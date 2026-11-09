"""Replay a causal CPU model of host-backed LRU on saved Top-K rankings."""

import argparse
from collections import Counter, OrderedDict
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("previous_step", "previous_layer", "rank_interleave")
CAPACITY_MULTIPLES = (1, 2, 4)
BUDGET_FRACTIONS = ((1, 2), (4, 5), (1, 1))


class LRU:
    def __init__(self, capacity, ranked):
        self.capacity = capacity
        self.blocks = OrderedDict.fromkeys(reversed(ranked))

    def touch_ranked(self, ranked):
        for block in reversed(ranked):
            self.blocks[block] = None
            self.blocks.move_to_end(block)
        while len(self.blocks) > self.capacity:
            self.blocks.popitem(last=False)

    def prefetch(self, ranked):
        transfers = [block for block in ranked if block not in self.blocks]
        self.touch_ranked(transfers)
        return set(transfers)

    def demand(self, ranked):
        missing = set(ranked).difference(self.blocks)
        self.touch_ranked(ranked)
        return missing


def predict(policy, previous_step, previous_layer, budget):
    if policy == "previous_step":
        return previous_step[:budget]
    if policy == "previous_layer":
        return previous_layer[:budget]
    if policy != "rank_interleave":
        raise ValueError(f"unknown policy: {policy}")
    # Same alternating unique-ID policy as predictor_rank_trace; importing
    # that experiment would load the GPU model stack in this CPU-only tool.
    sources = (iter(previous_step), iter(previous_layer))
    predicted = {}
    for slot in range(budget):
        source = sources[slot % 2]
        block = next(source)
        while block in predicted:
            block = next(source)
        predicted[block] = None
    return tuple(predicted)


def step_metrics(cache, prediction, ranked, baseline_missing):
    selected = set(ranked)
    before = selected.difference(cache.blocks)
    loaded = cache.prefetch(prediction)
    missing = cache.demand(ranked)
    covered = before - missing
    polluted = missing - before
    net_saved = len(baseline_missing) - len(missing)
    return {
        "baseline_demand_misses": len(baseline_missing),
        "candidate_demand_misses_before_prefetch": len(before),
        "candidate_demand_misses": len(missing),
        "prefetch_loaded_blocks": len(loaded),
        "prefetch_covered_own_misses": len(covered),
        "prefetch_blocks_unused_this_step": len(loaded - selected),
        "prefetch_pollution_misses": len(polluted),
        "baseline_only_misses": len(baseline_missing - missing),
        "candidate_only_misses": len(missing - baseline_missing),
        "net_demand_misses_saved": net_saved,
        "candidate_transfer_blocks": len(loaded) + len(missing),
        "extra_transfer_blocks": len(loaded) - net_saved,
        "head_observations": 1,
        "heads_with_fewer_demand_misses": int(net_saved > 0),
        "heads_with_more_demand_misses": int(net_saved < 0),
    }


def rank_rows(snapshot, layer, topk):
    rows = snapshot["ranked_ids_by_layer"][str(layer)]
    blocks = snapshot["complete_blocks_by_layer"][str(layer)]
    if not rows or any(len(r) != topk or len(set(r)) != topk or min(r) < 0 or max(r) >= blocks for r in rows):
        raise ValueError(f"invalid ranking at context {snapshot['context']}, step {snapshot['step']}, layer {layer}")
    return rows


def simulate(trace, bytes_per_block):
    topk = trace["topk"]
    rows = []
    for context in trace["contexts"]:
        snapshots = sorted((s for s in trace["snapshots"] if s["context"] == context), key=lambda s: s["step"])
        if [s["step"] for s in snapshots] != list(range(trace["decode_steps"])):
            raise ValueError(f"expected contiguous decode steps from zero at context {context}")
        for layer in trace["target_layers"]:
            ranks = [rank_rows(snapshot, layer, topk) for snapshot in snapshots]
            preceding = [rank_rows(snapshot, layer - 1, topk) for snapshot in snapshots]
            heads = len(ranks[0])
            if any(len(group) != heads for group in ranks + preceding):
                raise ValueError(f"query-head count changes at context {context}, layer {layer}")
            variants = [(policy, topk * numerator // denominator)
                        for numerator, denominator in BUDGET_FRACTIONS for policy in POLICIES]
            for multiple in CAPACITY_MULTIPLES:
                capacity = multiple * topk
                baselines = [LRU(capacity, ranked) for ranked in ranks[0]]
                candidates = {variant: [LRU(capacity, ranked) for ranked in ranks[0]] for variant in variants}
                for step in range(1, len(snapshots)):
                    demand = ranks[step]
                    base_missing = [cache.demand(ranked) for cache, ranked in zip(baselines, demand)]
                    available_blocks = snapshots[step]["complete_blocks_by_layer"][str(layer)]
                    for (policy, budget), caches in candidates.items():
                        totals = Counter()
                        for head, (cache, ranked, missing) in enumerate(zip(caches, demand, base_missing)):
                            prediction = predict(policy, ranks[step - 1][head], preceding[step][head], budget)
                            if any(block >= available_blocks for block in prediction):
                                raise ValueError(f"prediction references unavailable KV at context {context}, layer {layer}, step {step}")
                            totals.update(step_metrics(cache, prediction, ranked, missing))
                        totals["baseline_transfer_bytes"] = totals["baseline_demand_misses"] * bytes_per_block
                        totals["candidate_transfer_bytes"] = totals["candidate_transfer_blocks"] * bytes_per_block
                        totals["extra_transfer_bytes"] = totals["extra_transfer_blocks"] * bytes_per_block
                        rows.append({"context": context, "layer": layer, "step": step, "capacity": capacity,
                                     "prediction_budget": budget, "policy": policy, **totals})
    return rows


def summarize(rows, fields):
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        metrics = {name: value for name, value in row.items()
                   if name not in ("context", "layer", "step", "capacity", "prediction_budget", "policy")}
        groups.setdefault(key, Counter()).update(metrics)
    return [{**dict(zip(fields, key)), **totals} for key, totals in sorted(groups.items())]


def report(result, trace_path, result_path, report_path):
    from benchmarks.summarize import table

    def link(path):
        relative = Path(os.path.relpath(path.resolve(), report_path.parent.resolve())).as_posix()
        return f"[{path.name}]({relative})"

    setup = result["setup"]
    lines = ["# Host-backed KV 缓存的因果轨迹回放", "",
             f"在 CPU 上回放 {setup['model']} 的密集解码轨迹。选中集合来自外接 Quest 风格的块排序。上下文为 {setup['contexts']} token，目标层为 {setup['layers']}，每层每个查询头使用独立的逻辑 GPU 缓存。每个上下文第 0 步只用真实选中块热身。其后 {setup['measured_steps']} 步计入结果。{link(trace_path)}", "",
             "完整 KV 在逻辑 host 存储中可用。缓存按目标层与头独立管理，容量为 K、2K、4K 个块。每个策略和容量都从同一热身状态出发，随后各自演化。基线只按需搬运。候选在本层选择前使用上一解码步同层、当前步前一层，或两份排名交替去重后的前缀。跨层按查询头编号对应，预取目标层的 KV。", "",
             "只搬运预测列表里尚未驻留的块。预取写入 LRU，排名越靠前，写入后的访问记录越新。已驻留的预测块在需求前保持原有顺序。真实选集到达后，以需求开始前的缓存快照计算全部缺页，再统一填入并按排名更新 LRU。只读 KV 被驱逐时无需回写。模型假定预取在本步选择之前完成，搬运带宽、流依赖和排队时间均未计时。", "",
             f"逻辑块为 {setup['block_tokens']} token × {setup['head_dim']} 维 × {setup['element_bytes']} 字节 × K/V 两份，即每次 {setup['bytes_per_block']:,} 字节。逐查询头求和可能重复计算 GQA 共享的物理页。这个静态估算描述逻辑 H2D 字节。实测 H2D 时延、HiSparse 的 GPU 缓存行为和原生 QSA 均超出该轨迹的测量范围。{link(result_path)}", "",
             "`覆盖` 与 `污染` 在同一候选当步预取前后的状态上比较。`净少补` 等于独立按需基线缺块减候选需求缺块。额外搬运等于候选预取加需求，再减基线需求。未用预取指当步未被选择，之后仍可能有用。", "",
             "## 全部上下文和层的配对计数", ""]
    temporal = [row for row in result["summary"] if row["policy"] == "previous_step"]
    if temporal and all(row["prefetch_loaded_blocks"] == 0 for row in temporal):
        lines.extend(["上一解码步的选中块已由需求路径装入。容量至少为 K，且同一缓存的两次需求之间没有其他头或层写入。因此纯时间预测在本模型中发起零次搬运。若把预测用于注意力计算，`P ∩ resident` 仍可参与推测，无需额外 H2D。这个回放只统计搬运与驻留。", ""])
    moving = [row for row in result["summary"] if row["prefetch_loaded_blocks"]]
    if moving:
        best = max(moving, key=lambda row: row["net_demand_misses_saved"])
        fewer = sum(row["net_demand_misses_saved"] > 0 for row in moving)
        less_transfer = sum(row["extra_transfer_blocks"] < 0 for row in moving)
        lines.extend([f"发起搬运的 {len(moving)} 种配置中，{fewer} 种减少需求缺块，{less_transfer} 种减少总搬运。需求缺块降幅最大的配置为容量 {best['capacity']}、预测 {best['prediction_budget']}、`{best['policy']}`，需求缺块从 {best['baseline_demand_misses']:,} 降到 {best['candidate_demand_misses']:,}，多搬运 {best['extra_transfer_blocks']:,} 块。该配置当步直接覆盖 {best['prefetch_covered_own_misses']:,} 个自身缺块，其余净收益与历史缓存状态的变化有关。", ""])
    summary = []
    for row in result["summary"]:
        before = row["candidate_demand_misses_before_prefetch"]
        prefetch = row["prefetch_loaded_blocks"]
        helpful = row["prefetch_covered_own_misses"]
        coverage = f"{100 * helpful / before:.1f}%" if before else "—"
        extra_mib = row["extra_transfer_bytes"] / (1024 * 1024)
        summary.append([row["capacity"], row["prediction_budget"], f"`{row['policy']}`", row["baseline_demand_misses"],
                        row["candidate_demand_misses"], f"{row['net_demand_misses_saved']:+}", prefetch, f"{helpful} ({coverage})",
                        row["prefetch_blocks_unused_this_step"], row["prefetch_pollution_misses"], row["candidate_transfer_blocks"],
                        f"{row['extra_transfer_blocks']:+}", f"{extra_mib:+.2f}"])
    lines.extend([*table(["容量块数", "预测上限", "策略", "基线缺块", "候选缺块", "净少补", "预取搬运", "覆盖自身缺块",
                          "当步未用", "污染缺块", "候选总搬运", "额外搬运", "额外逻辑 MiB"], summary), ""])
    summary = []
    for row in result["by_context"]:
        summary.append([row["context"], row["capacity"], row["prediction_budget"], f"`{row['policy']}`",
                        f"{row['net_demand_misses_saved']:+}", f"{row['extra_transfer_blocks']:+}",
                        f"{row['heads_with_fewer_demand_misses']}/{row['heads_with_more_demand_misses']}/{row['head_observations']}"])
    lines.extend(["## 按上下文拆分", "", *table(["上下文 token", "容量块数", "预测上限", "策略", "净少补", "额外搬运",
                                                     "需求缺块少/多/总头观测"], summary), "",
                  "各快照的逐头合计、两个路径的独有需求缺块数和逻辑字节在原始结果中。回放只覆盖同一公开提示词的两段上下文，头观测跨层和解码步重复。", "",
                  "运行 `python -m experiments.hisparse_cache_trace` 可重新生成结果与本报告。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=ROOT / "results/window/predictor_rank_trace.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/hisparse/cache_trace.json")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/hisparse_trace.md")
    parser.add_argument("--head-dim", type=int, default=128, help="logical byte estimate only")
    parser.add_argument("--element-bytes", type=int, default=2, help="logical byte estimate only")
    args = parser.parse_args()
    if min(args.head_dim, args.element_bytes) <= 0:
        parser.error("byte estimate dimensions must be positive")
    raw = args.trace.read_bytes()
    trace = json.loads(raw.decode("utf-8-sig"))
    bytes_per_block = trace["block_size"] * args.head_dim * args.element_bytes * 2
    rows = simulate(trace, bytes_per_block)
    fields = ["capacity", "prediction_budget", "policy"]
    result = {
        "setup": {"model": trace["model"], "contexts": trace["contexts"], "layers": trace["target_layers"],
                  "measured_steps": trace["decode_steps"] - 1, "topk": trace["topk"], "block_tokens": trace["block_size"],
                  "head_dim": args.head_dim, "element_bytes": args.element_bytes, "bytes_per_block": bytes_per_block,
                  "trace_sha256": hashlib.sha256(raw).hexdigest(), "trace_file": args.trace.name},
        "scope": "CPU-only causal per-layer/query-head host-backed LRU model on dense Qwen with Quest proxy rankings",
        "warmup": "Step zero demand-only for all policies, excluded from metrics",
        "prefetch": "Only absent IDs transfer; higher predicted rank is newer; resident predictions are not promoted",
        "demand": "Misses are evaluated together before demand fills; higher selected rank is newer",
        "summary": summarize(rows, fields), "by_context": summarize(rows, ["context", *fields]), "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(report(result, args.trace, args.output, args.report), encoding="utf-8")


if __name__ == "__main__":
    main()
