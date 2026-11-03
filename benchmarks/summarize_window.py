"""Build the selector-window and predictor-rank report from raw JSON."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median

from benchmarks.summarize import interval, table
from benchmarks.summarize_round2 import source


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = "serial_exact"
FLASHINFER = "serial_flashinfer"
SAVED = "saved_overlap"
ZERO = "zero_repair_overlap"
POLICIES = ["temporal", "layer", "interleave"]


def stage_ratio(probe):
    return median(row["ms"]["selector"] / row["ms"]["attention"] for row in probe["raw_rounds"])


def case_label(cfg):
    return cfg["kind"], cfg["context"], cfg["topk"]


def probes(lines, inputs):
    if not inputs:
        return
    links = " · ".join(source(path) for path, _ in inputs)
    lines.extend(["## 串行阶段探测", "", "`S` 是选择，`A` 是已给定选中 ID 的串行注意力。阶段各自捕获 Graph，并打乱采样顺序。`S/A` 先逐轮求比再取中位数。串行总耗时来自同时包含两阶段的独立 Graph。" + links, ""])
    rows = []
    for _, result in inputs:
        for item in result["experiments"]:
            cfg = item["config"]
            measurement = item["serial_probe"]
            timing = [median(row["ms"][stage] for row in measurement["raw_rounds"]) * 1000
                      for stage in ["selector", "attention", "serial"]]
            backend = "分层" if cfg.get("hierarchical") else "分块"
            rows.append([*case_label(cfg), backend, *(f"{value:.4g}" for value in timing), f"{stage_ratio(measurement):.4g}"])
    lines.extend([*table(["形状", "上下文 token", "K", "注意力实现", "S p50 (µs)", "A p50 (µs)", "串行 p50 (µs)", "S/A (×)"], rows), ""])


def calibrations(lines, inputs):
    groups = defaultdict(list)
    natural = {}
    for _, run in inputs:
        for item in run["calibrations"]:
            label = case_label(item["config"])
            groups[(*label, item["window"])].append(item["calibration"])
            if item["window"] == "natural":
                natural.setdefault(label, []).append(stage_ratio(item["natural_stages"]))
    rows = []
    for (kind, context, topk, window), items in sorted(groups.items()):
        base = natural.get((kind, context, topk), [])
        verified = [stage_ratio(item["verified_stages"]) for item in items if "verified_stages" in item]
        counts = [item["iterations"] for item in items if item["iterations"]]
        rows.append([kind, context, topk, window, interval(base, 4), interval(verified, 4),
                     f"{len(counts)}/{len(items)}", interval(counts, 5)])
    lines.extend(["### 选择窗口校准", "", "当原始 `S < A` 时才增加一段 GPU 算术，并复测 `S/A`。`S ≥ A` 保留原始路径。1 CTA 与 64 CTA 两种填充只检验合成工作并行占用的影响。表中横线表示该列无独立填充复测。", "",
                  *table(["形状", "上下文 token", "K", "窗口", "自然 S/A (×)", "校准后 S/A (×)", "实际填充种子", "迭代次数"], rows), ""])


def pair_ratios(row):
    return [pair["baseline_ms"] / pair["candidate_ms"] for pair in row["pairs"]]


def score(rows):
    ratios = [pair_ratios(row) for row in rows]
    return [interval([median(values) for values in ratios], 4), f"{sum(median(values) > 1 for values in ratios)}/{len(rows)}",
            f"{sum(min(values) > 1 for values in ratios)}/{len(rows)}"]


def speed_rows(lines, cases):
    windows = defaultdict(lambda: defaultdict(list))
    configs = defaultdict(lambda: defaultdict(list))
    for case in cases:
        cfg = case["config"]
        group = (*case_label(cfg), cfg["window"])
        budget = (cfg["prediction_budget"], cfg["controlled_precision"], cfg["controlled_coverage"],
                  cfg["misses_per_head"], cfg["wrong_per_head"])
        for row in case["rows"]:
            path = row["baseline"], row["candidate"]
            windows[group][path].append(row)
            configs[(*group, *budget)][path].append(row)
    for kind in sorted({label[0] for label in windows}):
        relevant = [paths for label, paths in windows.items() if label[0] == kind]
        internal = [row for paths in relevant for row in paths[(IMPLEMENTATION, SAVED)]]
        vendor = [row for paths in relevant for row in paths[(FLASHINFER, SAVED)]]
        internal_speed, internal_wins, _ = score(internal)
        vendor_speed, vendor_wins, _ = score(vendor)
        lines.extend([f"`{kind}` 的完整严格推测路径相对同实现串行的比值为 {internal_speed}×，{internal_wins} 个测点较快。相对适配的 FlashInfer 串行为 {vendor_speed}×，{vendor_wins} 个测点较快。", ""])
    padded_with_misses = [row for case in cases if case["config"]["window"].startswith("pad_") and case["config"]["misses_per_head"] > 0
                          for row in case["rows"] if (row["baseline"], row["candidate"]) == (IMPLEMENTATION, SAVED)]
    if padded_with_misses:
        speed, wins, _ = score(padded_with_misses)
        lines.extend([f"在校准到约 1:1 且有漏块的测点，完整路径相对同实现串行的比值为 {speed}×，较快测点为 {wins}。", ""])
    pairs = [(IMPLEMENTATION, SAVED), (FLASHINFER, SAVED), (IMPLEMENTATION, ZERO), (IMPLEMENTATION, FLASHINFER)]
    rows = []
    for label, comparisons in sorted(windows.items()):
        for base, candidate in pairs:
            measured = comparisons.get((base, candidate), [])
            if measured:
                gaps = [1000 * median(pair["baseline_ms"] - pair["candidate_ms"] for pair in row["pairs"]) for row in measured]
                rows.append([*label, f"`{base}` / `{candidate}`", *score(measured), interval(gaps, 4)])
    lines.extend(["### 按窗口汇总配对耗时", "", "ABBA/BAAB 每段为 Graph 多次回放的均摊耗时。同一对的两段基线均值除以两段候选均值，逐测点报告比值中位数。大于 1 表示候选更快。`zero_repair_overlap` 只返回选中 ID 和推测状态。", "",
                  *table(["形状", "上下文 token", "K", "窗口", "分子 / 分母", "测点比值范围 (×)", "中位数 >1", "全部配对 >1",
                          "配对基线减候选 (µs)"], rows), ""])
    rows = []
    for (kind, context, k, window, budget, precision, coverage, miss, wrong), comparisons in sorted(configs.items()):
        internal = comparisons[(IMPLEMENTATION, SAVED)]
        vendor = comparisons[(FLASHINFER, SAVED)]
        rows.append([kind, context, k, window, budget, f"{precision * 100:.3g}%", f"{coverage * 100:.3g}%",
                     miss, wrong, *score(internal)[:2], *score(vendor)[:2]])
    lines.extend(["### 预算和命中率分层", "", "精确率为命中数 / 预测数，覆盖率为命中数 / 所选 K。按种子在同一形状、窗口、预算和精确率下配对。每个测点的原始记录保留每轮延迟。", "",
                  *table(["形状", "上下文 token", "K", "窗口", "预测数", "精确率", "覆盖率", "漏块/头", "错块/头",
                          "同实现比值 (×)", "较快种子", "FlashInfer 比值 (×)", "较快种子"], rows), ""])


def simulations(lines, inputs):
    if not inputs:
        return
    cases = [case for _, result in inputs for case in result["experiments"]]
    checks = [row for case in cases for row in case["post_timing_validation"].values()]
    links = " · ".join(source(path) for path, _ in inputs)
    seeds = [result.get("seed") for _, result in inputs]
    gpu = sorted({result["environment"]["gpu"] for _, result in inputs})
    lines.extend(["## 控制选择窗口与预测预算", "", f"设备为 {'、'.join(gpu)}，种子为 {'、'.join(str(seed) for seed in seeds)}，共 {len(cases)} 个测点，计时后 Graph 输出校验 {sum(row['passed'] for row in checks)}/{len(checks)}。完整注意力与 FP64 比较，无修复对照只校验选中 ID。{links}", ""])
    shapes = sorted({tuple(case["config"][key] for key in ["kind", "heads", "kv_heads", "dim", "block_size"]) for case in cases})
    lines.extend([*table(["形状", "查询头", "KV 头", "头维度", "块长 token"], shapes), ""])
    calibrations(lines, inputs)
    speed_rows(lines, cases)


def rank_summary(lines, path, data):
    cases = data["experiments"]
    lines.extend([f"## {data['model']} 的排名轨迹", "", f"上下文 {data['contexts']}，密集解码 {data['decode_steps']} 步，采样目标层 {data['target_layers']}，块长 {data['block_size']}，K={data['topk']}。共 {len(cases)} 个预算与快照组合。{source(path)}", "",
                  "`temporal` 取上一解码步同层排名，`layer` 取本步前一密集层排名，`interleave` 依次从两份排名交替取唯一 ID。跨层按同一查询头序号配对。目标块来自本步同层的 Quest 风格代理排名。各步和层的头样本用于描述性统计。轨迹分析的指标为排名和集合命中。", ""])
    cutoffs = [pair for snapshot in data["snapshots"] for layer in snapshot["score_ties_by_layer"].values()
               for pair in zip(layer["all_blocks_tied_at_cutoff"], layer["selected_blocks_tied_at_cutoff"])]
    if cutoffs:
        lines.extend([f"排序按分数降序，完全同分按块 ID 升序。采样排名中有 {sum(all_tied > used for all_tied, used in cutoffs)}/{len(cutoffs)} 个头的并列跨过 Top-K 截断位置。", ""])
    grouped = defaultdict(list)
    layered = defaultdict(list)
    source_bins = defaultdict(list)
    prefix = defaultdict(dict)
    for case in cases:
        group = case["context"], case["budget"]
        for row in case["heads"]:
            grouped[group].append(row)
            layered[case["context"], case["layer"], case["budget"]].append(row)
            if case["budget"] == data["topk"]:
                source_bins[case["context"]].append(row)
            key = (case["context"], case["step"], case["layer"], row["head"])
            prefix[key][case["budget_fraction"]] = (case["budget"], row)
    summary = []
    for budget in sorted({key[1] for key in grouped}):
        samples = [row for (_, p), rows in grouped.items() if p == budget for row in rows]
        for policy in POLICIES:
            precision = 100 * mean(row[policy]["precision"] for row in samples)
            coverage = 100 * mean(row[policy]["coverage"] for row in samples)
            summary.append([budget, f"`{policy}`", f"{precision:.3f}", f"{coverage:.3f}"])
    lines.extend(["下表把采样上下文和层合并作描述性汇总。各头跨层、跨步有重复观测。", "",
                  *table(["预测数", "策略", "平均精确率 (%)", "平均覆盖率 (%)"], summary), ""])
    rows = []
    for (context, budget), samples in sorted(grouped.items()):
        for policy in POLICIES:
            values = [row[policy] for row in samples]
            averages = [mean(value[key] for value in values) * 100 for key in ["precision", "coverage"]]
            misses = [value["miss"] for value in values]
            wrong = [value["wrong"] for value in values]
            rows.append([context, budget, f"`{policy}`", len(values), *(f"{x:.3f}" for x in averages), interval(misses), interval(wrong)])
    lines.extend([*table(["上下文 token", "预测数", "策略", "头样本数", "平均精确率 (%)", "平均覆盖率 (%)", "漏块范围/头", "错块范围/头"], rows), ""])
    rows = []
    for (context, layer, budget), samples in sorted(layered.items()):
        coverage = [100 * mean(row[policy]["coverage"] for row in samples) for policy in POLICIES]
        rows.append([context, layer, budget, len(samples), *(f"{x:.3f}" for x in coverage)])
    lines.extend(["### 按上下文和目标层分组", "", *table(["上下文 token", "目标层", "预测数", "头样本数", "时间覆盖率 (%)",
                  "前层覆盖率 (%)", "交错覆盖率 (%)"], rows), ""])
    rows = []
    for (context, budget), samples in sorted(grouped.items()):
        coverage = [sum(row["interleave"]["hits"] > row[policy]["hits"] for row in samples) for policy in ["temporal", "layer"]]
        exclusive = [mean(row["sources"][key] for row in samples) for key in ["common_hits", "temporal_only_hits", "layer_only_hits"]]
        rows.append([context, budget, *(f"{x:.3f}" for x in exclusive), *(f"{x}/{len(samples)}" for x in coverage)])
    lines.extend([*table(["上下文 token", "预测数", "两来源共同命中/头", "仅时间命中/头", "仅跨层命中/头",
                          "交错比时间多命中", "交错比跨层多命中"], rows), ""])
    retention(lines, source_bins, prefix)


def retention(lines, source_bins, prefix):
    rows = []
    for context, samples in sorted(source_bins.items()):
        for policy in POLICIES:
            for key, label in [("source_rank_quartile_retention", "来源排序被目标保留"),
                               ("target_rank_quartile_retention", "目标排序被预测覆盖")]:
                values = [row[policy][key] for row in samples if key in row[policy]]
                if values:
                    rates = [100 * mean(q) for q in zip(*values)]
                    rows.append([context, f"`{policy}`", label, *(f"{value:.3f}" for value in rates)])
    lines.extend(["### 排名位置与保留率", "", "四分位统计使用完整 K 预算。`Q1` 是最高分四分位。来源保留率问旧排名中的块能否进入当前 Top-K；目标覆盖率问当前排名中的块是否被旧排名预测到。", "",
                  *table(["上下文 token", "策略", "统计方向", "Q1 (%)", "Q2 (%)", "Q3 (%)", "Q4 (%)"], rows), ""])
    grouped = defaultdict(list)
    for key, variants in prefix.items():
        if .8 not in variants or 1. not in variants:
            continue
        top, truncated = variants[.8]
        size, full = variants[1.]
        for policy in ["temporal", "layer"]:
            prefix_rate = truncated[policy]["hits"] / top
            tail_rate = (full[policy]["hits"] - truncated[policy]["hits"]) / (size - top)
            grouped[key[0], policy, top, size].append((prefix_rate, tail_rate))
    rows = []
    for (context, policy, top, size), values in sorted(grouped.items()):
        top_rates, tail_rates = zip(*values)
        rows.append([context, f"`{policy}`", top, size - top, f"{100 * mean(top_rates):.3f}", f"{100 * mean(tail_rates):.3f}",
                     f"{sum(a > b for a, b in values)}/{len(values)}"])
    lines.extend(["前缀与尾部使用同一来源排名及相同目标 Top-K。前缀为 `floor(0.8K)` 个块。", "",
                  *table(["上下文 token", "策略", "前缀块数", "尾部块数", "前缀保留率 (%)", "尾部保留率 (%)", "前缀率较高的头样本"], rows), ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="*")
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "window_results.md")
    args = parser.parse_args()
    paths = args.inputs or sorted((ROOT / "results" / "window").glob("*.json"))
    data = [(path, json.loads(path.read_text(encoding="utf-8-sig"))) for path in paths]
    probe_runs = [(path, run) for path, run in data if run.get("experiments") and "serial_probe" in run["experiments"][0]]
    simulation_runs = [(path, run) for path, run in data if run.get("experiments") and "calibrations" in run]
    traces = [(path, run) for path, run in data if run.get("experiments") and "snapshots" in run]
    if not probe_runs and not simulation_runs and not traces:
        parser.error("no recognized window or predictor result JSON found")
    lines = ["# 选择窗口、推测预算与排名代理实验", "",
             "目标工作负载是 FlashNext 的 QSA。本轮没有加载其权重，而以 FP16 随机张量、目标形状和独立 Triton 打分构造 `qsa_shape`。打分对每个 indexer 头的 QK 点积取 ReLU，再求和并按头维度缩放，由 `torch.topk` 输出各查询头共享的块 ID。实验范围限于形状、评分代理和内核计时，不是原生 SGLang QSA 或真实模型运行。`current` 是前轮 Quest 风格逐头 min/max 界选择器。", "",
             "仅在原始选择耗时短于注意力时增加 GPU 算术，目标为 `S:A ≈ 1:1`。填充工作对串行与并行路径都计时。1 CTA 和 64 CTA 是两种合成争用条件。原始 `S ≥ A` 的配置使用自然窗口。完整路径包括选择、推测、流汇合与严格修复。FlashInfer 基线适配为每查询头一个虚拟请求，GPU 页表更新计时，固定规划在计时之外。", "",
             "合成实验独立控制预算和精确率。关于排名前缀稳定性，证据来自密集 Qwen3-4B Q/K 上的外接 Quest 风格排序。跨层按头序号对齐，结论适用于这一代理。权重训练、QSA 原生服务和生成质量均在本次测量范围之外。", ""]
    model_configs = sorted({run["model_configuration_reference"] for _, run in simulation_runs if "model_configuration_reference" in run})
    if model_configs:
        links = " · ".join(f"[FlashNext 公开配置]({url})" for url in model_configs)
        lines.extend([f"形状参考公开的查询头、KV 头、头维度与 indexer 压缩率。{links}", ""])
    probes(lines, probe_runs)
    simulations(lines, simulation_runs)
    for path, run in traces:
        rank_summary(lines, path, run)
    for path, run in data:
        if "stages" in run:
            rows = [[stage["name"], stage["returncode"]] for stage in run["stages"]]
            lines.extend(["## 执行记录", "", f"GPU 运行记录与源码摘要见 {source(path)}。", "", *table(["阶段", "退出码"], rows), ""])
    lines.extend(["新的打包与分层归并内核另在 FP16、BF16 下通过 36 项 FP64/Graph 变 ID 检查，见 [correctness.json](../results/window/correctness.json)。", "",
                  "![排名预测代理实验](../figures/predictor_retention.png)", "",
                  "运行 `python -m benchmarks.summarize_window` 可从原始 JSON 重新生成此文。", ""])
    args.report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
