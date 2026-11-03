"""Summarize the common FlashInfer backend ablation from raw pair timings."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

from benchmarks.summarize import interval, p95, table
from benchmarks.summarize_round2 import source


ROOT = Path(__file__).resolve().parents[1]
SERIAL = "serial_flashinfer"
LEGACY = "legacy_virtual_heads_serial"
STRICT = "strict_flashinfer_overlap"
SERIALIZED = "strict_flashinfer_serialized"
UNION = "union_flashinfer_overlap"


def ratios(row):
    return [p["baseline_ms"] / p["candidate_ms"] for p in row["pairs"]]


def ratio_stats(rows):
    pairs = [ratios(row) for row in rows]
    medians = [median(values) for values in pairs]
    return [interval(medians, 4), f"{sum(value > 1 for value in medians)}/{len(rows)}",
            f"{sum(min(values) > 1 for values in pairs)}/{len(rows)}"]


def timings(rows, field):
    samples = [[p[field] * 1000 for p in row["pairs"]] for row in rows]
    return interval([median(values) for values in samples]), interval([p95(values) for values in samples])


def group_by_shape(cases):
    groups = defaultdict(list)
    for case in cases:
        cfg = case["config"]
        groups[cfg["kind"], cfg["context"], cfg["topk"]].append(case)
    return groups


def matching(cases, baseline, candidate):
    return [row for case in cases for row in case["rows"]
            if (row["baseline"], row["candidate"]) == (baseline, candidate)]


def full_paths(lines, cases):
    lines.extend(["## 严格推测修复", "", "三组配对分别衡量完整路径相对当前 FlashInfer 串行、相对旧虚拟头串行，以及相同推测与修复工作重叠后的收益。比值大于 1 表示分母路径更快。", ""])
    strict_rows = []
    for baseline, label in [(SERIAL, "当前 FlashInfer 串行"), (LEGACY, "旧 FlashInfer 虚拟头串行"),
                            (SERIALIZED, "同一严格推测串行化")]:
        rows = matching(cases, baseline, STRICT)
        if rows:
            strict_rows.append([label, *timings(rows, "baseline_ms"), *timings(rows, "candidate_ms"), *ratio_stats(rows)])
    lines.extend([*table(["分子基线", "基线 p50 (µs)", "基线 p95 (µs)", "严格并行 p50 (µs)", "严格并行 p95 (µs)",
                          "跨测点比值 (×)", "中位数 > 1", "全部配对 > 1"], strict_rows), ""])
    comparisons = [(matching([case], SERIAL, STRICT), matching([case], LEGACY, STRICT)) for case in cases]
    combined = [(current[0], old[0]) for current, old in comparisons if current and old]
    if combined:
        faster = sum(median(ratios(current)) > 1 and median(ratios(old)) > 1 for current, old in combined)
        lines.extend([f"严格并行在 {faster}/{len(combined)} 个测点的两项完整路径配对中均快于各自的串行基线。", ""])
    by_kind = defaultdict(list)
    for case in cases:
        by_kind[case["config"]["kind"]].append(case)
    summary = []
    for kind, group in sorted(by_kind.items()):
        speed = [ratio_stats(matching(group, baseline, STRICT))[0] for baseline in [SERIAL, LEGACY, SERIALIZED]]
        heads = sum(case["config"]["heads"] for case in group)
        fallback = sum(case["post_timing_validation"][STRICT]["fallback_heads"] for case in group)
        summary.append([kind, len(group), *speed, f"{fallback}/{heads}"])
    lines.extend([*table(["形状", "测点", "当前串行 / 严格并行 (×)", "旧串行 / 严格并行 (×)", "严格串行化 / 严格并行 (×)",
                          "回退头数"], summary), ""])
    details = []
    groups = defaultdict(list)
    for case in cases:
        cfg = case["config"]
        label = (cfg["kind"], cfg["context"], cfg["topk"], cfg["prediction_budget"], cfg["precision"], cfg["coverage"])
        groups[label].append(case)
    for (kind, context, topk, budget, precision, coverage), group in sorted(groups.items()):
        internal = ratio_stats(matching(group, SERIAL, STRICT))
        overlap = ratio_stats(matching(group, SERIALIZED, STRICT))
        fallback = [case["post_timing_validation"][STRICT]["fallback_heads"] for case in group]
        heads = sum(case["config"]["heads"] for case in group)
        details.append([kind, context, topk, budget, f"{precision * 100:.3g}%", f"{coverage * 100:.3g}%", *internal[:2],
                        *overlap[:2], f"{sum(fallback)}/{heads}"])
    lines.extend(["预测精确率以预测块数为分母，覆盖率以 K 为分母。回退计数取计时后的严格并行复验结果。", "",
                  *table(["形状", "上下文 token", "K", "预测数", "精确率", "覆盖率", "当前串行比值 (×)", "较快种子",
                          "严格串行化比值 (×)", "较快种子", "回退头数"], details), ""])


def union_and_baseline(lines, groups):
    lines.extend(["## 并集语义与串行后端对照", "", "本地实现的并集对照保留误预测块，再补上遗漏块。FP64 校验针对并集，另记录输出相对严格 Top-K 的整体 L2 差异。", ""])
    union = []
    legacy = []
    for label, cases in sorted(groups.items()):
        rows = matching(cases, SERIAL, UNION)
        checks = [case["post_timing_validation"][UNION] for case in cases]
        union.append([*label, *ratio_stats(rows), f"{sum(r['passed'] for r in checks)}/{len(checks)}",
                      interval([r["relative_l2_vs_strict"] for r in checks], 4)])
        old_rows = matching(cases, LEGACY, SERIAL)
        legacy.append([*label, *ratio_stats(old_rows)])
    lines.extend([*table(["形状", "上下文 token", "K", "串行精确 / 并集并行 (×)", "中位数 > 1", "全部配对 > 1",
                         "并集 FP64 校验", "相对严格 L2"], union), "",
                  "新旧串行对照同时反映 Tensor Core 开关，以及共享块 ID 配置下的 GQA 分组布局变化。", "",
                  *table(["形状", "上下文 token", "K", "旧串行 / 当前串行 (×)", "中位数 > 1", "全部配对 > 1"], legacy), ""])


def diagnostics(lines, cases, groups, folder):
    checks = [value for case in cases for value in case["post_timing_validation"].values()]
    strict = [case["post_timing_validation"][STRICT] for case in cases]
    total_heads = sum(case["config"]["heads"] for case in cases)
    lines.extend(["## 数值与阶段诊断", "", f"计时后 Graph 复验通过 {sum(r['passed'] for r in checks)}/{len(checks)}。严格并行相对 FP64 的最大绝对误差为 {max(r['max_abs_error'] for r in strict):.3g}，复验回退 {sum(r['fallback_heads'] for r in strict)}/{total_heads} 个头。", ""])
    path = folder / "correctness.json"
    if path.is_file():
        result = json.loads(path.read_text(encoding="utf-8-sig"))
        values = [v for row in result["rows"] for v in row["checks"].values()]
        tolerances = {row["dtype"].removeprefix("torch."): (v["atol"], v["rtol"])
                      for row in result["rows"] if "dtype" in row for v in row["checks"].values()}
        bounds = "，".join(f"{dtype} 为 `{atol:g} + {rtol:g} × |参考值|`"
                          for dtype, (atol, rtol) in sorted(tolerances.items()))
        lines.extend([f"单独的图内数据变更和数值检查通过 {sum(v['passed'] for v in values)}/{len(values)}。容差按逐元素绝对误差计算，{bounds}。{source(path)}", ""])
    rows = []
    for label, group in sorted(groups.items()):
        stages = [case["isolated_stage_ms"] for case in group]
        summary = [interval([row[stage] * 1000 for row in stages]) for stage in ["selector", "attention", "prediction", "repair"]]
        rows.append([*label, *summary])
    lines.extend(["阶段单测使用独立 Graph，适合观察各阶段的耗时量级。完整路径结论使用相邻的完整路径配对。", "",
                  *table(["形状", "上下文 token", "K", "选择 p50 (µs)", "所选注意力 p50 (µs)", "预测 p50 (µs)", "修复 p50 (µs)"], rows), ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "vendor_backend_results.md")
    args = parser.parse_args()
    folder = ROOT / "results" / "vendor_shared"
    paths = args.inputs or sorted(folder.glob("benchmark_2*.json"))
    inputs = [(path, json.loads(path.read_text(encoding="utf-8-sig"))) for path in paths]
    if not inputs or any(not run.get("experiments") for _, run in inputs):
        parser.error("expected nonempty FlashInfer backend result JSON")
    hashes = {tuple(sorted(run["source_sha256"].items())) for _, run in inputs}
    if len(hashes) != 1:
        parser.error("benchmark results were captured from different source snapshots")
    cases = [case for _, result in inputs for case in result["experiments"]]
    groups = group_by_shape(cases)
    links = " · ".join(source(path) for path, _ in inputs)
    seeds = "、".join(str(result["seed"]) for _, result in inputs)
    devices = "、".join(sorted({result["environment"]["gpu"] for _, result in inputs}))
    versions = "、".join(sorted({result["environment"]["flashinfer"] for _, result in inputs}))
    lines = ["# 串行与推测两侧接入 FlashInfer", "",
             f"{devices}，FlashInfer {versions}，种子 {seeds}，共 {len(cases)} 个 FP16 随机输入测点。`qsa_shape` 使用 FlashNext 公开形状与独立评分内核，`current` 使用 Quest 风格 min/max 评分。各测点的输入、选择器、所选集合和预测集合固定。完整路径只覆盖合成快照上的选择、注意力与修复。{links}", "",
             "串行精确与推测均用 FlashInfer Tensor Core paged decode，推测额外取输出和 LSE。共享选集的 `qsa_shape` 每个 KV 头分组处理 12 个查询头。修复的补块和擦除块共用一次 FlashInfer FA2 paged prefill 调用。Triton 负责集合匹配、任务和页表构造、带符号合并。严格路径的保护性回退对标记头按最多 128 token 并行正向重算，再归并。无回退时两次 kernel 启动也计时，对应程序直接退出。", "",
             "修复请求最多 128 个 KV token，沿用 FlashInfer 0.3.1 FA2 的单 tile 调度条件。GPU 在计时中改写动态 CSR 页表，空请求仍运行一个占位页并在合并时屏蔽。该技巧绑定到该版本的内部调度。固定规划、块元数据和预测 ID 构造在计时区间外。[本地接入代码](../strict_topk/flashinfer_backend.py) · [FlashInfer v0.3.1 调度源码](https://github.com/flashinfer-ai/flashinfer/blob/v0.3.1/include/flashinfer/attention/scheduler.cuh)", "",
             "随机 ABBA 或 BAAB 每段回放同一 CUDA Graph 100 次，共 5 对。每对分别取基线和候选两段的均摊延迟，分子除以分母。比值范围覆盖各测点的配对中位数。p50、p95 是固定输入下五个配对均值的分位数。带符号合并的守卫提供经验性保护，实测 FP64 对照用于检验所测输入。", ""]
    full_paths(lines, cases)
    union_and_baseline(lines, groups)
    diagnostics(lines, cases, groups, folder)
    manifest = folder / "run_manifest.json"
    if manifest.is_file():
        lines.extend([f"执行状态和测量源码摘要见 {source(manifest)}。", ""])
    lines.extend(["运行 `python -m benchmarks.summarize_vendor` 可从原始 JSON 重新生成此文。", ""])
    args.report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
