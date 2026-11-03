"""Summarize round-two GQA critical-path JSON on a CPU-only machine."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

from benchmarks.summarize import interval, p95, table


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "serial_exact"
STRICT = "saved_overlap"
UNION = "union_overlap"
ZERO = "zero_repair_overlap"


def source(path):
    resolved = path.resolve()
    results = ROOT / "results"
    if not resolved.is_relative_to(results):
        return f"`{path.name}`"
    relative = resolved.relative_to(results)
    return f"[{path.name}](../results/{relative.as_posix()})"


def batch_times(row):
    return [elapsed / row["replays_per_batch"] for elapsed in row["raw_batch_ms"]]


def p50(row):
    return median(batch_times(row))


def rows_by_method(experiment):
    return {row["method"]: row for row in experiment["rows"] if "raw_batch_ms" in row}


def speed_summary(groups, method):
    pairs = [(g[BASELINE], g[method]) for g in groups if BASELINE in g and method in g]
    speed = [p50(base) / p50(candidate) for base, candidate in pairs]
    return interval(speed), f"{sum(s > 1 for s in speed)}/{len(speed)}"


def stage_table(groups):
    known = [BASELINE, STRICT, UNION, ZERO, "selector_only", "prepare_only", "saved_repair_only", "union_repair_only", "join_only", "empty_graph"]
    observed = {name for g in groups for name in g}
    rows = []
    for name in [*known, *sorted(observed.difference(known))]:
        measured = [g[name] for g in groups if name in g]
        if not measured:
            continue
        times = [batch_times(row) for row in measured]
        ratio = wins = "—"
        if name in (STRICT, UNION, ZERO):
            ratio, wins = speed_summary(groups, name)
        rows.append([f"`{name}`", len(measured), interval([1000 * median(t) for t in times]),
                     interval([1000 * p95(t) for t in times]), ratio, wins])
    return table(["路径", "测点", "p50 (µs)", "p95 (µs)", "串行精确 / 此路径 p50 (×)", "p50 低于串行精确"], rows)


def correctness(lines, experiments):
    rows = [row for experiment in experiments for row in experiment["rows"]]
    captures = sum("capture_error" in row for row in rows)
    strict = [row["numerics_vs_own_semantics"] for row in rows if row.get("semantics") == "attention(selected)"
              and "numerics_vs_own_semantics" in row]
    union = [row for row in rows if "numerics_vs_exact_topk" in row]
    lines.extend([f"共 {len(rows)} 条路径记录，{sum(row['passed'] for row in rows)} 条通过各自的检查，CUDA Graph 捕获失败 {captures} 条。", ""])
    if strict:
        lines.extend([f"严格注意力相对 FP64 通过 {sum(row['passed'] for row in strict)}/{len(strict)}，最大绝对误差 {max(row['max_abs_error'] for row in strict):.3g}。", ""])
    if union:
        own = [row["numerics_vs_own_semantics"] for row in union]
        exact = [row["numerics_vs_exact_topk"] for row in union]
        lines.extend([f"并集完整路径相对自身并集 FP64 oracle 通过 {sum(r['passed'] for r in own)}/{len(own)}。相对严格 Top-K 通过 {sum(r['passed'] for r in exact)}/{len(exact)}，最大绝对差异 {max(r['max_abs_error'] for r in exact):.3g}。", ""])


def critical_path(lines, experiments, groups):
    paired = [(g[BASELINE], g[ZERO]) for g in groups if BASELINE in g and ZERO in g]
    if not paired:
        return []
    headroom = [(p50(base) - p50(zero)) * 1000 for base, zero in paired]
    zero_ratio = [p50(zero) / p50(base) for base, zero in paired]
    lines.extend(["#### 关键路径余量", "", f"去掉修复后，双流准备与汇合仍占同点串行精确 p50 的 {interval([100 * value for value in zero_ratio])}%。余量 `serial_exact − zero_repair_overlap` 为 {interval(headroom)} µs，有 {sum(value <= 0 for value in headroom)}/{len(headroom)} 个点在这个对照中已没有正余量。`zero_repair_overlap` 返回选中 ID 和预测状态，尚未产出注意力结果。", ""])
    strict_union = [p50(g[STRICT]) / p50(g[UNION]) for g in groups if STRICT in g and UNION in g]
    if strict_union:
        lines.extend([f"同一输入、选择器和预测状态下，严格完整路径 / 并集完整路径的 p50 比为 {interval(strict_union)}×。比值反映两套成员处理与修复分支的实测差异。", ""])
    overlap_work = [p50(g[ZERO]) / max(p50(g['selector_only']), p50(g['prepare_only']))
                    for g in groups if all(name in g for name in [ZERO, "selector_only", "prepare_only"])]
    if overlap_work:
        lines.extend([f"双流准备与汇合的 p50 是两个单测阶段最大值的 {interval(overlap_work)}×。单测阶段来自独立 Graph，差值包含并发资源竞争与调度影响。", ""])
    suffix = []
    for name in [STRICT, UNION]:
        deltas = [(1000 * (p50(g[name]) - p50(g[ZERO])), p50(g[name]) >= p50(g[BASELINE]))
                  for g in groups if all(path in g for path in [name, ZERO, BASELINE])]
        if deltas:
            suffix.append([f"`{name}`", interval([delta for delta, _ in deltas]),
                           f"{sum(at_least for _, at_least in deltas)}/{len(deltas)}"])
    if suffix:
        lines.extend(["完整 Graph 和无修复 Graph 的 p50 差值包含整个输出后缀及两张图的调度差异。对同一个测点，当这个差值达到无修复余量时，完整路径耗尽该点余量。", "",
                      *table(["路径", "完整减无修复 p50 (µs)", "达到或超出余量"], suffix), ""])
    audits = []
    for name in [BASELINE, "empty_graph"]:
        ratios = [g[name]["single_vs_batched_median"] for g in groups if name in g and "single_vs_batched_median" in g[name]]
        if ratios:
            audits.append([f"`{name}`", len(ratios), interval(ratios)])
    if audits:
        lines.extend([*table(["计时审计", "测点", "单次 Graph / 批量每次 p50 (×)"], audits), ""])

    detail = []
    for exp, group in zip(experiments, groups):
        if not all(name in group for name in [BASELINE, ZERO, STRICT, UNION]):
            continue
        config = exp["config"]
        detail.append((config, group))
    return detail


def group_detail(lines, detail, synthetic):
    if not detail:
        return
    by_budget = defaultdict(list)
    for cfg, group in detail:
        by_budget[cfg['context'], cfg['topk'], cfg['predict_k']].append((cfg, group))
    summary = []
    for (context, topk, predict_k), items in sorted(by_budget.items()):
        configs, groups = zip(*items)
        strict, wins = speed_summary(groups, STRICT)
        union, union_wins = speed_summary(groups, UNION)
        gaps = [(p50(g[BASELINE]) - p50(g[ZERO])) * 1000 for g in groups]
        coverages = [100 * c["overlap_mean"] for c in configs]
        summary.append([context, topk, predict_k, interval(coverages), strict, wins, union, union_wins, interval(gaps)])
    lines.extend(["#### 按上下文与预算分组", "", *table(["上下文 token", "选中块", "预测块", "实际覆盖率 (%)", "严格 p50 (×)", "严格更快", "并集 p50 (×)", "并集更快", "无修复余量 (µs)"], summary), ""])
    if synthetic:
        strata = defaultdict(list)
        for cfg, g in detail:
            strata[cfg['requested_overlap'], cfg['predict_k'] == cfg['topk']].append((cfg, g))
        rows = []
        for (coverage, equal), items in sorted(strata.items()):
            configs, groups = zip(*items)
            strict, wins = speed_summary(groups, STRICT)
            union, union_wins = speed_summary(groups, UNION)
            rows.append([f"{coverage * 100:g}%", "同预算" if equal else "异预算",
                         interval([100 * cfg['overlap_mean'] for cfg in configs]), strict, wins, union, union_wins])
        lines.extend(["#### 按命中率分组", "", "覆盖率以所选块数为分母。预测与所选预算相同时，100% 覆盖也意味着预测无错块。", "",
                      *table(["请求覆盖率", "预算关系", "实际覆盖率 (%)", "严格 p50 (×)", "严格更快", "并集 p50 (×)", "并集更快"], rows), ""])


def seed_comparison(lines, results):
    if len(results) < 2:
        return
    matched = defaultdict(dict)
    seed_rows = []
    for _, result in results:
        seed = result["arguments"]["seed"]
        groups = []
        for case in result["experiments"]:
            cfg = case["config"]
            shape = tuple(cfg[key] for key in ["context", "topk", "predict_k", "requested_overlap"])
            group = rows_by_method(case)
            if all(name in group for name in [BASELINE, STRICT, UNION]):
                matched[shape][seed] = group
                groups.append(group)
        seed_rows.append([seed, *speed_summary(groups, STRICT), *speed_summary(groups, UNION)])
    complete = {shape: samples for shape, samples in matched.items() if len(samples) == len(results)}
    lines.extend(["#### 跨种子的可重复性", "", "同形状、同请求覆盖率配对，不同种子重新生成输入与块 ID。这组数据检验配置收益对种子的稳定性。", "",
                  *table(["种子", "严格 p50 (×)", "严格更快", "并集 p50 (×)", "并集更快"], seed_rows), ""])
    repeatable = []
    for name in [STRICT, UNION]:
        wins = 0
        for (context, topk, predict_k, overlap), samples in sorted(complete.items()):
            speeds = [p50(group[BASELINE]) / p50(group[name]) for group in samples.values()]
            if min(speeds) > 1:
                wins += 1
                repeatable.append([f"`{name}`", context, topk, predict_k, f"{overlap * 100:g}%", interval(speeds, 4)])
        lines.extend([f"`{name}` 在 {wins}/{len(complete)} 个配置中对全部 {len(results)} 个种子都取得较低的 p50。", ""])
    if repeatable:
        lines.extend([*table(["路径", "上下文 token", "选中块", "预测块", "请求覆盖率", "跨种子 p50 加速比 (×)"], repeatable), ""])


def budget_comparison(lines, results):
    matched = defaultdict(dict)
    for _, result in results:
        for case in result["experiments"]:
            cfg = case["config"]
            if cfg["topk"] == 64 and cfg["requested_overlap"] == .5:
                matched[cfg["context"], result["arguments"]["seed"]][cfg["predict_k"]] = rows_by_method(case)
    rows = []
    for name in [ZERO, STRICT, UNION]:
        for size in [64, 96]:
            ratios = [p50(g[size][name]) / p50(g[32][name]) for g in matched.values()
                      if name in g.get(size, {}) and name in g.get(32, {})]
            if ratios:
                rows.append([f"`{name}`", size, interval(ratios), f"{sum(r > 1 for r in ratios)}/{len(ratios)}"])
    if rows:
        lines.extend(["#### 固定命中块数，增加错块预算", "", "每头选 64 块并固定命中 32 块。对同一上下文与种子，额外预测只加入错块。以预测 32 块为分母，比值大于 1 表示额外预算提高了延迟。", "",
                      *table(["路径", "预测块", "p50 / 预测 32 块的 p50 (×)", "更慢的配对"], rows), ""])


def trace_wins(lines, detail):
    wins = [(cfg, name, p50(g[BASELINE]) / p50(g[name])) for cfg, g in detail for name in [STRICT, UNION]
            if p50(g[BASELINE]) > p50(g[name])]
    if 0 < len(wins) <= 5:
        rows = [[f"`{name}`", cfg["context"], cfg["layer"], cfg["step"], cfg["topk"], cfg["blocks"],
                 f"{100 * cfg['overlap_mean']:.3g}%", f"{ratio:.4f}"] for cfg, name, ratio in wins]
        lines.extend(["下表列出 p50 略低于串行精确的全部回放快照。", "",
                      *table(["路径", "上下文 token", "层", "步", "选中块", "总块数", "覆盖率", "p50 加速比 (×)"], rows), ""])


def add_results(lines, results):
    result = results[0][1]
    synthetic = "model" not in result
    experiments = [case for _, run in results for case in run["experiments"]]
    groups = [rows_by_method(case) for case in experiments]
    configs = [case["config"] for case in experiments]
    title = "合成 GQA" if synthetic else result["model"]
    links = " · ".join(source(path) for path, _ in results)
    lines.extend([f"## {title}", "", f"原始记录包含 {len(experiments)} 个测点。{links}", ""])
    if not experiments:
        return
    if synthetic:
        seeds = [run.get("arguments", {}).get("seed") for _, run in results]
        lines.extend([f"种子为 {', '.join(str(seed) for seed in seeds if seed is not None)}。不同种子重新生成随机 Q/K/V。", ""])
    else:
        lines.extend([f"生成 {result['steps']} 步，采样层 {result['layers']}。用上一步所选块 ID 预测本步，仅包含完整 32-token 块。Q/K/V 来自密集解码，注意力方法在固定快照上回放。", ""])
    counts = [r["batches"] for g in groups for r in g.values()]
    replays = [r["replays_per_batch"] for g in groups for r in g.values()]
    shapes = sorted({(c["heads"], c["kv_heads"], c["block_size"], c["dim"]) for c in configs})
    average_coverage = [c["overlap_mean"] * 100 for c in configs]
    lines.extend([f"形状按 `(查询头, KV 头, 块长, 头维度)` 排列。{', '.join(f'`{shape}`' for shape in shapes)}。每路径每测点 {interval(counts)} 批，每批 {interval(replays)} 次 Graph replay。各测点逐头平均覆盖率为 {interval(average_coverage)}%，测点均值为 {sum(average_coverage) / len(average_coverage):.3g}%。", ""])
    correctness(lines, experiments)
    lines.extend([*stage_table(groups), ""])
    detail = critical_path(lines, experiments, groups)
    group_detail(lines, detail, synthetic)
    if synthetic:
        seed_comparison(lines, results)
        budget_comparison(lines, results)
    else:
        trace_wins(lines, detail)


def methodology_audit(lines, path, result):
    lines.extend(["## 计时方法与实现交叉审计", "", "同一行对同一张 Graph 使用单次计时与批量计时。旧版与原生 GQA 路径使用同一输入和预测 ID，选择器的有序输出一致时才计时。共享选择器的比较同时包含原生 KV 布局和注意力实现的改变。" + source(path), ""])
    numeric = [r["numerics_vs_exact_topk"] for c in result["experiments"] for r in c["rows"]]
    lines.extend([f"数值校验通过 {sum(r['passed'] for r in numeric)}/{len(numeric)}。", ""])
    matched = []
    for case in result["experiments"]:
        cfg = case["config"]
        rows = {row["method"]: row for row in case["rows"]}
        measurements = []
        for name, row in sorted(rows.items()):
            single = median(row["single"]["raw_ms"])
            batched = p50(row["batched"])
            measurements.append([f"`{name}`", f"{single * 1000:.3g}", f"{batched * 1000:.3g}", f"{single / batched:.3g}"])
        lines.extend([f"#### 上下文 {cfg['context']}，选中 {cfg['topk']} 块", "",
                      *table(["路径", "单次 p50 (µs)", "批量每次 p50 (µs)", "单次 / 批量 (×)"], measurements), ""])
        batched = {name: p50(row["batched"]) for name, row in rows.items()}
        pair_names = [("legacy_exact", "legacy_saved_overlap"), ("legacy_exact", "legacy_compact_fused_overlap"),
                      ("native_exact", "native_saved_overlap"),
                      ("legacy_saved_overlap", "native_saved_overlap_shared_selector"),
                      ("native_saved_overlap_shared_selector", "native_saved_overlap")]
        matched.append([cfg["context"], *(f"{batched[base] / batched[variant]:.3g}" for base, variant in pair_names)])
    lines.extend(["下表比较批量计时 p50。前 3 列方法比值均以同一实现家族的串行精确为分子。最后两列分别比较共享选择器下的 saved 变化，以及固定原生注意力实现时的选择器变化。", "",
                  *table(["上下文 token", "旧 saved (×)", "旧 compact 融合 (×)", "原生 saved (×)",
                          "旧 saved / 原生 saved · 共享选择器 (×)", "原生 saved · 旧选择器 / 原生选择器 (×)"], matched), ""])


def paired_confirmation(lines, results):
    result = results[0][1]
    cases = [case for _, run in results for case in run["experiments"]]
    if "model" in result:
        title = f"{result['model']} 的配对复测"
    elif result.get("arguments", {}).get("idealized_estimate"):
        title = "合成命中率已知时的理想化调度对照"
    else:
        title = "跨种子的固定估计配对复测"
    links = " · ".join(source(path) for path, _ in results)
    lines.extend([f"## {title}", "", "每个候选路径与相邻的串行精确基线组成 ABBA 或 BAAB 配对，批量回放，并随机化方法顺序。每个配对内用基线与候选各两段的平均时间计算速度比。不同配对共享同一进程内的设备条件。" + links, "",
                  "`light_overlap` 用每查询头一个 CTA 在一次启动内匹配并顺序处理漏块。`auto_overlap` 用外部漏块比例估计在轻量修复与分块并行修复间切换，两条分支都计算严格 Top-K。", ""])
    groups = defaultdict(list)
    methods = defaultdict(list)
    for case in cases:
        for row in case["rows"]:
            methods[row["method"]].append(row)
        cfg = case["config"]
        groups[cfg["context"], cfg["topk"]].append(case)
    estimates = {case["config"].get("estimate_source", case["config"].get("expected_miss_fraction_source", ""))
                 for case in cases}
    if "idealized_requested_overlap" in estimates:
        lines.extend(["该合成诊断把请求命中率用于外部估计。真实回放需要基于此前观察估计未来漏块数。", ""])
    elif "fixed" in estimates:
        estimate = [c["config"]["expected_miss_fraction"] for c in cases]
        lines.extend([f"该合成诊断使用固定漏块比例估计 {interval(estimate)}。", ""])
    elif "model" in result:
        lines.extend(["QKV 取自密集模型解码。调度估计使用上一观测到的命中率，初始按保守估计。", ""])
    checks = [row["passed"] for case in cases for row in case["validation"].values()]
    lightweight = sum(c["config"]["heads"] >= 16 and c["config"]["topk"] * c["config"]["expected_miss_fraction"] <= 2 for c in cases)
    lines.extend([f"共 {len(cases)} 个测点，语义与诊断校验通过 {sum(checks)}/{len(checks)}。`auto_overlap` 的静态规则在 {lightweight}/{len(cases)} 个测点选择轻量修复。", ""])
    summary = []
    for method, rows in sorted(methods.items()):
        ratios = [[pair["baseline_over_candidate"] for pair in row["pairs"]] for row in rows]
        point_medians = [median(r) for r in ratios]
        summary.append([f"`{method}`", len(rows), interval(point_medians, 4),
                        f"{sum(r > 1 for r in point_medians)}/{len(rows)}",
                        f"{sum(min(r) > 1 for r in ratios)}/{len(rows)}"])
    lines.extend([*table(["路径", "测点", "测点内配对加速比中位数 (×)", "中位数 > 1", "测点内全部配对 > 1"], summary), ""])
    detail = []
    for (context, topk), cases in sorted(groups.items()):
        for method in [STRICT, "light_overlap", "auto_overlap", UNION]:
            values = [row["median_paired_ratio"] for case in cases for row in case["rows"] if row["method"] == method]
            if values:
                detail.append([context, topk, f"`{method}`", interval(values, 4), f"{sum(r > 1 for r in values)}/{len(values)}"])
    lines.extend([*table(["上下文 token", "选中块", "路径", "配对加速比中位数 (×)", "测点更快"], detail), ""])
    if len(results) > 1:
        repeatable_pairs(lines, results)


def repeatable_pairs(lines, results):
    grouped = defaultdict(dict)
    for _, run in results:
        seed = run["arguments"]["seed"]
        for case in run["experiments"]:
            c = case["config"]
            for row in case["rows"]:
                key = (row["method"], c["context"], c["topk"], c["requested_overlap"])
                grouped[key][seed] = row
    summary = []
    for (method, context, topk, overlap), runs in sorted(grouped.items()):
        if method == ZERO or len(runs) != len(results):
            continue
        medians = [row["median_paired_ratio"] for row in runs.values()]
        if min(medians) > 1:
            summary.append([f"`{method}`", context, topk, f"{100 * overlap:g}%", interval(medians, 4),
                            f"{sum(row['min_paired_ratio'] > 1 for row in runs.values())}/{len(runs)}"])
    lines.extend(["下表列出每个种子的配对中位数都大于 1 的完整路径配置。最后一列更严格，要求该种子的全部配对都大于 1。", "",
                  *table(["路径", "上下文 token", "选中块", "请求覆盖率", "跨种子配对中位数 (×)", "全部配对更快的种子"], summary), ""])


def vendor_baseline(lines, path, result):
    env = result["environment"]
    lines.extend(["## 厂商库与原型串行基线", "", "各方法共享选择器。FlashInfer 以页表接收所选块，计时包含页表更新。Flash attention 的 gather 路径先物化所选 KV。两种库路径将各查询头视作独立请求，输出 FP16，原生 split-GQA 路径输出 FP32。固定形状的规划、图捕获和块元数据生成在计时区间外。" + source(path), ""])
    if not env.get("flash_attn_available"):
        lines.extend(["`gather_torch_flash` 使用 PyTorch SDPA 的 Flash 后端。此运行未取得可调用的独立 `flash_attn_func`。", ""])
    if env.get("flashinfer"):
        lines.extend([f"FlashInfer 版本 `{env['flashinfer']}`。Paged 方法还在同一张 Graph 中更换块 ID，再与 FP64 校验。", ""])
    rows = []
    for case in result["experiments"]:
        timings = {name: [elapsed / case["replays_per_batch"] for elapsed in row["raw_batch_ms"]]
                   for name, row in case["methods"].items()}
        baseline = median(timings["serial_gqa"])
        for name, values in sorted(timings.items()):
            row = case["methods"][name]
            error = row["numerics_vs_fp64"]["max_abs_error"]
            changed = row.get("changed_graph_ids_vs_fp64", {}).get("passed")
            rows.append([case["blocks"] * case["block_size"], case["topk"], f"`{name}`", f"{1000 * median(values):.3g}",
                         f"{1000 * p95(values):.3g}", f"{baseline / median(values):.3g}", f"{error:.3g}",
                         "通过" if row["passed"] else "失败", {True: "通过", False: "失败", None: "—"}[changed]])
    lines.extend([*table(["上下文 token", "选中块", "方法", "p50 (µs)", "p95 (µs)", "原生串行 / 此方法 (×)",
                         "最大绝对误差", "FP64 校验", "改块 ID 校验"], rows), ""])


def classify(result):
    cases = result.get("experiments", [])
    if not cases:
        return None
    if "ordered_selectors_match" in cases[0]:
        return "audit"
    if "methods" in cases[0]:
        return "vendor"
    if "diagnostics" in cases[0]:
        return None  # signed repair has its own report and baseline semantics
    if "validation" in cases[0]:
        return "paired"
    return "trace" if "model" in result else "synthetic"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="*")
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "round2_results.md")
    args = parser.parse_args()
    lines = ["# 第二轮实验结果", "", "同一套原生 GQA 内核和选择器比较串行精确、严格推测修复和并集推测修复。`union_overlap` 仍从逐块状态合并，保留预测中的错块并按自身集合校验；本轮没有运行原 PRR 的每头聚合与流式漏块修复内核。`zero_repair_overlap` 只返回选中 ID 和预测状态，不产出注意力结果，用于估计现有双流调度移除修复后的余量。", "",
             "**这里的 Qwen 使用稠密注意力。** Q/K/V 捕获后，在固定快照上额外运行 Quest 风格的块级 min/max 打分与 `torch.topk`，每块 32 个 token。按当前 Q 各维正负取块 K 的最大或最小值并求内积界；它不是 Qwen 自带的 Top-K 选择阶段。预测采用上一解码步选中 ID，没有实现 EMA。完整路径只指回放的选择、推测和修复，没有接入模型注意力输出，也不代表模型解码延迟或稀疏任务质量。", "",
             "CUDA event 计时区间内批量回放固定的 Graph，按批总时间除以每批回放次数得到样本。p50、p95 取这些批均值的分位数。报告区间跨不同测点，速度比始终对同一测点配对。计时排除编译、图捕获、元数据生成和预测列表构造。比较完整路径性能时，以相邻基线的配对复测为准。独立测量用于观察完整配置和分阶段开销。", ""]
    paths = args.inputs or sorted((ROOT / "results" / "round2").glob("*.json"))
    data = [(path, json.loads(path.read_text(encoding="utf-8-sig"))) for path in paths]
    synthetic = [(path, run) for path, run in data if classify(run) == "synthetic"
                 and (args.inputs or run.get("arguments", {}).get("profile") != "smoke")]
    traces = [(path, run) for path, run in data if classify(run) == "trace"]
    audits = [(path, run) for path, run in data if classify(run) == "audit"]
    paired = [(path, run) for path, run in data if classify(run) == "paired"
              and (args.inputs or run.get("arguments", {}).get("profile") != "smoke")]
    vendor = [(path, run) for path, run in data if classify(run) == "vendor"]
    if not synthetic and not traces and not audits and not paired and not vendor:
        parser.error("no recognized round-two result JSON found")
    fixed = [(path, run) for path, run in paired if "model" not in run and not run["arguments"].get("idealized_estimate")]
    paired_traces = [item for item in paired if "model" in item[1]]
    if fixed:
        paired_confirmation(lines, fixed)
    for item in paired_traces:
        paired_confirmation(lines, [item])
    for item in paired:
        if item not in fixed and item not in paired_traces:
            paired_confirmation(lines, [item])
    if synthetic:
        add_results(lines, synthetic)
    for trace in traces:
        add_results(lines, [trace])
    for path, run in audits:
        methodology_audit(lines, path, run)
    for path, run in vendor:
        vendor_baseline(lines, path, run)
    for path, run in data:
        if "stages" in run and path.stem in ("run_manifest", "confirm_manifest"):
            stages = [[row["name"], row["returncode"], f"{row['wall_seconds']:.2f}"] for row in run["stages"]]
            lines.extend([f"## 执行记录 · `{path.stem}`", "", f"单张 A100-SXM4-80GB。GPU 运行记录和测量源码摘要见 {source(path)}。", "",
                          *table(["阶段", "退出码", "墙钟时间 (s)"], stages), ""])
    lines.extend(["## 与第一轮及 PRR 的比较范围", "", "第一轮使用展开 GQA 的回放或合成多头输入，Graph 计时采用逐次 event。第二轮同时改变布局、内核和计时方式，宜采用同轮配对比值。第二轮的 p95 描述同一固定输入的批均值波动。", ""])
    if synthetic and traces and audits and fixed and paired_traces and vendor:
        lines.extend(["第一轮 compact 的慢速有明确的实现成本。它多做一遍 Q/K 统计，分组后按头聚合低质量块。相同 Graph 的单次与批量计时差异远小于长上下文下旧 compact 与旧串行精确的差距。切到原生 GQA 的 saved 路径后，严格和并集完整路径的耗时接近。去掉严格错块过滤只能改变修复的一部分，选择和推测的并发成本、漏块计算及输出合并仍消耗余量。轻量修复在无漏块的合成条件下降低了这部分成本，真实回放的漏块使每头顺序处理的代价变大。", "",
                      "此处 FlashInfer 页式基线的增益是在固定形状、共同选择器与 A100 上测得的。PRR 使用专门的 Hopper 内核、EMA 和按选择窗口调节的推测预算，并报告真实解码。现有结果解释了原型当前瓶颈，无法把与 PRR 的性能差距归到一种因素。", ""])
    lines.extend(["[第一轮结果](results.md) · [设计说明与 PRR 对照](design_explained.md)", ""])
    args.report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
