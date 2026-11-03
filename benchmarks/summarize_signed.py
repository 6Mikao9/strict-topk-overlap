"""Regenerate the shared add/erase fusion report from raw paired timings."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

from benchmarks.summarize import interval, p95, table
from benchmarks.summarize_round2 import source


ROOT = Path(__file__).resolve().parents[1]


def ratio(row):
    return median(pair["baseline_ms"] / pair["candidate_ms"] for pair in row["pairs"])


def shape_key(config):
    return tuple(config[key] for key in ("heads", "blocks", "block_size", "dim", "topk", "overlap"))


def performance(lines, cases):
    groups = defaultdict(list)
    for case in cases:
        for row in case["rows"]:
            groups[row["method"]].append(row)
    for stage, title in [("repair", "修复阶段"), ("full", "完整路径")]:
        rows = []
        for name, measurements in sorted(groups.items()):
            if not name.startswith(stage + "_"):
                continue
            ratios = [ratio(row) for row in measurements]
            samples = [[pair["candidate_ms"] * 1000 for pair in row["pairs"]] for row in measurements]
            bases = {row["baseline"].removeprefix("repair_") for row in measurements}
            all_wins = sum(all(pair["baseline_ms"] > pair["candidate_ms"] for pair in row["pairs"]) for row in measurements)
            rows.append([f"`{name.removeprefix(stage + '_')}`", ", ".join(f"`{base}`" for base in sorted(bases)), len(measurements),
                         interval([median(s) for s in samples]), interval([p95(s) for s in samples]), interval(ratios, 4),
                         f"{sum(r > 1 for r in ratios)}/{len(ratios)}", f"{all_wins}/{len(ratios)}"])
        lines.extend([f"### {title}", "", *table(["方法", "分子基线", "测点", "p50 (µs)", "p95 (µs)", "配对中位数 (×)",
                                                 "测点中位数 > 1", "测点所有配对 > 1"], rows), ""])


def diagnostics(lines, cases):
    validation = [r for case in cases for r in case["validation"].values()]
    post_timing = [r for case in cases for r in case["post_timing_validation"].values()]
    lines.extend(["### 数值与调度", "", f"计时前 FP64 校验通过 {sum(r['passed'] for r in validation)}/{len(validation)}，计时后重新回放通过 {sum(r['passed'] for r in post_timing)}/{len(post_timing)}。计时后相对自身计算集合的最大绝对误差 {max(r['max_abs_error'] for r in post_timing):.3g}。", ""])
    groups = defaultdict(list)
    for case in cases:
        for name, diagnostic in case["diagnostics"].items():
            if name.startswith("repair_") and ("_signed_" in name or "_fused_" in name):
                groups[name.removeprefix("repair_")].append(diagnostic)
    rows = []
    for name, items in sorted(groups.items()):
        positive = [job for r in items for job in r.get("positive_jobs_per_head", [])]
        negative = [job for r in items for job in r.get("negative_jobs_per_head", [])]
        fallback = [r.get("fallback_heads", 0) for r in items]
        approximate = [r["approximate_heads"] for r in items if "approximate_heads" in r]
        bounds = [r["max_extra_mass_bound"] for r in items if "max_extra_mass_bound" in r]
        differences = [r["vs_strict_topk"]["max_abs_error"] for r in items]
        rows.append([f"`{name}`", interval(positive), interval(negative), sum(positive) if positive else "—",
                     sum(negative) if negative else "—", sum(fallback), sum(approximate) if approximate else "—",
                     f"{max(bounds):.3g}" if bounds else "—", interval(differences)])
    lines.extend([*table(["方法", "每头补块范围", "每头擦除范围", "补块总数", "擦除总数", "回退头总数", "近似头总数",
                         "最大记录质量上界", "相对严格的最大绝对误差范围"], rows), ""])


def synthetic_detail(lines, cases):
    groups = defaultdict(list)
    for case in cases:
        groups[shape_key(case["config"])].append(case)
    rows = []
    stable_mixed = []
    for shape, group in sorted(groups.items()):
        heads, blocks, block_size, dim, topk, overlap = shape
        config = f"{heads}/{blocks}/{block_size}/{dim}/{topk} · {100 * overlap:g}%"
        names = sorted(name.removeprefix("repair_") for name in group[0]["diagnostics"]
                       if name.startswith("repair_") and "_signed_" in name)
        for label in names:
            repair_rows = [row for case in group for row in case["rows"] if row["method"] == f"repair_{label}"]
            repair = [ratio(row) for row in repair_rows]
            full = [ratio(row) for case in group for row in case["rows"] if row["method"] == f"full_{label}"]
            diagnostic = [case["diagnostics"][f"repair_{label}"] for case in group]
            positive = [job for r in diagnostic for job in r["positive_jobs_per_head"]]
            negative = [job for r in diagnostic for job in r["negative_jobs_per_head"]]
            fallback = sum(r["fallback_heads"] for r in diagnostic)
            rows.append([config, f"`{label}`", interval(positive), interval(negative), fallback, interval(repair, 4),
                         f"{sum(r > 1 for r in repair)}/{len(repair)}", interval(full, 4), f"{sum(r > 1 for r in full)}/{len(full)}"])
            if len(repair) > 1 and label.endswith("eps0") and min(positive) > 0 and min(negative) > 0 and min(repair) > 1:
                stable_mixed.append((min(repair), config, label, positive, negative, fallback, repair_rows, repair, full))
    lines.extend(["### 按配置核对种子稳定性", "", "标签为 `H/N/B/D/K · 请求覆盖率`。各个种子重新生成 Q/K/V 和块 ID。修复和完整路径分别使用各自的基线。", "",
                  *table(["配置", "方法", "每头补块范围", "每头擦除范围", "回退头总数", "修复 (×)",
                          "修复较快种子", "完整 (×)", "完整较快种子"], rows), ""])
    if stable_mixed:
        _, config, label, positive, negative, fallback, repair_rows, repair, full = max(stable_mixed, key=lambda item: item[0])
        tails = [p95([pair["baseline_ms"] for pair in row["pairs"]]) / p95([pair["candidate_ms"] for pair in row["pairs"]])
                 for row in repair_rows]
        lines.extend([f"在每头同时有补块和擦除任务的严格配置中，`{label}` 的 `{config}` 取得最大的跨种子最小修复比值。每头补块 {interval(positive)} 个、擦除 {interval(negative)} 个，共回退 {fallback} 个头。修复配对中位数为 {interval(repair, 4)}×，两条修复路径的 p95 比为 {interval(tails, 4)}×。同一配置的完整路径为 {interval(full, 4)}×。", ""])


def dataset_section(lines, inputs, variant, synthetic=False):
    first = inputs[0][1]
    cases = [case for _, data in inputs for case in data["experiments"]]
    title = "合成输入" if synthetic else first["model"]
    links = " · ".join(source(path) for path, _ in inputs)
    rounds = [case["timing"]["rounds"] for case in cases]
    replays = [case["timing"]["replays"] for case in cases]
    gpu = ", ".join(sorted({data["environment"]["gpu"] for _, data in inputs}))
    lines.extend([f"## {variant} · {title}", "", f"{len(cases)} 个测点，{gpu}，每方法每点 {interval(rounds)} 轮配对，每段 {interval(replays)} 次 Graph replay。{links}", ""])
    if not synthetic:
        coverages = []
        for case in cases:
            cfg = case["config"]
            jobs = case["diagnostics"]["repair_hybrid_signed_eps0"]["positive_jobs_per_head"]
            coverages.append(100 * (1 - sum(jobs) / (cfg["heads"] * cfg["topk"])))
        lines.extend([f"上下文 {first['contexts']} token，密集模型解码 {first['steps']} 步，采样层 {first['layers']}。上一解码步的所选 ID 用于预测。各测点覆盖率为 {interval(coverages)}%，由补块任务计数除以选中块总数反推。选择器读取原生 GQA 元数据，各修复方法使用计时前已展开的 KV。", ""])
    performance(lines, cases)
    diagnostics(lines, cases)
    if synthetic:
        synthetic_detail(lines, cases)


def main_trace_summary(lines, traces):
    cases = [case for _, run in traces for case in run["experiments"]]
    if not cases:
        return
    links = " · ".join(source(path) for path, _ in traces)
    summary = []
    for mode in ["hybrid", "compact"]:
        tail = [row for case in cases for row in case["rows"] if row["method"] == f"repair_{mode}_signed_eps0"]
        full = [row for case in cases for row in case["rows"] if row["method"] == f"full_{mode}_signed_eps0"]
        tail_speed = [ratio(row) for row in tail]
        full_speed = [ratio(row) for row in full]
        summary.append([f"`{mode}`", interval(tail_speed, 4), f"{sum(x > 1 for x in tail_speed)}/{len(tail_speed)}",
                        interval(full_speed, 4), f"{sum(x > 1 for x in full_speed)}/{len(full_speed)}"])
    lines.extend(["## 主实现的真实 QKV 配对结果", "", "两款模型合并统计严格路径。旧 fused 的相同状态模式用于修复对照，串行精确用于完整路径对照。" + links, "",
                  *table(["状态模式", "修复加速比 (×)", "修复较快测点", "完整路径加速比 (×)", "完整较快测点"], summary), ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="*")
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "signed_results.md")
    args = parser.parse_args()
    folder = ROOT / "results" / "round2"
    paths = args.inputs or [*sorted(folder.glob("signed_2*.json")), *sorted(folder.glob("signed_fused_2*.json")),
                            *sorted(folder.glob("trace_signed_*.json"))]
    inputs = [(path, json.loads(path.read_text(encoding="utf-8-sig"))) for path in paths]
    if not inputs or any(not run.get("experiments") for _, run in inputs):
        parser.error("expected nonempty signed benchmark results")
    for path, run in inputs:
        if any("post_timing_validation" not in case for case in run["experiments"]):
            parser.error(f"{path.name} lacks post-timing validation; regenerate with the corrected benchmark")
    lines = ["# 补块与擦除共用 KV 读算的消融", "",
             "Qwen3 在回放采样时仍用 SDPA 执行密集解码。实验取出真实 Q/K/V，在每个固定快照上另接一个本地实现的 Quest 风格选择器。它按 32 token 分块，预先为每块的 K 向量按维度求最小值和最大值。本步 Q 的分量非负时取最大值，为负时取最小值，逐维乘 Q 后求和作为块分数，再用 `torch.topk` 选 K 块。严格 Top-K 指精确计算这个所选集合上的注意力。预测 ID 来自上一步选择。文中的完整路径只覆盖该原型的选择器、推测与修复，块元数据和预测 ID 构造均在计时区间外。", "",
             "先为每个头生成任务队列，补块在前，需要实际擦除的低质量错块在后。单个并行 kernel 处理两类任务的 KV，执行共用的 QK、指数和及加权 V 计算。每个任务由一个 CTA 计算，写出指数尺度及带符号的分母、分子状态。固定网格为每头预留 `|A| + |P|` 个任务位，无任务的 CTA 跳过 KV 加载。队列任务数在回退之前统计。", "",
             "主实现 `signed_fused` 在读算 kernel 内把负任务转换到低质量组尺度。compact 在该尺度直接求权重，hybrid 延续准备阶段的顺序，先按块求和再重缩放，避免逐 token 权重先下溢。最终合并内按头执行严格回退，整个修复共三次 kernel 启动。四启动参考版 `signed_repair` 把负块重缩放留给合并，并另起回退 kernel。两版共用任务队列生成逻辑。表内的 `signed` 指所在章节的版本，两版分别与各自运行中的旧 fused 或串行精确基线配对。", "",
             "修复阶段复用已选 ID 和准备好的预测状态，速度比为同模式同 `ε` 的旧 fused 修复延迟除以候选修复延迟。完整路径包含选择、推测、双流汇合与修复，速度比以该点的串行精确为分子。原合并融合、分阶段 split 和带符号任务融合共享状态布局。", "",
             "ABBA 或 BAAB 配对各含两段基线和两段候选 Graph replay。每对以两段均摊耗时计算比值，报告各点的比值中位数及跨点范围。p50、p95 对每个测点的配对均值计算。修复单测和完整路径来自各自的 Graph 与计时配对，两者的绝对值用于各自对照。计时排除编译、捕获、块元数据和预测 ID 构造。基准保留外部预测状态直到计时完成，随后对每张 Graph 再回放一次并校验输出。", "",
             "`ε=0` 按所选集合的 FP64 结果校验。`ε=0.05` 允许保留符合质量预算的低质量错块，按该方法实际保留的集合计算 FP64 oracle。记录的错误块质量上界与相对严格 Top-K 的输出差异分别统计。", ""]
    checks_path = folder / "signed_fused_correctness.json"
    if checks_path.is_file():
        checks = json.loads(checks_path.read_text(encoding="utf-8-sig"))
        changed = checks["graph_mutation_checks"]
        status = "通过" if checks["passed"] else "有失败"
        lines.extend([f"三启动主实现的对抗输入、质量预算和 CUDA Graph 复用检查{status}。在已捕获 Graph 中修改 V 后，FP16/BF16 的 saved、hybrid、compact 共 {sum(row['passed'] for row in changed)}/{len(changed)} 项通过 FP64 校验。{source(checks_path)}", ""])
        underflow = checks.get("hybrid_underflow_regression")
        if underflow:
            status = "通过" if underflow["passed"] else "失败"
            lines.extend([f"hybrid 下溢回归检查{status}，相对 FP64 的最大绝对误差为 {underflow['max_abs_error']:.3g}。{source(checks_path)}", ""])
    main_trace_summary(lines, [item for item in inputs if "signed_fused_" in item[0].name and "model" in item[1]])
    for fused, variant in [(True, "三启动主实现"), (False, "四启动参考版")]:
        selected = [item for item in inputs if ("signed_fused_" in item[0].name) == fused]
        synthetic = [item for item in selected if "model" not in item[1]]
        if synthetic:
            dataset_section(lines, synthetic, variant, synthetic=True)
        for item in selected:
            if "model" in item[1]:
                dataset_section(lines, [item], variant)
    manifests = [path for path in sorted(folder.glob("signed*manifest.json")) if path.is_file()]
    if manifests:
        lines.extend(["运行记录与测量源码摘要见 " + " · ".join(source(path) for path in manifests) + "。", ""])
    lines.extend(["运行 `python -m benchmarks.summarize_signed` 可重新生成此文。", ""])
    args.report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
