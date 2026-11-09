"""Render the checked-in JSON results using Python and matplotlib (no GPU needed)."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"saved": "#0072B2", "hybrid": "#D55E00", "compact": "#009E73", "prefetch": "#CC79A7"}


def interval(values, precision=3):
    if not values:
        return "—"
    if min(values) == max(values):
        return f"{min(values):.{precision}g}"
    return f"{min(values):.{precision}g}–{max(values):.{precision}g}"


def p95(samples):
    ordered = sorted(samples)
    pos = (len(ordered) - 1) * .95
    low = int(pos)
    return ordered[low] + (pos - low) * (ordered[min(low + 1, len(ordered) - 1)] - ordered[low])


def timing(row):
    samples = row["raw_ms"]
    return statistics.median(samples), p95(samples)


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"] + [
        "| " + " | ".join(map(str, row)) + " |" for row in rows
    ]


def source(path):
    return f"[{path.name}](../results/{path.name})"


def row_groups(result):
    if "experiments" in result:
        return [{row["method"]: row for row in item["rows"]} for item in result["experiments"]]
    groups = defaultdict(dict)
    for row in result["rows"]:
        if row.get("epsilon", 0) == 0:
            groups[row["layer"], row["step"]][row["method"]] = row
    return list(groups.values())


def comparison(groups):
    methods = defaultdict(list)
    for group in groups:
        base = group.get("serial_exact", {})
        for name, row in group.items():
            if "raw_ms" not in base or "raw_ms" not in row:
                continue
            median, tail = timing(row)
            baseline_median, baseline_tail = timing(base)
            serial = group.get(name.removesuffix("_overlap") + "_serial", {})
            paired = timing(serial)[0] / median if name.endswith("_overlap") and "raw_ms" in serial else None
            methods[name].append((median, tail, baseline_median / median, baseline_tail / tail, paired))
    return methods


def latency_section(lines, result, path, kind):
    groups = row_groups(result)
    rows = [row for group in groups for row in group.values()]
    methods = comparison(groups)
    lines.extend([f"### {kind} · `{path.stem}`", ""])
    samples = [len(row["raw_ms"]) for row in rows if "raw_ms" in row]
    failures = sum("capture_error" in row for row in rows)
    validation = result.get("validation", {}).get("passed", result.get("passed"))
    status = {True: "通过", False: "有失败", None: "未记录"}[validation]
    lines.append(f"共有 {len(groups)} 个测点，每方法每点 {interval(samples)} 次计时，校验状态为{status}，捕获失败 {failures} 条。{source(path)}")
    if kind == "Eager":
        configs = [case["config"] for case in result["experiments"]]
        shapes = sorted({(c["heads"], c["blocks"], c["block_size"], c["dim"], c["topk"]) for c in configs})
        lines.extend(["", "形状按 `(H, N, B, D, K)` 排列。" + "、".join(f"`{s}`" for s in shapes) + "。"])
        selectors = [timing(item["selector"]) for item in result["experiments"]]
        lines.append(f"单独运行 selector 的 p50 为 {interval([p[0] * 1000 for p in selectors])} µs，p95 为 {interval([p[1] * 1000 for p in selectors])} µs。")
        tails = [tail / median for row in rows if "raw_ms" in row for median, tail in [timing(row)]]
        lines.append(f"逐行 p95/p50 为 {interval(tails)}。这些重复样本来自同一基准进程，区间没有跨独立运行估计的置信界。")
    if kind != "Qwen QKV 回放":
        numerics = [r["numerics_vs_exact_topk"] for r in rows if r["method"] != "prr_union_overlap" and "numerics_vs_exact_topk" in r]
        if numerics:
            lines.extend(["", f"严格路径相对 FP64 oracle 通过 {sum(r['passed'] for r in numerics)}/{len(numerics)}，最大绝对误差 {max(r['max_abs_error'] for r in numerics):.3g}。"])
    else:
        overlaps = [r["overlap_fraction"] * 100 for r in rows if r["method"] == "serial_exact"]
        fallback = [r["fallback_heads"] for r in rows if r["method"] != "serial_exact"]
        errors = [r["numeric"]["max_abs_error"] for r in rows]
        execution = "、".join(sorted({r.get("timing", "eager") for r in rows}))
        lines.extend(["", f"模型为 `{result['model']}`，请求上下文 {result['context']} token，生成 {result['steps']} 步，采样层 {result['layers']}。从第二步起用上一解码步的块 ID 预测当前选择。计时使用 {execution}，每测点固定张量和块 ID。测点的平均逐头重叠率为 {statistics.mean(overlaps):.1f}%，范围 {interval(overlaps)}%。严格方法逐点回退头数 {interval(fallback)}，FP64 对照最大绝对误差 {max(errors):.3g}。"])
    summary = []
    for name, values in sorted(methods.items(), key=lambda item: (item[0] != "serial_exact", item[0])):
        median, tail, speed, tail_speed, paired = zip(*values)
        if name == "prr_union_overlap":
            speed_columns = ("—", "—", "—")
        else:
            speed_columns = (interval(speed), interval(tail_speed), f"{sum(x > 1 for x in speed)}/{len(values)}")
        summary.append([f"`{name}`", len(values), interval([x * 1000 for x in median]), interval([x * 1000 for x in tail]),
                        *speed_columns, interval([x for x in paired if x is not None])])
    lines.extend(["", *table(["方法", "测点", "p50 (µs)", "p95 (µs)", "相对基线 p50 (×)", "相对基线 p95 (×)", "p50 快于基线", "相对同算法串行 p50 (×)"], summary), ""])
    speculative = [name for name in methods if name.endswith(("_overlap", "_eps0")) and name.startswith(("saved", "hybrid", "compact"))]
    if speculative:
        wins = sum(v[2] > 1 for name in speculative for v in methods[name])
        total = sum(len(methods[name]) for name in speculative)
        paired = [v[4] for name in speculative for v in methods[name] if v[4] is not None]
        lines.extend([f"严格推测修复并行路径在 {wins}/{total} 个方法与测点配对中取得低于串行精确基线的 p50。", ""])
        if paired:
            lines.extend([f"并行路径在 {sum(x > 1 for x in paired)}/{len(paired)} 个方法与测点配对中取得低于同算法串行路径的 p50。", ""])
        if "experiments" in result:
            grouped = defaultdict(list)
            for group in groups:
                base = group.get("serial_exact", {})
                if "raw_ms" not in base:
                    continue
                c = base["config"]
                hit_rate = round(c["topk"] * c["overlap"]) / c["topk"]
                for name, values in comparison([group]).items():
                    if name in speculative:
                        grouped[hit_rate].append(values[0][2])
            summary = [[f"{100 * rate:g}%", interval(values), f"{sum(x > 1 for x in values)}/{len(values)}"]
                       for rate, values in sorted(grouped.items())]
            lines.extend([*table(["逐头块重叠率", "严格并行 p50 加速比 (×)", "p50 快于基线"], summary), ""])
    if "prr_union_overlap" in methods:
        union = [r for r in rows if r["method"] == "prr_union_overlap"]
        passed_union = sum(r["numerics_vs_own_semantics"]["passed"] for r in union)
        passed_strict = sum(r["numerics_vs_exact_topk"]["passed"] for r in union)
        max_error = max(r["numerics_vs_exact_topk"]["max_abs_error"] for r in union)
        lines.extend([f"`prr_union_overlap` 是自实现的并集语义对照。相对并集 FP64 oracle 通过 {passed_union}/{len(union)}，相对精确 Top-K 通过 {passed_strict}/{len(union)}，后者最大绝对误差 {max_error:.3g}。", ""])
    storage = []
    for mode in ["hybrid", "compact"]:
        for name in sorted(methods):
            if not (name.startswith(mode) and name.endswith("_overlap")):
                continue
            items = [r for r in rows if r["method"] == name and "prediction_live_tensor_storage_bytes" in r]
            if not items:
                continue
            state = [r["prediction_live_tensor_storage_bytes"] / 1024 for r in items]
            peak = [r["incremental_peak_torch_allocated_bytes"] / 1024 for r in items]
            kr = [r.get("logical_k_block_reads_estimate", r.get("logical_kv_block_reads_estimate")) / (r["config"]["heads"] * r["config"]["topk"]) for r in items]
            vr = [r.get("logical_v_block_reads_estimate", r.get("logical_kv_block_reads_estimate")) / (r["config"]["heads"] * r["config"]["topk"]) for r in items]
            storage.append([f"`{name}`", interval(state), interval(peak), interval(kr), interval(vr)])
    if storage:
        lines.extend(["预测存活状态按底层 tensor storage 去重统计。增量峰值来自 PyTorch allocator，包含计时路径临时 tensor。K、V 块读量按 `H × K` 归一化，是逻辑估算。compact 的统计阶段额外读取一次预测 K。", "",
                      *table(["方法", "预测存活状态 (KiB)", "增量峰值 (KiB)", "逻辑 K 块读 (×)", "逻辑 V 块读 (×)"], storage), ""])
        paired_storage = defaultdict(list)
        for group in groups:
            for name, compact in group.items():
                if not (name.startswith("compact_") and name.endswith("_overlap")):
                    continue
                hybrid = group.get(name.replace("compact_", "hybrid_", 1), {})
                if "prediction_live_tensor_storage_bytes" in compact and "prediction_live_tensor_storage_bytes" in hybrid:
                    for key in ["prediction_live_tensor_storage_bytes", "incremental_peak_torch_allocated_bytes"]:
                        paired_storage[key].append(100 * compact[key] / hybrid[key])
        if paired_storage:
            live = paired_storage["prediction_live_tensor_storage_bytes"]
            peaks = paired_storage["incremental_peak_torch_allocated_bytes"]
            lines.extend([f"同一形状、重叠率与 `τ` 配对时，compact 存活状态是 hybrid 的 {interval(live)}%，增量峰值是 hybrid 的 {interval(peaks, 5)}%。compact 每头保留槽数为 `min(S, floor(1/τ) + 1)`，`S` 为预测块数。槽数上限若等于 `S`，额外索引和标记仍占空间。", ""])
    if kind == "Qwen QKV 回放":
        trace_budget_section(lines, result)


def trace_budget_section(lines, result):
    groups = defaultdict(dict)
    for row in result["rows"]:
        groups[row["step"], row["layer"]][row["method"]] = row
    budgets = defaultdict(list)
    for group in groups.values():
        for row in group.values():
            if row.get("epsilon", 0) > 0:
                budgets[row["epsilon"]].append((row, group["serial_exact"], group["compact_fused_eps0"]))
    if not budgets:
        return
    lines.extend(["#### Qwen 回放的近似质量预算", "", "同一层、解码步和计时方式配对。严格数值校验与 `ε>0` 的近似误差分开统计。", ""])
    summary = []
    for epsilon, items in sorted(budgets.items()):
        measured = [timing(row) for row, _, _ in items]
        speeds = [timing(baseline)[0] / m[0] for (_, baseline, _), m in zip(items, measured)]
        strict = [timing(baseline)[0] / m[0] for (_, _, baseline), m in zip(items, measured)]
        bounds = [x for row, _, _ in items for x in row.get("extra_mass_bound", [])]
        approximate = [sum(row["approximate_heads"]) for row, _, _ in items if "approximate_heads" in row]
        errors = [row["numeric"]["max_abs_error"] for row, _, _ in items]
        summary.append([f"{epsilon:g}", len(items), interval([x[0] * 1000 for x in measured]), interval([x[1] * 1000 for x in measured]),
                        interval(speeds), f"{sum(x > 1 for x in speeds)}/{len(speeds)}", interval(strict), interval(approximate),
                        f"{max(bounds):.3g}" if bounds else "—", interval(errors)])
    lines.extend([*table(["ε", "测点", "p50 (µs)", "p95 (µs)", "串行精确 / 近似 p50 (×)", "p50 快于串行精确",
                         "ε=0 / 近似 p50 (×)", "近似头数", "最大质量上界", "最大绝对误差范围"], summary), ""])
    measurements = [item for items in budgets.values() for item in items]
    all_speed = [timing(exact)[0] / timing(row)[0] for row, exact, _ in measurements]
    all_relative = [timing(strict)[0] / timing(row)[0] for row, _, strict in measurements]
    lines.extend([f"Qwen 回放的近似完整路径在 {sum(x > 1 for x in all_speed)}/{len(all_speed)} 个配对中快于串行精确。相对 `ε=0` 的 p50 比为 {interval(all_relative)}×。", ""])


def correctness_section(lines, path, result):
    groups = defaultdict(list)
    for row in result["rows"]:
        groups[row["method"], row["dtype"]].append(row)
    adversarial = {row["case"] for row in result["rows"] if not row["case"].startswith("random")}
    lines.extend([f"### 数值正确性 · `{path.stem}`", "", f"Oracle 对相同的量化 Q/K/V 和精确选择的块 ID 使用 FP64。通过条件逐元素为 `|error| ≤ 3e-4 + 3e-4 × |reference|`，同时要求输出有限。覆盖一般随机输入、非二次幂形状、不同大小和无序块集合，以及 {len(adversarial)} 个对抗案例。" + source(path), ""])
    summary = [[method, dtype.removeprefix("torch."), f"{sum(r['passed'] for r in rows)}/{len(rows)}",
                f"{max(r['max_abs_error'] for r in rows):.3g}", f"{max(r['max_head_l2_relative'] for r in rows):.3g}",
                sum(r.get("fallback_heads", 0) for r in rows)] for (method, dtype), rows in sorted(groups.items())]
    lines.extend([*table(["方法", "输入精度", "通过", "最大绝对误差", "最大逐头相对 L2", "回退头数总和"], summary), ""])


def guard_section(lines, path, result):
    rows = result["rows"]
    groups = defaultdict(list)
    for row in rows:
        groups[row["min_remaining"], row["cancellation_tol"]].append(row)
    lines.extend(["### 数值守卫消融", "", f"共有 {len(rows)} 条记录，FP16 对抗案例与质量阈值 `τ` 的组合在不同守卫参数下测试。计时仅包括修复阶段。剩余质量和消去容差均为零时，非有限值等硬条件仍可触发回退。{source(path)}", "",
                  "消去守卫逐坐标比较相减残差与数值尺度。尺度计入低质量组的绝对分子包络，用于识别组内累加已经发生的消去。触发守卫的头重新计算精确选择集合上的注意力。", ""])
    summary = []
    for (mass, cancel), items in sorted(groups.items()):
        summary.append([f"{mass:g}", f"{cancel:g}", f"{sum(not r['passed'] for r in items)}/{len(items)}",
                        sum(r["fallback_heads"] for r in items), interval([timing(r)[0] * 1000 for r in items]),
                        interval([timing(r)[1] * 1000 for r in items])])
    lines.extend([*table(["剩余质量下限", "消去容差", "失败", "回退头数总和", "p50 (µs)", "p95 (µs)"], summary), ""])
    failures = sorted({r["case"] for r in rows if not r["passed"]})
    if failures:
        lines.extend(["出现失败的案例包括 " + "、".join(f"`{name}`" for name in failures) + "。", ""])


def fusion_groups(result):
    groups = defaultdict(dict)
    for row in result["rows"]:
        config = tuple(sorted(row["config"].items()))
        groups[row["timing"], row["stage"], config][row["method"]] = row
    return groups


def fusion_section(lines, path, result):
    rows = result["rows"]
    lines.extend([f"### 修复融合 · `{path.stem}`", "", "完整路径计时包含 selector、推测、流汇合和修复，修复单测直接复用已选 ID 和预测状态。按同一执行模式、阶段、配置与预测方法配对。`split / fused` 大于 1 表示融合路径更快。" + source(path), ""])
    if "correctness" in result:
        checks = result["correctness"]["rows"]
        lines.extend([f"融合正确性套件通过 {sum(r['passed'] for r in checks)}/{len(checks)}，最大绝对误差 {max(r['max_abs_error'] for r in checks):.3g}。", ""])
    if not rows:
        lines.extend(["该文件仅包含正确性结果。", ""])
        return
    lines.extend([f"计时输出相对 FP64 oracle 通过 {sum(r['numerics_vs_exact_topk']['passed'] for r in rows)}/{len(rows)}，每行 {interval([len(r['raw_ms']) for r in rows])} 次计时。", ""])
    groups = fusion_groups(result)
    selectors = sorted({r["config"].get("selector", "torch") for r in rows})
    if selectors == ["fused"]:
        lines.extend(["完整路径的各方法及其串行精确基线共用 Triton 融合 min/max 打分与 `torch.topk`。", ""])
    if any(r["config"].get("heterogeneous") for r in rows):
        lines.extend(["异质重叠配置的一半头重叠 0%，另一半头重叠 100%。", ""])
    compact_tails = [timing(g["compact_split"])[1] / timing(g["compact_fused"])[1]
                     for (kind, stage, _), g in groups.items() if (kind, stage) == ("graph", "repair")
                     and "compact_split" in g and "compact_fused" in g]
    if compact_tails:
        lines.extend([f"CUDA Graph 的 compact 修复阶段，融合前后 p95 比为 {interval(compact_tails)}×，融合在 {sum(x > 1 for x in compact_tails)}/{len(compact_tails)} 个配置缩短尾部延迟。", ""])
    for execution in ["eager", "graph"]:
        for stage in ["repair", "full"]:
            configurations = [group for (kind, part, _), group in groups.items() if kind == execution and part == stage]
            if not configurations:
                continue
            summary = []
            for name in ["serial_exact", "saved", "hybrid", "compact"]:
                if name == "serial_exact":
                    baseline = [timing(g[name]) for g in configurations if name in g]
                    if baseline:
                        median, tail = zip(*baseline)
                        summary.append([f"`{name}`", len(baseline), interval([m * 1000 for m in median]), interval([t * 1000 for t in tail]), "—", "—", "—", "—", "—"])
                    continue
                matched = [g for g in configurations if f"{name}_split" in g and f"{name}_fused" in g]
                if not matched:
                    continue
                split = [timing(g[f"{name}_split"]) for g in matched]
                fused = [timing(g[f"{name}_fused"]) for g in matched]
                median_ratio = [s[0] / f[0] for s, f in zip(split, fused)]
                tail_ratio = [s[1] / f[1] for s, f in zip(split, fused)]
                baseline_ratio = [timing(g["serial_exact"])[0] / f[0] for g, f in zip(matched, fused) if "serial_exact" in g]
                summary.append([f"`{name}`", len(matched), interval([s[0] * 1000 for s in split]), interval([s[1] * 1000 for s in split]),
                                interval([f[0] * 1000 for f in fused]), interval([f[1] * 1000 for f in fused]),
                                interval(median_ratio), interval(tail_ratio), interval(baseline_ratio)])
            title = "修复" if stage == "repair" else "完整路径"
            lines.extend([f"#### {execution} · {title}", "", *table([
                "方法", "配置", "split p50 (µs)", "split p95 (µs)", "fused p50 (µs)", "fused p95 (µs)",
                "split / fused p50 (×)", "split / fused p95 (×)", "串行精确 / fused p50 (×)"], summary), ""])
    for execution in ["eager", "graph"]:
        pairs = [(dict(cfg), name, timing(group["serial_exact"])[0] / timing(row)[0])
                 for (kind, stage, cfg), group in groups.items() if kind == execution and stage == "full" and "serial_exact" in group
                 for name, row in group.items() if name.endswith("_fused")]
        if not pairs:
            continue
        wins = [(cfg, method, ratio) for cfg, method, ratio in pairs if ratio > 1]
        lines.extend([f"{execution} 完整路径中，融合方法有 {len(wins)}/{len(pairs)} 个方法与配置配对的 p50 快于同选择器的串行精确基线。", ""])
        if 0 < len(wins) <= 5:
            cases = [f"`{method}`，H={cfg['heads']}、B={cfg['block_size']}、D={cfg['dim']}、K={cfg['topk']}、块重叠 {cfg['overlap'] * 100:g}%，{ratio:.3g}×" for cfg, method, ratio in wins]
            lines.extend([*(f"- {case}。" for case in cases), ""])


def plot_fusion(plt, datasets, output):
    runs = [(path, result) for kind, path, result in datasets if kind == "修复融合" and result["rows"]]
    if not runs:
        return False
    path, result = max(runs, key=lambda item: len(item[1]["rows"]))
    groups = fusion_groups(result)
    figure, axes = plt.subplots(2, 2, figsize=(10, 6.7), layout="constrained")
    for i, execution in enumerate(["eager", "graph"]):
        for j, stage in enumerate(["repair", "full"]):
            ax = axes[i, j]
            configurations = [g for (kind, part, _), g in groups.items() if kind == execution and part == stage]
            names = ["saved", "hybrid", "compact"]
            for index, name in enumerate(names):
                pairs = [(timing(g[f"{name}_split"]), timing(g[f"{name}_fused"])) for g in configurations
                         if f"{name}_split" in g and f"{name}_fused" in g]
                if not pairs:
                    continue
                for column, offset, marker in [(0, -.15, "o"), (1, .15, "s")]:
                    ratios = [s[column] / f[column] for s, f in pairs]
                    ax.hlines(index + offset, min(ratios), max(ratios), color=COLORS[name], lw=2)
                    ax.scatter(statistics.median(ratios), index + offset, color=COLORS[name], marker=marker, s=28)
            ax.set_yticks(range(len(names)), names)
            ax.invert_yaxis()
            ax.axvline(1, color="0.35", ls="--", lw=1)
            ax.set_title(f"{execution.capitalize()} · {stage} · {len(configurations)} configurations")
            ax.grid(axis="x", color="0.9", lw=.6)
            ax.set_xlabel("Split / fused latency (×)")
    figure.suptitle(f"Fusion ablation · {path.stem}\nRanges and medians across configurations (circles: p50 ratios, squares: p95 ratios)", fontsize=13)
    figure.savefig(output / "fusion_latency.png", dpi=180)
    plt.close(figure)
    return True


def plot_latency(plt, datasets, output):
    selected = []
    for kind in ["Eager", "CUDA Graph", "Qwen QKV 回放"]:
        matches = [(path, result) for category, path, result in datasets if category == kind]
        if matches:
            selected.append((kind, max(matches, key=lambda item: len(row_groups(item[1])))))
    if not selected:
        return False
    figure, axes = plt.subplots(1, len(selected), figsize=(5.3 * len(selected), 5.3), squeeze=False, layout="constrained")
    for ax, (kind, (path, result)) in zip(axes[0], selected):
        methods = comparison(row_groups(result))
        overlaps = sorted(name for name in methods if name.endswith(("_overlap", "_eps0")) and not name.startswith("prr_"))
        for idx, name in enumerate(overlaps):
            color = COLORS[name.split("_")[0]]
            for column, offset, marker in [(2, -.15, "o"), (3, .15, "s")]:
                values = [row[column] for row in methods[name]]
                ax.hlines(idx + offset, min(values), max(values), color=color, lw=2)
                ax.scatter(statistics.median(values), idx + offset, marker=marker, color=color, s=28, zorder=3)
        title = {"Eager": "Eager", "CUDA Graph": "CUDA Graph", "Qwen QKV 回放": "Qwen QKV replay"}[kind]
        ax.set_title(f"{title} · {len(row_groups(result))} points\n{path.stem}")
        ax.set_yticks(range(len(overlaps)), [name.removesuffix("_overlap").replace("_tau", " τ=").replace("_fused_eps0", " fused ε=0") for name in overlaps])
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.axvline(1, color="0.3", ls="--", lw=1)
        ax.grid(axis="x", which="both", color="0.9", lw=.6)
        ax.set_xlabel("Serial exact / overlap latency (×)")
    figure.suptitle("Latency ratios across matched workloads\nRanges and medians across points (circles: p50 ratios, squares: p95 ratios)", fontsize=13)
    figure.savefig(output / "latency_ranges.png", dpi=180)
    plt.close(figure)
    return True


def plot_storage(plt, datasets, output):
    runs = [(path, result) for kind, path, result in datasets if kind == "Eager"]
    if not runs:
        return False
    path, result = max(runs, key=lambda item: len(item[1]["experiments"]))
    shape_values = defaultdict(lambda: defaultdict(list))
    for experiment in result["experiments"]:
        c = experiment["config"]
        shape = (c["heads"], c["topk"], c["dim"])
        for row in experiment["rows"]:
            name = row["method"]
            if name.endswith("_overlap") and "prediction_live_tensor_storage_bytes" in row:
                shape_values[shape][name.removesuffix("_overlap")].append(row)
    if not shape_values:
        return False
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), layout="constrained")
    names = sorted({name for methods in shape_values.values() for name in methods})
    shapes = sorted(shape_values)
    for ax, key, ylabel in zip(axes, ["prediction_live_tensor_storage_bytes", "incremental_peak_torch_allocated_bytes"],
                               ["Live prediction storage (KiB)", "Incremental allocator peak (KiB)"]):
        width = .77 / len(names)
        for j, name in enumerate(names):
            color = COLORS[name.split("_")[0]]
            for i, shape in enumerate(shapes):
                values = [r[key] / 1024 for r in shape_values[shape][name]]
                if values:
                    x = i + width * (j - (len(names) - 1) / 2)
                    ax.vlines(x, min(values), max(values), color=color, lw=2)
                    ax.scatter(x, statistics.median(values), color=color, marker=["o", "s", "^"][j % 3], s=28,
                               label=name.replace("_tau", " τ=") if i == 0 else None)
        ax.set_xticks(range(len(shapes)), [f"{h}, {k}, {d}" for h, k, d in shapes])
        ax.set_xlabel("Heads, selected blocks, head dimension")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(axis="y", which="both", color="0.9", lw=.6)
    axes[-1].legend(loc="upper left", ncol=2, fontsize=8)
    figure.suptitle(f"Prediction-state storage and allocator peak · {path.stem}\nMarkers show medians; lines span tested configurations for each shape", fontsize=13)
    figure.savefig(output / "prediction_storage.png", dpi=180)
    plt.close(figure)
    return True


def pareto_section(lines, path, result):
    rows = result["rows"]
    lines.extend([f"### 允许额外注意力质量时的精度与延迟 · `{path.stem}`", "",
                  "固定形状的 CUDA Graph 使用 compact 推测及融合修复。`ε` 限制可留在输出中的错误块概率质量，`ε=0` 使用严格路径。质量上界按实现记录的逐头 `extra_mass_bound` 汇总，输出误差则以精确所选集合的 FP64 oracle 实测。两者量纲不同。" + source(path), ""])
    if not rows:
        return
    groups = defaultdict(list)
    for row in rows:
        groups[row["stage"], row["epsilon"]].append(row)
    status = {True: "通过", False: "有失败", None: "未记录"}
    lines.extend([f"严格精度检查{status[result.get('strict_checks_passed')]}，质量预算检查{status[result.get('mass_budget_checks_passed')]}。合成实验另外用 FP64 在所选块和保留错误块的并集上独立重算错误质量。每条记录 {interval([len(r['raw_ms']) for r in rows])} 次计时。", ""])
    summary = []
    for (stage, epsilon), items in sorted(groups.items()):
        medians, tails = zip(*(timing(r) for r in items))
        bounds = [value for r in items for value in r.get("extra_mass_bound", [])]
        actual = [value for r in items for value in r.get("fp64_actual_extra_mass", [])]
        errors = [r["numeric"]["max_abs_error"] for r in items]
        approx = [sum(r["approximate_heads"]) for r in items if "approximate_heads" in r]
        speed = [r["serial_exact_median_ms"] / median for r, median in zip(items, medians)] if stage == "full" else []
        summary.append(["完整" if stage == "full" else "修复", f"{epsilon:g}", len(items),
                        interval([x * 1000 for x in medians]), interval([x * 1000 for x in tails]),
                        interval(speed), interval(approx), f"{max(bounds):.3g}" if bounds else "—",
                        f"{max(actual):.3g}" if actual else "—",
                        interval(errors), sum(r["fallback_heads"] for r in items)])
    lines.extend([*table(["阶段", "ε", "配置", "p50 (µs)", "p95 (µs)", "串行精确 / 该方法 p50 (×)", "近似头数", "最大质量上界", "FP64 最大错误质量", "最大绝对误差范围", "回退头数总和"], summary), ""])
    full = [r for r in rows if r["stage"] == "full" and r["epsilon"] > 0]
    strict = {tuple(sorted(r["config"].items())): r for r in rows if r["stage"] == "full" and r["epsilon"] == 0}
    if full:
        wins = sum(timing(r)[0] < r["serial_exact_median_ms"] for r in full)
        pairs = [(strict[tuple(sorted(r["config"].items()))], r) for r in full if tuple(sorted(r["config"].items())) in strict]
        lines.extend([f"在 `ε>0` 的完整路径中，{wins}/{len(full)} 个配置与预算配对的 p50 快于串行精确基线。相对同配置 `ε=0`，p50 加速比为 {interval([timing(a)[0] / timing(b)[0] for a, b in pairs])}×。", ""])


def plot_pareto(plt, datasets, output):
    runs = [(path, result) for kind, path, result in datasets if kind == "质量预算" and result["rows"]]
    if not runs:
        return False
    path, result = max(runs, key=lambda item: len(item[1]["rows"]))
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 5.1), layout="constrained")
    for ax, stage, column in zip(axes, ["full", "repair"], [0, 1]):
        groups = defaultdict(list)
        for row in result["rows"]:
            if row["stage"] == stage:
                groups[tuple(sorted(row["config"].items()))].append(row)
        for config, rows in sorted(groups.items()):
            cfg = dict(config)
            points = []
            for row in sorted(rows, key=lambda row: row["epsilon"]):
                latency = timing(row)[column]
                normalized = latency / row["serial_exact_median_ms"] if stage == "full" else latency * 1000
                points.append((row["numeric"]["max_abs_error"], normalized))
            x, y = zip(*points)
            label = f"{100 * cfg['overlap']:g}% hit, Q × {cfg['q_scale']:g}"
            line, = ax.plot(x, y, color=None, alpha=.65, marker=".", lw=1, label=label)
            frontier = [(a, b) for a, b in set(points) if not any(c <= a and d <= b and (c, d) != (a, b) for c, d in points)]
            fx, fy = zip(*frontier)
            ax.scatter(fx, fy, color=line.get_color(), s=28, zorder=3)
            ax.scatter(x[0], y[0], facecolors="none", edgecolors=line.get_color(), marker="s", s=45, zorder=4)
        ax.set_xscale("symlog", linthresh=1e-7)
        ax.set_xlabel("Maximum absolute output error vs FP64")
        ax.grid(color="0.9", lw=.6)
    axes[0].set_title("Full path · p50 normalized to serial exact")
    axes[0].set_ylabel("Method / serial exact latency (×)")
    axes[0].axhline(1, color="0.3", ls="--", lw=1)
    axes[1].set_title("Repair only · p95")
    axes[1].set_ylabel("Latency (µs)")
    axes[1].legend(fontsize=8)
    figure.suptitle(f"Accuracy–latency sweep · {path.stem}\nSquares mark ε=0; large dots mark nondominated budgets within each workload", fontsize=13)
    figure.savefig(output / "accuracy_latency.png", dpi=180)
    plt.close(figure)
    return True


def classify(result):
    if result.get("scope", "").startswith("approximate compact fused repair"):
        return "质量预算"
    if result.get("scope", "").startswith("fusion ablation"):
        return "修复融合"
    if "experiments" in result:
        return "CUDA Graph" if result.get("scope", "").startswith("Static CUDA Graph") else "Eager"
    if "model" in result and "rows" in result:
        return "Qwen QKV 回放"
    if "oracle" in result:
        return "数值正确性"
    if result.get("scope", "").startswith("Repair-only timing"):
        return "守卫消融"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "results.md")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures")
    args = parser.parse_args()

    datasets = []
    for path in sorted(args.results.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8-sig"))
        kind = classify(result)
        if kind:
            datasets.append((kind, path, result))
    if not datasets:
        parser.error("no recognized result files found")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.figures.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Strict Top-K 实验结果", "",
             "本文保留第一轮结果。最新结果见 [PRR 基线接口对齐](paper_baseline_results.md)、[容器内 CUDA profiling](profiling.md)、[HiSparse 缓存回放](hisparse_trace.md)与[接入设计](hisparse_integration.md)。此前的同库对照见 [FlashInfer 双边接入报告](vendor_backend_results.md)，选择窗口与排名分析见 [窗口实验](window_results.md)。", "",
             "串行精确基线先执行块选择，再用同一套 split-block Triton 原型计算所选块注意力。块选择使用预先生成的 K 向量分量最小值和最大值、按 Q 符号取界并执行 `torch.topk`。Eager 的 CUDA event 区间包含主机提交间隙和 CUDA 流等待。CUDA Graph 回放固定输入和块 ID，排除图捕获时间。两类计时均排除编译、块元数据构建与预测 ID 构造。", "",
             "下表的 p50、p95 来自每测点的全部 CUDA event 样本，p95 采用线性插值。区间表示各测点统计量的最小值到最大值。加速比按同一测点的串行精确基线除以对应方法计算，大于 1 表示对应方法更快。`attention_only` 预先给定所选 ID，仅提供注意力部分的耗时参考。", ""]
    environments = set()
    for _, _, result in datasets:
        env = result.get("environment", {})
        keys = ["gpu", "cuda_runtime", "torch", "triton"]
        if all(key in env for key in keys):
            environments.add(tuple(str(env[key]) for key in keys))
    if environments:
        lines.extend(["## 环境", "", *table(["GPU", "CUDA runtime", "PyTorch", "Triton"], sorted(environments)), ""])
    if (args.results / "run_manifest.json").is_file():
        lines.extend(["GPU 基准的执行记录与源码摘要见 [run_manifest.json](../results/run_manifest.json)。", ""])

    if plot_latency(plt, datasets, args.figures):
        lines.extend(["![同测点的 p50 和 p95 加速比区间](../figures/latency_ranges.png)", ""])
    if plot_storage(plt, datasets, args.figures):
        lines.extend(["![预测存活状态和增量分配峰值](../figures/prediction_storage.png)", ""])
    if plot_fusion(plt, datasets, args.figures):
        lines.extend(["![融合前后的修复及完整路径延迟](../figures/fusion_latency.png)", ""])
    if plot_pareto(plt, datasets, args.figures):
        lines.extend(["![质量预算的精度与延迟测点](../figures/accuracy_latency.png)", ""])

    lines.extend(["## 延迟与内存", ""])
    for kind in ["Eager", "CUDA Graph", "Qwen QKV 回放"]:
        for category, path, result in datasets:
            if category == kind:
                latency_section(lines, result, path, kind)
    for kind, path, result in datasets:
        if kind == "修复融合":
            fusion_section(lines, path, result)
        if kind == "质量预算":
            pareto_section(lines, path, result)
    lines.extend(["## 数值与回退", ""])
    for kind, path, result in datasets:
        if kind == "数值正确性":
            correctness_section(lines, path, result)
        if kind == "守卫消融":
            guard_section(lines, path, result)
    lines.extend(["## 结论边界", "",
                  "合成实验使用受控重叠率、随机 Q/K/V 和单卡固定形状。回放使用真实密集解码的 Q/K/V、完整 32-token 块与扩展的 GQA KV。证据适用于这套原型的选择与注意力阶段。实际 HBM/L2 流量、与厂商优化内核的相对性能、完整模型吞吐及生成质量均需相应实验。", "",
                  "可用 `python -m benchmarks.summarize` 重新生成本文和图表，需安装 matplotlib。", ""])
    args.report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
