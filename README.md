# Strict Top-K Overlap

用当前 query 提前计算预测块，在正式 Top-K 完成后修正预测错误。项目比较逐块保存、混合扣除、紧凑状态及融合修正，并研究允许少量额外 softmax 质量时的精度—速度折中。

这是一个 A100 上实测的 Triton 研究原型。完整数字见 [第二轮实验报告](docs/round2_results.md)、[第一轮实验报告](docs/results.md) 与 [results](results/)。主要性能基线为同一选择器下的串行精确注意力；第二轮另测 FlashInfer 与 PyTorch Flash SDPA。尚未完成模型端到端加速。

**Qwen 在这些实验中仍执行稠密注意力。** 回放提取其 Q/K/V，再额外运行本项目的 Quest 风格块级 min/max 打分与 `torch.topk`；这不是 Qwen 原生的稀疏选择器，也没有接入模型的实际注意力输出。报告中的“完整路径”仅包含回放的选择、推测与修复，不包含整层或整模型解码。元数据构建与预测 ID 构造不计时；预测采用上一步选择列表，没有实现 PRR 的 EMA。当前结果不能证明稀疏模型吞吐或任务准确率改善。

第二轮使用原生 GQA、分块并行补算、轻量单次启动修复和相邻基线配对计时。轻量修复在 64K 上下文、选 64 块、100% 命中的合成配置中跨三个种子达到 1.144–1.158×；两款 Qwen 的 16K/约 32K 真实 QKV 回放各 18 个测点，所有推测完整路径均慢于串行基线。同一逐块后端下严格与并集路径耗时接近，不能把原型与 PRR 的差距归因于错块过滤。并集对照没有运行原 PRR 的聚合修复内核。[概念与 PRR 对照](docs/design_explained.md)

第一轮严格分离/融合路径分别通过 304 项 FP64 对照检查；融合使紧凑修正阶段的 p95 改善 1.05–1.60×，但 compact 完整路径没有净收益。允许额外质量可以少做撤销，目前仍未得到优于串行基线的实用精度—速度折中。

新增 [选择窗口与预测预算报告](docs/window_results.md)：先测串行 `score + topk` 与 attention，仅对选择较快的配置添加合成 GPU 工作至约 1:1，并在相同串行对照中计入它。三个种子、90 个测点仍未获得超过所测 FlashInfer 适配基线的净收益。FlashNext/QSA 部分采用公开形状和独立评分内核的随机输入，没有加载其模型权重。另用 dense Qwen3-4B 轨迹比较上一步、前一层与排名混合；取前 51/64 块把跨步预测精确率（命中数/预测数）从 74.3% 提高到 81.6%，覆盖率降到 65.0%，尚不能据此认定这一预算最快。

## 方法

- `saved`：每个预测块保存 FP32 softmax 状态，只合并命中块并补算漏块。
- `hybrid`：实际 attention 质量超过阈值的块单存，其余聚合；错块从聚合状态扣除。这一参考实现先物化所有逐块状态。
- `compact`：先计算块级 QK 统计，再生成低质量聚合和有界高质量状态；额外 QK 遍历换取较少常驻状态。
- `repair_fused`：在一个 kernel 中做集合判定、错块扣除、漏块补算、合并及保护性重算，减少选择完成后的修正尾部。
- `gqa` / `gqa_light`：直接读取共享 KV 头；分别提供分块并行修复、每头单次启动修复及基于漏块估计的调度。估计错误只影响性能，两条分支都计算严格 Top-K。
- `signed_fused`：漏选块与需撤销的低质量错块共用一个按块并行的 QK/V kernel，分别带正负号；另做任务排布和带守卫的状态合并，总计三次启动。合并内包含必要的严格回退。[加减融合实验](docs/signed_results.md)
- `gqa_hierarchical`：对四-token 小块按 8 块/CTA 打包计算，保留逐块状态并分层归并；共享选集时复用一份匹配表。与同轮旧实现比较，QSA 形状 attention 阶段从约 425 µs 降到 26 µs；这是注意力内核对照，不是模型解码收益。
- `select_blocks`：融合 min/max 界打分，避免大尺寸逐元素中间张量，随后执行 `torch.topk`。融合/预算/真实回放实验及其串行基线均使用这一选择器。
- 可选 `max_extra_mass`：对低质量误选块计算质量占比上界，小于预算时保留它们，跳过重读撤销。它控制额外 softmax 质量，不直接保证输出相对误差或任务准确率。

严格模式保持选中集合语义，但归约顺序和 FP32 舍入不同。分母检查和绝对值累加量用于触发重算，仍不是普适浮点误差证书。原始 PRR 的 union 语义与本项目不同；现有官方逐块状态接口是强基线，见 [相关工作](docs/related_work.md)。

## 复现

需要 NVIDIA GPU、兼容 CUDA 的 PyTorch 和 Triton。实测版本在 [requirements-tested.txt](requirements-tested.txt)。以下命令从仓库根目录运行，不会下载模型。

```bash
export PYTHONPATH="$PWD"
export TRITON_CACHE_DIR="$PWD/.triton-cache"
python -m tests.check_correctness --include-compact --output results/correctness.json
python -m benchmarks.run_benchmark --include-compact --profile main --warmup 5 --repeats 30 --output results/eager.json
python -m benchmarks.graph_benchmark --include-compact --repeats 60 --output results/graph.json
python -m benchmarks.guard_ablation --output results/guard_ablation.json
python -m tests.check_selector
python -m benchmarks.fusion_benchmark --selector fused --repeats 60 --output results/fusion.json
python -m benchmarks.pareto_benchmark --repeats 60 --output results/pareto.json
python -m experiments.trace_replay --model /path/to/Qwen3-0.6B-Base --context 2048 --steps 8 --graph --output results/model_trace.json
python benchmarks/summarize.py
```

第二轮复现命令（模型路径指向已有本地权重；厂商对照额外需要 FlashInfer 0.3.1）：

```bash
python -m experiments.run_round2 --model-small /path/to/Qwen3-0.6B-Base --model-large /path/to/Qwen3-4B
python -m experiments.run_round2_confirm --model-small /path/to/Qwen3-0.6B-Base --model-large /path/to/Qwen3-4B
python -m benchmarks.methodology_audit --output results/round2/methodology_audit.json
python -m benchmarks.vendor_baseline --output results/round2/vendor_baseline.json
python benchmarks/summarize_round2.py
python -m experiments.run_signed --model-small /path/to/Qwen3-0.6B-Base --model-large /path/to/Qwen3-4B
python -m experiments.run_signed_fused --model-small /path/to/Qwen3-0.6B-Base --model-large /path/to/Qwen3-4B
python benchmarks/summarize_signed.py
python -m tests.check_gqa_hierarchical
python -m experiments.run_window --model /path/to/Qwen3-4B
python -m benchmarks.summarize_window
python -m benchmarks.plot_predictor
```

真实 Qwen Q/K/V 回放使用仓库内自写公开提示词和上一解码步的选择列表。模型仍执行 dense generation；回放只处理完整的 32-token 块。第一轮展开 GQA KV，第二轮保持共享 KV 头的原生布局。两轮均不代表稀疏模型推理质量。

## 实验边界

CUDA Graph 数据、形状与预测列表固定；eager 结果包含 Python 提交间隙。合成实验控制预测重叠率，不包含真实预测器代价。选择器使用 block min/max 的 query-sign bound 与 `torch.topk`，预先构造元数据。无 CUPTI 的逻辑 KV 读取估计不等于 HBM 流量；touch-prefetch 不保证缓存驻留。第二轮 p95 是批量回放均值的分位数，不能解释为线上请求尾延迟。`zero_repair_overlap` 不产出注意力结果，只用于诊断现有调度的剩余时间。

实现细节：[内核](docs/KERNELS.md)、[融合](docs/FUSION.md)、[数值审查](docs/review.md)、[PRR 官方源码审查](docs/prr_implementation.md)。

提交作者与提交者均使用匿名身份；公开仓库的账号归属仍然可见。

各轮 `run_manifest.json` 保存执行时源码快照和阶段退出状态，`source_manifest.json` 保存该轮发布文件摘要；第一轮快照保留不变。第二轮另有 `confirm_manifest.json` 记录配对复测。报告生成器和厂商适配开发期间存在后续更新，具体快照差异由第二轮发布清单记录。
