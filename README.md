# Strict Top-K Overlap

用当前 query 提前计算预测块，在正式 Top-K 完成后修正预测错误。项目比较逐块保存、混合扣除、紧凑状态及融合修正，并研究允许少量额外 softmax 质量时的精度—速度折中。

这是一个 A100 上实测的 Triton 研究原型。完整数字、原始采样和图表见 [实验报告](docs/results.md) 与 [results](results/)。性能基线为同一 split-block attention 原型；没有声称超越 FlashAttention/FlashInfer 或完成模型端到端加速。

本轮结果：严格分离/融合路径分别通过 304 项 FP64 对照检查；融合使紧凑修正阶段的 p95 改善 1.05–1.60×。完整路径只在一组 100% 预测命中的合成配置中出现净加速（saved 1.14×、hybrid 1.09×）。Qwen 回放平均块重叠率 78.9%，严格和近似路径都未快于串行基线。允许额外质量可以少做撤销，但本轮尚未得到优于串行基线的实用精度—速度折中。

## 方法

- `saved`：每个预测块保存 FP32 softmax 状态，只合并命中块并补算漏块。
- `hybrid`：实际 attention 质量超过阈值的块单存，其余聚合；错块从聚合状态扣除。这一参考实现先物化所有逐块状态。
- `compact`：先计算块级 QK 统计，再生成低质量聚合和有界高质量状态；额外 QK 遍历换取较少常驻状态。
- `repair_fused`：在一个 kernel 中做集合判定、错块扣除、漏块补算、合并及保护性重算，减少选择完成后的修正尾部。
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

真实 Qwen Q/K/V 回放使用仓库内自写公开提示词和上一解码步的选择列表。模型仍执行 dense generation；回放只处理完整的 32-token 块，并展开 GQA KV，不代表稀疏模型推理质量。

## 实验边界

CUDA Graph 数据、形状与预测列表固定；eager 结果包含 Python 提交间隙。合成实验控制预测重叠率，不包含真实预测器代价。选择器使用 block min/max 的 query-sign bound 与 `torch.topk`，预先构造元数据。无 CUPTI 的逻辑 KV 读取估计不等于 HBM 流量；touch-prefetch 不保证缓存驻留。尾部 p95 是固定配置的重复采样，不是线上请求分布。

实现细节：[内核](docs/KERNELS.md)、[融合](docs/FUSION.md)、[数值审查](docs/review.md)。

提交作者与提交者均使用匿名身份；公开仓库的账号归属仍然可见。

`results/run_manifest.json` 保存 GPU 执行阶段的源码散列和退出状态。报告生成器随后更新，最终发布文件另由 `results/source_manifest.json` 记录；测量使用的其他 Python 源码均与发布版本一致。
