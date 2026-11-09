# HiSparse 接入设计与对照实验

当前证据来自 CPU 轨迹回放、独立 GPU 注意力基准和原型的 CUDA Graph 时间线。缓存回放使用密集 Qwen3-4B 的外接 Quest 排名，按层与查询头维护独立的逻辑缓存。其 18 个实际发起预取的配置有 10 个减少需求缺块，全部增加总搬运。纯上一解码步预测的搬运为零，因为容量至少为 K 且上一选集仍驻留。GPU 结果测量的是本项目原型。这些实验尚未接入 SGLang HiSparse 的缓存、搬运或真实解码路径。[CPU 缓存回放](hisparse_trace.md) · [GPU 注意力基准](paper_baseline_results.md) · [原型时间线](../results/profiling/summary.json)

## 官方路径与接入点

HiSparse 在 decode 端将完整 KV 保存在 pinned host pool，并为每个请求在 GPU 留热缓冲。选择器产出逻辑 Top-K 后，swap-in 查找驻留命中、更新 LRU、搬入缺失项，并返回后续注意力要读的设备 slot 表。源码的 `HiSparseCoordinator._run_swap_in_kernel` 按层读取设备缓冲和元数据。需按目标模型选择对应入口和布局，例如 DSA 的 `swap_in_selected_pages` 与 MiniMax M3 的 `swap_in_selected_blocks`。[SGLang HiSparse 指南](https://docs.sglang.io/docs/advanced_features/hisparse_guide) · [HiSparseCoordinator 源码](https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/managers/hisparse_coordinator.py)

原生 shared-index prefetch 在 anchor 已经算出精确选集后记录 host→device 的 miss plan。后续 skip 层按同一索引和同步的 slot 布局执行 copy-only，在使用前等待各自的完成事件。它依赖模型的 IndexShare 配置。将前一层排名用于具有独立选择器的下一层属于预测，需要在真实选择到达后校验和修复。原生 exact copy 的收益应单独测量。[SGLang shared-index prefetch](https://docs.sglang.io/docs/advanced_features/hisparse_guide#shared-index-prefetch-automatic) · [HiSparseCoordinator 源码](https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/managers/hisparse_coordinator.py)

## 第一阶段只使用已驻留的预测块

在目标层 Q 可用后启动两条流。`R` 为目标请求、目标层当前可读的驻留逻辑集合，`P` 为选择完成前已知的预测 ID，`A` 为当前原生选择器给出的真实选集。

1. 主流运行原生选择器。推测流仅查阅驻留表，在 `P ∩ R` 的稳定 slot 上用本步 Q 计算并保存逐块 softmax 状态。查表不更新原生 LRU，不产生 H2D。优先实现上一解码步同层 ID 的来源。
2. 第一版在原生 swap-in 之前等待推测读取完成。swap-in 仍按 `A` 和现有策略搬运并产生设备 slot 表。这样只在选择窗口内尝试隐藏驻留计算，保留官方的搬运及淘汰路径。`A \ R` 表示进入需求阶段时缺少的 KV，和注意力补算的集合是两个不同指标。
3. 令 `B` 为已完成且有效的逐块状态对应的 ID。按原生选集保留 `A ∩ B`，对 `A \ B` 用 swap-in 后的设备 slot 补算，并合并得到严格选集的注意力。`B = P ∩ R` 时，`A \ B` 包括 `A \ P` 与 `A ∩ (P \ R)`。其中也可包含已驻留但未预测的块。若无法产出可靠的逐块状态或 slot 解析失败，退回原生严格注意力。

例如 `R={1,2,3}`、`P={1,2,4}`、`A={2,3,5}`。推测读取 1、2，保留 2，需求 H2D 只搬 5，修复计算 3、5。块 3 已驻留却仍需算，块 1 的已算状态直接丢弃。

### Slot 生命周期与后续搬运

- **读完再覆盖。** 普通 CUDA `record_stream` 管理分配生命周期，无法阻止同一 KV pool slot 被后续 swap-in 原地覆盖。第一版依靠读取完成事件。进一步重叠搬运时，查表与标记读占用必须与受控淘汰协调。swap-in 对未占用 slot 可立即搬运，对候选占用 slot 等待读完或重选；当无可用 slot 时先释放占用再继续，尤其是容量恰好为 K 时。待写入的 slot 只在搬运完成后供读者使用。
- **标识与缓冲区。** 对跨流引用绑定请求代次、层、逻辑 ID、设备 slot 与映射版本。Graph 固定容量和地址，动态写入有效长度和 slot；结果输出、任务表、部分状态以及事件依赖必须跨 replay 保持正确。请求退出、槽位复用、batch padding 或跨步回放都要纳入正确性测试。slot 读占用只需持有到实际 KV 读完，后续合并读取单独保存的状态。
- **先验证驻留，再加少量 staging。** 第二阶段才从 `P \ R` 中筛少量高置信项，搬到有独立上限的暂存区。正式选择后仅接纳命中的暂存项，其余回收，保持按需路径的 hot LRU。未完成的可复用项可以等待或走按需搬运，需记录重复搬运。先在非共享组验证。若之后用于 anchor，anchor 因暂存而省掉的复制仍可能是 skip 层的真实缺页，需保留后者的精确复制计划及共同 slot 布局。两个对照的 GPU 预算都包含暂存区，先验证移动额外字节是否真的减少关键路径上的等待。
- **按物理复制单元计账。** 命中与修复算术可以按查询头统计，H2D 则按实际物理页或 token slot 及复制载荷计数。GQA 的多个查询头共享同一 KV 页时，一次共同拷贝记一次。不同层的 KV 各自计数，MLA 使用实际压缩布局的 `item_size_bytes`。日志分别保留按需 H2D、预测 H2D、原生 shared-index H2D、重复复制与 D2D 提交。

## 公平对照与验证顺序

| 对照 | 测量目的 |
| --- | --- |
| 官方按需 HiSparse，关闭 shared-index prefetch | 定位原生选择、swap-in 与设备注意力的阶段成本。 |
| 官方 HiSparse，保留该模型原生 shared-index prefetch | 作为具备 IndexShare 的部署基线，保留精确跨层搬运。 |
| 同样的官方配置 + 驻留推测计算 | 测驻留计算的重叠与修复净收益。先只改有独立选择的层或 anchor，skip 层保留官方执行。 |
| 同样的官方配置 + 有界 staging，分别开启和关闭驻留推测计算 | 区分额外 H2D、缓存影响与推测算术收益。 |

配对使用相同模型、选集、批量、硬件、精度、物理缓存字节预算、原生 attention 后端与 warmup。每种路径从同样的缓存状态开始并独立演化。做原生 Graph 的动态 ID、实际搬运与批量变化检查。先让关闭预测的接入版本和官方基线产生一致的 slot、输出和搬运计数，再按 `A` 的相同量化 KV 比较数值。覆盖无预测、全命中、全错、需要覆盖正在读的 slot、容量 K、请求退出及共享组的 anchor/skip 等场景。若后端为逐块状态需要改写接口，另报同一新后端下的串行对照，避免把接口更换的收益归给重叠。

计时同时保留两类样本。固定输入的 Graph 配对用于测选择、占用等待、真实 H2D、补算与回退。原生运行测不同上下文和并发下的每 token 延迟与吞吐，记录实际 H2D 字节、在关键路径上等待拷贝的时间、GPU 峰值内存和 native baseline 的数值偏差。时间线应标明实际 copy kernel、选择与注意力的重叠，独立阶段耗时仅作诊断。现有六条原型 profile 是带 profiler 的单次固定输入回放，不能估计 HiSparse 的搬运和并发。[原型时间线及配置](../results/profiling/summary.json)
