"""Regenerate the profiling report and timeline from recorded GPU events."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

from benchmarks.summarize import table

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def kernel_events(path):
    return sorted((e for e in read(path)['traceEvents'] if e.get('cat')=='kernel' and e.get('ph')=='X'),key=lambda e:e['ts'])


def check_sources(data):
    for f,digest in data['source_sha256'].items():
        assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==digest,f


def redact_dot_handles():
    # Kernel/edge structure is retained; process addresses have no analytical
    # value in a public dependency diagram. Timing traces are not modified.
    for path in (ROOT/'results/graph_diagnostic').glob('*.dot'):
        before=path.read_text();aliases={}
        def replace(match):
            value=match.group()
            if int(value,16)==0:return value
            if value not in aliases:aliases[value]=f'handle_{len(aliases)+1:03d}'
            return aliases[value]
        after=re.sub(r'0x[0-9A-Fa-f]+',replace,before)
        assert [s for s in before.splitlines() if ' -> ' in s]==[s for s in after.splitlines() if ' -> ' in s]
        if after!=before:path.write_text(after,encoding='utf-8')


def plot(profiles):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    colors={'Selector':'#168b83','Attention / prediction':'#3978b8','Set / page preparation':'#e7a43b',
            'Repair attention':'#8866b3','Signed merge':'#cf6471','Fallback':'#657786'}
    names=['serial','overlap','serialized_speculation','repair_only']
    fig,ax=plt.subplots(figsize=(11,4.5))
    for row,name in enumerate(names):
        events=kernel_events(Path('results/profiling')/profiles[name]['trace_file'])
        start=events[0]['ts'];repair=False
        for event in events:
            n=event['name']
            if n in ('_classify','_compact','_tables'):repair=True;kind='Set / page preparation'
            elif n.startswith('_fallback'):kind='Fallback'
            elif n=='_merge':kind='Signed merge'
            elif n=='_indexer_scores' or n.startswith('void at::'):kind='Selector'
            else:kind='Repair attention' if repair else 'Attention / prediction'
            band=(row-.31,.62)
            if name=='overlap' and kind=='Selector':band=(row-.31,.29)
            elif name=='overlap' and kind=='Attention / prediction':band=(row+.02,.29)
            ax.broken_barh([(event['ts']-start,event['dur'])],band,facecolors=colors[kind])
    ax.set_yticks(range(len(names)),names);ax.invert_yaxis()
    ax.set_xlabel('Microseconds from first GPU kernel (each row starts at zero)')
    ax.set_title('A100: one instrumented CUDA Graph replay per row')
    ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=c,label=n) for n,c in colors.items()],loc='upper center',
              bbox_to_anchor=(.5,-.2),ncol=3,frameon=False,fontsize=9)
    fig.tight_layout();fig.savefig(ROOT/'figures/profiling_timeline.png',dpi=180,bbox_inches='tight');plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--no-plot',action='store_true');args=ap.parse_args()
    redact_dot_handles()
    data=read('results/profiling/summary.json');check_sources(data)
    status=read('results/profiling/run_status.json');assert status['status']=='complete'
    p=data['profiles'];rows=[]
    for name,s in p.items():
        events=kernel_events(Path('results/profiling')/s['trace_file'])
        assert len(events)==s['cuda_kernel_count'] and s['validation']['passed']
        assert abs(sum(e['dur'] for e in events)-s['sum_kernel_us'])<.002
        rows.append([name,len(events),f"{s['sum_kernel_us']:.3f}",f"{s['first_to_last_kernel_span_us']:.3f}",
                     f"{s['no_kernel_interval_us']:.3f}",f"{s['concurrent_kernel_interval_us']:.3f}",s.get('fallback_heads','—')])
    pairs=data['unprofiled_full_path_pairs']
    b=statistics.median(x['baseline_ms'] for x in pairs)*1000
    c=statistics.median(x['candidate_ms'] for x in pairs)*1000
    ratio=statistics.median(x['ratio'] for x in pairs)
    repairs=p['repair_only']['kernels']
    groups=[('分类、压缩、页表',sum(repairs[n]['total_us'] for n in ['_classify','_compact','_tables'])),
            ('FlashInfer attention 与库内归并',sum(v['total_us'] for n,v in repairs.items() if n.startswith('void flashinfer::'))),
            ('带符号合并与误差守卫',repairs['_merge']['total_us']),
            ('保护性重算与归并',sum(v['total_us'] for n,v in repairs.items() if n.startswith('_fallback')))]
    lines=['# A100 容器内 profiling','',
        f"2026-11-09，在固定训练容器的 A100 SXM4 80GB GPU {status['gpu_index']} 上采集。PyTorch 2.8.0+cu128、Triton 3.4.0、FlashInfer 0.3.1。使用 PyTorch profiler / CUPTI 的真实 CUDA kernel 事件；环境没有可用的 Nsight Systems / Compute。本报告没有 SM 利用率、occupancy、HBM 带宽或 PCIe 硬件计数器。",'',
        '输入为随机 QSA 形状：8K 上下文，24 Q 头 / 2 KV 头，D=256，4-token 块，K=512。预测 409 块，其中 327 块命中；选择器是独立 QSA 公式加 torch.topk，没有加载原生 QSA 模型。没有 host KV 搬运。','',
        '## 无 profiler 的配对计时','',
        f'先运行 5 对 ABBA、每段 100 次 CUDA Graph 回放。直接串行路径的配对时长中位数为 **{b:.3f} µs**，严格推测路径为 **{c:.3f} µs**，配对比值中位数为 **{ratio:.4f}×**。此点尚未获得净加速。以下单次插桩时间线用于定位组成，不能代替这组正常计时。','',
        '## 单次 Graph 时间线','',
        *table(['路径','kernel 数','kernel 耗时和 µs','首尾跨度 µs','无 kernel 区间 µs','并发区间 µs','回退头'],rows),'',
        '![CUDA Graph kernel timeline](../figures/profiling_timeline.png)','',
        '图中每行的零点是该行第一个 GPU kernel。颜色按功能分类，不表示 CUDA stream；overlap 行的选择与预测分别画在上下半行以显示交叠。空白只表示没有记录到 GPU kernel，可能包含依赖等待或其他活动。`serialized_speculation` 和 `overlap` 执行相同计算量。','',
        f"本次 `overlap` trace 根据 GPU 时间戳实际测得 {p['overlap']['concurrent_kernel_interval_us']:.3f} µs 的并发区间。Graph 节点跟踪本身会扰动时间线，需要结合下方无 profiler 配对和依赖图诊断；trace lane 的数量本身不是并发证据，也不能据此推断 SM 占用或资源饱和。[NVIDIA 对 Graph 节点跟踪开销的说明](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/performance-issues.html)",'',
        '## 修复尾部与优化方向','',
        *table(['独立 repair_only 的组成','kernel 耗时和 µs'],[[n,f'{t:.3f}'] for n,t in groups]),'',
        f"此点 {p['repair_only'].get('fallback_heads',0)}/24 个查询头触发数值保护回退。回退占修复 kernel 耗时和的 {groups[-1][1]/p['repair_only']['sum_kernel_us']:.1%}。其余准备、库调用和合并仍然显著，所以只缩短擦除或放宽保护条件不足以解释、解决全部差距。",'',
        '优先级：先确认图调度能覆盖预测，再减少分类/压缩/表构建的启动与中间缓冲；让库 attention 直接产生所需的 FP32 部分状态，减少额外归并及低精度相减引起的回退。后者需要修改、验证内核接口，当前没有实现。保留逐块状态后只归并命中块是另一条无减法路线，应同时计入状态写入成本。单独 selector 中 torch.topk 占主要时间，也值得与原生 QSA selector 对照，不能通过只人为拖慢基线来制造收益。','',
        '## 图调度补查','']
    diagnostic=ROOT/'results/graph_diagnostic/summary.json'
    if diagnostic.exists():
        d=read('results/graph_diagnostic/summary.json');check_sources(d)
        ds=read('results/graph_diagnostic/run_status.json');assert ds['status']=='complete'
        assert all(v['passed'] for v in d['validation'].values())
        dr=[]
        for name,rr in d['pairs'].items():
            dr.append([name,f"{statistics.median(r['baseline_us'] for r in rr):.3f}",
                       f"{statistics.median(r['candidate_us'] for r in rr):.3f}",
                       f"{min(r['ratio'] for r in rr):.4f}–{max(r['ratio'] for r in rr):.4f}"])
        lines += [f"补查在同容器 GPU {ds['gpu_index']} 上进行，导出 Graph DOT 依赖图并对调选择/预测的提交先后。使用同一输入、内核、保护条件，四种输出均通过 FP64 校验。各对照单独进行五对 ABBA/BAAB，因此不同表行不能拼接计算比值。",'',
            *table(['基线 / 候选','基线中位 µs','候选中位 µs','逐对比值范围'],dr),'',
            '`selection_first` 依赖图中选择链 0→1→2→3 与预测链 4→5→6 独立，两者在修复节点 7 汇合，没有选择结束才能开始预测的依赖边。同工作量的无 profiler 配对中，重叠图比顺序图快，与 trace 中观察到的并发一致；不能把全部差异单独归因于并发或未采样的 GPU 时钟。完整路径仍慢于直接串行。调换提交顺序也不是普遍有效的优化，以下原始证据保留所有方向与失败结果。公开 DOT 的非零进程地址已替换为文件内一致的 handle 占位符，节点与边保持不变。','',
            '[诊断数据](../results/graph_diagnostic/summary.json) · [选择先提交的依赖图](../results/graph_diagnostic/selection_first.dot) · [预测先提交的依赖图](../results/graph_diagnostic/prediction_first.dot)','']
    lines += ['## 采样与复现边界','',
        '容器内进程列表可能隐藏其他容器的 PID。最终重测另用宿主机查询该 GPU 的进程数，并核对进程属于训练容器；每轮间隔 1 秒，另加查询耗时。[主测审计](../results/profiling/host_gpu_audit.json)和[图诊断审计](../results/graph_diagnostic/host_gpu_audit.json)仅公开计数，不包含用户、PID、主机地址或 UUID。采样未发现共享不等于能够排除采样间隔内的短暂活动。','',
        '主测宿主审计在任务选定 GPU 后约 4.1 秒才开始，首个样本已有一个本容器计算进程，因此没有覆盖基线启动初段；不能称为全程隔离证明。所有公开 profiler trace 的时间戳均在对应宿主审计范围内，图诊断审计从启动前开始。','',
        '每个路径只采一条单次重放 trace。CPU GraphLaunch API 时长受首次追踪和 profiler 影响，不能当稳态提交开销；摘要中的 DeviceSynchronize 汇总还包含 profiler 退出时的同步。没有据此宣称 CPU、显存或 PCIe 是瓶颈。','',
        '[原始摘要](../results/profiling/summary.json) · [串行 trace](../results/profiling/serial.trace.json) · [推测 trace](../results/profiling/overlap.trace.json) · [顺序推测 trace](../results/profiling/serialized_speculation.trace.json) · [修复 trace](../results/profiling/repair_only.trace.json) · [容器执行记录](../results/profiling/run_status.json)','',
        'GPU 采集：`python -m experiments.run_container_profile`；调度诊断：追加 `--stage diagnostic`。宿主机另运行 `python -m experiments.host_gpu_audit --gpu <index> --container <name> --output <file>`。报告和图：`python -m benchmarks.summarize_profiling`。','']
    (ROOT/'docs/profiling.md').write_text('\n'.join(lines),encoding='utf-8')
    if not args.no_plot:plot(p)
    print(json.dumps({'profiles':len(p),'serial_us':b,'overlap_us':c,'ratio':ratio}))


if __name__=='__main__':main()
