"""Plot observed predictor retention; no task-quality or latency inference."""
import json
from pathlib import Path
from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).resolve().parents[1]
    data=json.loads((root/'results/window/predictor_rank_trace.json').read_text())
    full=[h for e in data['experiments'] if e['budget']==64 for h in e['heads']]
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    colors=['#2463b1','#df7e33','#5d9973']
    for name,label,color in [('temporal','Previous step, same layer',colors[0]),('layer','Current step, prior dense layer',colors[1])]:
        values=[mean(h[name]['source_rank_quartile_retention'][i] for h in full) for i in range(4)]
        axes[0].plot(range(4),values,'o-',color=color,label=label)
        for i,value in enumerate(values):axes[0].annotate(f'{value:.0%}',(i,value),xytext=(0,6),textcoords='offset points',ha='center',fontsize=9)
    axes[0].set_xticks(range(4),['Top 25%','25–50%','50–75%','75–100%'])
    axes[0].set_title('Retention by source rank')
    axes[0].legend(fontsize=7,loc='lower left')
    budgets=[32,51,64]
    rows={p:[h for e in data['experiments'] if e['budget']==p for h in e['heads']] for p in budgets}
    for metric,label,style in [('precision','Prediction precision','o-'),('coverage','Target Top-K coverage','s--')]:
        values=[mean(h['temporal'][metric] for h in rows[p]) for p in budgets]
        axes[1].plot(budgets,values,style,label=label)
        for p,value in zip(budgets,values):axes[1].annotate(f'{value:.1%}',(p,value),xytext=(0,6),textcoords='offset points',ha='center',fontsize=9)
    axes[1].set_xticks(budgets,['32 (0.5K)','51 (~0.8K)','64 (K)'])
    axes[1].set_title('Temporal prediction budget')
    axes[1].legend(fontsize=8,loc='lower right')
    methods=['temporal','interleave','layer']
    values=[mean(h[name]['coverage'] for h in rows[51]) for name in methods]
    axes[2].bar(range(3),values,color=[colors[0],colors[2],colors[1]],width=.6)
    axes[2].set_xticks(range(3),['Previous step','Mixed ranks','Prior layer'])
    for i,value in enumerate(values):axes[2].text(i,value+.02,f'{value:.1%}',ha='center')
    axes[2].set_title('Equal budget: 51 blocks, target K=64')
    for ax in axes:
        ax.set_ylim(0,1.08);ax.set_yticks([0,.25,.5,.75,1.],['0%','25%','50%','75%','100%'])
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Dense Qwen3-4B + external Quest-style selector: proxy evidence, not native QSA',fontsize=12)
    dest=root/'figures';dest.mkdir(exist_ok=True)
    fig.savefig(dest/'predictor_retention.png',dpi=180)
    fig.savefig(dest/'predictor_retention.svg')


if __name__=='__main__':main()
