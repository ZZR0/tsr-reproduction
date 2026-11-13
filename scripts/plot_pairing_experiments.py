#!/usr/bin/env python3
"""Plot archived scalar results; no model loading, inference or API calls."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('archive',type=Path);a=ap.parse_args()
    root=a.archive;out=root/'figures';out.mkdir(exist_ok=True)
    rows=json.loads((root/'metrics/results.json').read_text())['evaluations']
    tags=['longer-gap-1112t','distinct-only-1112t','cap2500-judge1','cap2500-judge0']
    labels=['Longer + min gap\nJudge T=1','No length cap\nJudge T=1','Chosen <=2500 chars\nJudge T=1','Chosen <=2500 chars\nJudge T=0']
    selected=[next(r for r in rows if tag in r['run']) for tag in tags]
    colors=['#cc6677','#4477aa','#66aabb','#228833']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,(key,title,unit) in zip(axes.flat,[('lc_win','LC Win','%'),('win','Win','%'),('mean_characters','Mean response length','characters'),('truncation_percent','Outputs reaching 1024-token limit','%')]):
        values=[r[key] for r in selected]
        bars=ax.bar(range(4),values,color=colors,width=.65)
        ax.set_xticks(range(4),labels,fontsize=8)
        ax.set_title(title);ax.set_ylabel(unit);ax.set_ylim(0,max(values)*1.2)
        ax.bar_label(bars,labels=[f'{v:.0f}' if unit=='characters' else f'{v:.2f}' for v in values],padding=4,fontsize=9)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('TSR pairing ablations: final M2 on 805 AlpacaEval questions\nFresh SFT, training seed 42; answer generation T=0; one training seed per setting',fontsize=12)
    fig.savefig(out/'latest-four-comparison.png',dpi=180);fig.savefig(out/'latest-four-comparison.svg');plt.close(fig)
    cap=[r for r in selected if 'cap2500' in r['run']]
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    stages=['M0','Mf0','M1','Mf1','M2_future_tie_charcap_distinct']
    titles=['SFT M0','Future Mf0','TSR M1','Future Mf1','Final M2']
    for ax,stage,title in zip(axes.flat,stages,titles):
        for row,color in zip(cap,['#4477aa','#228833']):
            path=root/'runs'/row['run']/'training'/stage/'trainer_state.json'
            state=json.loads(path.read_text())
            values=[(v['step'],v['loss']) for v in state['log_history'] if 'step' in v and 'loss' in v]
            ax.plot([v[0] for v in values],[v[1] for v in values],color=color,linewidth=1.2,label=f"self-judge T={row['self_judge_temperature']:g}")
        ax.set_title(title);ax.set_xlabel('Optimizer step');ax.set_ylabel('Logged training loss');ax.grid(alpha=.2)
    axes.flat[-1].axis('off');axes.flat[0].legend(fontsize=8)
    fig.suptitle('Chosen <=2500 characters: native training loss\nDatasets differ between runs; these losses are not evaluation quality scores.',fontsize=12)
    fig.savefig(out/'cap2500-training-loss.png',dpi=180);fig.savefig(out/'cap2500-training-loss.svg');plt.close(fig)
    print('Saved two plots in PNG and SVG:',out)

if __name__=='__main__':main()
