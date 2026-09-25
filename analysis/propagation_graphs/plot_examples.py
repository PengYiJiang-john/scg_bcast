"""Plot four explicitly selected diagnostic cases from saved graph records."""
from pathlib import Path
import argparse
import os
import json
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from analyze import ACTIVE, CORRECTION, HERE, graph_metrics

SELECTED = [
    ('h22','h22_finance_cashflow__source_absent','ctrl_dense_6',
     '01_regex_empty_high_adoption','Chinese adoption can disappear from the regex proxy',
     'All six saved step statuses are absent. The final Pro score is 1.0; the text repeats positive cash flow. Empty proxy does not establish absent propagation.'),
    ('h22','h22_product_recall__source_absent','ctrl_tree_6',
     '02_nonchain_merge','An observed active-step proxy with two paths merging',
     'All six steps are regex-active. Two source paths merge at ctt5: maximum indegree 2, longest path 4/6. This proxy is not a chain.'),
    ('h22','h22_finance_cashflow__source_deemphasized','ctrl_tree_6',
     '03_chain_fragment','A three-node chain fragment does not validate the full semantic graph',
     'The proxy contains ctt2 -> ctt4 -> ctt5. The legacy edge into ctt6 (labeled corrected) is removed. Pro still scores the final output 1.0.'),
    ('scifact_legacy','scifact_105','debate_factcheck',
     '04_path_coverage_is_not_chain','A full-length directed path can coexist with shortcuts',
     'Saved labels yield a triangle: path coverage 3/3, but not a chain. Text at df4 actually disputes the claim despite its active label; this is not a validated semantic graph.'),
]
COLORS={'active':'#C85C36','correction':'#317B76','other':'#C3CDD5'}


def layout(nodes,edges):
    parents={n:[] for n in nodes}
    for u,v in edges:parents[v].append(u)
    depth={}
    todo=list(nodes)
    while todo:
        ready=[n for n in todo if all(p in depth for p in parents[n])]
        if not ready:raise ValueError('Cyclic graph')
        for n in ready:
            depth[n]=max((depth[p]+1 for p in parents[n]),default=0)
            todo.remove(n)
    maxdepth=max(depth.values(),default=0)
    levels={d:[n for n in nodes if depth[n]==d] for d in set(depth.values())}
    positions={}
    for d,group in levels.items():
        for j,n in enumerate(group):
            positions[n]=(d/max(maxdepth,1), .5+(len(group)-1)/2*.32-j*.32)
    return positions


def fmt(value):
    return 'undefined' if value is None else f'{value:.2f}'


def draw(ax,nodes,edges,positions,statuses,prefix_scores,title,metrics,active_edges,full):
    ax.set_xlim(-.14,1.14);ax.set_ylim(-.07,1.05);ax.axis('off')
    ax.set_title(title,loc='left',fontsize=11,fontweight='bold',pad=13)
    for u,v in edges:
        color=COLORS['active'] if (u,v) in active_edges else '#9AA6B1'
        samelevel=abs(positions[u][1]-positions[v][1])<1e-6
        # Arcs distinguish skip edges from the path at the same y-coordinate.
        curve=-.23 if samelevel and positions[v][0]-positions[u][0]>.48 else 0
        arrow=FancyArrowPatch(positions[u],positions[v],arrowstyle='-|>',mutation_scale=10,
                              color=color,linewidth=1.8 if (u,v) in active_edges else 1.25,
                              shrinkA=22,shrinkB=22,connectionstyle=f'arc3,rad={curve}',zorder=1)
        ax.add_patch(arrow)
    for n in nodes:
        x,y=positions[n]
        kind='active' if statuses[n] in ACTIVE else 'correction' if statuses[n] in CORRECTION else 'other'
        patch=FancyBboxPatch((x-.058,y-.055),.116,.11,boxstyle='round,pad=.006,rounding_size=.017',
                            facecolor=COLORS[kind],edgecolor='white',lw=1.2,zorder=2)
        ax.add_patch(patch)
        ax.text(x,y,n,ha='center',va='center',fontsize=8.2,fontweight='bold',
                color='white' if kind!='other' else '#263F51',zorder=3)
        if n in prefix_scores:
            ax.text(x,y-.092,f'I(prefix)={prefix_scores[n]:.2f}',ha='center',va='top',fontsize=7.0,color='#334657')
        if full and kind!='active':
            status=statuses[n].replace('_false','').replace('_correction',' corr.')
            ax.text(x,y+.088,status,ha='center',va='bottom',fontsize=6.7,color='#526374')
    if not nodes:
        ax.text(.5,.5,'EMPTY REGEX PROXY\nNo chain evidence',ha='center',va='center',
                fontsize=15,fontweight='bold',color='#9A472C')
    stats=(f"Nodes {metrics['node_count']} | Edges {metrics['edge_count']} | Max in/out {metrics['max_indegree']}/{metrics['max_outdegree']}\n"
           f"Branch/merge nodes {metrics['branch_nodes']}/{metrics['merge_nodes']} | Components {metrics['weak_components']}\n"
           f"Longest path {metrics['longest_path_nodes']}/{metrics['node_count']} | Coverage {fmt(metrics['longest_path_coverage'])} | Chain: {metrics['nontrivial_chain']}")
    ax.text(0,-.055,stats,transform=ax.transAxes,fontsize=8.2,va='top',color='#263F51',linespacing=1.55)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=HERE/'results')
    parser.add_argument('--output-dir',type=Path,default=HERE/'figures')
    args=parser.parse_args()
    graphs=json.loads((args.input_dir/'graph_records.json').read_text())
    indexed={(g['record']['dataset'],g['record']['case_id'],g['record']['workflow_id']):g for g in graphs}
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    selected=[]
    for dataset,case,wf,stem,title,note in SELECTED:
        g=indexed[(dataset,case,wf)];r=g['record']
        edges=list(map(tuple,g['edges']));activeedges=list(map(tuple,g['active_edges']))
        positions=layout(g['nodes'],edges)
        fig,axes=plt.subplots(1,2,figsize=(14,6.3))
        fig.subplots_adjust(left=.035,right=.98,top=.79,bottom=.29,wspace=.12)
        fig.suptitle(title,x=.036,y=.968,ha='left',fontsize=16,fontweight='bold',color='#1C3545')
        score='not available' if r['final_pro_intensity'] is None else f"{r['final_pro_intensity']:.2f}"
        fig.text(.036,.895,f'{dataset} | {case} | {wf} | Final Pro intensity: {score}',fontsize=9.3,color='#526374')
        draw(axes[0],g['nodes'],edges,positions,g['statuses'],g['prefix_intensities'],
             'Prescribed workflow dependencies',graph_metrics(g['nodes'],edges),set(activeedges),True)
        draw(axes[1],g['active_nodes'],activeedges,positions,g['statuses'],g['prefix_intensities'],
             'Legacy active-step induced proxy',graph_metrics(g['active_nodes'],activeedges),set(activeedges),False)
        fig.legend(handles=[Patch(facecolor=COLORS['active'],label='Regex active'),
                            Patch(facecolor=COLORS['correction'],label='Regex correction/challenge'),
                            Patch(facecolor=COLORS['other'],label='Other regex status')],
                   loc='lower left',bbox_to_anchor=(.028,.088),ncol=3,frameon=False,fontsize=9)
        fig.text(.036,.074,'\n'.join(textwrap.wrap(note,width=155)),fontsize=9.1,color='#263F51',va='top')
        fig.text(.036,.017,'Post hoc diagnostic selection. I(prefix) is a Flash system/prefix score, not a node-truth label. No recovered semantic-source graph or new LLM run is claimed.',
                 fontsize=7.8,color='#526374')
        fig.savefig(out/(stem+'.png'),dpi=180,facecolor='white')
        plt.close(fig)
        selected.append(dict(dataset=dataset,case_id=case,workflow_id=wf,figure=os.path.relpath(out/(stem+'.png'),args.input_dir),
                             selection_reason=note,metrics=r,raw_statuses=g['statuses'],
                             raw_texts=g['trace_texts'],active_edges=g['active_edges'],removed_legacy_edges=g['removed_edges']))
    (args.input_dir/'selected_examples.json').write_text(json.dumps(selected,indent=2,ensure_ascii=False)+'\n')
    print('Saved 4 diagnostic PNG figures and selected_examples.json in the requested output/input directories.')


if __name__=='__main__':
    main()
