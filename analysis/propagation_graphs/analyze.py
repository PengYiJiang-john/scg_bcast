"""Post hoc workflow/legacy-regex graph audit. Reads saved records; no model calls.

Nodes are workflow steps. Full edges are prescribed dependencies. Active nodes
use archived regex event statuses, and active edges require BOTH endpoints
active. These are proxies, not recovered semantic-source SCG graphs.
"""
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from statistics import mean

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[1]
ACTIVE = frozenset({'introduced_false', 'preserved_false', 'transformed_false',
                    'amplified_false', 'laundered_false', 'finalized_false'})
CORRECTION = frozenset({'corrected_false', 'challenged_false', 'preventive_correction'})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def graph_metrics(nodes, edges):
    """Simple directed DAG measures. Empty/singleton graphs never count as chains."""
    nodes = set(nodes)
    edges = set(map(tuple, edges))
    require(all(u in nodes and v in nodes for u, v in edges), 'Edge endpoint outside node set')
    ins = {u: set() for u in nodes}
    outs = {u: set() for u in nodes}
    for u, v in edges:
        ins[v].add(u)
        outs[u].add(v)
    remaining = {u: len(ins[u]) for u in nodes}
    ready = sorted(u for u in nodes if remaining[u] == 0)
    order = []
    while ready:
        u = ready.pop(0)
        order.append(u)
        for v in sorted(outs[u]):
            remaining[v] -= 1
            if remaining[v] == 0:
                ready.append(v)
                ready.sort()
    require(len(order) == len(nodes), 'Graph is cyclic; longest DAG path is undefined')
    paths = {}
    for u in order:
        candidates = [paths[p]+[u] for p in sorted(ins[u])]
        paths[u] = max(candidates, key=lambda p: (len(p), tuple(p))) if candidates else [u]
    longest = max(paths.values(), key=lambda p: (len(p), tuple(p))) if paths else []
    components, unseen = 0, set(nodes)
    while unseen:
        components += 1
        todo = [next(iter(unseen))]
        while todo:
            u = todo.pop()
            if u not in unseen:
                continue
            unseen.remove(u)
            todo.extend((ins[u] | outs[u]) & unseen)
    n, e = len(nodes), len(edges)
    max_in = max((len(v) for v in ins.values()), default=0)
    max_out = max((len(v) for v in outs.values()), default=0)
    branches = sum(len(v)>1 for v in outs.values())
    merges = sum(len(v)>1 for v in ins.values())
    chain = n >= 2 and components == 1 and e == n-1 and max_in <= 1 and max_out <= 1
    if n == 0:
        kind = 'empty'
    elif n == 1:
        kind = 'singleton'
    elif chain:
        kind = 'nontrivial_chain'
    elif components > 1:
        kind = 'disconnected'
    else:
        kind = 'branching_or_merging'
    return dict(node_count=n, edge_count=e, max_indegree=max_in, max_outdegree=max_out,
                branch_nodes=branches, merge_nodes=merges, weak_components=components,
                longest_path_nodes=len(longest), longest_path_coverage=(len(longest)/n if n else None),
                nontrivial_chain=chain, graph_kind=kind)


def check_sources():
    manifest = json.loads((HERE/'data/source_manifest.json').read_text())
    for row in manifest['files']:
        path = HERE/row['analysis_path']
        require(hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'], 'Changed input: '+row['analysis_path'])
    return manifest


def judge_index(scores_path, raw_path):
    scores = json.loads(scores_path.read_text())['scores']
    raw = read_lines(raw_path)
    raw_by_id = {row['prefix_id']: row for row in raw}
    require(len(raw_by_id)==len(raw)==len(scores), 'Unequal or duplicate judge coverage')
    result = {}
    for row in scores:
        response = raw_by_id[row['prefix_id']]
        match = re.search(r'\{.*\}', response['raw_response'], re.S)
        try:
            if match is None:
                raise ValueError('No complete JSON object')
            parsed = json.loads(match.group())
            numeric = float(parsed['intensity'])
            parse_status = 'strict_json'
        except (ValueError, KeyError, TypeError):
            # Preserve and disclose truncated responses; never substitute zero.
            recovered = re.search(r'"intensity"\s*:\s*([0-9]+(?:\.[0-9]+)?)', response['raw_response'])
            require(recovered is not None, 'No recoverable numeric score in raw judge response')
            numeric = float(recovered.group(1))
            parse_status = 'partial_json_numeric_score_recovered'
        require(0 <= numeric <= 1, 'Out-of-range raw judge score')
        require(numeric==float(row['intensity'])==float(response['intensity']), 'Judge score mismatch')
        key = (row['case_id'],row['workflow_id'],row['claim_id'],row['step_id'])
        require(key not in result, 'Duplicate judge key')
        result[key] = dict(intensity=float(row['intensity']), latest_content=row['latest_content'],
                           prefix_id=row['prefix_id'], model=response['judge_model'], raw_parse_status=parse_status)
    return result


def build_records():
    check_sources()
    h22 = read_lines(EXPERIMENTS/'archive/source_access/raw/runs.jsonl')
    old = json.loads((HERE/'data/scifact_legacy_matrix.json').read_text())['runs']
    pro = judge_index(EXPERIMENTS/'archive/source_access/raw/intensity_scores.json',
                      EXPERIMENTS/'archive/source_access/raw/direct_judgments.jsonl')
    flash = judge_index(HERE/'data/h22_flash_step_scores.json', HERE/'data/h22_flash_step_judgments.jsonl')
    require(len(h22)==120 and len(old)==20 and len(pro)==120 and len(flash)==720, 'Unexpected dataset sizes')
    records, node_rows, graphs = [], [], []
    for dataset, runs in [('h22', h22), ('scifact_legacy', old)]:
        for run in runs:
            case, workflow, trace = run['case'], run['workflow'], run['trace']
            case_id, workflow_id = case['case_id'], workflow['workflow_id']
            nodes = [step['step_id'] for step in workflow['steps']]
            require(len(nodes)==len(set(nodes)), 'Duplicate workflow step')
            edges = {(parent, step['step_id']) for step in workflow['steps'] for parent in step['depends_on']}
            trace_edges = {(parent, step['step_id']) for step in trace['steps'] for parent in step['depends_on']}
            require(edges==trace_edges, 'Trace dependencies differ from saved workflow')
            require(set(nodes)=={step['step_id'] for step in trace['steps']}, 'Trace/workflow node mismatch')
            final_steps = [step for step in trace['steps'] if step['is_final']]
            require(len(final_steps)==1, 'Expected one final step')
            final = final_steps[0]
            for claim in case['false_claims']:
                claim_id = claim['claim_id']
                events = [event for event in run['events'] if event['claim_id']==claim_id]
                by_event = {event['step_id']: event for event in events}
                require(len(events)==len(nodes) and set(by_event)==set(nodes), 'Incomplete event classification')
                statuses = {step: event['status'] for step,event in by_event.items()}
                active = {step for step,status in statuses.items() if status in ACTIVE}
                active_edges = {(u,v) for u,v in edges if u in active and v in active}
                links = [link for link in run['links'] if link['claim_id']==claim_id]
                raw_edges = {(link['source_step_id'],link['target_step_id']) for link in links}
                filtered = {(u,v) for u,v in raw_edges if u in active and v in active}
                require(raw_edges <= edges, 'Legacy stored link not in workflow DAG')
                require(filtered == active_edges, 'Filtered stored links disagree with induced active graph')
                if dataset=='h22':
                    base,condition = case_id.removeprefix('h22_').split('__')
                else:
                    base,condition = case_id,'not_applicable'
                record = dict(dataset=dataset,case_id=case_id,base_case=base,condition=condition,
                              workflow_id=workflow_id,topology=workflow['topology'],claim_id=claim_id)
                record.update({'full_'+key:value for key,value in graph_metrics(nodes,edges).items()})
                record.update({'active_'+key:value for key,value in graph_metrics(active,active_edges).items()})
                key=(case_id,workflow_id,claim_id,final['step_id'])
                final_pro = pro.get(key)
                final_flash = flash.get(key)
                if dataset=='h22':
                    require(final_pro and final_flash, 'Missing terminal judge')
                    require(final_pro['latest_content']==final['content'], 'Pro final text mismatch')
                record.update(legacy_link_count=len(raw_edges),
                              legacy_links_removed_nonactive_endpoint=len(raw_edges-filtered),
                              legacy_links_to_correction_or_challenge=sum(v not in active and statuses[v] in CORRECTION for u,v in raw_edges),
                              legacy_links_to_suppressed_or_other=sum(v not in active and statuses[v] not in CORRECTION for u,v in raw_edges),
                              filtered_stored_links_equal_active_induced=True,
                              final_regex_status=statuses[final['step_id']], final_regex_active=final['step_id'] in active,
                              final_pro_intensity=final_pro['intensity'] if final_pro else None,
                              final_flash_prefix_intensity=final_flash['intensity'] if final_flash else None,
                              cjk_step_count=sum(bool(re.search(r'[\u4e00-\u9fff]',step['content'])) for step in trace['steps']))
                for step in trace['steps']:
                    stepkey=(case_id,workflow_id,claim_id,step['step_id'])
                    flash_row=flash.get(stepkey)
                    if dataset=='h22':
                        require(flash_row is not None and flash_row['latest_content']==step['content'], 'Flash step text mismatch')
                    node_rows.append(dict(dataset=dataset,case_id=case_id,condition=condition,
                                          workflow_id=workflow_id,topology=workflow['topology'],claim_id=claim_id,
                                          step_id=step['step_id'],role=step['role'],is_final=step['is_final'],
                                          regex_status=statuses[step['step_id']],regex_active=step['step_id'] in active,
                                          flash_prefix_intensity=flash_row['intensity'] if flash_row else None,
                                          flash_raw_parse_status=flash_row['raw_parse_status'] if flash_row else None,
                                          flash_judge_model=flash_row['model'] if flash_row else None,
                                          has_cjk=bool(re.search(r'[\u4e00-\u9fff]',step['content'])),
                                          matched_false_patterns=json.dumps(by_event[step['step_id']]['matched_false_patterns'],ensure_ascii=False)))
                records.append(record)
                graphs.append(dict(record=record,nodes=nodes,edges=sorted(edges),active_nodes=sorted(active),
                                   active_edges=sorted(active_edges),statuses=statuses,legacy_edges=sorted(raw_edges),
                                   removed_edges=sorted(raw_edges-filtered),
                                   prefix_intensities={step['step_id']:flash[(case_id,workflow_id,claim_id,step['step_id'])]['intensity']
                                                     for step in trace['steps']} if dataset=='h22' else {},
                                   trace_texts={step['step_id']:step['content'] for step in trace['steps']},
                                   target_false_claim=claim['label']))
    require(len(records)==140 and len(node_rows)==810, 'Unexpected final graph/node count')
    return records,node_rows,graphs


def summarize_group(rows):
    result={'run_claim_count':len(rows)}
    for kind in ('full','active'):
        eligible=[r for r in rows if r[kind+'_node_count']>=2]
        counts=Counter(r[kind+'_graph_kind'] for r in rows)
        chains=sum(r[kind+'_nontrivial_chain'] for r in rows)
        summary=dict(graph_kind_counts=dict(counts),nontrivial_node_denominator=len(eligible),
                     nontrivial_chain_count=chains,
                     chain_fraction_among_graphs_with_at_least_two_nodes=chains/len(eligible) if eligible else None,
                     nonempty_count=sum(r[kind+'_node_count']>0 for r in rows),
                     at_least_one_edge_count=sum(r[kind+'_edge_count']>0 for r in rows),
                     longest_path_coverage_nontrivial_mean=mean(r[kind+'_longest_path_coverage'] for r in eligible) if eligible else None,
                     branch_graph_count=sum(r[kind+'_branch_nodes']>0 for r in rows),
                     merge_graph_count=sum(r[kind+'_merge_nodes']>0 for r in rows))
        for metric in ('node_count','edge_count','max_indegree','max_outdegree','branch_nodes','merge_nodes',
                       'weak_components','longest_path_nodes','longest_path_coverage'):
            values=[r[kind+'_'+metric] for r in rows if r[kind+'_'+metric] is not None]
            summary[metric]={'n_defined':len(values),'mean':mean(values) if values else None,
                             'min':min(values) if values else None,'max':max(values) if values else None}
        result[kind]=summary
    judged=[r for r in rows if r['final_pro_intensity'] is not None]
    high=[r for r in judged if r['final_pro_intensity']>=.8]
    result['pro_endpoint_cross_check']={
        'records_with_pro_score':len(judged),'exploratory_high_threshold':.8,
        'high_score_count':len(high),
        'high_score_with_empty_regex_graph':sum(r['active_node_count']==0 for r in high),
        'high_score_with_singleton_regex_graph':sum(r['active_node_count']==1 for r in high),
        'high_score_with_nonactive_final_regex_status':sum(not r['final_regex_active'] for r in high),
        'exact_one_score_with_empty_regex_graph':sum(r['final_pro_intensity']==1 and r['active_node_count']==0 for r in judged),
        'interpretation':'Measurement discordance only, not a false-negative rate; prefix/global intensity is not a node label.'}
    result['legacy_edge_audit']={'stored_edges':sum(r['legacy_link_count'] for r in rows),
                                 'removed_nonactive_endpoint_edges':sum(r['legacy_links_removed_nonactive_endpoint'] for r in rows),
                                 'removed_to_correction_or_challenge':sum(r['legacy_links_to_correction_or_challenge'] for r in rows),
                                 'removed_to_suppressed_or_other':sum(r['legacy_links_to_suppressed_or_other'] for r in rows)}
    return result


def grouped_summary(records):
    groupings=[('dataset',),('dataset','workflow_id'),('dataset','topology'),('dataset','condition'),
               ('dataset','condition','workflow_id')]
    output={}
    for fields in groupings:
        groups=defaultdict(list)
        for row in records:
            groups[tuple(row[field] for field in fields)].append(row)
        output['by_'+'_and_'.join(fields)]=[
            dict(group=dict(zip(fields,key)),**summarize_group(rows)) for key,rows in sorted(groups.items())]
    return output


def write_csv(path,rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def self_test():
    checks = 0
    def check(value, message):
        nonlocal checks
        require(value, message)
        checks += 1
    empty = graph_metrics([], [])
    check(empty['graph_kind']=='empty' and empty['weak_components']==0 and
          empty['longest_path_coverage'] is None and not empty['nontrivial_chain'], 'Empty graph policy')
    single = graph_metrics(['a'], [])
    check(single['graph_kind']=='singleton' and not single['nontrivial_chain'], 'Singleton policy')
    chain = graph_metrics('abcd', [('a','b'),('b','c'),('c','d')])
    check(chain['nontrivial_chain'] and chain['node_count']==4 and chain['edge_count']==3
          and chain['longest_path_coverage']==1, 'Chain counts')
    fork = graph_metrics('abc', [('a','b'),('a','c')])
    check(not fork['nontrivial_chain'] and fork['branch_nodes']==1 and fork['max_outdegree']==2, 'Fork')
    diamond = graph_metrics('abcd', [('a','b'),('a','c'),('b','d'),('c','d')])
    check(diamond['longest_path_coverage']==3/4 and diamond['merge_nodes']==1
          and diamond['branch_nodes']==1 and not diamond['nontrivial_chain'], 'Diamond')
    tree = graph_metrics('abcdef', [('a','c'),('b','d'),('c','e'),('d','e'),('e','f')])
    check(tree['node_count']==6 and tree['edge_count']==5 and tree['longest_path_coverage']==4/6
          and tree['max_indegree']==2 and not tree['nontrivial_chain'], 'Six-step two-source tree')
    disconnected = graph_metrics('abcd', [('a','b'),('c','d')])
    check(disconnected['weak_components']==2 and not disconnected['nontrivial_chain'], 'Disconnected paths')
    shortcut = graph_metrics('abc', [('a','b'),('b','c'),('a','c')])
    check(shortcut['longest_path_coverage']==1 and not shortcut['nontrivial_chain'], 'Path cover is not chain test')
    active = {'a','b'}
    retained = {(u,v) for u,v in [('a','b'),('b','correction')] if u in active and v in active}
    check(retained=={('a','b')}, 'Remove edge with corrected target')
    try:
        graph_metrics('ab', [('a','b'),('b','a')])
    except ValueError:
        check(True, 'Cycle rejected')
    else:
        check(False, 'Cycle was not rejected')
    print(f'{checks} independent graph checks passed.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=HERE/'results')
    parser.add_argument('--self-test',action='store_true',help='Run independent graph metric fixtures only')
    args=parser.parse_args()
    if args.self_test:
        self_test()
        return
    records,nodes,graphs=build_records()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_csv(args.output_dir/'per_run_graph_metrics.csv',records)
    write_csv(args.output_dir/'per_step_annotations.csv',nodes)
    summary=dict(analysis_kind='posthoc_exploratory_saved_workflow_and_legacy_regex_graph_analysis',
                 unit='one saved run and declared false-claim target; graph nodes are workflow steps',
                 active_statuses=sorted(ACTIVE),new_model_calls=0,
                 flash_step_judge_raw_parse_counts=dict(Counter(row['flash_raw_parse_status'] for row in nodes if row['flash_raw_parse_status'])),
                 flash_step_judge_model_counts=dict(Counter(row['flash_judge_model'] for row in nodes if row['flash_judge_model'])),
                 chain_definition='Connected DAG with >=2 nodes, n-1 edges, max indegree/outdegree <=1.',
                 path_coverage_definition='Longest directed DAG path node count divided by graph node count; undefined for empty.',
                 caveats=['Prescribed workflow edges are not recovered semantic-source dependencies.',
                          'Saved regex statuses can miss Chinese/paraphrases and misread correction context.',
                          'Active graph is an induced workflow proxy; both endpoints must be active.',
                          'Flash I_t measures the current prefix/system, not a local node false-claim label.',
                          'Pro endpoint score is a different measurement, not independent ground-truth graph labeling.',
                          'Empty/singleton graphs provide no nontrivial-chain evidence; long-path coverage alone does not rule out branches.',
                          'These are fixed archived cases, not new independent LLM runs or generalization evidence.'],
                 **grouped_summary(records))
    compact = []
    for group in summary['by_dataset_and_workflow_id']:
        active, full = group['active'], group['full']
        kinds = active['graph_kind_counts']
        compact.append(dict(**group['group'], runs=group['run_claim_count'],
                            full_nodes=full['node_count']['mean'], full_edges=full['edge_count']['mean'],
                            full_max_in=full['max_indegree']['max'], full_max_out=full['max_outdegree']['max'],
                            full_path_coverage=full['longest_path_coverage']['mean'],
                            active_empty=kinds.get('empty', 0), active_singleton=kinds.get('singleton', 0),
                            active_chain=kinds.get('nontrivial_chain', 0), active_disconnected=kinds.get('disconnected', 0),
                            active_branching_or_merging=kinds.get('branching_or_merging', 0),
                            active_eligible_n=active['nontrivial_node_denominator'],
                            active_chain_fraction=active['chain_fraction_among_graphs_with_at_least_two_nodes']))
    write_csv(args.output_dir/'by_workflow_compact.csv', compact)
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.output_dir/'graph_records.json').write_text(json.dumps(graphs,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({item['group']['dataset']:{'runs':item['run_claim_count'],
                     'active_kinds':item['active']['graph_kind_counts'],
                     'chain_fraction_nontrivial':item['active']['chain_fraction_among_graphs_with_at_least_two_nodes'],
                     'endpoint_cross_check':item['pro_endpoint_cross_check'],
                     'legacy_edge_audit':item['legacy_edge_audit']} for item in summary['by_dataset']},indent=2))


if __name__=='__main__':
    main()
