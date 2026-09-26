"""Verify all frozen workflows, graph references, and saved endpoint contrasts."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from natural_bcast.data import TaskCase, score
from natural_bcast.workflow import SPECS
from inspect_case import read_records


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_number(left, right):
    return math.isfinite(float(left)) and math.isfinite(float(right)) and math.isclose(
        float(left), float(right), rel_tol=0, abs_tol=1e-12)


def check_score(case, endpoint, saved):
    recalculated = score(case, endpoint['answer'], endpoint.get('scale', ''))
    require(saved.keys() == recalculated.keys(), f'{case.case_id}: score fields differ')
    for key, expected in recalculated.items():
        actual = saved[key]
        require(same_number(actual, expected) if isinstance(expected, (int, float)) else actual == expected,
                f'{case.case_id}: score mismatch in {key}')
    return recalculated


def check_pair(case, graph, row):
    relations = {b['behavior_id']: b for b in graph['behaviors']}
    preserved = {bid for bid, b in relations.items() if b['relation_type'] == 'PRESERVED'}
    target = row['target']
    background = set(row['background_ids'])
    require(row['case_id'] == case.case_id and row['task'] == case.task, 'Replay case mismatch')
    require(target in relations and relations[target]['relation_type'] in {'NEW', 'TRANSFORMED'},
            'Replay target must be NEW or TRANSFORMED')
    require(set(row['fixed_preserved_ids']) == preserved, 'Preservation metadata mismatch')
    require(background <= relations.keys() and not (background & preserved) and target not in background,
            'Invalid behavior background')
    require(row['status'] == 'cached_complete_pair' and row['identity']['applicable'],
            'Incomplete replay pair')
    for side, active, anchor in [('without', background | preserved, None),
                                 ('with_target', background | preserved | {target}, target)]:
        replay = row[side]
        require(set(replay['requested_active_behavior_ids']) == active, 'Active set mismatch')
        require(replay['anchored_target_id'] == anchor, 'Anchored target mismatch')
        require(set(replay['trace']) == set(SPECS), 'Incomplete six-agent replay')
        for role, spec in SPECS.items():
            require(set(replay['role_inputs'][role]) == set(spec.parents), 'Parent topology mismatch')
            for parent in spec.parents:
                require(replay['role_inputs'][role][parent] == replay['trace'][parent]['output'],
                        'Parent output differs from downstream input')
            require(not (set(replay['trace'][role]['cancelled']) & preserved), 'PRESERVED cancelled')
        final = replay['trace']['F']['output']
        require(replay['endpoint']['answer'] == final['proposed_answer'] and
                replay['endpoint'].get('scale', '') == final['answer_scale'], 'Endpoint binding mismatch')
        check_score(case, replay['endpoint'], replay['score'])
    require(same_number(row['without_loss'], row['without']['score']['loss']), 'Without loss mismatch')
    require(same_number(row['with_loss'], row['with_target']['score']['loss']), 'With loss mismatch')
    require(same_number(row['marginal_loss'], row['with_loss'] - row['without_loss']),
            'Marginal loss mismatch')


def pair_key(row):
    return row['case_id'], row['target'], row['sample'], tuple(row['background_ids'])


def verify(root=ROOT):
    config = json.loads((root / 'config.json').read_text())
    saved_summary = json.loads((root / 'summary.json').read_text())
    all_ids, all_pairs, case_rows, task_rows = set(), set(), [], []
    for task, expected_cases in config['tasks'].items():
        directory = root / 'data' / task
        inputs = list(read_records(directory / 'inputs.jsonl'))
        facts = list(read_records(directory / 'factual.jsonl'))
        graphs = list(read_records(directory / 'semantic_graphs.jsonl.gz'))
        require(len(inputs) == len(facts) == len(graphs) == expected_cases, f'{task}: case count mismatch')
        by_case, task_counts = {}, Counter()
        for position, (payload, factual, graph) in enumerate(zip(inputs, facts, graphs)):
            case = TaskCase.from_dict(payload)
            require(case.case_id == factual['case_id'] == graph['case_id'], 'Case order mismatch')
            require(case.task == factual['task'] == graph['task'] == task, 'Task mismatch')
            require(case.case_id not in all_ids, 'Duplicate case')
            all_ids.add(case.case_id)
            require(set(factual['outputs']) == set(SPECS), 'Incomplete factual workflow')
            final = factual['outputs']['F']
            require(factual['endpoint']['answer'] == final['proposed_answer'], 'Factual endpoint mismatch')
            require(factual['endpoint']['scale'] == final['answer_scale'], 'Factual scale mismatch')
            check_score(case, factual['endpoint'], factual['score'])
            nodes = {node['node_id'] for node in graph['nodes']}
            require(len(nodes) == len(graph['nodes']), 'Duplicate graph node')
            relations = graph['behaviors']
            require(len({b['behavior_id'] for b in relations}) == len(relations), 'Duplicate relation')
            counts = Counter(b['relation_type'] for b in relations)
            require(set(counts) <= {'NEW', 'TRANSFORMED', 'PRESERVED'}, 'Unknown semantic relation')
            for relation in relations:
                require(relation['owner'] in SPECS, 'Unknown relation owner')
                require(relation['output_node_id'] in nodes, 'Unknown output node')
                require(set(relation['input_node_ids']) <= nodes, 'Unknown input node')
            index = dict(task=task, case_id=case.case_id, locked_row_1based=position + 1,
                         dataset_split=case.split, experiment_split=factual.get('experiment_split', ''),
                         nodes=len(nodes), new=counts['NEW'], transformed=counts['TRANSFORMED'],
                         preserved=counts['PRESERVED'], factual_task_score=1 - factual['score']['loss'],
                         factual_loss=factual['score']['loss'], saved_replay_pairs=0)
            case_rows.append(index)
            by_case[case.case_id] = (case, graph, index)
            task_counts.update(counts)
        for row in read_records(directory / 'replay_pairs.jsonl.gz'):
            case, graph, index = by_case[row['case_id']]
            key = pair_key(row)
            require(key not in all_pairs, 'Duplicate replay pair')
            all_pairs.add(key)
            check_pair(case, graph, row)
            index['saved_replay_pairs'] += 1
            task_counts['pairs'] += 1
        task_rows.append({'task': task, 'cases': expected_cases, **task_counts})
    availability = list(read_records(root / 'replay_availability.jsonl.gz'))
    require(all(row['case_id'] in all_ids for row in availability), 'Availability references unknown case')
    expected_pairs = {pair_key(row) for row in availability if row['status'] == 'cached_complete_pair'}
    require(expected_pairs == all_pairs, 'Complete-pair coverage mismatch')
    require(dict(Counter(row['status'] for row in availability)) == saved_summary['replay_availability'],
            'Availability totals mismatch')
    require(task_rows == saved_summary['tasks'], 'Task summary mismatch')
    require(len(all_ids) == saved_summary['cases'], 'Total case count mismatch')
    with (root / 'case_index.csv').open(newline='') as handle:
        saved_index = list(csv.DictReader(handle))
    require(len(saved_index) == len(case_rows), 'Case index length mismatch')
    for old, new in zip(saved_index, case_rows):
        for field, value in new.items():
            require(same_number(old[field], value) if isinstance(value, (float, int)) else old[field] == value,
                    f'Case index mismatch: {new["case_id"]}, {field}')
    return {'cases': len(all_ids), 'tasks': task_rows, 'saved_replay_pairs': len(all_pairs),
            'endpoint_scores_checked': len(all_ids) + 2 * len(all_pairs), 'model_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    report = verify()
    output = json.dumps(report, indent=2) + '\n'
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / 'verification.json').write_text(output)
    print(output, end='')


if __name__ == '__main__':
    main()
