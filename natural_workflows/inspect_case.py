"""Read a frozen workflow and its saved semantic/replay records, offline."""
import argparse
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_records(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_case(case_id, root=ROOT):
    for task in json.loads((root / 'config.json').read_text())['tasks']:
        directory = root / 'data' / task
        case = next((row for row in read_records(directory / 'inputs.jsonl')
                     if row['case_id'] == case_id), None)
        if case is None:
            continue
        result = {'input': case}
        for section, filename in [('factual', 'factual.jsonl'),
                                  ('semantic_graph', 'semantic_graphs.jsonl.gz')]:
            result[section] = next(row for row in read_records(directory / filename)
                                   if row['case_id'] == case_id)
        result['replay_pairs'] = [row for row in read_records(directory / 'replay_pairs.jsonl.gz')
                                  if row['case_id'] == case_id]
        return result
    raise ValueError(f'Unknown case: {case_id}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--section', choices=['input', 'factual', 'semantic_graph', 'replay_pairs'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = load_case(args.case_id)
    if args.section:
        result = result[args.section]
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
