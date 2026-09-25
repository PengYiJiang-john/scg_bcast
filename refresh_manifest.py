"""Refresh or verify the experiment release file inventory; no manuscript files."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXCLUDED_PARTS = {'.git', '.venv', '__pycache__', '.pytest_cache'}


def inventory():
    rows = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in rel.parts):
            continue
        if str(rel) in {'manifest.json', '.DS_Store'} or rel.suffix in {'.pyc', '.pyo'}:
            continue
        if str(rel).startswith(('protocols/latex/', 'reproduced/local/')):
            continue
        content = path.read_bytes()
        rows.append({'path': rel.as_posix(), 'sha256': hashlib.sha256(content).hexdigest(), 'size_bytes': len(content)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    rows = inventory()
    if args.check:
        saved = json.loads((ROOT/'manifest.json').read_text())
        if rows != saved['files']:
            raise SystemExit('Manifest does not match current release files; review changes then refresh.')
        print(f'Checked {len(rows)} experiment release files.')
        return
    result = {
        'package': 'SCG_BCAST_Experiments_audited',
        'audit_date_local': '2026-09-26',
        'input_archive': 'SCG_BCAST_Experiments.zip',
        'input_archive_claims_preserved_in': 'provenance/original_manifest.json',
        'llm_experiments_E1_E2_E3_status': 'planned_not_executed',
        'numerical_status': 'locally_rerun_with_complete_constructed_witness_tables',
        'historical_H22_status': 'archived_raw_responses_recovered_and_reaggregated; generator_metadata_incomplete',
        'propagation_graph_status': 'posthoc_metrics_on_prescribed_workflows_and_legacy_rule_labels; not_semantic_ground_truth',
        'release_excludes': ['protocols/latex/', '.venv/', '.git/', '__pycache__/', 'reproduced/local/'],
        'notes': 'This manifest is a file-integrity inventory, not a certificate of empirical or theoretical validity.',
        'files': rows,
    }
    (ROOT/'manifest.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(f'Recorded {len(rows)} experiment release files. Manuscript LaTeX excerpts are excluded.')


if __name__ == '__main__':
    main()
