# Source-access observations

Saved traces and terminal judge scores for five cases, four source-presentation conditions, and six workflows: 120 traces in total. Each trace contains six steps. The score measures adoption of the target false claim on a scale from 0 to 1.

From the repository root:

```sh
python archive/source_access/recompute.py --output-dir reproduced/local/source_access
```

The script joins the records, checks saved scores against judge responses, and writes `per_run_scores.csv`, `condition_summary.csv`, and `summary.json`.

- `raw/`: traces, terminal prefixes, judge responses, and scores.
- `recomputed/`: saved per-trace scores and condition means.
- `source_manifest.json`: source file hashes.
