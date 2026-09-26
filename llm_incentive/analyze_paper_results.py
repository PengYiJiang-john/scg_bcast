from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
POSITIONS = ("A", "B", "C")
MODES = ("careful", "shortcut", "distorter")
MODE_LABELS = {"careful": "Careful", "shortcut": "Shortcut", "distorter": "Distorter"}
COLORS = {"careful": "#2F6B4F", "shortcut": "#D39A37", "distorter": "#B94B45"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild incentive tables and the English figure.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "reproduced/local/llm_incentive/paper")
    args = parser.parse_args()
    PAPER = args.output_dir
    PAPER.mkdir(parents=True, exist_ok=True)
    profiles = pd.read_csv(args.results_dir / "profile_game.csv")
    with gzip.open(args.results_dir / "profile_game_traces.json.gz", "rt", encoding="utf-8") as handle:
        traces = json.load(handle)
    trace_df = pd.DataFrame(
        [
            {
                "profile": item["profile"],
                "coalition": item["coalition"],
                "repeat": item["repeat"],
                "loss": item["loss"],
            }
            for item in traces
        ]
    )

    careful_coalitions = (
        trace_df[trace_df.profile == "careful|careful|careful"]
        .groupby("coalition", as_index=False)
        .loss.agg(["mean", "std"])
        .reset_index()
    )
    coalition_order = ["EMPTY", "A", "B", "C", "AB", "AC", "BC", "ABC"]
    careful_coalitions["coalition"] = pd.Categorical(
        careful_coalitions.coalition, categories=coalition_order, ordered=True
    )
    careful_coalitions = careful_coalitions.sort_values("coalition")
    careful_coalitions.to_csv(PAPER / "table_identity_cancellation.csv", index=False)

    homogeneous_rows = []
    for mode in MODES:
        profile = "|".join([mode] * 3)
        rows = trace_df[(trace_df.profile == profile) & (trace_df.coalition == "ABC")]
        homogeneous_rows.append(
            {
                "mode_profile": profile,
                "mean_loss": rows.loss.mean(),
                "std_over_replays": rows.loss.std(ddof=1),
                "replay_losses": ";".join(f"{value:.6f}" for value in rows.sort_values("repeat").loss),
            }
        )
    homogeneous = pd.DataFrame(homogeneous_rows)
    homogeneous.to_csv(PAPER / "table_mode_profiles.csv", index=False)

    incentive_rows = []
    for position in POSITIONS:
        selected = profiles.copy()
        for other in POSITIONS:
            if other != position:
                selected = selected[selected[f"mode_{other}"] == "careful"]
        for _, row in selected.iterrows():
            incentive_rows.append(
                {
                    "position": position,
                    "mode": row[f"mode_{position}"],
                    "shapley_loss_penalty": row[f"psi_{position}"],
                    "full_loss": row["full_loss"],
                }
            )
    incentives = pd.DataFrame(incentive_rows).sort_values(["position", "mode"])
    incentives.to_csv(PAPER / "table_unilateral_incentives.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.15))

    ax = axes[0]
    means = homogeneous.mean_loss.to_numpy()
    x = np.arange(3)
    ax.bar(x, means, width=0.58, color=[COLORS[mode] for mode in MODES], edgecolor="white")
    for index, mode in enumerate(MODES):
        profile = "|".join([mode] * 3)
        points = trace_df[(trace_df.profile == profile) & (trace_df.coalition == "ABC")].sort_values("repeat")
        jitter = np.linspace(-0.08, 0.08, len(points))
        ax.scatter(index + jitter, points.loss, color="#202020", s=13, zorder=3)
    ax.set_xticks(x, [MODE_LABELS[mode] for mode in MODES], rotation=12)
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("Terminal semantic loss")
    ax.set_title("(a) Mode-controlled outcomes")

    ax = axes[1]
    width = 0.23
    x = np.arange(3)
    for offset, mode in zip((-1, 0, 1), MODES, strict=True):
        rows = incentives[incentives["mode"] == mode].set_index("position").loc[list(POSITIONS)]
        ax.bar(
            x + offset * width,
            rows.shapley_loss_penalty,
            width,
            color=COLORS[mode],
            edgecolor="white",
            label=MODE_LABELS[mode],
        )
    ax.axhline(0, color="#555555", linewidth=0.7)
    ax.set_xticks(x, POSITIONS)
    ax.set_ylabel("Behavior Shapley loss penalty")
    ax.set_title("(b) Unilateral mode incentives")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.bar(
        np.arange(len(careful_coalitions)),
        careful_coalitions["mean"],
        color="#567AA3",
        edgecolor="white",
        width=0.7,
    )
    ax.set_xticks(np.arange(len(careful_coalitions)), coalition_order, rotation=25)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Terminal semantic loss")
    ax.set_title("(c) Identity-cancellation replay")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
        ax.tick_params(width=0.7, length=3)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(PAPER / "figure_correct_incentive_experiment.pdf", bbox_inches="tight")
    fig.savefig(PAPER / "figure_correct_incentive_experiment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
