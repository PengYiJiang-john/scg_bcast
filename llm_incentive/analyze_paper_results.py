from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_full"
PAPER = ROOT / "paper"
AGENTS = ("A", "B", "C")
MODES = ("careful", "shortcut", "distorter")
MODE_LABELS = {
    "careful": "Careful",
    "shortcut": "Shortcut",
    "distorter": "Distorter",
}
ENV_LABELS = {
    "modular": "Modular",
    "majority": "Majority",
    "gatekeeper": "Gatekeeper",
}


def bootstrap_mean_ci(values: np.ndarray, seed: int, draws: int = 10_000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    lower, upper = np.quantile(sampled, [0.025, 0.975])
    return float(values.mean()), float(lower), float(upper)


def profile_tuple(row: pd.Series) -> tuple[str, str, str]:
    return tuple(str(row[f"mode_{agent}"]) for agent in AGENTS)  # type: ignore[return-value]


def pure_equilibria(
    profiles: list[tuple[str, str, str]],
    payoff,
    tolerance: float = 1e-10,
) -> list[tuple[str, str, str]]:
    equilibria: list[tuple[str, str, str]] = []
    for profile in profiles:
        stable = True
        for index, agent in enumerate(AGENTS):
            current = payoff(profile, agent)
            for alternative in MODES:
                candidate = list(profile)
                candidate[index] = alternative
                if payoff(tuple(candidate), agent) < current - tolerance:
                    stable = False
                    break
            if not stable:
                break
        if stable:
            equilibria.append(profile)
    return equilibria


def format_profile(profile: tuple[str, str, str]) -> str:
    return "|".join(profile)


def analyze(results: Path, paper: Path) -> None:
    paper.mkdir(parents=True, exist_ok=True)
    mode_quality = pd.read_csv(results / "mode_quality.csv")
    profile_payoffs = pd.read_csv(results / "profile_payoffs.csv")
    coalition = pd.read_csv(results / "coalition_case_losses.csv")
    environment_summary = pd.read_csv(results / "environment_summary.csv")

    case_mode = (
        mode_quality.groupby(["case_id", "family", "agent", "mode"], as_index=False)
        .standalone_loss.mean()
    )
    mode_rows: list[dict[str, object]] = []
    for (agent, mode), group in case_mode.groupby(["agent", "mode"], sort=False):
        mean, low, high = bootstrap_mean_ci(group.standalone_loss.to_numpy(), 1000 + ord(agent) + MODES.index(mode))
        mode_rows.append(
            {
                "agent": agent,
                "mode": mode,
                "cases": group.case_id.nunique(),
                "mean_loss": mean,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    mode_ci = pd.DataFrame(mode_rows).sort_values(["agent", "mode"])
    mode_ci.to_csv(paper / "table_mode_quality_ci.csv", index=False)

    family_rows: list[dict[str, object]] = []
    for (family, mode), group in case_mode.groupby(["family", "mode"]):
        per_case = group.groupby("case_id").standalone_loss.mean().to_numpy()
        mean, low, high = bootstrap_mean_ci(per_case, 2000 + len(family) + MODES.index(mode))
        family_rows.append(
            {
                "task_family": family,
                "mode": mode,
                "cases": len(per_case),
                "mean_loss": mean,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    family_ci = pd.DataFrame(family_rows).sort_values(["task_family", "mode"])
    family_ci.to_csv(paper / "table_task_family_ci.csv", index=False)

    key_profiles = {mode: "|".join([mode] * 3) for mode in MODES}
    profile_case = (
        coalition[coalition.coalition == "ABC"]
        .groupby(["environment", "profile", "case_id"], as_index=False)
        .loss.mean()
    )
    workflow_rows: list[dict[str, object]] = []
    for environment in ENV_LABELS:
        for mode, profile in key_profiles.items():
            group = profile_case[(profile_case.environment == environment) & (profile_case.profile == profile)]
            mean, low, high = bootstrap_mean_ci(
                group.loss.to_numpy(),
                3000 + list(ENV_LABELS).index(environment) * 10 + MODES.index(mode),
            )
            workflow_rows.append(
                {
                    "environment": environment,
                    "profile": profile,
                    "cases": group.case_id.nunique(),
                    "mean_loss": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    workflow_ci = pd.DataFrame(workflow_rows)
    workflow_ci.to_csv(paper / "table_workflow_profiles_ci.csv", index=False)

    coalition_means = coalition.groupby(["environment", "profile", "coalition"]).loss.mean().to_dict()
    equilibrium_rows: list[dict[str, object]] = []
    gatekeeper_rows: list[dict[str, object]] = []
    for environment in ENV_LABELS:
        env_payoffs = profile_payoffs[profile_payoffs.environment == environment]
        profiles = [profile_tuple(row) for _, row in env_payoffs.iterrows()]
        shapley = {
            profile_tuple(row): {agent: float(row[f"psi_{agent}"]) for agent in AGENTS}
            for _, row in env_payoffs.iterrows()
        }
        full_loss = {profile_tuple(row): float(row.full_loss) for _, row in env_payoffs.iterrows()}
        loo: dict[tuple[str, str, str], dict[str, float]] = {}
        for profile in profiles:
            profile_name = format_profile(profile)
            loo[profile] = {}
            for agent in AGENTS:
                without_agent = "".join(item for item in AGENTS if item != agent)
                loo[profile][agent] = (
                    coalition_means[(environment, profile_name, "ABC")]
                    - coalition_means[(environment, profile_name, without_agent)]
                )

        methods = {
            "Behavior Shapley": lambda p, a: shapley[p][a],
            "LOO": lambda p, a: loo[p][a],
            "Team loss": lambda p, a: full_loss[p],
        }
        for method, payoff in methods.items():
            equilibria = pure_equilibria(profiles, payoff)
            equilibrium_rows.append(
                {
                    "environment": environment,
                    "incentive": method,
                    "pure_ne_count": len(equilibria),
                    "equilibria": "; ".join(format_profile(profile) for profile in equilibria),
                    "best_ne_loss": min(full_loss[profile] for profile in equilibria),
                    "worst_ne_loss": max(full_loss[profile] for profile in equilibria),
                    "c_absent_loss_best": min(
                        coalition_means[(environment, format_profile(profile), "AB")]
                        for profile in equilibria
                    ),
                    "c_absent_loss_worst": max(
                        coalition_means[(environment, format_profile(profile), "AB")]
                        for profile in equilibria
                    ),
                }
            )

        if environment == "gatekeeper":
            reference = ["careful", "careful", "careful"]
            for index, agent in enumerate(AGENTS):
                for mode in MODES:
                    profile_list = reference.copy()
                    profile_list[index] = mode
                    profile = tuple(profile_list)
                    gatekeeper_rows.append(
                        {
                            "agent": agent,
                            "mode": mode,
                            "full_loss": full_loss[profile],
                            "behavior_shapley_penalty": shapley[profile][agent],
                            "loo_penalty": loo[profile][agent],
                        }
                    )

    equilibria = pd.DataFrame(equilibrium_rows)
    equilibria.to_csv(paper / "table_equilibrium_comparison.csv", index=False)
    gatekeeper = pd.DataFrame(gatekeeper_rows)
    gatekeeper.to_csv(paper / "table_gatekeeper_incentives.csv", index=False)

    audit = environment_summary[
        [
            "environment",
            "global_optimum_loss",
            "equilibrium_count",
            "max_equilibrium_gap",
            "E",
            "two_E_bound",
            "all_27_starts_converged_to_NE",
            "mean_best_response_steps",
            "max_exact_potential_deviation_error",
        ]
    ].copy()
    audit.to_csv(paper / "table_theory_audit.csv", index=False)

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
    colors = {"careful": "#2F6B4F", "shortcut": "#D39A37", "distorter": "#B94B45"}
    fig, axes = plt.subplots(1, 3, figsize=(11.3, 3.2))

    ax = axes[0]
    x = np.arange(len(AGENTS))
    width = 0.23
    for offset, mode in zip((-1, 0, 1), MODES, strict=True):
        rows = mode_ci[mode_ci["mode"] == mode].set_index("agent").loc[list(AGENTS)]
        means = rows.mean_loss.to_numpy()
        yerr = np.vstack((means - rows.ci95_low.to_numpy(), rows.ci95_high.to_numpy() - means))
        ax.bar(
            x + offset * width,
            means,
            width,
            yerr=yerr,
            capsize=2,
            color=colors[mode],
            edgecolor="white",
            linewidth=0.5,
            label=MODE_LABELS[mode],
        )
    ax.set_xticks(x, AGENTS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Standalone loss")
    ax.set_title("(a) LLM behavior modes")
    ax.legend(frameon=False, ncol=1, fontsize=8)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)

    ax = axes[1]
    x = np.arange(len(ENV_LABELS))
    for offset, mode in zip((-1, 0, 1), MODES, strict=True):
        rows = workflow_ci[workflow_ci.profile == key_profiles[mode]].set_index("environment").loc[list(ENV_LABELS)]
        means = rows.mean_loss.to_numpy()
        yerr = np.vstack((means - rows.ci95_low.to_numpy(), rows.ci95_high.to_numpy() - means))
        ax.bar(
            x + offset * width,
            means,
            width,
            yerr=yerr,
            capsize=2,
            color=colors[mode],
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_xticks(x, [ENV_LABELS[item] for item in ENV_LABELS], rotation=16)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Full-workflow loss")
    ax.set_title("(b) End-to-end outcomes")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)

    ax = axes[2]
    method_order = ["Behavior Shapley", "LOO", "Team loss"]
    method_color = {"Behavior Shapley": "#2F6B4F", "LOO": "#567AA3", "Team loss": "#7B7B7B"}
    x = np.arange(len(ENV_LABELS))
    for offset, method in zip((-1, 0, 1), method_order, strict=True):
        rows = equilibria[equilibria.incentive == method].set_index("environment").loc[list(ENV_LABELS)]
        ax.bar(
            x + offset * width,
            rows.pure_ne_count.to_numpy(),
            width,
            color=method_color[method],
            edgecolor="white",
            linewidth=0.5,
            label=method,
        )
    ax.set_xticks(np.arange(len(ENV_LABELS)), [ENV_LABELS[item] for item in ENV_LABELS], rotation=16)
    ax.set_yticks([1, 3, 5, 7, 9])
    ax.set_ylim(0.5, 9.5)
    ax.set_ylabel("Number of pure equilibria")
    ax.set_title("(c) Incentive selectivity")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(width=0.7, length=3)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(paper / "figure_incentive_experiment.pdf", bbox_inches="tight")
    fig.savefig(paper / "figure_incentive_experiment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(mode_ci.to_string(index=False))
    print("\nEquilibrium comparison")
    print(equilibria[["environment", "incentive", "pure_ne_count", "best_ne_loss", "worst_ne_loss"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute task-level bootstrap tables and figure from incentive losses.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "reproduced" / "local" / "llm_incentive" / "paper")
    args = parser.parse_args()
    if args.output_dir.resolve() == PAPER.resolve():
        parser.error("Use a separate output directory to preserve the archived paper artifacts")
    analyze(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
