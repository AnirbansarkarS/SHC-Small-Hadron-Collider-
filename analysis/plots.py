"""Plotting helpers for Phase 2 analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_analysis(
    csv_path="data/datasets/phase2_runs.csv",
    output_dir="analysis",
    max_runs_per_label=10,
):
    df = pd.read_csv(csv_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    label_colors = {"stable": "green", "unstable": "red"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for label in sorted(df["label"].unique()):
        subset = df[df["label"] == label]
        ax.hist(
            subset["energy"],
            bins=60,
            alpha=0.6,
            label=label,
            color=label_colors.get(label, "gray"),
        )
    ax.set_title("Energy distribution")
    ax.set_xlabel("Energy (J)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path / "energy_distribution.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 6))
    for label in sorted(df["label"].unique()):
        runs = (
            df[df["label"] == label]["run_id"].drop_duplicates().head(max_runs_per_label)
        )
        for run_id in runs:
            run = df[df["run_id"] == run_id]
            ax.plot(
                run["x"],
                run["y"],
                lw=0.6,
                alpha=0.8,
                color=label_colors.get(label, "gray"),
            )
    ax.set_title("Trajectory overlays")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path / "trajectory_overlays.png", dpi=150)

    run_summary = df[["run_id", "label", "field_strength", "drift"]].drop_duplicates()

    fig, ax = plt.subplots(figsize=(6, 4))
    for label in sorted(run_summary["label"].unique()):
        subset = run_summary[run_summary["label"] == label]
        ax.scatter(
            subset["field_strength"],
            subset["drift"],
            s=18,
            alpha=0.8,
            label=label,
            color=label_colors.get(label, "gray"),
        )
    ax.set_title("B field vs drift")
    ax.set_xlabel("Field strength (T)")
    ax.set_ylabel("Drift (fraction)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path / "bfield_vs_drift.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 5))
    for label in sorted(df["label"].unique()):
        runs = (
            df[df["label"] == label]["run_id"].drop_duplicates().head(max_runs_per_label)
        )
        for run_id in runs:
            run = df[df["run_id"] == run_id]
            ax.plot(
                run["x"],
                run["vx"],
                lw=0.6,
                alpha=0.7,
                color=label_colors.get(label, "gray"),
            )
    ax.set_title("Phase-space diagram (x vs vx)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("vx (m/s)")
    fig.tight_layout()
    fig.savefig(output_path / "phase_space_x_vx.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5, 4))
    counts = run_summary["label"].value_counts()
    ax.bar(counts.index, counts.values, color=[label_colors.get(l, "gray") for l in counts.index])
    ax.set_title("Stability histogram")
    ax.set_xlabel("Label")
    ax.set_ylabel("Runs")
    fig.tight_layout()
    fig.savefig(output_path / "stability_histogram.png", dpi=150)

    plt.show()


if __name__ == "__main__":
    plot_analysis()
