"""
visualization/dashboard.py
==========================
Generate all dashboard plots and deep-dive figures.

Usage:
    from src.visualization.dashboard import plot_dashboard, plot_deepdive
    plot_dashboard(features)
    plot_deepdive(features, report_df)
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def plot_dashboard(
    features: pd.DataFrame,
    model_features: list[str],
    output_dir: str = "outputs/figures",
    dpi: int = 150,
):
    """Generate the 8-panel anomaly dashboard."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    anomalies = features[features["consensus"] == 1]

    # Reason data
    reason_data = (
        features.loc[features["consensus"] == 1, "reasons"]
        .str.split(" \\| ")
        .explode()
        .str.replace(r"\(\d+ companies\)", "", regex=True)
        .str.strip()
    )
    reason_counts = reason_data.value_counts()

    fig, axes = plt.subplots(4, 2, figsize=(20, 24))
    fig.suptitle(
        "Interpretable Anomaly Detection Dashboard\nColombia Power Grid 2020–2023",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # A. Timeline
    ax = axes[0, 0]
    ax.plot(
        features["timestamp"],
        features["if_score_norm"],
        alpha=0.3,
        linewidth=0.3,
        color="steelblue",
    )
    ax.scatter(
        anomalies["timestamp"],
        anomalies["if_score_norm"],
        color="red",
        s=8,
        zorder=5,
        label=f"Consensus anomalies (n={len(anomalies)})",
    )
    ax.set_title("A. Anomaly Scores Over Time")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # B. Feature importance
    imp = {
        f: abs(np.corrcoef(features[f], features["if_score"])[0, 1])
        for f in model_features
    }
    imp_s = dict(sorted(imp.items(), key=lambda x: x[1]))
    ax = axes[0, 1]
    ax.barh(list(imp_s.keys()), list(imp_s.values()), color="steelblue")
    ax.set_title("B. Feature Importance (|corr| with IF Score)")
    for i, (k, v) in enumerate(imp_s.items()):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)

    # C. Reasons
    ax = axes[1, 0]
    colors = [
        "#e74c3c",
        "#3498db",
        "#2ecc71",
        "#f39c12",
        "#9b59b6",
        "#1abc9c",
        "#e67e22",
    ]
    ax.barh(
        reason_counts.index, reason_counts.values, color=colors[: len(reason_counts)]
    )
    ax.set_title("C. Why Anomalies Were Flagged")
    for i, v in enumerate(reason_counts.values):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=9)

    # D. IF vs OCSVM
    ax = axes[1, 1]
    norm_sample = features[features["consensus"] == 0].sample(
        min(3000, len(features)), random_state=42
    )
    ax.scatter(
        norm_sample["if_score_norm"],
        norm_sample["ocsvm_score_norm"],
        alpha=0.15,
        s=3,
        c="gray",
        label="Normal",
    )
    ax.scatter(
        anomalies["if_score_norm"],
        anomalies["ocsvm_score_norm"],
        c="red",
        s=15,
        zorder=5,
        label="Consensus",
    )
    ax.set_xlabel("IF Score")
    ax.set_ylabel("OCSVM Score")
    ax.set_title("D. Model Agreement")
    ax.legend(fontsize=8)

    # E. Hour distribution
    ax = axes[2, 0]
    h = anomalies["timestamp"].dt.hour.value_counts().reindex(range(24), fill_value=0)
    ax.bar(h.index, h.values, color="coral", edgecolor="darkred")
    ax.set_xlabel("Hour of Day")
    ax.set_title("E. Anomalies by Hour of Day")
    ax.set_xticks(range(0, 24, 2))

    # F. Monthly
    ax = axes[2, 1]
    monthly = anomalies.set_index("timestamp").resample("ME").size()
    ax.bar(monthly.index, monthly.values, width=25, color="steelblue", edgecolor="navy")
    ax.set_title("F. Anomalies by Month")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # G. Total power
    ax = axes[3, 0]
    ax.plot(
        features["timestamp"],
        features["total_power"] / 1e6,
        alpha=0.4,
        linewidth=0.3,
        color="steelblue",
    )
    ax.scatter(
        anomalies["timestamp"], anomalies["total_power"] / 1e6, c="red", s=8, zorder=5
    )
    ax.set_ylabel("Total Power (millions)")
    ax.set_title("G. System Power with Anomalies Highlighted")

    # H. Widespread vs score
    ax = axes[3, 1]
    ax.scatter(
        features["n_flagged_3z"],
        features["if_score_norm"],
        c=features["consensus"],
        cmap="coolwarm",
        alpha=0.2,
        s=3,
    )
    ax.set_xlabel("# Companies with |z| > 3")
    ax.set_ylabel("IF Score")
    ax.set_title("H. More Companies Flagged → Higher Anomaly Score")

    path = f"{output_dir}/anomaly_dashboard.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


def plot_deepdive(
    features: pd.DataFrame,
    report_df: pd.DataFrame,
    n_days: int = 2,
    output_dir: str = "outputs/figures",
    dpi: int = 150,
):
    """Plot hour-by-hour scores for the top anomaly days."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    top_dates = report_df["timestamp"].dt.date.unique()[:n_days]

    fig, axes = plt.subplots(len(top_dates), 1, figsize=(16, 5 * len(top_dates)))
    if len(top_dates) == 1:
        axes = [axes]

    for i, date in enumerate(top_dates):
        ax = axes[i]
        day = features[features["timestamp"].dt.date == date]
        ax.plot(
            day["timestamp"].dt.hour,
            day["if_score_norm"],
            "b-o",
            ms=5,
            label="IF Score",
        )
        ax.plot(
            day["timestamp"].dt.hour,
            day["ocsvm_score_norm"],
            color="orange",
            marker="s",
            ms=4,
            label="OCSVM Score",
        )
        day_anom = day[day["consensus"] == 1]
        ax.scatter(
            day_anom["timestamp"].dt.hour,
            day_anom["if_score_norm"],
            c="red",
            s=80,
            zorder=5,
            label="Consensus",
        )
        ax.set_xlabel("Hour")
        ax.set_ylabel("Score")
        ax.set_title(f"Deep Dive: {date}")
        ax.legend(fontsize=8)
        ax.set_xticks(range(24))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{output_dir}/anomaly_deepdive.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
