"""
Pipeline 8: themed charts saved to outputs/images/.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

import src.threat_core as tc

T = tc.THEME


def apply_chart_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": T["bg"], "axes.facecolor": T["panel"], "savefig.facecolor": T["bg"],
        "axes.edgecolor": T["line"], "axes.labelcolor": T["muted"], "axes.titlecolor": T["text"],
        "axes.titlesize": 14, "axes.titleweight": "bold", "axes.grid": True,
        "grid.color": T["line"], "grid.alpha": 0.5, "grid.linewidth": 0.6,
        "xtick.color": T["muted"], "ytick.color": T["muted"], "text.color": T["text"],
        "legend.facecolor": T["panel"], "legend.edgecolor": T["line"], "font.size": 10,
    })


def save_fig(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(tc.IMAGES / name, dpi=150)
    plt.close(fig)
    print(f"        saved outputs/images/{name}")


def pretty(names) -> list[str]:
    return [tc.LABELS.get(n, n) for n in names]


def stage_visualize(ctx, args):
    apply_chart_theme()
    sev_color = {k: T[k] if k in T else T["safe"] for k in ("low", "medium", "high", "critical")}
    sev_color["none"] = T["safe"]
    class_color = {c: sev_color[tc.SEVERITY[c]] for c in tc.CLASSES}

    # 1 confusion matrix
    cm = confusion_matrix(ctx["y_test"], ctx["rf_pred"], labels=tc.CLASSES)
    cmap = LinearSegmentedColormap.from_list("watch", [T["panel"], T["low"], T["safe"]])
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False, linewidths=1, linecolor=T["bg"],
                xticklabels=pretty(tc.CLASSES), yticklabels=pretty(tc.CLASSES), ax=ax)
    ax.set(title="Random Forest confusion matrix", xlabel="Predicted", ylabel="Actual")
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save_fig(fig, "confusion_matrix.png")

    # 2 class distribution
    counts = ctx["df"]["label"].value_counts().reindex(tc.CLASSES)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(pretty(counts.index), counts.values, color=[class_color[c] for c in counts.index])
    ax.bar_label(bars, fmt="{:,.0f}", color=T["text"], padding=3)
    ax.set(title="Traffic mix in the dataset", ylabel="Connections")
    ax.grid(axis="x", visible=False)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    save_fig(fig, "attack_distribution.png")

    # 3 feature importance
    pre, model = ctx["rf"].named_steps["pre"], ctx["rf"].named_steps["model"]
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    imp = pd.Series(model.feature_importances_, index=names).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(imp.index, imp.values, color=T["low"])
    ax.set(title="What the Random Forest looks at most", xlabel="Importance")
    ax.grid(axis="y", visible=False)
    save_fig(fig, "feature_importance.png")

    # 4 anomaly scores
    is_attack = (ctx["y_test"] != "normal").values
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bins = np.linspace(ctx["iso_score"].min(), ctx["iso_score"].max(), 60)
    ax.hist(ctx["iso_score"][~is_attack], bins=bins, color=T["safe"], alpha=0.85, label="Normal")
    ax.hist(ctx["iso_score"][is_attack], bins=bins, color=T["critical"], alpha=0.85, label="Attack")
    ax.axvline(ctx["thr"], color=T["medium"], linestyle="--", linewidth=1.5, label="Alert threshold")
    ax.set(title="Isolation Forest anomaly scores", xlabel="Score (lower is more unusual)", ylabel="Connections")
    ax.legend()
    save_fig(fig, "anomaly_scores.png")

    # 5 alert timeline
    al = ctx["alerts"].copy()
    al["timestamp"] = pd.to_datetime(al["timestamp"])
    al["bucket"] = al["timestamp"].dt.floor("2min")
    order = [s for s in ("low", "medium", "high", "critical") if s in set(al["severity"])]
    piv = al.groupby(["bucket", "severity"]).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(piv))
    bottom = np.zeros(len(piv))
    for s in order:
        ax.bar(x, piv[s].values, bottom=bottom, width=0.85, color=sev_color[s], label=s)
        bottom += piv[s].values
    ax.set_xticks(range(0, len(piv), 3))
    ax.set_xticklabels([t.strftime("%H:%M") for t in piv.index[::3]], rotation=0)
    ax.set(title="Alerts over time (2-minute buckets)", xlabel="Time", ylabel="Alerts")
    ax.grid(axis="x", visible=False)
    ax.legend(title="Severity")
    save_fig(fig, "alert_timeline.png")
