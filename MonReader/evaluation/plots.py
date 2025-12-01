
from __future__ import annotations
from typing import Dict, List, Iterable, Optional
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------
# Bar chart: compare scalar metrics across models
def plot_metric_bars(model_metrics: Dict[str, Dict[str, float]]) -> None:
    """Improved grouped bar chart (industry‑style) for Acc / Prec / Rec / F1.
    Cleaner visuals, consistent colors, tighter spacing, clearer labels.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    metrics_order = ["acc", "prec", "rec", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]  

    model_names = list(model_metrics.keys())
    n_models = len(model_names) # how many clusters
    n_metrics = len(metrics_order) # how many bars per cluster

    
    # obtain values for each bar for each cluster
    values = np.array([[model_metrics[m][k] for m in model_names] for k in metrics_order])
    
    # Build x axis
    x = np.arange(n_models)
    width = 0.18 # bar width

    fig, ax = plt.subplots(figsize=(12, 6))

    # grouped bars
    for i, (label, color) in enumerate(zip(metric_labels, colors)):
        # for each metric, recenters each group of bars around the model's x position
        ax.bar(
            x + (i - (n_metrics - 1) / 2) * width,
            values[i],
            width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title("Validation Metrics per Model", fontsize=14, weight="bold")
    ax.legend(frameon=False, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# F1 curves: F1 vs epoch for each model
def plot_f1_curves(f1_histories: Dict[str, List[float]], phase_a_epochs: int = 5) -> None:
    """Industry‑style F1 vs epoch plot with Phase A/B divider."""

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    plt.figure(figsize=(12, 6))

    # --- Phase A shaded region ---
    plt.axvspan(
        0.5,                          # slightly before epoch 1
        phase_a_epochs + 0.5,         # slightly after last Phase A epoch
        color="lightgrey",
        alpha=0.50,
        label="Phase A (Linear Probe)",
    )

    for (name, f1_list), color in zip(f1_histories.items(), colors):
        epochs = np.arange(1, len(f1_list) + 1)
        plt.plot(
            epochs, f1_list,
            marker="o", markersize=5,
            linewidth=2.2,
            color=color,
            label=name,
        )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation F1", fontsize=12)
    plt.title("Validation F1 vs Epoch per Model", fontsize=14, weight="bold")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend(frameon=False, fontsize=11)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# Radar / spider plot: multi-metric profile per model

def plot_radar_metrics(model_metrics: Dict[str, Dict[str, float]]) -> None:
    """ 
    Radar / Spider Plot where each model is represented by a polygon spreading over axes: accuracy, precision, recall, F1 and size efficiency
    """

    # Axes: Accuracy, Precision, Recall, F1, Efficiency
    base_axes = ["acc", "prec", "rec", "f1", "efficiency"]
    axis_labels = ["Accuracy", "Precision", "Recall", "F1", "F1/MB"]

    # Compute F1 per MB efficiency
    raw_eff = {
        # for each model, take its F1 score and divid by its model size MB 
        m : model_metrics[m]['f1'] / model_metrics[m]['size_mb']
        for m in model_metrics
    }

    # normalize efficiency to [0,1] standard for radar
    # find max efficiency across models
    max_eff = max(raw_eff.values())
    eff_scores = {m: raw_eff[m] / max_eff for m in raw_eff}

    # Compute the angles around the circle
    n_axes = len(base_axes)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1] # append first angle to the end to connect first and last angle

    # Creat circular plot
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    # Loop over models and build their polygons
    for (model_name, metrics), color in zip(model_metrics.items(), colors):
        values = [
            metrics.get("acc", 0),
            metrics.get("prec", 0),
            metrics.get("rec", 0),
            metrics.get("f1", 0),
            eff_scores[model_name],
        ]
        values += values[:1]

        ax.plot(angles, values, linewidth=2.2, color=color, label=model_name)
        ax.fill(angles, values, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_yticklabels([])
    # Raise title higher by adjusting pad
    ax.set_title(
        "Radar Comparison of Model Performance and Efficiency", fontsize=14, weight="bold", pad=30
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), frameon=False)

    plt.tight_layout()
    plt.show()
   
# ---------------------------------------------------------------------
# Pareto frontier: F1 vs size (or any x/y) across models
def plot_pareto_frontier(model_metrics: Dict[str, Dict[str, float]]) -> None:
    """Plot the Pareto frontier for (size_mb, F1).
    Frontier = models not dominated by any other model in both F1↑ and size↓.
    """

    model_names = list(model_metrics.keys())
    # values for 2d coords plot
    sizes = np.array([model_metrics[m]["size_mb"] for m in model_names])
    f1s = np.array([model_metrics[m]["f1"] for m in model_names])

    # Sort by size (ascending)
    order = np.argsort(sizes)
    sizes = sizes[order]
    f1s = f1s[order]
    sorted_names = [model_names[i] for i in order]

    # Compute Pareto frontier
    # walks through modesl from smallest to largest and a model is added to the frontier if its F1 is higher than previous models
    frontier_idx = []
    best_f1 = -np.inf
    for i in range(len(sizes)):
        if f1s[i] > best_f1:
            frontier_idx.append(i)
            best_f1 = f1s[i]

    plt.figure(figsize=(10, 6))

    # Scatter all points
    plt.scatter(sizes, f1s, s=160, color="#4C72B0", edgecolor="black", linewidth=0.7)
    for name, s, f in zip(sorted_names, sizes, f1s):
        plt.text(s + 0.3, f, name, fontsize=10)

    # Draw frontier line
    plt.plot(sizes[frontier_idx], f1s[frontier_idx], color="#C44E52", linewidth=2.4, label="Pareto Frontier")

    plt.xlabel("Model Size (MB)", fontsize=12)
    plt.ylabel("Validation F1", fontsize=12)
    plt.title("Pareto Frontier: F1 vs Model Size", fontsize=15, weight="bold", pad=15)
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend(frameon=False, fontsize=11)
    plt.tight_layout()
    plt.show()


def display_academic_results_table(models_metrics: dict):
    """
    Display an academic‑style results table for multiple models.
    """
    from tabulate import tabulate

    # Convert dictionary structure into a list of rows
    headers = ["Model", 'Loss', "Accuracy", "Precision", "Recall", "F1 Score", "Size (MB)"]
    rows = []

    for model_name, metrics in models_metrics.items():
        rows.append([
            model_name,
            f"{metrics.get('loss', 0):.4f}",
            f"{metrics.get('accuracy', 0):.4f}",
            f"{metrics.get('precision', 0):.4f}",
            f"{metrics.get('recall', 0):.4f}",
            f"{metrics.get('f1', 0):.4f}",
            f"{metrics.get('size_mb', 0):.2f}"
        ])

    # Generate table in academic formatting
    table = tabulate(rows, headers=headers, tablefmt="github", numalign="center", stralign="center")
    print("\nResults Table:\n")
    print(table)


def plot_kd_summary_figure(model_metrics: dict):
    """ 
    Knowledge Distillation Summary Figure with 3 Panels:
    Panel A: Multi-metric bar chart comparison
    Panel B: Pareto Plot (Model size vs F1 Score)
    Panel C: Radar Plot (KD vs Baseline MobileNetV2)
    """
    model_names = list(model_metrics.keys())
    # extract metrics
    loss = [model_metrics[m]['loss'] for m in model_names]
    acc = [model_metrics[m]["accuracy"] for m in model_names]
    prec = [model_metrics[m]["precision"] for m in model_names]
    rec = [model_metrics[m]["recall"] for m in model_names]
    f1 = [model_metrics[m]["f1"] for m in model_names]
    size = [model_metrics[m]["size_mb"] for m in model_names]

    # Create 3 panel figure
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(20,12))
    gs = GridSpec(2, 2, width_ratios=[1,1.2], height_ratios=[1,1], figure=fig)

    # publication-style color palette
    pub_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # ---------------------------------------------------------------------
    # Panel A Multi-metric bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(model_names))
    width = 0.13
    # plot grouped bars for each metric
    ax1.bar(x-2*width, loss, width, label='Loss')
    ax1.bar(x - width, acc, width, label="Accuracy")
    ax1.bar(x, prec,  width, label="Precision")
    ax1.bar(x + width, rec, width, label="Recall")
    ax1.bar(x + 2*width, f1, width, label="F1 Score")
    #ax1.bar(x + 3*width, size, width, label="Size (MB)")
    # subplot 1 cosmetics
    ax1.set_title("Panel A: Multi-Metric Comparison", fontsize=14, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # ---------------------------------------------------------------------
    # Panel B Pareto Frontier Plot (Model Size vs F1)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(size, f1, s=120)
    for i, m in enumerate(model_names):
        ax2.annotate(m, (size[i], f1[i]), textcoords='offset points', xytext=(5,5))
    # Compute Pareto frontier (maxmize f1, minimize size)
    points = list(zip(size, f1, model_names))
    # sort by size 
    points.sort(key = lambda x: x[0])
    pareto = []
    best_f1 = -1
    for s, f1, n in points:
        if f1 > best_f1:
            pareto.append((s, f1))
            best_f1 = f1

    # Draw frontier line
    if len(pareto) > 1:
        pf_size, pf_f1 = zip(*pareto)
        ax2.plot(pf_size, pf_f1, linestyle="--", color='red', linewidth=2, label='Pareto Frontier')
        ax2.legend()
    ax2.set_xlabel("Model Size (MB)")
    ax2.set_ylabel("F1 Score")
    ax2.set_title('Panel B: Model Size vs F1 Score (Pareto Plot)', weight='bold')
    ax2.grid(linestyle='--', alpha=0.7)

    # ---------------------------------------------------------------------
    # Panel C Radar Plot
    # Metrics to normalize (only Loss + Efficiency)
    labels = ["Accuracy", "Precision", "Recall", "F1", "Loss (inverted)", "Efficiency (F1/Size)"]
    kd = model_metrics["MobileNetV2_KD"]
    base = model_metrics["MobileNetV2"]

    # Compute raw efficiency
    kd_eff = kd["f1"] / kd["size_mb"]
    base_eff = base["f1"] / base["size_mb"]

    # Collect raw values (keep original 0-1 metrics untouched)
    raw_kd = np.array([
        kd["accuracy"], kd["precision"], kd["recall"], kd["f1"], kd["loss"], kd_eff
    ])
    raw_base = np.array([
        base["accuracy"], base["precision"], base["recall"], base["f1"], base["loss"], base_eff
    ])

    # Build kd_norm and base_norm 
    # Keep Accuracy, Precision, Recall, F1 unchanged
    kd_norm = [raw_kd[0], raw_kd[1], raw_kd[2], raw_kd[3]]
    base_norm = [raw_base[0], raw_base[1], raw_base[2], raw_base[3]]

    # Normalize Loss (lower = better, invert to higher = better)
    vals_loss = np.array([raw_kd[4], raw_base[4]])
    min_l, max_l = vals_loss.min(), vals_loss.max()
    if max_l - min_l == 0:
        kd_loss_norm = base_loss_norm = 1.0
    else:
        kd_loss_norm = 1 - ((raw_kd[4] - min_l) / (max_l - min_l))
        base_loss_norm = 1 - ((raw_base[4] - min_l) / (max_l - min_l))

    kd_norm.append(kd_loss_norm)
    base_norm.append(base_loss_norm)

    # Normalize Efficiency (F1 / Size) 
    vals_eff = np.array([raw_kd[5], raw_base[5]])
    min_e, max_e = vals_eff.min(), vals_eff.max()
    if max_e - min_e == 0:
        kd_eff_norm = base_eff_norm = 1.0
    else:
        kd_eff_norm = (raw_kd[5] - min_e) / (max_e - min_e)
        base_eff_norm = (raw_base[5] - min_e) / (max_e - min_e)

    kd_norm.append(kd_eff_norm)
    base_norm.append(base_eff_norm)

    # Prepare radar coordinates
    num_vars = len(labels)
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()

    kd_plot = kd_norm + kd_norm[:1]
    base_plot = base_norm + base_norm[:1]
    angles += angles[:1]

    ax3 = fig.add_subplot(gs[:, 1], polar=True)  # Panel C occupies full right side  # Radar subplot (publication-ready)
    ax3.plot(angles, kd_plot, label="MobileNetV2_KD", linewidth=2)
    ax3.fill(angles, kd_plot, alpha=0.25)

    ax3.plot(angles, base_plot, label="MobileNetV2", linewidth=2)
    ax3.fill(angles, base_plot, alpha=0.25)

    ax3.set_title("Panel C: Radar Plot (KD vs Baseline) — Normalized", weight='bold')
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(labels)
    ax3.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25))

    plt.tight_layout()
    plt.show()

    