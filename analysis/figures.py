"""Publication-quality figure generation for paper reproduction.

Figures generated:
1. pipeline_diagram.png: Architecture of multi-view active grasp diffusion and risk control.
2. success_vs_views.png: Fixed-budget success rate vs camera observation budget.
3. bound_validation.png: Empirical test loss distributions across 200 re-splits vs alpha.
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_pipeline_diagram(output_path: str = "results/figures/pipeline_diagram.png"):
    """Generate pipeline schematic diagram."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.axis("off")

    # Define flowchart boxes
    boxes = [
        ("1. Multi-View Simulation\nGolden-Spiral Hemisphere\n128x128 RGB-D (PyBullet)", (0.05, 0.55), "#E8F0FE", "#1A73E8"),
        ("2. Multi-View Fusion\nDrop plane pixels\nSubsample M=2048 (xyz+rgb)", (0.28, 0.55), "#E6F4EA", "#137333"),
        ("3. SE(3) Diffusion Generator\n12-Step DDIM on (R, t, w)\nLocal + Global PointNet", (0.51, 0.55), "#FEF7E0", "#B06000"),
        ("4. Calibrated Scorer\nSuccess probability s*\nTemperature Scaling (ECE/AUROC)", (0.74, 0.55), "#FCE8E6", "#C5221F"),
    ]

    for title, (x, y), facecolor, edgecolor in boxes:
        ax.text(
            x, y, title,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.8", facecolor=facecolor, edgecolor=edgecolor, lw=2),
            fontsize=10, weight="bold", color="#202124"
        )

    # Arrows between top stages
    ax.annotate("", xy=(0.17, 0.55), xytext=(0.16, 0.55), arrowprops=dict(arrowstyle="->", lw=2.5, color="#5F6368"))
    ax.annotate("", xy=(0.40, 0.55), xytext=(0.39, 0.55), arrowprops=dict(arrowstyle="->", lw=2.5, color="#5F6368"))
    ax.annotate("", xy=(0.63, 0.55), xytext=(0.62, 0.55), arrowprops=dict(arrowstyle="->", lw=2.5, color="#5F6368"))

    # Bottom Decision Policy Block
    decision_box = (
        "5. Conformal Risk-Controlled Policy (Execute / Re-View / Abstain)\n"
        r"At each view $t \in [1, T]$: predict risk $\rho$. If $\rho \le \hat{\lambda}(\alpha)$, execute grasp." + "\n"
        r"If $\rho > \hat{\lambda}(\alpha)$, capture next viewpoint. If $t = T$ and $\rho > \hat{\lambda}$, abstain."
    )
    ax.text(
        0.50, 0.20, decision_box,
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=1.0", facecolor="#F3E8FD", edgecolor="#9334E6", lw=2.5),
        fontsize=10, weight="bold", color="#1F1F1F"
    )

    ax.annotate("", xy=(0.50, 0.32), xytext=(0.74, 0.45), arrowprops=dict(arrowstyle="->", lw=2.5, color="#9334E6"))

    plt.title("Vision-Based Diffusion Grasping with Conformal Risk-Controlled Policy", fontsize=13, weight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved pipeline diagram to {output_path}")


def plot_success_vs_views(
    success_rates: List[float],
    std_errors: List[float],
    output_path: str = "results/figures/success_vs_views.png",
):
    """Plot fixed-budget success rate vs camera observation count."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    views = np.arange(1, len(success_rates) + 1)

    plt.figure(figsize=(7, 5), dpi=300)
    rates_pct = np.array(success_rates) * 100
    errs_pct = np.array(std_errors) * 100

    plt.plot(views, rates_pct, marker="o", lw=2.5, markersize=8, color="#1A73E8", label="Diffusion Grasp Success")
    plt.fill_between(views, rates_pct - errs_pct, rates_pct + errs_pct, color="#1A73E8", alpha=0.2)

    plt.xlabel("Observation Budget (Views $k$)", fontsize=11, weight="bold")
    plt.ylabel("Success Rate (%)", fontsize=11, weight="bold")
    plt.title("Grasp Success vs. Camera View Budget", fontsize=12, weight="bold")
    plt.xticks(views)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved success vs views plot to {output_path}")


def plot_bound_validation(
    resplit_losses: Dict[str, np.ndarray],
    alpha: float = 0.10,
    output_path: str = "results/figures/bound_validation.png",
):
    """Plot empirical test loss distribution across 200 random re-splits."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.figure(figsize=(9, 5), dpi=300)

    model_names = list(resplit_losses.keys())
    data = [resplit_losses[m] for m in model_names]

    parts = plt.violinplot(data, showmeans=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#1A73E8")
        pc.set_edgecolor("#1558D6")
        pc.set_alpha(0.6)

    plt.axhline(alpha, color="#D93025", linestyle="--", lw=2.5, label=rf"Target Risk Bound $\alpha = {alpha}$")

    plt.xticks(np.arange(1, len(model_names) + 1), model_names, fontsize=10, weight="bold")
    plt.ylabel("Empirical Test Loss", fontsize=11, weight="bold")
    plt.title(rf"Conformal Bound Validation (200 Re-splits, $\alpha = {alpha}$)", fontsize=12, weight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved bound validation plot to {output_path}")
