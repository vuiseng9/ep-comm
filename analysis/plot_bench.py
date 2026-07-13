from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# -------------------------
# Config
# -------------------------
CSV_PATH = Path("ep_benchmark_olmoe_qwen3.csv")
OUT_PNG = Path("training_step_time_by_ep_backend.png")
OUT_SVG = Path("training_step_time_by_ep_backend.svg")

# Hide local from the plot.
EP_ORDER = ["host_nccl", "naive_symm", "pooled_symm", "zerocopy_symm"]

MODEL_ORDER = ["olmoe", "qwen3"]
MODEL_TITLES = {
    "olmoe": "OLMoE-1B-7B",
    "qwen3": "Qwen3-30B-A3B",
}

COLORS = {
    "forward": "#1A53FF",
    "backward": "#18DD69",
    "optimize": "#091B47",
    "other": "#D0D5DD",
}

LABELS = {
    "forward": "Forward",
    "backward": "Backward",
    "optimize": "Optimize",
    "other": "Other / residual",
}

XLIM_MAX = 220

FIGSIZE = (16, 10)
DPI = 360

# Font sizes
TITLE_MAIN_FS = 22
TITLE_SUB_FS = 18
PANEL_TITLE_FS = 14
AXIS_LABEL_FS = 13
TICK_LABEL_FS = 12.5
ANNOTATION_FS = 12.5
LEGEND_FS = 13

# Styling
BAR_HEIGHT = 0.70
BAR_EDGE_LW = 1.25
GRID_LW = 0.65
SPINE_LW = 1.1


# -------------------------
# Load data
# -------------------------
df = pd.read_csv(CSV_PATH)

# Residual so stacked bars add up to measured step_time.
df["other"] = df["step_time"] - df[["forward", "backward", "optimize"]].sum(axis=1)
df["other"] = df["other"].clip(lower=0)

y = np.arange(len(EP_ORDER))


# -------------------------
# Helpers
# -------------------------
def style_axis(ax, title, show_ylabel=False):
    ax.set_title(title, fontsize=PANEL_TITLE_FS, fontweight="bold", pad=9)
    ax.set_xlabel("elapsed (ms)", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("EP backend" if show_ylabel else "", fontsize=AXIS_LABEL_FS)

    ax.set_xlim(0, XLIM_MAX)
    ax.set_yticks(y)
    ax.set_yticklabels(EP_ORDER, fontsize=TICK_LABEL_FS)
    ax.set_ylim(len(EP_ORDER) - 0.5, -0.5)

    ax.tick_params(axis="x", labelsize=TICK_LABEL_FS)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FS)

    ax.grid(axis="x", linestyle="-", linewidth=GRID_LW, alpha=0.24)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(SPINE_LW)
    ax.spines["bottom"].set_linewidth(SPINE_LW)


def annotate_bar(ax, value, y_pos, ep_name):
    x_text = min(value + 5, XLIM_MAX - 8)
    ha = "left" if value + 5 <= XLIM_MAX - 8 else "right"

    ax.text(
        x_text,
        y_pos,
        f"{value:.1f}",
        va="center",
        ha=ha,
        fontsize=ANNOTATION_FS,
        fontweight="bold" if ep_name == "zerocopy_symm" else "normal",
        color="#111111",
    )


# -------------------------
# Plot
# -------------------------
fig, axes = plt.subplots(3, 2, figsize=FIGSIZE, facecolor="white")

for col_idx, model in enumerate(MODEL_ORDER):
    sub = (
        df[df["model"] == model]
        .set_index("ep")
        .loc[EP_ORDER]
        .reset_index()
    )

    # Row 1: stacked step time breakdown
    ax = axes[0, col_idx]
    left = np.zeros(len(sub))

    for comp in ["forward", "backward", "optimize", "other"]:
        ax.barh(
            y,
            sub[comp],
            left=left,
            color=COLORS[comp],
            edgecolor="white",
            linewidth=BAR_EDGE_LW,
            height=BAR_HEIGHT,
        )
        left += sub[comp].to_numpy()

    for i, row in sub.iterrows():
        annotate_bar(ax, row["step_time"], i, row["ep"])

    style_axis(ax, MODEL_TITLES[model], show_ylabel=(col_idx == 0))

    # Row 2: forward
    ax = axes[1, col_idx]
    ax.barh(
        y,
        sub["forward"],
        color=COLORS["forward"],
        edgecolor="white",
        linewidth=BAR_EDGE_LW,
        height=BAR_HEIGHT,
    )

    for i, row in sub.iterrows():
        annotate_bar(ax, row["forward"], i, row["ep"])

    style_axis(ax, "forward", show_ylabel=(col_idx == 0))

    # Row 3: backward
    ax = axes[2, col_idx]
    ax.barh(
        y,
        sub["backward"],
        color=COLORS["backward"],
        edgecolor="white",
        linewidth=BAR_EDGE_LW,
        height=BAR_HEIGHT,
    )

    for i, row in sub.iterrows():
        annotate_bar(ax, row["backward"], i, row["ep"])

    style_axis(ax, "backward", show_ylabel=(col_idx == 0))


# Main title: two lines, first bold, second regular.
fig.text(
    0.5,
    0.993,
    "Training Step Time",
    ha="center",
    va="top",
    fontsize=TITLE_MAIN_FS,
    fontweight="bold",
)

fig.text(
    0.5,
    0.960,
    "w.r.t. EP Backend on 8xH100_sxm5, EP=8, 1-layer MoE",
    ha="center",
    va="top",
    fontsize=TITLE_SUB_FS,
    fontweight="normal",
)


# Legend
legend_handles = [
    Patch(facecolor=COLORS["forward"], label=LABELS["forward"]),
    Patch(facecolor=COLORS["backward"], label=LABELS["backward"]),
    Patch(facecolor=COLORS["optimize"], label=LABELS["optimize"]),
    Patch(facecolor=COLORS["other"], label=LABELS["other"]),
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=4,
    frameon=False,
    fontsize=LEGEND_FS,
    bbox_to_anchor=(0.5, 0.006),
    handlelength=1.8,
    columnspacing=2.0,
)


# Layout and save
fig.tight_layout(rect=[0, 0.06, 1, 0.918], h_pad=2.0, w_pad=2.2)

fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_SVG, bbox_inches="tight", facecolor="white")

plt.show()