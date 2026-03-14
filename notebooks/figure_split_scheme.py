"""
Publication-quality figure: PLATE-VS two-axis dataset partitioning strategy.
Panel A: Conceptual 2D schematic (protein cluster × ligand distance)
Panel B: Stacked bar chart of train/test sizes at each distance threshold
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import os

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = "../data/output/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,   # editable text in Illustrator
    "ps.fonttype": 42,
})

# ── Colour palette (Okabe–Ito, colorblind-safe) ───────────────────────────────
C_TRAIN   = "#0072B2"   # blue  – training data
C_CHEM    = "#E69F00"   # amber – chemical generalisation test
C_PROT    = "#009E73"   # green – protein generalisation test
C_GRAY    = "#BBBBBB"   # light grey – "not used" region
C_OUTLINE = "#333333"

# ─────────────────────────────────────────────────────────────────────────────
# Data for Panel B
# ─────────────────────────────────────────────────────────────────────────────
thresholds  = [0.0,   0.1,   0.3,    0.5,    0.7,    0.9]
train_counts = [4168,  4741, 21316,  62549, 126905, 270363]
test_counts  = [275817, 275244, 258669, 217436, 153080,  9622]

total = [t + c for t, c in zip(train_counts, test_counts)]   # always 279,985

thresh_labels = [str(t) for t in thresholds]

# ─────────────────────────────────────────────────────────────────────────────
# Figure layout  (double-column width ≈ 183 mm / 7.2 in; height 3.2 in)
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.2, 3.4))

# Left panel A  – schematic  (occupies left ~42 % of figure)
# Right panel B – bar chart  (occupies right ~55 % of figure)
ax_a = fig.add_axes([0.03, 0.10, 0.40, 0.82])   # [left, bottom, w, h]
ax_b = fig.add_axes([0.52, 0.14, 0.46, 0.74])

# ═════════════════════════════════════════════════════════════════════════════
# PANEL A  –  conceptual schematic
# ═════════════════════════════════════════════════════════════════════════════
ax_a.set_xlim(0, 1)
ax_a.set_ylim(0, 1)
ax_a.set_aspect("equal")
ax_a.axis("off")

# ── grid lines ────────────────────────────────────────────────────────────────
ax_a.axvline(0.50, color=C_OUTLINE, lw=1.2, ls="--", zorder=3)
ax_a.axhline(0.50, color=C_OUTLINE, lw=1.2, ls="--", zorder=3)

# ── quadrant fills ────────────────────────────────────────────────────────────
# Bottom-left:  Protein-train  ×  Chemical-train  → MODEL TRAINING
ax_a.add_patch(FancyBboxPatch((0.02, 0.02), 0.46, 0.46,
    boxstyle="round,pad=0.01", facecolor=C_TRAIN, alpha=0.22, zorder=1))

# Top-left:     Protein-train  ×  Chemical-test   → CHEMICAL GENERALISATION
ax_a.add_patch(FancyBboxPatch((0.02, 0.52), 0.46, 0.46,
    boxstyle="round,pad=0.01", facecolor=C_CHEM, alpha=0.22, zorder=1))

# Right half:   Protein-test   ×  All chemicals   → PROTEIN GENERALISATION
ax_a.add_patch(FancyBboxPatch((0.52, 0.02), 0.46, 0.96,
    boxstyle="round,pad=0.01", facecolor=C_PROT, alpha=0.18, zorder=1))

# ── quadrant labels ───────────────────────────────────────────────────────────
kw = dict(ha="center", va="center", fontsize=7.5, zorder=4)

ax_a.text(0.25, 0.25, "Model\nTraining", color=C_TRAIN,
          fontweight="bold", **kw)

ax_a.text(0.25, 0.75, "Chemical\nGeneralisation\nTest",
          color="#B07800", fontweight="bold", **kw)

ax_a.text(0.75, 0.68, "Protein\nGeneralisation\nTest",
          color="#006050", fontweight="bold", **kw)

# ── count annotations (small) ─────────────────────────────────────────────────
ax_a.text(0.25, 0.13, "126,905 actives\n(at X = 0.7)",
          ha="center", va="center", fontsize=6.2, color="#444444", zorder=4)
ax_a.text(0.25, 0.87, "153,080 actives\n(at X = 0.7)",
          ha="center", va="center", fontsize=6.2, color="#444444", zorder=4)
ax_a.text(0.75, 0.28, "~10 % of\nprotein clusters\n(all actives)",
          ha="center", va="center", fontsize=6.2, color="#444444", zorder=4)

# ── axis arrows ───────────────────────────────────────────────────────────────
arrow_kw = dict(arrowstyle="-|>", color=C_OUTLINE, lw=1.0,
                mutation_scale=10, zorder=5)

# X-axis (protein clusters): left → right
ax_a.annotate("", xy=(1.0, -0.06), xytext=(0.0, -0.06),
              xycoords="axes fraction", textcoords="axes fraction",
              arrowprops=arrow_kw)
ax_a.text(0.50, -0.12, "Protein clusters",
          ha="center", va="top", fontsize=8, transform=ax_a.transAxes)
ax_a.text(0.25, -0.07, "90 % Training", ha="center", va="top",
          fontsize=6.5, color=C_TRAIN, transform=ax_a.transAxes)
ax_a.text(0.75, -0.07, "10 % Test", ha="center", va="top",
          fontsize=6.5, color=C_PROT, transform=ax_a.transAxes)

# Y-axis (chemical distance): bottom → top
ax_a.annotate("", xy=(-0.08, 1.0), xytext=(-0.08, 0.0),
              xycoords="axes fraction", textcoords="axes fraction",
              arrowprops=arrow_kw)
ax_a.text(-0.14, 0.50,
          "Tanimoto distance\nto PDB ligand",
          ha="center", va="center", fontsize=8,
          rotation=90, transform=ax_a.transAxes)
ax_a.text(-0.09, 0.25, "≤ X\n(train)", ha="right", va="center",
          fontsize=6.5, color=C_TRAIN, transform=ax_a.transAxes)
ax_a.text(-0.09, 0.75, "> X\n(test)", ha="right", va="center",
          fontsize=6.5, color="#B07800", transform=ax_a.transAxes)

# ── divider tick marks ────────────────────────────────────────────────────────
ax_a.text(0.50, -0.03, "|", ha="center", va="top", fontsize=8,
          color=C_OUTLINE, transform=ax_a.transAxes)
ax_a.text(-0.085, 0.50, "—", ha="right", va="center", fontsize=8,
          color=C_OUTLINE, transform=ax_a.transAxes)

# ── panel label ───────────────────────────────────────────────────────────────
ax_a.text(-0.12, 1.04, "A", transform=ax_a.transAxes,
          fontsize=11, fontweight="bold", va="top")

# ═════════════════════════════════════════════════════════════════════════════
# PANEL B  –  stacked bar chart
# ═════════════════════════════════════════════════════════════════════════════
x = np.arange(len(thresholds))
bar_w = 0.62

# Stacked bars: train (bottom) + chemical-test (top)
bars_train = ax_b.bar(x, train_counts, bar_w,
                      color=C_TRAIN, alpha=0.85, label="Chemical train",
                      zorder=3)
bars_test  = ax_b.bar(x, test_counts, bar_w,
                      bottom=train_counts,
                      color=C_CHEM, alpha=0.85, label="Chemical test",
                      zorder=3)

# Highlight recommended threshold (0.7, index 4)
for bar in [bars_train[4], bars_test[4]]:
    bar.set_edgecolor("#D55E00")
    bar.set_linewidth(1.6)

ax_b.set_xticks(x)
ax_b.set_xticklabels(thresh_labels)
ax_b.set_xlabel("Tanimoto distance threshold  X", fontsize=9)
ax_b.set_ylabel("Number of active compounds", fontsize=9)

# Y-axis: thousands
ax_b.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
ax_b.set_ylim(0, 330000)
ax_b.set_yticks([0, 50000, 100000, 150000, 200000, 250000, 300000])

# Grid
ax_b.yaxis.grid(True, color="#DDDDDD", lw=0.6, zorder=0)
ax_b.set_axisbelow(True)

# Spines
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)

# Recommended label
ax_b.annotate("Recommended\n(X = 0.7)",
              xy=(4, 285000), xytext=(4, 315000),
              ha="center", fontsize=6.8, color="#D55E00",
              arrowprops=dict(arrowstyle="-|>", color="#D55E00",
                              lw=0.9, mutation_scale=8))

# Legend
legend_patches = [
    mpatches.Patch(facecolor=C_TRAIN,  alpha=0.85, label="Chemical train  (dist ≤ X)"),
    mpatches.Patch(facecolor=C_CHEM,   alpha=0.85, label="Chemical test    (dist > X)"),
]
ax_b.legend(handles=legend_patches, loc="upper right",
            frameon=False, fontsize=7.2)

# Value labels on bars (train bottom segment, test top segment)
for i, (tr, te) in enumerate(zip(train_counts, test_counts)):
    if tr > 12000:
        ax_b.text(i, tr / 2, f"{tr/1000:.0f}k",
                  ha="center", va="center", fontsize=5.8,
                  color="white", fontweight="bold", zorder=4)
    if te > 12000:
        ax_b.text(i, tr + te / 2, f"{te/1000:.0f}k",
                  ha="center", va="center", fontsize=5.8,
                  color="white", fontweight="bold", zorder=4)

# Panel label
ax_b.text(-0.14, 1.04, "B", transform=ax_b.transAxes,
          fontsize=11, fontweight="bold", va="top")

# ── Save ──────────────────────────────────────────────────────────────────────
for fmt in ("pdf", "png", "svg"):
    path = f"{OUT_DIR}/split_scheme.{fmt}"
    fig.savefig(path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved: {path}")

plt.close(fig)
print("Done.")
