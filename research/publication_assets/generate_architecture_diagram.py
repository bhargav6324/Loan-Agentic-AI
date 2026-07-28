"""
Generate a publication-quality system architecture diagram for Springer LNCS.

Title: Overall Architecture of the Proposed Fairness-Aware
       Multi-Agent Loan Approval Framework

Style: Clean academic, white background, black outlines, subtle blue accents,
       rounded rectangles, diamonds for decisions, no gradients/shadows/3D.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.path as mpath
import numpy as np
import os

# ── Global Style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "text.color": "#1a1a1a",
    "axes.edgecolor": "#1a1a1a",
})

# Color palette — academic minimal
BLACK = "#1a1a1a"
BLUE_ACCENT = "#336699"
BLUE_LIGHT = "#e8f0fa"
WHITE = "#ffffff"
GRAY_LIGHT = "#f5f5f5"
GRAY_BORDER = "#999999"
DECISION_FILL = "#f0f4fa"

# Layout constants
FIG_W, FIG_H = 7.0, 10.5        # Springer single-column ~= 12.2 cm ≈ 4.8 in; we use 7 for clarity
BOX_W, BOX_H = 1.8, 0.48
SMALL_BOX_W, SMALL_BOX_H = 1.6, 0.40
DIA_SIZE = 0.55                  # diamond half-diagonal


def draw_rounded_box(ax, cx, cy, w, h, label, sublabel=None,
                     fill=WHITE, edge=BLACK, lw=1.2, fontsize=9,
                     bold=True, sublabel_fontsize=7.5):
    """Draw a rounded rectangle centred at (cx, cy)."""
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.06",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=3,
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(cx, cy + 0.06, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold" if bold else "normal",
                color=BLACK, zorder=4)
        ax.text(cx, cy - 0.10, sublabel, ha="center", va="center",
                fontsize=sublabel_fontsize, fontstyle="italic",
                color=GRAY_BORDER, zorder=4)
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold" if bold else "normal",
                color=BLACK, zorder=4)


def draw_diamond(ax, cx, cy, size, label, fill=DECISION_FILL, edge=BLACK,
                 lw=1.2, fontsize=8):
    """Draw a diamond (rotated square) centred at (cx, cy)."""
    verts = [
        (cx, cy + size),       # top
        (cx + size, cy),       # right
        (cx, cy - size),       # bottom
        (cx - size, cy),       # left
        (cx, cy + size),       # close
    ]
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * 3 + [mpath.Path.CLOSEPOLY]
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor=fill, edgecolor=edge,
                               linewidth=lw, zorder=3)
    ax.add_patch(patch)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=BLACK, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, color=BLACK, lw=1.0,
               head_width=0.06, head_length=0.04):
    """Draw a straight directional arrow."""
    dx, dy = x2 - x1, y2 - y1
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=12),
                zorder=2)


def draw_label_on_arrow(ax, x, y, text, fontsize=7.5, color=GRAY_BORDER):
    """Place a small label next to an arrow."""
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontstyle="italic", color=color, zorder=5,
            bbox=dict(boxstyle="round,pad=0.12", facecolor=WHITE,
                      edgecolor="none", alpha=0.95))


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    # ── Column centres ────────────────────────────────────────────────
    mid = FIG_W / 2            # 3.5
    left_col = mid - 2.1       # 1.4   — Risk Agent
    center_col = mid           # 3.5   — Policy Agent
    right_col = mid + 2.1      # 5.6   — Fairness Agent

    # ── Vertical positions (top to bottom) ────────────────────────────
    y_applicant   = 9.7
    y_loan_app    = 8.9
    y_planner     = 8.0
    y_parallel    = 6.8        # parallel agent row
    y_combine     = 5.7        # combine box
    y_decision    = 4.7        # diamond
    y_gov         = 3.4        # governance agent (right branch)
    y_report_no   = 3.4        # reporting agent (left branch, "No" path)
    y_report_yes  = 2.2        # reporting agent (right branch, after governance)
    y_final_no    = 2.4        # final report left
    y_final_yes   = 1.2        # final report right

    # Since both paths end at Reporting → Final, let's unify:
    # NO path:  Decision → Reporting Agent → Final Decision Report
    # YES path: Decision → Governance Agent → Reporting Agent → Final Decision Report
    # We'll place them side-by-side for clarity.

    left_path  = mid - 1.3     # "No" path centre
    right_path = mid + 1.3     # "Yes" path centre

    # ══════════════════════════════════════════════════════════════════
    #  ROW 1 — Applicant
    # ══════════════════════════════════════════════════════════════════
    draw_rounded_box(ax, mid, y_applicant, 1.6, 0.42, "Applicant",
                     fill=GRAY_LIGHT, edge=GRAY_BORDER)

    # Arrow: Applicant → Loan Application
    draw_arrow(ax, mid, y_applicant - 0.21, mid, y_loan_app + 0.22)

    # ══════════════════════════════════════════════════════════════════
    #  ROW 2 — Loan Application
    # ══════════════════════════════════════════════════════════════════
    draw_rounded_box(ax, mid, y_loan_app, 1.8, 0.42, "Loan Application",
                     fill=WHITE, edge=BLACK)

    # Arrow: Loan Application → Planner Agent
    draw_arrow(ax, mid, y_loan_app - 0.21, mid, y_planner + 0.26)

    # ══════════════════════════════════════════════════════════════════
    #  ROW 3 — Planner Agent
    # ══════════════════════════════════════════════════════════════════
    draw_rounded_box(ax, mid, y_planner, BOX_W, BOX_H, "Planner Agent",
                     sublabel="Coordinates workflow",
                     fill=BLUE_LIGHT, edge=BLUE_ACCENT, lw=1.4)

    # ── Fan-out arrows: Planner → three parallel agents ───────────────
    planner_bottom = y_planner - BOX_H / 2
    agent_top = y_parallel + BOX_H / 2

    draw_arrow(ax, mid, planner_bottom, left_col, agent_top)
    draw_arrow(ax, mid, planner_bottom, center_col, agent_top)
    draw_arrow(ax, mid, planner_bottom, right_col, agent_top)

    # ══════════════════════════════════════════════════════════════════
    #  ROW 4 — Parallel agents
    # ══════════════════════════════════════════════════════════════════
    draw_rounded_box(ax, left_col, y_parallel, BOX_W, BOX_H,
                     "Risk Agent",
                     sublabel="Predicts credit risk",
                     fill=WHITE, edge=BLACK)

    draw_rounded_box(ax, center_col, y_parallel, BOX_W, BOX_H,
                     "Policy Agent",
                     sublabel="Validates institutional rules",
                     fill=WHITE, edge=BLACK)

    draw_rounded_box(ax, right_col, y_parallel, BOX_W, BOX_H,
                     "Fairness Agent",
                     sublabel="Counterfactual fairness analysis",
                     fill=WHITE, edge=BLACK)

    # ── Dashed "parallel" bracket ─────────────────────────────────────
    brace_y_top = y_parallel + BOX_H / 2 + 0.12
    brace_y_bot = y_parallel - BOX_H / 2 - 0.12
    brace_x_left = left_col - BOX_W / 2 - 0.12
    brace_x_right = right_col + BOX_W / 2 + 0.12

    rect = mpatches.FancyBboxPatch(
        (brace_x_left, brace_y_bot),
        brace_x_right - brace_x_left,
        brace_y_top - brace_y_bot,
        boxstyle="round,pad=0.08",
        facecolor="none", edgecolor=GRAY_BORDER,
        linewidth=0.8, linestyle="--", zorder=1,
    )
    ax.add_patch(rect)
    ax.text(brace_x_right - 0.08, brace_y_top - 0.04,
            "parallel execution", ha="right", va="top",
            fontsize=7, fontstyle="italic", color=GRAY_BORDER, zorder=2)

    # ── Fan-in arrows: agents → Combine ───────────────────────────────
    agent_bottom = y_parallel - BOX_H / 2
    combine_top = y_combine + BOX_H / 2

    draw_arrow(ax, left_col, agent_bottom, mid, combine_top)
    draw_arrow(ax, center_col, agent_bottom, mid, combine_top)
    draw_arrow(ax, right_col, agent_bottom, mid, combine_top)

    # ══════════════════════════════════════════════════════════════════
    #  ROW 5 — Combine Outputs
    # ══════════════════════════════════════════════════════════════════
    draw_rounded_box(ax, mid, y_combine, 2.0, 0.42, "Combine Agent Outputs",
                     fill=GRAY_LIGHT, edge=GRAY_BORDER, bold=False, fontsize=8.5)

    # Arrow: Combine → Decision diamond
    draw_arrow(ax, mid, y_combine - 0.21, mid, y_decision + DIA_SIZE + 0.02)

    # ══════════════════════════════════════════════════════════════════
    #  ROW 6 — Decision diamond
    # ══════════════════════════════════════════════════════════════════
    draw_diamond(ax, mid, y_decision, DIA_SIZE, "Sensitive\nApplication?",
                 fontsize=7.5)

    # ── Branch: NO (left) ─────────────────────────────────────────────
    dia_left = mid - DIA_SIZE
    draw_arrow(ax, dia_left, y_decision, left_path, y_decision)
    draw_arrow(ax, left_path, y_decision, left_path, y_report_no + 0.22)
    draw_label_on_arrow(ax, dia_left - 0.32, y_decision + 0.14, "No")

    # ── Branch: YES (right) ───────────────────────────────────────────
    dia_right = mid + DIA_SIZE
    draw_arrow(ax, dia_right, y_decision, right_path, y_decision)
    draw_arrow(ax, right_path, y_decision, right_path, y_gov + 0.24)
    draw_label_on_arrow(ax, dia_right + 0.32, y_decision + 0.14, "Yes")

    # ══════════════════════════════════════════════════════════════════
    #  ROW 7a — Reporting Agent (NO path)
    # ══════════════════════════════════════════════════════════════════
    draw_rounded_box(ax, left_path, y_report_no, SMALL_BOX_W, SMALL_BOX_H,
                     "Reporting Agent",
                     fill=WHITE, edge=BLACK, fontsize=8.5)

    # Arrow: Reporting Agent → Final Decision Report (NO path)
    draw_arrow(ax, left_path, y_report_no - SMALL_BOX_H / 2,
               left_path, y_final_no + 0.22)

    draw_rounded_box(ax, left_path, y_final_no, SMALL_BOX_W, SMALL_BOX_H,
                     "Final Decision\nReport",
                     fill=GRAY_LIGHT, edge=GRAY_BORDER, fontsize=8, bold=False)

    # ══════════════════════════════════════════════════════════════════
    #  ROW 7b — Governance Agent (YES path)
    # ══════════════════════════════════════════════════════════════════
    draw_rounded_box(ax, right_path, y_gov, SMALL_BOX_W + 0.1, SMALL_BOX_H + 0.06,
                     "Governance Agent",
                     sublabel="Reviews sensitive cases",
                     fill=BLUE_LIGHT, edge=BLUE_ACCENT, lw=1.4, fontsize=8.5,
                     sublabel_fontsize=7)

    # Arrow: Governance → Reporting Agent (YES path)
    draw_arrow(ax, right_path, y_gov - (SMALL_BOX_H + 0.06) / 2,
               right_path, y_report_yes + SMALL_BOX_H / 2 + 0.02)

    draw_rounded_box(ax, right_path, y_report_yes, SMALL_BOX_W, SMALL_BOX_H,
                     "Reporting Agent",
                     fill=WHITE, edge=BLACK, fontsize=8.5)

    # Arrow: Reporting Agent → Final Decision Report (YES path)
    draw_arrow(ax, right_path, y_report_yes - SMALL_BOX_H / 2,
               right_path, y_final_yes + 0.22)

    draw_rounded_box(ax, right_path, y_final_yes, SMALL_BOX_W, SMALL_BOX_H,
                     "Final Decision\nReport",
                     fill=GRAY_LIGHT, edge=GRAY_BORDER, fontsize=8, bold=False)

    # ══════════════════════════════════════════════════════════════════
    #  Caption
    # ══════════════════════════════════════════════════════════════════
    ax.text(mid, 0.45,
            "Fig. 1. Overall architecture of the proposed fairness-aware\n"
            "multi-agent loan approval framework.",
            ha="center", va="center", fontsize=8.5, fontstyle="italic",
            color=BLACK)

    # ── Save ──────────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)

    svg_path = os.path.join(out_dir, "fig_system_architecture.svg")
    png_path = os.path.join(out_dir, "fig_system_architecture.png")
    pdf_path = os.path.join(out_dir, "fig_system_architecture.pdf")

    plt.savefig(svg_path, format="svg", bbox_inches="tight",
                facecolor=WHITE, edgecolor="none")
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight",
                facecolor=WHITE, edgecolor="none")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight",
                facecolor=WHITE, edgecolor="none")
    plt.close()

    print(f"✅  SVG  → {svg_path}")
    print(f"✅  PNG  → {png_path}")
    print(f"✅  PDF  → {pdf_path}")


if __name__ == "__main__":
    main()
