"""
Generate a Polished Academic Architecture Diagram for Springer LNCS / IEEE
────────────────────────────────────────────────────────────────────────────
Title: Overall Architecture of the Proposed Fairness-Aware Multi-Agent Loan Approval Framework

Style:
  - High-end academic architecture figure (hierarchical layering, modular grouping)
  - Clean white background, thin black outlines, subtle blue accents (#2b5c8f)
  - Rounded rectangles for modules and agents
  - No duplicated components, no flowchart look, clean information flow
  - Compact vertical layout fitting single-column Springer width (~60-70% page height)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Global Styling Parameters ─────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9.5,
    "text.color": "#1a1a1a",
    "axes.edgecolor": "#1a1a1a",
})

BLACK = "#1a1a1a"
GRAY_TEXT = "#555555"
GRAY_LIGHT = "#f8f9fa"
GRAY_BORDER = "#a0a0a0"
BLUE_ACCENT = "#2b5c8f"
BLUE_LIGHT = "#ebf3fa"
WHITE = "#ffffff"


def draw_box(ax, cx, cy, w, h, title, subtitle=None, fill=WHITE, edge=BLACK, lw=1.2,
             title_fontsize=9.5, subtitle_fontsize=7.8, bold_title=True, style="round,pad=0.06"):
    """Draw a rounded module box with optional subtitle."""
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=style,
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=3,
    )
    ax.add_patch(box)
    if subtitle:
        ax.text(cx, cy + h * 0.18, title, ha="center", va="center",
                fontsize=title_fontsize, fontweight="bold" if bold_title else "normal",
                color=BLACK, zorder=4)
        ax.text(cx, cy - h * 0.20, subtitle, ha="center", va="center",
                fontsize=subtitle_fontsize, fontstyle="italic",
                color=GRAY_TEXT, zorder=4, linespacing=1.2)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=title_fontsize, fontweight="bold" if bold_title else "normal",
                color=BLACK, zorder=4)
    return box


def draw_group_container(ax, x_min, y_min, x_max, y_max, label="", fill="none", edge=GRAY_BORDER,
                         lw=1.0, linestyle="--"):
    """Draw a bounding container for modular grouping (e.g. parallel suite or layer)."""
    box = FancyBboxPatch(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        boxstyle="round,pad=0.1",
        facecolor=fill, edgecolor=edge, linewidth=lw, linestyle=linestyle, zorder=1,
    )
    ax.add_patch(box)
    if label:
        ax.text((x_min + x_max) / 2, y_max, label,
                ha="center", va="center", fontsize=8.8, fontweight="bold",
                color=GRAY_TEXT, zorder=6, bbox=dict(boxstyle="round,pad=0.25", facecolor=WHITE, edgecolor="none", alpha=1.0))
    return box


def draw_arrow(ax, x1, y1, x2, y2, color=BLACK, lw=1.1, style="-|>", rad=0.0):
    """Draw a directional arrow between components."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                mutation_scale=13, connectionstyle=f"arc3,rad={rad}"),
                zorder=2)


def draw_edge_label(ax, x, y, text, fontsize=7.2, color=BLACK, bg_alpha=0.95):
    """Draw a clean background-padded label on an arrow or pathway."""
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontstyle="italic", color=color, zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=WHITE, edgecolor="none", alpha=bg_alpha))


def main():
    # Compact vertical dimensions ideal for Springer single column
    fig_w, fig_h = 8.0, 9.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    mid = fig_w / 2.0         # 4.0
    left_col = mid - 2.3      # 1.7
    right_col = mid + 2.3     # 6.3

    # ══════════════════════════════════════════════════════════════════
    #  LAYER 1: INPUT & PREPROCESSING (Top)
    # ══════════════════════════════════════════════════════════════════
    y_input = 9.2
    y_intake = 8.15

    draw_box(ax, mid, y_input, 4.4, 0.46, "Applicant Feature Vector & Loan Submission",
             subtitle="Demographic attributes, credit history, debt-to-income ratio",
             fill=GRAY_LIGHT, edge=GRAY_BORDER, bold_title=False, title_fontsize=9.2)
    
    draw_arrow(ax, mid, y_input - 0.23, mid, y_intake + 0.26)
    
    draw_box(ax, mid, y_intake, 4.8, 0.52, "Intake & Data Verification Agent",
             subtitle="KYC validation, fraud screening, and integrity checks",
             fill=WHITE, edge=BLACK, lw=1.3)

    # ══════════════════════════════════════════════════════════════════
    #  LAYER 2: MULTI-AGENT ORCHESTRATION & PARALLEL EVALUATION
    # ══════════════════════════════════════════════════════════════════
    y_planner = 6.9
    y_parallel = 5.25

    draw_arrow(ax, mid, y_intake - 0.26, mid, y_planner + 0.26)
    draw_edge_label(ax, mid, (y_intake + y_planner) / 2, "Verified Application Data")

    draw_box(ax, mid, y_planner, 4.4, 0.52, "Planner Agent (Supervisory Orchestrator)",
             subtitle="Dynamic DAG execution planning and task synchronization",
             fill=BLUE_LIGHT, edge=BLUE_ACCENT, lw=1.5)

    # Group container for parallel suite (with extra breathing room at top and bottom)
    draw_group_container(ax, 0.45, 4.25, 7.55, 6.2, label="Concurrent Multi-Agent Evaluation Suite",
                         fill="#fcfdfe", edge=GRAY_BORDER, lw=1.0, linestyle="--")

    # Three parallel agents (with multi-line subtitles to avoid overflow)
    draw_box(ax, left_col, y_parallel, 2.1, 0.95, "Risk Agent",
             subtitle="Random Forest ML &\nheuristic risk tiering",
             fill=WHITE, edge=BLACK, lw=1.2, subtitle_fontsize=7.6)
    
    draw_box(ax, mid, y_parallel, 2.1, 0.95, "Policy Agent",
             subtitle="Institutional rules &\nregulatory verification",
             fill=WHITE, edge=BLACK, lw=1.2, subtitle_fontsize=7.6)
    
    draw_box(ax, right_col, y_parallel, 2.1, 0.95, "Fairness Agent",
             subtitle="Counterfactual inversion &\nsensitivity audit (|ΔCF|)",
             fill=WHITE, edge=BLACK, lw=1.2, subtitle_fontsize=7.6)

    # Fan-out arrows from Planner to parallel suite
    planner_bot = y_planner - 0.26
    agent_top = y_parallel + 0.475
    draw_arrow(ax, mid, planner_bot, left_col, agent_top)
    draw_arrow(ax, mid, planner_bot, mid, agent_top)
    draw_arrow(ax, mid, planner_bot, right_col, agent_top)

    # ══════════════════════════════════════════════════════════════════
    #  LAYER 3: GOVERNANCE & DECISION RESOLUTION
    # ══════════════════════════════════════════════════════════════════
    y_sync = 3.6
    y_decision_engine = 2.35

    # Fan-in arrows from parallel suite to State Synchronization
    agent_bot = y_parallel - 0.475
    sync_top = y_sync + 0.26
    draw_arrow(ax, left_col, agent_bot, mid, sync_top)
    draw_arrow(ax, mid, agent_bot, mid, sync_top)
    draw_arrow(ax, right_col, agent_bot, mid, sync_top)

    draw_box(ax, mid, y_sync, 4.8, 0.52, "State Synchronization & Policy Join Module",
             subtitle="Aggregates risk scores, policy compliance, and bias flags",
             fill=WHITE, edge=BLACK, lw=1.3)

    draw_arrow(ax, mid, y_sync - 0.26, mid, y_decision_engine + 0.30)

    # Core Decision Engine Box (left-centered at 2.8, width 3.8 -> span 0.9 to 4.7)
    x_dec = 2.85
    draw_box(ax, x_dec, y_decision_engine, 3.8, 0.60, "Supervisory Decision Engine",
             subtitle="Evaluates composite risk against\ninstitutional underwriting thresholds",
             fill=WHITE, edge=BLACK, lw=1.3, subtitle_fontsize=7.6)

    # Governance & Human Oversight Module (right-centered at 6.1, width 2.2 -> span 5.0 to 7.2)
    x_gov = 6.15
    draw_box(ax, x_gov, y_decision_engine, 2.2, 0.60, "Governance Agent",
             subtitle="Human-in-the-loop review\n& semantic LLM debate",
             fill=BLUE_LIGHT, edge=BLUE_ACCENT, lw=1.4, subtitle_fontsize=7.6)

    # Bidirectional / Routing interaction between Decision Engine and Governance Module (0.3 gap)
    dec_right = x_dec + 1.9
    gov_left = x_gov - 1.1
    draw_arrow(ax, dec_right, y_decision_engine + 0.14, gov_left, y_decision_engine + 0.14, color=BLUE_ACCENT, lw=1.2)
    draw_arrow(ax, gov_left, y_decision_engine - 0.14, dec_right, y_decision_engine - 0.14, color=BLUE_ACCENT, lw=1.2)
    
    draw_edge_label(ax, (dec_right + gov_left) / 2, y_decision_engine + 0.28, "Sensitive (|ΔCF| > ε)", fontsize=7.1, color=BLUE_ACCENT)
    draw_edge_label(ax, (dec_right + gov_left) / 2, y_decision_engine - 0.28, "Audited Verdict", fontsize=7.1, color=BLUE_ACCENT)

    # ══════════════════════════════════════════════════════════════════
    #  LAYER 4: EXPLAINABILITY & AUDIT REPORTING (Bottom)
    # ══════════════════════════════════════════════════════════════════
    y_report = 1.15
    y_output = 0.25

    draw_arrow(ax, x_dec, y_decision_engine - 0.30, mid, y_report + 0.30)
    draw_edge_label(ax, (x_dec + mid) / 2, (y_decision_engine - 0.30 + y_report + 0.30) / 2, "Resolved Underwriting State", fontsize=7.5)

    draw_box(ax, mid, y_report, 5.2, 0.60, "Reporting & Explainability Agent",
             subtitle="Generates XAI decision justifications, counterfactual traces,\nand regulatory compliance audit logs",
             fill=WHITE, edge=BLACK, lw=1.3, subtitle_fontsize=7.6)

    draw_arrow(ax, mid, y_report - 0.30, mid, y_output + 0.23)

    draw_box(ax, mid, y_output, 4.6, 0.46, "Auditable Underwriting Decision & Explanation Report",
             subtitle="Transparent, regulatory-compliant financial recommendation",
             fill=GRAY_LIGHT, edge=GRAY_BORDER, bold_title=False, title_fontsize=9.2)

    # ── Figure Caption per Springer LNCS formatting ───────────────────
    ax.text(mid, -0.35,
            "Fig. 1. Overall architecture of the proposed fairness-aware multi-agent loan approval framework.",
            ha="center", va="top", fontsize=8.8, fontstyle="italic", color=BLACK)

    # ── Save Outputs ──────────────────────────────────────────────────
    out_dir = os.path.dirname(__file__)
    os.makedirs(out_dir, exist_ok=True)

    svg_path = os.path.join(out_dir, "fig_system_architecture_springer.svg")
    pdf_path = os.path.join(out_dir, "fig_system_architecture_springer.pdf")
    png_path = os.path.join(out_dir, "fig_system_architecture_springer.png")

    # Save with generous bottom bbox to include the caption
    plt.savefig(svg_path, format="svg", bbox_inches="tight", facecolor=WHITE, edgecolor="none")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=WHITE, edgecolor="none")
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight", facecolor=WHITE, edgecolor="none")
    plt.close()

    print("======================================================================")
    print("GENERATED SPRINGER ACADEMIC ARCHITECTURE DIAGRAM")
    print("======================================================================")
    print(f"✅  SVG  → {svg_path}")
    print(f"✅  PDF  → {pdf_path}")
    print(f"✅  PNG  → {png_path}")


if __name__ == "__main__":
    main()
