"""
Publication Asset Generator — Architecture Diagrams (SVG & 300 DPI PNG)
────────────────────────────────────────────────────────────────────────
Generates 3 publication-quality architectural flowcharts for Springer/IEEE:
  1. Overall LangGraph Multi-Agent Orchestration Architecture
  2. End-to-End Loan Underwriting Decision Workflow
  3. Counterfactual Fairness Audit and Escalation Workflow
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11


def draw_box(ax, x, y, width, height, text, facecolor="#e8f4f8", edgecolor="#2b5c8f", fontsize=11, weight="bold", style="round,pad=0.1"):
    box = mpatches.FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle=style,
        facecolor=facecolor, edgecolor=edgecolor, linewidth=2, zorder=3
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, fontweight=weight, zorder=4, color="#1a1a1a")
    return box


def draw_arrow(ax, x1, y1, x2, y2, text="", rad=0.0, color="#333333", lw=1.5, style="-|>"):
    arrow = mpatches.FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=style, mutation_scale=15,
        color=color, linewidth=lw, zorder=2
    )
    ax.add_patch(arrow)
    if text:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.03, text, ha="center", va="bottom", fontsize=9, fontweight="bold", color=color, zorder=5, bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.8))


def generate_langgraph_arch(out_dir):
    print("Generating Architecture Figure 1: Overall LangGraph Architecture...")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(5, 9.5, "LangGraph V7 Multi-Agent Orchestration Topology", ha="center", fontsize=16, fontweight="bold")

    # Nodes
    draw_box(ax, 5, 8.5, 2.8, 0.7, "Intake & Fraud Agent\n(KYC / Data Verification)", "#fff2e6", "#d95f02")
    draw_box(ax, 5, 7.2, 2.4, 0.6, "Planner Agent\n(DAG Routing Plan)", "#e6f2ff", "#2b5c8f")
    draw_box(ax, 5, 6.0, 1.8, 0.5, "Parallel Fan-Out", "#f2f2f2", "#666666", style="square,pad=0.1")

    # Parallel branches
    draw_box(ax, 3, 4.5, 2.4, 0.7, "Risk Agent\n(RF + Heuristic Tiering)", "#e6f9e6", "#2d862d")
    draw_box(ax, 7, 4.5, 2.4, 0.7, "Policy Agent\n(Regulatory Hard Rules)", "#e6e6fa", "#5c5c8a")

    # Join and Orchestrator
    draw_box(ax, 7, 3.0, 2.2, 0.6, "Policy Join Node\n(State Synchronization)", "#f2f2f2", "#666666", style="square,pad=0.1")
    draw_box(ax, 3, 3.0, 2.6, 0.7, "Orchestrator Decide\n(Central Supervisory Logic)", "#fff0f5", "#b30047")

    # Fairness & Reeval
    draw_box(ax, 1.2, 1.5, 2.0, 0.7, "Fairness Agent\n(Counterfactual Audit)", "#ffe6e6", "#cc0000")
    draw_box(ax, 4.8, 1.5, 2.0, 0.7, "Risk Re-evaluation\n(Score Adjustment)", "#e6f9e6", "#2d862d")

    # Terminal decisions
    draw_box(ax, 8.5, 1.5, 2.0, 0.7, "Terminal Actions\n(Approve / Reject / Escalate)", "#d9f2d9", "#006600", style="square,pad=0.1")

    # Arrows
    draw_arrow(ax, 5, 8.15, 5, 7.5, "Valid KYC")
    draw_arrow(ax, 5, 6.9, 5, 6.25)
    draw_arrow(ax, 4.1, 6.0, 3, 4.85, "Branch 1")
    draw_arrow(ax, 5.9, 6.0, 7, 4.85, "Branch 2")
    draw_arrow(ax, 3, 4.15, 3, 3.35)
    draw_arrow(ax, 7, 4.15, 7, 3.3)
    draw_arrow(ax, 7, 2.7, 7, 1.85, "Join to Terminal", rad=-0.2)
    
    # Orchestrator routing
    draw_arrow(ax, 2.0, 2.65, 1.5, 1.85, "Medium Risk")
    draw_arrow(ax, 3.8, 2.65, 4.5, 1.85, "Re-evaluate")
    draw_arrow(ax, 4.3, 3.0, 5.9, 3.0, "Synchronize")
    draw_arrow(ax, 1.5, 1.85, 2.2, 2.65, "Loopback", rad=-0.3, color="#cc0000")
    draw_arrow(ax, 4.8, 1.85, 3.5, 2.65, "Loopback", rad=0.3, color="#2d862d")
    draw_arrow(ax, 6.4, 8.5, 8.5, 2.0, "Fraud / Early Reject", rad=-0.4, color="#d95f02")
    draw_arrow(ax, 3.0, 2.65, 8.0, 2.0, "Auto-Approve / Reject", rad=0.2, color="#006600")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "arch1_langgraph_topology.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "arch1_langgraph_topology.svg"))
    plt.close()
    print("✅ Generated arch1_langgraph_topology (.png & .svg)")


def generate_decision_workflow(out_dir):
    print("Generating Architecture Figure 2: Loan Decision Workflow...")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis("off")

    ax.text(5, 10.5, "End-to-End Loan Underwriting Decision Workflow", ha="center", fontsize=15, fontweight="bold")

    draw_box(ax, 5, 9.5, 3.5, 0.6, "Applicant Feature Vector\n(Demographic, Credit, DTI, Employment)", "#f2f2f2", "#333333")
    draw_box(ax, 5, 8.2, 3.5, 0.7, "Pre-Screening & Hard Overrides\n(Check Hard Rejects & Fast-Tracks)", "#fff2e6", "#d95f02")
    
    # Fast track / hard reject branches
    draw_box(ax, 1.5, 8.2, 2.0, 0.6, "Hard Fast-Track\n(Score = 1.0)", "#d9f2d9", "#006600")
    draw_box(ax, 8.5, 8.2, 2.0, 0.6, "Hard Reject\n(Score = 0.0)", "#ffe6e6", "#cc0000")
    draw_arrow(ax, 3.25, 8.2, 2.5, 8.2, "Match")
    draw_arrow(ax, 6.75, 8.2, 7.5, 8.2, "Violation")

    draw_box(ax, 5, 6.7, 3.8, 0.7, "Risk Tiering Evaluation\n(0.4-0.8 ML Confidence + Heuristic Score)", "#e6f2ff", "#2b5c8f")
    draw_arrow(ax, 5, 7.85, 5, 7.05, "Standard Processing")

    # Risk tiers
    draw_box(ax, 2.0, 5.0, 2.2, 0.6, "Low Risk (≤ 35)\nAuto-Approve", "#d9f2d9", "#006600")
    draw_box(ax, 8.0, 5.0, 2.2, 0.6, "High Risk (> 99)\nAuto-Reject", "#ffe6e6", "#cc0000")
    draw_box(ax, 5.0, 5.0, 2.8, 0.6, "Medium Risk (35 - 99)\nRequires Regulatory & Fairness Audit", "#fff0f5", "#b30047")

    draw_arrow(ax, 4.0, 6.35, 2.5, 5.3, "Score ≤ 35")
    draw_arrow(ax, 6.0, 6.35, 7.5, 5.3, "Score > 99")
    draw_arrow(ax, 5.0, 6.35, 5.0, 5.3, "35 < Score ≤ 99")

    draw_box(ax, 5.0, 3.5, 3.5, 0.7, "Regulatory Policy Enforcement\n(Check Hard RBI Rules & DTI Caps)", "#e6e6fa", "#5c5c8a")
    draw_arrow(ax, 5.0, 4.7, 5.0, 3.85)
    draw_arrow(ax, 6.75, 3.5, 8.0, 4.7, "Policy Fail", rad=0.3, color="#cc0000")

    draw_box(ax, 5.0, 2.0, 3.5, 0.7, "Counterfactual Fairness Audit\n(Flip Sensitive Attribute & Check |ΔCF|)", "#ffe6e6", "#cc0000")
    draw_arrow(ax, 5.0, 3.15, 5.0, 2.35, "Policy Pass")

    draw_box(ax, 5.0, 0.5, 3.0, 0.6, "Terminal Action\n(Approve / Route to Override Rule)", "#d9f2d9", "#006600", style="square,pad=0.1")
    draw_arrow(ax, 5.0, 1.65, 5.0, 0.8)
    draw_arrow(ax, 2.0, 4.7, 3.5, 0.5, "Approved", rad=0.2, color="#006600")
    draw_arrow(ax, 1.5, 7.9, 3.5, 0.5, "Approved", rad=0.4, color="#006600")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "arch2_decision_workflow.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "arch2_decision_workflow.svg"))
    plt.close()
    print("✅ Generated arch2_decision_workflow (.png & .svg)")


def generate_fairness_workflow(out_dir):
    print("Generating Architecture Figure 3: Fairness Escalation Workflow...")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(5, 9.5, "Counterfactual Fairness Audit & Override Rule Trigger Workflow", ha="center", fontsize=15, fontweight="bold")

    draw_box(ax, 5, 8.3, 4.5, 0.7, "Medium-Risk Applicant Profile\n(Original Gender G, Original Risk Score R)", "#fff0f5", "#b30047")
    draw_box(ax, 5, 6.8, 4.5, 0.7, "Counterfactual Feature Cloning\n(Invert Protected Attribute: G_cf = 1 - G)", "#e6f2ff", "#2b5c8f")
    draw_arrow(ax, 5, 7.95, 5, 7.15)

    draw_box(ax, 5, 5.3, 4.5, 0.7, "Model & Heuristic Re-scoring\n(Predict Counterfactual Risk Score R_cf)", "#e6f2ff", "#2b5c8f")
    draw_arrow(ax, 5, 6.45, 5, 5.65)

    draw_box(ax, 5, 3.8, 4.5, 0.7, "Absolute Disparity Calculation\n(ΔCF = |R_cf - R| vs. Threshold ε = 6.0)", "#fff2e6", "#d95f02")
    draw_arrow(ax, 5, 4.95, 5, 4.15)

    draw_box(ax, 2.5, 2.0, 3.2, 0.7, "No Severe Bias (ΔCF ≤ 6.0)\nProceed to Standard Approval", "#d9f2d9", "#006600")
    draw_box(ax, 7.5, 2.0, 3.2, 0.7, "Bias Alert (ΔCF > 6.0)\nRoute to the Override Rule", "#ffe6e6", "#cc0000")
    draw_arrow(ax, 4.0, 3.45, 2.5, 2.35, "ΔCF ≤ 6.0", color="#006600")
    draw_arrow(ax, 6.0, 3.45, 7.5, 2.35, "ΔCF > 6.0", color="#cc0000")

    draw_box(ax, 7.5, 0.5, 3.5, 0.6, "Automated Override Rule\n(Override Harsh Bias / Confirm Policy)", "#f2f2f2", "#333333", style="square,pad=0.1")
    draw_arrow(ax, 7.5, 1.65, 7.5, 0.8, "Audit Case")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "arch3_fairness_workflow.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "arch3_fairness_workflow.svg"))
    plt.close()
    print("✅ Generated arch3_fairness_workflow (.png & .svg)")


def main():
    print("======================================================================")
    print("GENERATING ARCHITECTURE DIAGRAMS (SVG & 300 DPI PNG)")
    print("======================================================================")

    out_dir = os.path.join(os.path.dirname(__file__), "publication_assets", "figures")
    os.makedirs(out_dir, exist_ok=True)

    generate_langgraph_arch(out_dir)
    generate_decision_workflow(out_dir)
    generate_fairness_workflow(out_dir)

    print("\n✅ All 3 architecture diagrams generated successfully in SVG and 300 DPI PNG!")


if __name__ == "__main__":
    main()
