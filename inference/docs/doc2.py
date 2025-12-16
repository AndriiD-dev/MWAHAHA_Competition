import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from dataclasses import dataclass

# --- Configuration & Colors ---
COLORS = {
    "bg": "#FAFAFA",
    "input_fill": "#E3F2FD",  # Blue (Data)
    "input_edge": "#90CAF9",
    "step_fill": "#ECEFF1",  # Grey (Logic)
    "step_edge": "#B0BEC5",
    "model_fill": "#FFF3E0",  # Orange (External/Model)
    "model_edge": "#FFCC80",
    "decision_fill": "#F3E5F5",  # Purple (Validation)
    "decision_edge": "#CE93D8",
    "output_fill": "#E8F5E9",  # Green (Success)
    "output_edge": "#A5D6A7",
    "text_title": "#263238",
    "text_body": "#455A64",
    "arrow": "#546E7A"
}


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def top_mid(self): return (self.x + self.w / 2, self.y + self.h)

    @property
    def bottom_mid(self): return (self.x + self.w / 2, self.y)

    @property
    def left_mid(self): return (self.x, self.y + self.h / 2)

    @property
    def right_mid(self): return (self.x + self.w, self.y + self.h / 2)

    @property
    def center_y(self): return self.y + self.h / 2


def _draw_styled_box(ax, box: Box, title: str, content_lines: list, style="step"):
    # Select colors
    if style == "input":
        face, edge = COLORS["input_fill"], COLORS["input_edge"]
    elif style == "model":
        face, edge = COLORS["model_fill"], COLORS["model_edge"]
    elif style == "decision":
        face, edge = COLORS["decision_fill"], COLORS["decision_edge"]
    elif style == "output":
        face, edge = COLORS["output_fill"], COLORS["output_edge"]
    else:
        face, edge = COLORS["step_fill"], COLORS["step_edge"]

    # Shadow
    shadow = FancyBboxPatch(
        (box.x + 0.005, box.y - 0.005), box.w, box.h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="black", alpha=0.08, zorder=2, mutation_scale=1
    )
    ax.add_patch(shadow)

    # Box
    patch = FancyBboxPatch(
        (box.x, box.y), box.w, box.h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=face, edgecolor=edge, linewidth=1.5, zorder=3
    )
    ax.add_patch(patch)

    # Title - Moved up (margin 0.02)
    ax.text(
        box.x + box.w / 2, box.y + box.h - 0.025, title,
        ha="center", va="top", fontsize=11, fontweight="bold",
        color=COLORS["text_title"], zorder=4
    )

    # Content - Adjusted to fit strictly inside
    # Start Y is derived from box height to ensure consistency
    start_y = box.y + box.h - 0.06
    line_spacing = 0.025
    font_size = 10

    for i, line in enumerate(content_lines):
        ax.text(
            box.x + 0.015, start_y - (i * line_spacing), line,
            ha="left", va="top", fontsize=font_size, color=COLORS["text_body"],
            wrap=True, zorder=4
        )


def _draw_arrow(ax, p0, p1, connection_style="arc3,rad=0", label=None, color=None):
    c = color if color else COLORS["arrow"]
    arrow = FancyArrowPatch(
        p0, p1, arrowstyle="-|>", connectionstyle=connection_style,
        mutation_scale=15, linewidth=1.5, color=c, zorder=1
    )
    ax.add_patch(arrow)
    if label:
        mid_x = (p0[0] + p1[0]) / 2
        mid_y = (p0[1] + p1[1]) / 2

        # Smart label placement for elbows
        if "angle" in connection_style:
            # If line is mostly vertical, place label near the vertical segment
            # If line is mostly horizontal, place near horizontal
            if abs(p1[1] - p0[1]) > abs(p1[0] - p0[0]):
                mid_x = p1[0] - 0.04  # Shift left of vertical line
            else:
                mid_y = p0[1] + 0.02  # Shift above horizontal line

        ax.text(
            mid_x, mid_y, label, ha="center", va="center", fontsize=9,
            style='italic', bbox=dict(facecolor=COLORS["bg"], edgecolor="none", alpha=0.9, pad=3)
        )


def plot_inference_pipeline(out_path="inference_pipeline_fixed.png", show=False):
    # Wider and Taller Canvas to prevent cramping
    fig = plt.figure(figsize=(13, 16), facecolor=COLORS["bg"])
    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- LAYOUT LOGIC ---
    # We use a center column at X=0.40 to allow room on the left for the Retry loop
    c_x = 0.40
    w_main = 0.34  # Wide enough for long text

    # Vertical Positions (Top to Bottom)
    # Using explicit spacing to ensure gaps
    y_in = 0.88
    y_pass1 = 0.70
    y_pass2 = 0.50
    y_valid = 0.32
    y_out = 0.04

    # 1. Main Column Boxes
    box_input = Box(c_x, y_in, w_main, 0.08)
    box_pass1 = Box(c_x, y_pass1, w_main, 0.12)
    box_pass2 = Box(c_x, y_pass2, w_main, 0.15)  # Taller for 3 lines
    box_valid = Box(c_x, y_valid, w_main, 0.13)
    box_output = Box(c_x, y_out, w_main, 0.10)

    # 2. Side Components

    # RETRY LOOP (Left Side)
    # Positioned vertically between Validation and Generation to form a "C" loop
    retry_y = (y_valid + y_pass2) / 2 + 0.02  # ~0.43
    box_retry = Box(0.04, retry_y, 0.26, 0.12)

    # CONTEXT (Right Side)
    # Positioned between Pass 1 and Pass 2
    context_y = (y_pass1 + y_pass2) / 2 + 0.02  # ~0.62
    context_x = c_x + w_main + 0.04
    box_wiki = Box(context_x, context_y, 0.20, 0.10)

    # FALLBACK (Right Side)
    # Positioned below Valid
    fallback_y = 0.20
    fallback_x = c_x + w_main + 0.04
    box_fallback = Box(fallback_x, fallback_y, 0.20, 0.10)

    # --- DRAW BOXES ---
    _draw_styled_box(ax, box_input, "Input Data", [
        "task-a-two-words.csv",
        "Columns: word1, word2"
    ], style="input")

    _draw_styled_box(ax, box_wiki, "Context Enrichment", [
        "Wikipedia API (Cached)",
        "Build 'FACTS' block"
    ], style="model")

    _draw_styled_box(ax, box_pass1, "Pass 1: Planning", [
        "Input: Words + FACTS",
        "Temp: 0.4 (Creative but stable)",
        "Output: JSON Plan"
    ], style="step")

    _draw_styled_box(ax, box_pass2, "Pass 2: Generation", [
        "Input: Words + FACTS + Plan",
        "Temp: 0.8 (High creativity)",
        "Constraint: Include both words"
    ], style="step")

    _draw_styled_box(ax, box_valid, "Validation Logic", [
        "Check: required_words_present()",
        "1. Heuristic regex check",
        "2. Plural/Possessive handling"
    ], style="decision")

    _draw_styled_box(ax, box_retry, "Retry Loop (Max 5)", [
        "Increases Temp (up to 1.05)",
        "Injects Strict Suffix:",
        "'...INVALID unless...'"
    ], style="decision")

    _draw_styled_box(ax, box_fallback, "Fallback Strategy", [
        "Static Template",
        "Guarantees valid output",
        "Used if retries exhaust"
    ], style="step")

    _draw_styled_box(ax, box_output, "Final Output", [
        "Predictions.tsv + ZIP",
        "100% Valid Format"
    ], style="output")

    # --- DRAW ARROWS ---

    # 1. Main Flow (Down)
    _draw_arrow(ax, box_input.bottom_mid, box_pass1.top_mid)
    _draw_arrow(ax, box_pass1.bottom_mid, box_pass2.top_mid, label="Plan JSON")
    _draw_arrow(ax, box_pass2.bottom_mid, box_valid.top_mid, label="Draft Joke")

    # 2. Context Injections (From Right)
    # To Plan
    _draw_arrow(ax, box_wiki.left_mid, box_pass1.right_mid,
                connection_style="angle,angleA=180,angleB=90,rad=5")
    # To Gen (Use a slightly wider radius to separate lines)
    _draw_arrow(ax, box_wiki.left_mid, box_pass2.right_mid,
                connection_style="angle,angleA=180,angleB=90,rad=15", label="Facts")

    # 3. Validation: Success
    _draw_arrow(ax, box_valid.bottom_mid, box_output.top_mid, label="Valid", color="#4CAF50")

    # 4. Retry Logic (The "C" Loop on Left)
    # Fail: Valid Left -> Retry Bottom (Elbow Up)
    _draw_arrow(ax, box_valid.left_mid, box_retry.bottom_mid,
                connection_style="angle,angleA=180,angleB=270,rad=10",
                label="Missing Words", color="#E53935")

    # Regen: Retry Top -> Pass 2 Left (Elbow Right)
    _draw_arrow(ax, box_retry.top_mid, box_pass2.left_mid,
                connection_style="angle,angleA=90,angleB=180,rad=10",
                label="Re-generate", color="#E53935")

    # 5. Fallback Logic (Right)
    # Exhausted: Valid Right -> Fallback Top
    _draw_arrow(ax, box_valid.right_mid, box_fallback.top_mid,
                connection_style="angle,angleA=0,angleB=90,rad=10",
                label="Exhausted", color="#EF6C00")

    # To Output: Fallback Bottom -> Output Right
    _draw_arrow(ax, box_fallback.bottom_mid, box_output.right_mid,
                connection_style="angle,angleA=270,angleB=0,rad=10")

    # --- SAVE ---
    out_path = os.path.abspath(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=COLORS["bg"])
    print(f"Saved diagram to: {out_path}")

    if show:
        try:
            plt.show()
        except Exception:
            print("Plot saved (interactive show failed).")
    plt.close(fig)


if __name__ == "__main__":
    plot_inference_pipeline(show=True)