import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from dataclasses import dataclass

# --- Configuration & Colors ---
COLORS = {
    "bg": "#FAFAFA",  # Off-white background
    "box_fill": "#ECEFF1",  # Light Grey-Blue for process steps
    "box_edge": "#B0BEC5",  # Slate Grey border
    "data_fill": "#E3F2FD",  # Light Blue for Data Objects (Input)
    "data_edge": "#90CAF9",  # Blue border
    "target_fill": "#E8F5E9",  # Light Green for Targets
    "target_edge": "#A5D6A7",
    "external_fill": "#FFF3E0",  # Light Orange for External/Models (SpaCy/Wiki)
    "external_edge": "#FFCC80",
    "text_title": "#263238",  # Dark Slate
    "text_body": "#455A64",  # Medium Slate
    "arrow": "#546E7A"  # Arrow Color
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


def _draw_styled_box(ax, box: Box, title: str, content_lines: list, style="process"):
    """
    Draws a larger, cleaner box with a drop shadow.
    """
    # 1. Determine Colors based on style
    if style == "input" or style == "data":
        face, edge = COLORS["data_fill"], COLORS["data_edge"]
    elif style == "target" or style == "final":
        face, edge = COLORS["target_fill"], COLORS["target_edge"]
    elif style == "external":
        face, edge = COLORS["external_fill"], COLORS["external_edge"]
    else:  # process
        face, edge = COLORS["box_fill"], COLORS["box_edge"]

    # 2. Draw Shadow
    shadow = FancyBboxPatch(
        (box.x + 0.005, box.y - 0.005), box.w, box.h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="black", alpha=0.08, zorder=2, mutation_scale=1
    )
    ax.add_patch(shadow)

    # 3. Draw Main Box
    patch = FancyBboxPatch(
        (box.x, box.y), box.w, box.h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=face, edgecolor=edge, linewidth=1.5, zorder=3
    )
    ax.add_patch(patch)

    # 4. Text Content
    # Title (Centered, Bold)
    ax.text(
        box.x + box.w / 2, box.y + box.h - 0.035, title,
        ha="center", va="top", fontsize=11, fontweight="bold",
        color=COLORS["text_title"], zorder=4
    )

    # Body lines (Left Aligned)
    start_y = box.y + box.h - 0.08
    line_spacing = 0.028
    font_size = 10

    for i, line in enumerate(content_lines):
        ax.text(
            box.x + 0.02, start_y - (i * line_spacing), line,
            ha="left", va="top", fontsize=font_size, color=COLORS["text_body"],
            wrap=True, zorder=4
        )


def _draw_arrow(ax, p0, p1, connection_style="arc3,rad=0", label=None):
    arrow = FancyArrowPatch(
        p0, p1,
        arrowstyle="-|>",
        connectionstyle=connection_style,
        mutation_scale=15,
        linewidth=1.5,
        color=COLORS["arrow"],
        zorder=1
    )
    ax.add_patch(arrow)

    if label:
        mid_x = (p0[0] + p1[0]) / 2
        mid_y = (p0[1] + p1[1]) / 2

        ax.text(
            mid_x, mid_y, label,
            ha="center", va="center", fontsize=9, style='italic',
            bbox=dict(facecolor=COLORS["bg"], edgecolor="none", alpha=1.0, pad=3)
        )


def save_and_show(fig, path, show):
    path = os.path.abspath(path)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=COLORS["bg"])
    print(f"Saved diagram to: {path}")

    if show:
        try:
            plt.show()
        except AttributeError:
            print("\n[Warning] Could not display plot in PyCharm backend.")
            print("The image was saved successfully to disk.")
        except Exception as e:
            print(f"Could not show figure. Error: {e}")
    plt.close(fig)


# ==========================================
# DIAGRAM: Dataset Preparation Pipeline
# ==========================================
def plot_preparation_pipeline(show=False):
    out_path = "dataset_prep_pipeline.png"

    fig = plt.figure(figsize=(12, 14), facecolor=COLORS["bg"])
    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- LAYOUT LOGIC ---
    c_x = 0.30
    w_main = 0.40

    # Vertically distributed boxes
    # 1. Raw Data (Top)
    box_raw = Box(c_x, 0.85, w_main, 0.12)
    # 2. Preprocessing
    box_clean = Box(c_x, 0.66, w_main, 0.13)
    # 3. Sampling
    box_pairs = Box(c_x, 0.42, w_main, 0.16)
    # 4. Enrichment
    box_facts = Box(c_x, 0.21, w_main, 0.14)
    # 5. Final (Bottom)
    box_final = Box(c_x, 0.02, w_main, 0.13)

    # --- SIDE HELPERS (Adjusted Heights) ---

    # Height increased to 0.16 (was 0.12)
    side_h = 0.16

    # SpaCy aligns with Box 3 (Sampling) center
    spacy_y = box_pairs.center_y - (side_h / 2)
    box_spacy = Box(0.02, spacy_y, 0.22, side_h)

    # Wiki aligns with Box 4 (Enrichment) center
    wiki_y = box_facts.center_y - (side_h / 2)
    box_wiki = Box(0.76, wiki_y, 0.22, side_h)

    # --- DRAWING ---

    # 1. Raw Inputs
    _draw_styled_box(ax, box_raw, "1. Raw Data Sources", [
        "- SFT JSONL (1.2k rows)",
        "- DPO CSV (12k rows)"
    ], style="data")

    # 2. Cleaning
    _draw_styled_box(ax, box_clean, "2. Preprocessing", [
        "- Merge SFT & DPO",
        "- Normalize text",
        "- Deduplicate jokes"
    ], style="process")

    # 3. SpaCy (Side)
    _draw_styled_box(ax, box_spacy, "SpaCy NLP", [
        "- en_core_web_sm",
        "- Noun Chunking",
        "- Stopword Filter",
        "- POS Tagging"
    ], style="external")

    # 4. Sampling
    _draw_styled_box(ax, box_pairs, "3. Anchor Sampling", [
        "- Extract Candidates",
        "- Filter (min len, common)",
        "- Seeded Random Choice",
        "- Select 2 distinct words"
    ], style="process")

    # 5. Wikipedia (Side)
    _draw_styled_box(ax, box_wiki, "Wikipedia API", [
        "- Custom User-Agent",
        "- Local JSON Cache",
        "- Rate limit handling",
        "- Summary Extraction"
    ], style="external")

    # 6. Facts
    _draw_styled_box(ax, box_facts, "4. Context Enrichment", [
        "- Fetch summary for anchors",
        "- Fallback strategies",
        "- Format 'FACTS' block"
    ], style="process")

    # 7. Final Output
    _draw_styled_box(ax, box_final, "5. Prepared Dataset", [
        "- anchors_with_facts.tsv",
        "- Columns: anchor1, anchor2,",
        "  facts_block, joke"
    ], style="final")

    # --- ARROWS ---

    # Main flow
    _draw_arrow(ax, box_raw.bottom_mid, box_clean.top_mid)
    _draw_arrow(ax, box_clean.bottom_mid, box_pairs.top_mid, label="Cleaned DF")
    _draw_arrow(ax, box_pairs.bottom_mid, box_facts.top_mid, label="Anchor Pairs")
    _draw_arrow(ax, box_facts.bottom_mid, box_final.top_mid)

    # Side Injections
    _draw_arrow(ax, box_spacy.right_mid, box_pairs.left_mid, label="POS Tags")
    _draw_arrow(ax, box_wiki.left_mid, box_facts.right_mid, label="Summaries")

    save_and_show(fig, out_path, show)


if __name__ == "__main__":
    plot_preparation_pipeline(show=True)