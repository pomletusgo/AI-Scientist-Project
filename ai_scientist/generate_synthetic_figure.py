#!/usr/bin/env python3
"""
Paper-specific figure generator — LLM designs, matplotlib renders.
===============================================================
1. Feed paper idea to DeepSeek → get structured figure description (JSON)
2. matplotlib renders exactly what DeepSeek described
3. Each figure is unique to that paper's actual content

Usage:
  python ai_scientist/generate_synthetic_figure.py --idea-file results/01-physics-gravity/idea.json
"""

import argparse, json, os, sys, re, textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_scientist.llm import get_response_from_llm, create_client, extract_json_between_markers

# ================================================================
# Prompt: Ask DeepSeek to design the figure
# ================================================================

FIGURE_DESIGN_PROMPT = """Task: Design a scientific overview figure for a research paper.

Below is a paper idea. Your job is to produce a JSON description of a figure that
visually summarizes the paper's core problem, method, and outcome.

The JSON must have these exact keys: figure_title, blocks, flow, caption.
Each block needs: id, header, color, items.

IMPORTANT: Every item in every block MUST use specific terminology from THIS paper.
Use the paper's actual domain terms, method names, and metrics. No generic filler.

PAPER IDEA:
{idea_json}

OUTPUT (JSON only, no other text):
{{
  "figure_title": "concise 5-8 word title",
  "blocks": [
    {{
      "id": "challenge",
      "header": "short label",
      "color": "red",
      "items": ["paper-specific issue 1", "paper-specific issue 2"]
    }},
    {{
      "id": "approach",
      "header": "short label",
      "color": "blue",
      "items": ["paper-specific step 1", "paper-specific step 2", "paper-specific step 3"]
    }},
    {{
      "id": "impact",
      "header": "short label",
      "color": "green",
      "items": ["paper-specific result 1", "paper-specific result 2"]
    }}
  ],
  "flow": [
    {{"from": "challenge", "to": "approach", "label": "motivates"}},
    {{"from": "approach", "to": "impact", "label": "achieves"}}
  ],
  "caption": "Figure 1: one specific sentence describing this figure."
}}"""


def ask_deepseek_for_figure_design(idea, client=None, model=None):
    """Ask DeepSeek to design a paper-specific figure. Returns parsed JSON dict."""
    if client is None:
        client, model = create_client(None)

    prompt = FIGURE_DESIGN_PROMPT.format(
        idea_json=json.dumps(idea, indent=2, ensure_ascii=False)
    )

    print(f"  [FigureDesign] Asking DeepSeek to design figure...")
    resp, _ = get_response_from_llm(
        prompt,
        client=client,
        model=model,
        system_message="You are a scientific figure designer. Output ONLY valid JSON.",
        temperature=0.3,
    )

    # Extract JSON — try multiple strategies
    design = None

    # Strategy 1: Find JSON object containing "blocks" key (most reliable)
    match = re.search(r'\{[\s\S]*"blocks"[\s\S]*\}', resp)
    if match:
        try:
            design = json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 2: markdown fenced ```json ... ``` block
    if design is None:
        m = re.search(r'```json\s*([\s\S]*?)```', resp)
        if m:
            try:
                design = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

    # Strategy 3: any complete JSON object with blocks key
    if design is None:
        for m in re.finditer(r'\{[\s\S]*?\}', resp):
            try:
                candidate = json.loads(m.group())
                if isinstance(candidate, dict) and "blocks" in candidate:
                    design = candidate
                    break
            except json.JSONDecodeError:
                continue

    if design is None:
        print(f"  [FigureDesign] Failed to parse JSON from response. Using fallback.")
        return None

    # Validate required keys
    required = ["blocks", "flow", "caption", "figure_title"]
    for k in required:
        if k not in design:
            print(f"  [FigureDesign] Missing key '{k}' — using fallback.")
            return None

    print(f"  [FigureDesign] Got design: {len(design.get('blocks', []))} blocks, "
          f"{len(design.get('flow', []))} arrows, title='{design.get('figure_title', '')}'")
    return design


# ================================================================
# matplotlib Renderer — takes the LLM design and draws it
# ================================================================

COLOR_MAP = {
    "red":    ("#E74C3C", "#FADBD8"),
    "blue":   ("#2980B9", "#D6EAF8"),
    "green":  ("#27AE60", "#D5F5E3"),
    "orange": ("#E67E22", "#FDEBD0"),
    "purple": ("#8E44AD", "#E8DAEF"),
}
FALLBACK_COLORS = [
    ("#E74C3C", "#FADBD8"),
    ("#2980B9", "#D6EAF8"),
    ("#27AE60", "#D5F5E3"),
]


def render_figure(design, output_dir, paper_title="", verbose=True):
    """Render the LLM-designed figure with matplotlib."""
    blocks = design.get("blocks", [])
    flow = design.get("flow", [])
    figure_title = design.get("figure_title", "Method Overview")
    caption = design.get("caption", "")

    if len(blocks) < 2:
        print("  [Render] Need at least 2 blocks — using fallback renderer")
        return None

    # --- Layout math ---
    n_blocks = len(blocks)
    # Auto-calculate canvas size
    fig_w = max(12, n_blocks * 4.5)
    fig_h = 8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect('equal')
    ax.axis('off')

    BG_COLOR = '#FAFAFA'
    TEXT_COLOR = '#2C3E50'
    fig.patch.set_facecolor(BG_COLOR)

    # --- Block layout ---
    margin = 0.8
    gap = 0.6
    usable_w = fig_w - 2 * margin - (n_blocks - 1) * gap
    block_w = usable_w / n_blocks
    block_h = fig_h - 3.0  # leave room for title and caption

    block_positions = {}
    for i, block in enumerate(blocks):
        x0 = margin + i * (block_w + gap)
        y0 = 1.2
        block_positions[block["id"]] = (x0, y0, block_w, block_h)

    # --- Figure title ---
    ax.text(fig_w / 2, fig_h - 0.5, figure_title,
            ha='center', va='center', fontsize=16, fontweight='bold',
            color=TEXT_COLOR, fontfamily='sans-serif')

    # Subtitle: paper name
    paper_short = paper_title[:90] + "..." if len(paper_title) > 90 else paper_title
    ax.text(fig_w / 2, fig_h - 1.0, paper_short,
            ha='center', va='center', fontsize=9, fontstyle='italic',
            color='#95A5A6', fontfamily='sans-serif')

    # --- Draw each block ---
    for i, block in enumerate(blocks):
        x0, y0, bw, bh = block_positions[block["id"]]
        header = block.get("header", block["id"].title())
        items = block.get("items", [])
        color_name = block.get("color", "blue")
        primary, light = COLOR_MAP.get(color_name, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])

        # Block background
        rect = FancyBboxPatch((x0, y0), bw, bh,
                               boxstyle="round,pad=0.15",
                               facecolor=light, edgecolor=primary, linewidth=2.0)
        ax.add_patch(rect)

        # Block header bar
        header_h = 0.5
        header_rect = FancyBboxPatch((x0 + 0.15, y0 + bh - header_h - 0.15),
                                      bw - 0.3, header_h,
                                      boxstyle="round,pad=0.06",
                                      facecolor=primary, edgecolor='none', alpha=0.9)
        ax.add_patch(header_rect)
        ax.text(x0 + bw / 2, y0 + bh - header_h / 2 - 0.15, header,
                ha='center', va='center', fontsize=10.5, fontweight='bold',
                color='white', fontfamily='sans-serif')

        # Block items
        item_area_top = y0 + bh - header_h - 0.4
        item_area_bottom = y0 + 0.3
        item_area_h = item_area_top - item_area_bottom
        n_items = len(items)
        if n_items == 0:
            continue

        item_spacing = min(0.7, item_area_h / (n_items + 1))
        for j, item in enumerate(items):
            # Truncate long items
            item_text = item.strip()
            if len(item_text) > 70:
                # Wrap at word boundary
                item_text = textwrap.fill(item_text, width=55)

            iy = item_area_top - (j + 1) * item_spacing
            # Item background chip
            lines = item_text.count('\n') + 1
            chip_h = 0.3 + 0.2 * lines
            chip = FancyBboxPatch((x0 + 0.25, iy - chip_h / 2),
                                   bw - 0.5, chip_h,
                                   boxstyle="round,pad=0.08",
                                   facecolor='white', edgecolor=primary,
                                   linewidth=0.8, alpha=0.7)
            ax.add_patch(chip)
            ax.text(x0 + bw / 2, iy, item_text,
                    ha='center', va='center', fontsize=8.0,
                    color=TEXT_COLOR, fontfamily='sans-serif')

    # --- Draw flow arrows ---
    for f in flow:
        from_id = f.get("from", "")
        to_id = f.get("to", "")
        label = f.get("label", "")

        if from_id not in block_positions or to_id not in block_positions:
            continue

        fx0, fy0, fbw, fbh = block_positions[from_id]
        tx0, ty0, tbw, tbh = block_positions[to_id]

        # Arrow from right edge of "from" to left edge of "to"
        ax_start = (fx0 + fbw, fy0 + fbh / 2)
        ax_end = (tx0, ty0 + tbh / 2)

        arrow = FancyArrowPatch(ax_start, ax_end,
                                 arrowstyle='->', mutation_scale=20,
                                 color='#7F8C8D', linewidth=2.5, alpha=0.7,
                                 connectionstyle="arc3,rad=0.1")
        ax.add_patch(arrow)

        # Label on the arrow
        if label:
            mid_x = (ax_start[0] + ax_end[0]) / 2
            mid_y = (ax_start[1] + ax_end[1]) / 2 + 0.25
            ax.text(mid_x, mid_y, label,
                    ha='center', va='center', fontsize=8,
                    fontstyle='italic', color='#7F8C8D', fontfamily='sans-serif',
                    bbox=dict(boxstyle="round,pad=0.15", facecolor='white',
                              edgecolor='none', alpha=0.8))

    # --- Caption at bottom ---
    if caption:
        cap_wrapped = textwrap.fill(caption.strip(), width=120)
        ax.text(fig_w / 2, 0.3, cap_wrapped,
                ha='center', va='center', fontsize=9, fontstyle='italic',
                color='#7F8C8D', fontfamily='sans-serif')

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "method_figure.png")
    fig.savefig(png_path, dpi=150, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')

    pdf_path = os.path.join(output_dir, "method_overview.pdf")
    fig.savefig(pdf_path, dpi=150, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none', format='pdf')
    plt.close(fig)

    if verbose:
        print(f"  [Render] Saved: {png_path} ({os.path.getsize(png_path)} bytes)")
        print(f"  [Render] Saved: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")

    return png_path


# ================================================================
# Main entry point
# ================================================================

def generate_method_figure(idea, output_dir, client=None, model=None, verbose=True):
    """
    Generate a paper-specific method figure.

    1. Ask DeepSeek to design the figure content based on the paper idea
    2. Render with matplotlib

    Parameters
    ----------
    idea : dict — Paper idea with Title, Abstract, Method, Experiment keys.
    output_dir : str — Output directory for the figure.
    client : LLM client (optional, auto-creates DeepSeek client if None)
    model : model name (optional)

    Returns
    -------
    str or None — Path to saved figure.
    """
    paper_title = idea.get("Title", idea.get("Name", "Research Paper"))

    # Step 1: Get figure design from DeepSeek
    design = ask_deepseek_for_figure_design(idea, client=client, model=model)

    if design is None:
        # Fallback: generic three-column design
        print("  [FigureDesign] Using fallback generic design.")
        design = _make_fallback_design(idea)

    # Step 2: Render with matplotlib
    path = render_figure(design, output_dir, paper_title=paper_title, verbose=verbose)
    return path


def _make_fallback_design(idea):
    """Create a generic fallback design if LLM fails."""
    method_text = idea.get("Method", idea.get("Abstract", ""))
    abstract = idea.get("Abstract", "")

    # Extract keywords as fallback items
    import re
    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[a-z]+){1,3}', method_text + " " + abstract)
    unique_words = list(dict.fromkeys(words))[:8]

    return {
        "figure_title": "Research Overview",
        "layout": "three_column",
        "blocks": [
            {
                "id": "problem",
                "header": "Research Problem",
                "color": "red",
                "items": unique_words[:2] if len(unique_words) >= 2 else [
                    "Key limitation in current approaches",
                    "Critical gap to address"
                ]
            },
            {
                "id": "method",
                "header": "Proposed Approach",
                "color": "blue",
                "items": unique_words[2:5] if len(unique_words) >= 5 else [
                    "Novel methodological component",
                    "Integration framework",
                    "Evaluation protocol"
                ]
            },
            {
                "id": "outcome",
                "header": "Expected Impact",
                "color": "green",
                "items": unique_words[5:7] if len(unique_words) >= 7 else [
                    "Performance improvement",
                    "Practical contribution"
                ]
            },
        ],
        "flow": [
            {"from": "problem", "to": "method", "label": "drives"},
            {"from": "method", "to": "outcome", "label": "achieves"},
        ],
        "caption": "Figure 1: Overview of the research problem, proposed methodology, and key outcomes."
    }


def main():
    parser = argparse.ArgumentParser(description="Paper-specific figure generator (LLM + matplotlib)")
    parser.add_argument("--idea-file", required=True, help="Path to idea.json")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--model", default=None, help="LLM model (default: auto-detect)")
    args = parser.parse_args()

    with open(args.idea_file, "r", encoding="utf-8") as f:
        idea = json.load(f)

    client, model = create_client(args.model)
    path = generate_method_figure(idea, args.output_dir, client=client, model=model)
    if path:
        print(f"Figure saved: {path}")
    else:
        print("Figure generation failed.")


if __name__ == "__main__":
    main()
