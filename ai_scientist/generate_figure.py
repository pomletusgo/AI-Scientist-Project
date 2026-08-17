#!/usr/bin/env python3
"""
Dynamic Paper Figure Generator for AI-Scientist.
================================================
Given a paper's title, abstract, and core idea, generates a tailored method
figure using Google Gemini (NanoBanana) image generation.

The figure is automatically placed in the paper's output directory so the
LaTeX writeup pipeline can reference it.

Integration point:
    Called after experiments, before perform_writeup() in launch_scientist.py.

Usage (standalone):
    python ai_scientist/generate_figure.py --idea-file results/idea.json
    python ai_scientist/generate_figure.py --title "..." --abstract "..." --method "..."

Usage (programmatic):
    from ai_scientist.generate_figure import generate_paper_figure

    path = generate_paper_figure(
        idea={"Name": "...", "Title": "...", "Experiment": "...", ...},
        output_dir="results/my_paper/",
        api_key="AQ.xxx",
    )
"""

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ================================================================
# Prompt Builder — dynamically constructs image prompt from paper
# ================================================================

def build_dynamic_prompt(title, abstract, method_desc, experiment_notes="", section="method"):
    """
    Build a NanoBanana image prompt tailored to a specific paper.

    The prompt describes a scientific figure that illustrates:
      - The research problem / key trade-off
      - The proposed method / approach
      - The expected outcome

    Parameters
    ----------
    title : str
        Paper title.
    abstract : str
        Paper abstract (or summary of the research).
    method_desc : str
        Description of the core methodology.
    experiment_notes : str
        Notes about experiments (plots, metrics, etc.).
    section : str
        Figure type — "method" (single method figure), "pipeline" (full pipeline),
        or "overview" (research narrative overview).

    Returns
    -------
    dict with keys: prompt (str), title (str), filename (str)
    """
    # Truncate inputs for context window efficiency
    abstract_short = abstract[:600].strip()
    method_short = method_desc[:500].strip()

    # --- Detect domain from title/abstract keywords ---
    domain_keywords = {
        "optics": ["super-resolution", "imaging", "microscopy", "optical", "diffraction", "fluorescence", "photon", "lens"],
        "genomics": ["sequencing", "variant calling", "genome", "genomic", "DNA", "RNA", "read alignment", "haplotype"],
        "chemistry": ["catalysis", "catalyst", "reaction", "synthesis", "ligand", "cross-coupling", "polymer", "electrochemical"],
        "materials": ["alloy", "composite", "material", "strength", "ductility", "fatigue", "corrosion", "thermal"],
        "medicine": ["diagnostic", "biomarker", "clinical", "patient", "assay", "therapeutic", "point-of-care"],
        "economics": ["causal", "economic", "policy", "regression", "market", "trade", "inference"],
        "engineering": ["control system", "autonomous", "fault-tolerant", "robot", "sensor", "actuator", "vehicle"],
        "environmental": ["PM2.5", "climate", "emission", "pollution", "satellite", "air quality", "sustainability"],
        "nlp": ["language model", "NLP", "text", "transformer", "LLM", "token", "attention mechanism"],
        "ml_general": ["neural network", "deep learning", "training", "optimization", "generalization", "classification"],
    }

    detected_domain = "general scientific research"
    for domain, keywords in domain_keywords.items():
        text_lower = (title + " " + abstract_short).lower()
        if any(kw.lower() in text_lower for kw in keywords):
            detected_domain = domain
            break

    # --- Build the prompt based on section type ---
    if section == "pipeline":
        return _build_pipeline_prompt(title, abstract_short, method_short, detected_domain)
    elif section == "overview":
        return _build_overview_prompt(title, abstract_short, method_short, detected_domain)
    else:  # method (default)
        return _build_method_prompt(title, abstract_short, method_short, detected_domain)


def _build_method_prompt(title, abstract_short, method_short, domain):
    """Build a method-centric figure prompt."""
    prompt = f"""Create a professional scientific method diagram for a research paper titled:
"{title}"

The paper is in the domain of {domain}. Here is the core research:

ABSTRACT: {abstract_short}

METHOD: {method_short}

Design a clear, clean scientific figure with three columns (left to right):

LEFT COLUMN — "Problem / Challenge":
- Visual representation of the key research problem
- Show the main technical trade-off or limitation being addressed
- Use warm colors (red/orange) for the problem space
- Include a brief label describing the challenge

CENTER COLUMN — "Proposed Method":
- The core methodology / framework proposed in the paper
- Show key components and their interactions with connecting arrows
- Use blue tones for the method components
- Each main component should have a simple icon/box with 1-2 word label
- Show data flow or information flow between components

RIGHT COLUMN — "Outcome / Results":
- The expected improvements or solutions
- Use green tones for positive outcomes
- Show key metrics improving (arrows going up) or key issues resolved (checkmarks)
- Contrast with the problem state from the left column

STYLE REQUIREMENTS:
- Clean white background, professional academic style suitable for a top-tier scientific publication
- Vector-like flat illustration style, no photorealistic elements
- Sans-serif font style, no text overlapping, clear visual hierarchy
- 16:9 aspect ratio
- Color scheme: red/orange (challenge), blue (method), green (outcome)
- Minimal but clear labeling — prioritize visual clarity over text density
- The figure should be self-contained and understandable without reading the paper"""

    return {
        "title": f"Method Overview: {title}",
        "filename": "method_figure.png",
        "prompt": prompt,
    }


def _build_pipeline_prompt(title, abstract_short, method_short, domain):
    """Build a pipeline/architecture figure prompt."""
    prompt = f"""Create a professional pipeline architecture diagram for a research paper titled:
"{title}"

Domain: {domain}

ABSTRACT: {abstract_short}

METHOD: {method_short}

Design a top-to-bottom pipeline diagram with these stages:

STAGE 1 — "Problem Analysis":
- How the research problem is decomposed/understood
- Key inputs or data sources

STAGE 2 — "Proposed Approach":
- The main methodological innovation
- Key algorithms, models, or frameworks used
- How components interact

STAGE 3 — "Experimental Validation":
- Evaluation setup (datasets, metrics, baselines)
- Training or testing process

STAGE 4 — "Results & Insights":
- Key findings and outcomes
- Performance improvements or discoveries

STYLE REQUIREMENTS:
- Professional pipeline diagram, light gray or white background
- Distinct but harmonious color per stage
- Clean arrows connecting stages (top to bottom flow)
- Icon + concise text for each stage
- Flat vector illustration style suitable for a systems/ML paper
- 4:3 aspect ratio
- Minimal text — use visual elements and icons where possible"""

    return {
        "title": f"Pipeline: {title}",
        "filename": "pipeline_figure.png",
        "prompt": prompt,
    }


def _build_overview_prompt(title, abstract_short, method_short, domain):
    """Build a comprehensive research overview figure prompt."""
    prompt = f"""Create a comprehensive research overview illustration for a paper titled:
"{title}"

Domain: {domain}

ABSTRACT: {abstract_short}

METHOD: {method_short}

Design an infographic-style overview with three horizontal bands:

TOP BAND — "The Challenge":
- The research gap or problem being addressed
- Why existing approaches fall short
- Red/orange color scheme
- A central question or problem statement

MIDDLE BAND — "The Approach":
- Bridge between challenge and solution
- 3-4 key methodological pillars, each with a simple icon and brief label
- Blue color scheme
- Arrows showing how pillars reinforce each other

BOTTOM BAND — "The Impact":
- Key results and contributions
- Quantitative indicators if mentioned in the abstract
- Green color scheme
- Checkmarks or upward arrows for improvements

SIDE ELEMENT — A circular "Research Loop" showing: Analyze → Design → Evaluate → Refine

STYLE REQUIREMENTS:
- Professional infographic style, white background
- Clear visual hierarchy with distinct color zones
- Vector-like flat illustration
- 16:9 aspect ratio
- Suitable as a paper overview / graphical abstract"""

    return {
        "title": f"Research Overview: {title}",
        "filename": "overview_figure.png",
        "prompt": prompt,
    }


# ================================================================
# Image Generation via NanoBanana (Google Gemini)
# ================================================================

def generate_paper_figure(
    idea,
    output_dir,
    api_key=None,
    model="gemini-3.1-flash-image-preview",
    figure_type="method",
    verbose=True,
):
    """
    Generate a paper figure using NanoBanana and save it to the paper directory.

    Parameters
    ----------
    idea : dict
        Paper idea dictionary with keys: Name, Title, Abstract (optional),
        Experiment (optional), Method (optional).
    output_dir : str
        Path to the paper's output directory (e.g., results/my_paper/).
    api_key : str
        Google API key (AQ.xxx for Vertex AI Express, or AIza.xxx for Google AI).
    model : str
        Image generation model name.
    figure_type : str
        "method", "pipeline", or "overview".
    verbose : bool
        Print progress messages.

    Returns
    -------
    str or None
        Path to the saved figure file, or None if generation failed.
    """
    # --- Validate inputs ---
    if api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print("[FigureGen] ERROR: No API key. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
        return None

    # --- Extract idea metadata ---
    title = idea.get("Title", idea.get("Name", "Research Paper"))
    abstract = idea.get("Abstract", idea.get("Experiment", ""))
    method_desc = idea.get("Method", idea.get("Experiment", abstract))
    experiment_notes = idea.get("Experiment", "")

    # If the idea has a structured "Experiment" field with notes, use that
    if isinstance(experiment_notes, dict):
        experiment_notes = experiment_notes.get("notes", str(experiment_notes))

    # --- Build the prompt ---
    prompt_info = build_dynamic_prompt(
        title=title,
        abstract=abstract,
        method_desc=method_desc,
        experiment_notes=str(experiment_notes)[:400],
        section=figure_type,
    )

    if verbose:
        print(f"\n[FigureGen] Generating {figure_type} figure for: {title[:80]}...")
        print(f"[FigureGen] Domain detected from title/abstract keywords.")
        print(f"[FigureGen] Prompt length: {len(prompt_info['prompt'])} characters.")
        print(f"[FigureGen] Model: {model}")

    # --- Call NanoBanana API ---
    try:
        from google import genai
    except ImportError:
        print("[FigureGen] ERROR: google-genai package not installed.")
        print("[FigureGen] Install with: pip install google-genai")
        return None

    client = genai.Client(api_key=api_key, http_options={"timeout": 120000})  # 120s timeout

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt_info["prompt"],
            config={"response_modalities": ["IMAGE", "TEXT"]},
        )
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            print("[FigureGen] ERROR: API quota exhausted (429). Try again later.")
        elif "401" in error_str or "403" in error_str:
            print(f"[FigureGen] ERROR: API authentication failed. Check your key.")
        elif "location" in error_str.lower():
            print("[FigureGen] ERROR: API not available in your region.")
        else:
            print(f"[FigureGen] ERROR: API call failed: {e}")
        if verbose:
            traceback.print_exc()
        return None

    # --- Extract and save image ---
    output_path = None
    for i, part in enumerate(response.candidates[0].content.parts):
        if hasattr(part, "inline_data") and part.inline_data:
            image_data = part.inline_data.data
            mime_type = part.inline_data.mime_type
            ext = mime_type.split("/")[-1] if "/" in mime_type else "png"

            filename = prompt_info["filename"].rsplit(".", 1)[0] + f".{ext}"
            output_path = os.path.join(output_dir, filename)

            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(image_data)

            # Auto-convert JPEG → PNG for LaTeX compatibility
            if ext in ("jpeg", "jpg"):
                try:
                    from PIL import Image
                    import io
                    png_path = output_path.rsplit(".", 1)[0] + ".png"
                    img = Image.open(io.BytesIO(image_data))
                    img.save(png_path, "PNG")
                    output_path = png_path
                    if verbose:
                        print(f"[FigureGen] Auto-converted JPEG→PNG: {png_path}")
                except ImportError:
                    if verbose:
                        print(f"[FigureGen] PIL not available — saving as JPEG")

            if verbose:
                print(f"[FigureGen] Figure saved: {output_path} ({len(image_data)} bytes)")

        if hasattr(part, "text") and part.text and verbose:
            print(f"[FigureGen] Model feedback: {part.text[:200]}")

    if output_path is None:
        print("[FigureGen] WARNING: Model did not return an image.")
        if verbose:
            for i, cand in enumerate(response.candidates):
                for j, part in enumerate(cand.content.parts):
                    print(f"  Part {j}: text={bool(part.text)}, inline_data={bool(part.inline_data)}")

    return output_path


def generate_all_figures(idea, output_dir, api_key=None, model="gemini-3.1-flash-image-preview", verbose=True):
    """
    Generate all three figure types (method, pipeline, overview) for a paper.

    Returns
    -------
    dict
        {figure_type: file_path_or_None, ...}
    """
    results = {}
    for fig_type in ["method", "pipeline", "overview"]:
        try:
            path = generate_paper_figure(
                idea=idea,
                output_dir=output_dir,
                api_key=api_key,
                model=model,
                figure_type=fig_type,
                verbose=verbose,
            )
            results[fig_type] = path
        except Exception as e:
            print(f"[FigureGen] Failed to generate {fig_type}: {e}")
            results[fig_type] = None

    successful = sum(1 for v in results.values() if v is not None)
    if verbose:
        print(f"\n[FigureGen] Summary: {successful}/{len(results)} figures generated successfully.")

    return results


# ================================================================
# CLI Entry Point
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AI-Scientist Dynamic Paper Figure Generator (NanoBanana)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From an idea JSON file
  python ai_scientist/generate_figure.py --idea-file results/my_paper/idea.json

  # From command-line arguments
  python ai_scientist/generate_figure.py --title "My Paper" --abstract "..." --method "..."

  # Specific figure type
  python ai_scientist/generate_figure.py --idea-file results/idea.json --type pipeline
  python ai_scientist/generate_figure.py --idea-file results/idea.json --type all
        """,
    )
    # --- Input source ---
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--idea-file", type=str, help="Path to idea JSON file")
    input_group.add_argument("--title", type=str, help="Paper title (requires --abstract)")

    # --- Manual input fields ---
    parser.add_argument("--abstract", type=str, default="", help="Paper abstract")
    parser.add_argument("--method", type=str, default="", help="Method description")
    parser.add_argument("--experiment", type=str, default="", help="Experiment notes")

    # --- Configuration ---
    parser.add_argument("--output-dir", type=str, default="results/figures",
                        help="Output directory for generated figures")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Google API key (AQ.xxx or AIza.xxx)")
    parser.add_argument("--type", type=str, default="method",
                        choices=["method", "pipeline", "overview", "all"],
                        help="Figure type to generate")
    parser.add_argument("--model", type=str, default="gemini-3.1-flash-image-preview",
                        choices=["gemini-3.1-flash-image-preview", "gemini-2.5-flash-image"],
                        help="Image generation model")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only build and print the prompt, do not call API")

    args = parser.parse_args()

    # --- Build idea dict ---
    if args.idea_file:
        with open(args.idea_file, "r", encoding="utf-8") as f:
            idea = json.load(f)
    else:
        if not args.abstract:
            parser.error("--abstract is required when using --title")
        idea = {
            "Name": args.title,
            "Title": args.title,
            "Abstract": args.abstract,
            "Method": args.method or args.abstract,
            "Experiment": args.experiment or args.abstract,
        }

    # --- Get API key ---
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    # Dry run: just show the prompt
    if args.dry_run:
        prompt_info = build_dynamic_prompt(
            title=idea.get("Title", idea.get("Name", "")),
            abstract=idea.get("Abstract", ""),
            method_desc=idea.get("Method", idea.get("Abstract", "")),
            experiment_notes=str(idea.get("Experiment", ""))[:400],
            section=args.type if args.type != "all" else "method",
        )
        print("=" * 60)
        print(f"FIGURE TYPE: {args.type}")
        print(f"TITLE: {prompt_info['title']}")
        print(f"FILENAME: {prompt_info['filename']}")
        print("=" * 60)
        print(prompt_info["prompt"])
        return

    # --- Check API key ---
    if not api_key:
        print("ERROR: No API key. Set GOOGLE_API_KEY, GEMINI_API_KEY, or use --api-key.")
        print("Get a free key: https://aistudio.google.com/apikey")
        sys.exit(1)

    # --- Generate ---
    if args.type == "all":
        results = generate_all_figures(idea, args.output_dir, api_key, args.model)
    else:
        path = generate_paper_figure(idea, args.output_dir, api_key, args.model, args.type)
        results = {args.type: path}

    # Final summary
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    for fig_type, path in results.items():
        status = f"SAVED: {path}" if path else "FAILED"
        print(f"  [{fig_type}] {status}")


if __name__ == "__main__":
    main()
