#!/usr/bin/env python
"""
AI-Scientist LLM Pipeline Runner
=================================
Runs the LLM-only pipeline (ideation → skip experiments → writeup → review)
on Windows / CPU-only environments.

No GPU required. Uses mock experiment data to bridge between ideation and writeup stages.

Usage:
    # Paper review only (fully standalone)
    python run_llm_pipeline.py --task review --paper path/to/paper.pdf --model deepseek-v4-pro

    # Idea generation only
    python run_llm_pipeline.py --task ideation --experiment nanoGPT --model deepseek-v4-pro --num-ideas 5

    # Full pipeline with mock experiments
    python run_llm_pipeline.py --task pipeline --experiment nanoGPT --model deepseek-v4-pro --num-ideas 2

Models:
    deepseek-v4-pro  : DeepSeek V4 flagship (with reasoning)
    deepseek-v4-flash: DeepSeek V4 lightweight
    deepseek-r1      : DeepSeek R1 (legacy reasoning)
    deepseek-chat    : DeepSeek V2/V3 chat
    deepseek-reasoner: DeepSeek R1 (original name)
"""

import argparse
import json
import os
import os.path as osp
import sys
import shutil
from datetime import datetime

# Add the AI-Scientist directory to path
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from ai_scientist.llm import create_client, AVAILABLE_LLMS, get_response_from_llm, extract_json_between_markers
from ai_scientist.generate_ideas import generate_ideas, check_idea_novelty
from ai_scientist.perform_review import perform_review, load_paper


def run_review(paper_path, model, client_model, client, num_reflections=3, num_reviews_ensemble=3):
    """Run the automated paper review stage."""
    print(f"\n{'='*60}")
    print(f"Running Automated Paper Review")
    print(f"Paper: {paper_path}")
    print(f"Model: {client_model}")
    print(f"Reflections: {num_reflections}, Ensemble: {num_reviews_ensemble}")
    print(f"{'='*60}\n")

    # Load paper text from PDF
    print("Loading paper...")
    paper_text = load_paper(paper_path)
    print(f"Loaded {len(paper_text)} characters of text.\n")

    # Perform review
    print("Starting review...")
    review = perform_review(
        paper_text,
        model=client_model,
        client=client,
        num_reflections=num_reflections,
        num_fs_examples=1,
        num_reviews_ensemble=num_reviews_ensemble,
        temperature=0.75 if num_reviews_ensemble > 1 else 0.1,
    )

    # Save and display results
    output_path = osp.splitext(paper_path)[0] + "_review.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(review, f, indent=4, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("Review Results:")
    print(f"{'='*60}")
    for key in ["Overall", "Soundness", "Presentation", "Contribution", "Confidence", "Decision"]:
        if key in review:
            print(f"  {key}: {review[key]}")
    print(f"\n  Strengths: {review.get('Strengths', 'N/A')}")
    print(f"  Weaknesses: {review.get('Weaknesses', 'N/A')}")
    print(f"\nReview saved to: {output_path}")
    return review


def run_ideation(experiment, model, client_model, client, num_ideas=5, num_reflections=3):
    """Run the idea generation stage."""
    print(f"\n{'='*60}")
    print(f"Running Idea Generation")
    print(f"Experiment Template: {experiment}")
    print(f"Model: {client_model}")
    print(f"Num Ideas: {num_ideas}, Reflections: {num_reflections}")
    print(f"{'='*60}\n")

    base_dir = osp.join("templates", experiment)
    if not osp.exists(base_dir):
        print(f"Error: Template directory {base_dir} not found.")
        print("Available templates: nanoGPT, 2d_diffusion, grokking")
        return None

    # Load prompt and experiment code for context
    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)
    print(f"Domain: {prompt.get('task_description', 'N/A')[:100]}...")

    # Generate ideas
    ideas = generate_ideas(
        base_dir,
        client=client,
        model=client_model,
        skip_generation=False,
        max_num_generations=num_ideas,
        num_reflections=num_reflections,
    )

    print(f"\n{'='*60}")
    print(f"Generated {len(ideas)} ideas:")
    print(f"{'='*60}")
    for i, idea in enumerate(ideas):
        print(f"\n  Idea {i+1}: {idea.get('Name', 'N/A')}")
        print(f"    Title: {idea.get('Title', 'N/A')}")
        print(f"    Interestingness: {idea.get('Interestingness', 'N/A')}/10")
        print(f"    Feasibility: {idea.get('Feasibility', 'N/A')}/10")
        print(f"    Novelty: {idea.get('Novelty', 'N/A')}/10")

    return ideas


def create_mock_experiment_data(idea, template_dir, output_dir):
    """Create mock experiment data to bridge ideation and writeup stages."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(osp.join(output_dir, "latex"), exist_ok=True)
    os.makedirs(osp.join(output_dir, "run_0"), exist_ok=True)

    # Copy template files
    latex_template = osp.join(template_dir, "latex", "template.tex")
    if osp.exists(latex_template):
        shutil.copy(latex_template, osp.join(output_dir, "latex", "template.tex"))
    else:
        print(f"Warning: No LaTeX template found at {latex_template}")

    # Copy experiment.py and plot.py as reference
    for fname in ["experiment.py", "plot.py"]:
        src = osp.join(template_dir, fname)
        if osp.exists(src):
            shutil.copy(src, osp.join(output_dir, fname))

    # Create mock final_info.json (simulated experiment results)
    mock_results = {
        "train_loss": {"means": 2.35, "std": 0.12},
        "val_loss": {"means": 2.68, "std": 0.15},
        "test_loss": {"means": 2.71, "std": 0.14},
        "perplexity": {"means": 15.02, "std": 2.1},
    }
    with open(osp.join(output_dir, "run_0", "final_info.json"), "w") as f:
        json.dump(mock_results, f, indent=4)

    # Create notes.txt with experiment description
    notes = f"""# Title: {idea.get('Title', 'Untitled')}
# Experiment description: {idea.get('Experiment', 'No description provided.')}
## Run 0: Baseline
Results: {json.dumps(mock_results)}
Description: Baseline results from the vanilla model configuration.
## Run 1: Proposed Method
Results: train_loss=2.12, val_loss=2.45, test_loss=2.48, perplexity=11.93
Description: The proposed method reduces test loss by 8.5% compared to baseline.
"""
    with open(osp.join(output_dir, "notes.txt"), "w") as f:
        f.write(notes)

    # Copy the idea as ideas.json
    with open(osp.join(output_dir, "ideas.json"), "w") as f:
        json.dump([idea], f, indent=4)

    print(f"Mock experiment data created at: {output_dir}")
    print(f"  - notes.txt (experiment description + results)")
    print(f"  - run_0/final_info.json (baseline metrics)")
    print(f"  - experiment.py (reference code)")
    print(f"  - latex/template.tex (paper template)")
    return output_dir


def run_simple_writeup(idea, folder_name, model, client_model, client):
    """Generate a simple paper writeup using direct LLM calls (no Aider required)."""
    print(f"\n{'='*60}")
    print(f"Running Paper Writeup (Simplified)")
    print(f"Folder: {folder_name}")
    print(f"Model: {client_model}")
    print(f"{'='*60}\n")

    # Read the notes and template
    notes_path = osp.join(folder_name, "notes.txt")
    template_path = osp.join(folder_name, "latex", "template.tex")

    with open(notes_path, "r") as f:
        notes = f.read()

    if osp.exists(template_path):
        with open(template_path, "r") as f:
            template = f.read()
    else:
        template = r"\documentclass{article}"  # fallback

    # Generate a simple writeup via direct LLM
    writeup_system = """You are an AI researcher writing a scientific paper.
You have completed experiments and need to write up the results.
Be precise, use academic style, and only report results that are in the notes."""

    writeup_prompt = f"""Based on the following experiment notes, write a complete scientific paper section.

Experiment Notes:
```
{notes}
```

Please write:
1. A concise Title
2. An Abstract (200 words max)
3. Key Results summary

Format as a simple markdown document. Only report results that are explicitly in the notes.
Do not fabricate or extrapolate beyond the data provided.
"""

    print("Generating paper writeup via LLM...")
    text, msg_history = get_response_from_llm(
        writeup_prompt,
        client=client,
        model=client_model,
        system_message=writeup_system,
        temperature=0.5,
    )

    # Save the generated writeup
    writeup_path = osp.join(folder_name, "generated_paper.md")
    with open(writeup_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"\nWriteup generated and saved to: {writeup_path}")
    print(f"\n{'='*60}")
    print("Generated Paper Preview:")
    print(f"{'='*60}")
    print(text[:2000])
    if len(text) > 2000:
        print(f"\n... ({len(text) - 2000} more characters)")

    return writeup_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI-Scientist LLM Pipeline (CPU/GPU-free)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_llm_pipeline.py --task review --paper attention.pdf --model deepseek-v4-pro
  python run_llm_pipeline.py --task ideation --experiment nanoGPT --model deepseek-v4-flash --num-ideas 3
  python run_llm_pipeline.py --task pipeline --experiment nanoGPT --model deepseek-v4-pro --num-ideas 2
        """,
    )
    parser.add_argument(
        "--task", type=str, required=True,
        choices=["review", "ideation", "pipeline", "mock-data"],
        help="Which pipeline stage to run",
    )
    parser.add_argument(
        "--model", type=str, default="deepseek-v4-pro",
        choices=AVAILABLE_LLMS,
        help="LLM model to use",
    )
    parser.add_argument(
        "--paper", type=str, default=None,
        help="Path to PDF paper (for review task)",
    )
    parser.add_argument(
        "--experiment", type=str, default="nanoGPT",
        choices=["nanoGPT", "2d_diffusion", "grokking"],
        help="Experiment template to use",
    )
    parser.add_argument(
        "--num-ideas", type=int, default=3,
        help="Number of ideas to generate",
    )
    parser.add_argument(
        "--num-reflections", type=int, default=3,
        help="Number of reflection rounds for LLM",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory for results",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create LLM client
    print(f"Initializing {args.model} client...")
    client, client_model = create_client(args.model)
    print(f"Client created: {client_model}\n")

    if args.task == "review":
        if not args.paper:
            print("Error: --paper is required for review task")
            sys.exit(1)
        review = run_review(
            args.paper, args.model, client_model, client,
            num_reflections=args.num_reflections,
            num_reviews_ensemble=3,
        )

    elif args.task == "ideation":
        ideas = run_ideation(
            args.experiment, args.model, client_model, client,
            num_ideas=args.num_ideas,
            num_reflections=args.num_reflections,
        )

    elif args.task == "pipeline":
        # Step 1: Ideation
        ideas = run_ideation(
            args.experiment, args.model, client_model, client,
            num_ideas=args.num_ideas,
            num_reflections=args.num_reflections,
        )
        if not ideas:
            return

        # For each idea, create mock data and generate writeup
        for i, idea in enumerate(ideas):
            print(f"\n{'#'*60}")
            print(f"# Processing idea {i+1}/{len(ideas)}: {idea.get('Name', 'unknown')}")
            print(f"{'#'*60}")

            # Step 2: Create mock experiment data
            template_dir = osp.join("templates", args.experiment)
            idea_dir = osp.join(args.output_dir, f"idea_{i+1}_{idea.get('Name', 'unknown')}")
            create_mock_experiment_data(idea, template_dir, idea_dir)

            # Step 3: Generate simple writeup (no Aider needed)
            writeup_path = run_simple_writeup(idea, idea_dir, args.model, client_model, client)

        print(f"\n{'='*60}")
        print(f"Pipeline complete! Results in: {args.output_dir}")
        print(f"{'='*60}")

    elif args.task == "mock-data":
        # Just create mock experiment data for an existing idea
        template_dir = osp.join("templates", args.experiment)
        idea = {
            "Name": "test_experiment",
            "Title": "Test Experiment: Improving Model Performance",
            "Experiment": "Modify the attention mechanism to improve performance.",
            "Interestingness": 7,
            "Feasibility": 8,
            "Novelty": 6,
        }
        output_dir = osp.join(args.output_dir, "mock_experiment")
        create_mock_experiment_data(idea, template_dir, output_dir)


if __name__ == "__main__":
    main()
