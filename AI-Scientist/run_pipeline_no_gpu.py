#!/usr/bin/env python3
"""AI-Scientist GPU-Free Pipeline Runner - Skips GPU experiment execution."""
import argparse, json, os, os.path as osp, shutil, sys, random
from datetime import datetime
from ai_scientist.generate_ideas import generate_ideas, check_idea_novelty
from ai_scientist.llm import create_client, AVAILABLE_LLMS
from ai_scientist.perform_review import perform_review, load_paper
from ai_scientist.perform_writeup import perform_writeup
from ai_scientist.mock_experiments import generate_mock_results, save_mock_run

def parse_arguments():
    p = argparse.ArgumentParser(description="AI-Scientist GPU-Free Pipeline")
    p.add_argument("--experiment", type=str, default="nanoGPT", choices=["nanoGPT", "2d_diffusion", "grokking"])
    p.add_argument("--model", type=str, default="deepseek-v4-flash", choices=AVAILABLE_LLMS)
    p.add_argument("--review-model", type=str, default="deepseek-v4-pro", choices=AVAILABLE_LLMS)
    p.add_argument("--num-ideas", type=int, default=10)
    p.add_argument("--skip-ideation", action="store_true")
    p.add_argument("--skip-novelty-check", action="store_true")
    p.add_argument("--mock-runs", type=int, default=3)
    p.add_argument("--engine", type=str, default="semanticscholar", choices=["semanticscholar", "openalex"])
    return p.parse_args()
import random

def print_time():
    print(f"[{datetime.now().strftime('%H:%M:%S')}]", end=" ")

def check_dependencies():
    issues = []
    if not os.environ.get("DEEPSEEK_API_KEY"):
        issues.append("DEEPSEEK_API_KEY not set - set with: export DEEPSEEK_API_KEY='your-key'")
    if shutil.which("pdflatex") is None:
        issues.append("pdflatex not found - LaTeX PDF generation will fail")
    if issues:
        print("WARNING: Dependency issues:")
        for i in issues:
            print(f"  - {i}")
        if "DEEPSEEK_API_KEY" in str(issues[0]):
            return False
    return True

def run_gpu_free_pipeline(args):
    print("=" * 60)
    print("AI-SCIENTIST GPU-FREE PIPELINE")
    print(f"Template: {args.experiment}")
    print(f"Model: {args.model}")
    print(f"Review Model: {args.review_model}")
    print("=" * 60)
    
    client, client_model = create_client(args.model)
    review_client, review_model = create_client(args.review_model)
    
    base_dir = osp.join("templates", args.experiment)
    results_dir = osp.join("results", args.experiment)
    os.makedirs(results_dir, exist_ok=True)
    
    print_time(); print("STAGE 1: Generating research ideas...")
    ideas = generate_ideas(base_dir, client=client, model=client_model,
                          skip_generation=args.skip_ideation,
                          max_num_generations=args.num_ideas, num_reflections=3)
    
    if not args.skip_novelty_check:
        print_time(); print(f"STAGE 2: Checking novelty of {len(ideas)} ideas...")
        ideas = check_idea_novelty(ideas, base_dir=base_dir, client=client,
                                   model=client_model, engine=args.engine)
    else:
        print_time(); print("STAGE 2: Skipping novelty check")
    
    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(ideas, f, indent=4)
    
    novel_ideas = [i for i in ideas if i.get("novel", True)]
    print_time(); print(f"Novel ideas: {len(novel_ideas)}/{len(ideas)}")
    
    for idx, idea in enumerate(novel_ideas):
        print(); print("=" * 40)
        print_time(); print(f"IDEA {idx+1}/{len(novel_ideas)}: {idea['Name']}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        idea_name = f"{timestamp}_{idea['Name']}"
        folder_name = osp.join(results_dir, idea_name)
        
        if osp.exists(folder_name):
            print(f"  Folder exists, skipping.")
            continue
        
        shutil.copytree(base_dir, folder_name, dirs_exist_ok=True)
        
        print_time(); print("  Generating mock experiment results...")
        mock_runs = generate_mock_results(
            idea_name=idea.get("Name", "experiment"),
            num_runs=args.mock_runs,
            base_metric=random.uniform(0.3, 0.7),
            improvement_range=(0.02, 0.18),
        )
        save_mock_run(folder_name, idea.get("Name", "experiment"), args.mock_runs)
        
        notes_path = osp.join(folder_name, "notes.txt")
        with open(notes_path, "w", encoding="utf-8") as f:
            f.write(f"# Title: {idea.get('Title', 'Untitled')}\n")
            f.write(f"# Experiment: {idea.get('Experiment', 'No description')}\n")
            for run_idx in range(args.mock_runs):
                run_file = osp.join(folder_name, f"run_{run_idx+1}", "final_info.json")
                if osp.exists(run_file):
                    with open(run_file) as rf:
                        run_data = json.load(rf)
                    f.write(f"## Run {run_idx+1}\n")
                    f.write(f"Results: {json.dumps(run_data['means'])}\n\n")
        
        print_time(); print("  Writing LaTeX paper...")
        try:
            from aider.coders import Coder
            from aider.models import Model
            from aider.io import InputOutput
            
            writeup_file = osp.join(folder_name, "latex", "template.tex")
            exp_file = osp.join(folder_name, "experiment.py")
            fnames = [exp_file, writeup_file, notes_path]
            
            io = InputOutput(yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt")
            
            if "deepseek" in args.model:
                if "reasoner" in args.model or "r1" in args.model:
                    main_model = Model("deepseek/deepseek-reasoner")
                else:
                    main_model = Model("deepseek/deepseek-chat")
            else:
                main_model = Model(args.model)
            
            coder = Coder.create(main_model=main_model, fnames=fnames, io=io,
                                stream=False, use_git=False, edit_format="diff")
            perform_writeup(idea, folder_name, coder, cite_client=client,
                          cite_model=client_model, engine=args.engine)
            print_time(); print("  Paper written successfully!")
        except Exception as e:
            print_time(); print(f"  Paper writing failed: {e}")
            import traceback; traceback.print_exc()
            continue
        
        print_time(); print("  Performing automated review...")
        try:
            paper_pdf = osp.join(folder_name, f"{idea['Name']}.pdf")
            if osp.exists(paper_pdf):
                paper_text = load_paper(paper_pdf)
                review = perform_review(paper_text, model=review_model,
                                       client=review_client, num_reflections=3,
                                       num_fs_examples=1, num_reviews_ensemble=3,
                                       temperature=0.1)
                review_path = osp.join(folder_name, "review.json")
                with open(review_path, "w") as f:
                    json.dump(review, f, indent=4)
                print(f"    Overall Score: {review.get('Overall', 'N/A')}/10")
                print(f"    Decision: {review.get('Decision', 'N/A')}")
                print(f"    Originality: {review.get('Originality', 'N/A')}/4")
                print(f"    Quality: {review.get('Quality', 'N/A')}/4")
            else:
                print_time(); print("  PDF not found, skipping review.")
        except Exception as e:
            print_time(); print(f"  Review failed: {e}")
        
        print_time(); print(f"  Completed: {idea['Name']}")
    
    print(); print("=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Results saved to: {results_dir}")
    print("=" * 60)

if __name__ == "__main__":
    args = parse_arguments()
    if not check_dependencies():
        sys.exit(1)
    run_gpu_free_pipeline(args)
