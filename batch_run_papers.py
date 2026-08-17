#!/usr/bin/env python3
"""
Batch run all 10 papers from original paper/ through the pipeline.
Generates synthetic figures programmatically (no API dependency).
"""

import os, sys, json, subprocess, time, traceback

ORIGINAL_PAPER_DIR = r"d:\AI-Scientist\original paper"
PIPELINE_SCRIPT = r"d:\AI-Scientist\run_paper_pipeline.py"
RESULTS_BASE = r"d:\AI-Scientist\results"

PAPER_NAMES = [
    "01-physics-gravity",
    "02-ml-chemistry",
    "03-climate-albedo",
    "04-astronomy-einstein-ring",
    "05-medicine-microbiome",
    "06-neuroscience-memory",
    "07-ecology-biotic",
    "08-geoscience-crust",
    "09-materials-ml",
    "10-paleoceanography",
]

# Map original filenames to paper names
ORIGINAL_FILES = sorted([
    f for f in os.listdir(ORIGINAL_PAPER_DIR) if f.endswith(".pdf")
])

assert len(ORIGINAL_FILES) >= 10, f"Expected 10 PDFs, found {len(ORIGINAL_FILES)}: {ORIGINAL_FILES}"

# Create mapping
PAPER_MAP = list(zip(ORIGINAL_FILES, PAPER_NAMES))

def run_one(orig_file, paper_name):
    """Run pipeline for one paper."""
    out_dir = os.path.join(RESULTS_BASE, paper_name)

    # Skip if already completed (template.tex exists)
    output_tex = os.path.join(out_dir, "latex", "template.tex")
    if os.path.exists(output_tex) and os.path.getsize(output_tex) > 10000:
        print(f"\n{'='*60}")
        print(f"SKIPPING {paper_name} — already completed ({output_tex})")
        print(f"{'='*60}")
        return True

    print(f"\n{'='*60}")
    print(f"RUNNING {paper_name} from {orig_file}")
    print(f"{'='*60}")

    pdf_path = os.path.join(ORIGINAL_PAPER_DIR, orig_file)
    cmd = [
        sys.executable, PIPELINE_SCRIPT,
        "--paper", pdf_path,
        "--output", out_dir,
        "--experiments",
        "--no-review",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=r"d:\AI-Scientist",
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout per paper
            encoding="utf-8",
            errors="replace",
        )
        stdout = result.stdout
        stderr = result.stderr

        # Save log
        log_dir = os.path.join(out_dir, "batch_log.txt")
        os.makedirs(out_dir, exist_ok=True)
        with open(log_dir, "w", encoding="utf-8") as f:
            f.write("=== STDOUT ===\n")
            f.write(stdout)
            f.write("\n=== STDERR ===\n")
            f.write(stderr)

        if result.returncode != 0:
            print(f"  [FAIL] Return code: {result.returncode}")
            print(f"  Last 500 chars of stderr:\n{stderr[-500:]}")
            return False

        # Check output exists
        if os.path.exists(output_tex):
            size = os.path.getsize(output_tex)
            print(f"  [OK] Generated: {output_tex} ({size} bytes)")
            return True
        else:
            # Check for .tex files in output dir
            tex_files = [f for f in os.listdir(out_dir) if f.endswith(".tex")]
            if tex_files:
                print(f"  [OK] Generated tex: {tex_files}")
                return True
            print(f"  [WARN] No .tex output found")
            print(f"  Last 500 chars of stdout:\n{stdout[-500:]}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {paper_name} exceeded 30 minutes")
        return False
    except Exception as e:
        print(f"  [ERROR] {paper_name}: {e}")
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("BATCH RUN: 10 Papers")
    print("=" * 60)
    print(f"Original papers: {ORIGINAL_PAPER_DIR}")
    print(f"Results: {RESULTS_BASE}")
    print(f"Papers to process: {len(PAPER_MAP)}")
    print()

    results = {}
    for i, (orig_file, paper_name) in enumerate(PAPER_MAP):
        print(f"\n{'#'*60}")
        print(f"# PAPER {i+1}/{len(PAPER_MAP)}: {paper_name}")
        print(f"{'#'*60}")

        start = time.time()
        success = run_one(orig_file, paper_name)
        elapsed = time.time() - start

        results[paper_name] = {
            "success": success,
            "time": elapsed,
            "orig_file": orig_file,
        }
        print(f"  Time: {elapsed:.0f}s | Success: {success}")

    # Summary
    print(f"\n\n{'='*60}")
    print("BATCH SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for v in results.values() if v["success"])
    failed = sum(1 for v in results.values() if not v["success"])
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    for name, info in results.items():
        status = "✓" if info["success"] else "✗"
        print(f"  [{status}] {name} ({info['time']:.0f}s)")

    # Save summary JSON
    summary_path = os.path.join(RESULTS_BASE, "batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
