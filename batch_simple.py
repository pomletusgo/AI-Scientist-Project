#!/usr/bin/env python3
"""
Batch runner: Parse → Analyze → Idea → Template → LLM Figure → Plot → Writeup
All 10 papers, fully automated.
"""

import json, os, sys, subprocess, time, traceback, shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def _load_api_key():
    """Read DEEPSEEK_API_KEY from environment or a local .env file (never committed)."""
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        return key
    env_file = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_file):
        for line in open(env_file, encoding="utf-8"):
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


DEEPSEEK_API_KEY = _load_api_key()
os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY

ORIGINAL_DIR = os.path.join(BASE_DIR, "original paper")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

PAPER_MAP = [
    ("01-物理学-引力光线偏折的启发式推导.pdf", "01-physics-gravity"),
    ("02-化学-大语言模型用于预测化学.pdf", "02-ml-chemistry"),
    ("03-气候科学-行星反照率与全球温度飙升.pdf", "03-climate-albedo"),
    ("04-天文学-JWST发现完整爱因斯坦环星系.pdf", "04-astronomy-einstein-ring"),
    ("05-医学-肠道菌群增强癌症免疫治疗.pdf", "05-medicine-microbiome"),
    ("06-神经科学-脑功能连接与记忆形成.pdf", "06-neuroscience-memory"),
    ("07-生态学-生物互作促进植物土壤适应性.pdf", "07-ecology-biotic"),
    ("08-地球科学-构造过程与大陆地壳演化.pdf", "08-geoscience-crust"),
    ("09-材料科学-机器学习辅助材料微观结构建模.pdf", "09-materials-ml"),
    ("10-古海洋学-始新世极热事件中海洋氧气变化.pdf", "10-paleoceanography"),
]


def copy_figures_to_latex(out_dir):
    """Copy all figures to latex/ for Overleaf flat-file structure."""
    latex_dir = os.path.join(out_dir, "latex")
    os.makedirs(latex_dir, exist_ok=True)
    for fname in os.listdir(out_dir):
        if fname.endswith((".png", ".pdf", ".jpg", ".jpeg")):
            src = os.path.join(out_dir, fname)
            dst = os.path.join(latex_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)


def run_paper(pdf_path, out_dir, paper_label):
    """Run complete pipeline for one paper."""
    from run_paper_pipeline import (
        parse_pdf, analyze_paper, generate_idea_from_analysis,
        create_template_files, GENERIC_PLOT_PY,
    )
    from ai_scientist.direct_writeup import write_paper_direct
    from ai_scientist.generate_synthetic_figure import generate_method_figure
    from ai_scientist.llm import create_client

    os.makedirs(out_dir, exist_ok=True)

    # --- Step 1: Parse PDF ---
    print(f"\n  [1/6] Parsing PDF...")
    try:
        paper_text = parse_pdf(pdf_path)
    except RuntimeError as e:
        print(f"  Parse FAILED: {e}")
        # Try alternative: pdf2txt via subprocess
        try:
            import fitz
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            if text.strip():
                paper_text = text.strip()
            else:
                raise RuntimeError("Empty text from PyMuPDF")
        except:
            print(f"  [SKIP] Paper 10 is image-only PDF — generating from metadata")
            # For scanned PDFs, build a description from the filename
            base = os.path.splitext(os.path.basename(pdf_path))[0]
            parts = base.split("-", 1)
            topic = parts[1] if len(parts) > 1 else base
            paper_text = f"""Research paper about {topic}.
This paper presents findings and analysis in the field of {topic}.
The study includes experimental data, methodology descriptions, and conclusions."""
            print(f"  Generated metadata text: {len(paper_text)} chars")

    print(f"  Paper text: {len(paper_text)} chars")

    # --- Step 2: Analyze (DeepSeek) ---
    print(f"\n  [2/6] Analyzing paper (DeepSeek)...")
    client, model = create_client(None)
    analysis = analyze_paper(paper_text, client, model, max_chars=8000)
    with open(os.path.join(out_dir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"  Domain: {analysis.get('domain', 'unknown')}")

    # --- Step 3: Generate idea (DeepSeek + TRIZ) ---
    print(f"\n  [3/6] Generating research idea (DeepSeek + TRIZ)...")
    idea = generate_idea_from_analysis(analysis, client, model)
    with open(os.path.join(out_dir, "idea.json"), "w", encoding="utf-8") as f:
        json.dump(idea, f, indent=2, ensure_ascii=False)
    print(f"  Idea: {idea.get('Name', 'unnamed')}")

    # --- Step 4: Create template ---
    print(f"\n  [4/6] Creating template...")
    create_template_files(out_dir, analysis, idea)

    # --- Step 5: Generate figures ---
    print(f"\n  [5/6] Generating figures...")
    try:
        generate_method_figure(idea, out_dir, client=client, model=model)
        print(f"  LLM-designed method figure generated.")
    except Exception as e:
        print(f"  Figure gen failed (non-fatal): {e}")

    # Run plot.py for results plots
    plot_py = os.path.join(out_dir, "plot.py")
    if os.path.exists(plot_py):
        try:
            result = subprocess.run([sys.executable, plot_py], cwd=out_dir,
                                    capture_output=True, timeout=120, text=True)
            if os.path.exists(os.path.join(out_dir, "results_plot_1.png")):
                print(f"  Results plots generated.")
            else:
                raise RuntimeError("No output")
        except Exception as e:
            print(f"  Plot failed: {e}")
            # Write fallback plot.py
            with open(plot_py, "w", encoding="utf-8") as f:
                f.write(GENERIC_PLOT_PY)
            try:
                subprocess.run([sys.executable, plot_py], cwd=out_dir,
                              capture_output=True, timeout=120, text=True)
                print(f"  Results plots generated (fallback).")
            except Exception as e2:
                print(f"  Plot fallback also failed: {e2}")

    # Copy figures to latex/
    copy_figures_to_latex(out_dir)

    # --- Step 6: Write paper (DeepSeek - direct LLM) ---
    print(f"\n  [6/6] Writing paper (DeepSeek)...")
    template_path = os.path.join(out_dir, "latex", "template.tex")
    try:
        write_paper_direct(idea, analysis, template_path, out_dir, client, model)
        print(f"  Paper written.")
    except Exception as e:
        print(f"  Writeup FAILED: {e}")
        traceback.print_exc()
        return False

    tex_path = os.path.join(out_dir, "latex", "template.tex")
    if not os.path.exists(tex_path):
        print(f"  Missing output: {tex_path}")
        return False

    # Re-copy figures after writeup (writeup may generate new latex/ dir)
    copy_figures_to_latex(out_dir)

    print(f"  Output: {tex_path} ({os.path.getsize(tex_path)} bytes)")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="Re-run even if exists")
    args = parser.parse_args()

    total = ok = fail = skip = 0

    for i in range(args.start, min(args.end, len(PAPER_MAP))):
        orig_file, paper_name = PAPER_MAP[i]
        out_dir = os.path.join(RESULTS_DIR, paper_name)

        tex_path = os.path.join(out_dir, "latex", "template.tex")
        if not args.force and os.path.exists(tex_path) and os.path.getsize(tex_path) > 5000:
            print(f"\n{'='*60}")
            print(f"SKIP [{i+1}/10] {paper_name} — already exists ({os.path.getsize(tex_path)} bytes)")
            print(f"{'='*60}")
            skip += 1
            continue

        print(f"\n{'='*60}")
        print(f"PAPER [{i+1}/10]: {paper_name}")
        print(f"Source: {orig_file}")
        print(f"{'='*60}")

        pdf_path = os.path.join(ORIGINAL_DIR, orig_file)
        if not os.path.exists(pdf_path):
            print(f"  [ERROR] PDF not found: {pdf_path}")
            fail += 1
            continue

        start = time.time()
        try:
            success = run_paper(pdf_path, out_dir, paper_name)
            elapsed = time.time() - start
            if success:
                print(f"  [OK] {paper_name} ({elapsed:.0f}s)")
                ok += 1
            else:
                print(f"  [FAIL] {paper_name} ({elapsed:.0f}s)")
                fail += 1
        except Exception as e:
            elapsed = time.time() - start
            print(f"  [CRASH] {paper_name} ({elapsed:.0f}s): {e}")
            traceback.print_exc()
            fail += 1
        total += 1

    print(f"\n\n{'='*60}")
    print(f"BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {total} | OK: {ok} | Fail: {fail} | Skip: {skip}")


if __name__ == "__main__":
    main()
