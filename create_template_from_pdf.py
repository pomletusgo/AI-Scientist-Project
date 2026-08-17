#!/usr/bin/env python3
"""
PDF → AI-Scientist Template: Automatic extraction and template generation.

Takes a PDF/TXT paper, extracts structured research info via LLM,
and generates a complete AI-Scientist template ready to run.

Usage:
  python create_template_from_pdf.py --pdf paper.pdf --topic_name my_topic
  python create_template_from_pdf.py --pdf paper.pdf --topic_name my_topic --data my_data.csv
  python create_template_from_pdf.py --txt "paper text..." --topic_name quick_test
"""

import os, sys, json, shutil, argparse, re, requests

# ============================================================
# LLM
# ============================================================
DS_KEY = os.environ.get("DEEPSEEK_API_KEY", "")


def _get_client():
    if not DS_KEY:
        print("[ERROR] DEEPSEEK_API_KEY not set. Run: export DEEPSEEK_API_KEY=sk-...")
        sys.exit(1)
    from openai import OpenAI
    return OpenAI(api_key=DS_KEY, base_url="https://api.deepseek.com"), "deepseek-v4-pro"


def ask(sys_msg, usr_msg, temp=0.3, mt=2000):
    client, model = _get_client()
    for _ in range(3):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys_msg},
                          {"role": "user", "content": usr_msg}],
                temperature=temp, max_tokens=mt,
            )
            t = r.choices[0].message.content
            if t and len(t.strip()) > 50:
                return t
        except Exception as e:
            print(f"  LLM error: {e}")
        import time; time.sleep(2)
    return ""


def extract_json(text):
    for p in [r'```json\s*(.*?)\s*```', r'\{.*\}']:
        m = re.search(p, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1) if 'json' in p else m.group(0))
            except:
                pass
    return {}


# ============================================================
# PDF Reading
# ============================================================
def read_pdf(path):
    """Read PDF and return plain text."""
    # Try pymupdf4llm first (best quality)
    try:
        import pymupdf4llm
        text = pymupdf4llm.to_markdown(path)
        print(f"  Read {len(text)} chars via pymupdf4llm")
        return text[:15000]
    except ImportError:
        pass

    # Fallback to pymupdf
    try:
        import pymupdf
        doc = pymupdf.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
        print(f"  Read {len(text)} chars via pymupdf")
        return text[:15000]
    except ImportError:
        pass

    # Last resort: pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        print(f"  Read {len(text)} chars via pypdf")
        return text[:15000]
    except ImportError:
        print("[ERROR] No PDF reader available. Install pymupdf, pymupdf4llm, or pypdf.")
        sys.exit(1)


# ============================================================
# Phase 1: Extract paper info
# ============================================================
def extract_paper_info(content):
    """Extract structured info from paper content via LLM."""
    print("\n[Phase 1] Extracting paper structure...")

    resp = ask(
        "Extract structured research information from this paper. Output ONLY valid JSON.",
        f"""Analyze this paper and return JSON with these fields:

{{
  "title": "paper title (string)",
  "domain": "research field / subfield (string, e.g. 'natural language processing', 'computer vision')",
  "problem": "core problem the paper addresses (1-2 sentences)",
  "method": "proposed method / approach (1-2 sentences, the key idea)",
  "findings": ["key finding 1", "key finding 2", ...] (list of strings, at least 2),
  "limitations": ["limitation 1", "limitation 2", ...] (list of strings, at least 2),
  "datasets": ["dataset name 1", ...] (list of strings, datasets used in the paper),
  "open_questions": ["unanswered question or future direction 1", ...] (list of strings, at least 2),
  "keywords": ["keyword1", "keyword2", ...] (list of 3-5 keywords for literature search)
}}

Paper content:
{content[:12000]}""",
        temp=0.3, mt=2000,
    )

    info = extract_json(resp)
    if not info:
        print("[WARN] JSON extraction failed, using raw response")
        info = {"raw": resp[:500]}

    for key in ["title", "domain", "problem", "method"]:
        print(f"  {key}: {str(info.get(key, 'N/A'))[:100]}")
    return info


# ============================================================
# Phase 2: Generate seed ideas from open questions
# ============================================================
def generate_seed_ideas(info):
    """Convert open questions / limitations into concrete seed ideas."""
    print("\n[Phase 2] Generating seed ideas from open questions...")

    gaps = info.get("open_questions", []) + info.get("limitations", [])
    gaps_text = "\n".join(f"- {g}" for g in gaps[:6])

    resp = ask(
        "Generate concrete, implementable ML research ideas. Output ONLY valid JSON.",
        f"""Based on these research gaps from a paper:

{gaps_text}

Generate 2-3 concrete research ideas. Each idea should be a specific experiment
that can be implemented by modifying a PyTorch MLP training script.
The ideas should address the gaps using new approaches.

Output JSON:
[
  {{
    "Name": "lowercase_underscore_name",
    "Title": "Descriptive Paper Title",
    "Experiment": "Detailed implementation plan. What to modify in the code, what to compare, what metrics to report.",
    "Interestingness": 7,
    "Feasibility": 8,
    "Novelty": 7
  }}
]

Be realistic with ratings (1-10). An idea that directly addresses a stated gap
and proposes a clear experiment gets higher scores.""",
        temp=0.7, mt=2000,
    )

    ideas = extract_json(resp)
    if isinstance(ideas, dict):
        ideas = [ideas]
    if not ideas or not isinstance(ideas, list):
        print("[WARN] Could not generate seed ideas, using defaults")
        ideas = [{
            "Name": "improved_architecture",
            "Title": f"Improved Architecture for {info.get('domain', 'ML')}",
            "Experiment": "Modify MLP with additional layers/regularization. Compare with baseline.",
            "Interestingness": 6, "Feasibility": 8, "Novelty": 5,
        }]

    for i, idea in enumerate(ideas):
        print(f"  [{i+1}] {idea.get('Name')}: {idea.get('Title', '')[:80]}")
    return ideas


# ============================================================
# Phase 2.5: Auto-download real dataset
# ============================================================
def find_and_download_dataset(info, template_dir):
    """Search HuggingFace + OpenML + Kaggle for real datasets matching the topic."""
    keywords = info.get("keywords", [])
    domain = info.get("domain", "")
    queries = keywords + [domain] + [info.get("problem", "")[:80]]

    print("\n[Phase 2.5] Searching for real datasets...")
    all_datasets = []

    # --- HuggingFace ---
    for q in queries[:3]:
        try:
            r = requests.get(
                "https://huggingface.co/api/datasets",
                params={"search": q[:60], "sort": "downloads", "direction": -1, "limit": 3},
                timeout=15,
            )
            for d in r.json():
                all_datasets.append({
                    "id": d.get("id", ""),
                    "name": d.get("id", "").split("/")[-1] if "/" in d.get("id", "") else d.get("id", ""),
                    "downloads": d.get("downloads", 0),
                    "description": (d.get("description", "") or "")[:150],
                    "source": "huggingface",
                })
        except Exception as e:
            print(f"  HuggingFace ({q[:30]}...): {e}")

    # --- OpenML ---
    for q in queries[:2]:
        try:
            r = requests.get(
                "https://www.openml.org/api/v1/json/data/list",
                params={"tag": q[:30], "limit": 3}, timeout=15,
            )
            for d in r.json().get("data", {}).get("dataset", [])[:3]:
                all_datasets.append({
                    "id": f"openml:{d.get('did', '')}",
                    "name": d.get("name", ""),
                    "description": (d.get("description", "") or "")[:150],
                    "source": "openml",
                    "instances": d.get("NumberOfInstances", "?"),
                })
        except Exception as e:
            print(f"  OpenML ({q[:30]}...): {e}")

    if not all_datasets:
        print("  No datasets found. You will need to provide a CSV manually.")
        return None

    # Sort by downloads and pick best
    all_datasets.sort(key=lambda x: x.get("downloads", 0), reverse=True)
    best = all_datasets[0]
    print(f"  Found {len(all_datasets)} datasets. Best: {best['name']} ({best['source']})")

    # Try to download
    csv_path = os.path.join(template_dir, "data.csv")
    if best["source"] == "huggingface" and "/" in best["id"]:
        try:
            from datasets import load_dataset
            print(f"  Downloading {best['id']} from HuggingFace...")
            data = load_dataset(best["id"], split="train", trust_remote_code=True)
            df = data.to_pandas().head(10000)
            df.to_csv(csv_path, index=False)
            print(f"  Downloaded: {len(df)} rows × {len(df.columns)} cols → {csv_path}")
            return csv_path
        except Exception as e:
            print(f"  HuggingFace download failed: {e}")

    if best["source"] == "openml":
        try:
            from sklearn.datasets import fetch_openml
            oid = int(best["id"].replace("openml:", ""))
            print(f"  Downloading OpenML dataset {oid}...")
            data = fetch_openml(data_id=oid, as_frame=True, parser="auto")
            df = data.frame.head(10000)
            df.to_csv(csv_path, index=False)
            print(f"  Downloaded: {len(df)} rows × {len(df.columns)} cols → {csv_path}")
            return csv_path
        except Exception as e:
            print(f"  OpenML download failed: {e}")

    print(f"  Auto-download failed. Please manually get the dataset and save as {csv_path}")
    print(f"  Source: {best.get('url', best.get('id', ''))}")
    return None


# ============================================================
# Phase 3: Build template
# ============================================================
def build_template(topic_name, info, seed_ideas, data_path=None):
    """Create the complete template directory."""
    template_dir = os.path.join("templates", topic_name)
    os.makedirs(template_dir, exist_ok=True)
    os.makedirs(os.path.join(template_dir, "latex"), exist_ok=True)
    os.makedirs(os.path.join(template_dir, "run_0"), exist_ok=True)

    domain = info.get("domain", "machine learning")
    problem = info.get("problem", "improving model performance")
    method = info.get("method", "novel architectural modifications")
    keywords = ", ".join(info.get("keywords", [domain]))

    # --- prompt.json ---
    prompt = {
        "system": (
            f"You are an ambitious AI researcher in {domain}. "
            f"You are looking to publish a paper that will contribute significantly to the field. "
            f"Your expertise covers {keywords}."
        ),
        "task_description": (
            f"You are given a PyTorch training script for tabular data in the domain of {domain}. "
            f"The core research problem is: {problem}. "
            f"Prior work has explored: {method}. "
            f"Your goal is to propose improvements that advance beyond existing approaches. "
            f"You can modify: model architecture, training methodology, regularization, "
            f"loss functions, data preprocessing, or evaluation protocols."
        ),
    }
    _write_json(os.path.join(template_dir, "prompt.json"), prompt)
    print(f"  prompt.json")

    # --- seed_ideas.json ---
    _write_json(os.path.join(template_dir, "seed_ideas.json"), seed_ideas)
    print(f"  seed_ideas.json ({len(seed_ideas)} ideas)")

    # --- LaTeX template ---
    title = f"Advancing {domain.title()}: A Novel Approach to {problem[:60]}"
    _write_latex_template(template_dir, title)
    print(f"  latex/template.tex")

    # --- Generate paper.bib from keywords ---
    bib_entries = _generate_bib_from_keywords(info)
    _write_bib(os.path.join(template_dir, "paper.bib"), bib_entries)
    print(f"  paper.bib ({len(bib_entries)} entries)")

    # --- Copy experiment.py and plot.py from generic_tabular ---
    generic = os.path.join("templates", "generic_tabular")
    for fname in ["experiment.py", "plot.py"]:
        src = os.path.join(generic, fname)
        dst = os.path.join(template_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"  {fname} (copied from generic_tabular)")
        else:
            print(f"  [WARN] {fname} not found in generic_tabular, please provide one")

    # --- Data ---
    if data_path and os.path.exists(data_path):
        shutil.copy(data_path, os.path.join(template_dir, "data.csv"))
        print(f"  data.csv (copied from {data_path})")
    else:
        print(f"  [NOTE] No --data provided. Put your CSV as {template_dir}/data.csv before running.")

    # --- Baseline placeholder (only if not already populated) ---
    finfo = os.path.join(template_dir, "run_0", "final_info.json")
    if not os.path.exists(finfo):
        _write_json(finfo, {"_placeholder": True})
        print(f"  run_0/final_info.json (placeholder — run experiment.py to populate)")

    return template_dir


# ============================================================
# Helpers
# ============================================================
def _generate_bib_from_keywords(info):
    """Auto-generate basic BibTeX entries from LLM-extracted paper info."""
    entries = []

    title = info.get("title", "")
    domain = info.get("domain", "")
    keywords = info.get("keywords", [])

    # Entry 1: The original paper itself
    key = re.sub(r"[^a-z0-9]+", "", title.lower()[:30]) if title else "originalpaper"
    entries.append(f"""@article{{{key},
  title   = {{{{{title}}}}},
  author  = {{{See Original Paper}}},
  journal = {{{domain}}},
  year    = {{2024}}
}}""")

    # Entry 2-3: Well-known datasets/tools cited in the paper
    # Pytorch
    entries.append("""@article{paszke2019pytorch,
  title   = {{PyTorch}: An Imperative Style, High-Performance Deep Learning Library},
  author  = {Paszke, Adam and Gross, Sam and Massa, Francisco and others},
  journal = {Advances in Neural Information Processing Systems},
  volume  = {32},
  year    = {2019}
}""")

    # Adam
    entries.append("""@article{kingma2014adam,
  title   = {{Adam}: A Method for Stochastic Optimization},
  author  = {Kingma, Diederik P. and Ba, Jimmy},
  journal = {arXiv preprint arXiv:1412.6980},
  year    = {2014}
}""")

    # Entry 4: Deep learning textbook
    entries.append("""@book{goodfellow2016deep,
  title   = {Deep Learning},
  author  = {Goodfellow, Ian and Bengio, Yoshua and Courville, Aaron},
  publisher = {MIT Press},
  year    = {2016}
}""")

    return entries


def _write_bib(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(entries) + "\n")
def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _write_latex_template(template_dir, title):
    tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}
\usepackage{times}\usepackage{graphicx}\usepackage{amsmath,amssymb}
\usepackage{hyperref}\usepackage{url}\usepackage{booktabs}\usepackage{caption}\usepackage{float}\usepackage{microtype}
\usepackage[numbers,sort&compress]{natbib}
\setlength{\emergencystretch}{2em}
\sloppy
\graphicspath{{../}}

\begin{document}

\title{""" + title + r"""}
\author{AI Scientist \\ Automated Research Pipeline}
\date{\today}
\maketitle

\begin{abstract}
ABSTRACT HERE
\end{abstract}

\section{Introduction}
\label{sec:intro}
INTRODUCTION HERE

\section{Related Work}
\label{sec:related}
RELATED WORK HERE

\section{Method}
\label{sec:method}
METHOD HERE

\section{Experimental Setup}
\label{sec:setup}
EXPERIMENTAL SETUP HERE

\section{Results}
\label{sec:results}
RESULTS HERE

\section{Discussion}
\label{sec:discussion}
DISCUSSION HERE

\section{Conclusion}
\label{sec:conclusion}
CONCLUSION HERE

\nocite{*}
\bibliographystyle{plainnat}
\bibliography{paper}
\end{document}"""
    with open(os.path.join(template_dir, "latex", "template.tex"), "w", encoding="utf-8") as f:
        f.write(tex)


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF → AI-Scientist Template")
    parser.add_argument("--pdf", type=str, default=None, help="Path to PDF paper")
    parser.add_argument("--txt", type=str, default=None, help="Or paste paper text directly")
    parser.add_argument("--topic_name", type=str, required=True,
                        help="Template folder name (used in --experiment)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to CSV dataset (copied into template)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save extracted paper info as JSON")
    args = parser.parse_args()

    # Read input
    if args.pdf:
        if not os.path.exists(args.pdf):
            print(f"[ERROR] PDF not found: {args.pdf}")
            sys.exit(1)
        print(f"Reading PDF: {args.pdf}")
        content = read_pdf(args.pdf)
    elif args.txt:
        content = args.txt[:15000]
        print(f"Using {len(content)} chars of text input")
    else:
        print("[ERROR] Provide --pdf or --txt")
        sys.exit(1)

    # Phase 1: Extract
    info = extract_paper_info(content)
    if args.output:
        _write_json(args.output, info)
        print(f"\nSaved extracted info: {args.output}")

    # Phase 2: Seed ideas
    seed_ideas = generate_seed_ideas(info)

    # Create template dir early so we can put data in it
    template_dir = os.path.join("templates", args.topic_name)
    os.makedirs(template_dir, exist_ok=True)

    # Phase 2.5: Auto-download real dataset (skip if user provided --data)
    if args.data:
        print(f"\n[Phase 2.5] Using provided dataset: {args.data}")
        shutil.copy(args.data, os.path.join(template_dir, "data.csv"))
        print(f"  Copied to {template_dir}/data.csv")
    else:
        data_csv = find_and_download_dataset(info, template_dir)

    # Phase 3: Build
    print(f"\n[Phase 3] Building template 'templates/{args.topic_name}/' ...")
    template_dir = build_template(args.topic_name, info, seed_ideas, args.data)

    has_data = os.path.exists(os.path.join(template_dir, "data.csv"))

    print(f"\n{'=' * 60}")
    print(f"Template ready: {template_dir}/")
    print(f"{'=' * 60}")
    print(f"\nNext steps:")
    if not has_data:
        print(f"  1. Put your CSV data as templates/{args.topic_name}/data.csv")
        print(f"  2. Run baseline: cd templates/{args.topic_name} && python experiment.py --data data.csv --out_dir run_0")
        print(f"  3. Run AI-Scientist:")
        n = 3
    else:
        print(f"  1. Run baseline: cd templates/{args.topic_name} && python experiment.py --data data.csv --out_dir run_0")
        print(f"  2. Run AI-Scientist:")
        n = 2
    print(f"     python launch_scientist.py --experiment {args.topic_name} \\")
    print(f"         --model deepseek-v4-pro --triz-mode inject --gpus 0")
