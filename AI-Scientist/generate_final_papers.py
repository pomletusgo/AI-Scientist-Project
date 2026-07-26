#!/usr/bin/env python3
"""
为每个 GPU 实验结果文件夹生成完整的 LaTeX 论文。
输入：notes.txt + ideas.json + PNG 图表
输出：latex/paper_final.tex（结构完整、图表嵌入、真参考文献）
"""

import sys, os, json, time, re, glob, requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_scientist.llm import create_client, get_response_from_llm

# ============================================================
# 0. 初始化
# ============================================================
client, model_name = create_client("deepseek-v4-pro")
print(f"Using: {model_name}")

results_dir = "/data/lrd/AI-Scientist/results/nanoGPT"
folders = sorted(glob.glob(f"{results_dir}/20260720_*/"))

if not folders:
    print("No result folders found!")
    sys.exit(1)

print(f"Found {len(folders)} result folders\n")

# ============================================================
# 1. OpenAlex 文献搜索（免费，不限流）
# ============================================================
def search_papers(query, limit=3):
    try:
        rsp = requests.get(
            "https://api.openalex.org/works",
            params={"search": query, "per_page": limit, "sort": "cited_by_count:desc"},
            timeout=15,
        )
        rsp.raise_for_status()
        papers = []
        for w in rsp.json().get("results", []):
            authors = ", ".join([
                a.get("author", {}).get("display_name", "")
                for a in w.get("authorships", [])[:3]
            ])
            papers.append({
                "title": w.get("title", ""),
                "authors": authors,
                "year": str(w.get("publication_year", "")),
                "venue": w.get("primary_location", {}).get("source", {}).get("display_name", "Unknown"),
            })
        return papers
    except Exception as e:
        print(f"  Search error: {e}")
        return []

# ============================================================
# 2. 生成一篇论文
# ============================================================
def generate_paper_for_folder(folder_path):
    folder_name = os.path.basename(folder_path.rstrip("/"))
    print(f"{'='*60}")
    print(f"Processing: {folder_name}")
    print(f"{'='*60}")

    # 加载数据
    notes_path = os.path.join(folder_path, "notes.txt")
    ideas_path = os.path.join(folder_path, "ideas.json")
    if not os.path.exists(notes_path):
        print("  SKIP: no notes.txt")
        return False

    with open(notes_path) as f:
        notes = f.read()

    # 提取标题
    idea_name = "Improved Transformer Training"
    paper_title = "Adaptive Training for Character-Level Transformers"
    if os.path.exists(ideas_path):
        with open(ideas_path) as f:
            ideas = json.load(f)
        for idea in ideas:
            if isinstance(idea, dict) and "Title" in idea:
                paper_title = idea["Title"]
                idea_name = idea.get("Name", idea_name)
                break

    # 列出 PNG 图表
    pngs = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])
    print(f"  Title: {paper_title[:80]}...")
    print(f"  Charts: {len(pngs)} PNGs")
    print(f"  Notes: {len(notes)} chars")

    # 搜索真实文献
    print(f"  Searching references...")
    queries = ["transformer language model optimization", "character level language model", "efficient neural network training"]
    refs = []
    for q in queries[:2]:
        refs.extend(search_papers(q, limit=3))
    # 去重
    seen = set()
    refs = [r for r in refs if not (r["title"] in seen or seen.add(r["title"]))]
    print(f"  Found {len(refs)} real references")

    # ============================================================
    # 逐节生成论文
    # ============================================================
    paper_sys = "You are an AI researcher at a top ML conference (NeurIPS/ICML). Write rigorous, precise academic prose with LaTeX math. Use \\cite{...} for references. Write ONLY the section content requested."

    sections = {}
    prompts = [
        ("Abstract", f"Write a 200-word abstract for: \"{paper_title}\"\nBased on these experiment notes:\n{notes[:1500]}\nOne continuous paragraph. Be specific with numbers."),
        ("Introduction", f"Write a 500-word Introduction for: \"{paper_title}\"\nContext: {notes[:1000]}\nInclude: motivation, research gap, our approach, key results, bullet contributions. Use \\cite{{...}}."),
        ("Method", f"Write a 500-word Method section for: \"{paper_title}\"\nExperiment notes:\n{notes[:2000]}\nInclude mathematical formulation with $$...$$ equations. Describe the architecture and training procedure."),
        ("Experimental Setup", f"Write a 300-word Experimental Setup. Based on:\n{notes[:1500]}\nDescribe: datasets (shakespeare_char, enwik8, text8), model config (NanoGPT-based transformer), optimizer (AdamW), evaluation metrics (train/val/test loss, perplexity), 3 random seeds."),
        ("Results", f"Write a 400-word Results section. Use THESE exact numbers from experiments:\n{notes[500:3000]}\nPresent findings honestly. Note limitations. Mention training curves (shown in Figures 1-3)."),
        ("Conclusion", f"Write a 200-word Conclusion for: \"{paper_title}\"\nSummarize approach, key result, 3 future directions."),
    ]

    for i, (name, prompt) in enumerate(prompts):
        print(f"  [{i+1}/6] {name}...")
        resp, _ = get_response_from_llm(
            prompt, client=client, model=model_name,
            system_message=paper_sys, temperature=0.7
        )
        # 清理
        resp = resp.strip()
        resp = re.sub(r"^```\w*\n?", "", resp)
        resp = re.sub(r"\n?```$", "", resp)
        sections[name.lower().replace(" ", "_")] = resp
        time.sleep(0.5)

    # ============================================================
    # 构建图表环境（硬编码，LLM 不参与）
    # ============================================================
    figures = ""
    fig_pairs = []
    for ds in ["enwik8", "shakespeare_char", "text8"]:
        val = next((p for p in pngs if "val" in p and ds in p), "")
        train = next((p for p in pngs if "train" in p and ds in p), "")
        if val and train:
            fig_pairs.append((ds, val, train))

    for ds, val, train in fig_pairs:
        ds_label = ds.replace("_", "\\_")
        figures += f"""
\\begin{{figure}}[htbp]
    \\centering
    \\includegraphics[width=0.48\\textwidth]{{{val}}}
    \\hfill
    \\includegraphics[width=0.48\\textwidth]{{{train}}}
    \\caption{{Validation (left) and training (right) loss curves on {ds_label} dataset.}}
    \\label{{fig:{ds}}}
\\end{{figure}}
"""

    # ============================================================
    # 构建参考文献
    # ============================================================
    bib = ""
    for i, r in enumerate(refs[:8], 1):
        bib += f"\\bibitem{{ref{i}}} {r['authors']} ({r['year']}). \\textit{{{r['title']}}}. {r['venue']}.\n\n"

    # 兜底
    if not bib:
        bib = r"""\bibitem{ref1} Vaswani et al. (2017). \textit{Attention Is All You Need}. NeurIPS.
\bibitem{ref2} Kingma \& Ba (2015). \textit{Adam: A Method for Stochastic Optimization}. ICLR.
\bibitem{ref3} Karpathy (2023). \textit{nanoGPT}. GitHub.
\bibitem{ref4} Kaplan et al. (2020). \textit{Scaling Laws for Neural Language Models}. arXiv.
\bibitem{ref5} Hoffmann et al. (2022). \textit{Training Compute-Optimal Large Language Models}. NeurIPS.
"""

    # ============================================================
    # 组装 LaTeX
    # ============================================================
    latex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{times}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{caption}
\usepackage{float}

\graphicspath{{../}}

\title{""" + paper_title + r"""}
\author{AI Scientist \\ Sakana AI}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
""" + sections['abstract'] + r"""
\end{abstract}

\section{Introduction}
""" + sections['introduction'] + r"""

\section{Method}
""" + sections['method'] + r"""

\section{Experimental Setup}
""" + sections['experimental_setup'] + r"""

\section{Results}
""" + sections['results'] + r"""

""" + figures + r"""

\section{Conclusion}
""" + sections['conclusion'] + r"""

\begin{thebibliography}{99}
""" + bib + r"""
\end{thebibliography}

\end{document}
"""

    # 保存
    output_path = os.path.join(folder_path, "latex", "paper_final.tex")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"  \033[92mSaved: {output_path}\033[0m")
    print(f"  Paper: {len(latex.split())} words, {len(latex)} chars")
    return True


# ============================================================
# 3. 批量处理
# ============================================================
success = 0
for i, folder in enumerate(folders):
    print(f"\n[{i+1}/{len(folders)}]")
    if generate_paper_for_folder(folder):
        success += 1

print(f"\n{'='*60}")
print(f"Done! Generated {success}/{len(folders)} papers.")
print(f"Papers saved to: {results_dir}/*/latex/paper_final.tex")
print(f"To get PDF: download folder(s) + PNGs, upload to overleaf.com")
print(f"{'='*60}")
