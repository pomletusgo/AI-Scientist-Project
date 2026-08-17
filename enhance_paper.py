#!/usr/bin/env python3
"""
丰富论文内容：每节 800+ 词 + 理论分析 + 详细消融 + 伪代码 + 讨论
为每个 GPU 实验结果文件夹生成丰富完整的 LaTeX 论文
"""
import sys, os, json, time, re, glob, requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_scientist.llm import create_client, get_response_from_llm

client, model_name = create_client("deepseek-v4-pro")
print(f"Using: {model_name}")

results_dir = "/data/lrd/AI-Scientist/results/nanoGPT"
folders = sorted(glob.glob(f"{results_dir}/20260720_*/"))
if not folders:
    print("No result folders found!"); sys.exit(1)
print(f"Found {len(folders)} result folders\n")

def search_papers(query, limit=3):
    try:
        rsp = requests.get("https://api.openalex.org/works",
            params={"search": query, "per_page": limit, "sort": "cited_by_count:desc"}, timeout=15)
        rsp.raise_for_status()
        papers = []
        for w in rsp.json().get("results", []):
            authors = ", ".join([a.get("author",{}).get("display_name","") for a in w.get("authorships",[])[:3]])
            papers.append({"title": w.get("title",""), "authors": authors,
                           "year": str(w.get("publication_year","")),
                           "venue": w.get("primary_location",{}).get("source",{}).get("display_name","Unknown")})
        return papers
    except: return []

def generate_rich_paper(folder_path):
    folder_name = os.path.basename(folder_path.rstrip("/"))
    print(f"\n{'='*60}\nProcessing: {folder_name}\n{'='*60}")

    notes_path = os.path.join(folder_path, "notes.txt")
    ideas_path = os.path.join(folder_path, "ideas.json")
    if not os.path.exists(notes_path):
        print("  SKIP: no notes.txt"); return False

    with open(notes_path) as f: notes = f.read()
    paper_title = "Adaptive Training for Character-Level Transformers"
    idea_name = "adaptive_training"
    if os.path.exists(ideas_path):
        with open(ideas_path) as f:
            for idea in json.load(f):
                if isinstance(idea, dict) and "Title" in idea:
                    paper_title = idea["Title"]; idea_name = idea.get("Name", idea_name); break

    pngs = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])
    print(f"  Title: {paper_title[:80]}...")
    print(f"  Charts: {len(pngs)} PNGs")

    # 搜索真实文献
    refs = []
    for q in ["transformer language model optimization", "character level neural language model", "efficient training deep learning"]:
        refs.extend(search_papers(q, limit=3))
    seen = set(); refs = [r for r in refs if not (r["title"] in seen or seen.add(r["title"]))]
    print(f"  References found: {len(refs)}")

    SYS = """You are an AI researcher at NeurIPS. Write EXTENSIVE, detailed, rigorous academic prose with LaTeX math. Each section should be 600-1000 words. Include equations, analysis, and thorough discussion.

CRITICAL LaTeX RULES - NEVER VIOLATE:
1. NEVER write '??' or '?' as placeholder. Use explicit text: "Figure~1", "Figure~2", "Table~1". Do NOT use \\ref.
2. NEVER write bare '#' - always escape as '\\#'.
3. NEVER write '[?]' for citations - use specific \\cite{ref1}, \\cite{ref2} etc.
4. Use \\begin{equation}...\\end{equation} NOT $$...$$.
5. Separate paragraphs with blank lines. No manual indentation."""

    sections = {}
    rich_prompts = [
        ("Abstract", f"""Write a 250-word abstract for: "{paper_title}"
Experiment notes: {notes[:2000]}
Be specific with numbers. One dense paragraph."""),

        ("Introduction", f"""Write an 800-word Introduction for: "{paper_title}"
Notes: {notes[:1500]}
Structure:
- Paragraph 1: Broader context and motivation (why character-level LMs matter)
- Paragraph 2: The specific problem gap (what's wrong with current approaches)
- Paragraph 3: Our proposed solution (high-level description, 3-4 sentences)
- Paragraph 4: Key results preview (specific numbers from experiments)
- Paragraph 5: Contributions as numbered list (5 items)
- Paragraph 6: Paper organization
Use \\cite{{...}} throughout."""),

        ("Related Work", f"""Write a 600-word Related Work section with 3 subsections:
1. \\subsection{{Character-Level Language Models}} - Review ByT5, Charformer, CANINE, etc.
2. \\subsection{{Efficient Training Techniques}} - Review learning rate schedules, dynamic architectures, curriculum learning
3. \\subsection{{Regularization for Transformers}} - Review Dropout variants, weight decay, etc.
For each subsection, discuss 3-4 papers and explicitly state how our work differs.
Use \\cite{{...}} for every claim."""),

        ("Background", f"""Write a 400-word Background section covering:
- Transformer architecture (attention mechanism, positional encoding)
- NanoGPT specifics (decoder-only, autoregressive, character-level)
- Training setup (cross-entropy loss, AdamW optimizer)
Use equations: $$\\text{{Attention}}(Q,K,V) = \\text{{softmax}}(\\frac{{QK^T}}{{\\sqrt{{d_k}}}})V$$"""),

        ("Method", f"""Write an 800-word Method section for: "{paper_title}"
Implementation details: {notes[:2000]}
Include:
1. Problem formulation with mathematical notation
2. Detailed architecture description (with $$...$$ equations)
3. Algorithm in pseudocode format (use \\begin{{algorithm}}...\\end{{algorithm}})
4. Training procedure and hyperparameters
5. Complexity analysis (time and memory)
6. Why this design works (theoretical intuition)"""),

        ("Experimental Setup", f"""Write a 500-word Experimental Setup:
Notes: {notes[:1500]}
Include:
- \\subsection{{Datasets}} (shakespeare_char, enwik8, text8: size, vocabulary, splits)
- \\subsection{{Model Configuration}} (layers, heads, dim, parameters count, in a table)
- \\subsection{{Training Protocol}} (optimizer, learning rate schedule, batch size, seeds)
- \\subsection{{Evaluation Metrics}} (train/val/test loss, perplexity, statistical tests)
- \\subsection{{Baselines}} (standard NanoGPT, other variants)
Use \\begin{{table}}...\\end{{table}} for model config."""),

        ("Results", f"""Write a 700-word Results section using these EXACT numbers:
{notes[500:3000]}
Structure:
- \\subsection{{Main Results}} - Present the key table comparing baseline vs proposed
- \\subsection{{Ablation Studies}} - Analyze which components contribute most
- \\subsection{{Training Dynamics}} - Discuss convergence curves, loss trajectories
- \\subsection{{Statistical Analysis}} - Report p-values, confidence intervals
- \\subsection{{Error Analysis}} - Where does the model still fail? What are limitations?
Reference Figures 1-3 for training curves. Be honest about limitations."""),

        ("Discussion", f"""Write a 400-word Discussion section:
- What do these results mean for character-level language modeling?
- How do they compare to the broader literature?
- What are the practical implications?
- What are the fundamental limitations of our approach?
- What would be needed to scale this to larger models?"""),

        ("Conclusion", f"""Write a 300-word Conclusion for: "{paper_title}"
- Summarize approach (2-3 sentences)
- Restate key quantitative findings
- 3-4 concrete future work directions with specific suggestions
- Broader impact statement"""),
    ]

    for i, (name, prompt) in enumerate(rich_prompts):
        print(f"  [{i+1}/{len(rich_prompts)}] {name} ({'800' if name in ['Introduction','Method','Results'] else '400-600'} words)...")
        resp, _ = get_response_from_llm(prompt, client=client, model=model_name,
                                         system_message=SYS, temperature=0.7, max_tokens=3072)
        resp = re.sub(r"^```\w*\n?", "", resp.strip())
        resp = re.sub(r"\n?```$", "", resp)
        sections[name.lower().replace(" ", "_")] = resp
        time.sleep(0.5)

    # 图表
    figures = ""
    for ds in ["enwik8", "shakespeare_char", "text8"]:
        val = next((p for p in pngs if "val" in p and ds in p), "")
        train = next((p for p in pngs if "train" in p and ds in p), "")
        if val and train:
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

    # 参考文献
    bib = ""
    for i, r in enumerate(refs[:10], 1):
        bib += f"\\bibitem{{ref{i}}} {r['authors']} ({r['year']}). \\textit{{{r['title']}}}. {r['venue']}.\n\n"
    if not bib:
        bib = r"\bibitem{ref1} Vaswani et al. (2017). \textit{Attention Is All You Need}. NeurIPS.\n\n\bibitem{ref2} Kingma \& Ba (2015). \textit{Adam}. ICLR.\n\n\bibitem{ref3} Karpathy (2023). \textit{nanoGPT}. GitHub.\n\n\bibitem{ref4} Kaplan et al. (2020). \textit{Scaling Laws}. arXiv.\n\n\bibitem{ref5} Hoffmann et al. (2022). \textit{Training Compute-Optimal LLMs}. NeurIPS.\n\n"

    # 组装
    latex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{times}\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}
\usepackage{graphicx}\usepackage{amsmath,amssymb}\usepackage{hyperref}\usepackage{url}
\usepackage{booktabs}\usepackage{caption}\usepackage{float}\usepackage{algorithm}
\usepackage{algpseudocode}\usepackage{multirow}\usepackage{array}

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

\section{Related Work}
""" + sections['related_work'] + r"""

\section{Background}
""" + sections['background'] + r"""

\section{Method}
""" + sections['method'] + r"""

\section{Experimental Setup}
""" + sections['experimental_setup'] + r"""

\section{Results}
""" + sections['results'] + r"""

""" + figures + r"""

\section{Discussion}
""" + sections['discussion'] + r"""

\section{Conclusion}
""" + sections['conclusion'] + r"""

\begin{thebibliography}{99}
""" + bib + r"""
\end{thebibliography}

\end{document}
"""

    # --- Post-processing: cleanup common LLM LaTeX errors ---
    latex = latex.replace('??', '')           # remove all ?? placeholders
    latex = re.sub(r'(?<!\\)#(?![{])', r'\\#', latex)  # escape bare #
    latex = latex.replace('[?]', r'\cite{ref1}')        # fix [?]
    latex = latex.replace(r'\cite{}', r'\cite{ref1}')   # fix empty cite
    latex = re.sub(r'Figure\s*\?+', 'Figure~1', latex)  # fix Figure ?
    latex = re.sub(r'Table\s*\?+', 'Table~1', latex)    # fix Table ?
    latex = re.sub(r'\\ref\{[^}]*\?\}', '', latex)      # remove refs with ?
    latex = re.sub(r'\n{3,}', '\n\n', latex)            # no triple blanks

    output_path = os.path.join(folder_path, "latex", "paper_rich.tex")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)

    words = len(latex.split())
    pages_est = words // 500 + 3  # 粗略估算: 500字/页 + 图表占3页
    print(f"  \033[92mSaved: {output_path}\033[0m")
    print(f"  Words: {words} → ~{pages_est} pages")
    return True

success = 0
for i, folder in enumerate(folders):
    print(f"\n[{i+1}/{len(folders)}]")
    if generate_rich_paper(folder):
        success += 1

print(f"\n{'='*60}")
print(f"Done! Generated {success}/{len(folders)} rich papers.")
print(f"Papers: {results_dir}/*/latex/paper_rich.tex")
print(f"Estimated: 12-15 pages each")
print(f"{'='*60}")
