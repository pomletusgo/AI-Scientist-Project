#!/usr/bin/env python3
"""从零生成干净论文：标准LaTeX模板 + 完整内容，无重复、无占位符"""
import sys, os, json, time, re
sys.path.insert(0, '/data/lrd/AI-Scientist')
from ai_scientist.llm import create_client, get_response_from_llm

client, model = create_client("deepseek-v4-pro")

# 找到最新的结果文件夹
import glob
folders = sorted(glob.glob("/data/lrd/AI-Scientist/results/nanoGPT/20260720_0*/"))
folder = folders[-1]
print(f"Using: {folder}")

# 加载实验数据
with open(folder + "notes.txt") as f: notes = f.read()
with open(folder + "ideas.json") as f:
    ideas = [i for i in json.load(f) if isinstance(i, dict) and "Name" in i]
idea = ideas[0]
title = idea.get("Title", "Adaptive Transformer Training")

# 列出可用图片
pngs = [f for f in os.listdir(folder) if f.endswith('.png')]
print(f"Images: {pngs}")

SYSTEM = """You are an AI researcher at a top ML conference (NeurIPS/ICML).
Write rigorous, original academic prose. Use LaTeX math notation.
Be specific and precise. Do NOT repeat sections. Do NOT include placeholder text.
Each section should be written ONCE with complete content."""

sections = {}

prompts = [
    ("abstract", f"""Write the Abstract (200 words) for: "{title}"
Context: {notes[:500]}
Method outline: {idea.get('Experiment','')[:300]}
Key results from notes: {notes[500:1500]}
Write one continuous paragraph. Be specific with numbers."""),

    ("introduction", f"""Write the Introduction (~500 words) for: "{title}"
Include: (1) Motivation for efficient transformer training, (2) The specific gap,
(3) Our proposed approach, (4) Key findings from experiments.
End with bullet-point contributions. Use \\cite{{...}} for references."""),

    ("related_work", f"""Write Related Work (~300 words) covering:
- Prior work on adaptive training techniques for transformers
- Related methods for dynamic context/scheduling
- How our approach differs
Use proper \\cite{{...}} citations throughout."""),

    ("method", f"""Write the Method (~600 words) based on this implementation:
{idea.get('Experiment','')}
Include: Problem formulation with math notation, baseline description,
proposed modification with $$...$$ equations for key formulas,
training procedure details. Be technically precise."""),

    ("experimental_setup", f"""Write Experimental Setup (~300 words) based on:
{notes[:1500]}
Include datasets, model config, optimizer, evaluation metrics, baseline comparison."""),

    ("results", f"""Write Results (~500 words) using EXACTLY these numbers.
DO NOT fabricate or change any values:

{notes[500:3000]}

Present a markdown table of key results, then discuss:
- Main findings
- What the improvements mean
- Any trade-offs
- Limitations honestly

Reference the figures: {', '.join(pngs)}."""),

    ("conclusion", f"""Write Conclusion (~200 words) for: "{title}"
Summarize the approach, key result, implications, and 3 future directions."""),
]

for i, (name, prompt) in enumerate(prompts):
    print(f"[{i+1}/7] Writing {name}...")
    resp, _ = get_response_from_llm(
        prompt, client=client, model=model,
        system_message=SYSTEM, temperature=0.7
    )
    sections[name] = resp.strip()
    time.sleep(0.5)

# ============================================================
# 组装干净 LaTeX 文档
# ============================================================
latex = r"""\documentclass[11pt]{article}

% ── 页面设置 ──
\usepackage[margin=1in]{geometry}
\usepackage{times}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}

% ── 学术必备 ──
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage[nottoc]{tocbibind}

% ── 图表路径 ──
\graphicspath{{./}}

% ── 标题 ──
\title{""" + title + r"""}
\author{AI Scientist \\ \small Sakana AI}
\date{\today}

\begin{document}

\maketitle

% ── 摘要 ──
\begin{abstract}
""" + sections['abstract'] + r"""
\end{abstract}

% ── 正文 ──
\section{Introduction}
""" + sections['introduction'] + r"""

\section{Related Work}
""" + sections['related_work'] + r"""

\section{Method}
""" + sections['method'] + r"""

\section{Experimental Setup}
""" + sections['experimental_setup'] + r"""

\section{Results}
""" + sections['results'] + r"""

% ── 图表 ──
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.48\textwidth]{val_loss_enwik8.png}
    \hfill
    \includegraphics[width=0.48\textwidth]{train_loss_enwik8.png}
    \caption{Validation and training loss curves on enwik8 dataset.}
    \label{fig:enwik8}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.48\textwidth]{val_loss_shakespeare_char.png}
    \hfill
    \includegraphics[width=0.48\textwidth]{train_loss_shakespeare_char.png}
    \caption{Validation and training loss curves on Shakespeare character dataset.}
    \label{fig:shakespeare}
\end{figure}

\section{Conclusion}
""" + sections['conclusion'] + r"""

% ── 参考文献 ──
\begin{thebibliography}{99}

\bibitem{vaswani2017attention}
A.~Vaswani et al., ``Attention Is All You Need,'' \emph{NeurIPS}, 2017.

\bibitem{kingma2014adam}
D.~Kingma and J.~Ba, ``Adam: A Method for Stochastic Optimization,'' \emph{arXiv:1412.6980}, 2014.

\bibitem{loshchilov2017adamw}
I.~Loshchilov and F.~Hutter, ``Decoupled Weight Decay Regularization,'' \emph{arXiv:1711.05101}, 2017.

\bibitem{ba2016layer}
J.~Ba et al., ``Layer Normalization,'' \emph{arXiv:1607.06450}, 2016.

\bibitem{radford2019language}
A.~Radford et al., ``Language Models are Unsupervised Multitask Learners,'' \emph{OpenAI}, 2019.

\bibitem{kaplan2020scaling}
J.~Kaplan et al., ``Scaling Laws for Neural Language Models,'' \emph{arXiv:2001.08361}, 2020.

\bibitem{hoffmann2022training}
J.~Hoffmann et al., ``Training Compute-Optimal Large Language Models,'' \emph{NeurIPS}, 2022.

\bibitem{karpathy2023nanogpt}
A.~Karpathy, ``nanoGPT,'' \emph{GitHub repository}, 2023.

\bibitem{lu2024aiscientist}
C.~Lu et al., ``The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery,'' \emph{arXiv:2408.06292}, 2024.

\end{thebibliography}

\end{document}
"""

out = folder + "/paper_clean.tex"
with open(out, "w") as f:
    f.write(latex)

print(f"\n{'='*60}")
print(f"Clean paper: {out}")
print(f"Total: {len(latex)} chars, {len(latex.split())} words")
print(f"\nTo compile: Upload paper_clean.tex + all .png files to overleaf.com")
print(f"{'='*60}")
