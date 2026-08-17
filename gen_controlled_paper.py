#!/usr/bin/env python3
"""
受控论文生成：读取 research_config.json，限定课题范围、文献来源、论文内容
"""
import glob, os, json, re, time
from openai import OpenAI

# ===== 0. 读取配置 =====
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'research_config.json')
with open(config_path) as f:
    config = json.load(f)

domain = config['domain']
keywords = config['keywords']
core_papers = config['core_papers']
search_terms = config['search_terms_for_references']
directions = config['research_directions']
exclude_topics = config['constraints']['exclude_topics']
allowed_methods = config['constraints']['methods']
allowed_datasets = config['constraints']['datasets']

print('=' * 60)
print('Controlled Paper Generation')
print('Domain:', domain)
print('Keywords:', ', '.join(keywords[:3]), '...')
print('Core papers:', len(core_papers))
print('Search terms:', len(search_terms))
print('=' * 60)

# ===== 1. 初始化 =====
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com"
)

best = max(glob.glob('/data/lrd/AI-Scientist/results/nanoGPT/20260720_0*/'),
           key=lambda d: len([f for f in os.listdir(d) if f.endswith('.png')]))
print('Data folder:', os.path.basename(best))

with open(os.path.join(best, 'notes.txt')) as fp: notes = fp.read()
base_title = directions[0] if directions else "Safety Alignment Research"

if os.path.exists(os.path.join(best, 'ideas.json')):
    with open(os.path.join(best, 'ideas.json')) as fp:
        for i in json.load(fp):
            if isinstance(i, dict) and i.get('Title'): base_title = i['Title']; break

pngs = sorted([f for f in os.listdir(best) if f.endswith('.png')])

# ===== 2. 搜索领域特定文献（受约束） =====
def search_real_papers(query, limit=5):
    try:
        import requests
        rsp = requests.get(
            "https://api.openalex.org/works",
            params={"search": query, "per_page": limit, "sort": "cited_by_count:desc"},
            timeout=15
        )
        rsp.raise_for_status()
        papers = []
        for w in rsp.json().get("results", []):
            title = w.get("title", "").lower()
            # 排除不相关
            skip = False
            for excl in config['constraints']['exclude']:
                if excl.lower() in title:
                    skip = True
                    break
            if skip:
                continue
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
    except:
        return []

print('\nSearching domain-specific references...')
refs = []
for term in search_terms[:4]:
    found = search_real_papers(term, limit=5)
    refs.extend(found)
    print(f'  "{term[:50]}" -> {len(found)} papers')

# 去重
seen = set()
refs = [r for r in refs if not (r["title"] in seen or seen.add(r["title"]))]
print(f'Total unique references: {len(refs)}')

# ===== 3. 生成论文（受约束的系统提示词） =====
SYS = f"""You are an academic paper author specializing in {domain}.

RESEARCH SCOPE: {', '.join(keywords[:4])}
RESEARCH DIRECTIONS: {', '.join(directions[:3])}
ALLOWED METHODS: {', '.join(allowed_methods)}
ALLOWED DATASETS: {', '.join(allowed_datasets)}
{exclude_topics}

Write clean academic English prose. NO LaTeX commands, NO backslashes, NO braces.
Use (Author, Year) format for citations. Be specific and detailed."""

sections = {}
prompts = [
    ("abstract", "250 words",
     f"Write a 250-word abstract for a paper about {directions[0]}. "
     f"The paper addresses: {keywords[0]}, {keywords[1]}. "
     f"Use results from experiments: {notes[:800]}. One paragraph."),

    ("introduction", "600 words",
     f"Write a 600-word Introduction about {directions[0]}. "
     f"Context: {domain}. Problem: {keywords[0]}. "
     f"Core references to cite: {', '.join(p['title'] for p in core_papers[:3])}. "
     f"Five numbered contributions."),

    ("related_work", "450 words",
     f"Write 450-word Related Work. Discuss: "
     f"(a) {keywords[1]} methods, (b) {keywords[2]} approaches, "
     f"(c) {directions[1]} research. "
     f"Reference these real papers: {', '.join(r['title'][:50] for r in refs[:5] if r['title'])}."),

    ("method", "600 words",
     f"Write 600-word Method for {directions[0]}. "
     f"Use allowed methods: {', '.join(allowed_methods[:3])}. "
     f"Implementation details from: {notes[:2000]}. Include equations."),

    ("experimental_setup", "350 words",
     f"Write 350-word Setup. Use datasets: {', '.join(allowed_datasets[:3])}. "
     f"Baseline: standard model. Metrics: safety score, robustness, alignment. "
     f"Based on: {notes[:1000]}."),

    ("results", "550 words",
     f"Write 550-word Results. Use numbers from: {notes[800:3000]}. "
     f"Discuss safety metrics, robustness improvements. "
     f"Reference Figures 1-3. Note limitations."),

    ("discussion", "300 words",
     f"Write 300-word Discussion. How does our method address {keywords[0]}? "
     f"Compare to {core_papers[0]['title']} and {core_papers[1]['title']}. "
     f"Three limitations. Implications for {domain}."),

    ("conclusion", "250 words",
     f"Write 250-word Conclusion. Summarize contribution for {domain}. "
     f"Key finding. Broader impact. "
     f"Future directions: {', '.join(directions[1:4] if len(directions)>1 else directions)}."),
]

for sec_name, word_target, prompt in prompts:
    for attempt in range(3):
        print(f'  [{sec_name}] (attempt {attempt+1})...')
        try:
            r = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role":"system","content":SYS},{"role":"user","content":prompt}],
                temperature=0.7, max_tokens=2500
            )
            text = r.choices[0].message.content
            if not text or len(text.strip()) < 50:
                print('    Empty, retry...')
                time.sleep(2)
                continue

            # 清理
            text = re.sub(r'```\w*\n?', '', text)
            text = re.sub(r'[{}]', '', text)
            text = text.replace('\\', '')
            text = text.replace('??', '').replace('[?]', '')
            text = text.replace('_', r'\_')
            text = text.replace('&', 'and')
            text = text.replace('%', 'percent')
            text = text.replace('#', 'No.')
            text = text.replace('$', '')
            text = re.sub(r'[^\x20-\x7E\n\t]', ' ', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r' {2,}', ' ', text)

            wc = len(text.split())
            if wc >= 30:
                sections[sec_name] = text.strip()
                print(f'    -> {wc} words')
                break
            else:
                print(f'    Only {wc} words, retry...')
                time.sleep(2)
        except Exception as e:
            print(f'    Error: {e}')
            time.sleep(3)
    if sec_name not in sections:
        sections[sec_name] = "Content generation failed."

# ===== 4. 生成 BibTeX =====
print('\nGenerating BibTeX references...')
bib_prompt = f"""Generate BibTeX entries for these academic papers.
Use citation keys like firstauthorYYYYkeyword.

Papers from literature search:
{chr(10).join(f'- {r["authors"]} ({r["year"]}). {r["title"]}. {r["venue"]}.' for r in refs[:6])}

Core papers in {domain}:
{chr(10).join(f'- {p["title"]}' for p in core_papers[:3])}

Output ONLY valid @article or @inproceedings BibTeX entries. One per paper."""

r = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role":"system","content":"You are a BibTeX generator. Output ONLY valid BibTeX entries."},
              {"role":"user","content":bib_prompt}],
    temperature=0.3, max_tokens=2000
)
bib_tex = r.choices[0].message.content
bib_tex = bib_tex.replace('```bibtex', '').replace('```', '').strip()

bib_path = os.path.join(best, 'references.bib')
with open(bib_path, 'w', encoding='utf-8') as fp:
    fp.write(bib_tex)
print(f'Saved: {bib_path} ({len(bib_tex)} chars)')

# ===== 5. 图片 =====
figs = ""
for ds in ["enwik8","shakespeare_char","text8"]:
    val = next((p for p in pngs if "val" in p and ds in p), "")
    train = next((p for p in pngs if "train" in p and ds in p), "")
    if val and train:
        ds_label = ds.replace('_', r'\_')
        figs += r"""\begin{figure}[htbp]\centering
\includegraphics[width=0.48\textwidth]{""" + val + r"""}\hfill
\includegraphics[width=0.48\textwidth]{""" + train + r"""}
\caption{Loss on """ + ds_label + r""" dataset.}\end{figure}
"""

# ===== 6. 组装 LaTeX（BibTeX 模式） =====
tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}
\usepackage{times}\usepackage{graphicx}\usepackage{amsmath,amssymb}
\usepackage{hyperref}\usepackage{url}\usepackage{booktabs}\usepackage{caption}\usepackage{float}\usepackage{microtype}
\usepackage[numbers,sort&compress]{natbib}

\setlength{\emergencystretch}{2em}
\sloppy
\graphicspath{{./}}

\begin{document}
\title{""" + base_title + r"""}
\author{AI Scientist \\ Sakana AI}
\date{\today}
\maketitle

\begin{abstract}
""" + sections['abstract'] + r"""
\end{abstract}

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

""" + figs + r"""

\section{Discussion}
""" + sections['discussion'] + r"""

\section{Conclusion}
""" + sections['conclusion'] + r"""

\bibliographystyle{plainnat}
\bibliography{references}
\end{document}"""

out = os.path.join(best, 'CONTROLLED_PAPER.tex')
with open(out, 'w', encoding='utf-8') as fp: fp.write(tex)

total = sum(len(s.split()) for s in sections.values())
print(f'\n===== DONE =====')
print(f'Paper: {out}')
print(f'BibTeX: {bib_path}')
print(f'Words: {total} | Est. pages: {total//400+3}')
for k, v in sections.items():
    print(f'  {k}: {len(v.split())} words')
print(f'\nUpload to Overleaf:')
print(f'  1. CONTROLLED_PAPER.tex')
print(f'  2. references.bib')
print(f'  3. All PNG files')
print(f'\nCompile: Recompile TWICE (first for .aux, second for bibliography)')
