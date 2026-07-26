#!/usr/bin/env python3
"""
MultiAgent-Scientist：基于 Qwen 的多智能体科学假设生成系统

输入：文献 PDF / 文本 / 领域描述
输出：标准化的《科学假设与研究计划》
"""

import json
import os
import sys
import argparse
import time as time_mod
from datetime import datetime
from llm_interface import create_qwen_client, call_qwen

# --- 文献搜索（Semantic Scholar API，免费不需要 Key）---
import requests
import backoff

@backoff.on_exception(backoff.expo, requests.exceptions.HTTPError, max_tries=3)
def search_real_papers(query, limit=5):
    """搜索真实论文 —— OpenAlex（免费不限流）"""
    rsp = requests.get(
        "https://api.openalex.org/works",
        params={
            "search": query,
            "per_page": limit,
            "sort": "cited_by_count:desc",
        },
        timeout=15,
    )
    rsp.raise_for_status()
    data = rsp.json()
    papers = []
    for w in data.get("results", []):
        authors = ", ".join([
            a.get("author", {}).get("display_name", "")
            for a in w.get("authorships", [])[:3]
        ])
        papers.append({
            "title": w.get("title", ""),
            "authors": authors,
            "year": w.get("publication_year", ""),
            "venue": w.get("primary_location", {}).get("source", {}).get("display_name", ""),
            "citations": w.get("cited_by_count", 0),
            "abstract": "",
        })
    return papers

# ============================================================
# Agent 1: 问题分析 Agent —— 识别领域局限性
# ============================================================
def agent_problem_analysis(client, model, literature_text, domain, mode):
    system = """你是一位资深科研评审专家。你的任务是：
1. 仔细阅读提供的文献内容
2. 识别该领域当前存在的 3-5 个具体局限性或未解决的问题
3. 对每个问题进行清晰表述，并评估其重要性

输出 JSON 格式：
```json
{
  "problems": [
    {"issue": "具体问题描述", "severity": "high/medium/low", "evidence": "从文献中哪里看出来的"}
  ],
  "primary_problem": "最值得研究的一个核心问题"
}
```"""
    prompt = f"""文献内容：
{literature_text[:8000]}

领域：{domain}

请分析上述文献，识别该领域当前存在的具体局限性。"""

    resp = call_qwen(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        client, model, temperature=0.7, mode=mode
    )
    return _extract_json(resp)


# ============================================================
# Agent 2: 知识整合 Agent —— 关联发现 + 推导创新思路
# ============================================================
def agent_knowledge_integration(client, model, literature_text, problems, domain, mode):
    system = """你是一位跨学科研究科学家。你的任务是：
1. 基于文献和已识别的问题，关联该领域及相关领域的已有工作
2. 推导出一条创新的解决思路（Rationale），展示完整的逻辑推导链条
3. 找出支持该思路的关键参考文献

输出 JSON 格式：
```json
{
  "rationale": "基于逻辑推理的创新点阐述，展示推导链条",
  "related_work": [{"title": "...", "authors": "...", "year": 2023, "relevance": "如何相关"}],
  "innovation_angle": "核心创新点是什么",
  "feasibility": "为什么这个思路可行"
}
```"""
    prompt = f"""文献内容：{literature_text[:8000]}
领域：{domain}
已识别的问题：{json.dumps(problems, ensure_ascii=False)}

请整合知识，提出创新解决思路。"""

    resp = call_qwen(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        client, model, temperature=0.8, mode=mode
    )
    return _extract_json(resp)


# ============================================================
# Agent 3: 方法设计 Agent —— 技术手段 + 实施步骤
# ============================================================
def agent_method_design(client, model, problems, rationale, domain, mode):
    system = """你是一位机器学习/数据科学方法专家。你的任务是：
1. 基于问题和解决思路，设计具体的技术方案
2. 列出验证假设所需的全部技术栈（统计方法、ML模型、深度学习框架等）
3. 给出详细的实施步骤（Methods），包括模型架构或实验流程

输出 JSON 格式：
```json
{
  "technical_stack": {
    "programming": ["Python 3.10", "PyTorch 2.x"],
    "ml_methods": ["具体方法1", "具体方法2"],
    "statistics": ["统计检验方法"],
    "frameworks": ["框架名称"],
    "hardware": ["GPU型号或计算资源需求"]
  },
  "methods": {
    "step_1": "...",
    "step_2": "...",
    "step_3": "...",
    "step_n": "..."
  },
  "model_architecture": "如果涉及神经网络，描述架构设计",
  "expected_challenges": ["可能的困难1", "可能的困难2"]
}
```"""
    prompt = f"""领域：{domain}
问题：{json.dumps(problems, ensure_ascii=False)}
解决思路：{json.dumps(rationale, ensure_ascii=False)}

请设计具体的技术方案和实施步骤。"""

    resp = call_qwen(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        client, model, temperature=0.6, mode=mode
    )
    return _extract_json(resp)


# ============================================================
# Agent 4: 实验规划 Agent —— 数据 + 基线 + 指标 + 结果
# ============================================================
def agent_experiment_planning(client, model, problems, rationale, methods, domain, mode):
    system = """你是一位实验设计专家。你的任务是：
1. 设计验证假设所需的完整实验方案
2. 指定 Source 数据（历史数据）和 Target 数据（需采集的数据特征）
3. 确定基线方法（Baselines）和评估指标（Metrics）
4. 在一定范围内进行公式推导或可行性验证，给出预期实验结果

输出 JSON 格式：
```json
{
  "datasets": {
    "source": {"name": "数据集名称", "url": "来源链接", "description": "为什么选这个数据"},
    "target": {"features": ["需要的特征1", "特征2"], "collection_method": "如何采集"}
  },
  "baselines": [
    {"name": "Baseline 1", "description": "...", "why_chosen": "..."}
  ],
  "metrics": [
    {"name": "指标名", "formula": "计算公式", "interpretation": "如何解读"}
  ],
  "expected_results": {
    "quantitative": "预期数值结果（附推导过程）",
    "qualitative": "定性预期",
    "feasibility_analysis": "实验可行性分析"
  }
}
```"""
    prompt = f"""领域：{domain}
问题：{json.dumps(problems, ensure_ascii=False)}
思路：{json.dumps(rationale, ensure_ascii=False)}
方法：{json.dumps(methods, ensure_ascii=False)}

请设计完整的实验方案。"""

    resp = call_qwen(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        client, model, temperature=0.6, mode=mode
    )
    return _extract_json(resp)


# ============================================================
# Agent 5: 综合写作 Agent —— 标题 + 摘要 + 整合报告
# ============================================================
def agent_synthesis_writing(client, model, all_results, domain, mode):
    system = """你是一位顶会论文作者和科学作家。你的任务是：
1. 基于前面所有 Agent 的输出，撰写完整的科学假设与研究计划
2. 生成符合学术规范的标题
3. 撰写包含背景、方法、预期结果的完整摘要

输出 JSON 格式：
```json
{
  "paper_title": "符合学术规范的论文标题",
  "abstract": "包含背景、方法、预期结果的完整摘要（200-300字）",
  "keywords": ["关键词1", "关键词2", "关键词3"]
}
```"""
    prompt = f"""所有前置分析结果：
{json.dumps(all_results, ensure_ascii=False, indent=2)}

领域：{domain}

请撰写标题和摘要，整合所有内容。"""

    resp = call_qwen(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        client, model, temperature=0.7, mode=mode
    )
    return _extract_json(resp)


# ============================================================
# Agent 6: 论文写作 —— 逐节生成完整学术论文
# ============================================================
def agent_paper_writing(client, model, all_results, domain, mode, real_papers=None):
    """逐节生成论文，每节独立调用，避免 JSON 解析失败"""
    import time

    writing = all_results.get("writing", {})
    title = writing.get("paper_title", "Research Paper")
    abstract = writing.get("abstract", "")

    paper_sections = {}
    section_prompts = [
        ("introduction", f"""Write the Introduction (~500 words) for the paper: "{title}"
Abstract: {abstract}
Problem: {json.dumps(all_results.get('problems', {}), ensure_ascii=False)[:500]}
Include: background, research gap, our proposed approach, contributions as bullet points.
Use \\cite{{...}} for references. Write ONLY the section content, no JSON wrapping."""),

        ("related_work", f"""Write the Related Work (~300 words) for the paper: "{title}"
Rationale: {json.dumps(all_results.get('rationale', {}), ensure_ascii=False)[:500]}
Compare with prior work on this topic. Highlight how our approach differs.
Use \\cite{{...}}. Write ONLY the section content."""),

        ("method", f"""Write the Method section (~500 words) for the paper: "{title}"
Methods: {json.dumps(all_results.get('methods', {}), ensure_ascii=False)[:1000]}
Include mathematical formulation with $$...$$ equations. Describe the approach step by step.
Write ONLY the section content."""),

        ("experimental_setup", f"""Write the Experimental Setup (~300 words) for the paper: "{title}"
Experiments: {json.dumps(all_results.get('experiments', {}), ensure_ascii=False)[:800]}
Describe datasets, baselines, metrics, and implementation details.
Write ONLY the section content."""),

        ("results", f"""Write the Results and Discussion (~400 words) for the paper: "{title}"
Expected Results: {json.dumps(all_results.get('experiments', {}).get('expected_results', {}), ensure_ascii=False)[:800]}
Present expected findings, discuss implications, note limitations honestly.
Write ONLY the section content."""),

        ("conclusion", f"""Write the Conclusion (~200 words) for the paper: "{title}"
Summarize the approach, key contributions, and 2-3 future directions.
Write ONLY the section content."""),
    ]

    system = "You are a paper author at NeurIPS/ICML. Write rigorous academic prose with LaTeX math."

    for i, (name, prompt) in enumerate(section_prompts):
        print(f"    [{i+1}/6] Writing {name}...")
        resp = call_qwen(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            client, model, temperature=0.7, max_tokens=2048, mode=mode
        )
        paper_sections[name] = resp.strip()
        time.sleep(0.5)

    # 参考文献：直接用 Semantic Scholar 原始结果，LLM 不参与
    if real_papers:
        ref_lines = []
        for p in real_papers[:12]:
            ref_lines.append(f"{p['authors']} ({p['year']}). {p['title']}. {p['venue']}.")
        ref_resp = "\n".join(ref_lines)
    else:
        # fallback: LLM 生成
        ref_prompt = "List 5 well-known academic papers in this field. Format: Author (Year). Title. Venue."
        ref_resp = call_qwen(
            [{"role": "system", "content": system}, {"role": "user", "content": ref_prompt}],
            client, model, temperature=0.3, max_tokens=512, mode=mode
        )

    return {**paper_sections, "references_text": ref_resp}


def assemble_paper(all_results):
    """将论文各章节组装为 LaTeX 文档"""
    paper = all_results.get("paper", {})
    writing = all_results.get("writing", {})

    # 格式化为 BibTeX 条目
    refs_text = paper.get("references_text", "")
    ref_entries = []
    for i, line in enumerate(refs_text.strip().split("\n"), 1):
        line = line.strip()
        if line and len(line) > 10:
            # 去掉可能的编号前缀
            if line[0].isdigit() and ". " in line[:4]:
                line = line.split(". ", 1)[-1]
            ref_entries.append(f"\\bibitem{{ref{i}}} {line}")

    refs = "\n".join(ref_entries) if ref_entries else "\\bibitem{ref1} Reference list to be completed."

    latex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{times}\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}
\usepackage{graphicx}\usepackage{amsmath,amssymb}\usepackage{hyperref}\usepackage{booktabs}
\title{""" + writing.get("paper_title", "Research Paper") + r"""}
\author{MultiAgent-Scientist \\ Based on Qwen Series}
\date{\today}
\begin{document}
\maketitle
\begin{abstract}
""" + writing.get("abstract", "") + r"""
\end{abstract}
\section{Introduction}
""" + paper.get("introduction", "") + r"""
\section{Related Work}
""" + paper.get("related_work", "") + r"""
\section{Method}
""" + paper.get("method", "") + r"""
\section{Experimental Setup}
""" + paper.get("experimental_setup", "") + r"""
\section{Results and Discussion}
""" + paper.get("results", "") + r"""
\section{Conclusion}
""" + paper.get("conclusion", "") + r"""
\begin{thebibliography}{99}
""" + refs + r"""
\end{thebibliography}
\end{document}"""

    return latex


# ============================================================
# 辅助函数
# ============================================================
def _extract_json(text):
    """从 LLM 回复中提取 JSON"""
    import re
    # 尝试 ```json ... ``` 格式
    m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 尝试找 { ... }
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"raw_output": text, "error": "无法解析 JSON"}


def load_literature(sources):
    """加载文献：PDF / TXT / 直接文本"""
    texts = []
    for src in sources:
        if os.path.isfile(src):
            if src.endswith('.pdf'):
                try:
                    import pymupdf4llm
                    text = pymupdf4llm.to_markdown(src)
                except:
                    import pymupdf
                    doc = pymupdf.open(src)
                    text = "\n".join(page.get_text() for page in doc)
                texts.append(text[:5000])  # 每篇限制 5000 字
            elif src.endswith('.txt') or src.endswith('.md'):
                with open(src, 'r', encoding='utf-8') as f:
                    texts.append(f.read()[:5000])
        elif os.path.isdir(src):
            for fname in os.listdir(src):
                if fname.endswith('.pdf') or fname.endswith('.txt'):
                    texts.extend(load_literature([os.path.join(src, fname)]))
        else:
            texts.append(src[:5000])  # 直接当作文本
    return "\n\n---\n\n".join(texts)


def assemble_report(all_results):
    """组装最终的标准化报告"""
    problems = all_results.get("problems", {})
    rationale = all_results.get("rationale", {})
    methods = all_results.get("methods", {})
    experiments = all_results.get("experiments", {})
    writing = all_results.get("writing", {})

    report = f"""# 科学假设与研究计划
*生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

---

## 论文标题 (Paper Title)
{writing.get('paper_title', 'N/A')}

---

## 摘要 (Abstract)
{writing.get('abstract', 'N/A')}

**关键词**：{', '.join(writing.get('keywords', []))}

---

## 1. 待研究问题 (Problem Statement)
{problems.get('primary_problem', 'N/A')}

### 已识别的具体局限性
"""
    for i, p in enumerate(problems.get('problems', []), 1):
        report += f"\n{i}. **{p.get('issue', 'N/A')}** (严重程度: {p.get('severity', 'N/A')})\n   - 证据: {p.get('evidence', 'N/A')}\n"

    report += f"""

---

## 2. 解决思路 (Rationale)
{rationale.get('rationale', 'N/A')}

### 创新点
{rationale.get('innovation_angle', 'N/A')}

### 可行性论证
{rationale.get('feasibility', 'N/A')}

---

## 3. 技术手段 (Technical Details)
"""
    tech = methods.get('technical_stack', {})
    for category, items in tech.items():
        report += f"\n### {category}\n"
        if isinstance(items, list):
            for item in items:
                report += f"- {item}\n"
        else:
            report += f"{items}\n"

    report += f"""
### 模型架构
{methods.get('model_architecture', 'N/A')}

---

## 4. 方法论 (Methods)
"""
    for step, desc in methods.get('methods', {}).items():
        report += f"\n**{step}**：{desc}\n"

    report += f"""
### 预期挑战
"""
    for c in methods.get('expected_challenges', []):
        report += f"- {c}\n"

    report += f"""

---

## 5. 数据集 (Datasets)
### Source（历史数据）
"""
    ds = experiments.get('datasets', {})
    src_data = ds.get('source', {})
    report += f"- **名称**：{src_data.get('name', 'N/A')}\n"
    report += f"- **来源**：{src_data.get('url', 'N/A')}\n"
    report += f"- **说明**：{src_data.get('description', 'N/A')}\n"

    report += f"""
### Target（拟采集数据）
"""
    tgt_data = ds.get('target', {})
    report += f"- **特征**：{', '.join(tgt_data.get('features', []))}\n"
    report += f"- **采集方式**：{tgt_data.get('collection_method', 'N/A')}\n"

    report += f"""

---

## 6. 实验设计 (Experiments)
### 基线 (Baselines)
"""
    for b in experiments.get('baselines', []):
        report += f"- **{b.get('name', 'N/A')}**：{b.get('description', 'N/A')}\n"

    report += f"""
### 评估指标 (Metrics)
"""
    for m in experiments.get('metrics', []):
        report += f"- **{m.get('name', 'N/A')}**：{m.get('formula', 'N/A')}\n  - 解读：{m.get('interpretation', 'N/A')}\n"

    report += f"""

---

## 7. 预期结果 (Results)
"""
    er = experiments.get('expected_results', {})
    report += f"""
### 定量预期
{er.get('quantitative', 'N/A')}

### 定性预期
{er.get('qualitative', 'N/A')}

### 可行性分析
{er.get('feasibility_analysis', 'N/A')}

---

## 8. 参考文献 (References)
"""
    for i, ref in enumerate(rationale.get('related_work', []), 1):
        report += f"\n[{i}] {ref.get('authors', 'N/A')} ({ref.get('year', 'N/A')}). *{ref.get('title', 'N/A')}*. 相关性：{ref.get('relevance', 'N/A')}\n"

    report += """

---

*本报告由 MultiAgent-Scientist 系统自动生成，基于 Qwen 千问系列大模型。*
"""
    return report


# ============================================================
# 主流水线
# ============================================================
def run_pipeline(literature_sources, domain, output_dir="outputs", mode="dashscope", model_override=None, output_paper=False):
    """运行完整的五智能体流水线"""

    print("=" * 60)
    print("MultiAgent-Scientist：科学假设生成系统")
    print(f"领域：{domain}")
    print(f"模式：{mode}")
    print(f"输出：{'研究计划 + 论文' if output_paper else '研究计划'}")
    print("=" * 60)

    # 1. 加载文献
    print("\n[1/7] 加载文献...")
    literature_text = load_literature(literature_sources)
    print(f"  加载了 {len(literature_text)} 字符")

    # 2. 初始化 Qwen 客户端
    print("\n[2/7] 初始化 Qwen 模型...")
    client, default_model = create_qwen_client(mode)
    model = model_override or default_model
    print(f"  模型：{model}")

    # 3-7. 五个 Agent 流水线
    all_results = {}
    all_results["_output_paper"] = output_paper

    # Agent 1: 问题分析
    print("\n[3/7] Agent 1: 问题分析...")
    problems = agent_problem_analysis(client, model, literature_text, domain, mode)
    all_results["problems"] = problems
    print(f"  识别了 {len(problems.get('problems', []))} 个问题")
    print(f"  核心问题：{problems.get('primary_problem', 'N/A')[:100]}...")

    # Agent 2: 知识整合
    print("\n[4/7] Agent 2: 知识整合 + 创新推导...")
    rationale = agent_knowledge_integration(client, model, literature_text, problems, domain, mode)
    all_results["rationale"] = rationale
    print(f"  创新角度：{rationale.get('innovation_angle', 'N/A')[:100]}...")

    # Agent 2.5: 搜索真实文献
    print("\n[5/8] 搜索真实文献 (OpenAlex)...")
    real_papers = []
    # 从问题分析中提取搜索关键词
    queries = [domain]
    for p in problems.get("problems", [])[:3]:
        keywords = " ".join(p.get("issue", "").split()[:5])
        if len(keywords) > 10:
            queries.append(keywords)
    for q in queries[:4]:
        try:
            papers = search_real_papers(q, limit=5)
            real_papers.extend(papers)
            print(f"  搜索 '{q[:60]}' → {len(papers)} 篇")
        except Exception as e:
            print(f"  搜索失败: {e}")
    # 去重
    seen = set()
    real_papers = [p for p in real_papers if not (p['title'] in seen or seen.add(p['title']))]
    all_results["real_papers"] = real_papers
    print(f"  共找到 {len(real_papers)} 篇真实论文")

    # Agent 3: 方法设计
    print("\n[6/8] Agent 3: 方法设计...")
    methods = agent_method_design(client, model, problems, rationale, domain, mode)
    all_results["methods"] = methods
    print(f"  技术栈类别：{list(methods.get('technical_stack', {}).keys())}")

    # Agent 4: 实验规划
    print("\n[7/8] Agent 4: 实验规划...")
    experiments = agent_experiment_planning(client, model, problems, rationale, methods, domain, mode)
    all_results["experiments"] = experiments
    n_baselines = len(experiments.get('baselines', []))
    n_metrics = len(experiments.get('metrics', []))
    print(f"  基线 {n_baselines} 个，指标 {n_metrics} 个")

    # Agent 5: 综合写作
    print("\n[8/8] Agent 5: 综合写作...")
    writing = agent_synthesis_writing(client, model, all_results, domain, mode)
    all_results["writing"] = writing
    print(f"  标题：{writing.get('paper_title', 'N/A')}")

    # Agent 6: 论文写作（可选）
    if all_results.get("_output_paper"):
        print("\n[9/9] Agent 6: 生成完整论文...")
        paper = agent_paper_writing(client, model, all_results, domain, mode, all_results.get("real_papers", []))
        all_results["paper"] = paper
        print(f"  论文章节数：{len([k for k in paper if k != 'references'])}")

    # 组装报告
    print("\n" + "=" * 60)
    print("组装最终报告...")
    report = assemble_report(all_results)

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"hypothesis_report_{timestamp}.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    json_path = os.path.join(output_dir, f"raw_data_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n报告已保存：{report_path}")
    print(f"原始数据：{json_path}")

    # 如果生成论文，保存 .tex 文件
    if all_results.get("_output_paper") and all_results.get("paper"):
        paper_tex = assemble_paper(all_results)
        tex_path = os.path.join(output_dir, f"full_paper_{timestamp}.tex")
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(paper_tex)
        print(f"论文 LaTeX：{tex_path}")
        print("  → 上传到 overleaf.com 编译出 PDF")

    print("=" * 60)

    return report


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiAgent-Scientist")
    parser.add_argument("--sources", nargs="+", required=True,
                        help="文献来源：PDF文件路径 / TXT文件 / 目录 / 直接文本")
    parser.add_argument("--domain", type=str, required=True,
                        help="研究领域，如 '计算机视觉' '自然语言处理'")
    parser.add_argument("--mode", type=str, default="dashscope",
                        choices=["dashscope", "native", "local"],
                        help="Qwen 访问模式")
    parser.add_argument("--model", type=str, default=None,
                        help="模型名称，默认 qwen-plus")
    parser.add_argument("--output", type=str, default="outputs",
                        help="输出目录")
    parser.add_argument("--output-format", type=str, default="report",
                        choices=["report", "paper", "both"],
                        help="report=研究计划, paper=完整论文, both=两者都要")
    args = parser.parse_args()

    report = run_pipeline(
        literature_sources=args.sources,
        domain=args.domain,
        output_dir=args.output,
        mode=args.mode,
        model_override=args.model,
        output_paper=(args.output_format in ("paper", "both")),
    )
    print("\n" + report[:500])
