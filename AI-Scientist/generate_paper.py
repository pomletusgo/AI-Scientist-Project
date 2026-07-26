#!/usr/bin/env python
"""Generate a complete AI Scientist paper end-to-end using DeepSeek V4 Pro."""

import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DEEPSEEK_API_KEY", os.environ.get("DEEPSEEK_API_KEY", ""))

from ai_scientist.llm import create_client, get_response_from_llm, extract_json_between_markers

np.random.seed(42)
client, model_name = create_client("deepseek-v4-pro")
print(f"Using model: {model_name}")

# ============================================================
# STEP 1: Generate research idea
# ============================================================
print("=" * 60)
print("STEP 1: Generating Research Idea")
print("=" * 60)

with open("templates/nanoGPT/prompt.json") as f:
    prompt = json.load(f)
with open("templates/nanoGPT/experiment.py") as f:
    code = f.read()

idea_prompt = prompt["task_description"] + "\n\n<experiment.py>\n" + code[:3000] + "\n</experiment.py>\n\n"
idea_prompt += """Come up with ONE impactful and creative research idea for improving the NanoGPT model.
Focus on something novel yet feasible. Be specific about implementation.

Respond in JSON:
```json
{"Name": "...", "Title": "...", "Experiment": "(detailed plan, 200+ words)", "Interestingness": X, "Feasibility": X, "Novelty": X}
```"""

text, _ = get_response_from_llm(idea_prompt, client=client, model=model_name,
                                 system_message=prompt["system"], temperature=0.8)
idea = extract_json_between_markers(text)
assert idea, "Failed to extract idea JSON"
print(f"Idea: {idea['Name']}")
print(f"Title: {idea['Title']}")
print(f"Scores: I={idea['Interestingness']}/10 F={idea['Feasibility']}/10 N={idea['Novelty']}/10")

# ============================================================
# STEP 2: Create mock experiment results
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Creating Mock Experiment Results")
print("=" * 60)

results = {
    "baseline": {
        "train_loss": round(float(np.random.normal(2.35, 0.05)), 3),
        "val_loss": round(float(np.random.normal(2.68, 0.05)), 3),
        "test_loss": round(float(np.random.normal(2.71, 0.05)), 3),
        "perplexity": round(float(np.random.normal(15.02, 0.5)), 2),
        "train_time_s": 1847,
    },
    "proposed": {
        "train_loss": round(float(np.random.normal(2.08, 0.05)), 3),
        "val_loss": round(float(np.random.normal(2.41, 0.05)), 3),
        "test_loss": round(float(np.random.normal(2.44, 0.05)), 3),
        "perplexity": round(float(np.random.normal(11.47, 0.5)), 2),
        "train_time_s": 1923,
    },
    "ablation_1": {
        "desc": "Without component A",
        "test_loss": round(float(np.random.normal(2.58, 0.05)), 3),
        "perplexity": round(float(np.random.normal(13.21, 0.5)), 2),
    },
    "ablation_2": {
        "desc": "With alternative design B",
        "test_loss": round(float(np.random.normal(2.53, 0.05)), 3),
        "perplexity": round(float(np.random.normal(12.55, 0.5)), 2),
    },
}
b = results["baseline"]
p = results["proposed"]
improvement = round((b["test_loss"] - p["test_loss"]) / b["test_loss"] * 100, 1)
ppl_improvement = round((b["perplexity"] - p["perplexity"]) / b["perplexity"] * 100, 1)
results["proposed"]["improvement_pct"] = improvement
results["proposed"]["ppl_improvement_pct"] = ppl_improvement
results["proposed"]["p_value"] = 0.003

print(f"Baseline: test_loss={b['test_loss']}, perplexity={b['perplexity']}")
print(f"Proposed: test_loss={p['test_loss']}, perplexity={p['perplexity']}")
print(f"Improvement: {improvement}% test loss, {ppl_improvement}% perplexity")

# ============================================================
# STEP 3: Generate paper sections
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Writing Paper Sections")
print("=" * 60)

paper_sys = """You are an AI researcher writing for NeurIPS/ICML.
Write precisely, rigorously, with LaTeX math ($...$ inline, $$...$$ display).
Only report results that are explicitly provided.
Be honest about limitations. Do not fabricate references."""

sections = {}
section_prompts = {
    "Abstract": f"""Write a 150-200 word Abstract for: "{idea['Title']}"

Core idea: {idea['Experiment'][:300]}

Key results: Test loss improved from {b['test_loss']} to {p['test_loss']} ({improvement}%). Perplexity from {b['perplexity']} to {p['perplexity']} ({ppl_improvement}%).

Write one continuous paragraph. Be specific about the method and results.""",

    "Introduction": f"""Write the Introduction (~400 words) for: "{idea['Title']}"

1. Motivate why improving efficient language models matters
2. Identify the specific gap we address
3. Describe our proposed approach: {idea['Experiment'][:200]}
4. State key result: {improvement}% test loss improvement, {ppl_improvement}% perplexity reduction
5. End with bullet-point contributions

Use \\cite for placeholder references.""",

    "Method": f"""Write the Method section (~500 words with equations).

Detailed implementation: {idea['Experiment']}

Include:
1. Problem formulation with mathematical notation
2. Baseline NanoGPT architecture recap
3. Our proposed modification (step by step, with equations)
4. Training procedure
5. Design justification

Use $$...$$ for key equations, $...$ for inline symbols.""",

    "Experimental Setup": f"""Write Experimental Setup (~300 words).

Datasets: shakespeare_char, enwik8, text8 (character-level)
Model: GPT-based transformer, same base config as NanoGPT
Baseline: vanilla NanoGPT (dropout=0.2, AdamW, cosine LR)
Proposed: our modified model
Metrics: train/val/test loss, perplexity
3 random seeds, report mean +/- std""",

    "Results": f"""Write the Results section (~400 words) using EXACTLY these numbers:

| Metric | Baseline | Ours | Improvement |
|--------|----------|------|-------------|
| Train Loss | {b['train_loss']} | {p['train_loss']} | {round((b['train_loss']-p['train_loss'])/b['train_loss']*100,1)}% |
| Val Loss | {b['val_loss']} | {p['val_loss']} | {round((b['val_loss']-p['val_loss'])/b['val_loss']*100,1)}% |
| Test Loss | {b['test_loss']} | {p['test_loss']} | {improvement}% |
| Perplexity | {b['perplexity']} | {p['perplexity']} | {ppl_improvement}% |

Ablation:
- {results['ablation_1']['desc']}: test_loss={results['ablation_1']['test_loss']}, ppl={results['ablation_1']['perplexity']}
- {results['ablation_2']['desc']}: test_loss={results['ablation_2']['test_loss']}, ppl={results['ablation_2']['perplexity']}

Statistical significance: p = {p['p_value']} (paired t-test, 3 seeds).
Training time: comparable ({b['train_time_s']}s vs {p['train_time_s']}s).

Discuss main findings, ablation insights, and limitations honestly.""",

    "Conclusion": f"""Write the Conclusion (~150 words).

Summarize: what we proposed, key result ({improvement}% improvement), implications, and 2-3 future directions.""",
}

for i, (sec_name, sec_prompt) in enumerate(section_prompts.items()):
    print(f"[{i+1}/6] Writing {sec_name}...")
    text, _ = get_response_from_llm(sec_prompt, client=client, model=model_name,
                                     system_message=paper_sys, temperature=0.7)
    sections[sec_name.lower().replace(" ", "_")] = text.strip()
    print(f"   {sec_name}: {len(text.split())} words")

# ============================================================
# STEP 4: Assemble paper
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Assembling Complete Paper")
print("=" * 60)

template = """# {title}

**Generated by AI Scientist (SakanaAI) + DeepSeek V4 Pro**
**Date: {date}**

---

## Abstract

{abstract}

---

## 1. Introduction

{introduction}

---

## 2. Method

{method}

---

## 3. Experimental Setup

{experimental_setup}

---

## 4. Results

{results}

---

## 5. Conclusion

{conclusion}

---

## Appendix: Reproducibility

| Item | Value |
|------|-------|
| Model | GPT-based Transformer (NanoGPT) |
| Datasets | shakespeare_char, enwik8, text8 |
| Optimizer | AdamW (lr=0.001, weight_decay=0.1) |
| Seeds | 3 independent runs |
| Hardware | Simulated (8x NVIDIA H100 recommended) |

---

*This paper was automatically generated by the AI Scientist pipeline with DeepSeek V4 Pro.
Experimental results are simulated for demonstration. Real execution requires GPU hardware.
The idea, method design, and paper text were generated by LLM without human intervention.*
"""

paper = template.format(
    title=idea["Title"],
    date=time.strftime("%Y-%m-%d"),
    abstract=sections["abstract"],
    introduction=sections["introduction"],
    method=sections["method"],
    experimental_setup=sections["experimental_setup"],
    results=sections["results"],
    conclusion=sections["conclusion"],
)

os.makedirs("results", exist_ok=True)
output_path = "results/generated_paper.md"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(paper)

# Save metadata
with open("results/generated_paper_idea.json", "w") as f:
    json.dump({"idea": idea, "mock_results": results}, f, indent=4)

print(f"\nPaper saved: {output_path}")
print(f"Total: {len(paper.split())} words, {len(paper)} chars")
word_counts = {k: len(v.split()) for k, v in sections.items()}
print(f"Sections: " + " | ".join(f"{k}({v}w)" for k, v in word_counts.items()))
print(f"\nTop sections by length: {max(word_counts, key=word_counts.get)} ({max(word_counts.values())} words)")
