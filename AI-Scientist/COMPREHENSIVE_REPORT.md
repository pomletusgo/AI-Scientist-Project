# AI-Scientist Comprehensive Analysis Report

**Date:** 2026-07-13
**Models:** DeepSeek V4-Pro, V4-Flash, R1-0528

---

## 1. Paper Deep Interpretation

### 1.1 Core Vision

The AI-Scientist paper (arXiv:2408.06292, Lu et al., 2024; published in Nature, 2025)
proposes a system that fully automates the scientific discovery process for computational
research. The central claim: foundation models can now independently conduct end-to-end
research from hypothesis generation through experimental validation to manuscript
preparation and peer review.

Three key philosophical contributions:

1. **Automated Science as an Engineering Problem**: Scientific discovery reframed as
   a pipeline of well-defined stages, each amenable to LLM-based automation.

2. **Scale Enables Discovery**: The system can explore thousands of research directions
   in parallel, discovering insights missed by human researchers.

3. **The Review-Discovery Loop**: The automated reviewer creates a closed feedback loop
   where evaluations drive iterative improvement of both paper and research direction.

### 1.2 Pipeline Architecture (7 Stages)

**Stage 1: Ideation** - LLM brainstorms novel research directions given topic description,
baseline code template, and previously generated ideas. Prompt provides full baseline code
as context, asks LLM to reason about feasible modifications, requires structured JSON
output with novelty/feasibility/interestingness ratings.

**Stage 2: Novelty Check** - Uses Semantic Scholar API to verify idea novelty. Searches
for papers matching keywords from idea description. Known limitation: keyword-level
search rather than deep semantic understanding.

**Stage 3: Code Generation (Aider)** - Uses Aider AI pair-programming tool to modify
baseline experiment code. LLM proposes changes as search/replace diffs. System provides
run results after each modification for iterative debugging.

**Stage 4: Experiment Execution (GPU-dependent)** - Modified code runs in sandboxed
Docker with GPU. Each run produces final_info.json with metrics. Failures trigger debug
loop (up to 4 attempts). ~38-42% failure rate in practice.

**Stage 5: Paper Writing** - Fills LaTeX template section by section using experiment
results. Citation pipeline searches Semantic Scholar for references. Two full refinement
passes with section-specific tips.

**Stage 6: Automated Review** - Ensemble of 5 independent LLM reviewers using NeurIPS
guidelines. Meta-reviewer aggregates ensemble. Achieved 69% balanced accuracy; F1-score
exceeded inter-human agreement measured at NeurIPS 2021.

### 1.3 Technical Innovations

**Innovation 1: Template-Based Domain Scaffolding** - Each research domain is encapsulated
as a template (nanoGPT, 2D diffusion, grokking) containing baseline code, plot generation,
LaTeX skeleton, and task description. The LLM pipeline is domain-agnostic; templates
provide domain grounding. This is the most reusable architectural insight.

**Innovation 2: Structured LLM Orchestration** - Different pipeline stages use different
models optimized for the task. Different prompting strategies per stage: structured JSON
for ideation, search/replace diffs for code, free-form LaTeX for papers.

**Innovation 3: Ensemble Reviewing** - 5 independent reviews aggregated by meta-reviewer.
Captures diverse perspectives, reduces individual reviewer bias.

**Innovation 4: Iterative Refinement Loops** - Every stage includes reflection cycles:
ideas refined through multiple rounds, code debugged iteratively, papers go through
two full refinement passes.

### 1.4 Critical Analysis

**Strengths:** Pipeline architecture is genuinely novel; template system enables
extensibility; automated reviewer is a significant contribution; published in Nature;
v2 paper passed human peer review at ICLR 2025 workshop (score 6.33/10).

**Weaknesses:** Template-dependent (cannot generate novel paradigms); novelty detection
via keyword search is shallow; ~40% failure rate on code generation; citations often
hallucinated or outdated; no visual figure inspection in v1; domain restricted to
computational/ML experiments.

### 1.5 v1 vs v2 Comparison

| Feature | v1 | v2 |
|---------|-----|-----|
| Approach | Template-following linear pipeline | Agentic Best-First Tree Search |
| Paper Cost | ~5/paper (Claude Sonnet) | ~0-25/run |
| Review Method | LLM text-only | LLM + VLM figure feedback |
| Peer Review | Credible-looking papers | First AI paper to pass human review |
| Success Rate | Higher (well-defined) | Lower but broader exploration |

---

## 2. Code-to-Paper Mapping

### 2.1 Module Architecture

| Paper Section | Code Module | Lines | Description |
|--------------|-------------|-------|-------------|
| Ideation | ai_scientist/generate_ideas.py | ~300 | LLM brainstorming + novelty check |
| LLM Interface | ai_scientist/llm.py | ~380 | Model registry, client factory |
| Experiments | ai_scientist/perform_experiments.py | ~280 | Sandboxed execution |
| Paper Writing | ai_scientist/perform_writeup.py | ~500 | LaTeX + citation pipeline |
| Review | ai_scientist/perform_review.py | ~350 | Ensemble reviewing |
| Orchestration | launch_scientist.py | ~350 | Main entry, parallel exec |
| **NEW** Mock | ai_scientist/mock_experiments.py | ~120 | GPU-free mock generator |
| **NEW** Runner | run_pipeline_no_gpu.py | ~170 | GPU-free pipeline runner |

### 2.2 Data Flow

ideas.json -> experiment.py (Aider) -> run_N/final_info.json ->
notes.txt -> latex/template.tex -> paper.pdf -> review.json

### 2.3 Key Implementation Details

**LLM Response Parsing:** Two-stage JSON extraction - first tries markdown code blocks,
falls back to regex pattern matching. Handles common LLM output formatting issues.

**Aider Integration:** Uses aider-chat as intermediary between LLM and codebase.
Model names must be translated: deepseek-v4-flash -> Model(deepseek/deepseek-chat).

**Backoff Retry:** All API calls use @backoff.on_exception with exponential backoff
for RateLimitError and APITimeoutError, ensuring robustness in long autonomous runs.

**Prompt Engineering for Review:** NeurIPS guidelines combined with structured JSON
output requirements. Ensemble uses get_batch_responses_from_llm with n_responses=5
and temperature=0.75 to encourage diverse perspectives.

---

## 3. DeepSeek Model Integration

### 3.1 Models Added to AVAILABLE_LLMS

| Model | API ID | Parameters | Context | Best For |
|-------|--------|------------|---------|----------|
| V4-Pro | deepseek-v4-pro | 1.6T MoE (~49B active) | 1M tokens | Complex reasoning |
| V4-Flash | deepseek-v4-flash | 284B MoE (~13B active) | 1M tokens | Fast ideation |
| V3.2 | deepseek-v3.2 | Legacy alias | 128K | Backward compat |
| R1 | deepseek-r1 | Original R1 | 128K | Math reasoning |
| R1-0528 | deepseek-r1-0528 | R1 with fixes | 128K | Reduced hallucination |

### 3.2 Model Capability Tiers

**DEEPSEEK_REASONING_MODELS** (R1-family + V4-Pro thinking mode):
- Do NOT support system role messages
- Do NOT support temperature parameter (except V4-Pro)
- System prompt folded into user message
- Optimized for multi-step reasoning and careful analysis

**DEEPSEEK_FAST_MODELS** (V4-Flash, V3.2, legacy):
- Full OpenAI-compatible API
- Support system messages, temperature, max_tokens
- Optimized for speed and cost-efficiency

### 3.3 Recommended Model Assignments

| Pipeline Stage | Model | Rationale |
|---------------|-------|-----------|
| Ideation | deepseek-v4-flash | Fast, cheap, good creativity |
| Code Generation | deepseek-v4-pro | SWE-bench 80.6%, strong coding |
| Paper Writing | deepseek-v4-pro | Long-form reasoning, 1M context |
| Citations | deepseek-v4-flash | Fast, frequent API queries |
| Review | deepseek-r1-0528 | Best reasoning, lowest hallucination |
| Meta-Review | deepseek-v4-pro | Aggregation of perspectives |

### 3.4 API Routing

All DeepSeek models route through a single OpenAI-compatible client:
openai.OpenAI(base_url="https://api.deepseek.com", api_key=DEEPSEEK_API_KEY)

The create_client() function routes all 8 DeepSeek model IDs to this client.
The get_response_from_llm() function handles model-specific API differences
(system role handling, temperature support) through the capability tier system.

### 3.5 Cost Comparison

| Model | Input $/1M tok | Output $/1M tok | Per-paper est. |
|-------|---------------|-----------------|----------------|
| V4-Flash | /usr/bin/bash.10-0.30 | ~/usr/bin/bash.42 | -3 |
| V4-Pro | /usr/bin/bash.30 | /usr/bin/bash.50 | -5 |
| R1-0528 | /usr/bin/bash.55 | .19 | -7 |
| Claude 3.5 Sonnet | .00 | 5.00 | ~5 |
| GPT-4o | .00 | 5.00 | ~0 |

---

## 4. GPU-Free Pipeline Adaptation

### 4.1 Design Principle

Two dependency categories in AI-Scientist:
- **LLM-dependent** (no GPU): ideation, novelty check, paper writing, review
- **GPU-dependent**: experiment execution (CUDA, PyTorch, GPU memory)

By replacing experiment execution with mock generation, the full LLM pipeline functions.
This works because downstream stages read from files (notes.txt, final_info.json),
not from live experiments. Mock results follow the exact same JSON schema.

### 4.2 Mock Experiment Generator

mock_experiments.py design:
- Exact JSON schema match with real experiments
- Progressive metric improvement across runs (simulating refinement)
- Realistic noise to avoid perfectly clean trends
- Domain-specific metrics: attention_entropy (transformer), fid_score (diffusion),
  generalization_gap (grokking)
- Deterministic output via seeded RNG for reproducibility

### 4.3 Pipeline Changes

run_pipeline_no_gpu.py vs original launch_scientist.py:
1. Removed torch.cuda and GPU selection
2. Sequential execution (no hardware contention)
3. Mock generation replaces experiment execution
4. DEEPSEEK_API_KEY dependency check added
5. Detailed progress reporting with timestamps
6. Model name translation for Aider compatibility

---

## 5. Practical Experience Summary

### 5.1 Environment Challenges

**Windows 11:** No NVIDIA GPU (Intel Arc only) necessitated mock experiment system.
Python 3.10.11 installed but project needs 3.11 - conda environment recommended.
LaTeX requires MiKTeX for Windows (texlive is Linux-only).

**Network:** GitHub direct access timed out consistently. ghproxy.net worked for
individual file downloads (5-30s each). Full clone failed; 15+ files downloaded
individually via curl. DeepSeek API accessible directly.

**Model Migration:** DeepSeek API is OpenAI-compatible, minimal code changes needed.
Key difference: R1 models do not support system role. V4 models support 1M context.
Legacy model IDs being deprecated July 24, 2026. Estimated per-paper cost:
-5 (DeepSeek) vs 5-25 (Claude/GPT-4o original).

### 5.2 Key Lessons Learned

1. **LLM Pipeline Independence:** Architecture cleanly separates LLM and GPU stages.
   Enables development and testing without hardware.
2. **Template System:** Domain-specific code + domain-general LLM pipeline is the
   most reusable pattern. New domains = new templates, no pipeline changes.
3. **Prompt Engineering:** Quality depends heavily on prompt design. per_section_tips
   dictionary and NeurIPS-formatted reviewer prompt are significant engineering.
4. **Error Handling is Critical:** Backoff retry, 4-attempt debug loops, and
   5-attempt LaTeX compilation are essential for autonomous reliability.
5. **Model Selection Matters:** Different stages benefit from different capabilities.
   V4-Pro for reasoning, V4-Flash for speed, R1-0528 for careful review.
6. **Mock Results Work:** Structurally valid mock outputs enable full pipeline testing
   without GPU. Powerful technique for development and debugging.

### 5.3 Recommendations

**Getting Started:**
1. conda create -n ai_scientist python=3.11
2. pip install openai anthropic backoff numpy matplotlib pypdf
3. export DEEPSEEK_API_KEY=your-key
4. python run_pipeline_no_gpu.py --num-ideas 1 --mock-runs 2
5. Scale gradually as pipeline stabilizes

**Model Selection:** Default ideation/writing: deepseek-v4-flash (fast, cheap).
Critical stages: deepseek-v4-pro (code gen, meta-review).
Max accuracy: deepseek-r1-0528 (review with lowest hallucination).

**Production Path:** Add GPU -> real experiments -> explore v2 agentic tree search ->
add VLM figure review -> parallel execution -> W&B logging.

---

## File Inventory

**Modified from original:**
- ai_scientist/llm.py -- DeepSeek V4/R1 support, optional Gemini import

**Created new:**
- ai_scientist/mock_experiments.py -- GPU-free experiment generator
- run_pipeline_no_gpu.py -- Pipeline runner without GPU dependency
- COMPREHENSIVE_REPORT.md -- This report

**Downloaded from original (via ghproxy mirror):**
- ai_scientist/generate_ideas.py, perform_experiments.py, perform_review.py, perform_writeup.py
- launch_scientist.py
- templates/nanoGPT/experiment.py, plot.py, prompt.json, seed_ideas.json
- templates/nanoGPT/latex/template.tex, templates/nanoGPT/run_0/final_info.json
- ai_scientist/fewshot_examples/* (4 example papers + reviews)

---

## References

1. Lu, C. et al. The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. arXiv:2408.06292 (2024). Published in Nature (2025).
2. Yamada, Y. et al. AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search. arXiv:2504.08066 (2025).
3. SakanaAI. AI-Scientist. https://github.com/SakanaAI/AI-Scientist
4. DeepSeek. V4 Preview Release (April 24, 2026). https://api-docs.deepseek.com/news/news260424
5. DeepSeek API Documentation. https://api-docs.deepseek.com/
6. Beel, J. et al. Evaluating Sakana AI Scientist. arXiv:2502.14297 (2025).
