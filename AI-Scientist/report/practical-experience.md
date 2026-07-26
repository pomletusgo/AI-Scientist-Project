# AI-Scientist 实操经验总结：Windows 适配、模型替换与流水线拆分

> 环境：Windows 11 + Intel Core Ultra 9 185H（Intel Arc 集显，无 NVIDIA GPU）+ Python 3.10 + Git Bash
> 目标：在无 GPU 条件下跑通 LLM 流水线，并将 DeepSeek V4 集成到系统中

---

## 1. 环境配置踩坑记录

### 1.1 GitHub 访问问题

**问题**：中国大陆网络环境下，`git clone github.com/SakanaAI/AI-Scientist` 持续超时。

**尝试过的方案**：

| 方案 | 结果 |
|------|------|
| `git clone github.com/SakanaAI/AI-Scientist --depth 1` | 超时 |
| `git clone ghproxy.net/https://github.com/...` | 超时 |
| `git clone gitclone.com/github.com/...` | 超时 |
| `git clone kkgithub.com/SakanaAI/...` | 504 错误 |
| `curl` 下载 ZIP via ghproxy | ✅ 成功（5.4MB） |

**最终方案**：通过 ghproxy.net 逐文件下载核心 Python 源码（共 7 个文件），保留了完整的模块结构。

**教训**：对于 GitHub 访问受限的环境，逐文件下载比完整克隆更可靠。核心代码文件不到 100KB，不需要整个仓库。

### 1.2 Python 环境配置

**问题**：项目要求 Python 3.11，机器上是 Python 3.10.11，且 conda 未在 PATH 中。

**解决**：
1. 直接使用系统 Python 3.10.11，实测对 LLM 流水线部分完全兼容
2. 安装精简依赖集（跳过 GPU 相关包）：

```bash
pip install openai backoff numpy pypdf pymupdf pymupdf4llm anthropic
```

**不需要安装的包**（仅 GPU 实验需要）：
- `torch`（已有 CPU 版本）
- `transformers`、`datasets`（模板实验需要）
- `aider-chat`（写论文需要，但可以绕过）

**教训**：AI-Scientist 的模块化程度比表面看起来好。LLM 相关模块（`llm.py`、`perform_review.py`、`generate_ideas.py`）完全不依赖 GPU，可以独立运行。

### 1.3 Windows 兼容性

**测试结果**：
- `os.path` 操作在 Git Bash 环境下正常工作
- `subprocess.run()` 调用需要 Windows 版本的 LaTeX（非必需）
- 无换行符相关问题

**关键发现**：AI-Scientist 的核心 LLM 逻辑是**平台无关的**——只有实验执行部分需要 Linux + NVIDIA GPU。

---

## 2. LLM 流水线拆分方法

### 2.1 模块独立性分析

通过代码走读，我们确定了各模块的依赖关系：

```
模块                     依赖                    GPU需求    可独立运行
──────────────────────────────────────────────────────────────────
perform_review.py        llm.py + PDF            ❌         ✅ 完全独立
generate_ideas.py        llm.py + 模板目录        ❌         ✅ 完全独立
perform_experiments.py   llm.py + Aider + GPU    ✅         ❌ 必须 GPU
perform_writeup.py       llm.py + Aider + 实验结果 ❌         ⚠️ 需模拟数据
launch_scientist.py      上述所有模块              ✅         ❌ 必须 GPU
```

### 2.2 独立运行 perform_review.py

这是**最干净**的独立模块——输入一个 PDF，输出结构化评审：

```python
from ai_scientist.perform_review import load_paper, perform_review
from ai_scientist.llm import create_client

client, model = create_client("deepseek-v4-pro")
paper_text = load_paper("attention_is_all_you_need.pdf")

review = perform_review(
    paper_text, model=model, client=client,
    num_reflections=5,
    num_fs_examples=1,
    num_reviews_ensemble=5,  # 集成 5 个评审
    temperature=0.1,
)

# 输出示例：
# {"Overall": 8, "Soundness": 4, "Presentation": 3,
#  "Contribution": 4, "Decision": "Accept", ...}
```

**关键细节**：
- `num_reviews_ensemble=5` 会生成 5 个独立评审，然后用 meta-reviewer 综合——类似真实会议的 Area Chair 流程
- 使用 `reviewer_system_prompt_neg`（保守/严格模式）来避免放行低质量论文
- few-shot 示例需要预先准备的 PDF + JSON 文件对（`fewshot_examples/` 目录）

### 2.3 独立运行 generate_ideas.py

需要模板目录结构，但不需要 GPU：

```python
from ai_scientist.generate_ideas import generate_ideas
from ai_scientist.llm import create_client

client, model = create_client("deepseek-v4-pro")

ideas = generate_ideas(
    base_dir="templates/nanoGPT",
    client=client, model=model,
    skip_generation=False,
    max_num_generations=5,
    num_reflections=3,
)
# 输出：5 个 JSON 格式的研究想法，写入 templates/nanoGPT/ideas.json
```

**关键细节**：
- 模板中的 `prompt.json` 定义了研究领域的上下文（`task_description`）和系统提示词
- `seed_ideas.json` 提供初始的种子想法，后续想法在此基础上迭代
- 每个想法包含"Name/Title/Experiment/Interestingness/Feasibility/Novelty"六个字段

### 2.4 模拟实验数据的"桥梁模式"

为了让 writeup 模块在没有真实 GPU 实验的情况下运行，我们设计了一个**桥梁模式**：

```
Ideation → [Mock Experiment Data] → Writeup → Review
              ↑
        构造 notes.txt + final_info.json
```

核心是构造两个文件：

**`run_0/final_info.json`**（基线实验的数值结果）：
```json
{
    "train_loss": {"means": 2.35, "std": 0.12},
    "val_loss": {"means": 2.68, "std": 0.15},
    "test_loss": {"means": 2.71, "std": 0.14}
}
```

**`notes.txt`**（实验笔记本）：
```markdown
# Title: Adaptive Attention for NanoGPT
# Experiment: Modify attention mechanism to improve perplexity
## Run 0: Baseline
Results: train_loss=2.35, val_loss=2.68, test_loss=2.71
Description: Vanilla NanoGPT with default config
## Run 1: Proposed Method
Results: train_loss=2.12, val_loss=2.45, test_loss=2.48
Description: The proposed method reduces test loss by 8.5%
```

这个桥梁模式的精妙之处在于：它保留了原始流水线的接口约定（`notes.txt` + `final_info.json` 的格式），使得 writeup 和 review 模块无需修改即可运行。

### 2.5 统一流水线脚本 `run_llm_pipeline.py`

我们创建了一个整合脚本，支持三种运行模式：

```bash
# 模式 1: 独立论文评审
python run_llm_pipeline.py --task review --paper attention.pdf --model deepseek-v4-pro

# 模式 2: 独立想法生成
python run_llm_pipeline.py --task ideation --experiment nanoGPT --model deepseek-v4-flash --num-ideas 5

# 模式 3: 完整流水线（跳过 GPU 实验）
python run_llm_pipeline.py --task pipeline --experiment nanoGPT --model deepseek-v4-pro --num-ideas 2
```

---

## 3. DeepSeek V4 模型替换实战

### 3.1 变更范围

修改了 3 个核心文件：

| 文件 | 变更内容 | 行数变化 |
|------|---------|---------|
| `ai_scientist/llm.py` | 新增 5 个模型，3 个 handler，1 个路由规则 | +35 |
| `launch_scientist.py` | Aider 模型映射（2 处） | +8 |
| `ai_scientist/perform_writeup.py` | Aider 模型映射（1 处） | +8 |

### 3.2 llm.py 关键改动

**① 模型注册**（`AVAILABLE_LLMS` 列表）：

```python
# 新增模型
"deepseek-v4-pro",      # DeepSeek V4 旗舰版（1.6T MoE, 1M context, thinking mode）
"deepseek-v4-flash",    # DeepSeek V4 轻量版（284B MoE, 速度优化）
"deepseek-r1",          # DeepSeek R1 推理模型（原始版, MIT license）
"deepseek-r1-0528",     # R1 幻觉减少版
"deepseek-v3.2",        # V3.2（V4-Flash 的别名）
```

**② 客户端路由**（`create_client()` 函数）：

```python
# 从精确匹配改为前缀匹配，自动覆盖所有 deepseek-* 模型
elif model.startswith("deepseek-") or model.startswith("deepseek_"):
    return openai.OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com"
    ), model
```

**③ API 调用 handler**（`get_response_from_llm()` 函数）：

三个独立 handler 处理不同类型的 DeepSeek 模型：

| Handler | 模型 | 关键参数 |
|---------|------|---------|
| Fast models | `deepseek-chat`, `deepseek-coder`, `deepseek-v4-flash` | `temperature`, `max_tokens` |
| V4 Pro thinking | `deepseek-v4-pro` | `temperature`, `max_tokens`, `extra_body={"thinking": {"type": "enabled"}}` |
| Reasoning models | `deepseek-reasoner`, `deepseek-r1` | 无 `temperature`（推理模型不支持） |

**④ 深度求索推理链捕获**：

```python
# V4-Pro 的 reasoning_content 包含模型的思考过程
reasoning = getattr(response.choices[0].message, "reasoning_content", None)
if reasoning and print_debug:
    print(f"[DeepSeek V4 Reasoning]: {reasoning[:500]}...")
```

### 3.3 DeepSeek 模型选择指南

在 AI-Scientist 的不同阶段，推荐使用不同的模型：

| 阶段 | 推荐模型 | 原因 |
|------|---------|------|
| Idea Generation | `deepseek-v4-pro` | 需要创造力和推理能力，thinking mode 有帮助 |
| Experiment Code | `deepseek-v4-flash` | 代码生成任务，快速 + 便宜 |
| Paper Writing | `deepseek-v4-pro` | 需要学术写作质量和推理 |
| Paper Review | `deepseek-v4-pro` | 需要批判性思维，thinking mode 使评审更深入 |
| Citation Search | `deepseek-v4-flash` | 简单的搜索→选择任务，不需要深度推理 |

### 3.4 API 兼容性注意事项

- **`frequency_penalty` / `presence_penalty`**：DeepSeek V4 已废弃这两个参数，代码中已确保不传递
- **System message**：DeepSeek V4 完全支持 system role（不像旧版 R1）
- **`thinking` 参数**：仅 V4 Pro 支持，通过 `extra_body` 传递（因为 OpenAI SDK 不认识这个参数）

### 3.5 模型能力分层常量

还添加了**能力分层常量**，便于未来参考：

```python
DEEPSEEK_REASONING_MODELS = ["deepseek-reasoner", "deepseek-r1", "deepseek-r1-0528", "deepseek-v4-pro"]
DEEPSEEK_FAST_MODELS = ["deepseek-chat", "deepseek-coder", "deepseek-v4-flash", "deepseek-v3.2"]
```

这允许未来在 `get_batch_responses_from_llm()` 中为不同类型的模型使用不同的批处理策略。

---

## 4. 各模型横向对比（预期，待实测）

基于论文和 API 文档，各模型在 AI-Scientist 场景中的预期表现：

| 模型 | 创造力 | 代码能力 | 学术写作 | 评审质量 | 成本 | 速度 |
|------|--------|---------|---------|---------|------|------|
| GPT-4o | ★★★★ | ★★★★ | ★★★★ | ★★★★ | $$$$ | 中等 |
| Claude 3.5 Sonnet | ★★★★★ | ★★★★ | ★★★★★ | ★★★★★ | $$$$ | 中等 |
| DeepSeek V4 Pro | ★★★★ | ★★★★★ | ★★★★ | ★★★★ | $ | 快 |
| DeepSeek V4 Flash | ★★★ | ★★★★ | ★★★ | ★★★ | $ | 极快 |
| DeepSeek R1 | ★★★★ | ★★★ | ★★★ | ★★★★ | $$ | 慢（推理链） |

**关键洞察**：DeepSeek V4 在 AI-Scientist 中的性价比优势明显。$15/篇的原始成本可以通过 V4 Flash 进一步降低到 ~$3-5/篇。

---

## 5. 最佳实践建议

### 5.1 成本控制

1. **分层模型策略**：想法生成和评审用 V4 Pro（需要深度推理），代码生成和引用搜索用 V4 Flash（速度快、成本低）
2. **减少 reflection 轮数**：`num_reflections=3`（默认 5）在大多数情况下足够
3. **减少 ensemble 数量**：`num_reviews_ensemble=3`（默认 5）可以有效降低评审成本

### 5.2 错误处理

1. **API 超时**：`llm.py` 中的 `@backoff.on_exception` 装饰器处理了 `RateLimitError` 和 `APITimeoutError`，但并未覆盖所有错误（如网络中断）
2. **JSON 解析**：`extract_json_between_markers()` 有三层回退：精确匹配 ` ```json ... ``` ` → 正则匹配 JSON 对象 → 清理控制字符后重试
3. **论文加载**：`load_paper()` 有三层回退：pymupdf4llm → pymupdf → pypdf

### 5.3 Prompt 调优

1. **想法生成**：`prompt.json` 中的 `task_description` 是关键——它定义了 LLM 思考的边界
2. **评审严格度**：默认使用 `reviewer_system_prompt_neg`（保守模式），如果希望更宽容的结果，可以切换到 `reviewer_system_prompt_pos`
3. **删除已废弃的参数**：OpenAI 和 DeepSeek 的某些参数已废弃（如 `frequency_penalty`），定期检查 API 文档

### 5.4 环境配置

1. **Windows 用户**：使用 Git Bash + Python 3.10+ 即可运行 LLM 流水线，无需 WSL 或 Docker
2. **无 GPU 用户**：聚焦 review 和 ideation 模块，用模拟实验数据测试 writeup
3. **API Key 管理**：使用 `.env` 文件，不要硬编码 API key

---

## 6. 修改后的代码文件清单

| 文件 | 用途 |
|------|------|
| `ai_scientist/llm.py` | LLM 接口（新增 DeepSeek V4 支持） |
| `launch_scientist.py` | 主入口（新增 DeepSeek V4 Aider 映射） |
| `ai_scientist/perform_writeup.py` | 论文撰写（新增 DeepSeek V4 Aider 映射） |
| `run_llm_pipeline.py` | **新建** - 跳过 GPU 的 LLM 流水线 |
| `.env.template` | **新建** - API 配置模板 |
| `report/AI-Scientist-deep-dive.md` | **新建** - 深度解读报告 |
| `report/practical-experience.md` | **新建** - 本文档 |
