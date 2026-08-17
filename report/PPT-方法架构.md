# AI-Scientist 系统架构

> 用于 PPT 展示：一页方法架构图 + 一页领域调研

---

## 一、整体架构（适合 PPT 流程图）

```
                        ┌─────────────────────────────────────────┐
                        │           AI-Scientist 系统              │
                        │   全自动科学发现智能体 (Sakana AI, 2024)  │
                        └─────────────────────────────────────────┘

  输入层                    处理层                          输出层
┌──────────┐    ┌──────────────────────────────┐    ┌──────────────┐
│ 代码模板  │    │                              │    │              │
│ experiment│    │  ① 想法生成 (Ideation)        │    │ ideas.json   │
│   .py    │───→│  - 读代码 + prompt           │───→│ (研究想法)    │
│          │    │  - Chain-of-Thought 推理      │    │              │
│ prompt   │    │  - 自我反思打磨 (×3轮)         │    ├──────────────┤
│  .json   │    │  - Semantic Scholar 新颖性检查 │    │              │
│          │    │                              │    │ final_info   │
│ seed     │    │  ② 实验执行 (Experiment)       │    │   .json      │
│ ideas    │    │  - Aider 自动改代码            │───→│ (实验数据)    │
│  .json   │    │  - GPU 训练 (A100)            │    │              │
│          │    │  - 错误自动修复 (×4次重试)      │    ├──────────────┤
├──────────┤    │  - 记录 notes.txt             │    │              │
│ 研究主题  │    │                              │    │ *.png        │
│ 领域描述  │───→│  ③ 论文撰写 (Write-up)        │───→│ (训练曲线图)  │
│          │    │  - 逐节生成 (7节)              │    │              │
│          │    │  - Semantic Scholar 引用搜索    │    ├──────────────┤
│          │    │  - LaTeX 自动纠错              │    │              │
│          │    │  - pdflatex 编译               │    │ paper.pdf    │
│          │    │                              │    │ (完整论文)    │
│          │    │  ④ 自动评审 (Review)           │    │              │
│          │    │  - NeurIPS 标准评审            │───→│ review.txt   │
│          │    │  - 集成评审 (×3)               │    │ (评审意见)    │
│          │    │  - Meta-review 综合            │    │              │
│          │    └──────────────────────────────┘    └──────────────┘
└──────────┘
```

---

## 二、技术栈架构（适合展示技术细节）

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM 接口层 (llm.py)                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ DeepSeek │ │  OpenAI  │ │ Anthropic│ │    Google    │  │
│  │ V4 Pro   │ │ GPT-4o   │ │  Claude  │ │   Gemini     │  │
│  │ V4 Flash │ │ o1/o3    │ │  Sonnet  │ │  1.5/2.0     │  │
│  │ R1       │ │          │ │          │ │              │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘  │
│       └─────────────┴────────────┴──────────────┘          │
│                     统一调用接口                              │
│            create_client() + get_response_from_llm()         │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ Ideation │   │Experiment│   │ Write-up │
    │  Agent   │   │  Agent   │   │  Agent   │
    └────┬─────┘   └────┬─────┘   └────┬─────┘
         │              │              │
    ┌────▼─────┐   ┌────▼─────┐   ┌────▼─────┐
    │CoT +     │   │Aider     │   │LaTeX     │
    │Reflection│   │Code Edit │   │Compiler  │
    │Novelty   │   │GPU Train │   │Citation  │
    │Check     │   │Auto Fix  │   │Search    │
    └──────────┘   └──────────┘   └──────────┘
                          │
                    ┌─────▼─────┐
                    │  Review   │
                    │  Agent    │
                    │ Ensemble  │
                    │ NeurIPS   │
                    │ Standards │
                    └───────────┘
```

---

## 三、数据流（适合展示模块间关系）

```
templates/nanoGPT/                    results/nanoGPT/20260720_xxx/
┌─────────────────┐                   ┌─────────────────────────────┐
│ experiment.py   │──① 被读取────────→│ ideas.json                  │
│ prompt.json     │                   │ final_info.json (×6 runs)   │
│ seed_ideas.json │                   │ train_loss_*.png (×6)       │
│ plot.py         │                   │ val_loss_*.png (×6)         │
│ latex/          │                   │ notes.txt                   │
│   template.tex  │                   │ latex/                      │
└─────────────────┘                   │   paper_rich.tex ──→ PDF   │
                                      │ review.txt                  │
                                      └─────────────────────────────┘

  输入：人类提供代码模板              输出：AI 自动产出的完整论文
  (定义研究边界)                      (含图表、参考文献、评审)
```

---

## 四、关键创新点（适合单独页面展示）

| 创新点 | 描述 | 对应模块 |
|--------|------|---------|
| **自省式想法生成** | LLM 生成想法后自我批判打磨，最多 5 轮 | `generate_ideas.py` |
| **自动代码修改** | Aider 增量式编辑代码，非一次性生成 | `perform_experiments.py` |
| **错误自动恢复** | 实验失败时错误日志反馈 LLM，最多重试 4 次 | `perform_experiments.py` |
| **真实文献搜索** | Semantic Scholar / OpenAlex API 搜索真实论文 | `perform_writeup.py` |
| **集成评审** | 3 个独立评审 + meta-reviewer 综合 | `perform_review.py` |
| **模板抽象层** | 一个模板 = 一个研究领域，换模板即换方向 | `templates/` |
| **低成本** | 每篇论文 ~$15（GPT-4o），DeepSeek 更降至 ~$0.15 | 全系统 |

---

## 五、我们的改进（DeepSeek V4 适配）

```
原始系统                         我们的版本
─────────────────────          ─────────────────────
支持模型：GPT-4o / Claude       新增：DeepSeek V4 Pro/Flash/R1
想法生成：单 LLM                V4 Pro thinking 模式
论文撰写：单次生成               丰富版 9 节 800+ 词/节
文献搜索：Semantic Scholar      补增 OpenAlex（免费不限流）
LaTeX：服务器直接编译            本地 Overleaf 编译
GPU：需要 NVIDIA GPU            适配 A100 GPU
```
