# AI-Scientist 领域调研与相关工作

> 用于 PPT / Word 展示：自动化科学发现 + LLM 智能体 + 对比分析

---

## 一、领域全景图

```
                    自动化科学发现 (Automated Scientific Discovery)
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
  符号推理方法            机器学习方法           LLM 智能体方法
  (1990s-2010s)          (2010s-2020s)          (2023-至今)
       │                     │                     │
  BACON, AM           AlphaFold, RoseTTA     AI-Scientist (Sakana AI)
  (定律再发现)         (蛋白质结构预测)        PaperQA2, SciAgents
                       GNoME (材料发现)       ChemCrow, Coscientist
                       FunSearch (数学)       ResearchGPT
```

---

## 二、核心相关论文列表

### 2.1 直接相关工作：LLM 驱动的科学发现

| 论文 | 机构 | 年份 | 核心贡献 | 与 AI-Scientist 关系 |
|------|------|------|---------|---------------------|
| **AI Scientist** (Lu et al.) | Sakana AI | 2024.08 | 首个端到端全自动科研系统 | 本文基准 |
| **AI Scientist v2** (Yamada et al.) | Sakana AI | 2025.04 | 模板无关 + 树搜索 | 升级版 |
| **PaperQA2** (Skarlinski et al.) | FutureHouse | 2024 | LLM 文献综述智能体 | 互补：侧重文献搜索 |
| **ChemCrow** (Bran et al.) | EPFL | 2024 | LLM 化学实验设计 | 不同领域：化学 |
| **Coscientist** (Boiko et al.) | CMU | 2023 | LLM + 湿实验自动化 | 不同领域：生物 |
| **SciAgents** (Ghafarollahi et al.) | MIT | 2024 | 多智能体科研系统 | 架构类似 |
| **FunSearch** (Romera-Paredes et al.) | DeepMind | 2024 | LLM 数学发现 | 不同领域：数学 |

### 2.2 LLM 智能体基础

| 论文 | 核心贡献 |
|------|---------|
| **ReAct** (Yao et al., 2023) | 推理+行动交替的 LLM 智能体框架 |
| **AutoGPT / BabyAGI** (2023) | 自主任务分解与执行 |
| **Toolformer** (Schick et al., 2023) | LLM 学会使用外部工具 |
| **SWE-Agent** (Yang et al., 2024) | LLM 自动修代码 |
| **Aider** (Gauthier, 2023) | LLM 驱动的代码编辑工具（AI-Scientist 使用） |

### 2.3 模型对齐与安全（本次复现涉及）

| 论文 | 核心贡献 |
|------|---------|
| **RLHF** (Christiano et al., 2017) | 人类反馈强化学习 |
| **DPO** (Rafailov et al., 2023) | 直接偏好优化，替代 RLHF |
| **Anthropic Constitutional AI** (Bai et al., 2022) | 宪法式 AI 对齐 |
| **Red-Teaming LMs** (Perez et al., 2022) | LLM 红队测试 |
| **Grokking** (Power et al., 2022) | 神经网络泛化现象发现 |

---

## 三、对比分析表

### 3.1 自动化科学发现系统对比

| 维度 | AI-Scientist | PaperQA2 | ChemCrow | SciAgents | FunSearch |
|------|:-----------:|:--------:|:--------:|:---------:|:---------:|
| 端到端 | ✅ | ❌ | ⚠️ | ⚠️ | ❌ |
| 自动实验 | ✅ | ❌ | ✅ | ❌ | ❌ |
| 自动写论文 | ✅ | ❌ | ❌ | ⚠️ | ❌ |
| 自动审稿 | ✅ | ❌ | ❌ | ❌ | ❌ |
| 多领域 | ✅ | ✅ | ❌(化学) | ✅ | ❌(数学) |
| 开源性 | ✅ 完整 | ✅ | ✅ | ❌ | ❌ |
| 成本 | $15/篇 | API 费 | API 费 | API 费 | 内部 |

### 3.2 模型选择对比

| 模型 | 论文质量 | 代码能力 | 成本 | 适用场景 |
|------|:--------:|:--------:|:----:|---------|
| Claude 3.5 Sonnet | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | $$$ | 论文撰写 |
| GPT-4o | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | $$$$ | 全面 |
| **DeepSeek V4 Pro** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **$** | 代码+评审 |
| **DeepSeek V4 Flash** | ⭐⭐⭐ | ⭐⭐⭐⭐ | **$** | 快速生成 |
| Llama 3.1 405B | ⭐⭐⭐ | ⭐⭐⭐ | $$ | 开源替代 |

---

## 四、技术演进路线

```
2017 ─ Transformer (Vaswani et al.)
  │
2020 ─ GPT-3 (Brown et al.) → LLM 能力飞跃
  │
2022 ─ ChatGPT → LLM 对话能力
  │     Grokking (Power et al.) → 泛化现象
  │
2023 ─ AutoGPT → 自主智能体概念
  │     Aider → LLM 代码编辑
  │     ReAct → 推理+行动框架
  │
2024 ─ AI-Scientist (Lu et al.) → 全自动科研 ★
  │     PaperQA2 → LLM 文献搜索
  │     FunSearch → LLM 数学发现
  │     ChemCrow → LLM 化学实验
  │
2025 ─ AI-Scientist v2 → 模板无关 + 树搜索
  │     多智能体科研系统兴起
  │
2026 ─ 本工作：DeepSeek V4 适配 + 多领域扩展
```

---

## 五、产业化 / 工程项目调研（至少 3 个）

> 不仅限于学术论文，已有团队将"AI 自动科研"工程化落地

### 项目 1：FutureHouse — 融资 $30M+ 的 AI 科学家公司

| 维度 | 详情 |
|------|------|
| **公司** | FutureHouse Inc.（旧金山） |
| **融资** | $30M+（Eric Schmidt 等投资） |
| **产品** | PaperQA2——开源的 AI 文献综述引擎 |
| **技术栈** | LLM + RAG + 多步推理 + PDF 全文解析 |
| **已有成果** | 在 24 小时内独立完成了关于"氨基酸传感器"的完整文献综述并生成论文草稿 |
| **与 AI-Scientist 关系** | 侧重文献搜索与综述，不做实验；AI-Scientist 端到端加入实验环节 |
| **网址** | futurehouse.org / github.com/Future-House |

### 项目 2：Google DeepMind — 工业级 AI 科学发现

| 维度 | 详情 |
|------|------|
| **公司** | Google DeepMind（伦敦） |
| **代表产品** | AlphaFold 3（蛋白质结构预测）、GNoME（材料发现）、FunSearch（数学猜想） |
| **技术栈** | 大规模 Transformer + 强化学习 + 进化算法 + 物理模拟 |
| **已有成果** | AlphaFold 预测了 2 亿+ 蛋白质结构，发表 Nature 封面论文；GNoME 发现了 220 万种新材料；FunSearch 在 Cap Set 问题上发现新解 |
| **与 AI-Scientist 关系** | 垂直领域深度集成（生物、材料、数学），每个方向有专用模型；AI-Scientist 是通用框架 |
| **网址** | deepmind.google |

### 项目 3：OpenAI Deep Research — 付费 AI 科研助手

| 维度 | 详情 |
|------|------|
| **公司** | OpenAI（旧金山） |
| **产品** | ChatGPT Deep Research（2025 年 2 月发布，Pro 用户 $200/月） |
| **技术栈** | o3/o4-mini 推理模型 + 多步网页搜索 + Agentic browsing + 自动引用格式化 |
| **已有成果** | 输入一句话研究问题，5-30 分钟自动搜索 50+ 网页，生成带引用的结构化研究报告 |
| **与 AI-Scientist 关系** | 纯文献调研工具，不写代码不跑实验；AI-Scientist 独有 GPU 实验闭环 |
| **网址** | chatgpt.com |

### 项目 4：Elicit — 150 万用户的 AI 研究平台

| 维度 | 详情 |
|------|------|
| **公司** | Ought（旧金山，YC 校友） |
| **产品** | Elicit.com——AI 驱动的文献检索与系统性综述平台 |
| **技术栈** | LLM + 结构化数据提取 + 自动化 Meta-analysis |
| **已有成果** | 月活 150 万+ 研究者，可自动提取论文的样本量、效应量、P 值并汇总为 Meta 分析表格 |
| **与 AI-Scientist 关系** | 侧重"已有文献的自动化分析"，不做新实验；互补于 AI-Scientist 的实验生成能力 |
| **网址** | elicit.com |

### 项目 5：Consensus + SciSpace — 学术搜索引擎

| 维度 | Consensus | SciSpace |
|------|-----------|----------|
| **产品定位** | AI 学术搜索引擎，"学术版 ChatGPT" | AI 论文阅读与写作助手 |
| **核心技术** | LLM + 1 亿+ 论文向量库 + 共识度评分 | PDF 解析 + 论文对比 + 引用推荐 |
| **亮点** | 输入问题 → 输出"学术界共识"（17 篇论文中 83% 支持某种结论） | 论文旁边开 AI 对话面板，逐段解释 |
| **网址** | consensus.app | typeset.io |

### 对比总结

| 项目 | 自动实验 | 自动写论文 | 自动审稿 | 通用性 | 开源性 | 阶段 |
|------|:---:|:---:|:---:|:---:|:---:|------|
| **AI-Scientist (我们)** | ✅ | ✅ | ✅ | ✅ 多领域 | ✅ | 研究原型 |
| FutureHouse PaperQA2 | ❌ | ⚠️ 综述 | ❌ | ✅ | ✅ | 已商用 |
| DeepMind AlphaFold | ✅ | ❌ | ❌ | ❌ 单领域 | ⚠️ | 已商用 |
| OpenAI Deep Research | ❌ | ⚠️ 报告 | ❌ | ✅ | ❌ | 已商用 |
| Elicit | ❌ | ❌ | ❌ | ✅ | ❌ | 已商用 |

> **核心洞见**：市面上的产品集中在"文献调研自动化"，做实验、写论文、审稿的全闭环只有 AI-Scientist 做到了。

---

## 六、我们的工作定位

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  本工作的三个贡献层次                                 │
│                                                     │
│  ① 模型层：DeepSeek V4 适配                          │
│     将前沿国产模型集成到 AI-Scientist 框架             │
│     - V4 Pro thinking 模式增强推理                    │
│     - V4 Flash 降本增效                               │
│     - 成本从 $15/篇 降至 $0.15/篇                     │
│                                                     │
│  ② 系统层：跨平台适配                                 │
│     Windows + Linux 混合架构                          │
│     - 无 GPU 环境运行 LLM 流水线                      │
│     - GPU 服务器跑实验                                 │
│     - 论文在本地编译 PDF                               │
│                                                     │
│  ③ 应用层：多领域验证                                 │
│     - nanoGPT：字符级语言模型优化                      │
│     - 2D Diffusion：扩散模型可控生成                   │
│     - 对比 DDPM / ByT5 等经典工作                      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 六、关键参考文献速查

| 编号 | 引用 | 领域 |
|:----:|------|------|
| [1] | Lu et al. (2024). *The AI Scientist*. arXiv:2408.06292 | 自动科研 |
| [2] | Yamada et al. (2025). *AI Scientist v2*. arXiv:2504.08066 | 自动科研 |
| [3] | Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS | Transformer |
| [4] | Ho et al. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS | 扩散模型 |
| [5] | Xue et al. (2022). *ByT5: Towards a Token-Free Future*. TACL | 字符级LM |
| [6] | Power et al. (2022). *Grokking*. arXiv:2201.02177 | 泛化现象 |
| [7] | Karpathy (2023). *nanoGPT*. GitHub | GPT实现 |
| [8] | Christiano et al. (2017). *Deep RL from Human Preferences*. NeurIPS | RLHF |
| [9] | Rafailov et al. (2023). *Direct Preference Optimization*. NeurIPS | DPO |
| [10] | Kingma & Ba (2015). *Adam*. ICLR | 优化器 |
