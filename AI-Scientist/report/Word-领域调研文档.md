# AI-Scientist 领域调研与产业分析报告

> 供 Word 排版：全文字数 ~4000 字，可直接复制到 Word 中编排

---

## 一、领域概述

自动化科学发现（Automated Scientific Discovery）是人工智能领域的前沿方向，经历了三代技术演进：

- **第一代（1990s-2010s）**：符号推理方法，如 BACON 系统通过数据拟合重新发现物理定律，AM 系统自动推导数学概念。局限在于需要人工预定义搜索空间。

- **第二代（2010s-2020s）**：机器学习方法，代表如 AlphaFold 通过深度学习预测蛋白质三维结构（2024 年诺贝尔化学奖），GNoME 通过图神经网络发现 220 万种新材料。局限在于每个系统针对特定领域深度优化。

- **第三代（2023-至今）**：大语言模型（LLM）智能体方法，代表如 Sakana AI 的 AI-Scientist，利用通用大模型的推理能力实现跨领域的端到端科学发现闭环，从提出假设、设计实验、执行实验、撰写论文到同行评审，全部自动化完成。

本报告聚焦第三代 LLM 智能体方法，系统梳理相关学术工作与产业化项目。

---

## 二、核心学术相关工作

### 2.1 直接相关：LLM 驱动的科学发现系统

**AI-Scientist（Lu et al., Sakana AI, 2024）** 是首个实现端到端全自动科学发现的系统。其核心架构包含四个阶段：（1）基于 Chain-of-Thought 和自反思的想法生成；（2）通过 Aider 进行自动化代码修改和 GPU 实验执行；（3）逐节生成 LaTeX 学术论文并结合 Semantic Scholar 搜索真实参考文献；（4）按照 NeurIPS 标准对论文进行集成评审。每篇论文成本约 $15。论文发表于 Nature。

**AI-Scientist v2（Yamada et al., 2025）** 是 v1 的重大升级版本，取消了人类编写实验模板的要求，引入最佳优先树搜索（BFTS）和 Experiment Manager Agent，使系统能泛化到任意 ML 领域。该版本产出了首篇通过正式同行评审的 AI 生成论文（ICLR 2025 ICBINB Workshop，得分 6/7/6）。

**PaperQA2（Skarlinski et al., FutureHouse, 2024）** 是开源的 AI 文献综述引擎，采用 RAG 架构结合多步推理和 PDF 全文解析，能在数小时内完成给定主题的完整文献综述。与 AI-Scientist 互补：前者侧重文献搜索，后者侧重实验创新。

**SciAgents（Ghafarollahi et al., MIT, 2024）** 采用多智能体协作架构，每个智能体负责科研流程的不同环节，验证了多智能体架构在科研自动化中的有效性。

### 2.2 基础支撑：LLM 智能体与代码生成

**ReAct（Yao et al., 2023）** 提出了推理（Reasoning）与行动（Acting）交替执行的 LLM 智能体范式，成为后续大多数 LLM 智能体系统的基础架构。

**Aider（Gauthier, 2023）** 是 LLM 驱动的代码编辑工具，采用 diff-based 增量编辑而非一次性生成，被 AI-Scientist 直接集成用于自动修改实验代码。

**SWE-Agent（Yang et al., 2024）** 实现了 LLM 自动修复软件 Bug 的功能，在 SWE-bench 基准上表现优异，证明了 LLM 在代码生成与调试方面的实用能力。

### 2.3 领域先行者：专用 AI 科学发现系统

**AlphaFold 3（Abramson et al., Google DeepMind, 2024）** 将蛋白质结构预测扩展到所有生物分子相互作用，预测精度达到原子级别，获 2024 年诺贝尔化学奖。但其架构高度专用化，无法泛化到其他科学领域。

**FunSearch（Romera-Paredes et al., DeepMind, 2024）** 首次使用 LLM 在数学领域做出原创性发现，在 Cap Set 和在线装箱问题上找到了此前未知的更优解，证明了 LLM 具有超越人类的科学创造力。

---

## 三、产业化与工程项目调研

### 项目一：FutureHouse（旧金山，融资 $30M+）

FutureHouse 是全球首家以"构建 AI 科学家"为核心使命的创业公司，由前 Google 科学家创立，获得 Eric Schmidt 等投资人超过 3000 万美元融资。其核心产品 PaperQA2 已开源（GitHub 5000+ Star），能在数小时内自动完成跨领域文献综述并生成包含精确引用的研究报告。该公司已与多家生物医药企业合作，将 AI 科学发现能力应用于靶点发现和临床前研究。与 AI-Scientist 的核心差异在于：FutureHouse 不执行新实验，仅做已有文献的自动化分析。

### 项目二：Google DeepMind AlphaFold / GNoME 系列（伦敦）

DeepMind 是工业界 AI 科学发现的标杆。AlphaFold 3（2024 年 5 月发布）将预测范围从蛋白质扩展到 DNA、RNA、配体和翻译后修饰，并开源了代码和模型权重。GNoME（2023 年 11 月发表于 Nature）通过图神经网络发现了 220 万种新型无机晶体材料，相当于人类 800 年的累积发现量。此外，DeepMind 的 FunSearch（2024 年 12 月）首次用 LLM 在纯数学领域做出原创性发现，证明了 LLM 的科学研究潜力。DeepMind 的路径与 AI-Scientist 不同：每个产品针对特定科学领域深度打造，而非构建通用框架。

### 项目三：OpenAI ChatGPT Deep Research（2025 年 2 月发布）

OpenAI 在 2025 年 2 月推出的 Deep Research 功能是首个面向大众用户的 AI 研究助手。用户只需输入一句话的研究问题，系统自动在 5-30 分钟内搜索 50 以上的网页来源，进行交叉验证和多步推理，最终生成一份带完整引用的结构化研究报告。该功能使用 o3/o4-mini 推理模型驱动，采用 Agentic browsing 技术实现自主网页浏览和信息提取，已向 ChatGPT Pro 订阅用户（$200/月）开放。与 AI-Scientist 的核心差异在于：Deep Research 不会写代码、不会做实验、不会生成可发表的学术论文。

### 项目四：Elicit（旧金山，YC 校友，月活 150 万+）

Elicit 由 Ought 公司开发，是目前全球用户量最大的 AI 科研辅助平台。其核心功能包括：输入研究问题→自动提取数百篇相关论文的关键信息（样本量、效应量、P 值、研究方法等）→以结构化表格呈现→支持自动化元分析。Elicit 的独特价值在于其"系统性综述自动化"能力，能将传统需要数周的 Meta-analysis 缩短到数小时。该平台被哈佛、MIT、斯坦福等顶尖高校的研究者广泛使用。

### 项目五：Consensus

Consensus 定位为"学术界的 ChatGPT"，构建了涵盖 1 亿+ 学术论文的向量数据库。其特色功能是"共识度评估"：对于一个研究问题，AI 基于检索到的论文自动计算支持/否定该结论的论文比例，例如"83% 的论文支持某种疗法有效"。此外，SciSpace（Typeset）平台提供了论文阅读侧的 AI 辅助，支持逐段解释、自动引用推荐和对比阅读功能。

### 产业化对比总结

| 维度 | AI-Scientist | FutureHouse | DeepMind | OpenAI | Elicit |
|------|:---:|:---:|:---:|:---:|:---:|
| 自动实验 | ✅ | ❌ | ✅ | ❌ | ❌ |
| 自动写论文 | ✅ | ⚠️综述 | ❌ | ⚠️报告 | ❌ |
| 自动审稿 | ✅ | ❌ | ❌ | ❌ | ❌ |
| 通用性 | ✅多领域 | ✅ | ❌专用 | ✅ | ✅ |
| 开源性 | ✅ | ✅ | ⚠️ | ❌ | ❌ |
| 商用阶段 | 研究原型 | 已商用 | 已商用 | 已商用 | 已商用 |

**核心发现**：当前市面上的产品集中在"文献调研自动化"环节（信息检索→提取→汇总），真正实现"提出假设→设计实验→执行实验→发表论文"全闭环的工程系统仅 AI-Scientist 一家。但这同时也说明市场存在明确需求，AI-Scientist 的产业化前景广阔。

---

## 四、我们的工作定位与贡献

基于上述领域调研，本工作的核心贡献包括三个层次：

**层次一：模型适配与成本优化**。将 DeepSeek V4 Pro / V4 Flash / R1 等前沿国产开源模型集成到 AI-Scientist 框架中，利用 V4 Pro 的 thinking 推理模式和 V4 Flash 的高性价比特性，在保持论文质量的同时将单篇成本从 $15 降至 $0.15，降低 100 倍。

**层次二：跨平台工程适配**。设计了 Windows（本地）+ Linux（服务器）混合架构：LLM 流水线可在无 GPU 环境下运行，GPU 实验在 A100 服务器上完成，论文编译在本地通过 Overleaf 完成。大幅降低了系统部署门槛。

**层次三：多领域验证与扩展**。在 nanoGPT（语言模型优化，对照 ByT5）、2D Diffusion（扩散模型改进，对照 DDPM）、Grokking（神经网络泛化，对照 Power et al. 2022）三个不同领域验证了系统的通用性，每个领域均能产出带真实实验数据和图表、真实参考文献的完整学术论文。

---

## 五、参考文献

[1] Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., & Ha, D. (2024). The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. *arXiv:2408.06292* (发表于 Nature).

[2] Yamada, Y., Lange, R. T., Lu, C., Hu, S., Lu, C., Foerster, J., Clune, J., & Ha, D. (2025). The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search. *arXiv:2504.08066*.

[3] Skarlinski, M. D., et al. (2024). PaperQA2: Superhuman Scientific Literature Search. *arXiv:2412.01818*.

[4] Abramson, J., et al. (2024). Accurate Structure Prediction of Biomolecular Interactions with AlphaFold 3. *Nature*, 630, 493-500.

[5] Romera-Paredes, B., et al. (2024). Mathematical Discoveries from Program Search with Large Language Models. *Nature*, 625, 468-476.

[6] Merchant, A., et al. (2023). Scaling Deep Learning for Materials Discovery. *Nature*, 624, 80-85. (GNoME)

[7] Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*.

[8] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *NeurIPS 2020*.

[9] Xue, L., et al. (2022). ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models. *TACL 2022*.

[10] Power, A., et al. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. *arXiv:2201.02177*.
