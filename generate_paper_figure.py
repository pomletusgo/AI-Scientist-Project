#!/usr/bin/env python3
"""
NanoBanana (Gemini Image) 论文配图生成器
===========================================
使用 Google Gemini 图片生成模型为 AI-Scientist 论文生成配图。

模型: gemini-3.1-flash-image-preview (NanoBanana 2)
      gemini-2.5-flash-image           (NanoBanana 1)

用法:
    # 生成论文核心方法图（推荐）
    python generate_paper_figure.py

    # 指定输出路径
    python generate_paper_figure.py --output results/method_figure.png

    # 选择不同的图片类型
    python generate_paper_figure.py --type pipeline   # 流水线架构图
    python generate_paper_figure.py --type method     # 方法核心图（默认）
    python generate_paper_figure.py --type overview   # 全局概览图

环境变量:
    GOOGLE_API_KEY: Google Vertex AI Express 密钥 (AQ.xxx 格式)
    或 GEMINI_API_KEY: Gemini API 密钥 (AIza 格式)
"""

import argparse
import base64
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Fix Windows encoding for Unicode emoji (✅ ❌ ⚠️)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ============================================================
# 图片 Prompt 设计 —— 基于论文核心内容
# ============================================================

PROMPTS = {
    # -------- 方法核心图: Problem → Systematic Method → Solution --------
    "method": {
        "title": "Systematic LLM Safety Alignment via Multi-Agent Evaluation and Representation Control",
        "filename": "method_framework.png",
        "prompt": """Create a scientific method diagram for a machine learning research paper about
LLM safety alignment. The diagram should illustrate the core methodology clearly.

Layout (left to right flow, three main sections):

LEFT SECTION - "Safety Vulnerabilities in LLMs":
- A large language model (neural network icon) being attacked by adversarial inputs
- Red arrows labeled: "Jailbreak Prompts", "Reward Hacking", "Adversarial Injection"
- Below: a small box stating the core challenge: "Safety mechanisms degrade model capability"
- Visual contrast: the LLM shown as partially exposed / vulnerable

CENTER SECTION - "Systematic Safety Enhancement Framework":
A structured multi-component methodology showing:
  1. "Multi-Agent Red Teaming" (top left of center):
     - Three specialized evaluator agents: "Coverage Explorer", "Targeted Attacker", "Diversity Generator"
     - They probe the model from different angles, finding complementary vulnerabilities
  2. "Representation-Based Alignment" (top right of center):
     - Internal model representations being adjusted
     - Contrast between "unsafe activation pattern" (red) and "aligned activation pattern" (green)
  3. "Adversarial Robustness Training" (bottom of center):
     - A training loop: "Generate adversarial variants → Train on failures → Evaluate → Repeat"
     - Loss curves showing improved robustness over iterations
  All three components connected with arrows showing they work together

RIGHT SECTION - "Safety-Aligned LLM":
- A shielded neural network with security indicators
- Labels: "Robust to Jailbreak Attacks", "Maintained Task Performance", "Verified Safety Guarantees"
- Green checkmarks and a shield icon
- Contrast with the vulnerable model on the left

BOTTOM BANNER - "Automated Research Pipeline":
- Five connected modules showing the full discovery process:
  "Systematic Ideation" → "Automated Experiments" → "Paper Generation" → "Multi-Agent Review" → "Verification"
- Each module with a simple icon

Style: Clean scientific diagram, white background, professional academic style.
Color scheme: red/orange for vulnerabilities, blue for the methodology,
green for the safe model output. Sans-serif font style, no text overlapping.
Suitable for a NeurIPS/ICML paper. 16:9 aspect ratio, vector-like appearance.""",
    },

    # -------- 流水线架构图: AI-Scientist 完整流程 --------
    "pipeline": {
        "title": "AI-Scientist: Automated Research Pipeline for LLM Safety",
        "filename": "pipeline_architecture.png",
        "prompt": """Create a pipeline architecture diagram for a research paper about
an automated scientific discovery system for LLM safety alignment research.

The diagram shows a multi-stage pipeline flowing from top to bottom:

STAGE 1 - "Systematic Research Ideation":
- An LLM analyzing the problem space: safety vulnerabilities, performance trade-offs
- Structured problem decomposition into sub-problems
- Output: Novel research ideas with clearly identified approach and evaluation plan
- Visual: branching tree showing idea exploration

STAGE 2 - "Automated Experimentation":
- Code generation and execution (Python script icon)
- Training runs with loss curves showing convergence
- Evaluation metrics: safety scores, task performance, robustness measures
- Visual: experiment progress with charts

STAGE 3 - "Scientific Paper Generation":
- LaTeX document icon with sections being populated
- Structured writing: Abstract, Introduction, Method, Experiments, Results, Conclusion
- Automatic figure and table insertion from experiment outputs
- Visual: document being assembled

STAGE 4 - "Multi-Agent Peer Review":
- Three specialized reviewer agents:
  - "Methodology Reviewer" (blue) — checks technical soundness
  - "Experiments Reviewer" (green) — validates empirical results
  - "Literature Reviewer" (orange) — ensures contextual positioning
- Central "Response Agent" that evaluates critiques and applies improvements
- Iterative refinement loop with convergence indicator

STAGE 5 - "Quality Verification":
- Reference authenticity validation (database lookup icon)
- Parameter plausibility check (validated range indicator)
- Overall trust score computation
- Visual: checkmarks and confidence scores

STAGE 6 - "Output: Publication-Ready Research":
- Final compiled paper (PDF icon with checkmark)
- Novel contributions to LLM safety alignment research
- Reproducible experiment code and data

Style: Professional pipeline diagram, light gray background, distinct color per stage,
clean arrows connecting stages, icon + concise text labels.
Flat vector illustration style, suitable for a systems/ML paper. 4:3 aspect ratio.""",
    },

    # -------- 全局概览图: 完整研究叙事 --------
    "overview": {
        "title": "Systematic LLM Safety Alignment: From Vulnerability Analysis to Robust Defense",
        "filename": "research_overview.png",
        "prompt": """Create a comprehensive research overview illustration for a paper about
LLM safety alignment through systematic vulnerability analysis and defense.

The image integrates the complete research narrative:

TOP SECTION - "The Safety Challenge":
- Center: An LLM exposed to diverse attack vectors
- Left: Jailbreak scenarios — adversarial prompts bypassing safety training
- Right: Reward exploitation — the model gaming the reward signal during RLHF
- Central question: "How to achieve robust safety without sacrificing model capability?"

MIDDLE SECTION - "Systematic Methodology":
A bridge-like structure connecting challenge to solution, built from four pillars:
  1. "Comprehensive Vulnerability Discovery" — multi-perspective red teaming
  2. "Internal Representation Control" — manipulating model activations for safety
  3. "Adversarial Robustness Optimization" — training against worst-case inputs
  4. "Automated Quality Assurance" — verifying references, parameters, and claims
Each pillar with a simple icon and brief label. Arrows showing how they reinforce each other.

BOTTOM SECTION - "Results: Safe and Capable LLMs":
- A robust LLM with a protective shield
- Three key outcomes:
  1. "Enhanced Jailbreak Resistance" — significantly reduced attack success rate
  2. "Preserved Task Performance" — minimal degradation on standard benchmarks
  3. "Generalizable Safety" — defense transfers across attack types
- Quantitative indicators: improved safety scores with confidence intervals

SIDE PANEL - "The Research Loop":
- A circular feedback process: "Analyze Vulnerabilities → Design Defense → Evaluate → Refine"
- Each iteration producing stronger safety guarantees
- The loop driving continuous improvement

Style: Infographic/overview style, professional and clean, white background.
Color scheme: red/orange (challenge/vulnerability), blue (methodology),
green (solution/results), purple (research loop).
Clear visual hierarchy, suitable as a paper overview figure. 16:9 aspect ratio.""",
    },
}


def generate_image(prompt_text, output_path, api_key=None, model="gemini-3.1-flash-image-preview"):
    """
    使用 Google Gemini (NanoBanana) 生成图片。

    Args:
        prompt_text: 图片描述 prompt
        output_path: 输出图片路径
        api_key: Google API key (AQ.xxx 或 AIza.xxx)
        model: 图片生成模型名称

    Returns:
        str: 生成的图片文件路径
    """
    from google import genai

    if api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "请设置 GOOGLE_API_KEY 或 GEMINI_API_KEY 环境变量，"
            "或通过 --api-key 参数传入。\n"
            "获取免费 API Key: https://aistudio.google.com/apikey"
        )

    print(f"使用模型: {model}")
    print(f"API Key: {api_key[:12]}...{api_key[-4:]}")
    print(f"Prompt 长度: {len(prompt_text)} 字符")
    print()

    client = genai.Client(api_key=api_key)

    print("正在生成图片 (NanoBanana)...")
    print("这可能需要 10-30 秒，请稍候...")

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt_text,
            config={
                "response_modalities": ["IMAGE", "TEXT"],
            },
        )
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            print("\n❌ 配额已用完 (429 RESOURCE_EXHAUSTED)")
            print("   Google AI Studio 免费配额每日有限制。")
            print("   解决方案:")
            print("   1. 等待配额重置 (通常每天重置)")
            print("   2. 使用不同的 Google 账号获取新 Key")
            print("   3. 升级到付费计划")
            print(f"   获取 Key: https://aistudio.google.com/apikey")
        elif "location" in error_str.lower():
            print("\n❌ 地域限制")
            print("   Google API 在你所在的地区不可用。")
            print("   请使用 VPN 或代理服务。")
        else:
            print(f"\n❌ 生成失败: {e}")
        raise

    # 提取图片数据
    image_saved = False
    for i, part in enumerate(response.candidates[0].content.parts):
        if hasattr(part, "inline_data") and part.inline_data:
            image_data = part.inline_data.data
            mime_type = part.inline_data.mime_type
            print(f"获取到图片: {len(image_data)} bytes, 类型: {mime_type}")

            # 保存图片
            ext = mime_type.split("/")[-1] if "/" in mime_type else "png"
            if not output_path.endswith(f".{ext}"):
                output_path = output_path.rsplit(".", 1)[0] + f".{ext}"

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(image_data)
            print(f"✅ 图片已保存: {output_path}")
            image_saved = True

        if hasattr(part, "text") and part.text:
            print(f"模型返回文本: {part.text[:200]}")

    if not image_saved:
        print("⚠️ 模型未返回图片，检查 response:")
        print(f"   Candidates: {len(response.candidates)}")
        for i, cand in enumerate(response.candidates):
            print(f"   Candidate {i}: {len(cand.content.parts)} parts")
            for j, part in enumerate(cand.content.parts):
                print(f"     Part {j}: text={bool(part.text)}, inline_data={bool(part.inline_data)}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="NanoBanana 论文配图生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python generate_paper_figure.py                          # 默认生成方法核心图
  python generate_paper_figure.py --type pipeline           # 生成流水线架构图
  python generate_paper_figure.py --type overview           # 生成全局概览图
  python generate_paper_figure.py --type all                # 生成所有图片
  python generate_paper_figure.py --output my_figure.png    # 自定义输出路径
  python generate_paper_figure.py --api-key AQ.xxx          # 直接传入 API Key
        """,
    )
    parser.add_argument(
        "--type", type=str, default="method",
        choices=["method", "pipeline", "overview", "all"],
        help="图片类型: method(方法核心图), pipeline(流水线图), overview(全局概览), all(全部)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出图片路径 (默认: results/<type>_<timestamp>.png)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Google API Key (AQ.xxx 或 AIza.xxx 格式)",
    )
    parser.add_argument(
        "--model", type=str, default="gemini-3.1-flash-image-preview",
        choices=[
            "gemini-3.1-flash-image-preview",   # NanoBanana 2 (推荐)
            "gemini-2.5-flash-image",           # NanoBanana 1
            "gemini-3.1-flash-image",           # NanoBanana 2 正式版
        ],
        help="使用的图片生成模型",
    )
    parser.add_argument(
        "--show-prompts", action="store_true",
        help="仅显示所有 prompt 内容，不生成图片",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅打印配置，不实际调用 API",
    )
    args = parser.parse_args()

    # 仅显示 prompts
    if args.show_prompts:
        for ptype, pinfo in PROMPTS.items():
            print(f"\n{'='*70}")
            print(f"类型: {ptype} — {pinfo['title']}")
            print(f"文件名: {pinfo['filename']}")
            print(f"{'='*70}")
            print(pinfo["prompt"])
        return

    # 确定要生成的类型
    if args.type == "all":
        types_to_generate = list(PROMPTS.keys())
    else:
        types_to_generate = [args.type]

    # API Key
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN — 仅显示配置")
        print("=" * 60)
        print(f"模型: {args.model}")
        print(f"API Key: {'已设置' if api_key else '❌ 未设置'}")
        print(f"类型: {args.type}")
        for ptype in types_to_generate:
            info = PROMPTS[ptype]
            print(f"\n  [{ptype}] {info['title']}")
            print(f"  → {info['filename']}")
            print(f"  Prompt: {len(info['prompt'])} 字符")
        return

    # 检查 API Key
    if not api_key:
        print("❌ 错误: 未设置 API Key")
        print()
        print("请通过以下方式之一设置:")
        print("  1. 命令行参数: --api-key AQ.xxx")
        print("  2. 环境变量: export GOOGLE_API_KEY=AQ.xxx")
        print("  3. 环境变量: export GEMINI_API_KEY=AQ.xxx")
        print()
        print("获取免费 API Key: https://aistudio.google.com/apikey")
        sys.exit(1)

    # 生成图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    for ptype in types_to_generate:
        info = PROMPTS[ptype]
        print("=" * 60)
        print(f"生成: {info['title']}")
        print(f"类型: {ptype}")
        print("=" * 60)

        # 确定输出路径
        if args.output and len(types_to_generate) == 1:
            output_path = args.output
        else:
            base_name = info["filename"].rsplit(".", 1)[0]
            ext = info["filename"].rsplit(".", 1)[1]
            output_path = f"results/{base_name}_{timestamp}.{ext}"

        try:
            result_path = generate_image(
                info["prompt"],
                output_path,
                api_key=api_key,
                model=args.model,
            )
            results.append({"type": ptype, "path": result_path, "status": "success"})
        except Exception:
            results.append({"type": ptype, "path": output_path, "status": "failed"})
            if args.type != "all":
                sys.exit(1)

    # 总结
    print()
    print("=" * 60)
    print("生成总结")
    print("=" * 60)
    for r in results:
        symbol = "✅" if r["status"] == "success" else "❌"
        print(f"  {symbol} [{r['type']}] {r['path']}")

    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"\n成功: {success_count}/{len(results)}")
    return results


if __name__ == "__main__":
    main()
