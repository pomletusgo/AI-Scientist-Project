#!/usr/bin/env python3
"""Standalone debate runner — called via subprocess for terminal isolation."""
import json, os, sys, traceback

QWEN_API_KEY = "sk-ws-H.EEMHRMM.rn6X.MEYCIQCEBkq2oB96-KBcXFb0U-Q747ol5XEcYI87iztmCNwXBQIhAN01KYz3gnjs8HSrJwhH2k5zatd4Gxzi8vK0A6QEKH1j"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-plus"

def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"[DebateStep] Output dir: {output_dir}")

    # Suppress prompt_toolkit
    os.environ["TERM"] = "dumb"
    os.environ["PROMPT_TOOLKIT_NO_CPR"] = "1"

    sys.path.insert(0, r"d:\AI-Scientist")

    # 1) Create Qwen client for LLM calls
    from openai import OpenAI
    qwen_client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

    # 2) Run perform_review (pure LLM, no Aider)
    print("[DebateStep] Running 3-reviewer ensemble review...")
    tex_path = os.path.join(output_dir, "latex", "template.tex")
    with open(tex_path, "r", encoding="utf-8") as f:
        paper_text = f.read()

    from ai_scientist.perform_review import perform_review
    try:
        review = perform_review(
            paper_text,
            model=QWEN_MODEL,
            client=qwen_client,
            num_reflections=2,
            num_fs_examples=1,
            num_reviews_ensemble=3,
            temperature=0.1,
        )
        review_path = os.path.join(output_dir, "review_qwen.json")
        with open(review_path, "w", encoding="utf-8") as f:
            json.dump(review, f, indent=2, ensure_ascii=False, default=str)
        print(f"[DebateStep] Review saved: {review_path}")
    except Exception as e:
        print(f"[DebateStep] Review failed: {e}")
        traceback.print_exc()

    # 3) Run Aider debate (if coder can initialize)
    print("[DebateStep] Running multi-agent debate with Aider...")
    try:
        from aider.coders import Coder
        from aider.io import InputOutput
        from aider.models import Model
        from ai_scientist.perform_debate import debate_paper

        notes_file = os.path.join(output_dir, "notes.txt")
        io = InputOutput(
            yes=True,
            chat_history_file=os.path.join(output_dir, "debate_chat.json"),
            pretty=False,
        )

        os.environ["OPENAI_API_BASE"] = QWEN_BASE_URL
        os.environ["OPENAI_API_KEY"] = QWEN_API_KEY

        coder_model = Model(f"openai/{QWEN_MODEL}")
        coder = Coder.create(
            main_model=coder_model,
            fnames=[tex_path, notes_file],
            io=io,
            stream=False,
            use_git=False,
            edit_format="diff",
        )

        debate_paper(
            output_dir,
            client=qwen_client,
            model=QWEN_MODEL,
            coder=coder,
            max_rounds=3,
            consensus_rounds=2,
        )
        print("[DebateStep] Debate completed.")
    except Exception as e:
        print(f"[DebateStep] Aider debate failed (non-fatal): {e}")
        traceback.print_exc()

    print("[DebateStep] Done.")


if __name__ == "__main__":
    main()
