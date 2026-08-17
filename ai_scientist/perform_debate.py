#!/usr/bin/env python3
"""
Multi-Agent Consensus Review System.

Three specialized reviewer agents independently critique a paper,
then enter a CONSENSUS phase where they discuss disagreements and try to
reach agreement. If consensus cannot be reached after a configurable number
of rounds, the system falls back to MAJORITY VOTING.

This replaces the old single-agent "Paper Agent" accept/reject model with a
true peer-review consensus mechanism.

Design:
  Phase 1 — INDEPENDENT REVIEW: 3 reviewers produce independent reviews
  Phase 2 — CONSENSUS BUILDING: Reviewers see each other's critiques and
           discuss to reach agreement on each disputed point
  Phase 3 — VOTING FALLBACK: If consensus not reached after N rounds,
           majority vote decides; ties broken by a neutral Chair agent
  Phase 4 — APPLY & ITERATE: Agreed changes applied, loop to Phase 1
"""

import os, os.path as osp, json, re, threading, time
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from ai_scientist.perform_review import perform_review
from ai_scientist.llm import get_response_from_llm, extract_json_between_markers

# Hard limit for Aider coder.run() to prevent infinite refinement loops
AIDER_TIMEOUT_SECONDS = 600  # 10 minutes per coder.run() call

# ============================================================
# Three Reviewer Personas — same review format, different focus
# ============================================================

REVIEWER_PROMPTS = {
    "Methodology": (
        "You are an AI researcher specializing in METHODOLOGY review. "
        "Focus EXCLUSIVELY on: is the proposed method sound? Are assumptions "
        "justified? Are there logical gaps in the reasoning? Is the problem "
        "formulation correct? Is the approach novel or derivative? "
        "Be critical and specific — point to exact sections or formulas that "
        "need clarification or have flaws."
    ),
    "Experiments": (
        "You are an AI researcher specializing in EXPERIMENTAL review. "
        "Focus EXCLUSIVELY on: are the experiments sufficient to support the claims? "
        "Are baselines fair and comprehensive? Are hyperparameters justified? "
        "Is the evaluation metric appropriate? Are results statistically meaningful? "
        "Could the results be explained by factors other than the proposed method? "
        "Check for missing ablations, unfair comparisons, or cherry-picked results."
    ),
    "Literature": (
        "You are an AI researcher specializing in LITERATURE and CLARITY review. "
        "Focus EXCLUSIVELY on: is related work properly covered and fairly compared? "
        "Are citations accurate and complete? Is the argument clear and easy to follow? "
        "Are there missing references that should be cited? Is the writing quality "
        "appropriate for publication? Check for overclaims, vague statements, or "
        "unsupported assertions."
    ),
}

# NeurIPS-style review form — same as perform_review.py but without few-shot examples
REVIEW_FORM = """
Respond in the following format:

THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

In <THOUGHT>, briefly discuss your reasoning as a reviewer focused on your specialty.

In <JSON>, provide the review with these fields:
- "Summary": One-paragraph summary of the paper.
- "Strengths": A list of specific strengths (at least 2).
- "Weaknesses": A list of specific, actionable weaknesses (at least 3). Each must describe exactly what needs to change.
- "Originality": 1-4 (low to very high).
- "Quality": 1-4 (low to very high).
- "Clarity": 1-4 (low to very high).
- "Significance": 1-4 (low to very high).
- "Overall": 1-10 (very strong reject to award quality).
- "Decision": Accept or Reject.

This JSON will be automatically parsed, so ensure the format is precise.
"""

# ============================================================
# Consensus Building Prompt
# ============================================================

CONSENSUS_DISCUSSION_PROMPT = """You are a panel of THREE peer reviewers discussing a research paper.
All three reviewers have read the paper and produced independent reviews.

Below are the reviews from each reviewer. Your task is to DISCUSS and try to reach
CONSENSUS on each identified weakness.

================================================================================
PAPER (excerpts):
{paper_excerpt}
================================================================================

REVIEWS:
{reviews_text}
================================================================================

CONSENSUS DISCUSSION — For EACH weakness found by any reviewer, please:

1. STATE YOUR POSITION: Each reviewer briefly states whether they agree the
   weakness is valid, and why or why not.

2. DISCUSS: Compare perspectives. If reviewers disagree, explore why — is it a
   matter of expertise, interpretation, or priority?

3. PROPOSE RESOLUTION: For each weakness, propose:
   - Is it a real issue? (yes / no / needs clarification)
   - How should it be fixed? (specific actionable suggestion)
   - Priority: high / medium / low

4. REACH CONSENSUS:
   - CONSENSUS means at least 2 out of 3 reviewers agree on the validity and fix.
   - If all 3 agree → STRONG CONSENSUS
   - If 2 agree, 1 disagrees → WEAK CONSENSUS (majority carries)
   - If all 3 disagree (1-1-1 split) → NO CONSENSUS → mark as DISPUTED for voting

Output your discussion and conclusions in the following JSON format:

```json
{{
  "discussion_summary": "Brief narrative of the key discussion points and areas of agreement/disagreement",
  "consensus_points": [
    {{
      "weakness_id": 1,
      "weakness_text": "Brief description of the weakness",
      "source_reviewer": "Methodology|Experiments|Literature",
      "validity_consensus": "valid|invalid|disputed",
      "agreement_level": "strong_consensus|weak_consensus|no_consensus",
      "agreed_fix": "Specific actionable suggestion, or null if invalid/disputed",
      "priority": "high|medium|low",
      "reviewer_positions": {{
        "Methodology": "agree/disagree/neutral — brief reason",
        "Experiments": "agree/disagree/neutral — brief reason",
        "Literature": "agree/disagree/neutral — brief reason"
      }}
    }}
  ],
  "disputed_points": [
    {{
      "weakness_id": 1,
      "weakness_text": "Brief description",
      "positions": {{
        "Methodology": "position and reasoning",
        "Experiments": "position and reasoning",
        "Literature": "position and reasoning"
      }},
      "reason_for_dispute": "Why consensus could not be reached"
    }}
  ],
  "overall_assessment": {{
    "paper_ready": true/false,
    "remaining_major_issues": 0,
    "recommendation": "accept_as_is|minor_revision|major_revision|reject"
  }}
}}
```

IMPORTANT: Focus on reaching genuine consensus where possible. Only mark as DISPUTED
when discussion genuinely fails to resolve differences. Be specific and actionable."""


# ============================================================
# Voting Fallback Prompt (used when consensus fails)
# ============================================================

VOTING_PROMPT = """CONSENSUS COULD NOT BE REACHED on the following points after {consensus_rounds} rounds of discussion.

A neutral CHAIR agent will now resolve these disputes by VOTE.

For each DISPUTED point below, evaluate the arguments from each reviewer and CAST YOUR VOTE.
You are NOT any of the reviewers — you are a neutral, senior area chair with broad expertise.

DISPUTED POINTS:
{disputed_points_json}

PAPER EXCERPT (for context):
{paper_excerpt}

VOTING RULES:
- Vote "accept" if you believe the weakness is VALID and should be addressed
- Vote "reject" if you believe the weakness is NOT valid or NOT worth addressing
- Majority wins (2+ out of 3 reviewers already voted in discussion; your vote is the TIE-BREAKER for 1-1-1 splits)

Output your votes:

```json
{{
  "chair_votes": [
    {{
      "weakness_id": 1,
      "vote": "accept|reject",
      "reasoning": "Brief explanation of your decision (one sentence)"
    }}
  ],
  "chair_summary": "Brief overall assessment"
}}
```"""


# ============================================================
# Coder.run() wrapper with hard timeout to prevent infinite loops
# ============================================================

def _run_coder_with_limit(coder, message, timeout=AIDER_TIMEOUT_SECONDS):
    """Run coder.run() in a thread with a hard timeout."""
    result_container = {"done": False, "error": None}

    def _target():
        try:
            coder.run(message)
            result_container["done"] = True
        except Exception as e:
            result_container["error"] = str(e)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        print(f"  [AIDER TIMEOUT] coder.run() exceeded {timeout}s limit.")
        return True
    if result_container["error"]:
        print(f"  [AIDER ERROR]: {result_container['error']}")
        return False
    return result_container["done"]


# ============================================================
# Paper text extraction
# ============================================================

def _load_paper_text(folder_name):
    """Load paper text from the latex template."""
    tex_path = osp.join(folder_name, "latex", "template.tex")
    if not osp.exists(tex_path):
        return ""
    with open(tex_path, "r", encoding="utf-8") as f:
        text = f.read()
    # Strip LaTeX formatting for readability
    text = re.sub(r'\\\w+\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\\w+', '', text)
    text = re.sub(r'[%$]', '', text)
    return text[:8000]


# ============================================================
# Weakness merging with source tracking
# ============================================================

def _merge_review_findings(reviews: Dict[str, Dict]) -> List[Dict]:
    """
    Merge weaknesses from all reviewers, preserving source information.
    Deduplicates near-identical weaknesses.
    Returns list of {id, source, text, source_reviewer}.
    """
    all_weaknesses = []
    for agent_name, review in reviews.items():
        weaknesses = review.get("Weaknesses", [])
        if isinstance(weaknesses, str):
            weaknesses = [weaknesses]
        for w in weaknesses:
            if w and len(w.strip()) > 10:
                all_weaknesses.append({
                    "source": agent_name,
                    "text": w.strip(),
                })

    # Simple deduplication by first 80 chars
    unique = []
    seen = set()
    for w in all_weaknesses:
        key = w["text"][:80].lower()
        if key not in seen:
            seen.add(key)
            unique.append(w)

    # Add IDs
    for i, w in enumerate(unique):
        w["id"] = i + 1

    return unique


# ============================================================
# Phase 2: Build Consensus
# ============================================================

def _build_consensus(
    weaknesses: List[Dict],
    reviews: Dict[str, Dict],
    paper_text: str,
    client,
    model,
    round_num: int,
) -> Dict:
    """
    Have the three reviewers discuss and try to reach consensus on each weakness.

    Returns a structured consensus result with:
    - consensus_points: weaknesses with agreement
    - disputed_points: weaknesses where consensus failed
    - overall_assessment: readiness evaluation
    """
    # Format reviews for the discussion prompt
    reviews_text_parts = []
    for agent_name, review in reviews.items():
        review_summary = {
            "reviewer": agent_name,
            "summary": review.get("Summary", ""),
            "strengths": review.get("Strengths", []),
            "weaknesses": review.get("Weaknesses", []),
            "overall": review.get("Overall", "N/A"),
            "decision": review.get("Decision", "N/A"),
        }
        reviews_text_parts.append(json.dumps(review_summary, indent=2, ensure_ascii=False))
    reviews_text = "\n\n---\n\n".join(reviews_text_parts)

    prompt = CONSENSUS_DISCUSSION_PROMPT.format(
        paper_excerpt=paper_text[:5000],
        reviews_text=reviews_text,
    )

    system_msg = (
        "You are a PANEL of three peer reviewers (Methodology, Experiments, Literature) "
        "discussing a paper together. You must simulate ALL THREE reviewers discussing "
        "each weakness. Your goal is to reach GENUINE consensus where possible. "
        "Output ONLY valid JSON in the exact format specified. No markdown fences outside the JSON."
    )

    for attempt in range(3):
        try:
            text, _ = get_response_from_llm(
                prompt,
                client=client,
                model=model,
                system_message=system_msg,
                temperature=0.4,  # Lower temp for more consistent consensus
            )
            result = extract_json_between_markers(text)
            if result and "consensus_points" in result:
                # Validate structure
                valid = True
                for cp in result.get("consensus_points", []):
                    if "weakness_id" not in cp or "agreement_level" not in cp:
                        valid = False
                        break
                if valid:
                    return result

            if attempt == 1:
                print("  Retrying consensus with stricter format...")
                prompt = (
                    "CRITICAL: Output ONLY the JSON object. No extra text. No markdown.\n\n"
                    + prompt
                )
        except Exception as e:
            print(f"  Consensus attempt {attempt + 1} failed: {e}")
            time.sleep(2)

    # Smart fallback: auto-consensus based on reviewer agreement heuristics
    print("  Using heuristic fallback for consensus...")
    return _heuristic_consensus(weaknesses, reviews)


def _heuristic_consensus(weaknesses: List[Dict], reviews: Dict[str, Dict]) -> Dict:
    """
    Heuristic consensus when LLM consensus building fails.
    If multiple reviewers independently found the same issue → strong signal.
    """
    consensus_points = []
    disputed_points = []

    for w in weaknesses:
        # Check which reviewers found this issue
        finding_reviewers = []
        all_weakness_texts = {}
        for agent_name, review in reviews.items():
            rev_weaknesses = review.get("Weaknesses", [])
            if isinstance(rev_weaknesses, str):
                rev_weaknesses = [rev_weaknesses]
            for rw in rev_weaknesses:
                if w["text"][:60].lower() in rw.lower() or rw[:60].lower() in w["text"].lower():
                    finding_reviewers.append(agent_name)
                    all_weakness_texts[agent_name] = rw[:100]

        n_finders = len(set(finding_reviewers))

        if n_finders >= 2:
            # 2+ reviewers found it → consensus
            agreement = "strong_consensus" if n_finders == 3 else "weak_consensus"
            consensus_points.append({
                "weakness_id": w["id"],
                "weakness_text": w["text"][:120],
                "source_reviewer": w.get("source", "Unknown"),
                "validity_consensus": "valid",
                "agreement_level": agreement,
                "agreed_fix": w["text"][:200],
                "priority": "high" if n_finders == 3 else "medium",
                "reviewer_positions": {
                    r: "agree" if r in finding_reviewers else "neutral"
                    for r in REVIEWER_PROMPTS
                },
            })
        elif n_finders == 1:
            # Only one reviewer found it → weak signal but still include
            consensus_points.append({
                "weakness_id": w["id"],
                "weakness_text": w["text"][:120],
                "source_reviewer": w.get("source", "Unknown"),
                "validity_consensus": "valid",
                "agreement_level": "weak_consensus",
                "agreed_fix": w["text"][:200],
                "priority": "low",
                "reviewer_positions": {
                    r: "agree" if r in finding_reviewers else "neutral"
                    for r in REVIEWER_PROMPTS
                },
            })
        else:
            disputed_points.append({
                "weakness_id": w["id"],
                "weakness_text": w["text"][:120],
                "positions": {r: "no_position" for r in REVIEWER_PROMPTS},
                "reason_for_dispute": "No reviewer clearly identified this issue",
            })

    # Determine if the paper is ready based on remaining high-priority issues
    high_priority = sum(1 for cp in consensus_points if cp.get("priority") == "high")
    all_valid = [cp for cp in consensus_points if cp.get("validity_consensus") == "valid"]

    paper_ready = high_priority == 0 and len(all_valid) <= 2

    return {
        "discussion_summary": f"Heuristic consensus: {len(consensus_points)} points with agreement, "
                              f"{len(disputed_points)} disputed. High priority issues: {high_priority}.",
        "consensus_points": consensus_points,
        "disputed_points": disputed_points,
        "overall_assessment": {
            "paper_ready": paper_ready,
            "remaining_major_issues": high_priority,
            "recommendation": "accept_as_is" if paper_ready else "minor_revision",
        },
    }


# ============================================================
# Phase 3: Voting Fallback
# ============================================================

def _resolve_by_voting(
    disputed_points: List[Dict],
    paper_text: str,
    client,
    model,
    consensus_rounds: int,
) -> List[Dict]:
    """
    When consensus fails, a neutral Chair agent casts tie-breaking votes.
    Returns resolved decisions for each disputed point.
    """
    if not disputed_points:
        return []

    prompt = VOTING_PROMPT.format(
        consensus_rounds=consensus_rounds,
        disputed_points_json=json.dumps(disputed_points, indent=2, ensure_ascii=False),
        paper_excerpt=paper_text[:3000],
    )

    system_msg = (
        "You are a neutral SENIOR AREA CHAIR with broad ML expertise. "
        "You are NOT any of the reviewers. Your job is to cast tie-breaking votes "
        "on disputed review points. Output ONLY valid JSON. No extra text."
    )

    for attempt in range(3):
        try:
            text, _ = get_response_from_llm(
                prompt,
                client=client,
                model=model,
                system_message=system_msg,
                temperature=0.2,  # Low temp for consistent, fair voting
            )
            result = extract_json_between_markers(text)
            if result and "chair_votes" in result:
                return result["chair_votes"]
        except Exception as e:
            print(f"  Voting attempt {attempt + 1} failed: {e}")
            time.sleep(2)

    # Fallback: accept all disputed points (conservative — better to fix more than less)
    print("  Voting fallback: accepting all disputed points (conservative default).")
    return [
        {
            "weakness_id": dp.get("weakness_id", i + 1),
            "vote": "accept",
            "reasoning": "Default accept — could not resolve dispute, erring on side of improvement",
        }
        for i, dp in enumerate(disputed_points)
    ]


# ============================================================
# Phase 4: Apply improvements
# ============================================================

def _apply_improvements(consensus_points, disputed_votes, tex_path, coder):
    """Feed consensus + voted changes to Aider for paper revision."""
    # Collect accepted changes
    accepted = []

    # From consensus: include all "valid" points
    for cp in consensus_points:
        if cp.get("validity_consensus") == "valid" and cp.get("agreed_fix"):
            priority = cp.get("priority", "medium")
            accepted.append({
                "text": cp["agreed_fix"],
                "priority": priority,
                "agreement": cp.get("agreement_level", "weak_consensus"),
            })

    # From voting: include all "accept" votes
    for vote in disputed_votes or []:
        if vote.get("vote") == "accept":
            accepted.append({
                "text": vote.get("reasoning", "Fix as recommended by reviewers"),
                "priority": "medium",
                "agreement": "voting_resolved",
            })

    if not accepted:
        print("  No changes to apply.")
        return False, accepted

    # Sort by priority (high first) and agreement strength
    priority_order = {"high": 0, "medium": 1, "low": 2}
    agreement_order = {"strong_consensus": 0, "weak_consensus": 1, "voting_resolved": 2}
    accepted.sort(key=lambda x: (priority_order.get(x["priority"], 1),
                                  agreement_order.get(x["agreement"], 1)))

    # Apply all accepted changes (no hard cap — natural convergence handles this)
    items = "\n".join(
        f"{i + 1}. [{a['agreement']}] [{a['priority']}] {a['text']}"
        for i, a in enumerate(accepted)
    )

    improvement_prompt = f"""Apply these SPECIFIC improvements to template.tex.
These changes have been validated through peer consensus.

{items}

GUIDELINES:
- Make targeted edits for each point above. Do NOT rewrite the whole paper.
- Do NOT add new experiments, datasets, or results that are not in the notes.
- Focus on clarity, precision, and addressing the specific weaknesses identified.
- After applying all changes, STOP. This is a targeted revision.
- Preserve the paper's core contributions and structure."""

    try:
        success = _run_coder_with_limit(coder, improvement_prompt)
        return success, accepted
    except Exception as e:
        print(f"  Aider improvement failed: {e}")
        return False, accepted


# ============================================================
# Main Debate Orchestrator (Consensus + Voting)
# ============================================================

def debate_paper(
    folder_name,
    client,
    model,
    coder,
    max_rounds: int = 3,
    consensus_rounds: int = 2,
    min_accepted: int = 0,  # No longer a hard cutoff — informational only
):
    """
    Run multi-agent peer review with CONSENSUS + VOTING mechanism.

    Flow:
      1. INDEPENDENT REVIEW: 3 specialized reviewers produce reviews
      2. CONSENSUS BUILDING: Reviewers discuss disagreements, try to reach consensus
      3. VOTING FALLBACK: If consensus fails, neutral Chair votes on disputed points
      4. APPLY CHANGES: Consensus + voted changes applied via Aider
      5. ITERATE: Loop until natural convergence

    Convergence criteria (natural, no hard cutoffs):
      - No high-priority issues remain
      - All reviewers agree paper is ready
      - No new weaknesses found compared to previous round
      - OR all 3 reviewers independently vote Accept

    Args:
        folder_name: Path to the paper directory
        client: LLM client
        model: LLM model name
        coder: Aider Coder instance for applying edits
        max_rounds: Maximum outer debate rounds (default 3)
        consensus_rounds: Max consensus discussion rounds before voting (default 2)
        min_accepted: Informational only — log if accepted changes drop below this

    Returns:
        dict: debate log with full trace
    """
    tex_path = osp.join(folder_name, "latex", "template.tex")
    if not osp.exists(tex_path):
        print("[Debate] No paper template found, skipping.")
        return {"error": "No template.tex found"}

    debate_log = {
        "folder": folder_name,
        "started_at": datetime.now().isoformat(),
        "mechanism": "consensus+voting",
        "rounds": 0,
        "converged": False,
        "convergence_reason": "",
        "rounds_detail": [],
    }

    prev_all_accepted = False
    prev_high_priority_count = float("inf")

    for round_num in range(1, max_rounds + 1):
        print(f"\n{'=' * 60}")
        print(f"[Debate] Round {round_num}/{max_rounds}")
        print(f"{'=' * 60}")

        # Load current paper text
        paper_text = _load_paper_text(folder_name)
        if not paper_text:
            print("[Debate] Could not load paper text.")
            break

        # ============================================
        # PHASE 1: Independent Review
        # ============================================
        print("[Phase 1] Independent Review — 3 specialized reviewers...")
        reviews = {}
        all_accept = True
        for agent_name, system_prompt in REVIEWER_PROMPTS.items():
            print(f"  - {agent_name} reviewer...")
            try:
                review = perform_review(
                    paper_text,
                    model=model,
                    client=client,
                    num_reflections=1,
                    num_fs_examples=0,
                    num_reviews_ensemble=1,
                    temperature=0.7,
                    reviewer_system_prompt=system_prompt + "\n\n" + REVIEW_FORM,
                    review_instruction_form="",
                )
                reviews[agent_name] = review
                w_count = len(review.get("Weaknesses", []))
                decision = review.get("Decision", "Reject")
                score = review.get("Overall", "?")
                print(f"    {w_count} weaknesses, Score: {score}/10, Decision: {decision}")
                if decision != "Accept":
                    all_accept = False
            except Exception as e:
                print(f"    Failed: {e}")
                reviews[agent_name] = {"Weaknesses": [], "Decision": "Error", "error": str(e)}
                all_accept = False
            time.sleep(1)

        # Check natural convergence: all 3 reviewers independently Accept
        if all_accept:
            print("\n[Convergence] All 3 reviewers independently voted ACCEPT — paper is ready!")
            debate_log["converged"] = True
            debate_log["convergence_reason"] = "unanimous_accept"
            debate_log["rounds_detail"].append({
                "round": round_num,
                "phase": "independent_review",
                "reviews": reviews,
                "converged": True,
                "reason": "All reviewers accepted",
            })
            break

        # Merge weaknesses
        all_weaknesses = _merge_review_findings(reviews)
        print(f"\n[Phase 1 Result] {len(all_weaknesses)} unique weaknesses across all reviewers.")

        if not all_weaknesses:
            print("[Convergence] No weaknesses found — paper is ready!")
            debate_log["converged"] = True
            debate_log["convergence_reason"] = "no_weaknesses"
            break

        # ============================================
        # PHASE 2: Consensus Building
        # ============================================
        print(f"\n[Phase 2] Consensus Building (up to {consensus_rounds} discussion rounds)...")

        consensus_result = None
        for cr in range(1, consensus_rounds + 1):
            print(f"  Consensus round {cr}/{consensus_rounds}...")
            consensus_result = _build_consensus(
                all_weaknesses, reviews, paper_text, client, model, cr
            )

            n_consensus = len(consensus_result.get("consensus_points", []))
            n_disputed = len(consensus_result.get("disputed_points", []))
            strong = sum(1 for cp in consensus_result.get("consensus_points", [])
                        if cp.get("agreement_level") == "strong_consensus")
            weak = sum(1 for cp in consensus_result.get("consensus_points", [])
                      if cp.get("agreement_level") == "weak_consensus")
            print(f"    Consensus: {n_consensus} points (strong={strong}, weak={weak}), "
                  f"Disputed: {n_disputed}")

            if n_disputed == 0:
                print(f"    Full consensus reached after {cr} discussion round(s)!")
                break

        n_disputed = len(consensus_result.get("disputed_points", []))
        consensus_points = consensus_result.get("consensus_points", [])

        # ============================================
        # PHASE 3: Voting Fallback (only if disputed points remain)
        # ============================================
        disputed_votes = []
        if n_disputed > 0:
            print(f"\n[Phase 3] Voting Fallback — {n_disputed} disputed points unresolved.")
            print(f"  Consensus did not converge after {consensus_rounds} rounds.")
            print(f"  Switching to majority voting with neutral Chair tie-breaker...")

            disputed_votes = _resolve_by_voting(
                consensus_result["disputed_points"],
                paper_text,
                client,
                model,
                consensus_rounds,
            )
            n_accept = sum(1 for v in disputed_votes if v.get("vote") == "accept")
            n_reject = sum(1 for v in disputed_votes if v.get("vote") == "reject")
            print(f"  Voting result: {n_accept} accepted, {n_reject} rejected")
        else:
            print(f"\n[Phase 3] Skipped — no disputed points, consensus was sufficient.")

        # ============================================
        # PHASE 4: Apply & Iterate
        # ============================================
        print(f"\n[Phase 4] Applying changes via Aider...")
        success, applied = _apply_improvements(
            consensus_points, disputed_votes, tex_path, coder
        )

        # Count by agreement type
        high_priority = sum(1 for a in applied if a.get("priority") == "high")
        strong_consensus_count = sum(1 for a in applied if a.get("agreement") == "strong_consensus")
        weak_consensus_count = sum(1 for a in applied if a.get("agreement") == "weak_consensus")
        voting_count = sum(1 for a in applied if a.get("agreement") == "voting_resolved")

        print(f"  Applied: {len(applied)} changes")
        print(f"    Strong consensus: {strong_consensus_count}")
        print(f"    Weak consensus:   {weak_consensus_count}")
        print(f"    Voting resolved:  {voting_count}")
        print(f"    High priority:    {high_priority}")

        # Record round details
        debate_log["rounds_detail"].append({
            "round": round_num,
            "phase1_reviews": reviews,
            "phase2_consensus": {
                "consensus_points": consensus_points,
                "disputed_points": consensus_result.get("disputed_points", []),
                "overall_assessment": consensus_result.get("overall_assessment", {}),
            },
            "phase3_votes": disputed_votes,
            "phase4_applied": {
                "total": len(applied),
                "strong_consensus": strong_consensus_count,
                "weak_consensus": weak_consensus_count,
                "voting": voting_count,
                "high_priority": high_priority,
            },
            "applied": applied,
            "success": success,
        })

        # ============================================
        # NATURAL CONVERGENCE CHECK
        # ============================================

        # Criterion 1: No high-priority issues found
        if high_priority == 0 and len(applied) == 0:
            print("\n[Convergence] No issues to apply — paper is ready!")
            debate_log["converged"] = True
            debate_log["convergence_reason"] = "no_issues_to_apply"
            break

        # Criterion 2: Diminishing returns — high-priority count not decreasing
        if round_num > 1 and high_priority >= prev_high_priority_count and len(applied) <= 2:
            print(f"\n[Convergence] Diminishing returns — "
                  f"high priority issues stable at {high_priority}.")
            debate_log["converged"] = True
            debate_log["convergence_reason"] = "diminishing_returns"
            break

        # Criterion 3: Paper agent assessment says ready
        overall = consensus_result.get("overall_assessment", {})
        if overall.get("paper_ready"):
            print(f"\n[Convergence] Consensus assessment: paper is ready.")
            debate_log["converged"] = True
            debate_log["convergence_reason"] = "consensus_ready"
            break

        if not success:
            print("\n[Warning] Aider failed to apply changes this round.")

        prev_high_priority_count = high_priority
        debate_log["rounds"] = round_num

    # ============================================
    # Save debate log
    # ============================================
    log_path = osp.join(folder_name, "debate_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(debate_log, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[Debate] Log saved: {log_path}")

    # Print final summary
    print(f"\n{'=' * 60}")
    print(f"DEBATE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Rounds:           {debate_log['rounds']}")
    print(f"  Converged:        {debate_log['converged']}")
    print(f"  Reason:           {debate_log['convergence_reason']}")
    if debate_log["rounds_detail"]:
        last = debate_log["rounds_detail"][-1]
        applied = last.get("phase4_applied", {})
        print(f"  Final round changes:")
        print(f"    Strong consensus: {applied.get('strong_consensus', 0)}")
        print(f"    Weak consensus:   {applied.get('weak_consensus', 0)}")
        print(f"    Voting resolved:  {applied.get('voting', 0)}")
        print(f"    Total applied:    {applied.get('total', 0)}")

    return debate_log


# ============================================================
# Legacy compatibility wrapper
# ============================================================
# The old single-agent evaluate_critiques and related functions are replaced
# by the consensus+voting mechanism above. These stubs exist for reference.

def _evaluate_critiques(weaknesses, paper_text, client, model):
    """
    [DEPRECATED] Old single-agent evaluation.
    Replaced by _build_consensus() + _resolve_by_voting().
    """
    print("[WARNING] _evaluate_critiques is deprecated. Use consensus+voting instead.")
    return []


def _merge_weaknesses(reviews):
    """
    [DEPRECATED] Old merge function without source tracking.
    Replaced by _merge_review_findings().
    """
    return _merge_review_findings(reviews)
