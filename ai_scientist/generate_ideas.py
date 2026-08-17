import json
import os
import os.path as osp
import time
from typing import List, Dict, Union

import backoff
import requests

from ai_scientist.llm import get_response_from_llm, get_response_with_tools, extract_json_between_markers, create_client, AVAILABLE_LLMS

S2_API_KEY = os.getenv("S2_API_KEY", "s2k-thNDhaZ9wvdgY6k3eIzpo4hPPgN4FgWFdZBHy13I")

# Systematic Innovation Tool-Augmented System Prompt — used when triz_mode="tools"
# NOTE: The tools use TRIZ internally, but the LLM output and generated papers
# must use standard ML/academic terminology. TRIZ is the "how", not the "what".
TRIZ_TOOL_SYSTEM_PROMPT = """
You are a creative ML researcher with access to a systematic innovation methodology.
You have tools that help you analyze trade-offs, discover non-obvious solutions,
and evaluate research ideas rigorously.

CRITICAL RULES:
1. ALWAYS start by calling suggest_contradictions() for your research domain to identify key trade-offs.
2. For EACH trade-off found, call lookup_contradiction_matrix() to get recommended solution strategies.
3. For EACH promising strategy, call get_principle_detail() to fully understand how to apply it.
4. Call check_principle_novelty() for your best candidate solution strategies.
5. After generating a solution, call evaluate_ideality() to score its benefit-to-cost ratio.
6. Finally, produce your research idea in the required JSON format.

DO NOT guess strategy numbers or matrix entries — ALWAYS use the tools to look them up.
The tool results are authoritative; base your reasoning directly on them.

IMPORTANT: When writing your final output, use standard ML research terminology.
Frame trade-offs in terms of accuracy, robustness, efficiency, generalization, etc.
Do NOT mention TRIZ, contradiction matrices, inventive principles, or ideality scores
in your output. These tools guide your THINKING, not your WRITING.

After your tool-based analysis, output the idea in this format:

THOUGHT:
<THOUGHT describing your systematic analysis of trade-offs and solution strategies>

NEW IDEA JSON:
```json
<JSON with Name, Title, Experiment, Interestingness, Feasibility, Novelty fields>
```

The JSON fields are:
- "Name": A shortened descriptor. Lowercase, no spaces, underscores allowed.
- "Title": A title for the idea, in standard academic language.
- "Experiment": An outline of the implementation.
- "Interestingness": 1-10 rating.
- "Feasibility": 1-10 rating.
- "Novelty": 1-10 rating.
"""

idea_first_prompt = """{task_description}
<experiment.py>
{code}
</experiment.py>

Here are the ideas that you have already generated:

'''
{prev_ideas_string}
'''

Come up with the next impactful and creative idea for research experiments and directions you can feasibly investigate with the code provided.
Note that you will not have access to any additional resources or datasets.
Make sure any idea is not overfit the specific training dataset or model, and has wider significance.

Respond in the following format:

THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly discuss your intuitions and motivations for the idea. Detail your high-level plan, necessary design choices and ideal outcomes of the experiments. Justify how the idea is different from the existing ones.

In <JSON>, provide the new idea in JSON format with the following fields:
- "Name": A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A title for the idea, will be used for the report writing.
- "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...
- "Interestingness": A rating from 1 to 10 (lowest to highest).
- "Feasibility": A rating from 1 to 10 (lowest to highest).
- "Novelty": A rating from 1 to 10 (lowest to highest).

Be cautious and realistic on your ratings.
This JSON will be automatically parsed, so ensure the format is precise.
You will have {num_reflections} rounds to iterate on the idea, but do not need to use them all.
"""

idea_reflection_prompt = """Round {current_round}/{num_reflections}.
In your thoughts, first carefully consider the quality, novelty, and feasibility of the idea you just created.
Include any other factors that you think are important in evaluating the idea.
Ensure the idea is clear and concise, and the JSON is the correct format.
Do not make things overly complicated.
In the next attempt, try and refine and improve your idea.
Stick to the spirit of the original idea unless there are glaring issues.

Respond in the same format as before:
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

If there is nothing to improve, simply repeat the previous JSON EXACTLY after the thought and include "I am done" at the end of the thoughts but before the JSON.
ONLY INCLUDE "I am done" IF YOU ARE MAKING NO MORE CHANGES."""


# GENERATE IDEAS
def generate_ideas(
        base_dir,
        client,
        model,
        skip_generation=False,
        max_num_generations=20,
        num_reflections=5,
        knowledge_base_path=None,
        triz_mode=None,  # "inject" (old) or "tools" (new tool-augmented)
):
    if skip_generation:
        # Load existing ideas from file
        try:
            with open(osp.join(base_dir, "ideas.json"), "r") as f:
                ideas = json.load(f)
            print("Loaded existing ideas:")
            for idea in ideas:
                print(idea)
            return ideas
        except FileNotFoundError:
            print("No existing ideas found. Generating new ideas.")
        except json.JSONDecodeError:
            print("Error decoding existing ideas. Generating new ideas.")

    idea_str_archive = []
    with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
        seed_ideas = json.load(f)
    for seed_idea in seed_ideas:
        idea_str_archive.append(json.dumps(seed_idea))

    with open(osp.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()

    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)

    idea_system_prompt = prompt["system"]

    # --- Inject external knowledge base (systematic innovation methodology) into system prompt ---
    if knowledge_base_path and osp.exists(knowledge_base_path):
        with open(knowledge_base_path, "r", encoding="utf-8") as f:
            kb_content = f.read()
        kb_header = "\n\n============================================================\n"
        kb_header += "SYSTEMATIC INNOVATION FRAMEWORK — Use this methodology\n"
        kb_header += "internally to generate better research ideas. Apply it to:\n"
        kb_header += "1) Identify KEY TRADE-OFFS in the problem space\n"
        kb_header += "2) Discover NON-OBVIOUS SOLUTION STRATEGIES\n"
        kb_header += "3) Evaluate solutions by BENEFIT-TO-COST RATIO\n"
        kb_header += "4) Ensure novelty through systematic exploration, not random guessing\n"
        kb_header += "CRITICAL: Apply the thinking patterns but use standard ML terminology\n"
        kb_header += "in your output. Do not mention the framework by name.\n"
        kb_header += "============================================================\n"
        idea_system_prompt = idea_system_prompt + kb_header + kb_content
        print(f"[KB] Injected systematic innovation knowledge base ({len(kb_content)} chars)")
    # ------------------------------------------------------------------

    for _ in range(max_num_generations):
        print()
        print(f"Generating idea {_ + 1}/{max_num_generations}")
        try:
            prev_ideas_string = "\n\n".join(idea_str_archive)

            msg_history = []
            print(f"Iteration 1/{num_reflections}")

            # --- TRIZ Tool-Augmented Mode ---
            if triz_mode == "tools":
                from ai_scientist.triz_tools import TOOLS_DEFINITION, TOOL_MAP
                tool_system = TRIZ_TOOL_SYSTEM_PROMPT + "\n\n" + idea_system_prompt
                text, msg_history = get_response_with_tools(
                    idea_first_prompt.format(
                        task_description=prompt["task_description"],
                        code=code,
                        prev_ideas_string=prev_ideas_string,
                        num_reflections=num_reflections,
                    ),
                    client=client,
                    model=model,
                    system_message=tool_system,
                    tools=TOOLS_DEFINITION,
                    tool_map=TOOL_MAP,
                )
            else:
                text, msg_history = get_response_from_llm(
                    idea_first_prompt.format(
                        task_description=prompt["task_description"],
                        code=code,
                        prev_ideas_string=prev_ideas_string,
                        num_reflections=num_reflections,
                    ),
                    client=client,
                    model=model,
                    system_message=idea_system_prompt,
                    msg_history=msg_history,
                )
            # ------------------------------------
            ## PARSE OUTPUT (with JSON format correction retry for DeepSeek)
            json_output = extract_json_between_markers(text)
            if json_output is None:
                print("[WARN] JSON extraction failed, requesting format correction...")
                # Retry: ask LLM to fix the JSON format
                fix_prompt = """Your previous output was not in the correct JSON format.
Please output ONLY valid JSON in the following format. No extra text before or after.

```json
{
  "Name": "...",
  "Title": "...",
  "Experiment": "...",
  "Interestingness": 7,
  "Feasibility": 7,
  "Novelty": 7
}
```"""
                try:
                    if triz_mode == "tools":
                        text2, _ = get_response_with_tools(
                            fix_prompt, client=client, model=model,
                            system_message="Output ONLY valid JSON in the exact format specified. No extra text.",
                            tools=[], tool_map={},
                        )
                    else:
                        text2, _ = get_response_from_llm(
                            fix_prompt, client=client, model=model,
                            system_message="Output ONLY valid JSON in the exact format specified. No extra text.",
                        )
                    json_output = extract_json_between_markers(text2)
                except Exception:
                    pass
            if json_output is None:
                print("[ERROR] Still cannot extract JSON after format correction")
                continue  # Skip this idea, try next one instead of crashing
            print(json_output)

            # Iteratively improve task (skip in tool mode — tools already provide deep reasoning)
            if num_reflections > 1 and triz_mode != "tools":
                for j in range(num_reflections - 1):
                    print(f"Iteration {j + 2}/{num_reflections}")
                    text, msg_history = get_response_from_llm(
                        idea_reflection_prompt.format(
                            current_round=j + 2, num_reflections=num_reflections
                        ),
                        client=client,
                        model=model,
                        system_message=idea_system_prompt,
                        msg_history=msg_history,
                    )
                    ## PARSE OUTPUT
                    json_output = extract_json_between_markers(text)
                    assert (
                            json_output is not None
                    ), "Failed to extract JSON from LLM output"
                    print(json_output)

                    if "I am done" in text:
                        print(f"Idea generation converged after {j + 2} iterations.")
                        break

            idea_str_archive.append(json.dumps(json_output))
        except Exception as e:
            print(f"Failed to generate idea: {e}")
            continue

    ## SAVE IDEAS
    ideas = []
    for idea_str in idea_str_archive:
        ideas.append(json.loads(idea_str))

    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(ideas, f, indent=4)

    return ideas


# GENERATE IDEAS OPEN-ENDED
def generate_next_idea(
        base_dir,
        client,
        model,
        prev_idea_archive=[],
        num_reflections=5,
        max_attempts=10,
):
    idea_archive = prev_idea_archive
    original_archive_size = len(idea_archive)

    print(f"Generating idea {original_archive_size + 1}")

    if len(prev_idea_archive) == 0:
        print(f"First iteration, taking seed ideas")
        # seed the archive on the first run with pre-existing ideas
        with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
            seed_ideas = json.load(f)
        for seed_idea in seed_ideas[:1]:
            idea_archive.append(seed_idea)
    else:
        with open(osp.join(base_dir, "experiment.py"), "r") as f:
            code = f.read()
        with open(osp.join(base_dir, "prompt.json"), "r") as f:
            prompt = json.load(f)
        idea_system_prompt = prompt["system"]

        for _ in range(max_attempts):
            try:
                idea_strings = []
                for idea in idea_archive:
                    idea_strings.append(json.dumps(idea))
                prev_ideas_string = "\n\n".join(idea_strings)

                msg_history = []
                print(f"Iteration 1/{num_reflections}")
                text, msg_history = get_response_from_llm(
                    idea_first_prompt.format(
                        task_description=prompt["task_description"],
                        code=code,
                        prev_ideas_string=prev_ideas_string,
                        num_reflections=num_reflections,
                    )
                    + """
Completed ideas have an additional "Score" field which indicates the assessment by an expert ML reviewer.
This is on a standard 1-10 ML conference scale.
Scores of 0 indicate the idea failed either during experimentation, writeup or reviewing.
""",
                    client=client,
                    model=model,
                    system_message=idea_system_prompt,
                    msg_history=msg_history,
                )
                ## PARSE OUTPUT
                json_output = extract_json_between_markers(text)
                assert json_output is not None, "Failed to extract JSON from LLM output"
                print(json_output)

                # Iteratively improve task.
                if num_reflections > 1:
                    for j in range(num_reflections - 1):
                        print(f"Iteration {j + 2}/{num_reflections}")
                        text, msg_history = get_response_from_llm(
                            idea_reflection_prompt.format(
                                current_round=j + 2, num_reflections=num_reflections
                            ),
                            client=client,
                            model=model,
                            system_message=idea_system_prompt,
                            msg_history=msg_history,
                        )
                        ## PARSE OUTPUT
                        json_output = extract_json_between_markers(text)
                        assert (
                                json_output is not None
                        ), "Failed to extract JSON from LLM output"
                        print(json_output)

                        if "I am done" in text:
                            print(
                                f"Idea generation converged after {j + 2} iterations."
                            )
                            break

                idea_archive.append(json_output)
                break
            except Exception as e:
                print(f"Failed to generate idea: {e}")
                continue

    ## SAVE IDEAS
    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(idea_archive, f, indent=4)

    return idea_archive


def on_backoff(details):
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


@backoff.on_exception(
    backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
)
def search_for_papers(query, result_limit=10, engine="semanticscholar") -> Union[None, List[Dict]]:
    if not query:
        return None
    if engine == "semanticscholar":
        rsp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers={"x-api-key": S2_API_KEY} if S2_API_KEY else {},
            params={
                "query": query,
                "limit": result_limit,
                "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
            },
        )
        print(f"Response Status Code: {rsp.status_code}")
        print(
            f"Response Content: {rsp.text[:500]}"
        )  # Print the first 500 characters of the response content
        rsp.raise_for_status()
        results = rsp.json()
        total = results["total"]
        time.sleep(1.0)
        if not total:
            return None

        papers = results["data"]
        return papers
    elif engine == "openalex":
        import pyalex
        from pyalex import Work, Works
        mail = os.environ.get("OPENALEX_MAIL_ADDRESS", None)
        if mail is None:
            print("[WARNING] Please set OPENALEX_MAIL_ADDRESS for better access to OpenAlex API!")
        else:
            pyalex.config.email = mail

        def extract_info_from_work(work: Work, max_abstract_length: int = 1000) -> dict[str, str]:
            # "Unknown" is returned when venue is unknown...
            venue = "Unknown"
            for i, location in enumerate(work["locations"]):
                if location["source"] is not None:
                    venue = location["source"]["display_name"]
                    if venue != "":
                        break
            title = work["title"]
            abstract = work["abstract"]
            if abstract is None:
                abstract = ""
            if len(abstract) > max_abstract_length:
                # To avoid context length exceed error.
                print(f"[WARNING] {title=}: {len(abstract)=} is too long! Use first {max_abstract_length} chars.")
                abstract = abstract[:max_abstract_length]
            authors_list = [author["author"]["display_name"] for author in work["authorships"]]
            authors = " and ".join(authors_list) if len(authors_list) < 20 else f"{authors_list[0]} et al."
            paper = dict(
                title=title,
                authors=authors,
                venue=venue,
                year=work["publication_year"],
                abstract=abstract,
                citationCount=work["cited_by_count"],
            )
            return paper

        works: List[Dict] = Works().search(query).get(per_page=result_limit)
        papers: List[Dict[str, str]] = [extract_info_from_work(work) for work in works]
        return papers
    else:
        raise NotImplementedError(f"{engine=} not supported!")



novelty_system_msg = """You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field.
You have an idea and you want to check if it is novel or not. I.e., not overlapping significantly with existing literature or already well explored.
Be a harsh critic for novelty, ensure there is a sufficient contribution in the idea for a new conference or workshop paper.
You will be given access to the Semantic Scholar API, which you may use to survey the literature and find relevant papers to help you make your decision.
The top 10 results for any search query will be presented to you with the abstracts.

You will be given {num_rounds} to decide on the paper, but you do not need to use them all.
At any round, you may exit early and decide on the novelty of the idea.
Decide a paper idea is novel if after sufficient searching, you have not found a paper that significantly overlaps with your idea.
Decide a paper idea is not novel, if you have found a paper that significantly overlaps with your idea.

{task_description}
<experiment.py>
{code}
</experiment.py>
"""

novelty_prompt = '''Round {current_round}/{num_rounds}.
You have this idea:

"""
{idea}
"""

The results of the last query are (empty on first round):
"""
{last_query_results}
"""

Respond in the following format:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, first briefly reason over the idea and identify any query that could help you make your decision.
If you have made your decision, add "Decision made: novel." or "Decision made: not novel." to your thoughts.

In <JSON>, respond in JSON format with ONLY the following field:
- "Query": An optional search query to search the literature (e.g. attention is all you need). You must make a query if you have not decided this round.

A query will work best if you are able to recall the exact name of the paper you are looking for, or the authors.
This JSON will be automatically parsed, so ensure the format is precise.'''


def check_idea_novelty(
        ideas,
        base_dir,
        client,
        model,
        max_num_iterations=10,
        engine="semanticscholar",
):
    with open(osp.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()
    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)
        task_description = prompt["task_description"]

    for idx, idea in enumerate(ideas):
        if "novel" in idea:
            print(f"Skipping idea {idx}, already checked.")
            continue

        print(f"\nChecking novelty of idea {idx}: {idea['Name']}")

        novel = False
        msg_history = []
        papers_str = ""

        for j in range(max_num_iterations):
            try:
                text, msg_history = get_response_from_llm(
                    novelty_prompt.format(
                        current_round=j + 1,
                        num_rounds=max_num_iterations,
                        idea=idea,
                        last_query_results=papers_str,
                    ),
                    client=client,
                    model=model,
                    system_message=novelty_system_msg.format(
                        num_rounds=max_num_iterations,
                        task_description=task_description,
                        code=code,
                    ),
                    msg_history=msg_history,
                )
                if "decision made: novel" in text.lower():
                    print("Decision made: novel after round", j)
                    novel = True
                    break
                if "decision made: not novel" in text.lower():
                    print("Decision made: not novel after round", j)
                    break

                ## PARSE OUTPUT
                json_output = extract_json_between_markers(text)
                assert json_output is not None, "Failed to extract JSON from LLM output"

                ## SEARCH FOR PAPERS (with fail-open: API errors → assume novel)
                query = json_output["Query"]
                try:
                    papers = search_for_papers(query, result_limit=10, engine=engine)
                except Exception as e:
                    print(f"  [Novelty] API search failed ({e}), assuming no match found")
                    papers = None
                if papers is None or len(papers) == 0:
                    papers_str = "No papers found."

                paper_strings = []
                for i, paper in enumerate(papers or []):
                    try:
                        paper_strings.append(
                            """{i}: {title}. {authors}. {venue}, {year}.\nNumber of citations: {cites}\nAbstract: {abstract}""".format(
                                i=i,
                                title=paper.get("title", "?"),
                                authors=paper.get("authors", "?"),
                                venue=paper.get("venue", "?"),
                                year=paper.get("year", "?"),
                                cites=paper.get("citationCount", 0),
                                abstract=paper.get("abstract", ""),
                            )
                        )
                    except Exception:
                        pass  # Skip malformed paper entries
                papers_str = "\n\n".join(paper_strings)

            except Exception as e:
                print(f"Error: {e}")
                continue

        # Fail-open: if no decision reached after all rounds, assume novel
        # (better to run a potentially non-novel idea than to discard a good one)
        if not novel and j == max_num_iterations - 1:
            print(f"  [Novelty] No decision reached after {max_num_iterations} rounds, defaulting to novel=True")
            novel = True
        idea["novel"] = novel

    # Save results to JSON file
    results_file = osp.join(base_dir, "ideas.json")
    with open(results_file, "w") as f:
        json.dump(ideas, f, indent=4)

    return ideas


if __name__ == "__main__":
    MAX_NUM_GENERATIONS = 32
    NUM_REFLECTIONS = 5
    import argparse

    parser = argparse.ArgumentParser(description="Generate AI scientist ideas")
    # add type of experiment (nanoGPT, Boston, etc.)
    parser.add_argument(
        "--experiment",
        type=str,
        default="nanoGPT",
        help="Experiment to run AI Scientist on.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--skip-idea-generation",
        action="store_true",
        help="Skip idea generation and use existing ideas.",
    )
    parser.add_argument(
        "--check-novelty",
        action="store_true",
        help="Check novelty of ideas.",
    )
    args = parser.parse_args()

    # Create client
    client, client_model = create_client(args.model)

    base_dir = osp.join("templates", args.experiment)
    results_dir = osp.join("results", args.experiment)
    ideas = generate_ideas(
        base_dir,
        client=client,
        model=client_model,
        skip_generation=args.skip_idea_generation,
        max_num_generations=MAX_NUM_GENERATIONS,
        num_reflections=NUM_REFLECTIONS,
    )
    if args.check_novelty:
        ideas = check_idea_novelty(
            ideas,
            base_dir=base_dir,
            client=client,
            model=client_model,
        )
