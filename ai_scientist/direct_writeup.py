#!/usr/bin/env python3
"""
Direct Paper Writer — bypasses Aider, calls LLM directly.
Generates complete LaTeX paper + real references from idea + analysis.
"""
import json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_scientist.llm import get_response_from_llm, create_client

# ============================================================
# Figure availability — injected into prompts so LLM doesn't invent phantom figures
_FIGURE_INFO_TEMPLATE = """\n\n== AVAILABLE FIGURES (ON DISK) ==
The paper has EXACTLY these image files available: {figure_list}.
- Use method_figure.png for the methodological overview / framework schematic.
- Use results_plot_1.png for the primary quantitative result figure.
- Use results_plot_2.png for the secondary result figure (e.g., ablation, calibration, or comparison).
- IMPORTANT: Do NOT invent additional figure filenames (no fig1.pdf, fig_convergence.pdf, etc.).
- Reference ONLY the figures listed above in \\includegraphics and \\ref commands.
- Each figure should appear AT MOST ONCE in the paper.
- For the Results section: place results_plot_1.png AND results_plot_2.png at appropriate positions within the text (not all at the end)."""

SECTION_PROMPTS = {
    "title_abstract_title": """You are writing a scientific paper.

RESEARCH IDEA: {idea_json}
PAPER ANALYSIS: {analysis_json}

Write a compelling, specific title for this {domain} paper. Output ONLY the title text — no LaTeX commands, no \\title{{}}, no formatting. Just the plain title string.

The title should capture: the main innovation, the domain context, and hint at the key result or contribution.""",

    "title_abstract_abstract": """You are writing a scientific paper.

RESEARCH IDEA: {idea_json}
PAPER ANALYSIS: {analysis_json}

Write a 5-7 sentence abstract for this {domain} paper. Output ONLY the abstract text — no LaTeX commands, no \\begin{{abstract}} or \\end{{abstract}}. Just the plain paragraph.

Cover: (1) the problem/motivation, (2) the key trade-off or gap, (3) your proposed method/approach, (4) main results or findings, (5) broader impact or significance.""",

    "introduction": """RESEARCH IDEA: {idea_json}
PAPER ANALYSIS: {analysis_json}

Write a complete Introduction section (3-4 paragraphs). Cover: (1) broader context, (2) the key trade-off limiting current approaches, (3) existing methods and their limits, (4) your proposed approach and innovations, (5) list of contributions. Use \\cite{{...}} with MEANINGFUL, UNIQUE keys for each paper you reference (e.g., \\cite{{hell2009sted}}, \\cite{{betzig2006palm}}). Never use placeholder keys. Output the complete \\section{{Introduction}}... block.""",

    "related_work": """RESEARCH IDEA: {idea_json}
Write a Related Work section (2-3 paragraphs) for a {domain} paper. Compare and contrast prior work with your approach. Use \\cite{{...}} with unique, MEANINGFUL keys. Output the complete \\section{{Related Work}}... block.""",

    "background": """RESEARCH IDEA: {idea_json}
Write a Background section (2-3 paragraphs) covering foundational concepts and problem setting. Output the complete \\section{{Background}}... block.""",

    "method": """RESEARCH IDEA: {idea_json}
PAPER ANALYSIS: {analysis_json}
{available_figures}
Write a complete Method section (4-5 subsections). Include EXACTLY ONE figure block — referencing method_figure.png — placed at the beginning of the section (before the first subsection). Use \\begin{{figure}}[htbp] for the figure placement. Fill in the figure caption with a specific, detailed description of what the schematic shows. Use \\cite{{...}} with unique keys. Include mathematical formulation where appropriate.
CRITICAL: Write ONLY ONE \\begin{{figure}}...\\end{{figure}} block in this section. IMPORTANT: Use \\begin{{figure}}[htbp] NOT [t].""",

    "experimental_setup": """RESEARCH IDEA: {idea_json}
Write an Experimental Setup section covering datasets, metrics, baselines, implementation details. Be specific. Do NOT include any figures in this section. Output the complete \\section{{Experimental Setup}}... block.""",

    "results": """RESEARCH IDEA: {idea_json}
{available_figures}
Write a Results section presenting main quantitative results, ablation findings, and key insights. Include EXACTLY TWO figure blocks: results_plot_1.png (primary result) and results_plot_2.png (secondary analysis/ablation). Use \\begin{{figure}}[htbp] NOT [t]. Place them at natural positions within the text — do NOT cluster all figures together at the end. Fill in figure captions with specific descriptions referencing the actual data shown.
CRITICAL: Write EXACTLY TWO \\begin{{figure}}[htbp]...\\end{{figure}} blocks (one for results_plot_1.png, one for results_plot_2.png). No more, no less. Use [htbp] placement.""",

    "conclusion": """RESEARCH IDEA: {idea_json}
Write a Conclusions and Future Work section. Summarize contributions, discuss limitations, suggest future work. Do NOT include any figures. Output the complete \\section{{Conclusions and Future Work}}... block."""
}


def _get_available_figures_info(output_dir):
    """Return a string describing which figures are on disk, for injection into prompts."""
    parent_dir = os.path.dirname(output_dir.rstrip("/\\"))
    images = []
    for d in [output_dir, parent_dir]:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".pdf")):
                images.append(f)
    images = list(dict.fromkeys(images))  # deduplicate, preserve order
    if not images:
        return ""
    return _FIGURE_INFO_TEMPLATE.format(figure_list=", ".join(images))


def generate_section(name, idea, analysis, client, model, extra_kwargs=None):
    prompt = SECTION_PROMPTS[name].format(
        idea_json=json.dumps(idea, indent=2, ensure_ascii=False),
        analysis_json=json.dumps(analysis, indent=2, ensure_ascii=False),
        domain=analysis.get("domain", "scientific research"),
        available_figures=extra_kwargs.get("available_figures", "") if extra_kwargs else "",
    )
    system_msg = f"You are an expert scientific writer in {analysis.get('domain', 'research')}. Output ONLY valid LaTeX. No markdown fences."
    print(f"  [{name}] Generating...")
    resp, _ = get_response_from_llm(prompt, client=client, model=model, system_message=system_msg, temperature=0.3)
    resp = resp.strip()
    if resp.startswith("```"):
        resp = re.sub(r"^```\w*\n?", "", resp)
        resp = re.sub(r"\n?```$", "", resp)
    return resp


def _dedup_figures(template_text):
    """
    Remove duplicate \\begin{figure}...\\end{figure} blocks that reference
    the same image file. Keeps only the first occurrence of each unique image.
    Also removes figure blocks with no \\includegraphics command.
    """
    fig_re = re.compile(r'(\\begin\{figure\}.*?\\end\{figure\})', re.DOTALL)
    img_re = re.compile(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}')

    seen_images = set()
    cleaned = template_text
    for fm in reversed(list(fig_re.finditer(template_text))):
        block = fm.group(1)
        img_match = img_re.search(block)
        if not img_match:
            # No image reference — remove this block
            cleaned = cleaned[:fm.start()] + cleaned[fm.end():]
            continue
        img_name = img_match.group(1).strip()
        # Normalize: strip extensions for comparison
        img_key = img_name.rsplit(".", 1)[0] if "." in img_name else img_name
        if img_key in seen_images:
            print(f"  [dedup] Removed duplicate figure block referencing '{img_name}'")
            cleaned = cleaned[:fm.start()] + cleaned[fm.end():]
        else:
            seen_images.add(img_key)
    return cleaned


def generate_references(template_text, idea, analysis, client, model):
    """Extract citation keys from text, generate matching bibtex in batches."""
    cite_keys = re.findall(r"\\cite[a-z]*\{([^}]+)\}", template_text)
    all_keys = set()
    for ck in cite_keys:
        for k in ck.split(","):
            k = k.strip()
            if k and not k.startswith("placeholder") and not k.startswith("ref1") and len(k) > 3:
                all_keys.add(k)

    if not all_keys:
        return None

    keys_list = sorted(all_keys)
    print(f"  [references] Found {len(keys_list)} unique keys in text")

    domain = analysis.get("domain", "scientific research")
    all_entries = []

    # Process in batches of 5 keys for better LLM compliance
    batch_size = 5
    for batch_start in range(0, len(keys_list), batch_size):
        batch = keys_list[batch_start:batch_start + batch_size]
        print(f"  [references] Batch {batch_start//batch_size + 1}: {batch}")

        prompt = f"""Generate bibtex entries for EXACTLY these citation keys: {', '.join(batch)}

IMPORTANT: Use the keys EXACTLY as given above. DO NOT change them, DO NOT use placeholder2024.

CRITICAL: The first line of each entry MUST be:
@article{{{batch[0]},
  author = ...

For each key, create an @article entry with author, title, journal, year (2015-2025).
Use REAL paper titles and REAL author names from the {domain} literature.
Output ONLY bibtex. No markdown, no explanations, no code fences."""

        system_msg = f"Use these EXACT keys: {', '.join(batch)}. Each key MUST appear verbatim as the citation key."
        resp, _ = get_response_from_llm(prompt, client=client, model=model, system_message=system_msg, temperature=0.5)
        resp = resp.strip()
        if resp.startswith("```"):
            resp = re.sub(r"^```\w*\n?", "", resp)
            resp = re.sub(r"\n?```$", "", resp)
        all_entries.append(resp)

    combined = "\n\n".join(all_entries)
    # Verify keys are preserved
    for k in keys_list:
        if k not in combined:
            print(f"  [references] WARNING: Key '{k}' missing from generated bibtex!")
    return combined


def _collect_real_images(output_dir):
    """Return list of actual .png image files in output_dir (parent of latex/)."""
    parent_dir = os.path.dirname(os.path.dirname(output_dir.rstrip("/\\"))) if output_dir.rstrip("/\\").endswith("latex") else output_dir
    images = []
    for d in [output_dir, parent_dir]:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(".png") and os.path.isfile(os.path.join(d, f)):
                images.append({"name": f, "base": os.path.splitext(f)[0], "path": os.path.join(d, f)})
    # Deduplicate by name
    seen = set()
    unique = []
    for img in images:
        if img["name"] not in seen:
            seen.add(img["name"])
            unique.append(img)
    return unique


def _fix_graphicspath(template_text):
    """Fix \graphicspath for Overleaf compatibility. Handles {{../}}, {{}}, {{}}} etc."""
    # Match \graphicspath{...} with any number of inner braces
    return re.sub(r"\\graphicspath\{+\s*(?:[^}]*?)\s*\}+", r"\\graphicspath{{}}", template_text)


def _fix_figure_references(template_text, output_dir):
    """
    Replace LLM-invented figure filenames (e.g., fig1, fig2) with actual
    image files found on disk. Keeps LLM-generated captions intact.
    """
    real_images = _collect_real_images(output_dir)
    if not real_images:
        print("  [figures] No real images found — skipping figure fix")
        return template_text

    print(f"  [figures] Real images on disk: {[i['name'] for i in real_images]}")

    # Find all \includegraphics commands
    img_cmd_re = re.compile(r'(\\includegraphics(?:\[[^\]]*\])?)\{([^}]+)\}')

    def ref_file_exists(ref):
        """Check if ref (with/without extension) exists in output_dir or parent."""
        parent_dir = os.path.dirname(output_dir.rstrip("/\\"))
        for check_dir in [output_dir, parent_dir]:
            if os.path.isfile(os.path.join(check_dir, ref)):
                return True
            if "." not in ref:
                for ext in [".png", ".jpg", ".jpeg", ".pdf"]:
                    if os.path.isfile(os.path.join(check_dir, ref + ext)):
                        return True
        return False

    # Find references that don't exist on disk
    all_refs = [(m.group(2), m.start()) for m in img_cmd_re.finditer(template_text)]
    missing = [(ref, pos) for ref, pos in all_refs if not ref_file_exists(ref)]

    # Also fix method_overview.pdf → method_figure.png if method_figure.png exists
    overview_refs = [(ref, pos) for ref, pos in all_refs if ref in ("method_overview.pdf", "method_overview")]
    for ref, pos in overview_refs:
        if not missing or (ref, pos) not in missing:
            if "method_overview" in ref and ref_file_exists("method_figure.png"):
                old_cmd = re.search(r'(\\includegraphics(?:\[[^\]]*\])?)\{' + re.escape(ref) + r'\}',
                                    template_text)
                if old_cmd:
                    new_cmd = old_cmd.group(1) + "{method_figure.png}"
                    template_text = template_text.replace(old_cmd.group(0), new_cmd)
                    print(f"  [figures] Normalized '{ref}' → 'method_figure.png'")

    if not missing:
        print("  [figures] All figure references point to existing files — OK")
        return template_text

    print(f"  [figures] Missing references: {[r for r, _ in missing]}")

    # Classify real images: method_figure vs results_plot vs other
    method_imgs = [i for i in real_images if "method" in i["name"].lower()]
    result_imgs = [i for i in real_images if "result" in i["name"].lower() or "plot" in i["name"].lower()]
    other_imgs  = [i for i in real_images if i not in method_imgs and i not in result_imgs]

    # Find section boundaries for context-aware assignment
    method_sec_pos = template_text.find(r"\section{Method}")
    results_sec_pos = template_text.find(r"\section{Results}")
    conclusion_sec_pos = template_text.find(r"\section{Conclusions")

    # Assign real images to missing refs by section context
    used_method = []
    used_result = []
    used_other = []

    for ref, pos in missing:
        # Determine which section this ref is in
        after_method = method_sec_pos > 0 and pos > method_sec_pos
        after_results = results_sec_pos > 0 and pos > results_sec_pos
        after_conclusion = conclusion_sec_pos > 0 and pos > conclusion_sec_pos

        if after_results and not after_conclusion:
            # In Results section → use results images
            pool = [i for i in result_imgs if i["base"] not in used_result]
            if not pool:
                pool = [i for i in method_imgs if i["base"] not in used_method]
            if not pool:
                pool = [i for i in real_images if i["base"] not in used_other]
        elif after_method and not after_results:
            # In Method section → use method images
            pool = [i for i in method_imgs if i["base"] not in used_method]
            if not pool:
                pool = [i for i in real_images if i["base"] not in used_other]
        else:
            # Introduction / Related Work / Background / Conclusion
            pool = [i for i in other_imgs if i["base"] not in used_other]
            if not pool:
                pool = [i for i in real_images if i["base"] not in used_other]

        if pool:
            chosen = pool[0]
            old_cmd = re.search(
                r'(\\includegraphics(?:\[[^\]]*\])?)\{' + re.escape(ref) + r'\}',
                template_text
            )
            if old_cmd:
                new_cmd = old_cmd.group(1) + "{" + chosen["base"] + "}"
                template_text = template_text.replace(old_cmd.group(0), new_cmd)
                print(f"  [figures] Replaced '{ref}' → '{chosen['base']}'")
                if chosen in method_imgs:
                    used_method.append(chosen["base"])
                elif chosen in result_imgs:
                    used_result.append(chosen["base"])
                else:
                    used_other.append(chosen["base"])

    return template_text


def _inject_method_figure(template_text, output_dir):
    """If method_figure.png exists but is not referenced, inject a figure block before \section{Method}."""
    parent_dir = os.path.dirname(output_dir.rstrip("/\\"))
    method_png = None
    for d in [output_dir, parent_dir]:
        for fname in ["method_figure.png", "method_figure.jpeg", "method_figure.jpg"]:
            fp = os.path.join(d, fname)
            if os.path.isfile(fp):
                method_png = fname.replace(".jpeg", ".png").replace(".jpg", ".png")
                break
        if method_png:
            break

    if not method_png:
        return template_text

    # Check if method_figure is already referenced
    if method_png.rsplit(".", 1)[0] in template_text:
        return template_text
    if "method_figure" in template_text:
        return template_text

    method_sec_idx = template_text.find(r"\section{Method}")
    if method_sec_idx < 0:
        return template_text

    # Find the newline just before \section{Method}
    insert_idx = template_text.rfind("\n", 0, method_sec_idx)
    if insert_idx < 0:
        insert_idx = method_sec_idx

    figure_block = rf"""
\begin{{figure}}[htbp]
    \centering
    \includegraphics[width=0.95\textwidth]{{{method_png.rsplit('.', 1)[0]}}}
    \caption{{Overview of the proposed methodology. The framework
    addresses the key trade-off through a systematic approach.}}
    \label{{fig:method_overview}}
\end{{figure}}
"""
    template_text = template_text[:insert_idx] + figure_block + template_text[insert_idx:]
    print(f"  [figures] Injected method figure block referencing '{method_png}'")
    return template_text


def write_paper_direct(idea, analysis, template_path, output_dir, client, model):
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    # --- Fix 1: \graphicspath → fix for Overleaf flat-file structure ---
    template = _fix_graphicspath(template)

    # --- Compute available figures for prompt injection ---
    fig_info = _get_available_figures_info(output_dir)
    fig_kwargs = {"available_figures": fig_info} if fig_info else {}

    # --- Remove ALL template figure blocks — LLM will write its own ---
    fig_re = re.compile(r"\\begin\{figure\}.*?\\end\{figure\}", re.DOTALL)
    template_no_figs = fig_re.sub(
        lambda m: f"%%TEMPLATE_FIG_REMOVED_{m.start()}%%", template
    )

    # Sec 1a: Title — replace just the TITLE HERE placeholder, preserve LaTeX structure
    title_sec = generate_section("title_abstract_title", idea, analysis, client, model)
    if title_sec:
        title_clean = title_sec.strip().strip('"').strip("'")
        template_no_figs = template_no_figs.replace(r"\title{TITLE HERE}", rf"\title{{{title_clean}}}")
        print(f"  [title] Injected: {title_clean[:80]}...")

    # Sec 1b: Abstract — replace just ABSTRACT HERE placeholder
    abstract_sec = generate_section("title_abstract_abstract", idea, analysis, client, model)
    if abstract_sec:
        abstract_clean = abstract_sec.strip()
        template_no_figs = template_no_figs.replace("ABSTRACT HERE", abstract_clean)
        print(f"  [abstract] Injected ({len(abstract_clean)} chars)")

    # Sec 2-8: Body — methods/results get figure info injected into prompt
    SECTIONS_WITH_FIGURES = {"method", "results"}
    sections = [
        ("introduction", "Introduction"),
        ("related_work", "Related Work"),
        ("background", "Background"),
        ("method", "Method"),
        ("experimental_setup", "Experimental Setup"),
        ("results", "Results"),
        ("conclusion", "Conclusions and Future Work"),
    ]
    for name, heading in sections:
        extra = fig_kwargs if name in SECTIONS_WITH_FIGURES else None
        sec = generate_section(name, idea, analysis, client, model, extra_kwargs=extra)
        if not sec:
            continue
        # Match from \section{Heading} to next \section or \bibliographystyle or \end{document}
        pattern = rf"(\\section\{{{heading}\}}.*?)(?=\\section\{{|\\bibliographystyle|\\end\{{document\}})"
        try:
            template_no_figs = re.sub(pattern, lambda m: sec.strip() + "\n\n", template_no_figs, count=1, flags=re.DOTALL)
            print(f"  [{name}] Injected ({len(sec)} chars)")
        except Exception as e:
            print(f"  [{name}] Regex error: {e} — appending")
            hdr = f"\\section{{{heading}}}"
            idx = template_no_figs.find(hdr)
            if idx >= 0:
                end = template_no_figs.find("\\section{", idx + len(hdr))
                if end < 0:
                    end = template_no_figs.find("\\bibliographystyle", idx)
                if end < 0:
                    end = template_no_figs.find("\\end{document}", idx)
                if end > idx:
                    template_no_figs = template_no_figs[:idx] + sec.strip() + "\n\n" + template_no_figs[end:]

    # --- Fix: Replace [t] with [htbp] in figure environments to prevent LaTeX float pile-up ---
    template_no_figs = re.sub(r'(\\begin\{figure\})\[t\]', r'\1[htbp]', template_no_figs)

    # --- Clean up template figure placeholders (remove them, don't re-insert) ---
    template = re.sub(r"%%TEMPLATE_FIG_REMOVED_\d+%%", "", template_no_figs)
    # Also clean up any leftover blank lines from removal
    template = re.sub(r"\n{3,}", "\n\n", template)
    print(f"  [figures] Cleaned up template figure placeholders")

    # --- Fix 2: Replace LLM-invented figure filenames with actual disk files ---
    template = _fix_figure_references(template, output_dir)

    # --- Fix 3: Deduplicate figure blocks (keep only first occurrence of each image) ---
    template = _dedup_figures(template)

    # --- Fix 4: Inject method figure if missing ---
    template = _inject_method_figure(template, output_dir)

    # Sec 9: References
    refs = generate_references(template, idea, analysis, client, model)
    if refs:
        old_bib = re.search(r"(\\begin\{filecontents\}\{references\.bib\}).*?(\\end\{filecontents\})", template, re.DOTALL)
        if old_bib:
            template = template[:old_bib.start()] + old_bib.group(1) + "\n" + refs.strip() + "\n" + old_bib.group(2) + template[old_bib.end():]
            print(f"  [references] Injected {len(refs)} chars")

    output_path = os.path.join(output_dir, "latex", "template.tex")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(template)

    paper_path = os.path.join(output_dir, f"{idea.get('Name', 'paper')}.tex")
    with open(paper_path, "w", encoding="utf-8") as f:
        f.write(template)

    print(f"\nPaper written: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()
    if args.api_key:
        os.environ["DEEPSEEK_API_KEY"] = args.api_key
    with open(os.path.join(args.output_dir, "idea.json")) as f:
        idea = json.load(f)
    with open(os.path.join(args.output_dir, "analysis.json")) as f:
        analysis = json.load(f)
    client, cm = create_client(args.model)
    print(f"Title: {idea.get('Title', 'N/A')[:100]}")
    write_paper_direct(idea, analysis, os.path.join(args.output_dir, "latex", "template.tex"), args.output_dir, client, cm)
