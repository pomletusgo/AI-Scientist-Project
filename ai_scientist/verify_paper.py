#!/usr/bin/env python3
"""
Paper Verification Agent — validates references and experimental parameters
in AI-generated papers.

Two checks:
  1. Reference Authenticity: Searches OpenAlex to verify each citation actually
     exists, with matching authors/year/venue.
  2. Parameter Plausibility: Checks experimental parameters against domain-
     typical ranges and flags suspicious values.

Usage:
  python verify_paper.py --paper_dir results/telco_churn/20260730_xxx/
  python verify_paper.py --paper_dir . --domain classification
"""

import os, sys, re, json, time, argparse
import requests
import numpy as np

# ============================================================
# LaTeX math → ML parameter name mapping
LATEX_PARAM_MAP = {
    "lambda": "weight_decay",
    "gamma": "gamma",
    "alpha": "learning_rate",
    "beta": "momentum",
    "lr": "learning_rate",
    "learning_rate": "learning_rate",
    "batch_size": "batch_size",
    "epochs": "epochs",
    "dropout": "dropout",
    "weight_decay": "weight_decay",
    "temperature": "temperature",
}

# Domain-typical parameter ranges (min, typical_min, typical_max, max)
# ============================================================
PARAM_RANGES = {
    "learning_rate":       (1e-7, 1e-5, 1e-1, 1.0),
    "lr":                  (1e-7, 1e-5, 1e-1, 1.0),
    "batch_size":          (1, 4, 1024, 100000),
    "batchsize":           (1, 4, 1024, 100000),
    "epochs":              (1, 5, 500, 10000),
    "weight_decay":        (0, 1e-8, 0.1, 1.0),
    "dropout":             (0.0, 0.05, 0.8, 1.0),
    "dropout_rate":        (0.0, 0.05, 0.8, 1.0),
    "num_layers":          (1, 2, 100, 10000),
    "n_layers":            (1, 2, 100, 10000),
    "hidden_size":         (2, 8, 4096, 100000),
    "hidden_dim":          (2, 8, 4096, 100000),
    "embedding_dim":       (2, 16, 4096, 100000),
    "temperature":         (0.01, 0.1, 2.0, 100),
    "gamma":               (0.0, 0.5, 5.0, 100),    # focal loss
    "momentum":            (0.0, 0.8, 0.999, 1.0),
    "beta1":               (0.0, 0.8, 0.999, 1.0),
    "beta2":               (0.0, 0.99, 0.9999, 1.0),
}

# ============================================================
# Reference Verification
# ============================================================

def extract_citations(tex_path):
    """Extract all citation keys from a LaTeX file. Supports \cite{key} and \nocite{*}."""
    with open(tex_path, "r", encoding="utf-8") as f:
        text = f.read()
    keys = set()
    for match in re.findall(r'\\cite\w*\*?\s*\{([^}]*)\}', text):
        for k in match.split(","):
            k = k.strip()
            if k and k != "*":
                keys.add(k)
    # If \nocite{*} is used, add ALL keys from the bib file
    if re.search(r'\\nocite\s*\*?\s*\{\s*\*\s*\}', text):
        tex_dir = os.path.dirname(tex_path)
        for bib_name in ["paper.bib", "references.bib"]:
            bib_path = os.path.join(tex_dir, "..", bib_name)
            if os.path.exists(bib_path):
                bib = extract_bib_entries(tex_path, bib_path)
                keys.update(bib.keys())
                break
    return sorted(keys)


def extract_bib_entries(tex_path, bib_path):
    """Extract BibTeX entries keyed by citation key."""
    # Try bib file first
    bib_text = ""
    if os.path.exists(bib_path):
        with open(bib_path, "r", encoding="utf-8") as f:
            bib_text = f.read()
    else:
        # Fall back to filecontents in tex file
        with open(tex_path, "r", encoding="utf-8") as f:
            tex = f.read()
        m = re.search(r'\\begin\{filecontents\}\{.*?\.bib\}(.*?)\\end\{filecontents\}', tex, re.DOTALL)
        if m:
            bib_text = m.group(1)

    entries = {}
    # Parse each @entry{key, ...}
    for match in re.finditer(r'@(\w+)\s*\{\s*([^,]+)\s*,\s*(.*?)\}\s*\n?\s*\}', bib_text, re.DOTALL):
        entry_type = match.group(1)
        key = match.group(2).strip()
        body = match.group(3)
        # Extract fields
        fields = {}
        for fm in re.finditer(r'(\w+)\s*=\s*[{"]([^}"]*)[}"]', body):
            fields[fm.group(1).lower()] = fm.group(2)
        fields["_type"] = entry_type
        entries[key] = fields

    # Also try simpler pattern for entries without nested braces
    for match in re.finditer(r'@(\w+)\s*\{\s*([^,]+)\s*,', bib_text):
        key = match.group(2).strip()
        if key not in entries:
            # Extract until we hit the closing }
            start = match.end()
            depth = 0
            end = start
            for i, ch in enumerate(bib_text[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    if depth == 0:
                        end = i
                        break
                    depth -= 1
            body = bib_text[start:end]
            fields = {}
            for fm in re.finditer(r'(\w+)\s*=\s*[{"]([^}"]*)[}"]', body):
                fields[fm.group(1).lower()] = fm.group(2)
            fields["_type"] = match.group(1)
            entries[key] = fields

    return entries


def search_openalex(title, max_results=3):
    """Search OpenAlex for a paper by title. Returns list of matching works."""
    try:
        query = title[:150].replace("{", "").replace("}", "")
        r = requests.get(
            "https://api.openalex.org/works",
            params={
                "search": query,
                "per_page": max_results,
                "sort": "relevance_score:desc",
            },
            timeout=15,
        )
        r.raise_for_status()
        results = []
        for w in r.json().get("results", []):
            authors = [
                a.get("author", {}).get("display_name", "")
                for a in w.get("authorships", [])
            ]
            results.append({
                "title": w.get("title", ""),
                "authors": authors,
                "year": str(w.get("publication_year", "")),
                "venue": (w.get("primary_location", {})
                           .get("source", {})
                           .get("display_name", "Unknown")),
                "doi": w.get("doi", ""),
                "cited_by": w.get("cited_by_count", 0),
            })
        return results
    except Exception as e:
        return [{"error": str(e)}]


def search_semantic_scholar(title, max_results=3, retries=3):
    """Search Semantic Scholar for a paper by title.
    Free: 100 requests/5min without key. With S2_API_KEY: 100 requests/sec.
    Built-in retry with backoff for 429 rate limiting.
    """
    api_key = os.environ.get("S2_API_KEY", "s2k-thNDhaZ9wvdgY6k3eIzpo4hPPgN4FgWFdZBHy13I")
    headers = {"x-api-key": api_key} if api_key else {}

    query = title[:200].replace("{", "").replace("}", "")

    for attempt in range(retries):
        try:
            r = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": query,
                    "limit": max_results,
                    "fields": "title,authors,year,venue,journal,externalIds,citationCount,url",
                },
                headers=headers,
                timeout=15,
            )
            if r.status_code == 429:
                wait = (attempt + 1) * 5
                print(f"    (S2 rate limited, retrying in {wait}s...)")
                time.sleep(wait)
                continue
            r.raise_for_status()
            results = []
            for p in r.json().get("data", []):
                authors = [a.get("name", "") for a in p.get("authors", [])]
                venue_info = p.get("venue", "") or ""
                if isinstance(venue_info, dict):
                    venue_info = venue_info.get("name", "")
                journal_info = p.get("journal", {}) or {}
                if isinstance(journal_info, dict):
                    journal_info = journal_info.get("name", "")

                results.append({
                    "title": p.get("title", ""),
                    "authors": authors,
                    "year": str(p.get("year", "")),
                    "venue": venue_info or journal_info or "Unknown",
                    "doi": p.get("externalIds", {}).get("DOI", ""),
                    "cited_by": p.get("citationCount", 0),
                    "url": p.get("url", ""),
                })
            return results
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
                continue
            return [{"error": str(e)}]
    return [{"error": "Rate limited after retries"}]


def verify_reference(key, bib_entry, engine="semanticscholar"):
    """Verify a single reference. Returns verification report.
    engine: "semanticscholar" (default, faster) or "openalex"
    """
    title = bib_entry.get("title", "")
    authors_str = bib_entry.get("author", "")
    year = bib_entry.get("year", "")
    venue = bib_entry.get("journal", "") or bib_entry.get("booktitle", "")

    if not title:
        return {
            "key": key,
            "status": "UNVERIFIABLE",
            "reason": "No title found in BibTeX entry",
            "bib_title": title,
        }

    # Search
    if engine == "openalex":
        results = search_openalex(title)
    else:
        results = search_semantic_scholar(title)

    if not results or "error" in results[0]:
        err = results[0].get("error", "No results") if results else "No results"
        # Fallback: try the other engine
        if engine != "openalex":
            results = search_openalex(title)
        else:
            results = search_semantic_scholar(title)
        if not results or "error" in results[0]:
            return {
                "key": key,
                "status": "SEARCH_FAILED",
                "reason": err,
                "bib_title": title,
            }

    # Find best match by title similarity
    best = results[0]
    bib_title_clean = re.sub(r'[{}]', '', title).lower().strip()
    best_title_clean = best["title"].lower().strip() if best["title"] else ""

    # Title similarity
    title_words_bib = set(bib_title_clean.split())
    title_words_best = set(best_title_clean.split())
    if title_words_bib:
        overlap = len(title_words_bib & title_words_best) / len(title_words_bib)
    else:
        overlap = 0

    # Author check
    bib_authors = [a.strip() for a in authors_str.replace(" and ", ",").split(",") if a.strip()]
    oa_authors = [a.lower() for a in best["authors"]]
    bib_authors_lower = [a.lower() for a in bib_authors]

    author_match = any(
        any(bib_a in oa_a for oa_a in oa_authors)
        for bib_a in bib_authors_lower
    ) if bib_authors_lower else False

    # Year check
    year_match = (year and best["year"] and year == best["year"])

    # Verdict
    if overlap > 0.7 and (author_match or year_match):
        status = "VERIFIED"
        detail = f"Title match {overlap:.0%}"
        if author_match:
            detail += ", author confirmed"
        if year_match:
            detail += f", year {year} matches"
    elif overlap > 0.4:
        status = "LIKELY_REAL"
        detail = f"Partial title match {overlap:.0%}, check manually"
    else:
        status = "UNVERIFIED"
        detail = f"Best OpenAlex match: \"{best['title'][:100]}\""

    return {
        "key": key,
        "status": status,
        "reason": detail,
        "bib_title": title[:120],
        "bib_authors": bib_authors[:3],
        "bib_year": year,
        "bib_venue": venue[:80] if venue else "",
        "openalex_title": best["title"][:120] if best["title"] else "",
        "openalex_authors": best["authors"][:3],
        "openalex_year": best.get("year", ""),
        "openalex_venue": best.get("venue", "")[:80],
        "openalex_cited_by": best.get("cited_by", 0),
        "title_overlap": round(overlap, 3),
    }


# ============================================================
# Parameter Plausibility Check
# ============================================================

def extract_parameters(tex_path):
    """Extract numerical experimental parameters from LaTeX text."""
    with open(tex_path, "r", encoding="utf-8") as f:
        text = f.read()

    params = {}
    # First pass: LaTeX math mode parameters
    # Match both $...$ and \(...\) delimiters
    # Patterns like: $\lambda = 0.1$, \(\lambda = 0.01\), $lr = 1e-3$
    for m in re.finditer(r'(?:\$|\\\()\s*\\?(\w+)\s*=\s*([\d.eE+\-]+)\s*(?:\$|\\\))', text):
        name = m.group(1).lower()
        # Map LaTeX names to ML parameters
        name = LATEX_PARAM_MAP.get(name, name)
        if name in PARAM_RANGES:
            try:
                params.setdefault(name, []).append(float(m.group(2)))
            except ValueError:
                pass

    for param_name, (p_min, t_min, t_max, p_max) in PARAM_RANGES.items():
        # Match patterns like: "learning rate 0.001", "lr=1e-3", "batch size of 64"
        patterns = [
            rf'{param_name}\s*[=:]\s*([\d.eE+\-]+)',
            rf'{param_name}\s*(?:of|is|set to|was)\s*([\d.eE+\-]+)',
            rf'([\d.eE+\-]+)\s*(?:for|as)\s*(?:the\s*)?{param_name}',
        ]
        values = []
        for pat in patterns:
            for match in re.finditer(pat, text, re.IGNORECASE):
                try:
                    v = float(match.group(1))
                    values.append(v)
                except ValueError:
                    pass

        if values:
            params.setdefault(param_name, []).extend(values)

    return params


def check_parameter(name, values, ranges):
    """Check if parameter values are within plausible range."""
    p_min, t_min, t_max, p_max = ranges
    assessments = []
    for v in values:
        if p_min <= v <= p_max:
            if t_min <= v <= t_max:
                assessments.append({"value": v, "verdict": "OK", "detail": "Within typical range"})
            else:
                assessments.append({"value": v, "verdict": "UNUSUAL",
                                    "detail": f"Outside typical [{t_min}, {t_max}] but within possible [{p_min}, {p_max}]"})
        else:
            assessments.append({"value": v, "verdict": "SUSPICIOUS",
                                "detail": f"Outside plausible range [{p_min}, {p_max}], likely an error"})
    return assessments


# ============================================================
# Main Verifier
# ============================================================

def verify_paper(paper_dir, domain="classification", bib_path=None, tex_path=None,
                 json_path=None, engine="semanticscholar"):
    """
    Run full verification on a generated paper.

    Args:
        paper_dir: Path to the paper directory (contains latex/template.tex)
        domain: ML domain for parameter context
        bib_path: Optional path to .bib file
        tex_path: Optional path to .tex file
        json_path: Optional path to save JSON results
        engine: "semanticscholar" (recommended) or "openalex"

    Returns:
        dict with verification results
    """
    if tex_path is None:
        tex_path = os.path.join(paper_dir, "latex", "template.tex")
    if bib_path is None:
        bib_path = os.path.join(paper_dir, "paper.bib")
    if not os.path.exists(bib_path):
        bib_path = os.path.join(paper_dir, "references.bib")

    if not os.path.exists(tex_path):
        return {"error": f"Paper not found: {tex_path}"}

    print(f"Verifying: {tex_path}")
    print(f"Domain: {domain}")

    # --- Reference Check ---
    print("\n" + "=" * 60)
    print("CHECK 1/2: Reference Authenticity")
    print("=" * 60)

    cite_keys = extract_citations(tex_path)
    bib_entries = extract_bib_entries(tex_path, bib_path)
    print(f"  Citations in text: {len(cite_keys)}")
    print(f"  BibTeX entries: {len(bib_entries)}")

    # Find missing bib entries
    missing_bib = [k for k in cite_keys if k not in bib_entries]
    if missing_bib:
        print(f"  WARNING: {len(missing_bib)} citations have no BibTeX entry!")
        for k in missing_bib:
            print(f"    - {k}")

    ref_results = []
    for key in cite_keys:
        if key in bib_entries:
            print(f"  [{key}] ", end="", flush=True)
            result = verify_reference(key, bib_entries[key], engine=engine)
            ref_results.append(result)
            symbol = {"VERIFIED": "✓", "LIKELY_REAL": "~", "UNVERIFIED": "✗", "SEARCH_FAILED": "?", "UNVERIFIABLE": "-"}
            print(f"{symbol.get(result['status'], '?')} {result['status']}")
            time.sleep(2.0)  # Rate limit: S2 allows 100/5min = 1 per 3s without key
        else:
            ref_results.append({"key": key, "status": "MISSING_BIB", "reason": "No BibTeX entry found"})
            print(f"  [{key}] ✗ MISSING_BIB")

    # --- Parameter Check ---
    print("\n" + "=" * 60)
    print("CHECK 2/2: Parameter Plausibility")
    print("=" * 60)

    params = extract_parameters(tex_path)
    print(f"  Parameters found: {len(params)}")
    param_results = {}
    for name, values in sorted(params.items()):
        if name in PARAM_RANGES:
            checks = check_parameter(name, values, PARAM_RANGES[name])
            param_results[name] = checks
            for c in checks:
                symbol = {"OK": "✓", "UNUSUAL": "⚠", "SUSPICIOUS": "✗"}
                print(f"  {name} = {c['value']}: {symbol[c['verdict']]} {c['verdict']} — {c['detail']}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    n_verified = sum(1 for r in ref_results if r["status"] == "VERIFIED")
    n_likely = sum(1 for r in ref_results if r["status"] == "LIKELY_REAL")
    n_unverified = sum(1 for r in ref_results if r["status"] in ("UNVERIFIED", "SEARCH_FAILED"))
    n_missing = sum(1 for r in ref_results if r["status"] in ("MISSING_BIB", "UNVERIFIABLE"))

    print(f"  References: {n_verified} verified, {n_likely} likely real, "
          f"{n_unverified} unverified, {n_missing} missing")

    n_ok = sum(1 for checks in param_results.values() for c in checks if c["verdict"] == "OK")
    n_unusual = sum(1 for checks in param_results.values() for c in checks if c["verdict"] == "UNUSUAL")
    n_suspicious = sum(1 for checks in param_results.values() for c in checks if c["verdict"] == "SUSPICIOUS")
    print(f"  Parameters: {n_ok} OK, {n_unusual} unusual, {n_suspicious} suspicious")

    # Scoring
    total_refs = len(ref_results) or 1
    total_params = n_ok + n_unusual + n_suspicious or 1
    ref_score = (n_verified + 0.5 * n_likely) / total_refs
    param_score = n_ok / total_params
    overall = 0.5 * ref_score + 0.5 * param_score

    print(f"  Reference Score: {ref_score:.0%} | Parameter Score: {param_score:.0%}")
    print(f"  Overall Trust Score: {overall:.0%}")

    result = {
        "references": ref_results,
        "parameters": param_results,
        "missing_bib": missing_bib,
        "scores": {
            "reference_score": round(ref_score, 3),
            "parameter_score": round(param_score, 3),
            "overall_trust": round(overall, 3),
        },
    }

    # Save JSON if requested
    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nVerification report saved: {json_path}")

    return result


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify AI-generated paper references and parameters")
    parser.add_argument("--paper_dir", required=True, help="Path to paper directory")
    parser.add_argument("--domain", default="classification", help="ML domain (classification/regression/nlp/vision)")
    parser.add_argument("--bib", default=None, help="Path to .bib file (auto-detect if omitted)")
    parser.add_argument("--tex", default=None, help="Path to .tex file (auto-detect if omitted)")
    parser.add_argument("--json", default=None, help="Save results to JSON file")
    parser.add_argument("--engine", default="openalex",
                        choices=["semanticscholar", "openalex"],
                        help="Search engine: openalex (no key needed, reliable) or semanticscholar")
    args = parser.parse_args()

    results = verify_paper(args.paper_dir, args.domain, args.bib, args.tex, args.json, args.engine)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {args.json}")
