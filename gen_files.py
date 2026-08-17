#!/usr/bin/env python3
"""Generator script for AI-Scientist project files."""
import os

OUTDIR = r"C:/Users/liuru/AI-Scientist"
os.makedirs(OUTDIR, exist_ok=True)

def write_file(relpath, lines):
    path = os.path.join(OUTDIR, relpath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {len(lines)} lines to {relpath}")
    return path

# Build files below

