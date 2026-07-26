#!/usr/bin/env python3
"""修复 LaTeX 文字溢出问题：preamble + seqsplit + 表格 + 公式"""

import glob, os, re

best = max(glob.glob('/data/lrd/AI-Scientist/results/nanoGPT/20260720_0*/'),
           key=lambda d: len([f for f in os.listdir(d) if f.endswith('.png')]))
tex_path = os.path.join(best, 'PAPER_FINAL.tex')
with open(tex_path) as fp: tex = fp.read()

# ===== 1. 替换 preamble =====
old_preamble = tex[:tex.find(r'\begin{document}')]

new_preamble = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=1in,right=1.2in,left=1.1in,bottom=1.1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{times}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{caption}
\usepackage{float}
\usepackage{tabularx}
\usepackage{seqsplit}
\usepackage{microtype}
\usepackage{multirow}

\allowdisplaybreaks
\setlength{\emergencystretch}{2em}
\sloppy
\raggedbottom
\graphicspath{{./}}
"""

# 保留 title, author, date
title_match = re.search(r'\\title\{([^}]*)\}', old_preamble)
title = title_match.group(1) if title_match else "Research Paper"

new_preamble += '\n\\title{' + title + '}\n'
new_preamble += r'\author{AI Scientist \\ Sakana AI}' + '\n'
new_preamble += r'\date{\today}' + '\n'

# 找 body（从 \begin{document} 到 \end{document}）
body_start = tex.find(r'\begin{document}')
body_end = tex.find(r'\end{document}')
body = tex[body_start:body_end+16]

# ===== 2. 长连续字符串（40+ 字符无空格）包 seqsplit =====
def wrap_long(text):
    """找到连续 40+ 字符无空格的片段，用 seqsplit 包裹"""
    # 保护已有 LaTeX 命令
    cmds = re.findall(r'\\(?:[a-zA-Z]+|.)(?:\{[^}]*\})*', text)
    for i, c in enumerate(cmds):
        text = text.replace(c, '<<<CMD' + str(i) + '>>>')

    def replacer(m):
        s = m.group(0)
        if len(s) >= 40:
            return r'\seqsplit{' + s + '}'
        return s

    text = re.sub(r'[^\s\\{}]{40,}', replacer, text)

    for i, c in enumerate(cmds):
        text = text.replace('<<<CMD' + str(i) + '>>>', c)

    return text

body = wrap_long(body)

# ===== 3. 转换 tabular 为 tabularx =====
def tabular_to_tabularx(match):
    content = match.group(1)
    spec_match = re.search(r'\\begin\{tabular\}\{([^}]*)\}', content)
    if not spec_match:
        return match.group(0)
    spec = spec_match.group(1)
    ncols = len(spec.replace('|','').replace(' ',''))
    # 构建 tabularx 列规格
    new_spec = '|' + 'X|' * ncols
    new_content = content.replace(
        '\\begin{tabular}{' + spec + '}',
        '\\begin{tabularx}{\\textwidth}{' + new_spec + '}'
    )
    new_content = new_content.replace('\\end{tabular}', '\\end{tabularx}')
    return '\\begin{table}[htbp]\n\\centering\n\\small\n' + new_content + '\n\\end{table}'

body = re.sub(r'\\begin\{table\}.*?\\end\{table\}', tabular_to_tabularx, body, flags=re.DOTALL)

# ===== 4. 长行内公式转 display math =====
def inline_to_display(match):
    content = match.group(1)
    if len(content) > 50:
        return r'\begin{equation}' + content + r'\end{equation}'
    return '$' + content + '$'

body = re.sub(r'\$([^$]{50,})\$', inline_to_display, body)

# ===== 5. 组装 =====
tex = new_preamble + '\n' + body

with open(tex_path, 'w', encoding='utf-8') as fp: fp.write(tex)
print('SAVED:', tex_path)
print('Preamble replaced, seqsplit added, tabularx converted, sloppy enabled.')
print('Re-download PAPER_FINAL.tex + PNGs to Overleaf.')
