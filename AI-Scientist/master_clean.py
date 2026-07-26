#!/usr/bin/env python3
"""一次性强力清理 ALL LaTeX 格式问题"""
import glob, re, os

for f in sorted(glob.glob('/data/lrd/AI-Scientist/results/nanoGPT/*/latex/paper_rich.tex')):
    print(f'Cleaning: {os.path.basename(os.path.dirname(os.path.dirname(f)))}')
    with open(f) as fp: t = fp.read()
    before = len(t)

    # ═══════ 1. 删所有问号 ═══════
    t = re.sub(r'\?\?+', '', t)          # ?? 或更多
    t = re.sub(r'\[\?\]', '', t)         # [?]
    t = re.sub(r'\(\?\)', '', t)         # (?)
    t = re.sub(r' \? ', ' ', t)          # 单独 ?
    t = re.sub(r' \?\.', '.', t)         # ? 后接句号
    t = re.sub(r' \?,', ',', t)          # ? 后接逗号
    t = re.sub(r' \?\)', ')', t)         # ? 后接括号
    t = re.sub(r'\n\?', '\n', t)         # 行首 ?

    # ═══════ 2. 修 # ═══════
    t = t.replace('#', r'\#')

    # ═══════ 3. 删不可见字符 ═══════
    for ch in [' ',' ','​','‎','‏','﻿',' ',' ','‌','‍','']:
        t = t.replace(ch, ' ')

    # ═══════ 4. 修 LLM 编造的 LaTeX 命令 ═══════
    for bad, good in {
        '\\eps':'\\epsilon','\\lamdba':'\\lambda','\\sigmoid':'\\sigma',
        '\\relu':'\\text{ReLU}','\\gelu':'\\text{GELU}','\\softmax':'\\text{softmax}',
        '\\topk':'\\text{top-}k','\\layernorm':'\\text{LayerNorm}',
        '\\batchnorm':'\\text{BatchNorm}','\\argmax':'\\arg\\max',
        '\\argmin':'\\arg\\min','\\norm':'\\lVert','\\diag':'\\text{diag}',
        '\\concat':'\\parallel','\\encoder':'\\text{encoder}',
        '\\decoder':'\\text{decoder}','\\onehot':'\\text{one-hot}',
    }.items():
        if bad in t: t = t.replace(bad, good)

    # ═══════ 5. 修引用 ═══════
    t = t.replace(r'\cite{}', r'\cite{ref1}')
    t = re.sub(r'\\ref\{[^}]*\?[^}]*\}', '', t)
    t = re.sub(r'\\ref\{\}', '', t)

    # ═══════ 6. 修 $$ → equation ═══════
    t = re.sub(r'\$\$\s*(.*?)\s*\$\$', r'\\begin{equation}\n\1\n\\end{equation}', t, flags=re.DOTALL)

    # ═══════ 7. 修环境 ═══════
    for env in ['table', 'figure', 'algorithm']:
        t = t.replace(f'\\begin{{{env}}}', f'\\begin{{{env}}}[htbp]')

    # ═══════ 8. 修 algorithm 伪代码 ═══════
    if r'\begin{algorithm}' in t and r'\begin{algorithmic}' not in t:
        t = t.replace(r'\begin{algorithm}[htbp]',
            r'\begin{algorithm}[htbp]\caption{Algorithm}\begin{algorithmic}[1]')
    if r'\end{algorithm}' in t and r'\end{algorithmic}' not in t:
        t = t.replace(r'\end{algorithm}', r'\end{algorithmic}\end{algorithm}')

    # ═══════ 9. 加防报错头 ═══════
    if r'\DeclareUnicodeCharacter' not in t:
        t = t.replace(r'\documentclass[11pt]{article}',
            r'\documentclass[11pt]{article}\n\\usepackage{textcomp}\n\\DeclareUnicodeCharacter{202F}{ }\n\\DeclareUnicodeCharacter{00A0}{ }')

    # ═══════ 10. 多余空行 ═══════
    t = re.sub(r'\n{4,}', '\n\n\n', t)

    clean = f.replace('.tex', '_final.tex')
    with open(clean, 'w') as fp: fp.write(t)
    after = len(t)
    remaining = t.count('??') + t.count('[?]') + t.count(' ? ')
    print(f'  {before}→{after} chars, {remaining} issues remaining → {os.path.basename(clean)}')

print('\nDone! Download *_final.tex files to Overleaf.')
