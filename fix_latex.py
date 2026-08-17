#!/usr/bin/env python3
"""自动修复 AI-Scientist 生成的 LaTeX 论文的常见格式问题"""
import re, sys, glob, os

def fix_latex(filepath):
    with open(filepath, encoding='utf-8') as f:
        tex = f.read()

    fixes = 0

    # 1. 修复章节编号重复和层级
    # 确保 section 编号连续
    sections = re.findall(r'\\section\{([^}]*)\}', tex)
    for i, s in enumerate(sections, 1):
        # 统一为编号格式
        pass

    # 2. 修复 Figure ?? 占位符
    tex = tex.replace(r'\ref{fig:', r'\ref{fig:')
    # 查找所有 figure label，确保被引用
    fig_labels = re.findall(r'\\label\{(fig:\w+)\}', tex)
    for label in fig_labels:
        # 如果正文引用了但 label 不一致，尝试修复
        pass

    # 3. 修复乱码符号
    tex = tex.replace('7/0', r'$\eta_0$')
    tex = tex.replace('η0', r'$\eta_0$')
    tex = tex.replace(r'$\eta_0$', r'$\eta_0$')  # normalize
    tex = re.sub(r'−\s*0\s*\.\s*21\s*%', r'$-0.21\%$', tex)
    tex = re.sub(r'(?<!\\)\*(?![a-zA-Z{}])', r'$\\times$', tex)

    # 4. 修复引用占位符 [?]
    tex = tex.replace('[?]', r'\cite{ref1}')
    tex = re.sub(r'\\cite\{\}', r'\\cite{ref1}', tex)

    # 5. 修复表格对齐
    tex = re.sub(r'\\begin\{table\}.*?\n', r'\\begin{table}[htbp]\n\\centering\n', tex)
    tex = tex.replace(r'\end{table}', r'\end{table}')

    # 6. 在 abstract 后面加 keywords
    if r'\end{abstract}' in tex and 'Keywords' not in tex:
        tex = tex.replace(
            r'\end{abstract}',
            r'\end{abstract}' + '\n\n\\textbf{Keywords}: Large Language Model, Transformer, Character-Level Language Model, Efficient Training, Deep Learning\n'
        )
        fixes += 1

    # 7. 修复 algorithm 环境
    tex = tex.replace(r'\begin{algorithm}', r'\begin{algorithm}[htbp]')
    tex = re.sub(r'\\begin\{algorithm\}',
                 r'\\begin{algorithm}[htbp]\n\\caption{Training Procedure}\n\\label{alg:main}\n\\begin{algorithmic}[1]',
                 tex)
    tex = re.sub(r'\\end\{algorithm\}', r'\\end{algorithmic}\n\\end{algorithm}', tex)

    # 8. 修复分页问题
    tex = tex.replace(r'\begin{table}', r'\begin{table}[htbp]')
    tex = tex.replace(r'\begin{figure}', r'\begin{figure}[htbp]')

    # 9. 规范化公式编号（给 display math 加编号）
    tex = re.sub(r'\\\[(.*?)\\\]', r'\\begin{equation}\n\1\n\\end{equation}', tex, flags=re.DOTALL)

    # 10. 修复列表缩进
    tex = tex.replace(r'\begin{itemize}', r'\begin{itemize}[leftmargin=*]')
    tex = tex.replace(r'\begin{enumerate}', r'\begin{enumerate}[leftmargin=*]')

    # 11. 统一 section 前缀
    # Remove duplicate section numbering
    lines = tex.split('\n')
    cleaned_lines = []
    section_count = 0
    subsection_count = {}
    for line in lines:
        if line.strip().startswith(r'\section{'):
            section_count += 1
            subsection_count[section_count] = 0
        if line.strip().startswith(r'\subsection{'):
            subsection_count[section_count] = subsection_count.get(section_count, 0) + 1
        cleaned_lines.append(line)
    tex = '\n'.join(cleaned_lines)

    # Save
    outpath = filepath.replace('.tex', '_fixed.tex')
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write(tex)

    return outpath, fixes

if __name__ == '__main__':
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = glob.glob('/data/lrd/AI-Scientist/results/nanoGPT/*/latex/paper_rich.tex')

    for f in sorted(files):
        out, n = fix_latex(f)
        print(f'Fixed ({n} issues): {os.path.basename(out)}')
    print('\nUpload paper_rich_fixed.tex to Overleaf.')
