"""
Script para convertir latex a formato markdown
"""
import re
from pathlib import Path

def latex_to_markdown(latex_content: str) -> str:
    # Convertir tablas
    latex_content = re.sub(r'\\begin{tabular}{(.*?)}', r'| \1 |', latex_content)
    latex_content = re.sub(r'\\end{tabular}', '', latex_content)
    latex_content = re.sub(r'\\hline', '', latex_content)
    latex_content = re.sub(r'&', '|', latex_content)
    latex_content = re.sub(r'\\\\', '\n', latex_content)

    # Convertir negritas
    latex_content = re.sub(r'\\textbf{(.*?)}', r'**\1**', latex_content)

    # Convertir itálicas
    latex_content = re.sub(r'\\textit{(.*?)}', r'*\1*', latex_content)

    # Convertir notas de tabla
    latex_content = re.sub(r'\\begin{tablenotes}.*?\\end{tablenotes}', '', latex_content, flags=re.DOTALL)
    latex_content = re.sub(r'\\item', '- ', latex_content)
    latex_content = re.sub(r'\\small', '', latex_content)
    latex_content = re.sub(r'\\note:', 'Nota:', latex_content)
    return latex_content

from pathlib import Path
import pandas as pd
# Cargar datos de resultados estadísticos
latex_file = Path('data/results/anova_analysis/anova_table.tex')
latex_content = latex_file.read_text(encoding='utf-8')
markdown_content = latex_to_markdown(latex_content)
markdown_file = Path('data/results/anova_analysis/anova_table.md')
markdown_file.write_text(markdown_content, encoding='utf-8')
print(f"Tabla convertida y guardada en {markdown_file}")