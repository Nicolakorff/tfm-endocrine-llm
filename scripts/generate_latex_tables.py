"""
Genera tablas de comparación y las exporta en formatos legibles:
- LaTeX (.tex)
- Markdown (.md)
- CSV (.csv)
- HTML (.html)
"""

import pandas as pd
from pathlib import Path
import argparse

LATEX_DIR = Path("data/results/semantic_comparison/latex")
MARKDOWN_DIR = Path("data/results/semantic_comparison/markdown")
CSV_DIR = Path("data/results/semantic_comparison/csv")
HTML_DIR = Path("data/results/semantic_comparison/html")

for d in (LATEX_DIR, MARKDOWN_DIR, CSV_DIR, HTML_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Cargar datos
pairwise_df = pd.read_csv("data/results/semantic_comparison/analysis/pairwise_comparisons.csv")

# Filtrar para obtener la comparación léxico vs semántico (lexical_empathy vs semantic_empathy)
semantic_comparison = pairwise_df[
    (pairwise_df['comparison'] == 'Efecto del sesgo semántico') &
    (pairwise_df['cond1'] == 'lexical_empathy') &
    (pairwise_df['cond2'] == 'semantic_empathy')
]

# Crear estructura compatible con el formato esperado
stats_df = pd.DataFrame({
    'metric': semantic_comparison['metric'],
    'M_simple': semantic_comparison['mean1'],
    'M_semantic': semantic_comparison['mean2'],
    'difference': semantic_comparison['diff'],
    't_statistic': semantic_comparison['t'],
    'p_value': semantic_comparison['p'],
    'cohens_d': semantic_comparison['cohens_d']
})

# Mapeo legible de métricas
metric_map = {
    'distinct_2': 'Diversidad léxica',
    'sentiment_polarity': 'Polaridad',
    'repetition_rate': 'Tasa de repetición',
    'length': 'Longitud'
}

def p_sig(p: float) -> str:
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''

def format_row(row: pd.Series):
    sig = p_sig(row['p_value'])
    return {
        'Métrica': metric_map.get(row['metric'], row['metric']),
        'M_simple': f"{row['M_simple']:.3f}",
        'M_semántico': f"{row['M_semantic']:.3f}",
        'Diferencia': f"{row['difference']:+.3f}",
        't': f"{row['t_statistic']:.2f}",
        'p': f"{row['p_value']:.3f}" + (f"^{sig}" if sig else ''),
        'd de Cohen': f"{row['cohens_d']:.2f}",
        'significancia': sig
    }

# Tabla 1: Resultados de tests estadísticos
latex_table1 = r"""\begin{table}[h]
\centering
\caption{Comparación estadística entre sesgo simple y sesgo semántico}
\label{tab:semantic_comparison}
\begin{tabular}{lcccccc}
\hline
\textbf{Métrica} & \textbf{M}_{\text{simple}} & \textbf{M}_{\text{semántico}} & \textbf{Diferencia} & \textbf{t} & \textbf{p} & \textbf{d de Cohen} \\
\hline
"""

for _, row in stats_df.iterrows():
    metric_name = metric_map.get(row['metric'], row['metric'])

    sig = ''
    if row['p_value'] < 0.001:
        sig = '$^{***}$'
    elif row['p_value'] < 0.01:
        sig = '$^{**}$'
    elif row['p_value'] < 0.05:
        sig = '$^{*}$'

    latex_table1 += f"{metric_name} & "
    latex_table1 += f"{row['M_simple']:.3f} & "
    latex_table1 += f"{row['M_semantic']:.3f} & "
    latex_table1 += f"{row['difference']:+.3f} & "
    latex_table1 += f"{row['t_statistic']:.2f} & "
    latex_table1 += f"{row['p_value']:.3f}{sig} & "
    latex_table1 += f"{row['cohens_d']:.2f} \\\\\n"

latex_table1 += r"""\hline
\end{tabular}
\begin{tablenotes}
\small
\item Nota: $^{*}$p < 0.05, $^{**}$p < 0.01, $^{***}$p < 0.001
\item M = Media; d de Cohen: |d| < 0.2 (trivial), 0.2-0.5 (pequeño), 0.5-0.8 (mediano), > 0.8 (grande)
\end{tablenotes}
\end{table}
"""

def export_markdown(df: pd.DataFrame, path: Path):
    headers = ["Métrica", "M_simple", "M_semántico", "Diferencia", "t", "p", "d de Cohen"]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + " | ".join(["---"] * len(headers)) + "|")
    for _, row in df.iterrows():
        fr = format_row(row)
        lines.append("| " + " | ".join([fr[h] for h in headers]) + " |")
    content = "\n".join(lines)
    path.write_text(content, encoding="utf-8")

def export_csv(df: pd.DataFrame, path: Path):
    df_out = pd.DataFrame([format_row(r) for _, r in df.iterrows()])
    cols = ["Métrica", "M_simple", "M_semántico", "Diferencia", "t", "p", "d de Cohen", "significancia"]
    df_out[cols].to_csv(path, index=False)

def export_html(df: pd.DataFrame, path: Path):
    df_out = pd.DataFrame([format_row(r) for _, r in df.iterrows()])
    html = df_out.drop(columns=["significancia"]).to_html(index=False)
    path.write_text(html, encoding="utf-8")


def main(formats: list[str]):
    if "latex" in formats:
        tex_path = LATEX_DIR / "table_statistical_comparison.tex"
        tex_path.write_text(latex_table1, encoding="utf-8")
        print(f"Tabla LaTeX guardada: {tex_path}")
        print("\nContenido LaTeX:\n")
        print(latex_table1)

    if "md" in formats or "markdown" in formats:
        md_path = MARKDOWN_DIR / "table_statistical_comparison.md"
        export_markdown(stats_df, md_path)
        print(f"Tabla Markdown guardada: {md_path}")

    if "csv" in formats:
        csv_path = CSV_DIR / "table_statistical_comparison.csv"
        export_csv(stats_df, csv_path)
        print(f"Tabla CSV guardada: {csv_path}")

    if "html" in formats:
        html_path = HTML_DIR / "table_statistical_comparison.html"
        export_html(stats_df, html_path)
        print(f"Tabla HTML guardada: {html_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera tablas en múltiples formatos")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["latex", "markdown", "csv", "html"],
        help="Formatos a exportar: latex markdown csv html"
    )
    args = parser.parse_args()
    main(args.formats)
