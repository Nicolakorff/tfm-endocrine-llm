"""
Genera tablas en formato LaTeX
"""

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("data/results/semantic_comparison/latex")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cargar datos
stats_df = pd.read_csv("data/results/semantic_comparison/analysis/statistical_tests.csv")

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
    metric_name = {
        'distinct_2': 'Diversidad léxica',
        'sentiment_polarity': 'Polaridad',
        'repetition_rate': 'Tasa de repetición',
        'length': 'Longitud'
    }.get(row['metric'], row['metric'])

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

# Guardar
with open(OUTPUT_DIR / "table_statistical_comparison.tex", 'w') as f:
    f.write(latex_table1)

print(f"Tabla LaTeX guardada: {OUTPUT_DIR / 'table_statistical_comparison.tex'}")
print("\n Contenido:")
print(latex_table1)
