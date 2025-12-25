"""
Analiza correlaciones entre niveles hormonales y métricas de texto.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("="*80)
print(" ANÁLISIS DE CORRELACIÓN: HORMONAS VS MÉTRICAS")
print("="*80 + "\n")

# Configuración
DATA_DIR = Path("data/results/consolidated")
OUTPUT_DIR = DATA_DIR.parent / "correlation_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. CARGAR DATOS
print("1. Cargando datos...")
df = pd.read_csv(DATA_DIR / "all_experiments_consolidated.csv")

# Verificar columnas hormonales
hormone_cols = [col for col in df.columns if col.startswith('hormone_')]

if not hormone_cols:
    print("   No se encontraron columnas hormonales")
    print("   Intentando extraer de profile_name...")

    # Fallback: usar valores conocidos de perfiles
    from endocrine_llm import HORMONE_PROFILES

    def extract_hormones(profile_name):
        if profile_name in HORMONE_PROFILES:
            profile = HORMONE_PROFILES[profile_name]
            return pd.Series({
                'hormone_dopamine': profile.dopamine,
                'hormone_cortisol': profile.cortisol,
                'hormone_oxytocin': profile.oxytocin,
                'hormone_adrenaline': profile.adrenaline,
                'hormone_serotonin': profile.serotonin
            })
        return pd.Series({
            'hormone_dopamine': np.nan,
            'hormone_cortisol': np.nan,
            'hormone_oxytocin': np.nan,
            'hormone_adrenaline': np.nan,
            'hormone_serotonin': np.nan
        })

    hormone_df = df['profile_name'].apply(extract_hormones)
    df = pd.concat([df, hormone_df], axis=1)
    hormone_cols = hormone_df.columns.tolist()

print(f"   Columnas hormonales encontradas: {hormone_cols}")

# 2. SELECCIONAR MÉTRICAS
metrics = ['distinct_2', 'sentiment_polarity', 'repetition_rate', 'length']
if 'perplexity' in df.columns:
    metrics.append('perplexity')

print(f"   Métricas a analizar: {metrics}")

# 3. PREPARAR DATASET PARA CORRELACIÓN
print("\n2. Preparando datos para correlación...")

# Seleccionar columnas relevantes
corr_cols = hormone_cols + metrics
df_corr = df[corr_cols].dropna()

print(f"   Observaciones con datos completos: {len(df_corr)}")

if len(df_corr) < 30:
    print("   Datos insuficientes para análisis de correlación")
    exit(1)

# 4. MATRIZ DE CORRELACIÓN
print("\n3. Calculando matriz de correlación...")

# Correlación de Pearson
corr_matrix = df_corr.corr(method='pearson')

# Extraer submatriz: hormonas vs métricas
corr_hormones_metrics = corr_matrix.loc[hormone_cols, metrics]

print("\nMatriz de correlación (Hormonas × Métricas):")
print(corr_hormones_metrics.round(3))

# Guardar
corr_hormones_metrics.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

# 5. TESTS DE SIGNIFICANCIA
print("\n4. Tests de significancia (p-values)...")

p_matrix = pd.DataFrame(
    index=hormone_cols,
    columns=metrics,
    dtype=float
)

for hormone in hormone_cols:
    for metric in metrics:
        r, p = stats.pearsonr(df_corr[hormone], df_corr[metric])
        p_matrix.loc[hormone, metric] = p

print("\nP-values:")
print(p_matrix.round(4))

# Guardar
p_matrix.to_csv(OUTPUT_DIR / "correlation_pvalues.csv")

# 6. IDENTIFICAR CORRELACIONES SIGNIFICATIVAS
print("\n5. Correlaciones significativas (p < 0.05):")
print("-" * 80)

significant_corrs = []

for hormone in hormone_cols:
    for metric in metrics:
        r = corr_hormones_metrics.loc[hormone, metric]
        p = p_matrix.loc[hormone, metric]

        if p < 0.05:
            # Interpretar fuerza
            abs_r = abs(r)
            if abs_r < 0.3:
                strength = "débil"
            elif abs_r < 0.5:
                strength = "moderada"
            else:
                strength = "fuerte"

            direction = "positiva" if r > 0 else "negativa"

            print(f"\n{hormone.replace('hormone_', '').upper()} × {metric}:")
            print(f"   r = {r:.3f}, p = {p:.4f}")
            print(f"   Correlación {strength} {direction}")

            significant_corrs.append({
                'hormone': hormone.replace('hormone_', ''),
                'metric': metric,
                'correlation': r,
                'p_value': p,
                'strength': strength,
                'direction': direction
            })

# Guardar significativas
if significant_corrs:
    sig_df = pd.DataFrame(significant_corrs)
    sig_df.to_csv(OUTPUT_DIR / "significant_correlations.csv", index=False)
    print(f"\n   {len(significant_corrs)} correlaciones significativas guardadas")
else:
    print("\n   No se encontraron correlaciones significativas")

# 7. VISUALIZACIÓN - HEATMAP
print("\n6. Generando heatmap...")

fig, ax = plt.subplots(figsize=(10, 8))

# Crear máscara para p-values no significativos
mask = p_matrix.astype(float) >= 0.05

# Heatmap
sns.heatmap(
    corr_hormones_metrics,
    annot=True,
    fmt='.3f',
    cmap='RdBu_r',
    center=0,
    vmin=-1,
    vmax=1,
    mask=mask,
    cbar_kws={'label': 'Correlación de Pearson'},
    linewidths=0.5,
    ax=ax
)

ax.set_title('Correlaciones: Niveles Hormonales × Métricas de Texto\n'
             '(Solo correlaciones significativas, p < 0.05)',
             fontsize=14, fontweight='bold', pad=20)

ax.set_xlabel('Métricas de Texto', fontsize=12)
ax.set_ylabel('Niveles Hormonales', fontsize=12)

# Limpiar labels
ax.set_yticklabels([label.get_text().replace('hormone_', '').title() 
                    for label in ax.get_yticklabels()])
ax.set_xticklabels([label.get_text().replace('_', ' ').title() 
                    for label in ax.get_xticklabels()], rotation=45, ha='right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'correlation_heatmap.pdf', bbox_inches='tight')
print(f"   {OUTPUT_DIR / 'correlation_heatmap.png'}")

# 8. SCATTERPLOTS DE CORRELACIONES MÁS FUERTES
if significant_corrs:
    print("\n7. Generando scatterplots de correlaciones más fuertes...")

    # Top 4 correlaciones por valor absoluto
    sig_df_sorted = sig_df.sort_values('correlation', key=abs, ascending=False)
    top_corrs = sig_df_sorted.head(4)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(top_corrs.iterrows()):
        ax = axes[idx]

        hormone_col = f"hormone_{row['hormone']}"
        metric_col = row['metric']

        # Scatter plot
        ax.scatter(df_corr[hormone_col], df_corr[metric_col], 
                  alpha=0.5, s=30, edgecolors='black', linewidth=0.5)

        # Línea de tendencia
        z = np.polyfit(df_corr[hormone_col], df_corr[metric_col], 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(df_corr[hormone_col].min(), df_corr[hormone_col].max(), 100)
        ax.plot(x_line, p_line(x_line), "r--", linewidth=2, alpha=0.8)

        # Labels
        ax.set_xlabel(row['hormone'].title(), fontsize=11)
        ax.set_ylabel(metric_col.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f"r = {row['correlation']:.3f}, p = {row['p_value']:.4f}",
                    fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle('Correlaciones Más Fuertes: Niveles Hormonales × Métricas',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'correlation_scatterplots.png', dpi=300, bbox_inches='tight')
    print(f"   {OUTPUT_DIR / 'correlation_scatterplots.png'}")

# 9. TABLA LATEX
print("\n8. Generando tabla LaTeX...")

latex_table = r"""\begin{table}[h]
\centering
\caption{Correlaciones significativas entre niveles hormonales y métricas de texto}
\label{tab:hormone_correlations}
\begin{tabular}{llccc}
\hline
\textbf{Hormona} & \textbf{Métrica} & \textbf{r} & \textbf{p} & \textbf{Interpretación} \\
\hline
"""

if significant_corrs:
    for _, row in sig_df_sorted.iterrows():
        latex_table += f"{row['hormone'].title()} & "
        latex_table += f"{row['metric'].replace('_', ' ')} & "
        latex_table += f"{row['correlation']:.3f} & "

        # Formatear p-value
        if row['p_value'] < 0.001:
            p_str = "< 0.001$^{***}$"
        elif row['p_value'] < 0.01:
            p_str = f"{row['p_value']:.3f}$^{{**}}$"
        elif row['p_value'] < 0.05:
            p_str = f"{row['p_value']:.3f}$^{{*}}$"
        else:
            p_str = f"{row['p_value']:.3f}"

        latex_table += f"{p_str} & "
        latex_table += f"{row['strength']} {row['direction']} \\\\\n"
else:
    latex_table += r"\multicolumn{5}{c}{No se encontraron correlaciones significativas (p < 0.05)} \\" + "\n"

latex_table += r"""\hline
\end{tabular}
\begin{tablenotes}
\small
\item Nota: $^{*}$p < 0.05, $^{**}$p < 0.01, $^{***}$p < 0.001
\item Interpretación de r: |r| < 0.3 (débil), 0.3-0.5 (moderada), > 0.5 (fuerte)
\end{tablenotes}
\end{table}
"""

with open(OUTPUT_DIR / "correlation_table.tex", 'w') as f:
    f.write(latex_table)
print(f"   {OUTPUT_DIR / 'correlation_table.tex'}")

print("\n" + "="*80)
print(" ANÁLISIS DE CORRELACIÓN COMPLETADO")
print("="*80)
