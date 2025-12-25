"""
Análisis ANOVA completo de todos los experimentos con tests post-hoc.
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("="*80)
print(" ANÁLISIS ANOVA COMPLETO")
print("="*80 + "\n")

# Configuración
DATA_DIR = Path("data/results/consolidated")
OUTPUT_DIR = DATA_DIR.parent / "anova_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. CARGAR DATOS CONSOLIDADOS
print("1. Cargando datos consolidados...")
df = pd.read_csv(DATA_DIR / "all_experiments_consolidated.csv")
print(f"   Total observaciones: {len(df)}")
print(f"   Perfiles únicos: {df['profile_name'].nunique()}")

# 2. PREPARAR DATOS PARA ANOVA
print("\n2. Preparando datos para ANOVA...")

# Filtrar perfiles con suficientes observaciones (mínimo 30)
profile_counts = df['profile_name'].value_counts()
valid_profiles = profile_counts[profile_counts >= 30].index.tolist()

df_anova = df[df['profile_name'].isin(valid_profiles)].copy()
print(f"   Perfiles con n≥30: {len(valid_profiles)}")
print(f"   Observaciones para ANOVA: {len(df_anova)}")

# Perfiles incluidos
print("\n   Perfiles incluidos en ANOVA:")
for profile in valid_profiles:
    n = len(df_anova[df_anova['profile_name'] == profile])
    print(f"      {profile}: n={n}")

# 3. ANOVA PARA CADA MÉTRICA
print("\n3. Ejecutando ANOVA para cada métrica...")
print("-" * 80)

metrics = ['distinct_2', 'sentiment_polarity', 'repetition_rate', 'length']
if 'perplexity' in df_anova.columns:
    metrics.append('perplexity')

anova_results = []

for metric in metrics:
    print(f"\n MÉTRICA: {metric.upper()}")
    print("-" * 60)

    # Limpiar NaN
    df_metric = df_anova[[metric, 'profile_name']].dropna()

    if len(df_metric) < 50:
        print(f"   Datos insuficientes (n={len(df_metric)}), saltando...")
        continue

    # Verificar normalidad por grupo (test de Shapiro-Wilk en muestra)
    print("   Tests de normalidad (muestra de 50 por grupo):")
    normality_ok = True
    for profile in valid_profiles[:5]:  # Solo primeros 5 para no saturar
        sample = df_metric[df_metric['profile_name'] == profile][metric].sample(
            min(50, len(df_metric[df_metric['profile_name'] == profile]))
        )
        if len(sample) > 3:
            stat, p = stats.shapiro(sample)
            print(f"      {profile}: W={stat:.3f}, p={p:.3f} {'✓' if p > 0.05 else '✗'}")
            if p < 0.05:
                normality_ok = False

    # Test de Levene (homogeneidad de varianzas)
    groups = [df_metric[df_metric['profile_name'] == p][metric].values 
              for p in valid_profiles]
    levene_stat, levene_p = stats.levene(*groups)
    print("\n   Test de Levene (homogeneidad):")
    print(f"      F={levene_stat:.3f}, p={levene_p:.3f} {'✓' if levene_p > 0.05 else '✗'}")

    homogeneity_ok = levene_p > 0.05

    # ANOVA one-way
    f_stat, p_value = stats.f_oneway(*groups)

    print("\n   ANOVA One-Way:")
    print(f"      F({len(valid_profiles)-1}, {len(df_metric)-len(valid_profiles)}) = {f_stat:.3f}")
    print(f"      p = {p_value:.4f}")

    if p_value < 0.001:
        sig = "***"
        interpretation = "Diferencias MUY significativas entre grupos"
    elif p_value < 0.01:
        sig = "**"
        interpretation = "Diferencias significativas entre grupos"
    elif p_value < 0.05:
        sig = "*"
        interpretation = "Diferencias marginalmente significativas entre grupos"
    else:
        sig = "ns"
        interpretation = "No hay diferencias significativas entre grupos"

    print(f"      Significancia: {sig}")
    print(f"      Interpretación: {interpretation}")

    # Eta squared (effect size)
    grand_mean = df_metric[metric].mean()
    ss_between = sum(len(df_metric[df_metric['profile_name']==p]) * 
                     (df_metric[df_metric['profile_name']==p][metric].mean() - grand_mean)**2 
                     for p in valid_profiles)
    ss_total = sum((df_metric[metric] - grand_mean)**2)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0

    print(f"      η² = {eta_squared:.4f}", end="")
    if eta_squared < 0.01:
        print(" (efecto trivial)")
    elif eta_squared < 0.06:
        print(" (efecto pequeño)")
    elif eta_squared < 0.14:
        print(" (efecto mediano)")
    else:
        print(" (efecto grande)")

    # Guardar resultados
    anova_results.append({
        'metric': metric,
        'n_groups': len(valid_profiles),
        'n_total': len(df_metric),
        'F_statistic': f_stat,
        'p_value': p_value,
        'significance': sig,
        'eta_squared': eta_squared,
        'normality_assumption': 'OK' if normality_ok else 'Violated',
        'homogeneity_assumption': 'OK' if homogeneity_ok else 'Violated',
        'interpretation': interpretation
    })

    # 4. TEST POST-HOC (Tukey HSD) si ANOVA significativo
    if p_value < 0.05:
        print("\n    Test Post-Hoc (Tukey HSD):")

        tukey = pairwise_tukeyhsd(
            endog=df_metric[metric],
            groups=df_metric['profile_name'],
            alpha=0.05
        )

        # Guardar resultados completos
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], 
                                columns=tukey.summary().data[0])
        tukey_df.to_csv(OUTPUT_DIR / f"tukey_{metric}.csv", index=False)

        # Mostrar comparaciones más significativas
        tukey_df['meandiff'] = tukey_df['meandiff'].astype(float)
        tukey_df['reject'] = tukey_df['reject'].astype(bool)

        significant = tukey_df[tukey_df['reject'] == True].sort_values(
            'meandiff', key=abs, ascending=False
        )

        if len(significant) > 0:
            print(f"      Comparaciones significativas: {len(significant)}")
            print("\n      Top 5 diferencias más grandes:")
            for idx, row in significant.head(5).iterrows():
                print(f"         {row['group1']:20s} vs {row['group2']:20s}: "
                      f"Δ={row['meandiff']:+.4f} (p={row['p-adj']:.4f})")
        else:
            print("      No se encontraron diferencias significativas en post-hoc")

        print(f"\n      Resultados completos guardados: tukey_{metric}.csv")

# 5. GUARDAR RESUMEN DE ANOVA
print("\n" + "="*80)
print("5. Guardando resumen de ANOVA...")

anova_df = pd.DataFrame(anova_results)
anova_df.to_csv(OUTPUT_DIR / "anova_summary.csv", index=False)
print(f"   {OUTPUT_DIR / 'anova_summary.csv'}")

# 6. CREAR TABLA LATEX
print("\n6. Generando tabla LaTeX...")

latex_table = r"""\begin{table}[h]
\centering
\caption{Resultados de ANOVA para comparación entre perfiles hormonales}
\label{tab:anova_results}
\begin{tabular}{lcccccc}
\hline
\textbf{Métrica} & \textbf{F} & \textbf{df1} & \textbf{df2} & \textbf{p} & \textbf{$\eta^2$} & \textbf{Sig.} \\
\hline
"""

for _, row in anova_df.iterrows():
    metric_name = {
        'distinct_2': 'Diversidad léxica',
        'sentiment_polarity': 'Polaridad',
        'repetition_rate': 'Tasa de repetición',
        'length': 'Longitud',
        'perplexity': 'Perplejidad'
    }.get(row['metric'], row['metric'])

    df1 = row['n_groups'] - 1
    df2 = row['n_total'] - row['n_groups']

    latex_table += f"{metric_name} & "
    latex_table += f"{row['F_statistic']:.2f} & "
    latex_table += f"{df1} & "
    latex_table += f"{df2} & "
    latex_table += f"{row['p_value']:.4f} & "
    latex_table += f"{row['eta_squared']:.3f} & "
    latex_table += f"{row['significance']} \\\\\n"

latex_table += r"""\hline
\end{tabular}
\begin{tablenotes}
\small
\item Nota: $^{*}$p < 0.05, $^{**}$p < 0.01, $^{***}$p < 0.001, ns = no significativo
\item $\eta^2$: < 0.01 (trivial), 0.01-0.06 (pequeño), 0.06-0.14 (mediano), > 0.14 (grande)
\end{tablenotes}
\end{table}
"""

with open(OUTPUT_DIR / "anova_table.tex", 'w') as f:
    f.write(latex_table)
print(f"   {OUTPUT_DIR / 'anova_table.tex'}")

# 7. VISUALIZACIÓN
print("\n7. Generando visualizaciones...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Comparación de Métricas entre Perfiles Hormonales (ANOVA)', 
             fontsize=16, fontweight='bold')

for idx, metric in enumerate(metrics[:4]):
    ax = axes[idx // 2, idx % 2]

    # Datos limpios
    df_plot = df_anova[[metric, 'profile_name']].dropna()

    # Boxplot
    sns.boxplot(data=df_plot, x='profile_name', y=metric, ax=ax)
    ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Perfil Hormonal')
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    # Añadir resultado de ANOVA
    anova_result = anova_df[anova_df['metric'] == metric]
    if len(anova_result) > 0:
        row = anova_result.iloc[0]
        ax.text(0.02, 0.98, 
                f"F={row['F_statistic']:.2f}, p={row['p_value']:.4f} {row['significance']}",
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'anova_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'anova_comparison.pdf', bbox_inches='tight')
print(f"   {OUTPUT_DIR / 'anova_comparison.png'}")
print(f"   {OUTPUT_DIR / 'anova_comparison.pdf'}")

print("\n" + "="*80)
print(" ANÁLISIS ANOVA COMPLETADO")
print("="*80)

# Resumen ejecutivo
print("\n RESUMEN EJECUTIVO:")
print("-" * 80)
for _, row in anova_df.iterrows():
    print(f"\n{row['metric'].upper()}:")
    print(f"   {row['interpretation']}")
    print(f"   F={row['F_statistic']:.2f}, p={row['p_value']:.4f}, η²={row['eta_squared']:.3f}")
    if row['normality_assumption'] != 'OK' or row['homogeneity_assumption'] != 'OK':
        print(f"   Asunciones: Normalidad={row['normality_assumption']}, "
              f"Homogeneidad={row['homogeneity_assumption']}")

print("\n" + "="*80)
print(" Dataset final:")
print(f"   Filas: {len(df_anova)}")
print(f"   Columnas: {len(df_anova.columns)}")
print(f"   Tamaño: {df_anova.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print("="*80)
