"""
isolated_hormone_analysis.py

Análisis específico del efecto de hormonas individuales (Fase 1).
Compara cada hormona alta vs baseline para aislar efectos.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("="*80)
print(" ANÁLISIS: EFECTO DE HORMONAS INDIVIDUALES (AISLADAS)")
print("="*80 + "\n")

# Configuración
DATA_DIR = Path("data/results")
OUTPUT_DIR = DATA_DIR / "isolated_hormones_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. CARGAR DATOS DE FASE 1
print("1. Cargando datos de Fase 1 (hormonas individuales)...")

if not (DATA_DIR / "phase1_results.csv").exists():
    print(" ERROR: phase1_results.csv no encontrado")
    print("   Ejecuta primero: python scripts/run_experiment_phase1.py")
    exit(1)

df = pd.read_csv(DATA_DIR / "phase1_results.csv")
print(f"   Total generaciones: {len(df)}")

# 2. IDENTIFICAR PERFILES
print("\n2. Identificando perfiles de hormonas individuales...")

# Perfiles esperados
individual_profiles = [
    'baseline',
    'high_dopamine',
    'high_cortisol',
    'high_oxytocin',
    'high_adrenaline',
    'high_serotonin'
]

# Verificar que existen
available_profiles = df['profile_name'].unique()
missing = [p for p in individual_profiles if p not in available_profiles]

if missing:
    print(f"    Perfiles faltantes: {missing}")
    individual_profiles = [p for p in individual_profiles if p in available_profiles]

print(f"   Perfiles disponibles para análisis: {individual_profiles}")

# Filtrar solo perfiles individuales
df_isolated = df[df['profile_name'].isin(individual_profiles)].copy()
print(f"   Observaciones para análisis: {len(df_isolated)}")

# Verificar que tenemos baseline
if 'baseline' not in individual_profiles:
    print("    ERROR: No se encontró perfil 'baseline'")
    exit(1)

# 3. MÉTRICAS A ANALIZAR
metrics = ['distinct_2', 'sentiment_polarity', 'repetition_rate', 'length']
if 'perplexity' in df_isolated.columns:
    metrics.append('perplexity')

print(f"\n3. Métricas a analizar: {metrics}")

# 4. CALCULAR EFECTOS (diferencia vs baseline)
print("\n4. Calculando efectos de cada hormona vs baseline...")
print("-" * 80)

baseline_data = df_isolated[df_isolated['profile_name'] == 'baseline']

effects_results = []

for hormone_profile in individual_profiles:
    if hormone_profile == 'baseline':
        continue

    hormone_name = hormone_profile.replace('high_', '')
    hormone_data = df_isolated[df_isolated['profile_name'] == hormone_profile]

    print(f"\n {hormone_name.upper()}")
    print(f"   n_baseline={len(baseline_data)}, n_hormone={len(hormone_data)}")

    for metric in metrics:
        baseline_vals = baseline_data[metric].dropna()
        hormone_vals = hormone_data[metric].dropna()

        if len(baseline_vals) < 5 or len(hormone_vals) < 5:
            print(f"    {metric}: datos insuficientes")
            continue

        # Estadísticas
        baseline_mean = baseline_vals.mean()
        baseline_std = baseline_vals.std()
        hormone_mean = hormone_vals.mean()
        hormone_std = hormone_vals.std()

        # Diferencia absoluta
        diff = hormone_mean - baseline_mean

        # Test t
        t_stat, p_value = stats.ttest_ind(baseline_vals, hormone_vals)

        # Cohen's d (effect size)
        pooled_std = np.sqrt((baseline_vals.var() + hormone_vals.var()) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0

        # Interpretar
        if abs(cohens_d) < 0.2:
            effect_interpretation = "trivial"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "pequeño"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "mediano"
        else:
            effect_interpretation = "grande"

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

        # Dirección del efecto
        direction = "↑" if diff > 0 else "↓"

        print(f"\n   {metric}:")
        print(f"      Baseline: M={baseline_mean:.4f} (SD={baseline_std:.4f})")
        print(f"      {hormone_name}: M={hormone_mean:.4f} (SD={hormone_std:.4f})")
        print(f"      Diferencia: {direction} {abs(diff):.4f}")
        print(f"      t={t_stat:.3f}, p={p_value:.4f} {sig}")
        print(f"      Cohen's d={cohens_d:.3f} (efecto {effect_interpretation})")

        # Guardar resultados
        effects_results.append({
            'hormone': hormone_name,
            'metric': metric,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'hormone_mean': hormone_mean,
            'hormone_std': hormone_std,
            'difference': diff,
            'percent_change': (diff / baseline_mean * 100) if baseline_mean != 0 else 0,
            't_statistic': t_stat,
            'p_value': p_value,
            'significance': sig,
            'cohens_d': cohens_d,
            'effect_size': effect_interpretation,
            'direction': 'increase' if diff > 0 else 'decrease'
        })

# 5. GUARDAR RESULTADOS
print("\n5. Guardando resultados...")

effects_df = pd.DataFrame(effects_results)
effects_df.to_csv(OUTPUT_DIR / "hormone_effects_summary.csv", index=False)
print(f"    {OUTPUT_DIR / 'hormone_effects_summary.csv'}")

# 6. VISUALIZACIÓN PRINCIPAL: EFECTO POR HORMONA
print("\n6. Generando visualizaciones...")

# Figura 1: Diferencias por hormona y métrica
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Efecto de Hormonas Individuales vs Baseline',
             fontsize=16, fontweight='bold')

for idx, metric in enumerate(metrics[:4]):
    ax = axes[idx // 2, idx % 2]

    metric_data = effects_df[effects_df['metric'] == metric].copy()

    if len(metric_data) == 0:
        continue

    # Ordenar por diferencia absoluta
    metric_data = metric_data.sort_values('difference', key=abs, ascending=False)

    # Colores según dirección
    colors = ['green' if d > 0 else 'red' for d in metric_data['difference']]

    # Barplot horizontal
    y_pos = np.arange(len(metric_data))
    ax.barh(y_pos, metric_data['difference'], color=colors, alpha=0.7, edgecolor='black')

    # Línea en 0
    ax.axvline(0, color='black', linestyle='--', linewidth=1)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([h.title() for h in metric_data['hormone']])
    ax.set_xlabel('Diferencia vs Baseline')
    ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Añadir significancia
    for i, (_, row) in enumerate(metric_data.iterrows()):
        x_pos = row['difference'] + (0.01 if row['difference'] > 0 else -0.01)
        ax.text(x_pos, i, row['significance'],
               ha='left' if row['difference'] > 0 else 'right',
               va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'hormone_effects_barplot.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'hormone_effects_barplot.pdf', bbox_inches='tight')
print(f"    {OUTPUT_DIR / 'hormone_effects_barplot.png'}")

# Figura 2: Heatmap de efectos
fig, ax = plt.subplots(figsize=(10, 6))

# Pivot para heatmap
pivot_data = effects_df.pivot(index='hormone', columns='metric', values='difference')

# Máscara para no significativos
pivot_sig = effects_df.pivot(index='hormone', columns='metric', values='p_value')
mask = pivot_sig >= 0.05

sns.heatmap(
    pivot_data,
    annot=True,
    fmt='.3f',
    cmap='RdYlGn',
    center=0,
    mask=mask,
    cbar_kws={'label': 'Diferencia vs Baseline'},
    linewidths=0.5,
    ax=ax
)

ax.set_title('Mapa de Efectos: Hormonas Individuales\n(Solo efectos significativos, p < 0.05)',
            fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Métrica', fontsize=12)
ax.set_ylabel('Hormona', fontsize=12)

# Limpiar labels
ax.set_yticklabels([label.get_text().title() for label in ax.get_yticklabels()], rotation=0)
ax.set_xticklabels([label.get_text().replace('_', ' ').title()
                    for label in ax.get_xticklabels()], rotation=45, ha='right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'hormone_effects_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'hormone_effects_heatmap.pdf', bbox_inches='tight')
print(f"    {OUTPUT_DIR / 'hormone_effects_heatmap.png'}")

# Figura 3: Boxplots comparativos
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Distribuciones: Baseline vs Hormonas Individuales',
             fontsize=16, fontweight='bold')

for idx, metric in enumerate(metrics[:4]):
    ax = axes[idx // 2, idx % 2]

    # Datos para boxplot
    df_plot = df_isolated[[metric, 'profile_name']].dropna()

    # Ordenar: baseline primero, luego resto alfabético
    order = ['baseline'] + sorted([p for p in individual_profiles if p != 'baseline'])

    sns.boxplot(data=df_plot, x='profile_name', y=metric, order=order, ax=ax)

    ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    # Destacar baseline
    ax.get_xticklabels()[0].set_weight('bold')
    ax.get_xticklabels()[0].set_color('blue')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'hormone_effects_boxplots.png', dpi=300, bbox_inches='tight')
print(f"    {OUTPUT_DIR / 'hormone_effects_boxplots.png'}")

# 7. TABLA LATEX
print("\n7. Generando tabla LaTeX...")

latex_table = r"""\begin{table}[h]
\centering
\caption{Efecto de hormonas individuales sobre métricas de texto (comparación vs baseline)}
\label{tab:isolated_hormone_effects}
\begin{tabular}{llcccccc}
\hline
\textbf{Hormona} & \textbf{Métrica} & \textbf{M}_{\text{baseline}} & \textbf{M}_{\text{hormona}} & \textbf{Δ} & \textbf{t} & \textbf{p} & \textbf{d} \\
\hline
"""

# Agrupar por hormona
for hormone in ['dopamine', 'cortisol', 'oxytocin', 'adrenaline', 'serotonin']:
    hormone_data = effects_df[effects_df['hormone'] == hormone]

    if len(hormone_data) == 0:
        continue

    latex_table += f"\\multicolumn{{8}}{{l}}{{\\textbf{{{hormone.title()}}}}} \\\\\n"

    for _, row in hormone_data.iterrows():
        metric_name = row['metric'].replace('_', ' ')

        latex_table += f"& {metric_name} & "
        latex_table += f"{row['baseline_mean']:.3f} & "
        latex_table += f"{row['hormone_mean']:.3f} & "
        latex_table += f"{row['difference']:+.3f} & "
        latex_table += f"{row['t_statistic']:.2f} & "

        # Formatear p-value con significancia
        if row['p_value'] < 0.001:
            p_str = "< 0.001$^{***}$"
        elif row['p_value'] < 0.01:
            p_str = f"{row['p_value']:.3f}$^{{**}}$"
        elif row['p_value'] < 0.05:
            p_str = f"{row['p_value']:.3f}$^{{*}}$"
        else:
            p_str = f"{row['p_value']:.3f}"

        latex_table += f"{p_str} & "
        latex_table += f"{row['cohens_d']:.2f} \\\\\n"

    latex_table += "\\hline\n"

latex_table += r"""\end{tabular}
\begin{tablenotes}
\small
\item Nota: M = Media; Δ = Diferencia (hormona - baseline); d = d de Cohen
\item Significancia: $^{*}$p < 0.05, $^{**}$p < 0.01, $^{***}$p < 0.001
\item Tamaño del efecto: |d| < 0.2 (trivial), 0.2-0.5 (pequeño), 0.5-0.8 (mediano), > 0.8 (grande)
\end{tablenotes}
\end{table}
"""

with open(OUTPUT_DIR / "isolated_hormones_table.tex", 'w') as f:
    f.write(latex_table)
print(f"    {OUTPUT_DIR / 'isolated_hormones_table.tex'}")

# 8. RESUMEN TEXTUAL
print("\n8. Generando resumen textual...")

summary = """# Análisis de Hormonas Individuales (Aisladas)

## Resumen Ejecutivo

Este análisis evalúa el efecto de cada hormona individual (elevada a 0.9)
comparada contra el perfil baseline (todas en 0.5).

## Efectos Significativos Detectados

"""

# Efectos significativos por hormona
for hormone in ['dopamine', 'cortisol', 'oxytocin', 'adrenaline', 'serotonin']:
    hormone_data = effects_df[
        (effects_df['hormone'] == hormone) &
        (effects_df['p_value'] < 0.05)
    ].sort_values('cohens_d', key=abs, ascending=False)

    if len(hormone_data) == 0:
        summary += f"\n### {hormone.title()}\n"
        summary += "- No se detectaron efectos significativos (p < 0.05)\n"
        continue

    summary += f"\n### {hormone.title()}\n\n"

    for _, row in hormone_data.iterrows():
        direction = "aumentó" if row['difference'] > 0 else "disminuyó"
        summary += f"- **{row['metric'].replace('_', ' ').title()}** {direction} "
        summary += f"{abs(row['percent_change']):.1f}% "
        summary += f"(Δ={row['difference']:+.3f}, d={row['cohens_d']:.2f}, p={row['p_value']:.4f})\n"

summary += """

## Interpretación General

"""

# Contar efectos por hormona
effects_by_hormone = effects_df[effects_df['p_value'] < 0.05].groupby('hormone').size()

if len(effects_by_hormone) > 0:
    summary += "Número de efectos significativos por hormona:\n\n"
    for hormone, count in effects_by_hormone.sort_values(ascending=False).items():
        summary += f"- **{hormone.title()}**: {count} métricas afectadas\n"
else:
    summary += "No se detectaron efectos significativos de las hormonas individuales.\n"

summary += """

## Figuras Generadas

1. `hormone_effects_barplot.png` - Diferencias por hormona
2. `hormone_effects_heatmap.png` - Mapa de calor de efectos
3. `hormone_effects_boxplots.png` - Distribuciones comparativas

## Datos

- `hormone_effects_summary.csv` - Resultados completos
- `isolated_hormones_table.tex` - Tabla para LaTeX

---

**Conclusión:** Este análisis permite identificar el efecto específico de cada
hormona individual sobre diferentes aspectos de la generación de texto.
"""

with open(OUTPUT_DIR / "isolated_hormones_summary.md", 'w') as f:
    f.write(summary)
print(f"    {OUTPUT_DIR / 'isolated_hormones_summary.md'}")

print("\n" + "="*80)
print(" ANÁLISIS DE HORMONAS INDIVIDUALES COMPLETADO")
print("="*80)

print("\n Resumen de efectos significativos:")
sig_effects = effects_df[effects_df['p_value'] < 0.05]
print(f"   Total efectos significativos: {len(sig_effects)}")

if len(sig_effects) > 0:
    print("\n   Por hormona:")
    for hormone, count in sig_effects.groupby('hormone').size().sort_values(ascending=False).items():
        print(f"      {hormone.title()}: {count}")
