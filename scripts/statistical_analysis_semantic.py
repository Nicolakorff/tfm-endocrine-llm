"""
Análisis Estadístico Completo: 
Compara Sesgo Simple vs Sesgo Semántico
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUTPUT_DIR = Path("data/results/semantic_comparison/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cargar datos
df = pd.read_csv("data/results/semantic_comparison/comparison_results.csv")

print("="*80)
print("ANÁLISIS ESTADÍSTICO COMPLETO")
print("="*80 + "\n")

# 1. ESTADÍSTICAS DESCRIPTIVAS
print("1. ESTADÍSTICAS DESCRIPTIVAS")
print("-"*80)

desc_stats = df.groupby('condition').agg({
    'distinct_2': ['count', 'mean', 'std', 'min', 'max'],
    'sentiment_polarity': ['mean', 'std'],
    'repetition_rate': ['mean', 'std'],
    'length': ['mean', 'std']
}).round(4)

print(desc_stats)
desc_stats.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")

# 2. TESTS DE NORMALIDAD
print("\n2. TESTS DE NORMALIDAD (Shapiro-Wilk)")
print("-"*80)

for condition in df['condition'].unique():
    data = df[df['condition'] == condition]['distinct_2'].dropna()
    stat, p = stats.shapiro(data)
    print(f"{condition}: W={stat:.4f}, p={p:.4f} {'(normal)' if p > 0.05 else '(no normal)'}")

# 3. TESTS DE COMPARACIÓN
print("\n3. TESTS T DE STUDENT (independiente)")
print("-"*80)

simple = df[df['condition'] == 'simple_bias']
semantic = df[df['condition'] == 'semantic_bias']

results_table = []

for metric in ['distinct_2', 'sentiment_polarity', 'repetition_rate', 'length']:
    simple_vals = simple[metric].dropna()
    semantic_vals = semantic[metric].dropna()

    # T-test
    t_stat, p_value = stats.ttest_ind(simple_vals, semantic_vals)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((simple_vals.var() + semantic_vals.var()) / 2)
    cohens_d = (semantic_vals.mean() - simple_vals.mean()) / pooled_std

    # Interpretar effect size
    if abs(cohens_d) < 0.2:
        effect = "trivial"
    elif abs(cohens_d) < 0.5:
        effect = "pequeño"
    elif abs(cohens_d) < 0.8:
        effect = "mediano"
    else:
        effect = "grande"

    print(f"\n{metric}:")
    print(f"M_simple={simple_vals.mean():.4f}, SD={simple_vals.std():.4f}")
    print(f"M_semantic={semantic_vals.mean():.4f}, SD={semantic_vals.std():.4f}")
    print(f"Diferencia: {semantic_vals.mean() - simple_vals.mean():+.4f}")
    print(f"t({len(simple_vals)+len(semantic_vals)-2})={t_stat:.3f}, p={p_value:.4f}")
    print(f"Cohen's d={cohens_d:.3f} (efecto {effect})")

    results_table.append({
        'metric': metric,
        'M_simple': simple_vals.mean(),
        'SD_simple': simple_vals.std(),
        'M_semantic': semantic_vals.mean(),
        'SD_semantic': semantic_vals.std(),
        'difference': semantic_vals.mean() - simple_vals.mean(),
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'effect_size': effect,
        'significant': 'Yes' if p_value < 0.05 else 'No'
    })

# Guardar tabla de resultados
results_df = pd.DataFrame(results_table)
results_df.to_csv(OUTPUT_DIR / "statistical_tests.csv", index=False)

# 4. CREAR FIGURA PARA TFM
print("\n4. GENERANDO FIGURA PARA TFM...")
print("-"*80)

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Título principal
fig.suptitle('Comparación: Sesgo Simple vs Sesgo Semántico',
             fontsize=18, fontweight='bold')

# Plot 1: Diversidad Léxica (principal)
ax1 = fig.add_subplot(gs[0, :2])
sns.violinplot(data=df, x='condition', y='distinct_2', ax=ax1, inner='box')
ax1.set_title('Diversidad Léxica', fontsize=14, fontweight='bold')
ax1.set_xlabel('Tipo de Sesgo')
ax1.set_ylabel('Distinct-2')
ax1.set_xticklabels(['Simple', 'Semántico'])

# Añadir significancia
simple_d2 = simple['distinct_2'].values
semantic_d2 = semantic['distinct_2'].values
_, p = stats.ttest_ind(simple_d2, semantic_d2)
y_max = max(simple_d2.max(), semantic_d2.max())
ax1.plot([0, 1], [y_max*1.05, y_max*1.05], 'k-', lw=1.5)
sig_text = '' if p < 0.001 else '' if p < 0.01 else '' if p < 0.05 else 'ns'
ax1.text(0.5, y_max*1.07, sig_text, ha='center', fontsize=12)

# Plot 2: Sentimiento
ax2 = fig.add_subplot(gs[0, 2])
sns.boxplot(data=df, x='condition', y='sentiment_polarity', ax=ax2)
ax2.set_title('(B) Polaridad', fontsize=12, fontweight='bold')
ax2.set_xlabel('')
ax2.set_xticklabels(['Simple', 'Sem.'], rotation=45)

# Plot 3: Repetición
ax3 = fig.add_subplot(gs[1, 0])
sns.boxplot(data=df, x='condition', y='repetition_rate', ax=ax3)
ax3.set_title('(C) Repetición', fontsize=12, fontweight='bold')
ax3.set_xlabel('')
ax3.set_xticklabels(['Simple', 'Sem.'], rotation=45)

# Plot 4: Longitud
ax4 = fig.add_subplot(gs[1, 1])
sns.boxplot(data=df, x='condition', y='length', ax=ax4)
ax4.set_title('(D) Longitud', fontsize=12, fontweight='bold')
ax4.set_xlabel('')
ax4.set_xticklabels(['Simple', 'Sem.'], rotation=45)

# Plot 5: Activación semántica
if 'semantic_activation_empathy' in df.columns:
    ax5 = fig.add_subplot(gs[1, 2])
    semantic_only = df[df['condition'] == 'semantic_bias']
    ax5.hist(
        semantic_only['semantic_activation_empathy'],
        bins=15,
        alpha=0.7,
        color='coral',
        edgecolor='black'
    )
    ax5.axvline(semantic_only['semantic_activation_empathy'].mean(),
                color='red', linestyle='--', linewidth=2,
                label=f"M={semantic_only['semantic_activation_empathy'].mean():.2f}")
    ax5.set_title('(E) Activación Empathy', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Similitud Semántica')
    ax5.legend()

# Plot 6: Por categoría de prompt
ax6 = fig.add_subplot(gs[2, :])
cat_comparison = df.groupby(['prompt_category', 'condition'])['distinct_2'].mean().unstack()
cat_comparison.plot(kind='bar', ax=ax6, alpha=0.8, width=0.7)
ax6.set_title('(F) Diversidad Léxica por Categoría de Prompt', fontsize=12, fontweight='bold')
ax6.set_xlabel('Categoría de Prompt')
ax6.set_ylabel('Distinct-2 (promedio)')
ax6.legend(title='Tipo', labels=['Simple', 'Semántico'])
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
ax6.grid(axis='y', alpha=0.3)

plt.savefig(
    OUTPUT_DIR / 'figure_semantic_comparison.png',
    dpi=300,
    bbox_inches='tight'
)
print(f"Figura guardada: {OUTPUT_DIR / 'figure_semantic_comparison.png'}")

plt.savefig(
    OUTPUT_DIR / 'figure_semantic_comparison.pdf',
    bbox_inches='tight'
)
print(f"PDF guardado: {OUTPUT_DIR / 'figure_semantic_comparison.pdf'}")

print("\n" + "="*80)
print("Análisis estadístico completado")
print("="*80)
