"""
Visualizaciones para Experimento Léxico vs Semántico
===================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

RESULTS_FILE = Path("data/results/semantic_comparison/comparison_results.csv")
OUTPUT_DIR = Path("data/results/semantic_comparison/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Estilo
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# Paleta de colores
COLORS = {
    'baseline': '#95a5a6',
    'lexical_empathy': '#3498db',
    'semantic_empathy': '#e74c3c',
    'semantic_creativity': '#f39c12',
    'semantic_caution': '#9b59b6'
}

print("="*80)
print("GENERANDO VISUALIZACIONES")
print("="*80)
print()

# ============================================================================
# CARGAR DATOS
# ============================================================================

print("Cargando datos...")
df = pd.read_csv(RESULTS_FILE)
print(f"Total generaciones: {len(df)}\n")

# ============================================================================
# FIGURA 1: MÉTRICAS PRINCIPALES POR CONDICIÓN
# ============================================================================

print("Generando Figura 1: Métricas principales...")

metrics = ['distinct_2', 'sentiment_polarity', 'repetition_rate', 'length']
metric_labels = {
    'distinct_2': 'Diversidad Léxica\n(Distinct-2)',
    'sentiment_polarity': 'Polaridad de\nSentimiento',
    'repetition_rate': 'Tasa de\nRepetición',
    'length': 'Longitud\n(tokens)'
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    
    # Box plot
    data_to_plot = [
        df[df['condition'] == cond][metric].dropna().values
        for cond in sorted(df['condition'].unique())
    ]
    
    bp = ax.boxplot(
        data_to_plot,
        labels=sorted(df['condition'].unique()),
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markersize=6)
    )
    
    # Colorear cajas
    for patch, condition in zip(bp['boxes'], sorted(df['condition'].unique())):
        patch.set_facecolor(COLORS[condition])
        patch.set_alpha(0.7)
    
    ax.set_ylabel(metric_labels[metric], fontsize=11, fontweight='bold')
    ax.set_xlabel('Condición', fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig1_metrics_by_condition.png', dpi=300, bbox_inches='tight')
print(f"✓ Guardada: {OUTPUT_DIR / 'fig1_metrics_by_condition.png'}\n")
plt.close()

# ============================================================================
# FIGURA 2: COMPARACIÓN DIRECTA LÉXICO VS SEMÁNTICO-EMPATHY
# ============================================================================

print("Generando Figura 2: Léxico vs Semántico-Empathy...")

lexical = df[df['condition'] == 'lexical_empathy']
semantic = df[df['condition'] == 'semantic_empathy']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics_to_compare = ['distinct_2', 'sentiment_polarity', 'repetition_rate']

for idx, metric in enumerate(metrics_to_compare):
    ax = axes[idx]
    
    # Preparar datos
    data = pd.DataFrame({
        'Léxico': lexical[metric].values,
        'Semántico': semantic[metric].values[:len(lexical)]  # Igualar tamaños
    })
    
    # Violin plot
    positions = [1, 2]
    parts = ax.violinplot(
        [data['Léxico'].dropna(), data['Semántico'].dropna()],
        positions=positions,
        showmeans=True,
        showmedians=True
    )
    
    # Colorear
    for pc, color in zip(parts['bodies'], ['#3498db', '#e74c3c']):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    
    # Estadísticos
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(
        data['Léxico'].dropna(),
        data['Semántico'].dropna()
    )
    
    # Añadir texto con resultado
    significance = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    ax.text(
        1.5, ax.get_ylim()[1] * 0.95,
        f't={t_stat:.2f}\np={p_value:.4f}\n{significance}',
        ha='center',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    ax.set_xticks(positions)
    ax.set_xticklabels(['Léxico', 'Semántico'])
    ax.set_ylabel(metric_labels[metric], fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig2_lexical_vs_semantic.png', dpi=300, bbox_inches='tight')
print(f"✓ Guardada: {OUTPUT_DIR / 'fig2_lexical_vs_semantic.png'}\n")
plt.close()

# ============================================================================
# FIGURA 3: ACTIVACIÓN SEMÁNTICA
# ============================================================================

print("Generando Figura 3: Activación semántica...")

semantic_cols = [
    'semantic_activation_empathy',
    'semantic_activation_creativity',
    'semantic_activation_caution'
]

# Preparar datos para heatmap
activation_data = df.groupby('condition')[semantic_cols].mean()

# Renombrar columnas
activation_data.columns = ['Empathy', 'Creativity', 'Caution']

# Crear heatmap
fig, ax = plt.subplots(figsize=(10, 6))

sns.heatmap(
    activation_data,
    annot=True,
    fmt='.3f',
    cmap='RdYlGn',
    center=0.25,
    vmin=0,
    vmax=0.5,
    cbar_kws={'label': 'Activación Semántica'},
    linewidths=1,
    ax=ax
)

ax.set_xlabel('Categoría Semántica', fontweight='bold')
ax.set_ylabel('Condición Experimental', fontweight='bold')
ax.set_title('Activación Semántica por Condición y Categoría', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig3_semantic_activation.png', dpi=300, bbox_inches='tight')
print(f"✓ Guardada: {OUTPUT_DIR / 'fig3_semantic_activation.png'}\n")
plt.close()

# ============================================================================
# FIGURA 4: COHERENCIA SEMÁNTICA
# ============================================================================

print("Generando Figura 4: Coherencia semántica...")

semantic_conditions = df[df['semantic_coherence'].notna()]

if len(semantic_conditions) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Box plot de coherencia
    data_to_plot = [
        semantic_conditions[semantic_conditions['condition'] == cond]['semantic_coherence'].values
        for cond in sorted(semantic_conditions['condition'].unique())
    ]
    
    bp = ax.boxplot(
        data_to_plot,
        labels=sorted(semantic_conditions['condition'].unique()),
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markersize=8)
    )
    
    # Colorear
    for patch, condition in zip(bp['boxes'], sorted(semantic_conditions['condition'].unique())):
        patch.set_facecolor(COLORS[condition])
        patch.set_alpha(0.7)
    
    # Línea de referencia
    ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, 
               label='Umbral esperado (0.3)')
    
    ax.set_ylabel('Coherencia Semántica\n(similitud con categoría objetivo)', 
                  fontweight='bold')
    ax.set_xlabel('Condición', fontweight='bold')
    ax.set_title('Coherencia Semántica: ¿El sesgo aumenta similitud con categoría?',
                 fontsize=13, fontweight='bold', pad=15)
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig4_semantic_coherence.png', dpi=300, bbox_inches='tight')
    print(f"✓ Guardada: {OUTPUT_DIR / 'fig4_semantic_coherence.png'}\n")
    plt.close()
else:
    print("⚠ No hay datos de coherencia semántica para visualizar\n")

# ============================================================================
# FIGURA 5: COMPARACIÓN ENTRE CATEGORÍAS SEMÁNTICAS
# ============================================================================

print("Generando Figura 5: Comparación entre categorías semánticas...")

semantic_only = df[df['condition'].str.startswith('semantic_')]

if len(semantic_only) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Distinct-2
    ax = axes[0]
    semantic_only.boxplot(
        column='distinct_2',
        by='condition',
        ax=ax,
        patch_artist=True
    )
    ax.set_xlabel('Categoría Semántica', fontweight='bold')
    ax.set_ylabel('Diversidad Léxica (Distinct-2)', fontweight='bold')
    ax.set_title('Diversidad Léxica por Categoría Semántica')
    plt.sca(ax)
    plt.xticks(rotation=45)
    
    # Sentiment
    ax = axes[1]
    semantic_only.boxplot(
        column='sentiment_polarity',
        by='condition',
        ax=ax,
        patch_artist=True
    )
    ax.set_xlabel('Categoría Semántica', fontweight='bold')
    ax.set_ylabel('Polaridad de Sentimiento', fontweight='bold')
    ax.set_title('Sentimiento por Categoría Semántica')
    plt.sca(ax)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig5_semantic_categories_comparison.png', 
                dpi=300, bbox_inches='tight')
    print(f"✓ Guardada: {OUTPUT_DIR / 'fig5_semantic_categories_comparison.png'}\n")
    plt.close()

# ============================================================================
# RESUMEN
# ============================================================================

print("="*80)
print("VISUALIZACIONES COMPLETADAS")
print("="*80)
print(f"\nFiguras guardadas en: {OUTPUT_DIR}")
print("\nFiguras generadas:")
print("  1. fig1_metrics_by_condition.png - Métricas por condición")
print("  2. fig2_lexical_vs_semantic.png - Comparación directa")
print("  3. fig3_semantic_activation.png - Heatmap de activación")
print("  4. fig4_semantic_coherence.png - Coherencia semántica")
print("  5. fig5_semantic_categories_comparison.png - Entre categorías")