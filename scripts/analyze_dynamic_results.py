"""
Implementa análisis estadístico del experimento dinámico
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("="*80)
print("ANÁLISIS DE RESULTADOS DINÁMICOS")
print("="*80 + "\n")

# Cargar datos
DATA_FILE = Path("data/results/phase3_dynamic_results.csv")
OUTPUT_DIR = Path("data/results/dynamic_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)
print(f"Cargados {len(df)} registros")

# Separar estático vs dinámico
static_df = df[df['is_dynamic'] == False]
dynamic_df = df[df['is_dynamic'] == True]

print(f"Estático: {len(static_df)}")
print(f"Dinámico: {len(dynamic_df)}")

# 1. COMPARACIÓN ESTÁTICO VS DINÁMICO
print("\n1. Comparación Estático vs Dinámico")
print("-" * 80)

metrics = ['distinct_2', 'repetition_rate', 'sentiment_polarity']

results = []
for metric in metrics:
    static_vals = static_df[metric].dropna()
    dynamic_vals = dynamic_df[metric].dropna()

    # T-test
    t_stat, p_val = stats.ttest_ind(static_vals, dynamic_vals)

    # Cohen's d
    pooled_std = np.sqrt((static_vals.std()**2 + dynamic_vals.std()**2) / 2)
    cohens_d = (dynamic_vals.mean() - static_vals.mean()) / pooled_std

    results.append({
        'metric': metric,
        'static_mean': static_vals.mean(),
        'static_std': static_vals.std(),
        'dynamic_mean': dynamic_vals.mean(),
        'dynamic_std': dynamic_vals.std(),
        'difference': dynamic_vals.mean() - static_vals.mean(),
        't_statistic': t_stat,
        'p_value': p_val,
        'cohens_d': cohens_d,
        'significant': 'Yes' if p_val < 0.05 else 'No'
    })

    print(f"\n{metric}:")
    print(f"Estático:  {static_vals.mean():.4f} ± {static_vals.std():.4f}")
    print(f"Dinámico:  {dynamic_vals.mean():.4f} ± {dynamic_vals.std():.4f}")
    print(f"Diferencia: {dynamic_vals.mean() - static_vals.mean():+.4f}")
    print(f"t = {t_stat:.3f}, p = {p_val:.4f}, d = {cohens_d:.3f}")
    print(f"Significativo: {results[-1]['significant']}")

# Guardar resultados
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "static_vs_dynamic_comparison.csv", index=False)
print("\n Guardado: static_vs_dynamic_comparison.csv")

# 2. ANÁLISIS DE CAMBIOS HORMONALES
print("\n2. Análisis de Cambios Hormonales")
print("-" * 80)

hormones = ['dopamine', 'cortisol', 'oxytocin', 'adrenaline', 'serotonin']

for hormone in hormones:
    delta_col = f'delta_{hormone}'
    if delta_col in dynamic_df.columns:
        values = dynamic_df[delta_col].dropna()
        print(f"\n{hormone.capitalize()}:")
        print(f"  Media: {values.mean():+.4f}")
        print(f"  SD:    {values.std():.4f}")
        print(f"  Rango: [{values.min():+.4f}, {values.max():+.4f}]")

# 3. VISUALIZACIONES
print("\n3. Creando visualizaciones...")

# Figura 1: Comparación Boxplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, metric in enumerate(metrics):
    ax = axes[idx]

    data_plot = pd.DataFrame({
        'Estático': static_df[metric],
        'Dinámico': dynamic_df[metric]
    })

    bp = data_plot.boxplot(ax=ax, return_type='dict', patch_artist=True)

    # Colorear
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')

    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "static_vs_dynamic_boxplots.png", dpi=300, bbox_inches='tight')
print("Guardado: static_vs_dynamic_boxplots.png")
plt.close()

# Figura 2: Distribución de cambios hormonales
if 'total_hormone_change' in dynamic_df.columns:
    plt.figure(figsize=(10, 6))

    changes = dynamic_df['total_hormone_change'].dropna()

    plt.hist(changes, bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
    plt.axvline(changes.mean(), color='darkred', linestyle='--', linewidth=2,
                label=f'Media: {changes.mean():.4f}')

    plt.xlabel('Cambio Hormonal Total', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.title('Distribución de Cambios Hormonales (Sistema Dinámico)', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hormone_change_distribution.png", dpi=300, bbox_inches='tight')
    print("Guardado: hormone_change_distribution.png")
    plt.close()

# Figura 3: Cambios por categoría
if 'category' in dynamic_df.columns and 'total_hormone_change' in dynamic_df.columns:
    plt.figure(figsize=(10, 6))

    categories = dynamic_df['category'].unique()
    data_by_cat = [dynamic_df[dynamic_df['category'] == cat]['total_hormone_change'].dropna() 
                   for cat in categories]

    bp = plt.boxplot(data_by_cat, labels=categories, patch_artist=True)

    for patch in bp['boxes']:
        patch.set_facecolor('#e74c3c')
        patch.set_alpha(0.7)

    plt.ylabel('Cambio Hormonal Total', fontsize=12)
    plt.xlabel('Categoría de Prompt', fontsize=12)
    plt.title('Cambios Hormonales por Categoría de Prompt', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hormone_changes_by_category.png", dpi=300, bbox_inches='tight')
    print("Guardado: hormone_changes_by_category.png")
    plt.close()

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)
print(f"\n Resultados guardados en: {OUTPUT_DIR}")
print("\n Archivos generados:")
print("  - static_vs_dynamic_comparison.csv")
print("  - static_vs_dynamic_boxplots.png")
print("  - hormone_change_distribution.png")
print("  - hormone_changes_by_category.png")
