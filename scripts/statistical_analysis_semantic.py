"""
Análisis Estadístico Completo: Léxico vs Semántico
==================================================

"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

RESULTS_FILE = Path("data/results/semantic_comparison/comparison_results.csv")
OUTPUT_DIR = Path("data/results/semantic_comparison/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05  # Nivel de significancia

print("="*80)
print("ANÁLISIS ESTADÍSTICO: LÉXICO VS SEMÁNTICO")
print("="*80)
print()

# ============================================================================
# CARGAR DATOS
# ============================================================================

print("Cargando resultados...")
df = pd.read_csv(RESULTS_FILE)

print(f"Total generaciones: {len(df)}")
print(f"Condiciones: {df['condition'].unique().tolist()}")
print()

# ============================================================================
# 1. ESTADÍSTICAS DESCRIPTIVAS
# ============================================================================

print("1. ESTADÍSTICAS DESCRIPTIVAS")
print("-" * 80)

metrics = ['distinct_2', 'sentiment_polarity', 'repetition_rate', 'length']

desc_stats = df.groupby('condition')[metrics].describe().round(4)
print(desc_stats)
print()

# ============================================================================
# 2. ANOVA DE UNA VÍA
# ============================================================================

print("2. ANOVA DE UNA VÍA")
print("-" * 80)

anova_results = {}

for metric in metrics:
    print(f"\nMétrica: {metric}")
    print("-" * 40)
    
    # Preparar datos por grupo
    groups = [
        df[df['condition'] == cond][metric].dropna().values
        for cond in df['condition'].unique()
    ]
    
    # ANOVA
    f_stat, p_value = f_oneway(*groups)
    
    # Effect size (eta-squared)
    # η² = SS_between / SS_total
    grand_mean = df[metric].mean()
    ss_between = sum(
        len(df[df['condition'] == cond]) * 
        (df[df['condition'] == cond][metric].mean() - grand_mean)**2
        for cond in df['condition'].unique()
    )
    ss_total = sum((df[metric] - grand_mean)**2)
    eta_squared = ss_between / ss_total
    
    # Interpretar effect size
    if eta_squared < 0.01:
        effect_interpretation = "despreciable"
    elif eta_squared < 0.06:
        effect_interpretation = "pequeño"
    elif eta_squared < 0.14:
        effect_interpretation = "mediano"
    else:
        effect_interpretation = "grande"
    
    print(f"F({len(groups)-1}, {len(df)-len(groups)}) = {f_stat:.3f}")
    print(f"p-value = {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
    print(f"η² = {eta_squared:.4f} ({effect_interpretation})")
    
    anova_results[metric] = {
        'F': f_stat,
        'p': p_value,
        'eta_squared': eta_squared,
        'significant': p_value < ALPHA
    }

print()

# ============================================================================
# 3. POST-HOC TESTS (TUKEY HSD)
# ============================================================================

print("3. POST-HOC TESTS (Tukey HSD)")
print("-" * 80)

posthoc_results = {}

for metric in metrics:
    if anova_results[metric]['significant']:
        print(f"\nMétrica: {metric}")
        print("-" * 40)
        
        # Tukey HSD
        tukey = pairwise_tukeyhsd(
            endog=df[metric].dropna(),
            groups=df.loc[df[metric].notna(), 'condition'],
            alpha=ALPHA
        )
        
        print(tukey)
        posthoc_results[metric] = tukey
    else:
        print(f"\n{metric}: No se realizan post-hoc (ANOVA no significativa)")

print()

# ============================================================================
# 4. COMPARACIONES ESPECÍFICAS DE INTERÉS
# ============================================================================

print("4. COMPARACIONES ESPECÍFICAS")
print("-" * 80)

comparisons = [
    ('baseline', 'lexical_empathy', 'Efecto del sesgo léxico'),
    ('lexical_empathy', 'semantic_empathy', 'Efecto del sesgo semántico'),
    ('semantic_empathy', 'semantic_creativity', 'Empathy vs Creativity'),
    ('semantic_empathy', 'semantic_caution', 'Empathy vs Caution'),
]

comparison_results = []

for cond1, cond2, description in comparisons:
    print(f"\n{description}: {cond1} vs {cond2}")
    print("-" * 40)
    
    data1 = df[df['condition'] == cond1]
    data2 = df[df['condition'] == cond2]
    
    for metric in metrics:
        vals1 = data1[metric].dropna()
        vals2 = data2[metric].dropna()
        
        if len(vals1) > 0 and len(vals2) > 0:
            # t-test
            t_stat, p_value = stats.ttest_ind(vals1, vals2)
            
            # Cohen's d
            pooled_std = np.sqrt((vals1.std()**2 + vals2.std()**2) / 2)
            cohens_d = (vals2.mean() - vals1.mean()) / pooled_std
            
            # Interpretar Cohen's d
            if abs(cohens_d) < 0.2:
                effect_size = "trivial"
            elif abs(cohens_d) < 0.5:
                effect_size = "pequeño"
            elif abs(cohens_d) < 0.8:
                effect_size = "mediano"
            else:
                effect_size = "grande"
            
            print(f"\n  {metric}:")
            print(f"    {cond1}: M={vals1.mean():.4f}, SD={vals1.std():.4f}")
            print(f"    {cond2}: M={vals2.mean():.4f}, SD={vals2.std():.4f}")
            print(f"    Δ = {vals2.mean() - vals1.mean():+.4f}")
            print(f"    t({len(vals1)+len(vals2)-2}) = {t_stat:.3f}, p = {p_value:.4f}")
            print(f"    d = {cohens_d:.3f} ({effect_size})")
            
            comparison_results.append({
                'comparison': description,
                'cond1': cond1,
                'cond2': cond2,
                'metric': metric,
                'mean1': vals1.mean(),
                'mean2': vals2.mean(),
                'diff': vals2.mean() - vals1.mean(),
                't': t_stat,
                'p': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < ALPHA
            })

# Guardar comparaciones
comparison_df = pd.DataFrame(comparison_results)
comparison_df.to_csv(OUTPUT_DIR / "pairwise_comparisons.csv", index=False)
print(f"\n✓ Comparaciones guardadas en: {OUTPUT_DIR / 'pairwise_comparisons.csv'}")

print()

# ============================================================================
# 5. ANÁLISIS DE COHERENCIA SEMÁNTICA
# ============================================================================

print("5. ANÁLISIS DE COHERENCIA SEMÁNTICA")
print("-" * 80)

semantic_conditions = df[df['semantic_coherence'].notna()]

if len(semantic_conditions) > 0:
    print("\nCoherencia semántica por condición:")
    coherence_stats = semantic_conditions.groupby('condition')['semantic_coherence'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ])
    print(coherence_stats.round(3))
    
    # ANOVA para coherencia
    groups = [
        semantic_conditions[semantic_conditions['condition'] == cond]['semantic_coherence'].dropna().values
        for cond in semantic_conditions['condition'].unique()
    ]
    
    f_stat, p_value = f_oneway(*groups)
    print(f"\nANOVA coherencia semántica:")
    print(f"F = {f_stat:.3f}, p = {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
else:
    print("No hay datos de coherencia semántica")

print()

# ============================================================================
# 6. ACTIVACIÓN SEMÁNTICA
# ============================================================================

print("6. ANÁLISIS DE ACTIVACIÓN SEMÁNTICA")
print("-" * 80)

semantic_activation_cols = [
    'semantic_activation_empathy',
    'semantic_activation_creativity',
    'semantic_activation_caution'
]

print("\nActivación promedio por condición:")
activation_summary = df.groupby('condition')[semantic_activation_cols].mean()
print(activation_summary.round(3))

# Verificar si el sesgo semántico aumenta la activación correspondiente
print("\nVerificación de efectividad del sesgo semántico:")

semantic_conditions_map = {
    'semantic_empathy': 'semantic_activation_empathy',
    'semantic_creativity': 'semantic_activation_creativity',
    'semantic_caution': 'semantic_activation_caution'
}

for condition, activation_col in semantic_conditions_map.items():
    if condition in df['condition'].values:
        semantic_data = df[df['condition'] == condition][activation_col].dropna()
        lexical_data = df[df['condition'] == 'lexical_empathy'][activation_col].dropna()
        
        if len(semantic_data) > 0 and len(lexical_data) > 0:
            t_stat, p_value = stats.ttest_ind(semantic_data, lexical_data)
            cohens_d = (semantic_data.mean() - lexical_data.mean()) / np.sqrt(
                (semantic_data.std()**2 + lexical_data.std()**2) / 2
            )
            
            print(f"\n{condition} vs lexical_empathy en {activation_col}:")
            print(f"  Semántico: M={semantic_data.mean():.3f}")
            print(f"  Léxico: M={lexical_data.mean():.3f}")
            print(f"  Incremento: {semantic_data.mean() - lexical_data.mean():+.3f}")
            print(f"  t={t_stat:.3f}, p={p_value:.4f}, d={cohens_d:.3f}")

print()

# ============================================================================
# 7. RESUMEN EJECUTIVO
# ============================================================================

print("7. RESUMEN EJECUTIVO")
print("=" * 80)

significant_effects = {k: v for k, v in anova_results.items() if v['significant']}

if len(significant_effects) > 0:
    print("\n✓ Efectos significativos encontrados:")
    for metric, results in significant_effects.items():
        print(f"  - {metric}: F={results['F']:.3f}, p={results['p']:.4f}, η²={results['eta_squared']:.4f}")
else:
    print("\n⚠ No se encontraron efectos significativos en ninguna métrica")

print("\n Métricas clave:")
print(f"  - Alpha level: {ALPHA}")
print(f"  - Total comparaciones: {len(comparison_results)}")
print(f"  - Comparaciones significativas: {sum(1 for r in comparison_results if r['significant'])}")

print("\n Archivos generados:")
print(f"  - {OUTPUT_DIR / 'pairwise_comparisons.csv'}")

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)