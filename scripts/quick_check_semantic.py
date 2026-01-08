"""Quick check de resultados semánticos"""

import pandas as pd

df = pd.read_csv("data/results/semantic_comparison/comparison_results.csv")

print("="*60)
print("VERIFICACIÓN RÁPIDA DE RESULTADOS")
print("="*60)

# Contar generaciones por condición
print("\n1. Generaciones por condición:")
print(df['condition'].value_counts())

# Verificar si hay diferencias
print("\n2. Métricas promedio:")
comparison = df.groupby('condition')[['distinct_2', 'sentiment_polarity']].mean()
print(comparison)

# Diferencia absoluta
baseline_d2 = df[df['condition']=='baseline']['distinct_2'].mean()
lexical_d2 = df[df['condition']=='lexical_empathy']['distinct_2'].mean()
semantic_conditions = df[df['condition'].str.startswith('semantic_')]
semantic_d2 = semantic_conditions['distinct_2'].mean()

print(f"\n3. Diferencia Baseline vs Léxico: {lexical_d2 - baseline_d2:+.4f}")
print(f"   Diferencia Baseline vs Semántico: {semantic_d2 - baseline_d2:+.4f}")

# Verificar activación semántica
if 'semantic_activation_empathy' in df.columns:
    semantic_only = df[df['condition'].str.startswith('semantic_')]
    print(f"\n4. Activación semántica promedio: {semantic_only['semantic_activation_empathy'].mean():.3f}")

    if semantic_only['semantic_activation_empathy'].mean() > 0.3:
        print("Activación semántica detectada")
    else:
        print("Activación semántica baja")

print("\n" + "="*60)
print(" Verificación rápida completada")
