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
simple_d2 = df[df['condition']=='simple_bias']['distinct_2'].mean()
semantic_d2 = df[df['condition']=='semantic_bias']['distinct_2'].mean()
diff = semantic_d2 - simple_d2

print(f"\n3. Diferencia en Distinct-2: {diff:+.4f}")

if abs(diff) > 0.01:
    print("   Diferencia detectada")
else:
    print("   Diferencia muy pequeña - revisar configuración")

# Verificar activación semántica
if 'semantic_activation_empathy' in df.columns:
    semantic_only = df[df['condition']=='semantic_bias']
    print(f"\n4. Activación semántica promedio: {semantic_only['semantic_activation_empathy'].mean():.3f}")
    
    if semantic_only['semantic_activation_empathy'].mean() > 0.3:
        print("   Activación semántica detectada")
    else:
        print("   Activación semántica baja")

print("\n" + "="*60)
print(" Verificación rápida completada")