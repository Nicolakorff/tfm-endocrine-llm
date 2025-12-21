# scripts/quick_analysis.py
"""
Análisis rápido de resultados de Fase 1 y 2
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Cargar resultados
phase1 = pd.read_csv("data/results/phase1_results.csv")
phase2 = pd.read_csv("data/results/phase2_results.csv")

print(" RESUMEN DE DATOS")
print(f"Fase 1: {len(phase1)} generaciones")
print(f"Fase 2: {len(phase2)} generaciones")

# Combinar
all_data = pd.concat([phase1, phase2], ignore_index=True)

# Estadísticas por perfil
print("\n MÉTRICAS PROMEDIO POR PERFIL:")
metrics = ['distinct_2', 'sentiment_polarity', 'perplexity', 'repetition_rate']
summary = all_data.groupby('profile_name')[metrics].mean().round(3)
print(summary)

# Visualización rápida
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Análisis Rápido de Resultados', fontsize=16, fontweight='bold')

# Plot 1: Distinct-2
axes[0, 0].boxplot(
    [all_data[all_data['profile_name'] == p]['distinct_2'].dropna()
     for p in all_data['profile_name'].unique()],
    labels=all_data['profile_name'].unique())
axes[0, 0].set_title('Diversidad Léxica (Distinct-2)')
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Sentiment
axes[0, 1].boxplot(
    [all_data[all_data['profile_name'] == p]['sentiment_polarity'].dropna()
     for p in all_data['profile_name'].unique()],
    labels=all_data['profile_name'].unique())
axes[0, 1].set_title('Polaridad de Sentimiento')
axes[0, 1].tick_params(axis='x', rotation=45)

# Plot 3: Perplexity
axes[1, 0].boxplot(
    [all_data[all_data['profile_name'] == p]['perplexity'].dropna()
     for p in all_data['profile_name'].unique()],
    labels=all_data['profile_name'].unique())
axes[1, 0].set_title('Perplexity')
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot 4: Repetition
axes[1, 1].boxplot(
    [all_data[all_data['profile_name'] == p]['repetition_rate'].dropna()
     for p in all_data['profile_name'].unique()],
    labels=all_data['profile_name'].unique())
axes[1, 1].set_title('Tasa de Repetición')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('data/results/quick_analysis.png', dpi=300, bbox_inches='tight')
print("\n Visualización guardada: data/results/quick_analysis.png")

# Verificar que hay diferencias significativas
print("\n VERIFICACIÓN DE DIFERENCIAS:")
baseline = all_data[all_data['profile_name'] == 'baseline']['distinct_2'].mean()
empathic = all_data[all_data['profile_name'] == 'empathic']['distinct_2'].mean()
creative = all_data[all_data['profile_name'] == 'creative']['distinct_2'].mean()

print(f"Baseline distinct_2: {baseline:.3f}")
print(f"Empathic distinct_2: {empathic:.3f} (diff: {empathic-baseline:+.3f})")
print(f"Creative distinct_2: {creative:.3f} (diff: {creative-baseline:+.3f})")

if abs(empathic - baseline) > 0.01 or abs(creative - baseline) > 0.01:
    print("\n Se detectan diferencias entre perfiles")
else:
    print("\n Las diferencias son muy pequeñas - considera ajustar parámetros hormonales")
