"""
Análisis rápido de resultados de Fase 1 y 2
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Obtener el directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).parent.parent

# Cargar resultados
phase1 = pd.read_csv(PROJECT_ROOT / "data/results/phase1/phase1_results.csv")
phase2 = pd.read_csv(PROJECT_ROOT / "data/results/phase2/phase2_results.csv")

print("RESUMEN DE DATOS")
print(f"Fase 1: {len(phase1)} generaciones")
print(f"Fase 2: {len(phase2)} generaciones")

# phase 1
phase1['profile_name'] = phase1['profile_name'].astype('category')
# Estadísticas por perfil
print("\n MÉTRICAS PROMEDIO POR PERFIL:")
metrics = ['distinct_2', 'sentiment_polarity', 'perplexity', 'repetition_rate']
summary = phase1.groupby('profile_name')[metrics].mean().round(3)
print(summary)

# Visualización rápida
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Análisis Rápido de Resultados', fontsize=16, fontweight='bold')

# Plot 1: Distinct-2
axes[0, 0].boxplot(
    [phase1[phase1['profile_name'] == p]['distinct_2'].dropna()
     for p in phase1['profile_name'].unique()],
    labels=phase1['profile_name'].unique())
axes[0, 0].set_title('Diversidad Léxica (Distinct-2)')
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Sentiment
axes[0, 1].boxplot(
    [phase1[phase1['profile_name'] == p]['sentiment_polarity'].dropna()
     for p in phase1['profile_name'].unique()],
    labels=phase1['profile_name'].unique())
axes[0, 1].set_title('Polaridad de Sentimiento')
axes[0, 1].tick_params(axis='x', rotation=45)

# Plot 3: Perplexity
axes[1, 0].boxplot(
    [phase1[phase1['profile_name'] == p]['perplexity'].dropna()
     for p in phase1['profile_name'].unique()],
    labels=phase1['profile_name'].unique())
axes[1, 0].set_title('Perplexity')
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot 4: Repetition
axes[1, 1].boxplot(
    [phase1[phase1['profile_name'] == p]['repetition_rate'].dropna()
     for p in phase1['profile_name'].unique()],
    labels=phase1['profile_name'].unique())
axes[1, 1].set_title('Tasa de Repetición')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('data/results/phase1/quick_analysis_phase1.png', dpi=300, bbox_inches='tight')
print("\n Visualización guardada: data/results/phase1/quick_analysis_phase1.png")

# Verificar que hay diferencias significativas
print("\n VERIFICACIÓN DE DIFERENCIAS:")
baseline = phase1[phase1['profile_name'] == 'baseline']['distinct_2'].mean()
empathic = phase1[phase1['profile_name'] == 'empathic']['distinct_2'].mean()
creative = phase1[phase1['profile_name'] == 'creative']['distinct_2'].mean()

print(f"Baseline distinct_2: {baseline:.3f}")
print(f"Empathic distinct_2: {empathic:.3f} (diff: {empathic-baseline:+.3f})")
print(f"Creative distinct_2: {creative:.3f} (diff: {creative-baseline:+.3f})")

if abs(empathic - baseline) > 0.01 or abs(creative - baseline) > 0.01:
    print("\n Se detectan diferencias entre perfiles")
else:
    print("\n Las diferencias son muy pequeñas - considera ajustar parámetros hormonales")

# phase 2
phase2['profile_name'] = phase2['profile_name'].astype('category')
# Estadísticas por perfil
print("\n MÉTRICAS PROMEDIO POR PERFIL:")
metrics = ['distinct_2', 'sentiment_polarity', 'perplexity', 'repetition_rate']
summary = phase2.groupby('profile_name')[metrics].mean().round(3)
print(summary)

# Visualización rápida
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Análisis Rápido de Resultados', fontsize=16, fontweight='bold')

# Plot 1: Distinct-2
axes[0, 0].boxplot(
    [phase2[phase2['profile_name'] == p]['distinct_2'].dropna()
     for p in phase2['profile_name'].unique()],
    labels=phase2['profile_name'].unique())
axes[0, 0].set_title('Diversidad Léxica (Distinct-2)')
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Sentiment
axes[0, 1].boxplot(
    [phase2[phase2['profile_name'] == p]['sentiment_polarity'].dropna()
     for p in phase2['profile_name'].unique()],
    labels=phase2['profile_name'].unique())
axes[0, 1].set_title('Polaridad de Sentimiento')
axes[0, 1].tick_params(axis='x', rotation=45)

# Plot 3: Perplexity
axes[1, 0].boxplot(
    [phase2[phase2['profile_name'] == p]['perplexity'].dropna()
     for p in phase2['profile_name'].unique()],
    labels=phase2['profile_name'].unique())
axes[1, 0].set_title('Perplexity')
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot 4: Repetition
axes[1, 1].boxplot(
    [phase2[phase2['profile_name'] == p]['repetition_rate'].dropna()
     for p in phase2['profile_name'].unique()],
    labels=phase2['profile_name'].unique())
axes[1, 1].set_title('Tasa de Repetición')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('data/results/phase2/quick_analysis_phase2.png', dpi=300, bbox_inches='tight')
print("\n Visualización guardada: data/results/phase2/quick_analysis_phase2.png")

# Verificar que hay diferencias significativas
print("\n VERIFICACIÓN DE DIFERENCIAS:")
baseline = phase2[phase2['profile_name'] == 'baseline']['distinct_2'].mean()
empathic = phase2[phase2['profile_name'] == 'empathic']['distinct_2'].mean()
creative = phase2[phase2['profile_name'] == 'creative']['distinct_2'].mean()

print(f"Baseline distinct_2: {baseline:.3f}")
print(f"Empathic distinct_2: {empathic:.3f} (diff: {empathic-baseline:+.3f})")
print(f"Creative distinct_2: {creative:.3f} (diff: {creative-baseline:+.3f})")

if abs(empathic - baseline) > 0.01 or abs(creative - baseline) > 0.01:
    print("\n Se detectan diferencias entre perfiles")
else:
    print("\n Las diferencias son muy pequeñas - considera ajustar parámetros hormonales")

