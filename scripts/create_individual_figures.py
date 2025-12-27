"""
Crea figuras individuales de alta calidad para cada sección del TFM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

print("="*80)
print("CREANDO FIGURAS INDIVIDUALES PARA TFM")
print("="*80 + "\n")

DATA_DIR = Path("data/results")
OUTPUT_DIR = DATA_DIR / "tfm_figures/individual"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Estilo
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# 1. FIGURA: ARQUITECTURA DEL SISTEMA
print("1. Creando diagrama de arquitectura del sistema...")

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Título
ax.text(0.5, 0.95, 'Arquitectura del Sistema de Neuromodulación Endocrina',
        ha='center', va='top', fontsize=16, fontweight='bold')

# Modelo base
rect_model = plt.Rectangle((0.35, 0.7), 0.3, 0.15, 
                           facecolor='lightblue', edgecolor='black', linewidth=2)
ax.add_patch(rect_model)
ax.text(0.5, 0.775, 'Modelo de Lenguaje\n(GPT-2)', 
       ha='center', va='center', fontsize=11, fontweight='bold')

# Vector hormonal
rect_hormones = plt.Rectangle((0.1, 0.4), 0.8, 0.2,
                              facecolor='lightcoral', edgecolor='black', linewidth=2)
ax.add_patch(rect_hormones)
ax.text(0.5, 0.55, 'Vector Hormonal H = [d, c, o, a, s]',
       ha='center', va='center', fontsize=12, fontweight='bold')

# Hormonas individuales
hormones = ['Dopamina\n(exploración)', 'Cortisol\n(cautela)', 'Oxitocina\n(empatía)',
           'Adrenalina\n(activación)', 'Serotonina\n(estabilidad)']
x_positions = np.linspace(0.15, 0.85, 5)

for x, hormone in zip(x_positions, hormones):
    circle = plt.Circle((x, 0.5), 0.04, facecolor='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(circle)
    ax.text(x, 0.35, hormone, ha='center', va='top', fontsize=8)

# LogitsProcessor
rect_processor = plt.Rectangle((0.35, 0.15), 0.3, 0.15,
                               facecolor='lightgreen', edgecolor='black', linewidth=2)
ax.add_patch(rect_processor)
ax.text(0.5, 0.225, 'HormonalLogitsProcessor\n(Modulación)', 
       ha='center', va='center', fontsize=11, fontweight='bold')

# Flechas
ax.annotate('', xy=(0.5, 0.7), xytext=(0.5, 0.6),
           arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.annotate('', xy=(0.5, 0.4), xytext=(0.5, 0.3),
           arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Output
ax.text(0.5, 0.05, 'Texto Generado Modulado',
       ha='center', va='center', fontsize=12, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black', linewidth=2))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig1_architecture.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'fig1_architecture.pdf', bbox_inches='tight')
plt.close()
print(f"fig1_architecture.png/pdf")

# 2. FIGURA: PERFILES HORMONALES
print("\n2. Creando visualización de perfiles hormonales...")

from endocrine_llm import HORMONE_PROFILES

# Seleccionar perfiles clave
profiles_to_plot = ['baseline', 'empathic', 'creative', 'cautious', 'euphoric', 'stressed']
hormones = ['dopamine', 'cortisol', 'oxytocin', 'adrenaline', 'serotonin']

# Datos
data = []
for profile_name in profiles_to_plot:
    if profile_name in HORMONE_PROFILES:
        profile = HORMONE_PROFILES[profile_name]
        for hormone in hormones:
            data.append({
                'profile': profile_name,
                'hormone': hormone,
                'level': getattr(profile, hormone)
            })

df_profiles = pd.DataFrame(data)

# Heatmap
fig, ax = plt.subplots(figsize=(10, 6))

pivot = df_profiles.pivot(index='profile', columns='hormone', values='level')
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', 
           linewidths=1, cbar_kws={'label': 'Nivel Hormonal'},
           vmin=0, vmax=1, ax=ax)

ax.set_title('Perfiles Hormonales Predefinidos', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Hormona', fontsize=12, fontweight='bold')
ax.set_ylabel('Perfil', fontsize=12, fontweight='bold')
ax.set_yticklabels([p.title() for p in pivot.index], rotation=0)
ax.set_xticklabels([h.title() for h in pivot.columns], rotation=45, ha='right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig2_hormone_profiles.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'fig2_hormone_profiles.pdf', bbox_inches='tight')
plt.close()
print(f"fig2_hormone_profiles.png/pdf")

# 3. FIGURA: EJEMPLO DE GENERACIÓN
print("\n3. Creando figura de ejemplo comparativo...")

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Título
ax.text(0.5, 0.95, 'Ejemplo de Generación: Efecto de Perfil Hormonal',
       ha='center', va='top', fontsize=14, fontweight='bold')

# Prompt
prompt_text = "I'm feeling anxious about my exam tomorrow."
ax.text(0.5, 0.88, f'Prompt: "{prompt_text}"',
       ha='center', va='top', fontsize=11, style='italic',
       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

# Ejemplos (placeholder - reemplazar con tus datos reales)
examples = [
    ("Baseline", "I'm feeling anxious about my exam tomorrow. I think I should study more and try to stay calm..."),
    ("Empathic", "I'm feeling anxious about my exam tomorrow. I understand that feeling. It's completely normal to feel nervous..."),
    ("Creative", "I'm feeling anxious about my exam tomorrow. Imagine your anxiety as a butterfly, fluttering...")
]

y_start = 0.75
y_step = 0.22

for i, (profile, text) in enumerate(examples):
    y_pos = y_start - (i * y_step)

    # Caja de perfil
    rect = plt.Rectangle((0.05, y_pos - 0.15), 0.9, 0.18,
                         facecolor=['lightblue', 'lightcoral', 'lightgreen'][i],
                         edgecolor='black', linewidth=2, alpha=0.3)
    ax.add_patch(rect)

    # Nombre del perfil
    ax.text(0.08, y_pos - 0.02, f'{profile}:',
           ha='left', va='top', fontsize=11, fontweight='bold')

    # Texto generado
    ax.text(0.08, y_pos - 0.05, text,
           ha='left', va='top', fontsize=9, wrap=True)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig3_generation_example.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'fig3_generation_example.pdf', bbox_inches='tight')
plt.close()
print(f"fig3_generation_example.png/pdf")

# 4. FIGURA: DISTRIBUCIÓN DE MÉTRICAS
print("\n4. Creando figura de distribución de métricas...")

df_all = pd.read_csv(DATA_DIR / "consolidated/all_experiments_consolidated.csv")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribución de Métricas Lingüísticas (Todos los Experimentos)',
            fontsize=14, fontweight='bold')

metrics_info = [
    ('distinct_2', 'Diversidad Léxica (Distinct-2)', (0, 1)),
    ('sentiment_polarity', 'Polaridad de Sentimiento', (-1, 1)),
    ('repetition_rate', 'Tasa de Repetición', (0, 1)),
    ('length', 'Longitud (tokens)', (0, None))
]

for ax, (metric, title, xlim) in zip(axes.flat, metrics_info):
    data = df_all[metric].dropna()

    ax.hist(data, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {data.mean():.3f}')
    ax.axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {data.median():.3f}')

    ax.set_xlabel(title, fontsize=11)
    ax.set_ylabel('Frecuencia', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    if xlim[1] is not None:
        ax.set_xlim(xlim)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig4_metrics_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'fig4_metrics_distribution.pdf', bbox_inches='tight')
plt.close()
print(f"fig4_metrics_distribution.png/pdf")

# 5. FIGURA: COMPARACIÓN TOP PERFILES
print("\n5. Creando comparación de top perfiles...")

# Top 6 perfiles por frecuencia
top6 = df_all['profile_name'].value_counts().head(6).index.tolist()
df_top6 = df_all[df_all['profile_name'].isin(top6)]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Comparación de Perfiles Hormonales Principales',
            fontsize=14, fontweight='bold')

metrics_plot = ['distinct_2', 'sentiment_polarity', 'repetition_rate', 'length']
titles = ['Diversidad Léxica', 'Polaridad', 'Repetición', 'Longitud']

for ax, metric, title in zip(axes.flat, metrics_plot, titles):
    sns.boxplot(data=df_top6, x='profile_name', y=metric, ax=ax, palette='Set2')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel(title, fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig5_top_profiles_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'fig5_top_profiles_comparison.pdf', bbox_inches='tight')
plt.close()
print(f"fig5_top_profiles_comparison.png/pdf")

print("\n" + "="*80)
print("FIGURAS INDIVIDUALES CREADAS")
print("="*80)
print("\n Total figuras generadas: 5")
print(f"Ubicación: {OUTPUT_DIR}")
