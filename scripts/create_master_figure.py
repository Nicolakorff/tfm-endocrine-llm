"""
Crea la figura maestra que resume TODOS los resultados experimentales
para incluir en el TFM como figura principal.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

print("="*80)
print("🎨 CREANDO FIGURA MAESTRA PARA TFM")
print("="*80 + "\n")

# Configuración
DATA_DIR = Path("data/results")
OUTPUT_DIR = DATA_DIR / "tfm_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Estilo
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# 1. CARGAR TODOS LOS DATOS
print("1. Cargando datos...")

# Datos consolidados
df_all = pd.read_csv(DATA_DIR / "consolidated/all_experiments_consolidated.csv")
print(f"   Dataset consolidado: {len(df_all)} generaciones")

# Resultados ANOVA
anova_df = pd.read_csv(DATA_DIR / "anova_analysis/anova_summary.csv")
print(f"   Resultados ANOVA: {len(anova_df)} métricas")

# Comparación semántica
if (DATA_DIR / "semantic_comparison/comparison_results.csv").exists():
    df_semantic = pd.read_csv(DATA_DIR / "semantic_comparison/comparison_results.csv")
    has_semantic = True
    print(f"   Comparación semántica: {len(df_semantic)} generaciones")
else:
    has_semantic = False
    print("   Comparación semántica no disponible")

# Hormonas aisladas
if (DATA_DIR / "isolated_hormones_analysis/hormone_effects_summary.csv").exists():
    effects_df = pd.read_csv(DATA_DIR / "isolated_hormones_analysis/hormone_effects_summary.csv")
    has_isolated = True
    print(f"   Hormonas aisladas: {len(effects_df)} efectos")
else:
    has_isolated = False
    print("   Análisis de hormonas aisladas no disponible")

# 2. CREAR FIGURA MAESTRA (4x4 grid)
print("\n2. Creando figura maestra (4x4)...")

fig = plt.figure(figsize=(20, 16))
gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)

# Título principal
fig.suptitle('Sistema de Neuromodulación Endocrina para LLMs: Resultados Experimentales',
             fontsize=20, fontweight='bold', y=0.98)

# PANEL A: Diversidad Léxica por Perfil (2x2 - esquina superior izquierda)
print("   Panel A: Diversidad léxica por perfil...")
ax_a = fig.add_subplot(gs[0:2, 0:2])

# Top 10 perfiles por frecuencia
top_profiles = df_all['profile_name'].value_counts().head(10).index.tolist()
df_top = df_all[df_all['profile_name'].isin(top_profiles)]

# Violin plot
parts = ax_a.violinplot(
    [df_top[df_top['profile_name']==p]['distinct_2'].dropna() for p in top_profiles],
    positions=range(len(top_profiles)),
    widths=0.7,
    showmeans=True,
    showmedians=True
)

# Colorear
for pc in parts['bodies']:
    pc.set_facecolor('skyblue')
    pc.set_alpha(0.7)
    pc.set_edgecolor('black')
    pc.set_linewidth(1)

ax_a.set_xticks(range(len(top_profiles)))
ax_a.set_xticklabels([p.replace('_', '\n') for p in top_profiles], rotation=45, ha='right', fontsize=9)
ax_a.set_ylabel('Diversidad Léxica (Distinct-2)', fontsize=11, fontweight='bold')
ax_a.set_title('(A) Diversidad Léxica por Perfil Hormonal', fontsize=12, fontweight='bold', pad=10)
ax_a.grid(axis='y', alpha=0.3)
ax_a.set_ylim([0, 1])

# Destacar baseline
if 'baseline' in top_profiles:
    baseline_idx = top_profiles.index('baseline')
    ax_a.axvline(baseline_idx, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax_a.text(baseline_idx, 0.95, 'Baseline', ha='center', va='top', 
             fontsize=9, color='red', fontweight='bold')

# PANEL B: Efectos ANOVA (1x2 - esquina superior derecha)
print("   Panel B: Resultados ANOVA...")
ax_b = fig.add_subplot(gs[0, 2:])

metrics_display = {
    'distinct_2': 'Diversidad\nLéxica',
    'sentiment_polarity': 'Polaridad\nSentimiento',
    'repetition_rate': 'Tasa\nRepetición',
    'length': 'Longitud'
}

# Barras de F-statistic con colores según significancia
x_pos = np.arange(len(anova_df))
colors = ['green' if p < 0.001 else 'orange' if p < 0.01 else 'yellow' if p < 0.05 else 'gray' 
          for p in anova_df['p_value']]

bars = ax_b.bar(x_pos, anova_df['F_statistic'], color=colors, alpha=0.7, edgecolor='black')

ax_b.set_xticks(x_pos)
ax_b.set_xticklabels([metrics_display.get(m, m) for m in anova_df['metric']], fontsize=9)
ax_b.set_ylabel('F-statistic', fontsize=11, fontweight='bold')
ax_b.set_title('(B) ANOVA: Efecto de Perfiles Hormonales', fontsize=12, fontweight='bold', pad=10)
ax_b.grid(axis='y', alpha=0.3)

# Añadir etiquetas de significancia
for i, (bar, row) in enumerate(zip(bars, anova_df.itertuples())):
    height = bar.get_height()
    sig = row.significance
    ax_b.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             sig, ha='center', va='bottom', fontweight='bold', fontsize=10)

# Leyenda
legend_elements = [
    mpatches.Patch(color='green', alpha=0.7, label='p < 0.001 (***)'),
    mpatches.Patch(color='orange', alpha=0.7, label='p < 0.01 (**)'),
    mpatches.Patch(color='yellow', alpha=0.7, label='p < 0.05 (*)'),
    mpatches.Patch(color='gray', alpha=0.7, label='n.s.')
]
ax_b.legend(handles=legend_elements, loc='upper right', fontsize=8)

# PANEL C: Tamaño de efecto (eta²)
print("   Panel C: Tamaño de efecto...")
ax_c = fig.add_subplot(gs[1, 2:])

# Categorizar tamaño de efecto
def categorize_effect(eta2):
    if eta2 < 0.01:
        return 'Trivial'
    elif eta2 < 0.06:
        return 'Pequeño'
    elif eta2 < 0.14:
        return 'Mediano'
    else:
        return 'Grande'

anova_df['effect_category'] = anova_df['eta_squared'].apply(categorize_effect)

# Barras apiladas
categories = ['Trivial', 'Pequeño', 'Mediano', 'Grande']
cat_colors = ['lightgray', 'yellow', 'orange', 'red']

x_pos = np.arange(len(anova_df))

for i, category in enumerate(categories):
    heights = [row.eta_squared if row.effect_category == category else 0 
               for row in anova_df.itertuples()]

    if i == 0:
        ax_c.bar(x_pos, heights, label=category, color=cat_colors[i], alpha=0.7, edgecolor='black')
        bottoms = heights
    else:
        ax_c.bar(x_pos, heights, bottom=bottoms, label=category, 
                color=cat_colors[i], alpha=0.7, edgecolor='black')
        bottoms = [b + h for b, h in zip(bottoms, heights)]

ax_c.set_xticks(x_pos)
ax_c.set_xticklabels([metrics_display.get(m, m) for m in anova_df['metric']], fontsize=9)
ax_c.set_ylabel('η² (Eta cuadrado)', fontsize=11, fontweight='bold')
ax_c.set_title('(C) Tamaño del Efecto (Effect Size)', fontsize=12, fontweight='bold', pad=10)
ax_c.legend(loc='upper right', fontsize=8)
ax_c.grid(axis='y', alpha=0.3)

# PANEL D: Hormonas Individuales (si disponible)
print("   Panel D: Efectos de hormonas individuales...")
ax_d = fig.add_subplot(gs[2, :2])

if has_isolated and len(effects_df) > 0:
    # Filtrar efectos significativos en distinct_2
    sig_effects = effects_df[
        (effects_df['metric'] == 'distinct_2') & 
        (effects_df['p_value'] < 0.05)
    ].sort_values('difference')

    if len(sig_effects) > 0:
        y_pos = np.arange(len(sig_effects))
        colors_d = ['green' if d > 0 else 'red' for d in sig_effects['difference']]

        ax_d.barh(y_pos, sig_effects['difference'], color=colors_d, alpha=0.7, edgecolor='black')
        ax_d.set_yticks(y_pos)
        ax_d.set_yticklabels([h.title() for h in sig_effects['hormone']], fontsize=10)
        ax_d.axvline(0, color='black', linestyle='--', linewidth=1)
        ax_d.set_xlabel('Diferencia vs Baseline (Distinct-2)', fontsize=11, fontweight='bold')
        ax_d.set_title('(D) Efecto de Hormonas Individuales', fontsize=12, fontweight='bold', pad=10)
        ax_d.grid(axis='x', alpha=0.3)
 
        # Añadir significancia
        for i, (_, row) in enumerate(sig_effects.iterrows()):
            x_pos_text = row['difference'] + (0.005 if row['difference'] > 0 else -0.005)
            ax_d.text(x_pos_text, i, row['significance'], 
                     ha='left' if row['difference'] > 0 else 'right',
                     va='center', fontweight='bold', fontsize=9)
    else:
        ax_d.text(0.5, 0.5, 'No hay efectos significativos\nen diversidad léxica',
                 ha='center', va='center', fontsize=12, transform=ax_d.transAxes)
        ax_d.set_title('(D) Efecto de Hormonas Individuales', fontsize=12, fontweight='bold', pad=10)
else:
    ax_d.text(0.5, 0.5, 'Datos no disponibles',
             ha='center', va='center', fontsize=12, transform=ax_d.transAxes)
    ax_d.set_title('(D) Efecto de Hormonas Individuales', fontsize=12, fontweight='bold', pad=10)

ax_d.set_xlim([-0.15, 0.15])

# PANEL E: Comparación Semántica (si disponible)
print("   Panel E: Comparación sesgo simple vs semántico...")
ax_e = fig.add_subplot(gs[2, 2:])

if has_semantic:
    # Boxplot comparativo
    simple_d2 = df_semantic[df_semantic['condition']=='simple_bias']['distinct_2'].dropna()
    semantic_d2 = df_semantic[df_semantic['condition']=='semantic_bias']['distinct_2'].dropna()

    bp = ax_e.boxplot([simple_d2, semantic_d2], 
                       labels=['Simple', 'Semántico'],
                       patch_artist=True,
                       widths=0.6)

    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax_e.set_ylabel('Diversidad Léxica (Distinct-2)', fontsize=11, fontweight='bold')
    ax_e.set_title('(E) Comparación: Sesgo Simple vs Semántico', fontsize=12, fontweight='bold', pad=10)
    ax_e.grid(axis='y', alpha=0.3)

    # Test estadístico
    from scipy import stats
    t_stat, p_val = stats.ttest_ind(simple_d2, semantic_d2)

    y_max = max(simple_d2.max(), semantic_d2.max())
    ax_e.plot([1, 2], [y_max*1.05, y_max*1.05], 'k-', lw=1.5)
    sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    ax_e.text(1.5, y_max*1.07, sig_text, ha='center', fontsize=12, fontweight='bold')
else:
    ax_e.text(0.5, 0.5, 'Datos no disponibles',
             ha='center', va='center', fontsize=12, transform=ax_e.transAxes)
    ax_e.set_title('(E) Comparación: Sesgo Simple vs Semántico', fontsize=12, fontweight='bold', pad=10)

# PANEL F: Distribución de Métricas
print("   Panel F: Distribución general de métricas...")
ax_f = fig.add_subplot(gs[3, :])

# Histogramas superpuestos
metrics_to_plot = ['distinct_2', 'sentiment_polarity', 'repetition_rate']
colors_hist = ['blue', 'green', 'red']
labels_hist = ['Diversidad Léxica', 'Polaridad', 'Repetición']

for metric, color, label in zip(metrics_to_plot, colors_hist, labels_hist):
    data = df_all[metric].dropna()
    ax_f.hist(data, bins=30, alpha=0.5, color=color, label=label, density=True)

ax_f.set_xlabel('Valor Normalizado', fontsize=11, fontweight='bold')
ax_f.set_ylabel('Densidad', fontsize=11, fontweight='bold')
ax_f.set_title('(F) Distribución General de Métricas (Todos los Experimentos)', 
              fontsize=12, fontweight='bold', pad=10)
ax_f.legend(loc='upper right', fontsize=10)
ax_f.grid(axis='y', alpha=0.3)

# Añadir nota al pie
fig.text(0.5, 0.01, 
         f'Datos: {len(df_all)} generaciones totales | '
         f'Perfiles: {df_all["profile_name"].nunique()} | '
         f'Experimentos: {df_all["experiment"].nunique() if "experiment" in df_all.columns else "N/A"}',
         ha='center', fontsize=10, style='italic')

# 3. GUARDAR
print("\n3. Guardando figura maestra...")

plt.savefig(OUTPUT_DIR / 'master_figure.png', dpi=300, bbox_inches='tight')
print(f"   PNG: {OUTPUT_DIR / 'master_figure.png'}")

plt.savefig(OUTPUT_DIR / 'master_figure.pdf', bbox_inches='tight')
print(f"   PDF: {OUTPUT_DIR / 'master_figure.pdf'}")

plt.savefig(OUTPUT_DIR / 'master_figure.svg', bbox_inches='tight')
print(f"   SVG: {OUTPUT_DIR / 'master_figure.svg'}")

print("\n" + "="*80)
print(" FIGURA MAESTRA CREADA")
print("="*80)
print(f"\nEsta figura resume TODOS los resultados experimentales en un solo panel.")
print(f"Incluye: ANOVA, hormonas individuales, comparación semántica y distribuciones.")
print(f"\n Dimensiones: 20x16 pulgadas (ideal para página completa en TFM)")
print(f" Resolución: 300 DPI (calidad publicación)")
