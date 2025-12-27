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
from scipy import stats

print("="*80)
print("CREANDO FIGURA MAESTRA PARA TFM")
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
consolidated_file = DATA_DIR / "consolidated/all_experiments_consolidated.csv"
if not consolidated_file.exists():
    print(f"ERROR: No se encontró {consolidated_file}")
    print("Ejecuta primero consolidate_all_experiments.py")
    exit(1)

df_all = pd.read_csv(consolidated_file)
print(f"Dataset consolidado: {len(df_all)} generaciones")

# Verificar si tenemos datos de Fase 3 (dinámico)
has_dynamic = 'is_dynamic' in df_all.columns and df_all['is_dynamic'].notna().any()
if has_dynamic:
    print("Datos dinámicos detectados")
    dynamic_count = df_all['is_dynamic'].sum()
    print(f"Dinámico: {dynamic_count} | Estático: {len(df_all) - dynamic_count}")

# Resultados ANOVA
anova_file = DATA_DIR / "anova_analysis/anova_summary.csv"
if anova_file.exists():
    anova_df = pd.read_csv(anova_file)
    has_anova = True
    print(f"Resultados ANOVA: {len(anova_df)} métricas")
else:
    has_anova = False
    print("Resultados ANOVA no disponibles")

# Comparación semántica
semantic_file = DATA_DIR / "semantic_comparison/comparison_results.csv"
if semantic_file.exists():
    df_semantic = pd.read_csv(semantic_file)
    has_semantic = True
    print(f"Comparación semántica: {len(df_semantic)} generaciones")
else:
    has_semantic = False
    print("Comparación semántica no disponible")

# Hormonas aisladas
effects_file = DATA_DIR / "isolated_hormones_analysis/hormone_effects_summary.csv"
if effects_file.exists():
    effects_df = pd.read_csv(effects_file)
    has_isolated = True
    print(f"Hormonas aisladas: {len(effects_df)} efectos")
else:
    has_isolated = False
    print("Análisis de hormonas aisladas no disponible")

# 2. CREAR FIGURA MAESTRA
print("\n2. Creando figura maestra...")

# Si tenemos datos dinámicos, usar grid 5x4, sino 4x4
if has_dynamic:
    print("Usando layout extendido (5x4) para incluir análisis dinámico")
    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(5, 4, figure=fig, hspace=0.35, wspace=0.3)
else:
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)

# Título principal
fig.suptitle('Sistema de Neuromodulación Endocrina para LLMs: Resultados Experimentales',
             fontsize=20, fontweight='bold', y=0.98)

# PANEL A: Diversidad Léxica por Perfil (2x2 - esquina superior izquierda)
print("Panel A: Diversidad léxica por perfil...")
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
print("Panel B: Resultados ANOVA...")
ax_b = fig.add_subplot(gs[0, 2:])

if has_anova and len(anova_df) > 0:
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
        sig = row.significance if hasattr(row, 'significance') else ''
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
else:
    ax_b.text(0.5, 0.5, 'Datos ANOVA no disponibles',
             ha='center', va='center', fontsize=12, transform=ax_b.transAxes)
    ax_b.set_title('(B) ANOVA: Efecto de Perfiles Hormonales', fontsize=12, fontweight='bold', pad=10)

# PANEL C: Tamaño de efecto (eta²)
print("Panel C: Tamaño de efecto...")
ax_c = fig.add_subplot(gs[1, 2:])

if has_anova and len(anova_df) > 0 and 'eta_squared' in anova_df.columns:
    # Categorizar tamaño de efecto
    def categorize_effect(eta2):
        if pd.isna(eta2):
            return 'Trivial'
        if eta2 < 0.01:
            return 'Trivial'
        elif eta2 < 0.06:
            return 'Pequeño'
        elif eta2 < 0.14:
            return 'Mediano'
        else:
            return 'Grande'

    anova_df['effect_category'] = anova_df['eta_squared'].apply(categorize_effect)

    # Barras directas (sin apilar para simplificar)
    x_pos = np.arange(len(anova_df))
    colors_effect = [
        'lightgray' if cat == 'Trivial' else 
        'yellow' if cat == 'Pequeño' else 
        'orange' if cat == 'Mediano' else 'red' 
        for cat in anova_df['effect_category']
    ]

    bars = ax_c.bar(x_pos, anova_df['eta_squared'], color=colors_effect, alpha=0.7, edgecolor='black')

    ax_c.set_xticks(x_pos)
    metrics_display = {
        'distinct_2': 'Diversidad\nLéxica',
        'sentiment_polarity': 'Polaridad\nSentimiento',
        'repetition_rate': 'Tasa\nRepetición',
        'length': 'Longitud'
    }
    ax_c.set_xticklabels([metrics_display.get(m, m) for m in anova_df['metric']], fontsize=9)
    ax_c.set_ylabel('η² (Eta cuadrado)', fontsize=11, fontweight='bold')
    ax_c.set_title('(C) Tamaño del Efecto (Effect Size)', fontsize=12, fontweight='bold', pad=10)

    # Leyenda
    legend_elements = [
        mpatches.Patch(color='lightgray', alpha=0.7, label='Trivial (< 0.01)'),
        mpatches.Patch(color='yellow', alpha=0.7, label='Pequeño (0.01-0.06)'),
        mpatches.Patch(color='orange', alpha=0.7, label='Mediano (0.06-0.14)'),
        mpatches.Patch(color='red', alpha=0.7, label='Grande (> 0.14)')
    ]
    ax_c.legend(handles=legend_elements, loc='upper right', fontsize=8)
    ax_c.grid(axis='y', alpha=0.3)
else:
    ax_c.text(0.5, 0.5, 'Datos de eta² no disponibles',
             ha='center', va='center', fontsize=12, transform=ax_c.transAxes)
    ax_c.set_title('(C) Tamaño del Efecto (Effect Size)', fontsize=12, fontweight='bold', pad=10)

# PANEL D: Hormonas Individuales
print("Panel D: Efectos de hormonas individuales...")
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
            sig_text = row['significance'] if 'significance' in row else ''
            ax_d.text(x_pos_text, i, sig_text, 
                     ha='left' if row['difference'] > 0 else 'right',
                     va='center', fontweight='bold', fontsize=9)

        ax_d.set_xlim([-0.15, 0.15])
    else:
        ax_d.text(0.5, 0.5, 'No hay efectos significativos\nen diversidad léxica',
                 ha='center', va='center', fontsize=12, transform=ax_d.transAxes)
        ax_d.set_title('(D) Efecto de Hormonas Individuales', fontsize=12, fontweight='bold', pad=10)
else:
    ax_d.text(0.5, 0.5, 'Datos no disponibles',
             ha='center', va='center', fontsize=12, transform=ax_d.transAxes)
    ax_d.set_title('(D) Efecto de Hormonas Individuales', fontsize=12, fontweight='bold', pad=10)

# PANEL E: Comparación Semántica
print("Panel E: Comparación sesgo simple vs semántico...")
ax_e = fig.add_subplot(gs[2, 2:])

if has_semantic and 'condition' in df_semantic.columns:
    # Boxplot comparativo
    simple_d2 = df_semantic[df_semantic['condition']=='simple_bias']['distinct_2'].dropna()
    semantic_d2 = df_semantic[df_semantic['condition']=='semantic_bias']['distinct_2'].dropna()

    if len(simple_d2) > 0 and len(semantic_d2) > 0:
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
        t_stat, p_val = stats.ttest_ind(simple_d2, semantic_d2)

        y_max = max(simple_d2.max(), semantic_d2.max())
        ax_e.plot([1, 2], [y_max*1.05, y_max*1.05], 'k-', lw=1.5)
        sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        ax_e.text(1.5, y_max*1.07, sig_text, ha='center', fontsize=12, fontweight='bold')
    else:
        ax_e.text(0.5, 0.5, 'Datos insuficientes',
                 ha='center', va='center', fontsize=12, transform=ax_e.transAxes)
        ax_e.set_title('(E) Comparación: Sesgo Simple vs Semántico', fontsize=12, fontweight='bold', pad=10)
else:
    ax_e.text(0.5, 0.5, 'Datos no disponibles',
             ha='center', va='center', fontsize=12, transform=ax_e.transAxes)
    ax_e.set_title('(E) Comparación: Sesgo Simple vs Semántico', fontsize=12, fontweight='bold', pad=10)

# PANEL F: Distribución de Métricas
print("Panel F: Distribución general de métricas...")
ax_f = fig.add_subplot(gs[3, :])

# Histogramas superpuestos
metrics_to_plot = []
colors_hist = []
labels_hist = []

if 'distinct_2' in df_all.columns:
    metrics_to_plot.append('distinct_2')
    colors_hist.append('blue')
    labels_hist.append('Diversidad Léxica')

if 'sentiment_polarity' in df_all.columns:
    metrics_to_plot.append('sentiment_polarity')
    colors_hist.append('green')
    labels_hist.append('Polaridad')

if 'repetition_rate' in df_all.columns:
    metrics_to_plot.append('repetition_rate')
    colors_hist.append('red')
    labels_hist.append('Repetición')

for metric, color, label in zip(metrics_to_plot, colors_hist, labels_hist):
    data = df_all[metric].dropna()
    if len(data) > 0:
        ax_f.hist(data, bins=30, alpha=0.5, color=color, label=label, density=True)

ax_f.set_xlabel('Valor Normalizado', fontsize=11, fontweight='bold')
ax_f.set_ylabel('Densidad', fontsize=11, fontweight='bold')
ax_f.set_title('(F) Distribución General de Métricas (Todos los Experimentos)', 
              fontsize=12, fontweight='bold', pad=10)
ax_f.legend(loc='upper right', fontsize=10)
ax_f.grid(axis='y', alpha=0.3)

# PANEL G: Análisis Dinámico (solo si hay datos)
if has_dynamic:
    print("Panel G: Sistema dinámico (Estático vs Dinámico)...")
    ax_g1 = fig.add_subplot(gs[4, 0:2])
    ax_g2 = fig.add_subplot(gs[4, 2:])

    static_df = df_all[df_all['is_dynamic'] == False]
    dynamic_df = df_all[df_all['is_dynamic'] == True]

    # Subplot G1: Boxplot comparativo
    if len(static_df) > 0 and len(dynamic_df) > 0:
        bp = ax_g1.boxplot(
            [static_df['distinct_2'].dropna(), dynamic_df['distinct_2'].dropna()],
            labels=['Estático', 'Dinámico'],
            patch_artist=True,
            widths=0.6
        )

        for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax_g1.set_ylabel('Diversidad Léxica (Distinct-2)', fontsize=11, fontweight='bold')
        ax_g1.set_title('(G1) Estático vs Dinámico', fontsize=12, fontweight='bold', pad=10)
        ax_g1.grid(axis='y', alpha=0.3)

        # Test estadístico
        t_stat, p_val = stats.ttest_ind(
            static_df['distinct_2'].dropna(),
            dynamic_df['distinct_2'].dropna()
        )
        sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        ax_g1.text(0.5, 0.95, f'p {sig_text}', transform=ax_g1.transAxes,
                  ha='center', va='top', fontsize=10, fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # Subplot G2: Distribución de cambios hormonales
    if 'total_hormone_change' in dynamic_df.columns:
        changes = dynamic_df['total_hormone_change'].dropna()
        if len(changes) > 0:
            ax_g2.hist(changes, bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
            ax_g2.axvline(changes.mean(), color='darkred', linestyle='--', linewidth=2,
                         label=f'Media: {changes.mean():.4f}')
            ax_g2.set_xlabel('Cambio Hormonal Total', fontsize=11, fontweight='bold')
            ax_g2.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
            ax_g2.set_title('(G2) Distribución de Cambios Hormonales', fontsize=12, fontweight='bold', pad=10)
            ax_g2.legend()
            ax_g2.grid(axis='y', alpha=0.3)

# Añadir nota al pie
experiments_count = df_all['experiment'].nunique() if 'experiment' in df_all.columns else 'N/A'
fig.text(0.5, 0.01 if not has_dynamic else 0.005, 
         f'Datos: {len(df_all):,} generaciones totales | '
         f'Perfiles: {df_all["profile_name"].nunique()} | '
         f'Experimentos: {experiments_count}',
         ha='center', fontsize=10, style='italic')

# 3. GUARDAR
print("\n3. Guardando figura maestra...")

plt.savefig(OUTPUT_DIR / 'master_figure.png', dpi=300, bbox_inches='tight')
print(f"PNG: {OUTPUT_DIR / 'master_figure.png'}")

plt.savefig(OUTPUT_DIR / 'master_figure.pdf', bbox_inches='tight')
print(f"PDF: {OUTPUT_DIR / 'master_figure.pdf'}")

plt.savefig(OUTPUT_DIR / 'master_figure.svg', bbox_inches='tight')
print(f"SVG: {OUTPUT_DIR / 'master_figure.svg'}")

plt.close()

print("\n" + "="*80)
print("FIGURA MAESTRA CREADA")
print("="*80)
print("\n Esta figura resume TODOS los resultados experimentales en un solo panel.")
print("Incluye: ANOVA, hormonas individuales, comparación semántica y distribuciones.")
if has_dynamic:
    print(" + Análisis del sistema dinámico (Fase 3)")
print(f"\n Dimensiones: {'20x20' if has_dynamic else '20x16'} pulgadas")
print("Resolución: 300 DPI (calidad publicación)")
print(f"Paneles: {'7 (A-G)' if has_dynamic else '6 (A-F)'}")
