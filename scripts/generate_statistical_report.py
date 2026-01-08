"""
Genera reporte ejecutivo con todos los análisis estadísticos
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("data/results")
REPORT_FILE = OUTPUT_DIR / "statistical_report.md"

# Cargar resultados
anova_df = pd.read_csv(OUTPUT_DIR / "anova_analysis/anova_summary.csv")
corr_df = pd.read_csv(OUTPUT_DIR / "correlation_analysis/significant_correlations.csv")

# Generar reporte
report = f"""# Reporte Estadístico Completo
## TFM: Sistema de Neuromodulación Endocrina para LLMs

**Fecha:** {datetime.now().strftime('%Y-%m-%d')}

---

## 1. Resumen Ejecutivo

Este reporte presenta el análisis estadístico completo de todos los experimentos
realizados para evaluar el efecto de la modulación hormonal artificial en la 
generación de texto.

### Experimentos Incluidos
- Fase 1: Hormonas individuales
- Fase 2: Perfiles combinados
- Fase 3: Comparación semántica: Sesgo simple vs embeddings
- Fase 4: Comparación sistema dinámico vs estático

---

## 2. Análisis ANOVA

### 2.1 Resultados Generales

"""

# Añadir resultados ANOVA
for _, row in anova_df.iterrows():
    report += f"\n#### {row['metric'].replace('_', ' ').upper()}\n\n"
    report += f"- **F-statistic:** {row['F_statistic']:.2f}\n"
    report += f"- **p-value:** {row['p_value']:.4f} {row['significance']}\n"
    report += f"- **η² (effect size):** {row['eta_squared']:.3f}\n"
    report += f"- **Interpretación:** {row['interpretation']}\n"
    report += f"- **Asunciones:** Normalidad={row['normality_assumption']}, "
    report += f"Homogeneidad={row['homogeneity_assumption']}\n\n"

report += """

### 2.2 Tests Post-Hoc (Tukey HSD)

Los resultados completos de las comparaciones pareadas se encuentran en:
```
data/results/anova_analysis/tukey_*.csv
```

**Interpretación:** Las comparaciones post-hoc indican qué pares de perfiles 
hormonales difieren significativamente entre sí.

---

## 3. Análisis de Correlación

### 3.1 Correlaciones Significativas

"""

if len(corr_df) > 0:
    for _, row in corr_df.iterrows():
        report += f"\n- **{row['hormone'].upper()} × {row['metric']}:** "
        report += f"r={row['correlation']:.3f}, p={row['p_value']:.4f} "
        report += f"({row['strength']} {row['direction']})\n"
else:
    report += "\nNo se encontraron correlaciones significativas (p < 0.05).\n"

report += """

### 3.2 Visualizaciones

Ver:
- `data/results/correlation_analysis/correlation_heatmap.png`
- `data/results/correlation_analysis/correlation_scatterplots.png`

---

## 4. Conclusiones Estadísticas

### 4.1 Efecto de los Perfiles Hormonales

"""

# Añadir conclusiones basadas en resultados
significant_metrics = anova_df[anova_df['p_value'] < 0.05]

if len(significant_metrics) > 0:
    report += "Los perfiles hormonales muestran efectos significativos en:\n\n"
    for _, row in significant_metrics.iterrows():
        report += f"- **{row['metric']}**: con un tamaño de efecto "
        if row['eta_squared'] < 0.06:
            report += "pequeño"
        elif row['eta_squared'] < 0.14:
            report += "mediano"
        else:
            report += "grande"
        report += f" (η²={row['eta_squared']:.3f})\n"
else:
    report += "No se detectaron efectos significativos de los perfiles hormonales.\n"

report += """

### 4.2 Relaciones Hormonas-Métricas

"""

if len(corr_df) > 0:
    report += f"Se identificaron {len(corr_df)} correlaciones significativas, "
    report += "indicando que ciertos niveles hormonales están asociados con "
    report += "características específicas del texto generado.\n"
else:
    report += "No se encontraron correlaciones lineales fuertes entre niveles "
    report += "hormonales individuales y métricas de texto.\n"

report += """

---

## 5. Limitaciones

1. **Tamaño muestral:** Variable según experimento
2. **Asunciones:** Algunas métricas violan asunción de normalidad
3. **Modelo:** Resultados basados en GPT-2 (pequeño)
4. **Causalidad:** Correlaciones no implican causalidad

---

## 6. Archivos Generados

### Tablas
- `anova_summary.csv`
- `tukey_*.csv` (por métrica)
- `correlation_matrix.csv`
- `significant_correlations.csv`

### Figuras
- `anova_comparison.png/pdf`
- `correlation_heatmap.png/pdf`
- `correlation_scatterplots.png`

### LaTeX
- `anova_table.tex`
- `correlation_table.tex`

---

**Fin del reporte**
"""

# Guardar reporte
with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"Reporte guardado: {REPORT_FILE}")
print(f"\n Para ver: cat {REPORT_FILE}")
