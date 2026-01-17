# Reporte Estadístico Completo
## TFM: Sistema de Neuromodulación Endocrina para LLMs

**Fecha:** 2026-01-10

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


#### DISTINCT 2

- **F-statistic:** 50.35
- **p-value:** 0.0000 ***
- **η² (effect size):** 0.104
- **Interpretación:** Diferencias MUY significativas entre grupos
- **Asunciones:** Normalidad=Violated, Homogeneidad=Violated


#### SENTIMENT POLARITY

- **F-statistic:** 2.02
- **p-value:** 0.0079 **
- **η² (effect size):** 0.005
- **Interpretación:** Diferencias significativas entre grupos
- **Asunciones:** Normalidad=OK, Homogeneidad=Violated


#### REPETITION RATE

- **F-statistic:** 27.27
- **p-value:** 0.0000 ***
- **η² (effect size):** 0.059
- **Interpretación:** Diferencias MUY significativas entre grupos
- **Asunciones:** Normalidad=Violated, Homogeneidad=Violated


#### LENGTH

- **F-statistic:** 45.92
- **p-value:** 0.0000 ***
- **η² (effect size):** 0.096
- **Interpretación:** Diferencias MUY significativas entre grupos
- **Asunciones:** Normalidad=Violated, Homogeneidad=OK


#### PERPLEXITY

- **F-statistic:** 473.40
- **p-value:** 0.0000 ***
- **η² (effect size):** 0.522
- **Interpretación:** Diferencias MUY significativas entre grupos
- **Asunciones:** Normalidad=Violated, Homogeneidad=Violated



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


- **DOPAMINE × distinct_2:** r=0.149, p=0.0000 (débil positiva)

- **DOPAMINE × repetition_rate:** r=-0.032, p=0.0094 (débil negativa)

- **DOPAMINE × perplexity:** r=0.249, p=0.0000 (débil positiva)

- **CORTISOL × distinct_2:** r=-0.107, p=0.0000 (débil negativa)

- **CORTISOL × repetition_rate:** r=-0.044, p=0.0003 (débil negativa)

- **CORTISOL × perplexity:** r=-0.233, p=0.0000 (débil negativa)

- **OXYTOCIN × distinct_2:** r=0.040, p=0.0014 (débil positiva)

- **OXYTOCIN × perplexity:** r=0.067, p=0.0000 (débil positiva)

- **ADRENALINE × distinct_2:** r=-0.224, p=0.0000 (débil negativa)

- **ADRENALINE × repetition_rate:** r=0.135, p=0.0000 (débil positiva)

- **ADRENALINE × length:** r=-0.030, p=0.0159 (débil negativa)

- **ADRENALINE × perplexity:** r=-0.154, p=0.0000 (débil negativa)

- **SEROTONIN × distinct_2:** r=0.168, p=0.0000 (débil positiva)

- **SEROTONIN × repetition_rate:** r=-0.045, p=0.0003 (débil negativa)

- **SEROTONIN × length:** r=0.031, p=0.0126 (débil positiva)

- **SEROTONIN × perplexity:** r=0.193, p=0.0000 (débil positiva)


### 3.2 Visualizaciones

Ver:
- `data/results/correlation_analysis/correlation_heatmap.png`
- `data/results/correlation_analysis/correlation_scatterplots.png`

---

## 4. Conclusiones Estadísticas

### 4.1 Efecto de los Perfiles Hormonales

Los perfiles hormonales muestran efectos significativos en:

- **distinct_2**: con un tamaño de efecto mediano (η²=0.104)
- **sentiment_polarity**: con un tamaño de efecto pequeño (η²=0.005)
- **repetition_rate**: con un tamaño de efecto pequeño (η²=0.059)
- **length**: con un tamaño de efecto mediano (η²=0.096)
- **perplexity**: con un tamaño de efecto grande (η²=0.522)


### 4.2 Relaciones Hormonas-Métricas

Se identificaron 16 correlaciones significativas, indicando que ciertos niveles hormonales están asociados con características específicas del texto generado.


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
