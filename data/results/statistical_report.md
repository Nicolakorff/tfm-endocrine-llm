# Reporte Estadístico Completo
## TFM: Sistema de Neuromodulación Endocrina para LLMs

**Fecha:** 2025-12-27

---

## 1. Resumen Ejecutivo

Este reporte presenta el análisis estadístico completo de todos los experimentos
realizados para evaluar el efecto de la modulación hormonal artificial en la 
generación de texto.

### Experimentos Incluidos
- Fase 1: Hormonas individuales
- Fase 2: Perfiles combinados
- Comparación semántica: Sesgo simple vs embeddings
- Comparación sistema dinámico vs estático

---

## 2. Análisis ANOVA

### 2.1 Resultados Generales


#### DISTINCT 2

- **F-statistic:** 19.91
- **p-value:** 0.0000 ***
- **η² (effect size):** 0.093
- **Interpretación:** Diferencias MUY significativas entre grupos
- **Asunciones:** Normalidad=Violated, Homogeneidad=Violated


#### SENTIMENT POLARITY

- **F-statistic:** 2.25
- **p-value:** 0.0023 **
- **η² (effect size):** 0.011
- **Interpretación:** Diferencias significativas entre grupos
- **Asunciones:** Normalidad=Violated, Homogeneidad=Violated


#### REPETITION RATE

- **F-statistic:** 10.39
- **p-value:** 0.0000 ***
- **η² (effect size):** 0.051
- **Interpretación:** Diferencias MUY significativas entre grupos
- **Asunciones:** Normalidad=Violated, Homogeneidad=Violated


#### LENGTH

- **F-statistic:** 47.24
- **p-value:** 0.0000 ***
- **η² (effect size):** 0.196
- **Interpretación:** Diferencias MUY significativas entre grupos
- **Asunciones:** Normalidad=Violated, Homogeneidad=Violated


#### PERPLEXITY

- **F-statistic:** 231.80
- **p-value:** 0.0000 ***
- **η² (effect size):** 0.544
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


- **DOPAMINE × distinct_2:** r=0.133, p=0.0000 (débil positiva)

- **DOPAMINE × perplexity:** r=0.449, p=0.0000 (moderada positiva)

- **CORTISOL × distinct_2:** r=-0.080, p=0.0000 (débil negativa)

- **CORTISOL × repetition_rate:** r=-0.061, p=0.0020 (débil negativa)

- **CORTISOL × perplexity:** r=-0.439, p=0.0000 (moderada negativa)

- **OXYTOCIN × distinct_2:** r=0.051, p=0.0096 (débil positiva)

- **OXYTOCIN × perplexity:** r=0.121, p=0.0000 (débil positiva)

- **ADRENALINE × distinct_2:** r=-0.210, p=0.0000 (débil negativa)

- **ADRENALINE × repetition_rate:** r=0.126, p=0.0000 (débil positiva)

- **ADRENALINE × perplexity:** r=-0.265, p=0.0000 (débil negativa)

- **SEROTONIN × distinct_2:** r=0.112, p=0.0000 (débil positiva)

- **SEROTONIN × sentiment_polarity:** r=-0.043, p=0.0300 (débil negativa)

- **SEROTONIN × perplexity:** r=0.337, p=0.0000 (moderada positiva)


### 3.2 Visualizaciones

Ver:
- `data/results/correlation_analysis/correlation_heatmap.png`
- `data/results/correlation_analysis/correlation_scatterplots.png`

---

## 4. Conclusiones Estadísticas

### 4.1 Efecto de los Perfiles Hormonales

Los perfiles hormonales muestran efectos significativos en:

- **distinct_2**: con un tamaño de efecto mediano (η²=0.093)
- **sentiment_polarity**: con un tamaño de efecto pequeño (η²=0.011)
- **repetition_rate**: con un tamaño de efecto pequeño (η²=0.051)
- **length**: con un tamaño de efecto grande (η²=0.196)
- **perplexity**: con un tamaño de efecto grande (η²=0.544)


### 4.2 Relaciones Hormonas-Métricas

Se identificaron 13 correlaciones significativas, indicando que ciertos niveles hormonales están asociados con características específicas del texto generado.


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
