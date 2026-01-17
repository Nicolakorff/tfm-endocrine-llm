# Reporte Estadístico Completo
## TFM: Sistema de Neuromodulación Endocrina para LLMs

**Fecha:** 2026-01-08

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

- **F-statistic:** 16.72
- **p-value:** 0.0000 ***
- **η² (effect size):** 0.079
- **Interpretación:** Diferencias MUY significativas entre grupos
- **Asunciones:** Normalidad=Violated, Homogeneidad=Violated


#### SENTIMENT POLARITY

- **F-statistic:** 0.64
- **p-value:** 0.8595 ns
- **η² (effect size):** 0.003
- **Interpretación:** No hay diferencias significativas entre grupos
- **Asunciones:** Normalidad=Violated, Homogeneidad=OK


#### REPETITION RATE

- **F-statistic:** 6.87
- **p-value:** 0.0000 ***
- **η² (effect size):** 0.034
- **Interpretación:** Diferencias MUY significativas entre grupos
- **Asunciones:** Normalidad=Violated, Homogeneidad=Violated


#### LENGTH

- **F-statistic:** 46.29
- **p-value:** 0.0000 ***
- **η² (effect size):** 0.192
- **Interpretación:** Diferencias MUY significativas entre grupos
- **Asunciones:** Normalidad=Violated, Homogeneidad=Violated


#### PERPLEXITY

- **F-statistic:** 229.93
- **p-value:** 0.0000 ***
- **η² (effect size):** 0.542
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


- **DOPAMINE × distinct_2:** r=0.167, p=0.0000 (débil positiva)

- **DOPAMINE × perplexity:** r=0.456, p=0.0000 (moderada positiva)

- **CORTISOL × distinct_2:** r=-0.117, p=0.0000 (débil negativa)

- **CORTISOL × perplexity:** r=-0.430, p=0.0000 (moderada negativa)

- **OXYTOCIN × distinct_2:** r=0.046, p=0.0181 (débil positiva)

- **OXYTOCIN × perplexity:** r=0.125, p=0.0000 (débil positiva)

- **ADRENALINE × distinct_2:** r=-0.190, p=0.0000 (débil negativa)

- **ADRENALINE × repetition_rate:** r=0.104, p=0.0000 (débil positiva)

- **ADRENALINE × perplexity:** r=-0.231, p=0.0000 (débil negativa)

- **SEROTONIN × distinct_2:** r=0.138, p=0.0000 (débil positiva)

- **SEROTONIN × repetition_rate:** r=-0.040, p=0.0406 (débil negativa)

- **SEROTONIN × perplexity:** r=0.284, p=0.0000 (débil positiva)


### 3.2 Visualizaciones

Ver:
- `data/results/correlation_analysis/correlation_heatmap.png`
- `data/results/correlation_analysis/correlation_scatterplots.png`

---

## 4. Conclusiones Estadísticas

### 4.1 Efecto de los Perfiles Hormonales

Los perfiles hormonales muestran efectos significativos en:

- **distinct_2**: con un tamaño de efecto mediano (η²=0.079)
- **repetition_rate**: con un tamaño de efecto pequeño (η²=0.034)
- **length**: con un tamaño de efecto grande (η²=0.192)
- **perplexity**: con un tamaño de efecto grande (η²=0.542)


### 4.2 Relaciones Hormonas-Métricas

Se identificaron 12 correlaciones significativas, indicando que ciertos niveles hormonales están asociados con características específicas del texto generado.


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
