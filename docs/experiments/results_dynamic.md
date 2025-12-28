# 📊 Resultados: Sistema Dinámico vs Estático (Fase 3)

**Versión:** 1.0  
**Fecha:** Enero 2025  
**Estado:** Listo para datos experimentales

---

## 📋 Resumen Ejecutivo

Este documento presenta los resultados de la comparación entre el **sistema hormonal dinámico** (con actualización en tiempo real) y el **sistema estático** (perfiles fijos).

### Configuración Experimental

- **Modelo Base:** DistilGPT2 (82M parámetros)
- **Prompts:** 40 (10 por categoría: creative, empathetic, factual, reasoning)
- **Perfiles:** 6 (3 estáticos + 3 dinámicos equivalentes)
- **Repeticiones:** 3 por combinación
- **Total Generaciones:** 720 (360 estáticas + 360 dinámicas)
- **Parámetros:**
  - `max_new_tokens`: 50
  - `update_interval`: 5
  - `learning_rate`: 0.15 (dinámico)

---

## 📊 Resultados Principales

### 1. Diversidad Léxica (Distinct-2)

| Sistema | Media | SD | Min | Max | N |
|---------|-------|-----|-----|-----|---|
| Estático | 0.XXX | 0.XXX | 0.XXX | 0.XXX | 360 |
| Dinámico | 0.XXX | 0.XXX | 0.XXX | 0.XXX | 360 |
| **Diferencia** | **+0.XXX** | - | - | - | - |

**Análisis Estadístico:**
- **t-test:** t(718) = X.XX, p = 0.XXX
- **Cohen's d:** X.XX (pequeño/medio/grande)
- **IC 95%:** [X.XXX, X.XXX]

**Interpretación:**  
> [Completar con tus resultados]

---

### 2. Tasa de Repetición

| Sistema | Media | SD |
|---------|-------|-----|
| Estático | 0.XXX | 0.XXX |
| Dinámico | 0.XXX | 0.XXX |
| **Diferencia** | **-0.XXX** | - |

**Análisis:**
- **t-test:** t(718) = X.XX, p = 0.XXX
- **Cohen's d:** X.XX

**Interpretación:**
> [Completar con tus resultados]

---

### 3. Cambios Hormonales (Solo Sistema Dinámico)

#### Cambio Hormonal Total

**Estadísticos:**
- **Media:** 0.XXX
- **SD:** 0.XXX
- **Mediana:** 0.XXX
- **Rango:** [0.XXX, 0.XXX]
- **% con cambio > 0.10:** XX%

#### Cambios por Hormona Individual

| Hormona | Δ Media | SD | Rango |
|---------|---------|-----|-------|
| Dopamine | +0.XXX | 0.XXX | [X.XXX, X.XXX] |
| Cortisol | -0.XXX | 0.XXX | [X.XXX, X.XXX] |
| Oxytocin | +0.XXX | 0.XXX | [X.XXX, X.XXX] |
| Adrenaline | +0.XXX | 0.XXX | [X.XXX, X.XXX] |
| Serotonin | +0.XXX | 0.XXX | [X.XXX, X.XXX] |

**Interpretación:**
> [Describir qué hormonas cambian más y por qué]

---

### 4. Análisis por Categoría de Prompt

#### Cambio Hormonal por Categoría

| Categoría | N | Cambio Total (Media) | SD |
|-----------|---|----------------------|----|
| Empathetic | XX | 0.XXX | 0.XXX |
| Creative | XX | 0.XXX | 0.XXX |
| Factual | XX | 0.XXX | 0.XXX |
| Reasoning | XX | 0.XXX | 0.XXX |

**ANOVA:**
- **F(3, 356) = X.XX, p = 0.XXX**
- **η² = 0.XXX**

**Post-hoc (Tukey HSD):**
> [Describir comparaciones significativas entre categorías]

---

## 📈 Visualizaciones

### Figura 1: Comparación Boxplots

![Estático vs Dinámico](../results/dynamic_analysis/static_vs_dynamic_boxplots.png)

**Descripción:** Boxplots comparativos de Distinct-2, Repetition Rate y Sentiment.

---

### Figura 2: Distribución de Cambios Hormonales

![Distribución de Cambios](../results/dynamic_analysis/hormone_change_distribution.png)

**Descripción:** Histograma mostrando la distribución de `total_hormone_change` en sistema dinámico.

---

### Figura 3: Cambios por Categoría

![Cambios por Categoría](../results/dynamic_analysis/hormone_changes_by_category.png)

**Descripción:** Boxplots de cambios hormonales según categoría de prompt.

---

## 🔍 Análisis Cualitativo

### Ejemplos Representativos

#### Ejemplo 1: Prompt Empático

**Prompt:** "I'm feeling stressed about work."

**Estático (empathic):**
```
[Tu texto generado aquí]
```
- Distinct-2: 0.XXX
- Repetition: 0.XXX

**Dinámico (empathic, dynamic):**
```
[Tu texto generado aquí]
```
- Distinct-2: 0.XXX
- Repetition: 0.XXX
- Cambios: Oxytocina +0.XXX, Cortisol -0.XXX

**Observaciones:**
> [Comentar diferencias cualitativas]

---

#### Ejemplo 2: Prompt Creativo

**Prompt:** "Write a creative story about"

**Estático (creative):**
```
[Tu texto generado aquí]
```

**Dinámico (creative, dynamic):**
```
[Tu texto generado aquí]
```
- Cambios: Dopamina +0.XXX, Serotonina -0.XXX

**Observaciones:**
> [Comentar diferencias]

---

## 🎯 Validación de Hipótesis

### H1: Diversidad Léxica ✓/✗

**Hipótesis:** Dinámico > Estático en Distinct-2

**Resultado:**
- Diferencia: +0.XXX (X.X%)
- p = 0.XXX
- **[✓ CONFIRMADA / ✗ RECHAZADA]**

---

### H2: Cambios Hormonales Significativos ✓/✗

**Hipótesis:** Total change > 0.10

**Resultado:**
- Media: 0.XXX
- % > 0.10: XX%
- **[✓ CONFIRMADA / ✗ RECHAZADA]**

---

### H3: Adaptación Contextual ✓/✗

**Hipótesis:** Cambios difieren por categoría

**Resultado:**
- ANOVA: F(3,356) = X.XX, p = 0.XXX
- **[✓ CONFIRMADA / ✗ RECHAZADA]**

---

### H4: Reducción de Repetición ✓/✗

**Hipótesis:** Dinámico < Estático en repetición

**Resultado:**
- Diferencia: -0.XXX (-XX%)
- p = 0.XXX
- **[✓ CONFIRMADA / ✗ RECHAZADA]**

---

## 💡 Conclusiones

### Hallazgos Principales

1. **[Conclusión 1 basada en tus datos]**

2. **[Conclusión 2]**

3. **[Conclusión 3]**

### Implicaciones

- **Teóricas:** [Qué aporta a la comprensión de sistemas dinámicos]
- **Prácticas:** [Aplicaciones potenciales]
- **Metodológicas:** [Lecciones aprendidas]

---

## ⚠️ Limitaciones

1. **Muestra:** 40 prompts de 200 totales (20%)
2. **Learning rate:** Fijo en 0.15 (no optimizado)
3. **Modelo:** DistilGPT2 pequeño (82M)
4. **Sin evaluación humana:** Solo métricas automáticas
5. **Update interval:** Fijo en 5 tokens

---

## 🔮 Trabajo Futuro

### Corto Plazo
1. Aumentar muestra a 200 prompts completos
2. Grid search de learning_rate [0.05, 0.30]
3. Probar update_intervals [3, 5, 7, 10]

### Medio Plazo
4. Evaluación humana (N≥30 anotadores)
5. Modelos más grandes (GPT-2 Large, Llama 2)
6. Análisis de clustering de trayectorias

### Largo Plazo
7. Aprendizaje de learning_rate óptimo por contexto
8. Meta-aprendizaje de estrategias de adaptación
9. Integración con RL para optimización

---

## 📁 Archivos Generados

### Datos
- `data/results/phase3_dynamic_results.csv` - Dataset completo (720 filas)
- `data/results/dynamic_analysis/static_vs_dynamic_comparison.csv` - Comparación estadística

### Visualizaciones
- `static_vs_dynamic_boxplots.png`
- `hormone_change_distribution.png`
- `hormone_changes_by_category.png`
- `example_trajectory_high_change.png`
- `example_trajectory_low_change.png`

### Análisis
- `statistical_tests.csv` - Todos los t-tests
- `anova_results.csv` - ANOVA por categoría
- `hormone_deltas_summary.csv` - Resumen de cambios hormonales

---

## 📚 Referencias para Interpretación

### Umbrales de Efecto (Cohen's d)
- d < 0.2: Trivial
- 0.2 ≤ d < 0.5: Pequeño
- 0.5 ≤ d < 0.8: Medio
- d ≥ 0.8: Grande

### Significancia
- p < 0.05: Significativo (*)
- p < 0.01: Muy significativo (**)
- p < 0.001: Extremadamente significativo (***)

---

**Documento preparado para:** TFM - Máster en Grandes Modelos de Lenguaje  
**Estado:** Listo para integración de datos experimentales

---

## 📝 Notas para Completar

**IMPORTANTE:** Este documento contiene placeholders (0.XXX) que deben ser reemplazados con los datos reales de tu experimento.

### Pasos para completar:

1. **Ejecutar experimento:**
   ```bash
   python scripts/run_dynamic_experiment.py
   ```

2. **Analizar resultados:**
   ```bash
   python scripts/analyze_dynamic_results.py
   ```

3. **Reemplazar placeholders** con valores de:
   - `data/results/dynamic_analysis/static_vs_dynamic_comparison.csv`

4. **Añadir textos de ejemplo** de generaciones representativas

5. **Completar secciones** de interpretación y observaciones

6. **Verificar figuras** están en rutas correctas
