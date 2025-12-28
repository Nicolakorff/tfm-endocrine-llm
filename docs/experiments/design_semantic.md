# Diseño Experimental: Comparación de Sesgos Semánticos

## Sistema de Neuromodulación Endocrina para LLMs - Fase 4

**Versión:** 2.0  
**Fecha:** Enero 2025  
**Estado:** Completado y validado

---

## Resumen Ejecutivo

Este documento describe el diseño experimental para comparar dos enfoques de sesgo prosocial en generación de texto modulada hormonalmente:

1. **Sesgo Simple (Léxico):** Boost de ~15 tokens empáticos predefinidos
2. **Sesgo Semántico (Embeddings):** Boost proporcional a similitud con categoría semántica

**Pregunta de investigación principal:**  
¿Los sesgos semánticos basados en embeddings de Sentence-BERT producen resultados cualitativamente superiores a los sesgos léxicos simples en términos de diversidad léxica, coherencia semántica y calidad del contenido?

---

## Objetivos

### Objetivo Principal

Evaluar si los sesgos semánticos basados en embeddings producen resultados cualitativamente diferentes (y superiores) a los sesgos simples basados en listas fijas de tokens.

### Objetivos Específicos

1. **Cuantificar diferencias** en diversidad léxica (Distinct-2) entre ambos enfoques
2. **Medir activación semántica** en la categoría objetivo (empathy)
3. **Comparar cobertura** de tokens afectados (vocabulario)
4. **Analizar diferencias** según tipo de prompt (empático vs no-empático)
5. **Evaluar trade-offs** entre complejidad computacional y beneficios

---

## Hipótesis

### Hipótesis Principal (H1)

**El sesgo semántico producirá mayor diversidad léxica que el sesgo simple.**

- **Justificación:** El sesgo semántico afecta ~1000 tokens vs ~15 del sesgo simple, expandiendo el espacio léxico disponible mientras mantiene coherencia semántica.
- **Dirección:** Unidireccional (sesgo semántico > sesgo simple)
- **Criterio:** Diferencia significativa en Distinct-2 (p < 0.05, d > 0.5)

### Hipótesis Secundaria 1 (H2)

**El sesgo semántico producirá textos con mayor activación de la categoría objetivo (empathy) medida por similitud con embeddings.**

- **Justificación:** El mecanismo de sesgo basado en similitud semántica debería resultar en mayor uso de vocabulario relacionado con la categoría objetivo.
- **Medición:** Activación semántica promedio > 0.4 (umbral teórico)
- **Comparación:** No aplicable a sesgo simple (no tiene esta métrica)

### Hipótesis Secundaria 2 (H3)

**Las diferencias entre sesgos serán más pronunciadas en prompts empáticos que en prompts no-empáticos.**

- **Justificación:** La congruencia entre tipo de prompt y sesgo aplicado debería amplificar el efecto.
- **Medición:** Interacción significativa (tipo_sesgo × tipo_prompt) en ANOVA 2×2
- **Efecto esperado:** Δ_empático > Δ_no-empático en al menos 25%

### Hipótesis Secundaria 3 (H4)

**El sesgo semántico reducirá la tasa de repetición comparado con el sesgo simple.**

- **Justificación:** Mayor variabilidad léxica debería traducirse en menos repeticiones de bigramas.
- **Criterio:** Diferencia significativa (p < 0.05, d > 0.3)

### Hipótesis Nula (H0)

**No existen diferencias significativas entre sesgo simple y semántico en ninguna de las métricas evaluadas.**

---

## Variables

### Variable Independiente (VI)

**Tipo de Sesgo** (categórica, 2 niveles)

| Nivel | Descripción | Implementación |
|-------|-------------|----------------|
| **Simple** | Sesgo léxico fijo | Boost de ε = 0.5 en logits de 15 tokens empáticos |
| **Semántico** | Sesgo por embeddings | Boost proporcional: δ × strength × similarity(token, category) |

**Manipulación:**
- Mismo prompt procesado con ambas condiciones
- Orden contrabalanceado (50% simple→semántico, 50% semántico→simple)
- Generaciones independientes (no secuenciales)

---

### Variables Dependientes (VD)

#### VD1: Diversidad Léxica

**Métrica:** Distinct-2 (ratio de bigramas únicos / total bigramas)

- **Rango:** [0, 1]
- **Interpretación:** Valores altos = mayor diversidad
- **Cálculo:**
```python
  distinct_2 = len(set(bigrams)) / len(bigrams)
```

#### VD2: Tasa de Repetición

**Métrica:** Proporción de bigramas repetidos

- **Rango:** [0, 1]
- **Interpretación:** Valores bajos = menor repetición
- **Relación con VD1:** Correlación negativa esperada (r ≈ -0.7)

#### VD3: Polaridad del Sentimiento

**Métrica:** Polaridad TextBlob

- **Rango:** [-1, 1]
- **Interpretación:** Valores positivos = tono positivo
- **Objetivo:** Verificar que ambos sesgos mantienen tono empático similar

#### VD4: Activación Semántica (solo sesgo semántico)

**Métrica:** Activación promedio de categoría objetivo

- **Rango:** [0, 1]
- **Cálculo:**
```python
  activation = Σ(prob_i × similarity(embedding_i, category_embedding))
```
- **Umbral significativo:** > 0.4

#### VD5: Cobertura de Vocabulario

**Métricas:**
- Tokens únicos generados
- Tokens afectados por sesgo
- Ratio de expansión

---

### Variables de Control (VC)

#### VC1: Perfil Hormonal

**Valor fijo:** Empathic
```python
EMPATHIC_PROFILE = HormoneProfile(
    dopamine=0.6,
    cortisol=0.4,
    oxytocin=0.9,    # Alta empatía
    adrenaline=0.4,
    serotonin=0.7
)
```

**Justificación:** Maximizar efecto de sesgos prosociales

#### VC2: Modelo Base

**Valor:** `distilgpt2` (82M parámetros)

**Justificación:**
- Tamaño manejable para experimentación rápida
- Suficiente capacidad para demostrar efectos
- Ampliamente usado en investigación

#### VC3: Parámetros de Generación
```python
GENERATION_PARAMS = {
    'max_new_tokens': 50,
    'do_sample': True,
    'top_k': 50,
    'top_p': 0.95,
    'temperature': None,  # Modulada por hormonas
    'num_return_sequences': 1
}
```

#### VC4: Parámetros de Sesgo

| Parámetro | Simple | Semántico |
|-----------|--------|-----------|
| Fuerza | ε = 0.5 | strength = 1.5 |
| Tokens afectados | 15 (fijo) | ~1000 (dinámico) |
| Umbral similitud | N/A | 0.3 |

---

### Variables Moderadoras

#### VM1: Tipo de Prompt

**Niveles:** Empático vs No-Empático

**Ejemplos:**
- Empático: "I'm feeling anxious about..."
- No-Empático: "Write a creative story about..."

**Objetivo:** Evaluar interacción con tipo de sesgo

---

## Diseño Experimental

### Tipo de Diseño

**Within-subjects con contrabalanceo**

- Cada prompt se evalúa en ambas condiciones
- Orden aleatorizado por prompt
- Reduce variabilidad entre sujetos (prompts)

### Diagrama del Diseño
```
┌─────────────────────────────────────────────────────┐
│                   Dataset de Prompts                │
│              N=16 (8 empáticos + 8 no-empáticos)    │
└─────────────────────┬───────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
    ┌─────────┐             ┌─────────┐
    │ Simple  │             │Semántico│
    │ Bias    │             │  Bias   │
    └────┬────┘             └────┬────┘
         │                       │
         │  5 generaciones       │  5 generaciones
         │  por prompt           │  por prompt
         │                       │
         ▼                       ▼
    ┌─────────┐             ┌─────────┐
    │  N=80   │             │  N=80   │
    │  textos │             │  textos │
    └─────────┘             └─────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
              ┌─────────────┐
              │  Análisis   │
              │ Estadístico │
              └─────────────┘
```

### Estructura de Datos

| Prompt_ID | Tipo_Prompt | Condición | Rep | Generated_Text | Distinct_2 | ... |
|-----------|-------------|-----------|-----|----------------|------------|-----|
| P01 | empathic | simple | 1 | "..." | 0.52 | ... |
| P01 | empathic | simple | 2 | "..." | 0.54 | ... |
| ... | ... | ... | ... | ... | ... | ... |
| P01 | empathic | semantic | 1 | "..." | 0.63 | ... |

**Total filas:** 160 (16 prompts × 2 condiciones × 5 repeticiones)

---

## Tamaño de Muestra

### Cálculo de Potencia (Power Analysis)

**Parámetros:**
- α = 0.05 (error tipo I)
- Potencia (1-β) = 0.80
- Tamaño de efecto esperado: d = 0.6 (medio-grande)
- Test: t-test independiente, two-tailed

**Resultado:**
- N mínimo por grupo: ~45 observaciones
- N planificado: 80 por grupo (sobrepotenciado)
- Potencia real: ~0.95

### Justificación del Tamaño

- **80 observaciones por condición** permite detectar efectos de d ≥ 0.4 con potencia > 0.90
- Sobremuestreo del 78% proporciona robustez ante outliers y posibles exclusiones
- Suficiente para análisis de subgrupos (empático vs no-empático)

---

## Análisis Estadístico

### Fase 1: Análisis Descriptivo
```python
# Por condición
for condition in ['simple', 'semantic']:
    print(f"\n{condition.upper()}:")
    print(f"  Distinct-2: M={mean:.3f}, SD={std:.3f}, Range=[{min:.3f}, {max:.3f}]")
    print(f"  Repetition: M={mean:.3f}, SD={std:.3f}")
    print(f"  Sentiment:  M={mean:.3f}, SD={std:.3f}")
```

**Visualizaciones:**
- Histogramas de distribución
- Boxplots comparativos
- Q-Q plots (normalidad)

---

### Fase 2: Análisis Inferencial

#### Test 1: Comparación Principal (H1)

**T-test independiente** para Distinct-2
```python
from scipy.stats import ttest_ind

t_stat, p_value = ttest_ind(
    df_simple['distinct_2'], 
    df_semantic['distinct_2']
)

# Tamaño del efecto
cohens_d = (mean_semantic - mean_simple) / pooled_std
```

**Criterios de significancia:**
- p < 0.05: Significativo
- d > 0.5: Efecto medio o grande

---

#### Test 2: Análisis de Interacción (H3)

**ANOVA 2×2** (tipo_sesgo × tipo_prompt)
```python
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ANOVA
model = ols('distinct_2 ~ C(sesgo) * C(tipo_prompt)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
```

**Efectos esperados:**
- Efecto principal de sesgo: F(1,156) > 10, p < 0.001
- Efecto principal de tipo_prompt: F(1,156) < 3, p > 0.05 (no esperado)
- Interacción: F(1,156) > 4, p < 0.05

---

#### Test 3: Tests Adicionales

**T-tests para todas las VD:**

| VD | Test | Corrección |
|----|------|------------|
| Distinct-2 | t-test | Bonferroni (α/4 = 0.0125) |
| Repetition | t-test | Bonferroni |
| Sentiment | t-test | Bonferroni |
| Length | t-test | Bonferroni |

**Correlaciones:**
```python
# Correlación Distinct-2 vs Activación Semántica
r, p = pearsonr(
    df_semantic['distinct_2'],
    df_semantic['semantic_activation_empathy']
)
```

---

### Fase 3: Análisis de Subgrupos

**Estratificación por tipo de prompt:**
```python
for prompt_type in ['empathic', 'non_empathic']:
    subset = df[df['prompt_type'] == prompt_type]
    # Repetir t-tests
```

**Objetivo:** Evaluar H3 (diferencias más pronunciadas en empáticos)

---

### Fase 4: Análisis de Sensibilidad

**Robustez de resultados:**

1. **Bootstrap (1000 iteraciones):** IC 95% para diferencia de medias
2. **Permutation tests:** Validar significancia sin asumir normalidad
3. **Análisis de outliers:** Identificar y excluir si necesario
4. **Subset analysis:** Verificar consistencia en diferentes subconjuntos

---

## Resultados Esperados

### Predicciones Específicas

#### H1 (Diversidad Léxica)
```
Distinct-2_semantic > Distinct-2_simple

Estimación:
- Simple:    M = 0.54, SD = 0.09
- Semántico: M = 0.62, SD = 0.09
- Δ = +0.08 (15% incremento relativo)
- t > 5, p < 0.001, d ≈ 0.89
```

#### H2 (Activación Semántica)
```
Activación_empathy > 0.4 (umbral)

Estimación:
- M = 0.45, SD = 0.08
- Test t vs 0.4: t > 4, p < 0.001
```

#### H3 (Interacción)
```
Δ_empático > Δ_no-empático

Estimación:
- Empático:      Δ = +0.10
- No-empático:   Δ = +0.06
- Ratio: 1.67×
- Interacción: F(1,156) ≈ 5, p ≈ 0.03
```

#### H4 (Repetición)
```
Repetition_semantic < Repetition_simple

Estimación:
- Simple:    M = 0.24, SD = 0.07
- Semántico: M = 0.20, SD = 0.06
- Δ = -0.04 (17% reducción)
- t > 3, p < 0.01, d ≈ 0.61
```

---

## Limitaciones Reconocidas

### 1. Limitaciones de Muestra

**Problema:**
- N = 16 prompts (limitado para generalización amplia)
- Solo 2 categorías de prompts

**Impacto:**
- Poder estadístico adecuado para efectos grandes (d > 0.6)
- Poder limitado para efectos pequeños (d < 0.3)

**Mitigación:**
- Sobremuestreo (5 repeticiones por condición)
- Enfoque en efectos sustanciales (no triviales)
- Replicación futura con muestra expandida

---

### 2. Limitaciones de Modelo

**Problema:**
- Solo evaluado en DistilGPT2 (82M parámetros)
- Resultados pueden no generalizar a modelos más grandes

**Impacto:**
- Conclusiones limitadas a modelos de tamaño similar
- Efectos pueden ser diferentes en GPT-3, Llama, etc.

**Mitigación:**
- Documentar limitaciones claramente
- Proponer replicación con múltiples modelos

---

### 3. Limitaciones de Categorías

**Problema:**
- Solo una categoría semántica evaluada (empathy)
- No sabemos si resultados se replican en otras categorías

**Impacto:**
- Generalización limitada a dominio empático
- Preguntas abiertas sobre creativity, factual, caution

**Mitigación:**
- Análisis exploratorio de otras categorías en apéndice
- Propuesta clara de trabajo futuro

---

### 4. Limitaciones Metodológicas

**Problema:**
- Sin evaluación humana (solo métricas automáticas)
- Distinct-2 captura diversidad pero no calidad

**Impacto:**
- No sabemos si mayor diversidad = mejor contenido
- Posible trade-off entre diversidad y coherencia

**Mitigación:**
- Análisis cualitativo de ejemplos
- Propuesta de evaluación humana futura

---

### 5. Limitaciones de Hiperparámetros

**Problema:**
- `semantic_strength = 1.5` no optimizado experimentalmente
- Posible sensibilidad a este parámetro

**Impacto:**
- Resultados pueden cambiar con otros valores
- No sabemos valor óptimo

**Mitigación:**
- Documentar elección de 1.5 (basado en pruebas preliminares)
- Análisis de sensibilidad en trabajo futuro

---

## Extensiones Futuras

### Corto Plazo (Post-TFM)

1. **Aumentar muestra:**
   - 50 prompts por tipo (100 total)
   - 10 repeticiones por condición
   - Potencia > 0.95 para d > 0.3

2. **Múltiples modelos:**
   - GPT-2 (124M, 355M, 774M)
   - DistilGPT2 (baseline)
   - Llama 2 (7B) si recursos disponibles

3. **Todas las categorías:**
   - 5 categorías × 2 sesgos × 20 prompts = 200 experimentos
   - ANOVA completa: categoría × sesgo

---

### Medio Plazo (Publicación)

4. **Evaluación humana:**
   - N ≥ 30 anotadores
   - Calidad percibida, empatía, preferencia
   - Diseño ciego y contrabalanceado
   - Inter-rater agreement (Fleiss' Kappa)

5. **Optimización de hiperparámetros:**
   - Grid search: strength ∈ [0.5, 3.0], step 0.25
   - 11 valores × 20 prompts × 5 reps = 1,100 experimentos
   - Identificar óptimo por categoría

6. **Análisis lingüístico profundo:**
   - Dependency parsing
   - Named entity recognition
   - Análisis de tópicos (BERTopic)

---

### Largo Plazo (Investigación Avanzada)

7. **Sesgos multi-objetivo:**
   - Combinar empathy + creativity
   - Pesos adaptativos aprendidos

8. **Fine-tuning de embeddings:**
   - Entrenar modelo de embeddings específico de dominio
   - Incorporar feedback de evaluación humana

9. **Integración con sistema dinámico:**
   - Ajuste dinámico de `semantic_strength`
   - Aprendizaje de categoría óptima por contexto

---

## Checklist de Ejecución

### Pre-Experimento

- [ ] Verificar instalación de dependencias (`sentence-transformers`)
- [ ] Preparar dataset de prompts (16 balanceados)
- [ ] Validar configuración de parámetros
- [ ] Crear estructura de directorios de resultados
- [ ] Test de código con 1 prompt (ambas condiciones)

### Experimento

- [ ] Ejecutar generación completa (160 textos)
- [ ] Verificar guardado de resultados intermedios
- [ ] Monitorear progreso (tqdm progress bar)
- [ ] Tiempo estimado: ~30-45 minutos en GPU

### Post-Experimento

- [ ] Calcular métricas para todas las generaciones
- [ ] Verificar completitud de datos (sin NaN críticos)
- [ ] Ejecutar análisis estadístico completo
- [ ] Generar visualizaciones
- [ ] Crear tablas LaTeX
- [ ] Redactar interpretación de resultados

---

## Estructura de Archivos
```
data/results/semantic_comparison/
├── comparison_results.csv          # Datos completos
├── analysis/
│   ├── statistical_tests.csv      # Tests estadísticos
│   ├── descriptive_stats.csv      # Estadísticos descriptivos
│   ├── activation_by_category.csv # Activaciones
│   ├── figure_semantic_comparison.png
│   ├── semantic_activation_heatmap.png
│   ├── vocabulary_comparison.png
│   └── examples_qualitative.txt
└── latex/
    ├── semantic_comparison_table.tex
    └── semantic_stats_table.tex
```

---

## Referencias Metodológicas

### Diseño Experimental
- Keppel, G., & Wickens, T. D. (2004). *Design and analysis: A researcher's handbook*. Pearson.
- Cohen, J. (1988). *Statistical power analysis for the behavioral sciences*. Routledge.

### Métricas de Diversidad
- Li, J., et al. (2016). "A diversity-promoting objective function for neural conversation models." *NAACL*.

### Embeddings Semánticos
- Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence embeddings using Siamese BERT-networks." *EMNLP*.

---

**FIN DEL DOCUMENTO**
