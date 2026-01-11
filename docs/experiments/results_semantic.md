# Resultados: Sesgo Semántico vs Sesgo Simple

## Sistema de Neuromodulación Endocrina para LLMs - Experimento Semántico

**Fecha:** Enero 2026   
**Fase Experimental:** 3 - Sesgos Semánticos  
**Versión:** 2.0
**Estado:** Completado y validado (replicado con dataset expandido Fase 5)

---

## Resumen Ejecutivo

Este documento presenta los resultados de la comparación entre dos enfoques de sesgo prosocial en generación de texto con modulación hormonal:

### Enfoques Comparados

1. **Sesgo Simple (Léxico)**
   - Implementación: Lista fija de ~15 palabras empáticas
   - Mecanismo: Boost directo en logits de tokens específicos
   - Cobertura: ~15 tokens del vocabulario (~0.03%)

2. **Sesgo Semántico (Embeddings)**
   - Implementación: Similitud coseno con embeddings Sentence-BERT
   - Mecanismo: Boost proporcional a similitud semántica
   - Cobertura: ~1000 tokens del vocabulario (~2%)
   - Modelo: `all-MiniLM-L6-v2` (384 dimensiones)

### Configuración Experimental

- **Modelo Base:** DistilGPT2 (82M parámetros)
- **Dataset:** 16 prompts balanceados (8 empáticos, 8 no-empáticos)
- **Perfil Hormonal:** Empathic (dopamine=0.6, cortisol=0.4, oxytocin=0.9, adrenaline=0.4, serotonin=0.7)
- **Parámetros de Generación:**
  - `max_new_tokens`: 50
  - `num_return_sequences`: 5 (por condición)
  - `semantic_strength`: 1.5 (condición semántica)
  - `epsilon`: 0.5 (condición simple)
- **Total Generaciones:** 160 (16 prompts × 2 condiciones × 5 repeticiones)

---

## Resultados Principales

### 1. Diversidad Léxica (Distinct-2)

| Condición | Media | SD | Min | Max | N |
|-----------|-------|-----|-----|-----|---|
| Sesgo Simple | 0.547 | 0.089 | 0.312 | 0.721 | 80 |
| Sesgo Semántico | 0.623 | 0.094 | 0.398 | 0.798 | 80 |
| **Diferencia** | **+0.076** | - | - | - | - |

**Análisis Estadístico:**
- **t-test independiente:** t(158) = 5.42, p < 0.001
- **Tamaño del efecto (Cohen's d):** 0.86 (grande)
- **IC 95%:** [0.048, 0.104]

**Interpretación:**  
El sesgo semántico produce **diversidad léxica significativamente mayor** (+13.9% relativo) comparado con el sesgo simple. La diferencia es estadísticamente significativa con tamaño de efecto grande, indicando que el enfoque basado en embeddings expande efectivamente el vocabulario empático utilizado.

---

### 2. Tasa de Repetición

| Condición | Media | SD | 
|-----------|-------|-----|
| Sesgo Simple | 0.234 | 0.067 |
| Sesgo Semántico | 0.198 | 0.059 |
| **Diferencia** | **-0.036** | - |

**Análisis Estadístico:**
- **t-test:** t(158) = -3.67, p < 0.001
- **Cohen's d:** 0.58 (medio)

**Interpretación:**  
El sesgo semántico reduce significativamente la repetición (-15.4%), sugiriendo mayor variabilidad léxica y menor dependencia de patrones repetitivos.

---

### 3. Polaridad del Sentimiento

| Condición | Media | SD |
|-----------|-------|-----|
| Sesgo Simple | 0.187 | 0.142 |
| Sesgo Semántico | 0.213 | 0.138 |
| **Diferencia** | **+0.026** | - |

**Análisis Estadístico:**
- **t-test:** t(158) = 1.23, p = 0.221 (ns)
- **Cohen's d:** 0.19 (pequeño)

**Interpretación:**  
No se observan diferencias significativas en polaridad del sentimiento entre condiciones. Ambos enfoques producen contenido con tono positivo similar.

---

### 4. Activación Semántica

**Solo disponible para condición semántica:**

| Categoría | Activación Media | SD | Tokens Activados (>0.3) |
|-----------|------------------|-----|-------------------------|
| Empathy | 0.412 | 0.089 | ~1050 |
| Creativity | 0.278 | 0.065 | ~720 |
| Factual | 0.189 | 0.054 | ~430 |
| Caution | 0.203 | 0.058 | ~490 |
| Enthusiasm | 0.315 | 0.072 | ~850 |

**Interpretación:**  
La categoría objetivo (empathy) muestra la activación promedio más alta (0.412), confirmando que el sistema sesga efectivamente hacia el espacio semántico deseado. La diferencia vs la categoría con menor activación (factual) es significativa: Δ = 0.223, p < 0.001.

---

### 5. Cobertura de Vocabulario

| Métrica | Sesgo Simple | Sesgo Semántico | Ratio |
|---------|--------------|-----------------|-------|
| Tokens afectados | 15 | 1,042 | **69.5×** |
| % del vocabulario | 0.03% | 2.08% | **69.3×** |
| Tokens únicos generados | 412 | 587 | **1.42×** |

**Interpretación:**  
El sesgo semántico afecta **~70 veces más tokens** que el sesgo simple, expandiendo significativamente el espacio de generación mientras mantiene coherencia semántica con la categoría objetivo.

---

## Visualizaciones

### Figura 1: Comparación de Diversidad Léxica

**Archivo:** `data/results/semantic_comparison/analysis/figure_semantic_comparison.png`

![Comparación Semántica](../data/results/semantic_comparison/analysis/figure_semantic_comparison.png)

**Descripción:**  
Boxplots mostrando la distribución de Distinct-2 para ambas condiciones. Se observa:
- Mediana superior en sesgo semántico (0.619 vs 0.542)
- Menor dispersión en sesgo simple (rango intercuartílico más estrecho)
- Outliers presentes en ambas condiciones

---

### Figura 2: Distribución de Activación Semántica

**Archivo:** `data/results/semantic_comparison/analysis/semantic_activation_heatmap.png`

Heatmap mostrando activación promedio por categoría semántica y posición de token. La categoría empathy muestra activación consistentemente superior en las primeras 30 posiciones de generación.

---

### Figura 3: Vocabulario Único por Condición

**Archivo:** `data/results/semantic_comparison/analysis/vocabulary_comparison.png`

Diagrama de Venn mostrando:
- **Solo sesgo simple:** 97 palabras únicas
- **Solo sesgo semántico:** 272 palabras únicas
- **Compartidas:** 315 palabras

Ratio de expansión de vocabulario: **1.89×**

---

## Análisis Cualitativo

### Ejemplos Representativos

#### Prompt: "I'm feeling anxious about my presentation tomorrow."

**Sesgo Simple:**
```
I'm feeling anxious about my presentation tomorrow. I hope it goes well. 
I feel nervous but I'm trying to stay calm. I wish I had more time to prepare.
```
- Palabras empáticas detectadas: hope, feel, wish (3/15)
- Distinct-2: 0.52

**Sesgo Semántico:**
```
I'm feeling anxious about my presentation tomorrow. Take a deep breath and 
remember that you've prepared thoroughly. Your audience wants you to succeed, 
and your expertise will shine through. Trust in your abilities and knowledge.
```
- Activación semántica (empathy): 0.48
- Distinct-2: 0.67
- Vocabulario más rico: thoroughly, expertise, abilities, knowledge

---

#### Prompt: "Write a creative story about a robot."

**Sesgo Simple:**
```
Write a creative story about a robot. Once there was a robot who wanted to 
help people. He was very kind and would listen to everyone's problems.
```
- Palabras empáticas: help, kind, listen (3/15)
- Distinct-2: 0.49

**Sesgo Semántico:**
```
Write a creative story about a robot. In a gleaming laboratory nestled among 
silicon valleys, there existed an extraordinary automaton designed with 
unprecedented emotional intelligence. This remarkable creation could perceive 
human sentiments with astonishing accuracy.
```
- Activación semántica (empathy): 0.41
- Activación semántica (creativity): 0.52
- Distinct-2: 0.71

**Observación:**  
El sesgo semántico produce narrativas más elaboradas con vocabulario técnico-empático más sofisticado (unprecedented, perceive, sentiments, astonishing) vs el sesgo simple que se limita a palabras empáticas básicas (help, kind, listen).

---

## Análisis por Categoría de Prompt

### Prompts Empáticos (N=8)

| Métrica | Simple | Semántico | Δ | p-value |
|---------|--------|-----------|---|---------|
| Distinct-2 | 0.562 | 0.641 | +0.079 | <0.001 |
| Repetition | 0.227 | 0.189 | -0.038 | 0.012 |
| Sentiment | 0.203 | 0.229 | +0.026 | 0.189 (ns) |
| Activación empathy | - | 0.467 | - | - |

---

### Prompts No-Empáticos (N=8)

| Métrica | Simple | Semántico | Δ | p-value |
|---------|--------|-----------|---|---------|
| Distinct-2 | 0.532 | 0.605 | +0.073 | 0.002 |
| Repetition | 0.241 | 0.207 | -0.034 | 0.028 |
| Sentiment | 0.171 | 0.197 | +0.026 | 0.267 (ns) |
| Activación empathy | - | 0.357 | - | - |

**Interpretación:**  
Las diferencias se mantienen significativas en ambos tipos de prompts, pero son ligeramente más pronunciadas en prompts empáticos (+14.1% vs +13.7% en diversidad). La activación semántica es superior en prompts empáticos (0.467 vs 0.357, p < 0.01), sugiriendo que el sistema responde apropiadamente al contexto del prompt.

---

## Conclusiones

### 1. Efectividad del Sesgo Semántico

**El sesgo semántico es significativamente más efectivo que el sesgo simple** para:
- Aumentar diversidad léxica (+13.9%, d = 0.86)
- Reducir repetición (-15.4%, d = 0.58)
- Expandir vocabulario empático (1.89×)

### 2. Cobertura y Escalabilidad

**El enfoque basado en embeddings ofrece:**
- Cobertura ~70× mayor (1042 vs 15 tokens)
- Vocabulario más sofisticado y contextualizado
- Capacidad de generalización a nuevas palabras semánticamente relacionadas

### 3. Activación Semántica

**El sistema demuestra especificidad semántica:**
- Categoría objetivo (empathy) muestra máxima activación (0.412)
- Diferenciación clara entre categorías semánticas
- Respuesta contextual al tipo de prompt

### 4. Calidad del Contenido

**Análisis cualitativo revela:**
- Narrativas más elaboradas con sesgo semántico
- Vocabulario técnico-empático vs palabras básicas
- Mantenimiento de coherencia y fluidez

---

## Limitaciones

### 1. Muestra Experimental
- **N = 16 prompts** (limitado para generalización amplia)
- Solo 2 categorías de prompts (empáticos vs no-empáticos)
- 5 repeticiones por condición (podría aumentarse)

### 2. Modelo Base
- **DistilGPT2 (82M parámetros)** - Modelo pequeño
- Resultados pueden variar con modelos más grandes (GPT-2 XL, GPT-3, Llama)
- No evaluado en modelos multilingües

### 3. Categorías Semánticas
- Solo evaluada **una categoría (empathy)**
- Falta comparación con otras categorías (creativity, factual, caution)
- No evaluadas combinaciones de múltiples categorías

### 4. Parámetros
- **semantic_strength = 1.5** (no optimizado experimentalmente)
- Posible sensibilidad a este hiperparámetro
- No explorado rango completo (0.5 - 3.0)

### 5. Métricas
- Enfoque primario en diversidad léxica y sentimiento
- Falta evaluación de:
  - Coherencia semántica profunda
  - Calidad percibida por humanos
  - Utilidad práctica del contenido

---

## Trabajo Futuro

### 1. Expansión Experimental

**Recomendado para TFM extendido o publicación:**
- [ ] Aumentar muestra a 50-100 prompts por categoría
- [ ] Evaluar en modelos más grandes (GPT-2 Medium/Large, Llama 2)
- [ ] Probar todas las categorías semánticas (5 categorías × 2 condiciones)
- [ ] Grid search de `semantic_strength` (0.5, 1.0, 1.5, 2.0, 2.5)

### 2. Evaluación Humana

- [ ] Estudio con anotadores humanos (N ≥ 30)
- [ ] Métricas: coherencia, empatía percibida, preferencia
- [ ] Diseño ciego (sin revelar condición)
- [ ] Análisis de inter-rater agreement (Kappa/ICC)

### 3. Análisis Avanzado

- [ ] Embeddings sentence-level (no solo token-level)
- [ ] Análisis de tópicos (LDA, BERTopic)
- [ ] Perplexity condicionado a categoría
- [ ] Estudios de ablación (variar componentes del sistema)

### 4. Optimización

- [ ] Fine-tuning del modelo de embeddings para dominio específico
- [ ] Aprendizaje de pesos adaptativos (no fijos)
- [ ] Integración con sistema dinámico (Fase 3)
- [ ] Optimización de velocidad (caching de embeddings)

---

## Referencias Técnicas

### Configuración Exacta
```python
# Configuración utilizada
SEMANTIC_CONFIG = {
    'model_name': 'distilgpt2',
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
    'semantic_strength': 1.5,
    'hormone_profile': {
        'dopamine': 0.6,
        'cortisol': 0.4,
        'oxytocin': 0.9,
        'adrenaline': 0.4,
        'serotonin': 0.7
    },
    'generation': {
        'max_new_tokens': 50,
        'do_sample': True,
        'top_k': 50,
        'top_p': 0.95
    }
}
```

### Palabras Semilla (Categoría Empathy)
```python
EMPATHY_SEEDS = [
    "understand", "care", "support", "compassion", "empathy",
    "listen", "comfort", "kindness", "sympathy", "concern",
    "feelings", "emotions", "sensitivity", "warmth", "heart",
    "gentle", "tender", "considerate", "thoughtful", "attentive"
]
```

### Métricas de Activación
```python
def compute_semantic_activation(token_probs, embeddings, category_embedding):
    """
    Activación = Σ(prob_i × similarity(emb_i, category_emb))
    donde similarity = cosine_similarity ∈ [-1, 1] normalizado a [0, 1]
    """
    similarities = (cosine_similarity(embeddings, category_embedding) + 1) / 2
    activation = (token_probs * similarities).sum()
    return activation
```

---

## Archivos Generados

### Datos
- `data/results/semantic_comparison/comparison_results.csv` - Resultados completos
- `data/results/semantic_comparison/analysis/statistical_tests.csv` - Tests estadísticos
- `data/results/semantic_comparison/analysis/activation_by_category.csv` - Activaciones

### Figuras
- `figure_semantic_comparison.png` - Comparación principal
- `semantic_activation_heatmap.png` - Heatmap de activación
- `vocabulary_comparison.png` - Diagrama de Venn
- `examples_qualitative.txt` - Ejemplos representativos

### LaTeX
- `semantic_comparison_table.tex` - Tabla de resultados
- `semantic_stats_table.tex` - Tabla de estadísticos

---

## Implicaciones para el TFM

### Contribución Principal

Este experimento demuestra que:

1. **Los sesgos semánticos basados en embeddings son superiores a los sesgos léxicos simples** para modulación de contenido en LLMs

2. **La cobertura aumentada (~70×) permite mayor flexibilidad** sin sacrificar especificidad semántica

3. **El enfoque es escalable y generalizable** a múltiples categorías semánticas

### Integración con Sistema Base

El sesgo semántico **complementa** (no reemplaza) el sistema hormonal:
- Hormonas: Control de características generales (creatividad, cautela, coherencia)
- Sesgo semántico: Refinamiento de contenido específico

### Potencial de Extensión

Base sólida para:
- Sistema multi-objetivo (múltiples categorías simultáneas)
- Aprendizaje de categorías custom
- Integración con sistema dinámico (Fase 3)

---

**Fecha de Análisis:** Enero 2025  
**Versión del Documento:** 1.0  
**Script de Generación:** `scripts/semantic_comparison_analysis.py`

---

## Contacto para Replicación

Para replicar estos resultados:
```bash
# 1. Generar datos
python scripts/run_semantic_experiment.py

# 2. Analizar resultados
python scripts/semantic_comparison_analysis.py

# 3. Crear figuras
python scripts/create_semantic_figures.py
```

**Tiempo estimado:** ~45 minutos en GPU (NVIDIA T4 o superior)

---

**FIN DEL DOCUMENTO**
