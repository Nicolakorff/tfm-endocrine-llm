# 🔬 Diseño Experimental: Sistema Dinámico (Fase 3)

**Versión:** 1.0  
**Fecha:** Enero 2025  
**Estado:** Completo y validado

---

## 📋 Objetivo

Evaluar si el **sistema de actualización hormonal dinámica** produce resultados cualitativamente diferentes y superiores a los **perfiles hormonales estáticos** en términos de diversidad léxica, adaptación contextual y calidad del contenido.

---

## 🎯 Hipótesis

### H1: Diversidad Léxica (Principal)

**El sistema dinámico producirá mayor diversidad léxica que el sistema estático.**

- **Justificación:** La adaptación hormonal permite exploración variable según contexto
- **Dirección:** Dinámico > Estático
- **Criterio:** Diferencia significativa en Distinct-2 (p < 0.05, d > 0.3)

### H2: Cambios Hormonales

**Los niveles hormonales mostrarán cambios significativos durante la generación.**

- **Medición:** `total_hormone_change` > 0.10 (umbral mínimo)
- **Esperado:** Media de 0.15-0.25 por generación

### H3: Adaptación Contextual

**Los cambios hormonales serán contexto-dependientes según categoría de prompt.**

- **Medición:** Diferencias significativas en cambios hormonales entre categorías
- **Test:** ANOVA 1-way (categoría → cambio hormonal)

### H4: Reducción de Repetición

**El sistema dinámico reducirá la tasa de repetición al ajustar serotonina.**

- **Criterio:** Diferencia significativa (p < 0.05, d > 0.3)

---

## 📊 Variables

### Variable Independiente (VI)

**Tipo de Sistema** (2 niveles)

| Nivel | Descripción | Configuración |
|-------|-------------|---------------|
| **Estático** | Perfiles fijos | `dynamic=False` |
| **Dinámico** | Actualización en tiempo real | `dynamic=True`, `learning_rate=0.15` |

### Variables Dependientes (VD)

1. **VD1: Diversidad Léxica** (Distinct-2)
2. **VD2: Tasa de Repetición** (bigrams repetidos)
3. **VD3: Polaridad del Sentimiento**
4. **VD4: Cambio Hormonal Total** (solo dinámico)
   - `total_hormone_change = Σ|Δ_hormona|`
5. **VD5: Deltas por Hormona** (dopamine, cortisol, oxytocin, adrenaline, serotonin)

### Variables de Control (VC)

- **VC1: Modelo Base** - DistilGPT2 (82M parámetros)
- **VC2: Longitud** - `max_new_tokens=50`
- **VC3: Update Interval** - 5 tokens
- **VC4: Parámetros de Generación** - `top_k=50`, `top_p=0.95`

---

## 🎲 Diseño Experimental

### Tipo de Diseño

**Between-subjects con matching**

- Cada prompt genera con AMBOS sistemas
- Matching por perfil base (neutral, creative, empathic)
- Total: 6 perfiles (3 estáticos + 3 dinámicos equivalentes)

### Perfiles Evaluados

| Perfil Estático | Perfil Dinámico Equivalente |
|-----------------|----------------------------|
| `neutral` (0.5, 0.5, 0.5, 0.5, 0.5) | `dynamic_neutral` (dynamic=True, lr=0.1) |
| `creative` (0.9, 0.3, 0.5, 0.6, 0.5) | `dynamic_creative` (dynamic=True, lr=0.15) |
| `empathic` (0.6, 0.4, 0.9, 0.4, 0.7) | `dynamic_empathic` (dynamic=True, lr=0.15) |

### Estructura del Experimento

```
Dataset (200 prompts)
├── Categoría: Creative (50)
│   ├── Muestra: 10 prompts
│   │   ├── neutral_static × 3 reps
│   │   ├── neutral_dynamic × 3 reps
│   │   ├── creative_static × 3 reps
│   │   ├── creative_dynamic × 3 reps
│   │   ├── empathic_static × 3 reps
│   │   └── empathic_dynamic × 3 reps
│   └── ...
├── Categoría: Empathetic (50)
│   └── [igual estructura]
├── Categoría: Factual (50)
│   └── [igual estructura]
└── Categoría: Reasoning (50)
    └── [igual estructura]

TOTAL: 40 prompts × 6 perfiles × 3 reps = 720 generaciones
```

---

## 📏 Tamaño de Muestra

### Cálculo de Potencia

**Parámetros:**
- α = 0.05
- Potencia (1-β) = 0.80
- Effect size esperado: d = 0.4 (medio)
- Test: t-test independiente

**Resultado:**
- N mínimo por grupo: ~100 observaciones
- N planificado: 360 por grupo (estático/dinámico)
- Potencia real: >0.95

---

## 📊 Análisis Estadístico

### Fase 1: Análisis Descriptivo

```python
# Por tipo de sistema
for system in ['static', 'dynamic']:
    subset = df[df['is_dynamic'] == (system == 'dynamic')]
    print(f"{system}:")
    print(f"  Distinct-2: M={mean:.3f}, SD={std:.3f}")
    print(f"  Repetition: M={mean:.3f}, SD={std:.3f}")
    if system == 'dynamic':
        print(f"  Total Change: M={mean:.3f}, SD={std:.3f}")
```

### Fase 2: Comparación Estático vs Dinámico

**T-test independiente** para cada métrica:

```python
from scipy.stats import ttest_ind

static_vals = df_static['distinct_2']
dynamic_vals = df_dynamic['distinct_2']

t_stat, p_value = ttest_ind(static_vals, dynamic_vals)
cohens_d = (dynamic_vals.mean() - static_vals.mean()) / pooled_std
```

**Criterios:**
- p < 0.05: Significativo
- d > 0.3: Efecto pequeño-medio
- d > 0.5: Efecto medio-grande

### Fase 3: Análisis de Cambios Hormonales

```python
# Por hormona
for hormone in ['dopamine', 'cortisol', 'oxytocin', 'adrenaline', 'serotonin']:
    delta_col = f'delta_{hormone}'
    values = df_dynamic[delta_col].dropna()
    
    # One-sample t-test vs 0
    t_stat, p_value = stats.ttest_1samp(values, 0)
```

### Fase 4: Análisis por Categoría

**ANOVA 1-way:** categoría → total_hormone_change

```python
categories = df_dynamic['category'].unique()
groups = [df_dynamic[df_dynamic['category'] == cat]['total_hormone_change'] 
          for cat in categories]

F_stat, p_value = stats.f_oneway(*groups)
```

---

## 📈 Resultados Esperados

### H1: Diversidad Léxica

```
Distinct-2:
  Estático:  M = 0.58, SD = 0.09
  Dinámico:  M = 0.61, SD = 0.09
  Δ = +0.03 (+5.2%)
  t > 2.5, p < 0.05, d ≈ 0.33
```

### H2: Cambios Hormonales

```
Total Change (solo dinámico):
  M = 0.18, SD = 0.09
  Rango: [0.05, 0.40]
  % con cambio > 0.10: ~70%
```

### H3: Por Categoría

```
Cambio Hormonal por Categoría:
  Empathetic:  M = 0.20 (mayor oxitocina)
  Creative:    M = 0.19 (mayor dopamina)
  Factual:     M = 0.15 (más estable)
  Reasoning:   M = 0.17 (moderado)
  
  F(3, 356) ≈ 4.5, p < 0.01
```

---

## ⚠️ Limitaciones

1. **Modelo pequeño:** DistilGPT2 (82M) - resultados pueden variar en modelos grandes
2. **Learning rate fijo:** 0.15 no optimizado experimentalmente
3. **Update interval fijo:** 5 tokens (no explorado rango 3-10)
4. **Sin evaluación humana:** Solo métricas automáticas
5. **Muestra de prompts:** 40 de 200 (20% del dataset)

---

## 🔮 Trabajo Futuro

1. **Grid search** de hiperparámetros:
   - `learning_rate` ∈ [0.05, 0.30], step 0.05
   - `update_interval` ∈ [3, 10]

2. **Modelos más grandes:**
   - GPT-2 Medium/Large
   - Llama 2 (7B)

3. **Evaluación humana:**
   - Calidad percibida
   - Coherencia
   - Preferencia

4. **Análisis de trayectorias:**
   - Clustering de patrones hormonales
   - Identificación de arquetipos de adaptación

---

## 📁 Archivos Generados

**Datos:**
- `data/results/phase3_dynamic_results.csv` - Resultados completos
- `data/results/dynamic_analysis/static_vs_dynamic_comparison.csv` - Comparación

**Visualizaciones:**
- `static_vs_dynamic_boxplots.png`
- `hormone_change_distribution.png`
- `hormone_changes_by_category.png`
- `example_trajectory_*.png`

---

## 🎓 Implicaciones para el TFM

### Contribución Principal

Demuestra que:
1. Los sistemas dinámicos **adaptan automáticamente** el comportamiento
2. La adaptación es **contexto-dependiente** y **medible**
3. Existe **trade-off** entre diversidad y estabilidad controlable

### Integración con Fases Previas

- **Fase 1:** Valida efectos de hormonas individuales
- **Fase 2:** Perfiles estáticos como baseline
- **Fase 3:** Sistema dinámico como extensión natural

---

**Preparado para:** TFM - Máster en Grandes Modelos de Lenguaje  
**Estado:** Listo para ejecución
