# Diseño Experimental: Hormonas Aisladas

## Sistema de Neuromodulación Endocrina para LLMs - Fase 2

**Versión:** 2.0  
**Fecha:** Diciembre 2024  
**Estado:** Completado y validado

---

## Objetivo

Evaluar el **efecto individual de cada hormona artificial** sobre la generación de texto, aislando su contribución independiente antes de evaluar combinaciones complejas.

**Pregunta de investigación:**  
¿Cada hormona artificial produce efectos medibles, significativos y específicos sobre dimensiones lingüísticas?

---

## Hipótesis

### H1: Dopamina → Diversidad

**Dopamina alta aumentará la diversidad léxica (Distinct-2).**

- **Mecanismo:** ↑ Temperatura → ↑ Exploración → ↑ Variabilidad
- **Predicción:** Δ_distinct-2 > +0.05 vs baseline
- **Criterio:** p < 0.01, d > 0.5

### H2: Cortisol → Conservadurismo

**Cortisol alto reducirá la perplejidad.**

- **Mecanismo:** Suavizado de distribución → ↓ Sorpresa
- **Predicción:** Δ_perplexity < -10%

### H3: Serotonina → Estabilidad

**Serotonina alta reducirá la tasa de repetición.**

- **Mecanismo:** Penalización de n-gramas repetidos
- **Predicción:** Δ_repetition < -0.03

### H4: Adrenalina → Intensidad

**Adrenalina alta reducirá la longitud.**

- **Mecanismo:** Amplificación + boost EOS
- **Predicción:** ↓ Longitud promedio

### H5: Oxitocina → Empatía

**Oxitocina alta aumentará palabras empáticas.**

- **Mecanismo:** Boost de tokens prosociales
- **Predicción:** ↑ Empathy word score

---

## Variables

### Variable Independiente

**Perfil Hormonal** (6 niveles)

| Perfil | D | C | O | A | S |
|--------|---|---|---|---|---|
| baseline | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| high_dopamine | **0.9** | 0.5 | 0.5 | 0.5 | 0.5 |
| high_cortisol | 0.5 | **0.9** | 0.5 | 0.5 | 0.5 |
| high_oxytocin | 0.5 | 0.5 | **0.9** | 0.5 | 0.5 |
| high_adrenaline | 0.5 | 0.5 | 0.5 | **0.9** | 0.5 |
| high_serotonin | 0.5 | 0.5 | 0.5 | 0.5 | **0.9** |

### Variables Dependientes

1. **Distinct-2** - Diversidad léxica
2. **Perplejidad** - Sorpresa del modelo
3. **Repetición** - Tasa de bigramas repetidos
4. **Longitud** - Número de tokens
5. **Polaridad** - Sentimiento
6. **Empathy Score** - Palabras empáticas

---

## Diseño

**Tipo:** Between-subjects factorial

**Estructura:**
- 6 perfiles × 200 prompts × 3 reps = **3,600 generaciones**

**Contrabalanceo:**
- Orden aleatorizado
- Seed diferente por repetición

---

## Tamaño de Muestra

- N por grupo: 600
- Potencia: >0.99 para d=0.5
- Sobremuestreo: 14×

---

## Análisis

1. **Descriptivo:** Media, SD por perfil
2. **T-tests:** Cada hormona vs baseline (Bonferroni: α=0.01)
3. **ANOVA:** Efecto global de perfil
4. **Post-hoc:** Tukey HSD si ANOVA significativo

---

## Resultados Esperados

| Hormona | Distinct-2 | Perplejidad | Efecto Principal |
|---------|------------|-------------|------------------|
| Baseline | 0.54 | 25.0 | - |
| Dopamine | **0.62** | 28.0 | ↑ Diversidad |
| Cortisol | 0.50 | **22.0** | ↓ Perplejidad |
| Serotonin | 0.55 | 26.0 | ↓ Repetición |
| Adrenaline | 0.53 | **21.5** | ↓ Longitud |

---

**Ver resultados en:** [results_phase1.md](results_phase1.md)

**FIN DEL DOCUMENTO**
