# Diseño Experimental: Comparación de Sesgos

## Objetivo

Evaluar si los sesgos semánticos basados en embeddings producen 
resultados cualitativamente diferentes a los sesgos simples basados 
en listas de tokens.

## Hipótesis

**H1:** El sesgo semántico producirá mayor diversidad léxica debido a 
que afecta a más tokens del vocabulario.

**H2:** El sesgo semántico producirá textos con mayor activación de 
la categoría objetivo (empathy) medida por similitud semántica.

**H3:** Las diferencias serán más pronunciadas en tareas empáticas 
que en tareas creativas.

## Variables

### Variable Independiente
- **Tipo de sesgo**: Simple vs Semántico (2 niveles)

### Variables Dependientes
1. Diversidad léxica (Distinct-2)
2. Polaridad de sentimiento
3. Tasa de repetición
4. Activación semántica (solo para sesgo semántico)

### Variables de Control
- Perfil hormonal: Empathic (fijo)
- Modelo base: GPT-2 (fijo)
- Longitud máxima: 60 tokens (fijo)
- Temperatura: Modulada por hormonas (fijo)

## Diseño

- **Tipo:** Within-subjects (mismo prompt, ambas condiciones)
- **N prompts:** 16 (8 empáticos + 8 creativos)
- **Generaciones/condición:** 5
- **Total generaciones:** 16 × 2 × 5 = 160

## Análisis Estadístico

1. **Descriptivo:** Media, SD por condición
2. **Inferencial:** t-test independiente para cada métrica
3. **Tamaño del efecto:** Cohen's d
4. **Visualización:** Boxplots, histogramas

## Resultados Esperados

- Diferencia significativa en distinct_2 (p < 0.05)
- Mayor activación semántica en categoría objetivo
- Efecto más fuerte en prompts empáticos

## Limitaciones

- Muestra de prompts limitada
- Un solo modelo (GPT-2)
- Una sola categoría semántica evaluada
