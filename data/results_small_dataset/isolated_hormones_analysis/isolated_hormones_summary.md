# Análisis de Hormonas Individuales (Aisladas)

## Resumen Ejecutivo

Este análisis evalúa el efecto de cada hormona individual (elevada a 0.9)
comparada contra el perfil baseline (todas en 0.5).

## Efectos Significativos Detectados


### Dopamine

- **Perplexity** aumentó 31.3% (Δ=+5.174, d=0.62, p=0.0000)
- **Distinct 2** aumentó 1.9% (Δ=+0.019, d=0.43, p=0.0000)
- **Repetition Rate** disminuyó 62.7% (Δ=-0.008, d=-0.26, p=0.0106)

### Cortisol

- **Perplexity** disminuyó 12.1% (Δ=-2.000, d=-0.42, p=0.0000)
- **Repetition Rate** disminuyó 81.4% (Δ=-0.011, d=-0.35, p=0.0006)

### Oxytocin
- No se detectaron efectos significativos (p < 0.05)

### Adrenaline

- **Perplexity** disminuyó 26.8% (Δ=-4.426, d=-0.60, p=0.0000)
- **Distinct 2** disminuyó 2.3% (Δ=-0.022, d=-0.33, p=0.0010)
- **Repetition Rate** aumentó 88.2% (Δ=+0.012, d=0.21, p=0.0326)

### Serotonin

- **Perplexity** aumentó 19.1% (Δ=+3.162, d=0.58, p=0.0000)
- **Distinct 2** aumentó 1.1% (Δ=+0.011, d=0.24, p=0.0167)
- **Repetition Rate** disminuyó 51.2% (Δ=-0.007, d=-0.20, p=0.0412)


## Interpretación General

Número de efectos significativos por hormona:

- **Adrenaline**: 3 métricas afectadas
- **Dopamine**: 3 métricas afectadas
- **Serotonin**: 3 métricas afectadas
- **Cortisol**: 2 métricas afectadas


## Figuras Generadas

1. `hormone_effects_barplot.png` - Diferencias por hormona
2. `hormone_effects_heatmap.png` - Mapa de calor de efectos
3. `hormone_effects_boxplots.png` - Distribuciones comparativas

## Datos

- `hormone_effects_summary.csv` - Resultados completos
- `isolated_hormones_table.tex` - Tabla para LaTeX

---

**Conclusión:** Este análisis permite identificar el efecto específico de cada
hormona individual sobre diferentes aspectos de la generación de texto.
