# Análisis de Hormonas Individuales (Aisladas)

## Resumen Ejecutivo

Este análisis evalúa el efecto de cada hormona individual (elevada a 0.9)
comparada contra el perfil baseline (todas en 0.5).

## Efectos Significativos Detectados


### Dopamine

- **Perplexity** aumentó 27.0% (Δ=+4.329, d=0.84, p=0.0000)
- **Distinct 2** aumentó 1.2% (Δ=+0.011, d=0.29, p=0.0042)
- **Repetition Rate** disminuyó 57.5% (Δ=-0.006, d=-0.22, p=0.0261)

### Cortisol

- **Repetition Rate** disminuyó 89.0% (Δ=-0.010, d=-0.37, p=0.0002)
- **Perplexity** disminuyó 7.5% (Δ=-1.202, d=-0.23, p=0.0208)

### Oxytocin
- No se detectaron efectos significativos (p < 0.05)

### Adrenaline

- **Perplexity** disminuyó 26.4% (Δ=-4.241, d=-0.91, p=0.0000)
- **Distinct 2** disminuyó 3.1% (Δ=-0.030, d=-0.51, p=0.0000)
- **Repetition Rate** aumentó 141.1% (Δ=+0.015, d=0.35, p=0.0005)

### Serotonin

- **Perplexity** aumentó 24.3% (Δ=+3.903, d=0.73, p=0.0000)


## Interpretación General

Número de efectos significativos por hormona:

- **Adrenaline**: 3 métricas afectadas
- **Dopamine**: 3 métricas afectadas
- **Cortisol**: 2 métricas afectadas
- **Serotonin**: 1 métricas afectadas


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
