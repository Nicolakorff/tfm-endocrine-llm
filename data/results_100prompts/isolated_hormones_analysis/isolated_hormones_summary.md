# Análisis de Hormonas Individuales (Aisladas)

## Resumen Ejecutivo

Este análisis evalúa el efecto de cada hormona individual (elevada a 0.9)
comparada contra el perfil baseline (todas en 0.5).

## Efectos Significativos Detectados


### Dopamine

- **Perplexity** aumentó 22.4% (Δ=+4.279, d=0.26, p=0.0000)
- **Distinct 2** aumentó 1.0% (Δ=+0.010, d=0.23, p=0.0002)
- **Repetition Rate** disminuyó 40.3% (Δ=-0.005, d=-0.17, p=0.0066)

### Cortisol

- **Repetition Rate** disminuyó 84.3% (Δ=-0.010, d=-0.43, p=0.0000)
- **Distinct 2** aumentó 0.7% (Δ=+0.007, d=0.18, p=0.0055)

### Oxytocin
- No se detectaron efectos significativos (p < 0.05)

### Adrenaline

- **Distinct 2** disminuyó 3.2% (Δ=-0.031, d=-0.47, p=0.0000)
- **Perplexity** disminuyó 33.3% (Δ=-6.369, d=-0.36, p=0.0000)
- **Repetition Rate** aumentó 154.3% (Δ=+0.018, d=0.34, p=0.0000)

### Serotonin

- **Distinct 2** aumentó 1.5% (Δ=+0.014, d=0.36, p=0.0000)
- **Perplexity** aumentó 19.3% (Δ=+3.700, d=0.24, p=0.0001)
- **Repetition Rate** disminuyó 49.7% (Δ=-0.006, d=-0.22, p=0.0005)


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
