# Índice de Figuras para TFM

## Figuras Principales

### Figura 1: Arquitectura del Sistema
**Archivo:** `data/results/tfm_figures/individual/fig1_architecture.png`
**Ubicación en TFM:** Capítulo 4 (Metodología)
**Descripción:** Diagrama que muestra la arquitectura completa del sistema de neuromodulación endocrina, incluyendo el modelo base, el vector hormonal y el procesador de logits.
**Tamaño recomendado:** Página completa o 0.8\textwidth
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/fig1_architecture.png}
\caption{Arquitectura del sistema de neuromodulación endocrina para LLMs}
\label{fig:architecture}
\end{figure}
```

### Figura 2: Perfiles Hormonales
**Archivo:** `data/results/tfm_figures/individual/fig2_hormone_profiles.png`
**Ubicación en TFM:** Capítulo 4 (Metodología) - Sección Perfiles
**Descripción:** Mapa de calor mostrando los niveles hormonales para cada perfil predefinido.
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.7\textwidth]{figures/fig2_hormone_profiles.png}
\caption{Perfiles hormonales predefinidos utilizados en los experimentos}
\label{fig:profiles}
\end{figure}
```

### Figura 3: Ejemplo de Generación
**Archivo:** `data/results/tfm_figures/individual/fig3_generation_example.png`
**Ubicación en TFM:** Capítulo 5 (Resultados) - Análisis Cualitativo
**Descripción:** Ejemplo comparativo de generación con diferentes perfiles hormonales.

### Figura 4: Distribución de Métricas
**Archivo:** `data/results/tfm_figures/individual/fig4_metrics_distribution.png`
**Ubicación en TFM:** Capítulo 5 (Resultados) - Estadísticas Descriptivas
**Descripción:** Histogramas de las métricas principales en todos los experimentos.

### Figura 5: Comparación Top Perfiles
**Archivo:** `data/results/tfm_figures/individual/fig5_top_profiles_comparison.png`
**Ubicación en TFM:** Capítulo 5 (Resultados) - Comparación de Perfiles
**Descripción:** Boxplots comparando los 6 perfiles hormonales más frecuentes.

### Figura 6: Figura Maestra (Resultados Integrados)
**Archivo:** `data/results/tfm_figures/master_figure.png`
**Ubicación en TFM:** Capítulo 5 (Resultados) - Inicio o Apéndice
**Descripción:** Figura comprehensiva que resume todos los resultados experimentales en 6 paneles.
**Tamaño:** Página completa o doble página
```latex
\begin{figure}[p]
\centering
\includegraphics[width=\textwidth]{figures/master_figure.png}
\caption{Resumen integrado de resultados experimentales: (A) Diversidad léxica por perfil, (B) Resultados ANOVA, (C) Tamaño de efecto, (D) Hormonas individuales, (E) Comparación semántica, (F) Distribución general de métricas}
\label{fig:master}
\end{figure}
```

## Figuras de Análisis Específico

### Figura 7: ANOVA Comparison
**Archivo:** `data/results/anova_analysis/anova_comparison.png`
**Ubicación:** Capítulo 5 - Análisis Inferencial

### Figura 8: Correlación Hormonas-Métricas
**Archivo:** `data/results/correlation_analysis/correlation_heatmap.png`
**Ubicación:** Capítulo 5 - Análisis de Correlación

### Figura 9: Efectos de Hormonas Individuales
**Archivo:** `data/results/isolated_hormones_analysis/hormone_effects_barplot.png`
**Ubicación:** Capítulo 5 - Hormonas Aisladas

### Figura 10: Comparación Semántica
**Archivo:** `data/results/semantic_comparison/analysis/figure_semantic_comparison.png`
**Ubicación:** Capítulo 5 - Experimento Semántico

## Guía de Uso

### Copiar figuras al documento LaTeX
```bash
# Crear carpeta de figuras en tu proyecto LaTeX
mkdir -p tfm_latex/figures

# Copiar figuras principales
cp data/results/tfm_figures/individual/*.pdf tfm_latex/figures/
cp data/results/tfm_figures/master_figure.pdf tfm_latex/figures/

# Copiar figuras de análisis
cp data/results/anova_analysis/*.pdf tfm_latex/figures/
cp data/results/correlation_analysis/*.pdf tfm_latex/figures/
cp data/results/isolated_hormones_analysis/*.pdf tfm_latex/figures/
```

### Template LaTeX para figuras
```latex
% Preámbulo
\usepackage{graphicx}
\usepackage{subcaption}
\graphicspath{{figures/}}

% Figura simple
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{nombre_figura.pdf}
\caption{Descripción de la figura}
\label{fig:etiqueta}
\end{figure}

% Figura con subfiguras (2x2)
\begin{figure}[h]
\centering
\begin{subfigure}[b]{0.48\textwidth}
    \includegraphics[width=\textwidth]{fig_a.pdf}
    \caption{Panel A}
    \label{fig:panel_a}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.48\textwidth}
    \includegraphics[width=\textwidth]{fig_b.pdf}
    \caption{Panel B}
    \label{fig:panel_b}
\end{subfigure}
\caption{Descripción general}
\label{fig:combined}
\end{figure}
```

## Resolución y Formato

- **Resolución:** Todas las figuras están en 300 DPI (calidad publicación)
- **Formatos disponibles:**
  - PNG: Para visualización rápida
  - PDF: Para inclusión en LaTeX (vector, mejor calidad)
  - SVG: Para edición posterior si necesario

## Orden Sugerido en el TFM

1. Fig 1: Arquitectura (Metodología)
2. Fig 2: Perfiles (Metodología)
3. Fig 4: Distribución de métricas (Resultados - Descriptivo)
4. Fig 5: Comparación top perfiles (Resultados - Descriptivo)
5. Fig 7: ANOVA (Resultados - Inferencial)
6. Fig 9: Hormonas individuales (Resultados - Análisis específico)
7. Fig 8: Correlaciones (Resultados - Análisis de correlación)
8. Fig 10: Comparación semántica (Resultados - Experimento adicional)
9. Fig 3: Ejemplo generación (Resultados - Cualitativo)
10. Fig 6: Figura maestra (Resumen final o Apéndice)

---

**Total de figuras:** 10 principales + variantes

