# Índice de Figuras para TFM

## Sistema de Neuromodulación Endocrina para LLMs

**Autor:** Nicola Korff  
**Tutor:** Matías Nuñez  
**Fecha:** Enero 2025

---

## Figuras Principales

### Figura 1: Arquitectura del Sistema
**Archivo:** `data/results/tfm_figures/individual/fig1_architecture.png` / `.pdf`  
**Ubicación en TFM:** Capítulo 4 (Metodología) - Sección 4.1 Diseño del Sistema  
**Descripción:** Diagrama completo mostrando la arquitectura del sistema de neuromodulación endocrina, incluyendo:
- Modelo base (LLM)
- Vector hormonal (5 componentes)
- Procesador de logits hormonal
- Flujo de datos y modulación

**Tamaño recomendado:** Página completa o `0.8\textwidth`
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=0.8\textwidth]{figures/fig1_architecture.pdf}
\caption{Arquitectura del sistema de neuromodulación endocrina para LLMs. 
El sistema integra cinco hormonas artificiales (dopamina, cortisol, oxitocina, 
adrenalina, serotonina) que modulan la distribución de probabilidad de tokens 
mediante transformaciones en el espacio de logits.}
\label{fig:architecture}
\end{figure}
```

**Referencias en texto:**
- "Como se muestra en la Figura \ref{fig:architecture}, el sistema..."
- "La arquitectura propuesta (ver Figura \ref{fig:architecture}) integra..."

---

### Figura 2: Perfiles Hormonales Predefinidos
**Archivo:** `data/results/tfm_figures/individual/fig2_hormone_profiles.png` / `.pdf`  
**Ubicación en TFM:** Capítulo 4 (Metodología) - Sección 4.3 Perfiles Hormonales  
**Descripción:** Mapa de calor (heatmap) mostrando los niveles hormonales para cada uno de los 12 perfiles predefinidos:
- baseline (control)
- 5 hormonas individuales (high_X)
- 6 perfiles combinados (empathic, creative, stressed, cautious, euphoric, stable)

**Colormap:** Viridis o RdYlGn (0.0 = azul, 1.0 = amarillo/verde)
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=0.7\textwidth]{figures/fig2_hormone_profiles.pdf}
\caption{Perfiles hormonales predefinidos utilizados en los experimentos. 
Cada fila representa un perfil y cada columna una hormona. El baseline 
(primera fila) tiene todos los valores en 0.5 como control.}
\label{fig:profiles}
\end{figure}
```

---

### Figura 3: Ejemplo Comparativo de Generación
**Archivo:** `data/results/tfm_figures/individual/fig3_generation_example.png` / `.pdf`  
**Ubicación en TFM:** Capítulo 5 (Resultados) - Sección 5.4 Análisis Cualitativo  
**Descripción:** Tabla visual mostrando el mismo prompt generado con 4-5 perfiles diferentes:
- Prompt: "I'm feeling anxious about my presentation tomorrow."
- Perfiles: baseline, empathic, creative, cautious
- Destacar diferencias léxicas y tonales

**Formato:** Tabla con fondo de color diferente por perfil
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=0.9\textwidth]{figures/fig3_generation_example.pdf}
\caption{Ejemplo comparativo de generación de texto con diferentes perfiles 
hormonales para el mismo prompt de entrada. Nótese las diferencias en tono, 
vocabulario y enfoque según el perfil aplicado.}
\label{fig:generation_example}
\end{figure}
```

---

### Figura 4: Distribución de Métricas Principales
**Archivo:** `data/results/tfm_figures/individual/fig4_metrics_distribution.png` / `.pdf`  
**Ubicación en TFM:** Capítulo 5 (Resultados) - Sección 5.1 Estadísticas Descriptivas  
**Descripción:** Grid de histogramas (2×3 o 3×2) mostrando la distribución de:
- Distinct-1, Distinct-2, Distinct-3
- Tasa de repetición
- Polaridad del sentimiento
- Longitud (tokens)

**Layout:** 6 subplots con misma escala de densidad
```latex
\begin{figure}[p]
\centering
\includegraphics[width=\textwidth]{figures/fig4_metrics_distribution.pdf}
\caption{Distribución de las métricas principales en todos los experimentos. 
(A) Diversidad léxica Distinct-1, (B) Distinct-2, (C) Distinct-3, 
(D) Tasa de repetición, (E) Polaridad del sentimiento, (F) Longitud en tokens.}
\label{fig:metrics_dist}
\end{figure}
```

---

### Figura 5: Comparación de Top Perfiles
**Archivo:** `data/results/tfm_figures/individual/fig5_top_profiles_comparison.png` / `.pdf`  
**Ubicación en TFM:** Capítulo 5 (Resultados) - Sección 5.2 Comparación de Perfiles  
**Descripción:** Violin plots o boxplots comparando los 6-10 perfiles más frecuentes en la métrica Distinct-2

**Elementos visuales:**
- Baseline destacado con línea vertical
- Ordenado por mediana
- Colores diferenciados
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=0.85\textwidth]{figures/fig5_top_profiles_comparison.pdf}
\caption{Comparación de diversidad léxica (Distinct-2) entre los perfiles 
hormonales más frecuentes. La línea roja vertical indica el baseline. 
Los perfiles creativos muestran mayor diversidad mientras que los cautelosos 
presentan menor variabilidad.}
\label{fig:profiles_comparison}
\end{figure}
```

---

### Figura 6: Figura Maestra (Resultados Integrados)
**Archivo:** `data/results/tfm_figures/master_figure.png` / `.pdf`  
**Ubicación en TFM:** Capítulo 5 (Resultados) - Inicio del capítulo o Apéndice A  
**Descripción:** Figura comprehensiva en grid 4×4 (sin dinámico) o 5×4 (con dinámico) que resume todos los resultados:

**Paneles:**
- **(A)** Diversidad léxica por perfil (violin plots)
- **(B)** Resultados ANOVA (F-statistics con significancia)
- **(C)** Tamaño de efecto (η² por métrica)
- **(D)** Efectos de hormonas individuales (barras horizontales)
- **(E)** Comparación sesgo simple vs semántico (boxplots)
- **(F)** Distribución general de métricas (histogramas superpuestos)
- **(G1)** Sistema dinámico: Estático vs Dinámico (solo si Fase 3)
- **(G2)** Distribución de cambios hormonales (solo si Fase 3)

**Tamaño:** Página completa o doble página
```latex
\begin{figure}[p]
\centering
\includegraphics[width=\textwidth]{figures/master_figure.pdf}
\caption{Resumen integrado de resultados experimentales. 
(A) Diversidad léxica por perfil hormonal, 
(B) Resultados ANOVA mostrando efectos significativos (***p<0.001, **p<0.01, *p<0.05), 
(C) Tamaño del efecto (η²) por métrica, 
(D) Efectos individuales de cada hormona vs baseline, 
(E) Comparación entre sesgo léxico simple y sesgo semántico, 
(F) Distribución general de métricas principales en todos los experimentos.
\ifDynamicPhase
(G1) Comparación sistema estático vs dinámico,
(G2) Distribución de cambios hormonales en sistema dinámico.
\fi}
\label{fig:master}
\end{figure}
```

---

## Figuras de Análisis Específico

### Figura 7: Resultados ANOVA Detallados
**Archivo:** `data/results/anova_analysis/anova_comparison.png` / `.pdf`  
**Ubicación en TFM:** Capítulo 5 (Resultados) - Sección 5.3 Análisis Inferencial  
**Descripción:** Barras de F-statistic para cada métrica con anotaciones de significancia y p-values
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=0.75\textwidth]{figures/anova_comparison.pdf}
\caption{Resultados del ANOVA para cada métrica evaluada. 
Todas las métricas muestran efectos significativos del perfil hormonal 
(p < 0.001), validando la efectividad del sistema de modulación.}
\label{fig:anova}
\end{figure}
```

---

### Figura 8: Matriz de Correlación Hormonas-Métricas
**Archivo:** `data/results/correlation_analysis/correlation_heatmap.png` / `.pdf`  
**Ubicación en TFM:** Capítulo 5 (Resultados) - Sección 5.5 Análisis de Correlación  
**Descripción:** Heatmap de correlaciones de Pearson entre niveles hormonales y métricas de texto

**Elementos:**
- Valores de correlación anotados
- Colormap divergente (RdBu_r)
- Asteriscos para significancia estadística
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=0.8\textwidth]{figures/correlation_heatmap.pdf}
\caption{Matriz de correlación entre niveles hormonales y métricas de texto. 
Se observa correlación positiva significativa entre dopamina y diversidad 
léxica (r=0.42, p<0.001), y negativa entre cortisol y tasa de repetición 
(r=-0.38, p<0.001). *p<0.05, **p<0.01, ***p<0.001}
\label{fig:correlation}
\end{figure}
```

---

### Figura 9: Efectos de Hormonas Individuales
**Archivo:** `data/results/isolated_hormones_analysis/hormone_effects_barplot.png` / `.pdf`  
**Ubicación en TFM:** Capítulo 5 (Resultados) - Sección 5.6 Análisis de Hormonas Aisladas  
**Descripción:** Barras horizontales mostrando la diferencia vs baseline para cada hormona individual en Distinct-2

**Código de colores:**
- Verde: Efecto positivo
- Rojo: Efecto negativo
- Anotaciones de significancia
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=0.7\textwidth]{figures/hormone_effects_barplot.pdf}
\caption{Efectos individuales de cada hormona sobre la diversidad léxica 
(Distinct-2) comparado con baseline. La dopamina muestra el mayor efecto 
positivo (+0.08, p<0.001), seguida de la adrenalina (+0.05, p<0.01). 
El cortisol muestra efecto negativo (-0.03, p<0.05).}
\label{fig:hormone_effects}
\end{figure}
```

---

### Figura 10: Comparación Sesgo Simple vs Semántico
**Archivo:** `data/results/semantic_comparison/analysis/figure_semantic_comparison.png` / `.pdf`  
**Ubicación en TFM:** Capítulo 5 (Resultados) - Sección 5.7 Experimento Semántico  
**Descripción:** Panel comparativo (2×2 o 1×3) mostrando:
- Boxplots de Distinct-2 (simple vs semántico)
- Diferencias en activación semántica
- Cobertura de tokens afectados
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=0.9\textwidth]{figures/semantic_comparison.pdf}
\caption{Comparación entre sesgo léxico simple y sesgo semántico basado en 
embeddings. El sesgo semántico muestra (A) mayor diversidad léxica 
(p<0.01), (B) activación significativamente superior en categoría objetivo 
(+28\%, p<0.001), y (C) cobertura ~67× mayor (1000 vs 15 tokens).}
\label{fig:semantic_comparison}
\end{figure}
```

---

### Figura 11: Sistema Dinámico - Trayectorias Hormonales
**Archivo:** `data/results/dynamic_experiment/trajectories/example_trajectory_max_change.png` / `.pdf`  
**Ubicación en TFM:** Capítulo 5 (Resultados) - Sección 5.8 Sistema Dinámico  
**Descripción:** Gráfico de líneas mostrando la evolución de niveles hormonales durante la generación token-por-token

**Elementos:**
- 5 líneas (una por hormona)
- Eje X: Update step
- Eje Y: Hormone level [0,1]
- Leyenda clara
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=0.8\textwidth]{figures/dynamic_trajectory_example.pdf}
\caption{Ejemplo de trayectoria hormonal durante generación dinámica. 
Se observa aumento progresivo de oxitocina (+0.12) y disminución de 
cortisol (-0.08) en respuesta a contenido empático generado. 
Learning rate: 0.15, Update interval: 5 tokens.}
\label{fig:dynamic_trajectory}
\end{figure}
```

---

### Figura 12: Sistema Dinámico - Comparación Estático vs Dinámico
**Archivo:** `data/results/dynamic_experiment/analysis/static_vs_dynamic_comparison.png` / `.pdf`  
**Ubicación en TFM:** Capítulo 5 (Resultados) - Sección 5.8 Sistema Dinámico  
**Descripción:** Panel comparativo mostrando diferencias entre sistema estático y dinámico

**Subpaneles:**
- Boxplots de Distinct-2
- Distribución de cambios hormonales
- Scatter plot cambio vs diversidad
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=\textwidth]{figures/dynamic_comparison.pdf}
\caption{Comparación entre sistema estático y dinámico. (A) El sistema 
dinámico muestra mayor diversidad léxica promedio (0.61 vs 0.58, p<0.05). 
(B) Distribución de cambios hormonales totales (media: 0.18±0.09). 
(C) Relación entre magnitud de cambio hormonal y diversidad resultante.}
\label{fig:dynamic_comparison}
\end{figure}
```

---

### Figura 13: Cambios Hormonales por Categoría de Prompt
**Archivo:** `data/results/dynamic_experiment/analysis/hormone_changes_by_category.png` / `.pdf`  
**Ubicación en TFM:** Capítulo 5 (Resultados) - Sección 5.8 Sistema Dinámico  
**Descripción:** Boxplots o violin plots mostrando cambios hormonales específicos según categoría de prompt (creative, empathetic, factual, reasoning)
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=0.85\textwidth]{figures/hormone_changes_by_category.pdf}
\caption{Cambios hormonales observados según categoría de prompt. 
Los prompts empáticos inducen mayor aumento de oxitocina (+0.10 promedio), 
mientras que los creativos aumentan dopamina (+0.08). Los factual muestran 
menor variabilidad hormonal general.}
\label{fig:changes_by_category}
\end{figure}
```

---

## Guía de Uso

### 1. Copiar Figuras al Proyecto LaTeX
```bash
#!/bin/bash
# Script: copy_figures_to_latex.sh

# Crear estructura de directorios
mkdir -p tfm_latex/figures

# Copiar figuras principales
echo "Copiando figuras principales..."
cp data/results/tfm_figures/individual/*.pdf tfm_latex/figures/
cp data/results/tfm_figures/master_figure.pdf tfm_latex/figures/

# Copiar figuras de análisis
echo "Copiando figuras de análisis..."
cp data/results/anova_analysis/*.pdf tfm_latex/figures/ 2>/dev/null
cp data/results/correlation_analysis/*.pdf tfm_latex/figures/ 2>/dev/null
cp data/results/isolated_hormones_analysis/*.pdf tfm_latex/figures/ 2>/dev/null
cp data/results/semantic_comparison/analysis/*.pdf tfm_latex/figures/ 2>/dev/null

# Copiar figuras dinámicas (Fase 3)
echo "Copiando figuras de sistema dinámico..."
cp data/results/dynamic_experiment/trajectories/*.pdf tfm_latex/figures/ 2>/dev/null
cp data/results/dynamic_experiment/analysis/*.pdf tfm_latex/figures/ 2>/dev/null

echo "✓ Figuras copiadas exitosamente a tfm_latex/figures/"
ls -lh tfm_latex/figures/ | wc -l
```

### 2. Template LaTeX para Preámbulo
```latex
% figures_setup.tex - Incluir en preámbulo del TFM

\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{float}  % Para control preciso de posición [H]
\usepackage{caption}

% Configurar ruta de figuras
\graphicspath{{figures/}}

% Personalizar captions
\captionsetup{
    font=small,
    labelfont=bf,
    format=hang,
    justification=justified,
    singlelinecheck=false,
    margin=0pt,
    figurewithin=section  % Numeración por sección
}

% Definir comandos útiles
\newcommand{\figref}[1]{Figura~\ref{#1}}
\newcommand{\tabref}[1]{Tabla~\ref{#1}}
```

### 3. Templates de Figuras

#### Figura Simple
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=0.8\textwidth]{nombre_figura.pdf}
\caption{Descripción clara y concisa de la figura. 
Incluir contexto necesario para interpretación independiente.}
\label{fig:etiqueta_unica}
\end{figure}
```

#### Figura con Subfiguras (2×2)
```latex
\begin{figure}[p]
\centering
\begin{subfigure}[b]{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{fig_a.pdf}
    \caption{Panel A - Descripción breve}
    \label{fig:panel_a}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{fig_b.pdf}
    \caption{Panel B - Descripción breve}
    \label{fig:panel_b}
\end{subfigure}

\vspace{0.5cm}

\begin{subfigure}[b]{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{fig_c.pdf}
    \caption{Panel C - Descripción breve}
    \label{fig:panel_c}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{fig_d.pdf}
    \caption{Panel D - Descripción breve}
    \label{fig:panel_d}
\end{subfigure}

\caption{Descripción general de la figura completa. 
Explicar relación entre paneles y conclusión principal.}
\label{fig:combined_results}
\end{figure}
```

#### Figura de Página Completa
```latex
\begin{figure}[p]  % [p] = dedicar página completa
\centering
\includegraphics[width=\textwidth,height=0.95\textheight,keepaspectratio]{master_figure.pdf}
\caption{Figura maestra con resultados integrados...}
\label{fig:master}
\end{figure}
```

---

## Orden Sugerido en el TFM

### Capítulo 4: Metodología
1. **Fig 1:** Arquitectura del sistema
2. **Fig 2:** Perfiles hormonales predefinidos

### Capítulo 5: Resultados

#### 5.1 Estadísticas Descriptivas
3. **Fig 4:** Distribución de métricas principales

#### 5.2 Comparación de Perfiles
4. **Fig 5:** Comparación de top perfiles

#### 5.3 Análisis Inferencial
5. **Fig 7:** Resultados ANOVA detallados
6. **Fig 6:** Figura maestra (puede ir aquí o al inicio del capítulo)

#### 5.4 Análisis Cualitativo
7. **Fig 3:** Ejemplo comparativo de generación

#### 5.5 Análisis de Correlación
8. **Fig 8:** Matriz de correlación hormonas-métricas

#### 5.6 Hormonas Aisladas
9. **Fig 9:** Efectos de hormonas individuales

#### 5.7 Experimento Semántico
10. **Fig 10:** Comparación sesgo simple vs semántico

#### 5.8 Sistema Dinámico (Fase 3)
11. **Fig 11:** Ejemplo de trayectoria hormonal
12. **Fig 12:** Comparación estático vs dinámico
13. **Fig 13:** Cambios hormonales por categoría

---

## Especificaciones Técnicas

### Resolución y Formatos

| Formato | Uso | Ventajas |
|---------|-----|----------|
| **PDF** | LaTeX (principal) | Vector, escalable, mejor calidad |
| **PNG** | Presentaciones, web | Raster, preview rápido |
| **SVG** | Edición posterior | Vector editable |

- **Resolución:** 300 DPI mínimo (600 DPI para figuras críticas)
- **Tamaño:** Optimizado para A4 (210×297 mm)
- **Colormap:** Viridis, RdBu_r (colorblind-friendly)
- **Fonts:** Arial o Helvetica, mínimo 8pt

### Checklist de Calidad

- [ ] Todos los ejes tienen etiquetas claras
- [ ] Leyendas legibles y bien posicionadas
- [ ] Colores distinguibles (test colorblindness)
- [ ] Resolución ≥300 DPI
- [ ] Formato vectorial (PDF) para LaTeX
- [ ] Caption descriptivo y auto-contenido
- [ ] Referenciado en el texto antes de aparecer
- [ ] Numeración consecutiva
- [ ] Tamaño apropiado (no pixelado al escalar)

---

## Referencias en Texto

### Ejemplos de Referencias
```latex
% Primera mención
Como se muestra en la \figref{fig:architecture}, el sistema integra...

% Múltiples figuras
Los resultados experimentales (\figref{fig:anova} y \figref{fig:correlation}) 
demuestran...

% Subfiguras
La distribución de diversidad léxica (\figref{fig:metrics_dist}a) muestra...

% Rango
Las figuras \ref{fig:first} a \ref{fig:last} resumen...

% Al final de párrafo
...con diferencias estadísticamente significativas (ver \figref{fig:anova}).
```

---

## Resumen Cuantitativo

**Total de figuras principales:** 13  
**Figuras de metodología:** 2  
**Figuras de resultados:** 11  
**Figuras con subfiguras:** 3-4  
**Figuras de página completa:** 2-3  

**Distribución por fase:**
- Sistema base: 7 figuras
- Sesgos semánticos: 1 figura
- Sistema dinámico: 3 figuras
- Figura maestra: 1 figura (integrada)

---

## Paleta de Colores Recomendada
```python
# Paleta consistente para todas las figuras
PALETTE = {
    'baseline': '#95a5a6',      # Gris
    'empathic': '#3498db',      # Azul
    'creative': '#e74c3c',      # Rojo
    'cautious': '#f39c12',      # Naranja
    'stressed': '#9b59b6',      # Morado
    'euphoric': '#2ecc71',      # Verde
    'static': '#3498db',        # Azul
    'dynamic': '#e74c3c',       # Rojo
}
```

---

**Última actualización:** Enero 2025  
**Versión:** 2.0 (con Sistema Dinámico)

---

## Notas Importantes

1. **Siempre usar formato PDF en LaTeX** para mejor calidad
2. **Caption auto-contenido**: El lector debe entender la figura sin leer el texto
3. **Referenciar antes de mostrar**: Mencionar figura antes de que aparezca
4. **Consistencia visual**: Usar mismos colores/estilos en todas las figuras
5. **Resolución verificada**: Comprobar que no se pixela al imprimir

---

## Soporte

Para regenerar cualquier figura:
```bash
# Figura maestra
python scripts/create_master_figure.py

# Análisis específicos
python scripts/isolated_hormone_analysis.py
python scripts/hormone_correlation_analysis.py

# Sistema dinámico
python scripts/visualize_hormone_trajectories.py
python scripts/analyze_dynamic_results.py
```

---

**Preparado para:** TFM - Máster en Grandes Modelos de Lenguaje  
**Formato:** LaTeX (pdflatex o XeLaTeX recomendado)

