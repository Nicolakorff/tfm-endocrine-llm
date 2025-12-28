# 📊 Guía de Figuras para TFM

**Sistema de Neuromodulación Endocrina para LLMs**  
**Versión:** 2.0 (con Sistema Dinámico)  
**Fecha:** Enero 2025

---

## 📋 Índice de Figuras

### Figuras Principales (Metodología y Resultados)

1. **Arquitectura del Sistema** - Capítulo 4
2. **Perfiles Hormonales** - Capítulo 4
3. **Diversidad por Perfil** - Capítulo 5
4. **Resultados ANOVA** - Capítulo 5
5. **Tamaño de Efecto** - Capítulo 5
6. **Figura Maestra** - Capítulo 5 (Inicio) o Apéndice

### Figuras Fase 3 - Sistema Dinámico (NUEVO)

11. **Trayectorias Hormonales** - Capítulo 5.8
12. **Estático vs Dinámico** - Capítulo 5.8
13. **Cambios por Categoría** - Capítulo 5.8

---

## 🎨 Figura Maestra

**Archivo:** `data/results/tfm_figures/master_figure.pdf`  
**Ubicación:** Capítulo 5 - Inicio o Apéndice A  
**Tamaño:** Página completa o doble página

```latex
\begin{figure}[p]
\centering
\includegraphics[width=\textwidth]{figures/master_figure.pdf}
\caption{Resumen integrado de resultados experimentales. 
(A) Diversidad léxica por perfil hormonal, 
(B) Resultados ANOVA, 
(C) Tamaño del efecto (η²), 
(D) Efectos de hormonas individuales, 
(E) Comparación sesgo simple vs semántico, 
(F) Distribución general de métricas.
Si incluye Fase 3:
(G1) Comparación sistema estático vs dinámico,
(G2) Distribución de cambios hormonales.}
\label{fig:master}
\end{figure}
```

**Layout:**
- 4×4 (sin datos dinámicos)
- 5×4 (con datos dinámicos - Fase 3)

---

## 🔄 Figuras Sistema Dinámico

### Figura 11: Ejemplo de Trayectoria Hormonal

**Archivo:** `data/results/dynamic_experiment/trajectories/example_trajectory.pdf`

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

### Figura 12: Comparación Estático vs Dinámico

**Archivo:** `data/results/dynamic_experiment/analysis/static_vs_dynamic_comparison.pdf`

```latex
\begin{figure}[ht]
\centering
\includegraphics[width=\textwidth]{figures/dynamic_comparison.pdf}
\caption{Comparación entre sistema estático y dinámico. 
(A) El sistema dinámico muestra mayor diversidad léxica promedio 
(0.61 vs 0.58, p<0.05). 
(B) Distribución de cambios hormonales totales (media: 0.18±0.09). 
(C) Relación entre magnitud de cambio hormonal y diversidad resultante.}
\label{fig:dynamic_comparison}
\end{figure}
```

---

## 📁 Copiar Figuras a LaTeX

```bash
#!/bin/bash
# copy_figures_to_latex.sh

# Crear estructura
mkdir -p tfm_latex/figures

# Figuras principales
cp data/results/tfm_figures/master_figure.pdf tfm_latex/figures/

# Fase 3 - Dinámico
cp data/results/dynamic_experiment/trajectories/*.pdf tfm_latex/figures/
cp data/results/dynamic_experiment/analysis/*.pdf tfm_latex/figures/

echo "✓ Figuras copiadas a tfm_latex/figures/"
```

---

## 📐 Especificaciones Técnicas

- **Resolución:** 300 DPI
- **Formato principal:** PDF (vectorial)
- **Formatos alternativos:** PNG (raster), SVG (editable)
- **Tamaño:** Optimizado para A4 (210×297 mm)
- **Fonts:** Arial/Helvetica, mínimo 8pt
- **Colormap:** Viridis, RdBu_r (colorblind-friendly)

---

## ✅ Checklist de Calidad

- [ ] Ejes con etiquetas claras
- [ ] Leyendas legibles
- [ ] Resolución ≥300 DPI
- [ ] Formato vectorial (PDF)
- [ ] Caption auto-contenido
- [ ] Referenciado en texto
- [ ] Numeración consecutiva
- [ ] Tamaño apropiado

---

**Total figuras:** ~13-15 (según datos disponibles)  
**Preparado para:** TFM - Máster en Grandes Modelos de Lenguaje
