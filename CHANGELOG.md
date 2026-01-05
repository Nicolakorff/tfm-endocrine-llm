# Changelog

Todos los cambios notables en este proyecto serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Sistema de feedback visual en tiempo real
- Integración con modelos más grandes (Llama 2, Mistral)
- Optimización de rendimiento con batching
- Dashboard interactivo (Streamlit)
- API REST para producción

---

## [0.5.0] - 2025-01-XX

### Added - Sistema Dinámico (Fase 4)

**Core Module (`core.py`):**
- `HormoneProfile` con soporte para actualización dinámica (`dynamic=True`, `learning_rate`)
- Método `update()` en `HormoneProfile` para ajuste basado en feedback:
  - Dopamina: ↑ con alta confianza (>0.8), ↓ con baja (<0.3)
  - Cortisol: ↑ con alta entropía (>0.7), ↓ con baja (<0.3)
  - Oxitocina: ↑ con sentimiento positivo (>0.3), ↓ con negativo (<-0.3)
  - Serotonina: ↓ con alta repetición (>0.5), ↑ con baja (<0.2)
  - Adrenalina: ↑ cuando cortisol alto + dopamina baja (estrés sin recompensa)
- Método `clone()` en `HormoneProfile` para crear copias independientes
- `HormonalLogitsProcessor` con tracking de métricas en tiempo real:
  - `_track_generation_metrics()` - Captura max probability y entropía
  - `get_feedback()` - Retorna métricas promediadas para actualización hormonal
  - `add_token()` - Mantiene historial para detección de repetición
- `generate_with_dynamic_hormones()` en `EndocrineModulatedLLM`:
  - Generación token-por-token con actualización hormonal
  - Intervalos configurables (`update_interval`, default: 5)
  - Ajuste dinámico de `NoRepeatNGramLogitsProcessor`
  - Opción de retornar trayectoria hormonal completa
- Perfiles dinámicos predefinidos:
  - `dynamic_neutral` (learning_rate=0.1)
  - `dynamic_adaptive` (learning_rate=0.2)
  - `dynamic_conservative` (learning_rate=0.05)

**Experiment Module (`experiment.py`):**
- `run_dynamic_experiment()` en `ExperimentRunner`:
  - Compara 3 perfiles dinámicos vs 3 estáticos equivalentes
  - Tracking de niveles hormonales iniciales y finales
  - Cálculo de deltas por hormona y cambio total
  - Sampling de 10 prompts por categoría
  - Exportación a CSV con columnas extendidas

**Analysis & Visualization:**
- `consolidate_all_experiments.py` actualizado:
  - Soporte para Fase 4 (`phase4_dynamic_results.csv`)
  - Normalización de columnas entre fases
  - Manejo de columnas hormonales dinámicas (`init_*`, `final_*`, `delta_*`)
  - Estadísticas específicas de sistema dinámico
- `create_master_figure.py` mejorado:
  - Layout adaptativo (4×4 sin dinámico, 5×4 con dinámico)
  - Paneles G1 y G2 para análisis estático vs dinámico
  - Visualización de distribución de cambios hormonales

**Documentation:**
- Notebooks actualizados:
  - `02_sistema_dinamico.ipynb` - Demo completa del sistema dinámico
  - Visualización de trayectorias hormonales
  - Comparación estático vs dinámico
  - Análisis de learning rates
- Guías experimentales:
  - `docs/experiments/design_dynamic.md`
  - `docs/experiments/results_dynamic.md`

### Changed
- **Imports:** Añadidos `collections.deque` y `numpy` en `core.py`
- **`__init__.py`:** Simplificado con versión única (0.5.0)
- **README.md:** Reducido a ~200 líneas con links a documentación detallada
- **Estructura de documentación:** Reorganizada en `docs/` con especialización clara

### Fixed
- Variables indefinidas en `consolidate_all_experiments.py` (`results_dir` → `DATA_DIR`)
- Código duplicado en `experiment.py` (loose code movido a métodos)
- Conflictos de columnas entre diferentes fases experimentales
- Error al sobrescribir niveles hormonales en consolidación

### Performance
- Generación token-por-token optimizada con caching
- Tracking selectivo solo cuando `dynamic=True`

---

## [0.4.0] - 2025-01-06

### Added - Sistema de Sesgos Semánticos (Fase 3)

**Semantic Module (`semantic.py`):**
- `SemanticBiasManager` para gestión de categorías semánticas
- `SemanticLogitsProcessor` compatible con pipeline de transformers
- 5 categorías predefinidas: empathy, creativity, factual, caution, enthusiasm
- Método `add_custom_category()` para categorías definidas por usuario
- Función `analyze_semantic_activation()` para análisis post-generación

**Core Integration:**
- `generate_with_semantic_bias()` en `EndocrineModulatedLLM`
- Basado en Sentence-BERT (all-MiniLM-L6-v2)
- Cobertura de ~1000 tokens (vs ~15 del sesgo léxico)
- Parámetro `semantic_strength` para control de intensidad

**Experiment & Analysis:**
- Script comparativo: `scripts/run_semantic_experiment.py`
- Análisis estadístico completo con t-tests
- Visualizaciones: boxplots, heatmaps, violinplots
- Tablas LaTeX formateadas para TFM

**Results:**
- Incremento de +13.9% en diversidad léxica (p<0.001, d=0.86)
- Reducción de -15.4% en tasa de repetición (p<0.001, d=0.58)
- Activación semántica promedio: 0.412 en categoría objetivo
- Ratio de cobertura: 67× vs sesgo léxico simple

**Documentation:**
- `docs/experiments/design_semantic.md` - Diseño experimental
- `docs/experiments/results_semantic.md` - Resultados completos
- `03_semantic_bias_demo.ipynb` - Notebook interactivo

### Changed
- `pyproject.toml`: `sentence-transformers` movido a opcional `[semantic]`
- `requirements.txt`: Separado en básico, semántico y desarrollo

### Experimental
- Sistema de sesgos semánticos (puede requerir ajuste de parámetros según modelo)

---

## [0.3.0] - 2024-12-30

### Added - Fases Experimentales 1 y 2

**Experiments:**
- Fase 1: Hormonas individuales (baseline + 5 high hormones)
- Fase 2: Perfiles combinados (6 perfiles: euphoric, stressed, empathic, cautious, creative, stable)
- Dataset de 40 prompts balanceados (8 por categoría)
- 3 generaciones por combinación prompt-perfil

**Testing:**
- Suite completa de tests unitarios (`tests/`)
- Tests para `core.py`, `metrics.py`, `experiment.py`
- Tests de integración end-to-end
- Coverage >70%
- Fixtures reutilizables con pytest

**Analysis Scripts:**
- `analyze_results.py` - Análisis estadístico completo
- `isolated_hormone_analysis.py` - Efectos de hormonas individuales
- `hormone_correlation_analysis.py` - Correlaciones hormonas-métricas
- ANOVA con corrección Bonferroni

**Results:**
- Dopamina: +0.08 en diversidad léxica (p<0.001)
- Cortisol: -0.12 en repetición (p<0.001)
- Oxitocina: +45% en palabras empáticas (p<0.01)
- ANOVA: Efecto significativo en todas las métricas (F>15, p<0.001)
- Effect sizes: η² = 0.12-0.18 (mediano-grande)

**Documentation:**
- `01_demo_basico.ipynb` - Introducción completa
- README completo con ejemplos
- Guías de interpretación de resultados

### Changed
- `ExperimentRunner`: Checkpoints automáticos cada 50 generaciones
- Métricas optimizadas con modo batch
- Formato de guardado: JSON + CSV

### Fixed
- Error en cálculo de perplexity para textos cortos
- Problema con tokens especiales en métricas de diversidad
- Race condition en guardado de checkpoints

---

## [0.2.0] - 2024-12-23

### Added - Sistema de Experimentación

**ExperimentRunner (`experiment.py`):**
- Ejecución batch de múltiples configuraciones
- Guardado automático (JSON/CSV)
- Método `get_summary_statistics()` para análisis rápido
- Método `compare_profiles()` para comparación directa
- Método `export_examples()` para exportar textos

**Advanced Metrics (`metrics.py`):**
- `AdvancedMetrics` class:
  - Perplexity calculada con modelo base
  - ROUGE-L para similitud con prompt
  - Entropía de distribución de tokens
- `EmpathyMetrics` class:
  - Contador de palabras empáticas
  - Score normalizado
  - Lexicón empático (15 palabras)

**Testing:**
- `test_core.py` - Tests de perfiles y generación
- `test_metrics.py` - Tests de cálculo de métricas
- `test_experiment.py` - Tests de experimentación
- Fixtures compartidos

### Changed
- `TextMetrics.compute_all()` retorna diccionario flat
- Métricas opcionales calculadas bajo demanda
- Mejor manejo de memoria en cálculos batch

### Fixed
- División por cero en `distinct_n` para textos vacíos
- Error en tokenización para modelos sin `pad_token`
- Compatibilidad con pandas >=2.0

---

## [0.1.0] - 2024-12-19

### Added - Sistema Base

**Core Module (`core.py`):**
- `HormoneProfile` class:
  - 5 hormonas: dopamine, cortisol, oxytocin, adrenaline, serotonin
  - Validación de rangos [0,1]
  - Serialización con `to_dict()`
- `HormonalLogitsProcessor`:
  - Temperatura adaptativa (dopamina ↑, cortisol ↓)
  - Suavizado de distribución (serotonina)
  - Top-K dinámico (adrenalina)
  - Sesgo prosocial (oxitocina)
  - Compatible con `transformers.LogitsProcessor`
- `EndocrineModulatedLLM`:
  - Wrapper de modelos HuggingFace
  - `generate_with_hormones()` - Generación modulada
  - `generate_baseline()` - Generación sin modulación
  - Lexicón empático automático (15 palabras)
  - Auto-configuración de `pad_token`
- `HORMONE_PROFILES` predefinidos:
  - baseline, high_dopamine, high_cortisol, high_oxytocin, high_adrenaline, high_serotonin
  - euphoric, stressed, empathic, cautious, creative, stable

**Metrics Module (`metrics.py`):**
- `TextMetrics` class:
  - Distinct-1, Distinct-2, Distinct-3
  - Tasa de repetición (bigramas)
  - Análisis de sentimiento (TextBlob)
  - Longitud en tokens

**Infrastructure:**
- Estructura de paquete Python (`endocrine_llm/`)
- Setup con `pyproject.toml`
- Dependencias mínimas: transformers, torch, textblob, tqdm
- Compatible con Python ≥3.8

### Technical Details
- Tested con GPT-2, DistilGPT-2
- GPU opcional (auto-detect CUDA)
- Reproducibilidad con `torch.manual_seed()`

---

## Tipos de Cambios

- **Added** - Nuevas características
- **Changed** - Cambios en funcionalidad existente
- **Deprecated** - Características que se eliminarán pronto
- **Removed** - Características eliminadas
- **Fixed** - Correcciones de bugs
- **Security** - Correcciones de seguridad
- **Performance** - Mejoras de rendimiento
- **Experimental** - Características experimentales

---

## Links de Versiones

- [0.5.0]: https://github.com/Nicolakorff/tfm-endocrine-llm/releases/tag/v0.5.0
- [0.4.0]: https://github.com/Nicolakorff/tfm-endocrine-llm/releases/tag/v0.4.0
- [0.3.0]: https://github.com/Nicolakorff/tfm-endocrine-llm/releases/tag/v0.3.0
- [0.2.0]: https://github.com/Nicolakorff/tfm-endocrine-llm/releases/tag/v0.2.0
- [0.1.0]: https://github.com/Nicolakorff/tfm-endocrine-llm/releases/tag/v0.1.0
- [Unreleased]: https://github.com/Nicolakorff/tfm-endocrine-llm/compare/v0.5.0...HEAD

**FIN DEL DOCUMENTO**
