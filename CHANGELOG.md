# Changelog

Todos los cambios notables en este proyecto serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
y este proyecto se adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Sistema de feedback visual en tiempo real
- Integración con modelos más grandes (Llama, Mistral)
- Optimización de rendimiento con batching
- Dashboard interactivo para exploración de resultados

---

## [0.5.0] - 2025-01-XX (Por hacer)

### Added
- **Sistema Hormonal Dinámico (Fase 3)**:
  - `HormoneProfile` con soporte para actualización dinámica (`dynamic=True`)
  - Método `update()` en `HormoneProfile` para ajuste basado en feedback
  - `generate_with_dynamic_hormones()` en `EndocrineModulatedLLM`
  - Tracking de métricas en tiempo real (`HormonalLogitsProcessor`)
  - Feedback automático basado en confianza, entropía y repetición
  - Método `run_dynamic_experiment()` en `ExperimentRunner`
  - Perfiles dinámicos predefinidos: `dynamic_neutral`, `dynamic_adaptive`, `dynamic_conservative`

- **Mejoras en análisis**:
  - Script consolidado `consolidate_all_experiments.py` con soporte para Fase 3
  - Normalización automática de columnas entre fases
  - Estadísticas específicas de sistema dinámico
  - Panel G en figura maestra para comparación estático vs dinámico
  - Metadata extendida con información de componentes

- **Documentación**:
  - Ejemplos de uso del sistema dinámico
  - Guía de configuración de `learning_rate`
  - Interpretación de trayectorias hormonales

### Changed
- `core.py`: Añadidos imports `collections.deque` y `numpy`
- `HormonalLogitsProcessor`: Ahora incluye tracking con `_track_generation_metrics()`
- `HORMONE_PROFILES`: Expandido con 3 perfiles dinámicos
- `consolidate_all_experiments.py`: Manejo robusto de columnas heterogéneas
- `create_master_figure.py`: Layout adaptativo (4x4 o 5x4 según datos)
- `__init__.py`: Simplificado y versionado único (`0.5.0`)

### Fixed
- Variables indefinidas en scripts de consolidación y visualización
- Código duplicado en `experiment.py`
- Conflictos de columnas entre diferentes fases experimentales
- Error al sobrescribir niveles hormonales en consolidación

### Performance
- Generación token-por-token optimizada para sistema dinámico
- Tracking selectivo solo cuando `dynamic=True`

---

## [0.4.0] - 2025-01-06

### Added
- **Sistema de sesgos semánticos** basados en Sentence-BERT embeddings:
  - `SemanticBiasManager` para gestionar categorías semánticas
  - `SemanticLogitsProcessor` compatible con API existente
  - Experimento comparativo sesgo simple vs semántico
  - 5 categorías semánticas predefinidas: `empathy`, `creativity`, `factual`, `caution`, `enthusiasm`
  - Soporte para categorías custom definidas por usuario

- **Análisis y visualización**:
  - Script `semantic_comparison_analysis.py` con estadísticas completas
  - Visualizaciones comparativas (boxplots, violins, heatmaps)
  - Tablas LaTeX formateadas para TFM
  - Análisis de activación semántica por token
  - Mapas de calor de similitud semántica

- **Métricas semánticas**:
  - `semantic_activation_empathy` - Activación de categoría empática
  - Análisis de distribución de activaciones
  - Comparación estadística (t-tests) entre condiciones

### Changed
- `EndocrineModulatedLLM` ahora incluye método `generate_with_semantic_bias()`
- Mejorada documentación con resultados experimentales completos
- Dataset de prompts expandido con categorización semántica

### Experimental
- Sistema de sesgos semánticos (puede requerir ajuste de parámetros según modelo)
- Integración experimental con modelos de embeddings alternativos

### Fixed
- Lazy loading de `sentence-transformers` para evitar dependencias obligatorias
- Manejo de errores en cálculo de similitud semántica
- Normalización de scores para diferentes tamaños de vocabulario

---

## [0.3.0] - 2024-12-30

### Added
- **Experimentos completos Fase 1 y Fase 2**:
  - Fase 1: Hormonas individuales (baseline + 5 hormonas high)
  - Fase 2: Perfiles combinados (6 perfiles: euphoric, stressed, empathic, cautious, creative, stable)
  - Dataset de 200 prompts balanceados (50 por categoría)
  - 3 generaciones por combinación prompt-perfil

- **Suite completa de tests**:
  - Tests unitarios para `core.py`, `metrics.py`, `experiment.py`
  - Tests de integración end-to-end
  - Coverage >70%
  - Fixtures reutilizables con `pytest`

- **Scripts de análisis**:
  - `analyze_results.py` - Análisis estadístico completo
  - `isolated_hormone_analysis.py` - Efectos individuales
  - `hormone_correlation_analysis.py` - Correlaciones
  - ANOVA con corrección Bonferroni

- **Documentación**:
  - Guía de uso completa en README
  - Ejemplos de código funcionales
  - Interpretación de resultados

### Changed
- `ExperimentRunner`: Mejorado manejo de errores con try-except por generación
- Checkpoints automáticos cada 50 generaciones
- Optimizado cálculo de métricas (modo batch para métricas avanzadas)
- Formato de guardado: JSON + CSV para compatibilidad

### Fixed
- Error en cálculo de perplexity para textos muy cortos
- Problema con tokens especiales en métricas de diversidad
- Race condition en guardado de checkpoints

---

## [0.2.0] - 2024-12-23

### Added
- **Sistema de experimentación (`ExperimentRunner`)**:
  - Ejecución batch de múltiples configuraciones
  - Guardado automático de resultados (JSON/CSV)
  - Estadísticas resumidas por perfil
  - Exportación de ejemplos a archivo de texto
  - Método `compare_profiles()` para comparación directa

- **Métricas avanzadas (`AdvancedMetrics`)**:
  - Perplejidad calculada con el modelo base
  - ROUGE-L para similitud con prompt
  - Entropía de distribución de tokens
  - Coherencia contextual

- **Métricas empáticas (`EmpathyMetrics`)**:
  - Contador de palabras empáticas en texto
  - Score normalizado de empatía
  - Lexicón empático básico (15 palabras)

- **Tests completos**:
  - `test_core.py` - Tests de perfiles y generación
  - `test_metrics.py` - Tests de cálculo de métricas
  - `test_experiment.py` - Tests de experimentación
  - Fixtures compartidos

### Changed
- `TextMetrics.compute_all()` ahora retorna diccionario flat
- Métricas opcionales calculadas solo si se solicitan
- Mejor manejo de memoria en cálculos batch

### Fixed
- División por cero en cálculo de `distinct_n` para textos vacíos
- Error en tokenización para modelos sin pad_token
- Compatibilidad con pandas >2.0

---

## [0.1.0] - 2024-12-19

### Added
- **Sistema base de modulación hormonal**:
  - Clase `HormoneProfile` con 5 hormonas: dopamina, cortisol, oxitocina, adrenalina, serotonina
  - Validación automática de rangos [0,1]
  - Serialización a diccionario con `to_dict()`

- **Procesamiento de logits (`HormonalLogitsProcessor`)**:
  - Temperatura adaptativa (dopamina ↑, cortisol ↓)
  - Suavizado de distribución (serotonina)
  - Top-K dinámico (adrenalina)
  - Sesgo prosocial (oxitocina)
  - Integrado con `transformers.LogitsProcessor`

- **Modelo base (`EndocrineModulatedLLM`)**:
  - Wrapper de modelos HuggingFace
  - Método `generate_with_hormones()` con modulación completa
  - Método `generate_baseline()` sin modulación
  - Lexicón empático automático (15 palabras)
  - Configuración automática de pad_token

- **Perfiles hormonales predefinidos (`HORMONE_PROFILES`)**:
  - `baseline` - Todos en 0.5 (control)
  - Hormonas individuales: `high_dopamine`, `high_cortisol`, `high_oxytocin`, `high_adrenaline`, `high_serotonin`
  - Perfiles combinados: `euphoric`, `stressed`, `empathic`, `cautious`, `creative`, `stable`

- **Métricas básicas (`TextMetrics`)**:
  - Diversidad léxica: Distinct-1, Distinct-2, Distinct-3
  - Tasa de repetición (bigramas)
  - Análisis de sentimiento (polaridad y subjetividad)
  - Longitud en tokens

- **Infraestructura**:
  - Estructura de paquete Python (`endocrine_llm/`)
  - Setup con `pyproject.toml`
  - Dependencias mínimas: transformers, torch, textblob, tqdm

### Technical Details
- Compatible con Python ≥3.8
- Tested con GPT-2, DistilGPT-2
- GPU opcional (auto-detect CUDA)
- Reproducibilidad con `torch.manual_seed()`

---

## Tipos de cambios

- `Added` - Nuevas características
- `Changed` - Cambios en funcionalidad existente
- `Deprecated` - Características que se eliminarán pronto
- `Removed` - Características eliminadas
- `Fixed` - Correcciones de bugs
- `Security` - Correcciones de seguridad
- `Performance` - Mejoras de rendimiento
- `Experimental` - Características experimentales

---

## Links

- [0.4.0]: Comparar con tag anterior cuando esté disponible
- [0.3.0]: Comparar con tag anterior cuando esté disponible
- [0.2.0]: Comparar con tag anterior cuando esté disponible
- [0.1.0]: Versión inicial
