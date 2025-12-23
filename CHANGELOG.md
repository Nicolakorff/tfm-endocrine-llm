## [0.4.0] - 2025-01-06

### Added
- Sistema de sesgos semánticos basados en Sentence-BERT embeddings
- SemanticBiasManager para gestionar categorías semánticas
- SemanticLogitsProcessor compatible con API existente
- Experimento comparativo sesgo simple vs semántico
- Análisis estadístico completo con visualizaciones
- Tablas LaTeX para TFM
- 5 categorías semánticas predefinidas (empathy, creativity, factual, caution, enthusiasm)
- Soporte para categorías custom

### Changed
- EndocrineModulatedLLM ahora incluye método `generate_with_semantic_bias()`
- Mejorada documentación con resultados experimentales

### Experimental
- Sistema de sesgos semánticos (puede requerir ajuste de parámetros)

## [0.3.0] - 2024-12-30

### Added
- Experimentos Fase 1 y Fase 2 completados
- Suite completa de tests (coverage >70%)
- Documentación de uso
- Scripts de análisis rápido

### Changed
- Mejorado manejo de errores en ExperimentRunner
- Optimizado cálculo de métricas

## [0.2.0] - 2024-12-23

### Added
- Sistema de experimentación (ExperimentRunner)
- Métricas avanzadas (perplexity, ROUGE-L)
- Tests completos

## [0.1.0] - 2024-12-19

### Added
- Sistema base de modulación hormonal
- LogitsProcessor integrado
- Perfiles hormonales predefinidos