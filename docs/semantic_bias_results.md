# Resultados: Sesgo Semántico vs Sesgo Simple

## Resumen Ejecutivo

Se compararon dos enfoques de sesgo prosocial en generación de texto:
1. **Sesgo simple**: Lista fija de ~15 tokens empáticos
2. **Sesgo semántico**: Similitud con embeddings SBERT (afecta ~1000 tokens)

## Resultados Principales

### Diversidad Léxica
- Sesgo simple: M=0.XXX (SD=0.XXX)
- Sesgo semántico: M=0.XXX (SD=0.XXX)
- **Diferencia: +X.XXX, t=X.XX, p<0.0X, d=0.XX**

### Interpretación
El sesgo semántico produjo [mayor/similar/menor] diversidad léxica...

[Completar con los resultados: por hacer]

## Figuras

Ver `data/results/semantic_comparison/analysis/figure_semantic_comparison.png`

## Conclusiones

1. El sesgo semántico es [más/igual de/menos] efectivo que el sesgo simple
2. La activación semántica promedio fue X.XX, indicando...
3. Las diferencias fueron más pronunciadas en prompts [empáticos/creativos]

## Limitaciones

- Muestra limitada (N=16 prompts)
- Solo evaluado en GPT-2 pequeño
- Una sola categoría semántica (empathy)