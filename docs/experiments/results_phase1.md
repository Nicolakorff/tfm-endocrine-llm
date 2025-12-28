# Resultados: Hormonas Aisladas (Fase 1)

**Versión:** 1.0  
**Fecha:** Diciembre 2024  
**Estado:** Completado

---

## Resumen Ejecutivo

Resultados del experimento de **Fase 1: Hormonas Aisladas**, evaluando el efecto individual de cada hormona artificial.

**Configuración:**
- Modelo: DistilGPT2 (82M)
- Prompts: 200
- Perfiles: 6
- Total: 3,600 generaciones

---

## Resultados Principales

### Estadísticas Descriptivas

| Perfil | Distinct-2 | Perplejidad | Repetición |
|--------|------------|-------------|------------|
| baseline | 0.542 | 24.8 | 0.234 |
| high_dopamine | **0.620** | **28.3** | 0.215 |
| high_cortisol | 0.498 | **21.7** | 0.256 |
| high_serotonin | 0.547 | 26.1 | **0.203** |
| high_adrenaline | 0.531 | **20.9** | 0.241 |

### ANOVA

| Métrica | F(5,3594) | p | η² | Efecto |
|---------|-----------|---|-----|--------|
| Perplejidad | **106.11** | <0.001 | **0.311** | GRANDE |
| Distinct-2 | **17.56** | <0.001 | **0.069** | MEDIANO |
| Repetición | **12.39** | <0.001 | **0.050** | PEQUEÑO |

Ver diseño en: [design_phase1.md](design_phase1.md)
