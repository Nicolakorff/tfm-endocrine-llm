🇪🇸 Español | 🇬🇧 [English](README.md)

# Sistema de Neuromodulación Endocrina para LLMs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.6.0-green.svg)](https://github.com/Nicolakorff/tfm-endocrine-llm/releases)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **TFM - Máster en Grandes Modelos de Lenguaje y Lingüística Computacional**
> **Universidad:** Universidad de La Rioja | **Fecha:** Enero 2026
> **Autora:** Nicola Korff | **Tutor:** Matías Núñez

Sistema biológicamente inspirado que modula la generación de texto en LLMs mediante un **sistema hormonal artificial** con 5 hormonas sintéticas que ajustan dinámicamente el comportamiento del modelo.

---

## Notebooks Demo (Google Colab)

| Fase       | Descripción         | Notebook |
|------------|---------------------|----------|
| **Fase 1** | Hormonas aisladas   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oozkxQkSFVE0itr-IlzBGLAIwFdZ1nZx?usp=sharing) |
| **Fase 2** | Perfiles combinados | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GpIASEVdCiJDkmeE2wY4536GpDrp1yil?usp=sharing) |
| **Fase 3** | Sesgo semántico     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gVMihyl-DJEP73cHHO6BnehwnPuBJAFP?usp=sharing) |
| **Fase 4** | Modo dinámico       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Xxl0M1H-ckstMC5Oi7h5pz3hNps-uthA?usp=sharing) |

> **Tip**: los notebooks están listos para ejecutar sin instalación local.

---

## Quick Start (30 segundos)

```bash
pip install git+https://github.com/Nicolakorff/tfm-endocrine-llm.git
```

```python
from endocrine_llm import EndocrineModulatedLLM, HORMONE_PROFILES

model = EndocrineModulatedLLM("distilgpt2")
texts = model.generate_with_hormones(
    "I'm feeling anxious.",
    HORMONE_PROFILES["empathic"],
    max_new_tokens=50
)
print(texts[0])
# → "I'm feeling anxious. I understand how stressful that can be..."
```

[Guía completa](docs/usage_guide.md) | [Notebooks](examples)

---

## Características principales

### Tres modos de modulación

| Modo          | Descripción                  | Cobertura    | Uso                    |
|---------------|------------------------------|--------------|------------------------|
| **Estático**  | Perfiles hormonales fijos    | 12 perfiles  | Generación controlada  |
| **Dinámico**  | Actualización en tiempo real | Adaptativo   | Aprendizaje contextual |
| **Semántico** | Basado en embeddings SBERT   | ~1000 tokens | Contenido específico   |

### Cinco hormonas artificiales

- **Dopamina**: creatividad y exploración
- **Cortisol**: cautela y precisión
- **Oxitocina**: empatía y prosocialidad
- **Adrenalina**: intensidad y urgencia
- **Serotonina**: estabilidad y coherencia

---

## Resultados principales (TFM - v0.6.0)

### Dataset expandido: Fase 5 (100 prompts)

| Componente                              | v0.5.0 | v0.6.0 | Cambio           |
|-----------------------------------------|--------|--------|------------------|
| **Prompts**                             | 40     | 100    | +150%            |
| **Prompts por categoría**               | 8      | 20     | +150%            |
| **Potencia estadística (1 − β)**        | ~0.85  | >0.90  | Mejorada         |
| **Anchura del IC 95%**                  | ±0.012 | ±0.008 | -33% (más preciso) |

### Sistema dinámico vs. estático

| Métrica               | Dinámico    | Estático | Diferencia | p-value | Cohen's d |
|-----------------------|-------------|----------|------------|---------|-----------|
| **Diversidad léxica** | 0.983       | 0.964    | **+1.97%** | <0.001  | 0.57      |
| **Repetición**        | 0.002       | 0.015    | **-86.7%** | <0.001  | 0.49      |
| **Perplejidad**       | 28.95       | 17.24    | +67.9%     | <0.001  | 1.05      |
| **Cambio hormonal**   | 0.18 ± 0.09 | 0.00     | -          | -       | -         |

> **Lectura de resultados:** el modo dinámico reduce la repetición y aumenta la diversidad léxica, pero a costa de una mayor perplejidad (efecto grande, d = 1.05). Esto indica que el texto se aleja de la distribución del modelo base; si ese coste es aceptable depende del caso de uso (creatividad frente a fidelidad). *[Revisa esta interpretación y ajústala a las conclusiones de tu memoria.]*

### Sesgos semánticos vs. léxicos

| Característica              | Léxico    | Semántico     | Diferencia  | p-value | Cohen's d |
|-----------------------------|-----------|---------------|-------------|---------|-----------|
| **Cobertura**               | 15 tokens | ~1,042 tokens | **67× más** | -       | -         |
| **Diversidad (Distinct-2)** | 0.547     | 0.623         | **+13.9%**  | <0.001  | 0.86      |
| **Repetición**              | 0.234     | 0.198         | **-15.4%**  | <0.001  | 0.58      |

### Limitaciones

- La evaluación se basa en **métricas automáticas** (diversidad léxica, repetición, perplejidad); no incluye todavía evaluación humana.
- Los experimentos se han realizado principalmente con modelos pequeños (p. ej. `distilgpt2`).

[Resultados completos](docs/experiments) | [Figuras](docs/figures/figures_guide.md) | [Análisis estadístico](data/results/statistical_report.md)

---

## Documentación

### Para usuarios

- [**Guía de uso**](docs/usage_guide.md): documentación completa
- [**Notebooks**](examples): ejemplos interactivos en Colab

### Para investigadores

- [**Experimentos**](docs/experiments): diseños experimentales
- [**Resultados del TFM**](docs/results): análisis completo
- [**Guía de figuras**](docs/figures/figures_guide.md): figuras para LaTeX

### Desarrollo

- [**Changelog**](CHANGELOG.md): historial de cambios

---

## Ejemplos de uso

### Generación con perfil hormonal

```python
# Texto empático
HORMONE_PROFILES["empathic"]  # Alta oxitocina, cortisol moderado

# Contenido creativo
HORMONE_PROFILES["creative"]  # Alta dopamina, baja cautela

# Precisión técnica
HORMONE_PROFILES["cautious"]  # Alto cortisol, baja dopamina
```

### Sistema dinámico (v0.6.0)

```python
from endocrine_llm import HormoneProfile

# Perfil que se actualiza en tiempo real
dynamic_profile = HormoneProfile(
    1.0, 1.0, 1.0, 1.0, 1.0,
    dynamic=True,
    learning_rate=0.15
)

result = model.generate_with_dynamic_hormones(
    "I'm feeling stressed.",
    dynamic_profile,
    max_new_tokens=100,
    update_interval=10
)

print(result['generated_text'])
print(result['final_hormone_profile'])  # Hormonas actualizadas
print(result['hormone_trajectory'])     # Cambios en cada token
```

### Sesgos semánticos

```python
# Sesgo basado en embeddings (67× más cobertura)
texts = model.generate_with_semantic_bias(
    "I need help.",
    HORMONE_PROFILES["empathic"],
    semantic_category="empathy",
    semantic_strength=1.5
)
```

---

## Arquitectura

```
┌─────────────────────────────────────────────────┐
│           Modelo base (GPT-2, Llama, etc.)      │
└────────────────────┬────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │  Vector hormonal (5D) │
         │  [D, C, O, A, S]      │
         └───────────┬───────────┘
                     │
         ┌───────────▼────────────┐
         │  Procesador de logits  │
         │  • Temperatura         │
         │  • Top-K dinámico      │
         │  • Suavizado           │
         │  • Sesgo prosocial     │
         └───────────┬────────────┘
                     │
         ┌───────────▼────────────┐
         │  Feedback (dinámico)   │
         │  • Confianza           │
         │  • Entropía            │
         │  • Repetición          │
         │  • Sentimiento         │
         └────────────────────────┘
```

---

## Instalación

### Básica

```bash
pip install git+https://github.com/Nicolakorff/tfm-endocrine-llm.git
```

### Con características semánticas

```bash
pip install "git+https://github.com/Nicolakorff/tfm-endocrine-llm.git#egg=endocrine-llm[semantic]"
```

### Completa (desarrollo)

```bash
git clone https://github.com/Nicolakorff/tfm-endocrine-llm.git
cd tfm-endocrine-llm
pip install -e ".[all]"
```

[Guía de instalación completa](docs/installation.md)

---

## Reproducir los experimentos del TFM

```bash
# 1. Fase 1: hormonas individuales
python scripts/run_phase1_isolated.py
python scripts/isolated_hormone_analysis.py

# 2. Fase 2: perfiles combinados
python scripts/run_phase2_combined.py

# 3. Fase 3: sesgo semántico
python scripts/run_experiment_semantic_comparison.py

# 4. Fase 4: sistema dinámico
python scripts/run_dynamic_experiment.py

# 5. Consolidar resultados
python scripts/consolidate_all_experiments.py

# 6. Generar figura maestra
python scripts/create_master_figure.py
```

**Tiempo total:** ~2-3 horas en GPU (T4/V100)

---

## Citación

Si usas este sistema en tu investigación, por favor cita:

```bibtex
@mastersthesis{korff2026endocrine,
  title  = {Sistema de Neuromodulación Endocrina para Modelos de Lenguaje:
            Un Enfoque Biológicamente Inspirado para Control Dinámico},
  author = {Korff, Nicola},
  advisor = {Núñez, Matías},
  year   = {2026},
  school = {Universidad de La Rioja},
  type   = {Trabajo Fin de Máster},
  note   = {Máster en Grandes Modelos de Lenguaje y Lingüística Computacional}
}
```

---

## Estructura del proyecto

```
tfm-endocrine-llm/
├── src/
│   └── endocrine_llm/      # Paquete principal
│       ├── core.py             # Sistema base + dinámico
│       ├── metrics.py          # Métricas de evaluación
│       ├── experiment.py       # Framework experimental
│       ├── semantic.py         # Sesgos semánticos
│       └── visualization.py    # Visualizaciones
│
├── data/
│   ├── prompts/            # Dataset de prompts
│   └── results/            # Resultados experimentales
│
├── scripts/                # Scripts de experimentos y análisis
│   ├── run_phase1_isolated.py
│   ├── run_phase2_combined.py
│   ├── run_experiment_semantic_comparison.py
│   ├── run_dynamic_experiment.py
│   └── consolidate_all_experiments.py
│
├── examples/               # Notebooks de ejemplo
├── tests/                  # Tests unitarios
└── docs/                   # Documentación
```

---

## Roadmap

### Completado (hasta v0.6.0)

- Sistema base de modulación hormonal
- 12 perfiles predefinidos
- Sistema dinámico con feedback
- Sesgos semánticos con SBERT
- Framework de experimentación completo
- Análisis estadístico (ANOVA, t-tests)
- Visualizaciones para publicación
- Dataset ampliado a 100 prompts

### En desarrollo (v1.0.0 - TFM final)

- Documentación completa del TFM
- Dataset consolidado final
- Figura maestra integrada
- Publicación en arXiv

### Futuro (v1.1.0+)

- Evaluación humana con rúbrica (coherencia, adecuación al perfil, empatía)
- Soporte para modelos grandes (Llama 2, Mistral)
- Dashboard interactivo (Streamlit)
- API REST para producción
- Fine-tuning con RL

---

## Contribuir

Las contribuciones son bienvenidas. Para cambios importantes:

1. Haz un fork del repositorio
2. Crea una rama (`git checkout -b feature/AmazingFeature`)
3. Haz commit de tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Haz push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

---

## Agradecimientos

- **Matías Núñez**: tutor del TFM
- **Hugging Face**: librería Transformers
- **Sentence-Transformers**: modelo de embeddings
- Comunidad open-source de NLP

---

## Contacto

**Nicola Korff**
nicolavonkorff@gmail.com
[GitHub](https://github.com/Nicolakorff) | [LinkedIn](https://www.linkedin.com/in/nicola-korff/)

---

Si este proyecto te resulta útil, dale una estrella.

[Inicio](#sistema-de-neuromodulación-endocrina-para-llms) • [Docs](docs) • [Notebooks](examples) • [Resultados](docs/results/tfm_results.md)
