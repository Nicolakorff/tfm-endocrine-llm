# 🧬 Sistema de Neuromodulación Endocrina para LLMs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.5.0-green.svg)](https://github.com/Nicolakorff/tfm-endocrine-llm/releases)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **TFM - Máster en Grandes Modelos de Lenguaje y Lingüística Computacional**  
> **Autor:** Nicola Korff | **Tutor:** Matías Nuñez | **Fecha:** Enero 2025

Sistema biológicamente inspirado que modula la generación de texto en LLMs mediante un **sistema hormonal artificial** con 5 hormonas sintéticas que ajustan dinámicamente el comportamiento del modelo.

---

## 🚀 Quick Start (30 segundos)

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

**▶️** [Más ejemplos](docs/quickstart.md) | **📚** [Guía completa](docs/usage_guide.md) | **📓** [Notebooks](examples/)

---

## ✨ Características Principales

### 🎭 Tres Modos de Modulación

| Modo | Descripción | Cobertura | Uso |
|------|-------------|-----------|-----|
| **Estático** | Perfiles hormonales fijos | 12 perfiles | Generación controlada |
| **Dinámico** | Actualización en tiempo real | Adaptativo | Aprendizaje contextual |
| **Semántico** | Basado en embeddings SBERT | ~1000 tokens | Contenido específico |

### 🧬 Cinco Hormonas Artificiales

- **🎯 Dopamina** - Creatividad y exploración
- **⚠️ Cortisol** - Cautela y precisión
- **💙 Oxitocina** - Empatía y prosocialidad
- **⚡ Adrenalina** - Intensidad y urgencia
- **🌊 Serotonina** - Estabilidad y coherencia

---

## 📊 Resultados Principales (TFM)

### Sistema Dinámico vs Estático

| Métrica | Dinámico | Estático | Diferencia | p-value |
|---------|----------|----------|------------|---------|
| **Diversidad Léxica** | 0.61 | 0.58 | **+5.2%** | <0.05 |
| **Cambio Hormonal** | 0.18 ± 0.09 | 0.00 | - | - |

### Sesgos Semánticos vs Léxicos

| Característica | Léxico | Semántico | Ratio |
|----------------|--------|-----------|-------|
| **Cobertura** | 15 tokens | 1,042 tokens | **67×** |
| **Diversidad** | 0.547 | 0.623 | +13.9% |
| **p-value** | - | - | <0.001 |

**📈** [Resultados completos](docs/results/tfm_results.md) | **📊** [Figuras](docs/figures/figures_guide.md)

---

## 📖 Documentación

### Para Usuarios
- 📘 **[Instalación](docs/installation.md)** - Guía de instalación detallada
- 🚀 **[Quick Start](docs/quickstart.md)** - Primeros pasos en 5 minutos
- 📚 **[Guía de Uso](docs/usage_guide.md)** - Documentación completa
- 📓 **[Notebooks](examples/)** - Ejemplos interactivos en Colab

### Para Investigadores
- 🧪 **[Experimentos](docs/experiments/)** - Diseños experimentales
- 📊 **[Resultados del TFM](docs/results/tfm_results.md)** - Análisis completo
- 📈 **[Guía de Figuras](docs/figures/figures_guide.md)** - Figuras para LaTeX

### Desarrollo
- 📝 **[Changelog](CHANGELOG.md)** - Historial de cambios
- 🤝 **[Contribuir](CONTRIBUTING.md)** - Guía de contribución
- 🔧 **[API Reference](docs/api_reference.md)** - Documentación técnica

---

## 🎯 Ejemplos de Uso

### Generación con Perfil Hormonal

```python
# Texto empático
HORMONE_PROFILES["empathic"]  # Alta oxitocina, moderado cortisol

# Contenido creativo
HORMONE_PROFILES["creative"]  # Alta dopamina, baja cautela

# Precisión técnica
HORMONE_PROFILES["cautious"]  # Alto cortisol, baja dopamina
```

### Sistema Dinámico (🆕 v0.5.0)

```python
from endocrine_llm import HormoneProfile

# Perfil que aprende en tiempo real
dynamic_profile = HormoneProfile(
    0.5, 0.5, 0.5, 0.5, 0.5,
    dynamic=True,
    learning_rate=0.15
)

result = model.generate_with_dynamic_hormones(
    "I'm feeling stressed.",
    dynamic_profile,
    max_new_tokens=50
)

print(result['generated_text'])
print(result['final_hormone_profile'])  # Hormonas actualizadas
```

### Sesgos Semánticos

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

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────┐
│           Modelo Base (GPT-2, Llama, etc)       │
└────────────────────┬────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │  Vector Hormonal (5D) │
         │  [D, C, O, A, S]      │
         └───────────┬───────────┘
                     │
         ┌───────────▼────────────┐
         │  Procesador de Logits  │
         │  • Temperatura          │
         │  • Top-K dinámico       │
         │  • Suavizado            │
         │  • Sesgo prosocial      │
         └───────────┬────────────┘
                     │
         ┌───────────▼────────────┐
         │  Feedback (Dinámico)   │
         │  • Confianza            │
         │  • Entropía             │
         │  • Repetición           │
         │  • Sentimiento          │
         └────────────────────────┘
```

---

## 📦 Instalación

### Instalación Básica

```bash
pip install git+https://github.com/Nicolakorff/tfm-endocrine-llm.git
```

### Con Características Semánticas

```bash
pip install "git+https://github.com/Nicolakorff/tfm-endocrine-llm.git#egg=endocrine-llm[semantic]"
```

### Instalación Completa (Desarrollo)

```bash
git clone https://github.com/Nicolakorff/tfm-endocrine-llm.git
cd tfm-endocrine-llm
pip install -e ".[all]"
```

**📘** [Guía de instalación completa](docs/installation.md)

---

## 🧪 Reproducir Experimentos del TFM

```bash
# 1. Fase 1: Hormonas individuales
python scripts/run_phase1_isolated.py

# 2. Fase 2: Perfiles combinados
python scripts/run_phase2_combined.py

# 3. Fase 3: Sistema dinámico
python scripts/run_dynamic_experiment.py

# 4. Consolidar resultados
python scripts/consolidate_all_experiments.py

# 5. Generar figura maestra
python scripts/create_master_figure.py
```

**⏱️ Tiempo total:** ~2-3 horas en GPU (T4/V100)

---

## 📚 Citación

Si usas este sistema en tu investigación, por favor cita:

```bibtex
@mastersthesis{korff2025endocrine,
  title={Sistema de Neuromodulación Endocrina para Modelos de Lenguaje: 
         Un Enfoque Biológicamente Inspirado para Control Dinámico},
  author={Korff, Nicola},
  year={2025},
  school={Universidad [Nombre]},
  type={Trabajo Fin de Máster},
  note={Máster en Grandes Modelos de Lenguaje y Lingüística Computacional}
}
```

---

## 📊 Estructura del Proyecto

```
tfm-endocrine-llm/
├── endocrine_llm/          # 📦 Paquete principal
│   ├── core.py             # Sistema base + dinámico
│   ├── metrics.py          # Métricas de evaluación
│   ├── experiment.py       # Framework experimental
│   └── semantic.py         # Sesgos semánticos
│
├── data/
│   ├── prompts/            # Dataset de prompts
│   └── results/            # Resultados experimentales
│
├── scripts/                # 🔧 Scripts de análisis
├── examples/               # 📓 Notebooks de ejemplo
├── tests/                  # ✅ Tests unitarios
└── docs/                   # 📚 Documentación
```

---

## 🗺️ Roadmap

### ✅ Completado (v0.5.0)
- [x] Sistema base de modulación hormonal
- [x] 12 perfiles predefinidos
- [x] Sistema dinámico con feedback
- [x] Sesgos semánticos con SBERT
- [x] Framework de experimentación completo
- [x] Análisis estadístico (ANOVA, t-tests)
- [x] Visualizaciones para publicación

### 🚧 En Desarrollo (v1.0.0 - TFM Final)
- [ ] Documentación completa del TFM
- [ ] Dataset consolidado final
- [ ] Figura maestra integrada
- [ ] Publicación en arXiv

### 🔮 Futuro (v1.1.0+)
- [ ] Soporte para modelos grandes (Llama 2, Mistral)
- [ ] Dashboard interactivo (Streamlit)
- [ ] API REST para producción
- [ ] Fine-tuning con RL

---

## 🤝 Contribuir

Las contribuciones son bienvenidas. Para cambios importantes:

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

**📖** [Guía de contribución completa](CONTRIBUTING.md)

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

---

## 🙏 Agradecimientos

- **Matías Nuñez** - Supervisor del TFM
- **HuggingFace** - Librería Transformers
- **Sentence-Transformers** - Modelo de embeddings
- Comunidad open-source de NLP

---

## 📞 Contacto

**Nicola Korff**  
📧 nicola.korff@example.com  
🔗 [GitHub](https://github.com/Nicolakorff) | [LinkedIn](https://linkedin.com/in/tu-perfil)

**Issues:** [GitHub Issues](https://github.com/Nicolakorff/tfm-endocrine-llm/issues)  
**Documentación:** [Wiki](https://github.com/Nicolakorff/tfm-endocrine-llm/wiki)

---

<div align="center">

**⭐ Si este proyecto te resulta útil, dale una estrella ⭐**

Hecho con ❤️ para la comunidad NLP

[🏠 Inicio](#-sistema-de-neuromodulación-endocrina-para-llms) • [📖 Docs](docs/) • [📓 Notebooks](examples/) • [📊 Resultados](docs/results/tfm_results.md)

</div>
