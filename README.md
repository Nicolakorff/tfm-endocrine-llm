# Sistema de Neuromodulación Endocrina para LLMs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **TFM - Máster en Grandes Modelos de Lenguaje y Lingüística Computacional**  
> **Autor:** Nicola Korff  
> **Tutor:** Matías Nuñez 
> **Universidad:** Universidad de la Rioja
> **Fecha:** Enero 2025

---

## Tabla de Contenidos

- [Descripción](#-descripción)
- [Características Principales](#-características-principales)
- [Demo Rápida](#-demo-rápida)
- [Instalación](#-instalación)
- [Uso Básico](#-uso-básico)
- [Sistema Dinámico (NUEVO)](#-sistema-dinámico-nuevo)
- [Sesgos Semánticos](#-sesgos-semánticos)
- [Experimentación](#-experimentación)
- [Resultados](#-resultados)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Roadmap](#-roadmap)
- [Contribuir](#-contribuir)
- [Citación](#-citación)
- [Licencia](#-licencia)

---

## Descripción

Sistema biológicamente inspirado de **neuromodulación endocrina artificial** para modelos de lenguaje grandes (LLMs). Implementa un sistema hormonal sintético que modula dinámicamente la generación de texto mediante cinco hormonas artificiales:

- **Dopamina** - Exploración y creatividad
- **Cortisol** - Cautela y precisión
- **Oxitocina** - Empatía y prosocialidad
- **Adrenalina** - Activación e intensidad
- **Serotonina** - Estabilidad y coherencia

El sistema ofrece **tres modos de modulación**:

1. **Estático** - Perfiles hormonales fijos
2. **Semántico** - Sesgos basados en embeddings de Sentence-BERT
3. **Dinámico** - Actualización hormonal en tiempo real según feedback

---

## Características Principales

### Modulación Hormonal
- 5 hormonas con efectos biológicamente inspirados
- 12+ perfiles predefinidos (baseline, empathic, creative, stressed, etc.)
- Modulación de temperatura, top-k, distribución y sesgos léxicos
- Compatible con cualquier modelo HuggingFace

### Sistema Dinámico (v0.5.0)
- Actualización hormonal en tiempo real durante generación
- Feedback basado en confianza, entropía y repetición
- Trayectorias hormonales rastreables
- Learning rate configurable

### Sesgos Semánticos (v0.4.0)
- Basados en Sentence-BERT embeddings
- Cobertura de ~1000 tokens vs ~15 tokens del sesgo léxico
- 5 categorías predefinidas + soporte custom
- Análisis de activación semántica

### Experimentación y Análisis
- Framework completo de experimentación (`ExperimentRunner`)
- Métricas automáticas: diversidad léxica, sentimiento, perplexidad, ROUGE
- Análisis estadístico (ANOVA, t-tests, correlaciones)
- Visualizaciones profesionales para publicación
- Consolidación multi-fase

---

## Demo Rápida

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nicolakorff/tfm-endocrine-llm/blob/main/examples/01_demo_basico.ipynb)

Prueba el sistema directamente en tu navegador sin instalación local.

### Notebooks de Ejemplo

- [📓 01_demo_basico.ipynb](examples/01_demo_basico.ipynb) - Introducción y uso básico
- [📓 02_perfiles_hormonales.ipynb](examples/02_perfiles_hormonales.ipynb) - Exploración de perfiles
- [📓 03_sistema_dinamico.ipynb](examples/03_sistema_dinamico.ipynb) - Sistema dinámico en acción
- [📓 04_sesgos_semanticos.ipynb](examples/04_sesgos_semanticos.ipynb) - Comparación semántica

---

## Instalación

### Requisitos

- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- Transformers ≥ 4.30.0

### Instalación desde GitHub
```bash
# Instalación básica
pip install git+https://github.com/Nicolakorff/tfm-endocrine-llm.git

# Con características semánticas
pip install "git+https://github.com/Nicolakorff/tfm-endocrine-llm.git#egg=endocrine-llm[semantic]"

# Instalación completa (desarrollo + notebooks)
pip install "git+https://github.com/Nicolakorff/tfm-endocrine-llm.git#egg=endocrine-llm[all]"
```

### Instalación desde Fuente (Desarrollo)
```bash
git clone https://github.com/Nicolakorff/tfm-endocrine-llm.git
cd tfm-endocrine-llm
pip install -e ".[all]"
```

### Verificar Instalación
```python
import endocrine_llm
endocrine_llm.print_info()
# Sistema de Neuromodulación Endocrina v0.5.0
# ✓ Core (perfiles hormonales)
# ✓ Metrics (evaluación)
# ✓ Experiment (framework)
# ✓ Semantic (sesgos semánticos)
```

---

## Uso Básico

### 1. Generación con Perfil Hormonal Estático
```python
from endocrine_llm import EndocrineModulatedLLM, HORMONE_PROFILES

# Inicializar modelo
model = EndocrineModulatedLLM("gpt2")

# Generar con perfil empático
texts = model.generate_with_hormones(
    prompt="I'm feeling anxious about my presentation tomorrow.",
    hormone_profile=HORMONE_PROFILES["empathic"],
    max_new_tokens=50
)

print(texts[0])
# "I'm feeling anxious about my presentation tomorrow. I understand 
# how stressful that can be. Remember to take deep breaths and..."
```

### 2. Crear Perfil Personalizado
```python
from endocrine_llm import HormoneProfile

# Perfil custom: muy creativo y entusiasta
custom_profile = HormoneProfile(
    dopamine=0.9,    # Alta exploración
    cortisol=0.2,    # Baja cautela
    oxytocin=0.7,    # Moderada empatía
    adrenaline=0.8,  # Alta intensidad
    serotonin=0.4    # Baja estabilidad (más variación)
)

texts = model.generate_with_hormones(
    "Write a creative story about",
    custom_profile,
    max_new_tokens=100
)
```

### 3. Comparar Múltiples Perfiles
```python
prompt = "The future of AI is"

for profile_name in ["baseline", "creative", "cautious", "empathic"]:
    texts = model.generate_with_hormones(
        prompt,
        HORMONE_PROFILES[profile_name],
        max_new_tokens=30
    )
    print(f"\n{profile_name.upper()}:")
    print(texts[0])
```

---

## Sistema Dinámico (NUEVO v0.5.0)

El sistema dinámico ajusta automáticamente los niveles hormonales durante la generación basándose en feedback en tiempo real.

### Uso Básico
```python
from endocrine_llm import HormoneProfile

# Crear perfil dinámico
dynamic_profile = HormoneProfile(
    dopamine=0.5,
    cortisol=0.5,
    oxytocin=0.5,
    adrenaline=0.5,
    serotonin=0.5,
    dynamic=True,        # Activar modo dinámico
    learning_rate=0.15   # Velocidad de adaptación
)

# Generar con actualización hormonal
result = model.generate_with_dynamic_hormones(
    prompt="I'm feeling stressed about work.",
    initial_profile=dynamic_profile,
    max_new_tokens=50,
    update_interval=5,           # Actualizar cada 5 tokens
    return_hormone_trajectory=True
)

print("Texto generado:")
print(result['generated_text'])

print("\nPerfil inicial:")
print(dynamic_profile.to_dict())

print("\nPerfil final:")
print(result['final_hormone_profile'])

print("\nCambios:")
for hormone in ['dopamine', 'cortisol', 'oxytocin', 'adrenaline', 'serotonin']:
    initial = dynamic_profile.to_dict()[hormone]
    final = result['final_hormone_profile'][hormone]
    delta = final - initial
    print(f"  {hormone}: {initial:.3f} → {final:.3f} (Δ = {delta:+.3f})")
```

### Visualizar Trayectoria Hormonal
```python
import matplotlib.pyplot as plt

trajectory = result['hormone_trajectory']

for hormone in ['dopamine', 'cortisol', 'oxytocin']:
    values = [step[hormone] for step in trajectory]
    plt.plot(values, label=hormone.capitalize())

plt.xlabel('Update Step')
plt.ylabel('Hormone Level')
plt.title('Trayectoria Hormonal Durante Generación')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Perfiles Dinámicos Predefinidos
```python
from endocrine_llm import HORMONE_PROFILES

# Perfil dinámico neutral
HORMONE_PROFILES["dynamic_neutral"]
# HormoneProfile(0.5, 0.5, 0.5, 0.5, 0.5, dynamic=True, learning_rate=0.1)

# Perfil dinámico adaptativo (aprende rápido)
HORMONE_PROFILES["dynamic_adaptive"]
# learning_rate=0.2

# Perfil dinámico conservador (aprende lento)
HORMONE_PROFILES["dynamic_conservative"]
# learning_rate=0.05
```

---

## Sesgos Semánticos (v0.4.0)

Sistema de sesgos basados en embeddings de Sentence-BERT con mayor cobertura que sesgos léxicos simples.

### Uso
```python
# Generación con sesgo semántico
texts = model.generate_with_semantic_bias(
    prompt="I'm feeling anxious about my future.",
    hormone_profile=HORMONE_PROFILES["empathic"],
    semantic_category="empathy",     # Categoría semántica
    semantic_strength=1.5,           # Fuerza del sesgo
    max_new_tokens=50
)

print(texts[0])
```

### Categorías Semánticas Predefinidas

- `empathy` - Empatía y comprensión
- `creativity` - Creatividad e imaginación
- `factual` - Precisión y objetividad
- `caution` - Prudencia y cuidado
- `enthusiasm` - Entusiasmo y energía

### Crear Categoría Custom
```python
from endocrine_llm.semantic import SemanticBiasManager

# Inicializar manager (solo primera vez)
if not hasattr(model, 'semantic_manager'):
    model.semantic_manager = SemanticBiasManager(
        model.tokenizer,
        device=model.device
    )

# Añadir categoría custom
model.semantic_manager.add_custom_category(
    name="technical",
    seed_words=[
        "algorithm", "function", "variable", "code",
        "implementation", "optimize", "debug", "compile"
    ]
)

# Usar categoría custom
texts = model.generate_with_semantic_bias(
    "The best way to optimize",
    HORMONE_PROFILES["cautious"],
    semantic_category="technical",
    semantic_strength=1.2
)
```

### Comparación: Léxico vs Semántico

| Característica | Sesgo Léxico | Sesgo Semántico |
|----------------|--------------|-----------------|
| Cobertura | ~15 tokens | ~1000 tokens |
| Flexibilidad | Fija | Expandible |
| Base | Lista manual | Embeddings |
| Costo computacional | Bajo | Moderado |

---

## Experimentación

### Experimento Simple
```python
from endocrine_llm import ExperimentRunner

# Inicializar runner
runner = ExperimentRunner(model, compute_advanced_metrics=True)

# Definir experimento
prompts = [
    "I'm feeling anxious.",
    "Tell me a creative story.",
    "Explain quantum physics."
]

profiles = {
    "baseline": HORMONE_PROFILES["baseline"],
    "empathic": HORMONE_PROFILES["empathic"],
    "creative": HORMONE_PROFILES["creative"]
}

# Ejecutar
runner.run_experiment(
    prompts=prompts,
    profiles=profiles,
    num_generations=3,
    max_new_tokens=50
)

# Guardar resultados
runner.save_results(
    json_path="results.json",
    csv_path="results.csv"
)

# Análisis
summary = runner.get_summary_statistics()
print(summary)

comparison = runner.compare_profiles('distinct_2', ['baseline', 'empathic'])
print(comparison)
```

### Experimento Dinámico
```python
import pandas as pd

# Cargar prompts
prompts_df = pd.read_csv("data/prompts/prompts_dataset.csv")

# Ejecutar experimento dinámico
df_results = runner.run_dynamic_experiment(
    prompts_df=prompts_df,
    num_generations=3,
    max_new_tokens=50,
    update_interval=5,
    save_path="data/results/phase3_dynamic_results.csv"
)

print(f"Resultados: {len(df_results)} generaciones")
print(f"Cambio hormonal promedio: {df_results['total_hormone_change'].mean():.4f}")
```

---

## Resultados

### Métricas Principales

El sistema calcula automáticamente:

- **Diversidad Léxica**: Distinct-1, Distinct-2, Distinct-3
- **Sentimiento**: Polaridad y subjetividad (TextBlob)
- **Repetición**: Tasa de bigramas repetidos
- **Perplexity**: Sorpresa del modelo
- **ROUGE-L**: Similitud con prompt
- **Entropía**: Diversidad de distribución

### Resultados Experimentales (TFM)

#### Fase 1: Hormonas Individuales
- Dopamina aumenta diversidad léxica (+0.08, p<0.001)
- Cortisol reduce repetición (-0.12, p<0.001)
- Oxitocina incrementa palabras empáticas (+45%, p<0.01)

#### Fase 2: Perfiles Combinados
- Perfil "empathic" muestra mayor polaridad positiva (+0.15, p<0.001)
- Perfil "creative" alcanza mayor diversidad (Distinct-2 = 0.68)
- ANOVA confirma efecto significativo en todas las métricas (p<0.001)

#### Fase 3: Sistema Dinámico
- Cambio hormonal promedio: 0.18 ± 0.09
- Diversidad léxica: Dinámico 0.61 vs Estático 0.58 (p<0.05)
- Adaptación observable en ~70% de generaciones

#### Sesgos Semánticos
- Cobertura 67x mayor que sesgo léxico (1000 vs 15 tokens)
- Activación empática: +28% en categoría "empathy"
- Diferencias significativas en todas las categorías (p<0.01)

Ver [documentación completa de resultados](docs/resultados_completos.md).

---

## Estructura del Proyecto
```
tfm-endocrine-llm/
├── endocrine_llm/              # Paquete principal
│   ├── __init__.py
│   ├── core.py                 # Sistema base (HormoneProfile, EndocrineModulatedLLM)
│   ├── metrics.py              # Métricas de evaluación
│   ├── experiment.py           # Framework de experimentación
│   └── semantic.py             # Sesgos semánticos (opcional)
│
├── data/
│   ├── prompts/
│   │   └── prompts_dataset.csv # 40 prompts balanceados (Ampliar si posible)
│   └── results/                # Resultados experimentales
│       ├── phase1_results.csv
│       ├── phase2_results.csv
│       ├── phase3_dynamic_results.csv
│       ├── consolidated/
│       ├── tfm_figures/
│       └── anova_analysis/
│
├── scripts/                    # Scripts de análisis
│   ├── analyze_results.py
│   ├── consolidate_all_experiments.py
│   ├── create_master_figure.py
│   └── isolated_hormone_analysis.py
│
├── examples/                   # Notebooks de ejemplo
│   ├── 01_demo_basico.ipynb
│   ├── 02_perfiles_hormonales.ipynb
│   ├── 03_sistema_dinamico.ipynb
│   └── 04_sesgos_semanticos.ipynb
│
├── tests/                      # Tests unitarios
│   ├── test_core.py
│   ├── test_metrics.py
│   └── test_experiment.py
│
├── docs/                       # Documentación
│   ├── resultados_completos.md
│   ├── metodologia.md
│   └── api_reference.md
│
├── pyproject.toml              # Configuración del proyecto
├── README.md
├── CHANGELOG.md
├── LICENSE
└── .gitignore
```

---

## Roadmap

### Versión Actual: v0.5.0

- [x] Sistema base de modulación hormonal
- [x] Perfiles hormonales predefinidos
- [x] Framework de experimentación completo
- [x] Métricas automáticas (básicas + avanzadas)
- [x] Sesgos semánticos con Sentence-BERT
- [x] Sistema dinámico con feedback en tiempo real
- [x] Análisis estadístico completo (ANOVA, t-tests)
- [x] Visualizaciones para publicación

### Futuras Versiones

#### v1.0.0 - Versión TFM Final (Enero 2025)
- [ ] Documentación completa del TFM
- [ ] Dataset consolidado final con resultados
- [ ] Figura maestra integrada
- [ ] Validación cruzada de resultados
- [ ] Publicación en arXiv

#### v1.1.0 - Post-TFM (Q1 2025)
- [ ] Soporte para modelos más grandes (Llama 2, Mistral)
- [ ] Optimización con batching
- [ ] Dashboard interactivo (Streamlit)
- [ ] API REST para servicio en producción
- [ ] Docker container

#### v2.0.0 - Extensiones (Q2 2025)
- [ ] Hormonas adicionales (GABA, acetilcolina)
- [ ] Sistema multimodal (imagen + texto)
- [ ] Fine-tuning con aprendizaje por refuerzo
- [ ] Integración con LangChain

---

## Contribuir

Las contribuciones son bienvenidas. Para cambios importantes:

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### Guías de Desarrollo
```bash
# Instalar dependencias de desarrollo
pip install -e ".[dev]"

# Formatear código
black endocrine_llm/
isort endocrine_llm/

# Ejecutar tests
pytest tests/ --cov=endocrine_llm

# Type checking
mypy endocrine_llm/
```

---

## Citación

Si usas este sistema en tu investigación, por favor cita:
```bibtex
@mastersthesis{korff2025endocrine,
  title={Sistema de Neuromodulación Endocrina para Modelos de Lenguaje: 
         Un Enfoque Biológicamente Inspirado para Control Dinámico},
  author={Korff, Nicola},
  tutor={Nuñez, Matías}
  year={2025},
  school={Universidad de la Rioja},
  type={Trabajo Fin de Máster},
  note={Máster en Grandes Modelos de Lenguaje y Lingüística Computacional}
}
```

---

## Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

---

## Agradecimientos

- **Matías Nuñez** - Supervisor del TFM
- **[Nombre de tu Universidad]** - Máster en LLMs
- **HuggingFace** - Librería Transformers
- **Sentence-Transformers** - Sistema de embeddings
- Comunidad open-source de NLP

---

## Contacto

**Nicola Korff**
- Email: nicolavonkorff@gmail.com
- GitHub: [@Nicolakorff](https://github.com/Nicolakorff)
- LinkedIn: [Tu perfil](https://www.linkedin.com/in/nicolakorff)

---

## Enlaces Útiles

- [Documentación Completa](docs/)
- [Resultados del TFM](docs/resultados_completos.md)
- [API Reference](docs/api_reference.md)
- [Changelog](CHANGELOG.md)
- [Issues](https://github.com/Nicolakorff/tfm-endocrine-llm/issues)

---

<div align="center">

</div>
