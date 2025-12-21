# Sistema de Neuromodulación Endocrina para LLMs

TFM - Máster en Grandes Modelos de Lenguaje y Lingüística Computacional
Tutor - Matías Nuñez

# README.md (actualizar sección demo)

## Demo Rápida

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nicolakorff/tfm-endocrine-llm/blob/main/examples/01_demo_basico.ipynb)

Prueba el sistema directamente en tu navegador sin instalación.

## Instalación
```bash
pip install git+https://github.com/Nicolakorff/tfm-endocrine-llm.git@v0.1.0
```

## Uso Básico
```python
from endocrine_llm import EndocrineModulatedLLM, HORMONE_PROFILES

model = EndocrineModulatedLLM("gpt2")
text = model.generate_with_hormones(
    "I'm feeling anxious.",
    HORMONE_PROFILES["empathic"]
)
print(text)
```

## Estado del Proyecto

- [x] v0.1.0 - Sistema base
- [ ] v0.2.0 - Sesgos semánticos
- [ ] v0.3.0 - Experimentos completos
