# Sistema de Neuromodulación Endocrina para LLMs

TFM - Máster en Grandes Modelos de Lenguaje y Lingüística Computacional

## 🚀 Demo Rápida

[![Open In Colab](https://colab.research.google.com/drive/19o0i3AqptAxGdIdYMx0Q_mFjYGOwdcRY?usp=sharing)

Prueba el sistema directamente en tu navegador sin instalación.

## 📦 Instalación
```bash
pip install git+https://github.com/Nicolakorff/tfm-endocrine-llm.git@v0.1.0
```

## 💡 Uso Básico
```python
from endocrine_llm import EndocrineModulatedLLM, HORMONE_PROFILES

model = EndocrineModulatedLLM("gpt2")
text = model.generate_with_hormones(
    "I'm feeling anxious.",
    HORMONE_PROFILES["empathic"]
)
print(text)
```

## 📊 Estado del Proyecto

- [x] v0.1.0 - Sistema base
- [ ] v0.2.0 - Sesgos semánticos
- [ ] v0.3.0 - Experimentos completos
