# Guía de Uso

## Instalación
```bash
pip install git+https://github.com/Nicolakorff/tfm-endocrine-llm.git@v0.3.0
```

## Uso Básico

### 1. Generación Simple
```python
from endocrine_llm import EndocrineModulatedLLM, HORMONE_PROFILES

# Inicializar modelo
model = EndocrineModulatedLLM("gpt2")

# Generar texto
text = model.generate_with_hormones(
    prompt="I'm feeling anxious.",
    hormone_profile=HORMONE_PROFILES["empathic"],
    max_new_tokens=50
)

print(text[0])
```

### 2. Ejecutar Experimento
```python
from endocrine_llm import ExperimentRunner
import pandas as pd

# Cargar prompts
prompts = ["Hello", "How are you?"]

# Definir perfiles
profiles = {
    "baseline": HORMONE_PROFILES["baseline"],
    "empathic": HORMONE_PROFILES["empathic"]
}

# Ejecutar
runner = ExperimentRunner(model)
runner.run_experiment(prompts, profiles, num_generations=5)

# Analizar resultados
df = runner.get_dataframe()
summary = runner.get_summary_statistics()
```

### 3. Perfiles Disponibles

- `baseline`: Todos los niveles en 0.5
- `high_dopamine`: Alta exploración
- `high_cortisol`: Alta cautela
- `high_oxytocin`: Alta prosocialidad
- `empathic`: Configuración empática
- `creative`: Configuración creativa
- `cautious`: Configuración cautelosa

## API Reference

Ver docstrings en el código para detalles completos.