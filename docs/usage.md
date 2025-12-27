# Guía de Uso - Sistema de Neuromodulación Endocrina para LLMs

**Versión:** 1.0  
**Última actualización:** Enero 2025  
**Nivel:** Principiante a Avanzado

---

## Tabla de Contenidos

- [Instalación](#-instalación)
- [Inicio Rápido](#-inicio-rápido)
- [Uso Básico](#-uso-básico)
- [Uso Avanzado](#-uso-avanzado)
- [Sistema Dinámico](#-sistema-dinámico-nuevo)
- [Sesgos Semánticos](#-sesgos-semánticos)
- [Experimentación](#-experimentación)
- [Análisis de Resultados](#-análisis-de-resultados)
- [Perfiles Disponibles](#-perfiles-disponibles)
- [API Reference](#-api-reference)
- [Troubleshooting](#-troubleshooting)
- [Ejemplos Completos](#-ejemplos-completos)

---

## Instalación

### Instalación desde GitHub
```bash
# Versión más reciente
pip install git+https://github.com/Nicolakorff/tfm-endocrine-llm.git

# Versión específica (recomendado para reproducibilidad)
pip install git+https://github.com/Nicolakorff/tfm-endocrine-llm.git@v0.5.0

# Con características semánticas
pip install "git+https://github.com/Nicolakorff/tfm-endocrine-llm.git#egg=endocrine-llm[semantic]"

# Instalación completa (desarrollo + notebooks)
pip install "git+https://github.com/Nicolakorff/tfm-endocrine-llm.git#egg=endocrine-llm[all]"
```

### Instalación desde Fuente (Desarrollo)
```bash
# Clonar repositorio
git clone https://github.com/Nicolakorff/tfm-endocrine-llm.git
cd tfm-endocrine-llm

# Instalar en modo editable
pip install -e .

# O con todas las dependencias
pip install -e ".[all]"
```

### Verificar Instalación
```python
import endocrine_llm
endocrine_llm.print_info()

# Salida esperada:
# ============================================================
# Sistema de Neuromodulación Endocrina v0.5.0
# ============================================================
# Componentes disponibles:
#   ✓ Core (perfiles hormonales)
#   ✓ Metrics (evaluación)
#   ✓ Experiment (framework)
#   ✓ Semantic (sesgos semánticos)
# ============================================================
```

---

## Inicio Rápido

### Tu Primer Texto Modulado (30 segundos)
```python
from endocrine_llm import EndocrineModulatedLLM, HORMONE_PROFILES

# 1. Inicializar modelo
model = EndocrineModulatedLLM("distilgpt2")  # Modelo pequeño, rápido

# 2. Generar con perfil empático
texts = model.generate_with_hormones(
    prompt="I'm feeling anxious about my presentation tomorrow.",
    hormone_profile=HORMONE_PROFILES["empathic"],
    max_new_tokens=50
)

# 3. Ver resultado
print(texts[0])
```

**Salida esperada:**
```
I'm feeling anxious about my presentation tomorrow. I understand how 
stressful that can be. Take a deep breath and remember that you've 
prepared well. Your audience wants you to succeed.
```

---

## Uso Básico

### 1. Generación con Diferentes Perfiles
```python
from endocrine_llm import EndocrineModulatedLLM, HORMONE_PROFILES

model = EndocrineModulatedLLM("gpt2")

prompt = "The future of artificial intelligence is"

# Comparar múltiples perfiles
for profile_name in ["baseline", "creative", "cautious", "empathic"]:
    texts = model.generate_with_hormones(
        prompt=prompt,
        hormone_profile=HORMONE_PROFILES[profile_name],
        max_new_tokens=30,
        num_return_sequences=1
    )
    
    print(f"\n{'='*60}")
    print(f"PERFIL: {profile_name.upper()}")
    print(f"{'='*60}")
    print(texts[0])
```

**Salida esperada:**
```
============================================================
PERFIL: BASELINE
============================================================
The future of artificial intelligence is uncertain, but it 
will likely involve more automation and data processing...

============================================================
PERFIL: CREATIVE
============================================================
The future of artificial intelligence is a shimmering tapestry 
woven with dreams of sentient machines and digital consciousness...

============================================================
PERFIL: CAUTIOUS
============================================================
The future of artificial intelligence is complex and requires 
careful consideration of ethical implications and potential risks...

============================================================
PERFIL: EMPATHIC
============================================================
The future of artificial intelligence is one where technology 
serves humanity's deepest needs, helping us connect and understand...
```

---

### 2. Crear Perfil Personalizado
```python
from endocrine_llm import HormoneProfile

# Perfil custom: Muy creativo pero cauteloso
my_profile = HormoneProfile(
    dopamine=0.9,    # Alta exploración/creatividad
    cortisol=0.7,    # Moderada-alta cautela
    oxytocin=0.5,    # Empatía neutra
    adrenaline=0.6,  # Moderada intensidad
    serotonin=0.6    # Moderada estabilidad
)

texts = model.generate_with_hormones(
    "Write a story about",
    my_profile,
    max_new_tokens=100
)

print(texts[0])
```

---

### 3. Generación Baseline (Sin Modulación)
```python
# Para comparación: generación sin modulación hormonal
baseline_text = model.generate_baseline(
    prompt="The future of AI is",
    max_new_tokens=50,
    temperature=1.0
)

print("BASELINE (sin modulación):")
print(baseline_text[0])
```

---

### 4. Controlar Parámetros de Generación
```python
texts = model.generate_with_hormones(
    prompt="Tell me about",
    hormone_profile=HORMONE_PROFILES["creative"],
    
    # Parámetros de control
    max_new_tokens=100,        # Longitud máxima
    num_return_sequences=3,    # Generar 3 variantes
    do_sample=True,            # Usar muestreo (vs greedy)
    top_k=50,                  # Top-K sampling
    top_p=0.95,                # Nucleus sampling
)

# Mostrar todas las variantes
for i, text in enumerate(texts, 1):
    print(f"\n--- Variante {i} ---")
    print(text)
```

---

## Uso Avanzado

### 1. Examinar Efecto de Cada Hormona
```python
from endocrine_llm import HormoneProfile

prompt = "The weather today is"

# Crear perfiles con solo una hormona alta
hormones = ['dopamine', 'cortisol', 'oxytocin', 'adrenaline', 'serotonin']

for hormone_name in hormones:
    # Baseline + hormona específica alta
    profile = HormoneProfile(
        dopamine=0.9 if hormone_name == 'dopamine' else 0.5,
        cortisol=0.9 if hormone_name == 'cortisol' else 0.5,
        oxytocin=0.9 if hormone_name == 'oxytocin' else 0.5,
        adrenaline=0.9 if hormone_name == 'adrenaline' else 0.5,
        serotonin=0.9 if hormone_name == 'serotonin' else 0.5,
    )
    
    texts = model.generate_with_hormones(prompt, profile, max_new_tokens=30)
    
    print(f"\n{hormone_name.upper()}: {texts[0]}")
```

---

### 2. Calcular Métricas de Texto
```python
from endocrine_llm import TextMetrics

generated_text = "I understand how you feel. Let me help you with that."

# Calcular todas las métricas
metrics = TextMetrics.compute_all(generated_text)

print("Métricas del texto:")
for metric_name, value in metrics.items():
    print(f"  {metric_name:25s}: {value:.4f}")

# Salida:
# length                   : 12.0000
# distinct_1               : 0.9167
# distinct_2               : 1.0000
# distinct_3               : 1.0000
# repetition_rate          : 0.0000
# sentiment_polarity       : 0.1750
# sentiment_subjectivity   : 0.4500
```

---

### 3. Métricas Avanzadas (requiere más tiempo)
```python
from endocrine_llm import AdvancedMetrics

# Inicializar calculador de métricas avanzadas
advanced = AdvancedMetrics(
    model.model,
    model.tokenizer,
    model.device
)

prompt = "Hello, I am"
generated = "Hello, I am feeling very happy today."

# Calcular métricas avanzadas
perplexity = advanced.compute_perplexity(generated)
rouge_l = advanced.compute_rouge_l(prompt, generated)
entropy = advanced.compute_entropy(generated)

print(f"Perplejidad: {perplexity:.2f}")
print(f"ROUGE-L:     {rouge_l:.4f}")
print(f"Entropía:    {entropy:.4f}")
```

---

## Sistema Dinámico (NUEVO)

### 1. Generación Dinámica Básica
```python
from endocrine_llm import HormoneProfile

# Crear perfil dinámico
dynamic_profile = HormoneProfile(
    dopamine=0.5,
    cortisol=0.5,
    oxytocin=0.5,
    adrenaline=0.5,
    serotonin=0.5,
    dynamic=True,        # 🔑 Activar modo dinámico
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

print("\n Perfil hormonal final:")
for hormone, value in result['final_hormone_profile'].items():
    initial = dynamic_profile.to_dict()[hormone]
    print(f"  {hormone:12s}: {initial:.3f} → {value:.3f} (Δ = {value-initial:+.3f})")
```

**Salida esperada:**
```
Texto generado:
I'm feeling stressed about work. Take a moment to breathe deeply 
and prioritize your tasks. Remember that it's okay to ask for help.

Perfil hormonal final:
  dopamine    : 0.500 → 0.523 (Δ = +0.023)
  cortisol    : 0.500 → 0.456 (Δ = -0.044)
  oxytocin    : 0.500 → 0.587 (Δ = +0.087)
  adrenaline  : 0.500 → 0.478 (Δ = -0.022)
  serotonin   : 0.500 → 0.512 (Δ = +0.012)
```

---

### 2. Visualizar Trayectoria Hormonal
```python
import matplotlib.pyplot as plt

trajectory = result['hormone_trajectory']

# Extraer evolución de cada hormona
hormones = ['dopamine', 'cortisol', 'oxytocin', 'adrenaline', 'serotonin']
colors = {
    'dopamine': '#e74c3c',
    'cortisol': '#f39c12',
    'oxytocin': '#3498db',
    'adrenaline': '#9b59b6',
    'serotonin': '#2ecc71'
}

plt.figure(figsize=(12, 6))

for hormone in hormones:
    values = [step[hormone] for step in trajectory]
    steps = list(range(len(values)))
    plt.plot(steps, values, marker='o', label=hormone.capitalize(), 
             color=colors[hormone], linewidth=2, markersize=4)

plt.xlabel('Update Step', fontsize=12)
plt.ylabel('Hormone Level', fontsize=12)
plt.title('Trayectoria Hormonal Durante Generación', fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig('hormone_trajectory.png', dpi=300)
plt.show()
```

---

### 3. Comparar Estático vs Dinámico
```python
prompt = "I'm feeling anxious."

# Versión estática
static_profile = HormoneProfile(0.5, 0.5, 0.5, 0.5, 0.5, dynamic=False)
static_texts = model.generate_with_hormones(
    prompt, static_profile, max_new_tokens=50, num_return_sequences=1
)

# Versión dinámica
dynamic_profile = HormoneProfile(0.5, 0.5, 0.5, 0.5, 0.5, dynamic=True, learning_rate=0.15)
dynamic_result = model.generate_with_dynamic_hormones(
    prompt, dynamic_profile, max_new_tokens=50
)

print("ESTÁTICO:")
print(static_texts[0])
print("\nDINÁMICO:")
print(dynamic_result['generated_text'])
print(f"\nCambio hormonal total: {sum(abs(dynamic_result['final_hormone_profile'][h] - 0.5) for h in hormones):.3f}")
```

---

## Sesgos Semánticos

### 1. Uso Básico de Sesgo Semántico
```python
texts = model.generate_with_semantic_bias(
    prompt="I'm feeling sad and alone.",
    hormone_profile=HORMONE_PROFILES["empathic"],
    semantic_category="empathy",     # Categoría objetivo
    semantic_strength=1.5,           # Fuerza del sesgo
    max_new_tokens=50
)

print(texts[0])
```

---

### 2. Categorías Semánticas Predefinidas
```python
# Categorías disponibles
categories = ['empathy', 'creativity', 'factual', 'caution', 'enthusiasm']

prompt = "Tell me about artificial intelligence."

for category in categories:
    texts = model.generate_with_semantic_bias(
        prompt,
        HORMONE_PROFILES["baseline"],
        semantic_category=category,
        semantic_strength=1.5,
        max_new_tokens=40
    )
    
    print(f"\n{category.upper()}:")
    print(texts[0][:200] + "...")
```

---

### 3. Crear Categoría Semántica Custom
```python
from endocrine_llm.semantic import SemanticBiasManager

# Inicializar manager (solo primera vez)
if not hasattr(model, 'semantic_manager'):
    model.semantic_manager = SemanticBiasManager(
        model.tokenizer,
        device=model.device
    )

# Definir nueva categoría con palabras semilla
model.semantic_manager.add_custom_category(
    name="technical",
    seed_words=[
        "algorithm", "function", "variable", "code", "implementation",
        "optimize", "debug", "compile", "syntax", "framework",
        "architecture", "module", "class", "method", "parameter"
    ]
)

# Usar categoría custom
texts = model.generate_with_semantic_bias(
    "The best way to optimize a neural network is",
    HORMONE_PROFILES["cautious"],
    semantic_category="technical",
    semantic_strength=1.2,
    max_new_tokens=60
)

print(texts[0])
```

---

### 4. Comparar Sesgo Simple vs Semántico
```python
prompt = "I need help with my problem."

# Sesgo simple (léxico)
simple_texts = model.generate_with_hormones(
    prompt,
    HORMONE_PROFILES["empathic"],  # Usa sesgo léxico interno
    max_new_tokens=50
)

# Sesgo semántico
semantic_texts = model.generate_with_semantic_bias(
    prompt,
    HORMONE_PROFILES["empathic"],
    semantic_category="empathy",
    semantic_strength=1.5,
    max_new_tokens=50
)

print("SESGO SIMPLE:")
print(simple_texts[0])
print("\nSESGO SEMÁNTICO:")
print(semantic_texts[0])
```

---

## Experimentación

### 1. Experimento Simple
```python
from endocrine_llm import ExperimentRunner

# Inicializar runner
runner = ExperimentRunner(model, compute_advanced_metrics=True)

# Definir prompts
prompts = [
    "I'm feeling anxious.",
    "Tell me a creative story.",
    "Explain quantum physics.",
    "How can I help someone in need?"
]

# Definir perfiles a comparar
profiles = {
    "baseline": HORMONE_PROFILES["baseline"],
    "empathic": HORMONE_PROFILES["empathic"],
    "creative": HORMONE_PROFILES["creative"],
    "cautious": HORMONE_PROFILES["cautious"]
}

# Ejecutar experimento
runner.run_experiment(
    prompts=prompts,
    profiles=profiles,
    num_generations=3,       # 3 repeticiones por combinación
    max_new_tokens=50,
    save_every=10            # Guardar checkpoint cada 10 generaciones
)

# Guardar resultados
runner.save_results(
    json_path="my_experiment.json",
    csv_path="my_experiment.csv"
)

print(f"Experimento completado: {len(runner.results)} generaciones")
```

---

### 2. Experimento Dinámico
```python
import pandas as pd

# Cargar dataset de prompts
prompts_df = pd.read_csv("data/prompts/prompts_dataset.csv")
# Esperado: columnas 'prompt' y 'category'

# Ejecutar experimento dinámico
df_results = runner.run_dynamic_experiment(
    prompts_df=prompts_df,
    num_generations=3,
    max_new_tokens=50,
    update_interval=5,
    save_path="dynamic_experiment_results.csv"
)

print(f"Resultados dinámicos: {len(df_results)} registros")
print(f"\nCambio hormonal promedio: {df_results['total_hormone_change'].mean():.4f}")
print(f"Diversidad léxica promedio (dinámico): {df_results[df_results['is_dynamic']==True]['distinct_2'].mean():.4f}")
```

---

## Análisis de Resultados

### 1. Estadísticas Resumidas
```python
# Obtener DataFrame con todos los resultados
df = runner.get_dataframe()

# Estadísticas por perfil
summary = runner.get_summary_statistics()
print(summary)

# Salida:
#                  distinct_2       repetition_rate  sentiment_polarity
#                  mean    std     mean    std      mean    std
# profile_name                                                         
# baseline         0.542   0.089   0.234   0.067    0.124   0.142
# empathic         0.589   0.094   0.198   0.059    0.213   0.138
# creative         0.623   0.102   0.187   0.071    0.089   0.156
# cautious         0.498   0.076   0.267   0.054    0.156   0.121
```

---

### 2. Comparar Perfiles Específicos
```python
# Comparar dos perfiles en una métrica
comparison = runner.compare_profiles(
    metric='distinct_2',
    profiles=['baseline', 'empathic', 'creative']
)

print(comparison)

# Salida:
#               count   mean    std     min     max
# profile_name                                      
# creative      12.0    0.623   0.102   0.398   0.798
# empathic      12.0    0.589   0.094   0.412   0.721
# baseline      12.0    0.542   0.089   0.312   0.689
```

---

### 3. Exportar Ejemplos de Texto
```python
# Exportar ejemplos representativos
runner.export_examples(
    output_path="text_examples.txt",
    profile_name="empathic",  # Solo perfil empático
    num_examples=10           # 10 ejemplos
)

print("Ejemplos exportados a text_examples.txt")
```

---

### 4. Visualizaciones
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Boxplot comparativo
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='profile_name', y='distinct_2')
plt.xticks(rotation=45)
plt.title('Diversidad Léxica por Perfil Hormonal')
plt.ylabel('Distinct-2')
plt.xlabel('Perfil')
plt.tight_layout()
plt.savefig('diversity_comparison.png', dpi=300)
plt.show()
```

---

## Perfiles Disponibles

### Perfiles Básicos

| Perfil | Descripción | Uso Recomendado |
|--------|-------------|-----------------|
| `baseline` | Todos en 0.5 | Control experimental |
| `high_dopamine` | Solo dopamina alta (0.9) | Generación creativa |
| `high_cortisol` | Solo cortisol alto (0.9) | Generación cautelosa |
| `high_oxytocin` | Solo oxitocina alta (0.9) | Contenido empático |
| `high_adrenaline` | Solo adrenalina alta (0.9) | Contenido intenso |
| `high_serotonin` | Solo serotonina alta (0.9) | Contenido estable |

### Perfiles Combinados

| Perfil | Dopamine | Cortisol | Oxytocin | Adrenaline | Serotonin | Descripción |
|--------|----------|----------|----------|------------|-----------|-------------|
| `empathic` | 0.6 | 0.4 | 0.9 | 0.4 | 0.7 | Empático y comprensivo |
| `creative` | 0.9 | 0.3 | 0.5 | 0.6 | 0.5 | Creativo y exploratorio |
| `cautious` | 0.3 | 0.8 | 0.5 | 0.4 | 0.6 | Cauteloso y preciso |
| `euphoric` | 0.9 | 0.2 | 0.6 | 0.5 | 0.8 | Positivo y entusiasta |
| `stressed` | 0.3 | 0.9 | 0.4 | 0.8 | 0.3 | Estresado y urgente |
| `stable` | 0.5 | 0.5 | 0.5 | 0.3 | 0.9 | Equilibrado y coherente |

### Perfiles Dinámicos (v0.5.0)

| Perfil | Learning Rate | Descripción |
|--------|---------------|-------------|
| `dynamic_neutral` | 0.1 | Adaptación moderada |
| `dynamic_adaptive` | 0.2 | Adaptación rápida |
| `dynamic_conservative` | 0.05 | Adaptación lenta |

### Visualizar Todos los Perfiles
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Extraer datos de perfiles
profile_data = []
for name, profile in HORMONE_PROFILES.items():
    if not profile.dynamic:  # Solo perfiles estáticos
        profile_data.append({
            'Profile': name,
            **profile.to_dict()
        })

df_profiles = pd.DataFrame(profile_data)

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    df_profiles.set_index('Profile')[['dopamine', 'cortisol', 'oxytocin', 'adrenaline', 'serotonin']],
    annot=True,
    fmt='.2f',
    cmap='viridis',
    vmin=0,
    vmax=1,
    cbar_kws={'label': 'Nivel Hormonal'}
)
plt.title('Mapa de Perfiles Hormonales', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('hormone_profiles_heatmap.png', dpi=300)
plt.show()
```

---

## API Reference

### Core Classes

#### `HormoneProfile`
```python
HormoneProfile(
    dopamine: float = 0.5,
    cortisol: float = 0.5,
    oxytocin: float = 0.5,
    adrenaline: float = 0.5,
    serotonin: float = 0.5,
    dynamic: bool = False,
    learning_rate: float = 0.1
)
```

**Métodos:**
- `to_dict() -> Dict[str, float]`: Convertir a diccionario
- `clone() -> HormoneProfile`: Crear copia independiente
- `update(feedback: Dict[str, float])`: Actualizar niveles (solo si `dynamic=True`)

---

#### `EndocrineModulatedLLM`
```python
EndocrineModulatedLLM(
    model_name: str = "gpt2",
    device: str = None
)
```

**Métodos principales:**
```python
generate_with_hormones(
    prompt: str,
    hormone_profile: HormoneProfile,
    max_new_tokens: int = 50,
    num_return_sequences: int = 1,
    do_sample: bool = True,
    top_k: int = 50,
    top_p: float = 0.95
) -> List[str]
```
```python
generate_with_dynamic_hormones(
    prompt: str,
    initial_profile: HormoneProfile,
    max_new_tokens: int = 50,
    update_interval: int = 5,
    return_hormone_trajectory: bool = False,
    **kwargs
) -> Dict
```
```python
generate_with_semantic_bias(
    prompt: str,
    hormone_profile: HormoneProfile,
    semantic_category: str,
    semantic_strength: float = 1.0,
    max_new_tokens: int = 50,
    num_return_sequences: int = 1
) -> List[str]
```

---

#### `ExperimentRunner`
```python
ExperimentRunner(
    model: EndocrineModulatedLLM,
    compute_advanced_metrics: bool = True
)
```

**Métodos:**
- `run_experiment(...)`: Ejecutar experimento estático
- `run_dynamic_experiment(...)`: Ejecutar experimento dinámico
- `get_dataframe() -> pd.DataFrame`: Obtener resultados
- `get_summary_statistics() -> pd.DataFrame`: Estadísticas por perfil
- `compare_profiles(...) -> pd.DataFrame`: Comparar perfiles
- `save_results(...)`: Guardar JSON/CSV
- `export_examples(...)`: Exportar ejemplos de texto

---

### Metrics Classes

#### `TextMetrics`
```python
TextMetrics.compute_all(text: str) -> Dict[str, float]
```

**Métricas retornadas:**
- `length`: Longitud en tokens
- `distinct_1`, `distinct_2`, `distinct_3`: Diversidad léxica
- `repetition_rate`: Tasa de repetición de bigramas
- `sentiment_polarity`: Polaridad del sentimiento [-1, 1]
- `sentiment_subjectivity`: Subjetividad [0, 1]

---

#### `AdvancedMetrics`
```python
AdvancedMetrics(model, tokenizer, device)

compute_perplexity(text: str) -> float
compute_rouge_l(reference: str, hypothesis: str) -> float
compute_entropy(text: str) -> float
```

---

## Troubleshooting

### Problema 1: Error al importar `sentence-transformers`

**Error:**
```
ModuleNotFoundError: No module named 'sentence_transformers'
```

**Solución:**
```bash
pip install sentence-transformers
# O
pip install "endocrine-llm[semantic]"
```

---

### Problema 2: CUDA out of memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solución:**
```python
# Usar modelo más pequeño
model = EndocrineModulatedLLM("distilgpt2")  # En lugar de "gpt2"

# O forzar CPU
model = EndocrineModulatedLLM("gpt2", device="cpu")

# O reducir batch size
texts = model.generate_with_hormones(
    prompt,
    profile,
    num_return_sequences=1  # En lugar de 5
)
```

---

### Problema 3: Generaciones muy repetitivas

**Solución:**
```python
# Aumentar temperatura efectiva con dopamina alta
profile = HormoneProfile(dopamine=0.9, ...)

# O ajustar parámetros de sampling
texts = model.generate_with_hormones(
    prompt,
    profile,
    top_p=0.95,  # Nucleus sampling más agresivo
    top_k=50     # Top-K más alto
)
```

---

### Problema 4: Resultados no reproducibles

**Solución:**
```python
import torch
import random
import numpy as np

# Fijar semillas
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

---

## Ejemplos Completos

### Ejemplo 1: Análisis Comparativo Completo

Ver: [`examples/01_comparative_analysis.py`](examples/01_comparative_analysis.py)

### Ejemplo 2: Sistema Dinámico en Acción

Ver: [`examples/02_dynamic_system.py`](examples/02_dynamic_system.py)

### Ejemplo 3: Experimento con Dataset

Ver: [`examples/03_full_experiment.py`](examples/03_full_experiment.py)

### Ejemplo 4: Análisis Estadístico

Ver: [`examples/04_statistical_analysis.py`](examples/04_statistical_analysis.py)

---

## Soporte

**Documentación completa:** [GitHub Wiki](https://github.com/Nicolakorff/tfm-endocrine-llm/wiki)  
**Issues:** [GitHub Issues](https://github.com/Nicolakorff/tfm-endocrine-llm/issues)  
**Email:** nicola.korff@example.com

---

## Licencia

MIT License - Ver [LICENSE](LICENSE) para detalles

---

**Última actualización:** Enero 2025  
**Versión del sistema:** 0.5.0
