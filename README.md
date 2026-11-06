🇬🇧 English | 🇪🇸 [Español](README.es.md)

# Endocrine Neuromodulation System for LLMs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.6.0-green.svg)](https://github.com/Nicolakorff/tfm-endocrine-llm/releases)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Master's Thesis (TFM) – Master's in Large Language Models and Computational Linguistics**
> **University:** Universidad de La Rioja | **Date:** January 2026
> **Author:** Nicola Korff | **Advisor:** Matías Núñez

A biologically inspired system that modulates text generation in LLMs through an **artificial hormonal system**: five synthetic hormones that dynamically adjust the model's behavior.

---

## Demo Notebooks (Google Colab)

| Phase       | Description         | Notebook |
|-------------|---------------------|----------|
| **Phase 1** | Isolated hormones   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oozkxQkSFVE0itr-IlzBGLAIwFdZ1nZx?usp=sharing) |
| **Phase 2** | Combined profiles   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GpIASEVdCiJDkmeE2wY4536GpDrp1yil?usp=sharing) |
| **Phase 3** | Semantic bias       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gVMihyl-DJEP73cHHO6BnehwnPuBJAFP?usp=sharing) |
| **Phase 4** | Dynamic mode        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Xxl0M1H-ckstMC5Oi7h5pz3hNps-uthA?usp=sharing) |

> **Tip**: the notebooks run out of the box, with no local installation.

---

## Quick Start (30 seconds)

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

[Full guide](docs/usage_guide.md) | [Notebooks](examples)

---

## Key Features

### Three modulation modes

| Mode         | Description                 | Coverage     | Use case              |
|--------------|-----------------------------|--------------|-----------------------|
| **Static**   | Fixed hormonal profiles     | 12 profiles  | Controlled generation |
| **Dynamic**  | Real-time updates           | Adaptive     | Contextual adaptation |
| **Semantic** | Based on SBERT embeddings   | ~1,000 tokens| Topic-specific content|

### Five artificial hormones

- **Dopamine**: creativity and exploration
- **Cortisol**: caution and precision
- **Oxytocin**: empathy and prosociality
- **Adrenaline**: intensity and urgency
- **Serotonin**: stability and coherence

---

## Main Results (TFM – v0.6.0)

### Expanded dataset: Phase 5 (100 prompts)

| Component                          | v0.5.0 | v0.6.0 | Change             |
|------------------------------------|--------|--------|--------------------|
| **Prompts**                        | 40     | 100    | +150%              |
| **Prompts per category**           | 8      | 20     | +150%              |
| **Statistical power (1 − β)**      | ~0.85  | >0.90  | Improved           |
| **95% CI width**                   | ±0.012 | ±0.008 | −33% (more precise)|

### Dynamic vs. static system

| Metric                | Dynamic     | Static | Difference | p-value | Cohen's d |
|-----------------------|-------------|--------|------------|---------|-----------|
| **Lexical diversity** | 0.983       | 0.964  | **+1.97%** | <0.001  | 0.57      |
| **Repetition**        | 0.002       | 0.015  | **−86.7%** | <0.001  | 0.49      |
| **Perplexity**        | 28.95       | 17.24  | +67.9%     | <0.001  | 1.05      |
| **Hormonal change**   | 0.18 ± 0.09 | 0.00   | –          | –       | –         |

> **Interpreting the results:** dynamic mode reduces repetition and increases lexical diversity, but at the cost of higher perplexity (large effect, d = 1.05). This suggests the generated text drifts further from the base model's distribution; whether that trade-off is acceptable depends on the use case (creativity vs. fidelity). 

### Semantic vs. lexical bias

| Feature                     | Lexical   | Semantic      | Difference   | p-value | Cohen's d |
|-----------------------------|-----------|---------------|--------------|---------|-----------|
| **Coverage**                | 15 tokens | ~1,042 tokens | **67× more** | –       | –         |
| **Diversity (Distinct-2)**  | 0.547     | 0.623         | **+13.9%**   | <0.001  | 0.86      |
| **Repetition**              | 0.234     | 0.198         | **−15.4%**   | <0.001  | 0.58      |

### Limitations

- Evaluation relies on **automatic metrics** (lexical diversity, repetition, perplexity); human evaluation is not yet included.
- Experiments were run mainly on small models (e.g., `distilgpt2`).

[Full results](docs/experiments) | [Figures](docs/figures/figures_guide.md) | [Statistical analysis](data/results/statistical_report.md)

---

## Documentation

### For users

- [**Usage guide**](docs/usage_guide.md): complete documentation
- [**Notebooks**](examples): interactive examples on Colab

### For researchers

- [**Experiments**](docs/experiments): experimental designs
- [**TFM results**](docs/results): full analysis
- [**Figures guide**](docs/figures/figures_guide.md): publication-ready figures for LaTeX

### Development

- [**Changelog**](CHANGELOG.md): version history

---

## Usage Examples

### Generation with a hormonal profile

```python
# Empathic text
HORMONE_PROFILES["empathic"]  # High oxytocin, moderate cortisol

# Creative content
HORMONE_PROFILES["creative"]  # High dopamine, low caution

# Technical precision
HORMONE_PROFILES["cautious"]  # High cortisol, low dopamine
```

### Dynamic system (v0.6.0)

```python
from endocrine_llm import HormoneProfile

# Profile that updates in real time
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
print(result['final_hormone_profile'])  # Updated hormone levels
print(result['hormone_trajectory'])     # Changes at each step
```

### Semantic bias

```python
# Embedding-based bias (67× more coverage)
texts = model.generate_with_semantic_bias(
    "I need help.",
    HORMONE_PROFILES["empathic"],
    semantic_category="empathy",
    semantic_strength=1.5
)
```

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│           Base model (GPT-2, Llama, etc.)       │
└────────────────────┬────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │  Hormone vector (5D)  │
         │  [D, C, O, A, S]      │
         └───────────┬───────────┘
                     │
         ┌───────────▼────────────┐
         │  Logits processor      │
         │  • Temperature         │
         │  • Dynamic Top-K       │
         │  • Smoothing           │
         │  • Prosocial bias      │
         └───────────┬────────────┘
                     │
         ┌───────────▼────────────┐
         │  Feedback (dynamic)    │
         │  • Confidence          │
         │  • Entropy             │
         │  • Repetition          │
         │  • Sentiment           │
         └────────────────────────┘
```

---

## Installation

### Basic

```bash
pip install git+https://github.com/Nicolakorff/tfm-endocrine-llm.git
```

### With semantic features

```bash
pip install "git+https://github.com/Nicolakorff/tfm-endocrine-llm.git#egg=endocrine-llm[semantic]"
```

### Full (development)

```bash
git clone https://github.com/Nicolakorff/tfm-endocrine-llm.git
cd tfm-endocrine-llm
pip install -e ".[all]"
```

[Full installation guide](docs/installation.md)

---

## Reproducing the TFM Experiments

```bash
# 1. Phase 1: individual hormones
python scripts/run_phase1_isolated.py
python scripts/isolated_hormone_analysis.py

# 2. Phase 2: combined profiles
python scripts/run_phase2_combined.py

# 3. Phase 3: semantic bias
python scripts/run_experiment_semantic_comparison.py

# 4. Phase 4: dynamic system
python scripts/run_dynamic_experiment.py

# 5. Consolidate results
python scripts/consolidate_all_experiments.py

# 6. Generate the master figure
python scripts/create_master_figure.py
```

**Total time:** ~2–3 hours on GPU (T4/V100)

---

## Citation

If you use this system in your research, please cite:

```bibtex
@mastersthesis{korff2026endocrine,
  title   = {Sistema de Neuromodulación Endocrina para Modelos de Lenguaje:
             Un Enfoque Biológicamente Inspirado para Control Dinámico},
  author  = {Korff, Nicola},
  advisor = {Núñez, Matías},
  year    = {2026},
  school  = {Universidad de La Rioja},
  type    = {Master's thesis (Trabajo Fin de Máster)},
  note    = {Master's in Large Language Models and Computational Linguistics}
}
```

---

## Project Structure

```
tfm-endocrine-llm/
├── src/
│   └── endocrine_llm/      # Main package
│       ├── core.py             # Base + dynamic system
│       ├── metrics.py          # Evaluation metrics
│       ├── experiment.py       # Experimental framework
│       ├── semantic.py         # Semantic bias
│       └── visualization.py    # Visualizations
│
├── data/
│   ├── prompts/            # Prompt dataset
│   └── results/            # Experimental results
│
├── scripts/                # Experiment and analysis scripts
│   ├── run_phase1_isolated.py
│   ├── run_phase2_combined.py
│   ├── run_experiment_semantic_comparison.py
│   ├── run_dynamic_experiment.py
│   └── consolidate_all_experiments.py
│
├── examples/               # Example notebooks
├── tests/                  # Unit tests
└── docs/                   # Documentation
```

---

## Roadmap

### Completed (up to v0.6.0)

- Base hormonal modulation system
- 12 predefined profiles
- Dynamic system with feedback
- Semantic bias with SBERT
- Full experimentation framework
- Statistical analysis (ANOVA, t-tests)
- Publication-ready visualizations
- Dataset expanded to 100 prompts

### In progress (v1.0.0 – final TFM)

- Complete TFM documentation
- Final consolidated dataset
- Integrated master figure
- arXiv publication

### Future (v1.1.0+)

- Human evaluation with a rubric (coherence, profile adequacy, empathy)
- Support for larger models (Llama 2, Mistral)
- Interactive dashboard (Streamlit)
- REST API for production
- Fine-tuning with RL

---

## Contributing

Contributions are welcome. For major changes:

1. Fork the repository
2. Create a branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Matías Núñez**: thesis advisor
- **Hugging Face**: Transformers library
- **Sentence-Transformers**: embedding models
- The open-source NLP community

---

## Contact

**Nicola Korff**
nicolavonkorff@gmail.com
[GitHub](https://github.com/Nicolakorff) | [LinkedIn](https://www.linkedin.com/in/nicolakorff/)

---

If you find this project useful, please give it a star.

[Top](#endocrine-neuromodulation-system-for-llms) • [Docs](docs) • [Notebooks](examples) • [Results](docs/results/tfm_results.md)
