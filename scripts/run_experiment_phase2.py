"""
Experimento Fase 2: Evaluar perfiles hormonales combinados
"""

from endocrine_llm import (
    EndocrineModulatedLLM,
    HORMONE_PROFILES,
    ExperimentRunner
)
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("data/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Inicializar
print(" Inicializando modelo...")
model = EndocrineModulatedLLM("gpt2")

# Cargar prompts
prompts_df = pd.read_csv("data/prompts/prompts_dataset.csv")
prompts = prompts_df['prompt'].tolist()

# Perfiles Fase 2
profiles_phase2 = {
    "baseline": HORMONE_PROFILES["baseline"],
    "euphoric": HORMONE_PROFILES["euphoric"],
    "stressed": HORMONE_PROFILES["stressed"],
    "empathic": HORMONE_PROFILES["empathic"],
    "cautious": HORMONE_PROFILES["cautious"],
    "creative": HORMONE_PROFILES["creative"],
    "stable": HORMONE_PROFILES["stable"],
}

# Ejecutar
print("\n Ejecutando Fase 2...")
runner = ExperimentRunner(model, compute_advanced_metrics=True)

runner.run_experiment(
    prompts=prompts,
    profiles=profiles_phase2,
    num_generations=5,
    max_new_tokens=60,
    save_every=50
)

# Guardar
runner.save_results(
    json_path=str(OUTPUT_DIR / "phase2_results.json"),
    csv_path=str(OUTPUT_DIR / "phase2_results.csv")
)

# Estadísticas
summary = runner.get_summary_statistics()
print("\n ESTADÍSTICAS FASE 2:")
print(summary)

runner.export_examples(
    str(OUTPUT_DIR / "phase2_examples.txt"),
    num_examples=3
)

print("\n Fase 2 completada")
