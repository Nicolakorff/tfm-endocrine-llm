"""
Experimento Fase 1: Evaluar efecto de hormonas individuales
"""

from endocrine_llm import (
    EndocrineModulatedLLM,
    HORMONE_PROFILES,
    ExperimentRunner
)
import pandas as pd
from pathlib import Path

# Configuración
OUTPUT_DIR = Path("data/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Inicializar
print("🧠 Inicializando modelo...")
model = EndocrineModulatedLLM("gpt2")  # o "gpt2-medium" si tienes GPU potente

# Cargar prompts
print("📝 Cargando prompts...")
prompts_df = pd.read_csv("data/prompts/prompts_dataset.csv")
prompts = prompts_df['prompt'].tolist()

print(f"   Total prompts: {len(prompts)}")

# Seleccionar perfiles para Fase 1
profiles_phase1 = {
    "baseline": HORMONE_PROFILES["baseline"],
    "high_dopamine": HORMONE_PROFILES["high_dopamine"],
    "high_cortisol": HORMONE_PROFILES["high_cortisol"],
    "high_oxytocin": HORMONE_PROFILES["high_oxytocin"],
    "high_adrenaline": HORMONE_PROFILES["high_adrenaline"],
    "high_serotonin": HORMONE_PROFILES["high_serotonin"],
}

# Ejecutar experimento
print("\n🧪 Ejecutando Fase 1...")
runner = ExperimentRunner(model, compute_advanced_metrics=True)

runner.run_experiment(
    prompts=prompts,
    profiles=profiles_phase1,
    num_generations=5,  # 5 generaciones por combinación
    max_new_tokens=60,
    save_every=50  # Checkpoint cada 50 generaciones
)

# Guardar resultados
print("\n💾 Guardando resultados...")
runner.save_results(
    json_path=str(OUTPUT_DIR / "phase1_results.json"),
    csv_path=str(OUTPUT_DIR / "phase1_results.csv")
)

# Mostrar estadísticas
print("\n📊 ESTADÍSTICAS RESUMIDAS:")
summary = runner.get_summary_statistics()
print(summary)

# Exportar ejemplos
runner.export_examples(
    str(OUTPUT_DIR / "phase1_examples.txt"),
    num_examples=3
)

print(f"\n✅ Fase 1 completada")
print(f"   Total generaciones: {len(runner.results)}")
print(f"   Resultados en: {OUTPUT_DIR}")