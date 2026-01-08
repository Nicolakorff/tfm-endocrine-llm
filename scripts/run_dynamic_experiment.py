"""
Experimento dinámico (Fase 4):
Compara sistema estático vs dinámico
"""

import pandas as pd
from pathlib import Path
from endocrine_llm import EndocrineModulatedLLM, ExperimentRunner
import torch

print("="*80)
print("EXPERIMENTO DINÁMICO - FASE 4")
print("="*80 + "\n")

# Configuración
DATA_DIR = Path("data")
PROMPTS_FILE = DATA_DIR / "prompts/prompts_dataset.csv"
OUTPUT_FILE = DATA_DIR / "results/phase4_dynamic_results.csv"

# Parámetros del experimento
NUM_GENERATIONS = 3          # Repeticiones por combinación
MAX_NEW_TOKENS = 50         # Longitud de generación
UPDATE_INTERVAL = 5         # Actualizar hormonas cada 5 tokens

# Verificar GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {device}")
if device == "cpu":
    print("Usando CPU - El experimento será más lento")
    print("Considera usar Google Colab con GPU gratuita")

# 1. CARGAR DATASET DE PROMPTS
print("\n1. Cargando dataset de prompts...")
if not PROMPTS_FILE.exists():
    print(f"ERROR: No se encontró {PROMPTS_FILE}")
    print("\n Primero debes crear el dataset de prompts:")
    print("python scripts/create_prompts_dataset.py")
    exit(1)

prompts_df = pd.read_csv(PROMPTS_FILE)
print(f"Cargados {len(prompts_df)} prompts")
print(f"Categorías: {prompts_df['category'].value_counts().to_dict()}")

# 2. INICIALIZAR MODELO
print("\n 2. Inicializando modelo...")
model = EndocrineModulatedLLM("Gpt2", device=device)
print("Modelo cargado")

# 3. INICIALIZAR RUNNER
print("\n 3. Inicializando ExperimentRunner...")
runner = ExperimentRunner(model, compute_advanced_metrics=True)  # True para incluir perplexity
print("Runner inicializado")
print("Métricas avanzadas activadas (incluye perplexity)")

# 4. EJECUTAR EXPERIMENTO DINÁMICO
print("\n 4. Ejecutando experimento dinámico...")
print("Configuración:")
print(f"- Prompts: {len(prompts_df)}")
print(f"- Generaciones por prompt: {NUM_GENERATIONS}")
print("- Perfiles: 6 (3 estáticos + 3 dinámicos)")
print(f"- Total generaciones: {len(prompts_df) * NUM_GENERATIONS * 6}")
print(f"- Update interval: {UPDATE_INTERVAL} tokens")
print("\n Tiempo estimado: ~45-60 minutos en GPU, ~2-3 horas en CPU")

input("\n Presiona ENTER para comenzar o Ctrl+C para cancelar...")

try:
    df_results = runner.run_dynamic_experiment(
        prompts_df=prompts_df,
        num_generations=NUM_GENERATIONS,
        max_new_tokens=MAX_NEW_TOKENS,
        update_interval=UPDATE_INTERVAL,
        save_path=OUTPUT_FILE
    )
    
    print("\n EXPERIMENTO COMPLETADO")
    print(f"Resultados guardados en: {OUTPUT_FILE}")
    
    # 5. ESTADÍSTICAS RESUMIDAS
    print("\n 5. Estadísticas resumidas:")
    print("-" * 80)
    
    print(f"\n Total generaciones: {len(df_results)}")
    
    print("\n Por condición:")
    for is_dynamic in [False, True]:
        subset = df_results[df_results['is_dynamic'] == is_dynamic]
        label = "Dinámico" if is_dynamic else "Estático"
        print(f"{label:12s}: {len(subset):4d} generaciones")
    
    print("\n Por perfil:")
    for profile in df_results['profile_name'].value_counts().items():
        print(f"{profile[0]:25s}: {profile[1]:4d}")
    
    print("\n Por categoría:")
    for category in df_results['category'].value_counts().items():
        print(f"{category[0]:20s}: {category[1]:4d}")
    
    # Métricas clave
    dynamic_subset = df_results[df_results['is_dynamic'] == True]
    static_subset = df_results[df_results['is_dynamic'] == False]
    
    print("\n Métricas promedio:")
    print("\n Diversidad Léxica (Distinct-2):")
    print(f"Estático:  {static_subset['distinct_2'].mean():.4f} ± {static_subset['distinct_2'].std():.4f}")
    print(f"Dinámico:  {dynamic_subset['distinct_2'].mean():.4f} ± {dynamic_subset['distinct_2'].std():.4f}")
    print(f"Diferencia: {dynamic_subset['distinct_2'].mean() - static_subset['distinct_2'].mean():+.4f}")
    
    if 'total_hormone_change' in dynamic_subset.columns:
        print("\n Cambio Hormonal Total (solo dinámico):")
        print(f"Media:  {dynamic_subset['total_hormone_change'].mean():.4f}")
        print(f"SD:     {dynamic_subset['total_hormone_change'].std():.4f}")
        print(f"Rango:  [{dynamic_subset['total_hormone_change'].min():.4f}, {dynamic_subset['total_hormone_change'].max():.4f}]")
    
    print("\n" + "="*80)
    print("EXPERIMENTO DINÁMICO COMPLETADO EXITOSAMENTE")
    print("="*80)
    
    print("\n Próximos pasos:")
    print("1. Consolidar resultados:")
    print("python scripts/consolidate_all_experiments.py")
    print("\n 2. Crear figura maestra:")
    print("python scripts/create_master_figure.py")
    print("\n 3. Análisis estadístico:")
    print("python scripts/analyze_dynamic_results.py")

except KeyboardInterrupt:
    print("\n\n Experimento interrumpido por el usuario")
    print("Resultados parciales pueden estar guardados")

except Exception as e:
    print("\n ERROR durante el experimento:")
    print(f"   {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)
