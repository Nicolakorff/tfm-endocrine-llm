"""
Experimento Fase 3: Comparación de Sesgo Simple vs Sesgo Semántico

Compara directamente el efecto de:
1. Sesgo simple (lista de tokens)
2. Sesgo semántico (embeddings SBERT)

Sobre las mismas métricas y los mismos prompts.
"""

from endocrine_llm import (
    EndocrineModulatedLLM,
    HORMONE_PROFILES,
    ExperimentRunner,
    TextMetrics
)
from endocrine_llm.semantic import (
    SemanticBiasManager,
    analyze_semantic_activation
)
import pandas as pd
import json
from pathlib import Path
from tqdm.auto import tqdm

# Configuración
OUTPUT_DIR = Path("data/results/semantic_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("EXPERIMENTO: SESGO SIMPLE VS SESGO SEMÁNTICO")
print("="*80 + "\n")

# 1. Inicializar modelo
print("Inicializando modelo...")
model = EndocrineModulatedLLM("gpt2")

# 2. Inicializar semantic manager
print("Inicializando SemanticBiasManager...")
semantic_manager = SemanticBiasManager(model.tokenizer, device=model.device)

# 3. Cargar prompts
print("Cargando prompts...")
prompts_df = pd.read_csv("data/prompts/prompts_dataset.csv")

# Seleccionar solo prompts de categorías relevantes
relevant_categories = ['empathic_support', 'creative_writing']
prompts_df = prompts_df[prompts_df['category'].isin(relevant_categories)]

print(f"Total prompts: {len(prompts_df)}")
print(f"Categorías: {prompts_df['category'].value_counts().to_dict()}")

# 4. Configuración experimental
EXPERIMENT_CONFIG = {
    'num_generations': 5,
    'max_new_tokens': 60,
    'hormone_profile': HORMONE_PROFILES["empathic"],  # Perfil con oxitocina alta
    'semantic_category': 'empathy',
    'semantic_strength': 1.5
}

print("\n Configuración:")
for key, value in EXPERIMENT_CONFIG.items():
    print(f"   {key}: {value}")

# 5. Ejecutar experimento
results = []

print("\n Ejecutando experimento...")
print("Condiciones: SIMPLE vs SEMANTIC")
print(f"Total generaciones: {len(prompts_df) * EXPERIMENT_CONFIG['num_generations'] * 2}\n")

with tqdm(total=len(prompts_df), desc="Prompts") as pbar:
    for idx, row in prompts_df.iterrows():
        prompt = row['prompt']
        category = row['category']

        for gen_idx in range(EXPERIMENT_CONFIG['num_generations']):

            # CONDICIÓN 1: Sesgo Simple (solo tokens)
            try:
                texts_simple = model.generate_with_hormones(
                    prompt=prompt,
                    hormone_profile=EXPERIMENT_CONFIG['hormone_profile'],
                    max_new_tokens=EXPERIMENT_CONFIG['max_new_tokens'],
                    num_return_sequences=1
                )

                text_simple = texts_simple[0]

                # Calcular métricas
                metrics_simple = TextMetrics.compute_all(text_simple)

                # Guardar resultado
                results.append({
                    'prompt': prompt,
                    'prompt_category': category,
                    'condition': 'simple_bias',
                    'generation_idx': gen_idx,
                    'generated_text': text_simple,
                    'hormone_profile': EXPERIMENT_CONFIG['hormone_profile'].to_dict(),
                    **metrics_simple
                })
            except Exception as e:  # noqa: W0718
                print(f"\n Error en condición SIMPLE: {e}")
                continue

        # CONDICIÓN 2: Sesgo Semántico (embeddings)
        try:
            texts_semantic = model.generate_with_semantic_bias(
                prompt=prompt,
                hormone_profile=EXPERIMENT_CONFIG['hormone_profile'],
                semantic_category=EXPERIMENT_CONFIG['semantic_category'],
                semantic_strength=EXPERIMENT_CONFIG['semantic_strength'],
                max_new_tokens=EXPERIMENT_CONFIG['max_new_tokens'],
                num_return_sequences=1
            )

            text_semantic = texts_semantic[0]

            # Calcular métricas básicas
            metrics_semantic = TextMetrics.compute_all(text_semantic)

            # Calcular activación semántica
            semantic_analysis = analyze_semantic_activation(
                text_semantic,
                semantic_manager
            )

            # Guardar resultado
            results.append({
                'prompt': prompt,
                'prompt_category': category,
                'condition': 'semantic_bias',
                'generation_idx': gen_idx,
                'generated_text': text_semantic,
                'hormone_profile': EXPERIMENT_CONFIG['hormone_profile'].to_dict(),
                'semantic_category': EXPERIMENT_CONFIG['semantic_category'],
                'semantic_strength': EXPERIMENT_CONFIG['semantic_strength'],
                'semantic_activation_empathy': semantic_analysis['similarities']['empathy'],
                'semantic_activation_creativity': semantic_analysis['similarities']['creativity'],
                'dominant_semantic_category': semantic_analysis['dominant_category'],
                **metrics_semantic
            })

        except Exception as e:  # noqa: W0718
            print(f"\n Error en condición SEMANTIC: {e}")
            continue

    pbar.update(1)

print(f"\n Experimento completado: {len(results)} generaciones")

# 6. Guardar resultados
print("\n Guardando resultados...")

# Guardar como JSON
with open(OUTPUT_DIR / "comparison_results.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Guardar como CSV para análisis
df = pd.DataFrame(results)
df.to_csv(OUTPUT_DIR / "comparison_results.csv", index=False)
print(f"JSON: {OUTPUT_DIR / 'comparison_results.json'}")
print(f"CSV: {OUTPUT_DIR / 'comparison_results.csv'}")

# 7. Análisis preliminar
print("\n" + "="*80)
print("ANÁLISIS PRELIMINAR")
print("="*80 + "\n")

# Comparar por condición
comparison = df.groupby('condition')[['distinct_2', 'sentiment_polarity', 'repetition_rate', 'length']].mean()

print("Métricas promedio por condición:")
print(comparison)
print()

# Test de diferencias
from scipy import stats

simple_data = df[df['condition'] == 'simple_bias']
semantic_data = df[df['condition'] == 'semantic_bias']

print("Tests estadísticos (Simple vs Semantic):")
print("-" * 60)

for metric in ['distinct_2', 'sentiment_polarity', 'repetition_rate']:
    simple_values = simple_data[metric].dropna()
    semantic_values = semantic_data[metric].dropna()

if len(simple_values) > 0 and len(semantic_values) > 0:
    t_stat, p_value = stats.ttest_ind(simple_values, semantic_values)

    print(f"\n{metric}:")
    print(f"Simple:   {simple_values.mean():.4f} (SD={simple_values.std():.4f})")
    print(f"Semantic: {semantic_values.mean():.4f} (SD={semantic_values.std():.4f})")
    print(f"Diferencia: {semantic_values.mean() - simple_values.mean():+.4f}")
    print(f"t={t_stat:.3f}, p={p_value:.4f} {'***' if p_value < 0.001 else
                                               '**' if p_value < 0.01 else
                                               '*' if p_value < 0.05 else 'ns'}")

# 8. Activación semántica
if 'semantic_activation_empathy' in df.columns:
    semantic_only = df[df['condition'] == 'semantic_bias']

    print(f"\n" + "="*80)
    print("ANÁLISIS DE ACTIVACIÓN SEMÁNTICA")
    print("="*80 + "\n")

    print(f"Activación promedio de 'empathy': {semantic_only['semantic_activation_empathy'].mean():.3f}")
    print(f"Activación promedio de 'creativity': {semantic_only['semantic_activation_creativity'].mean():.3f}")

    print("\n Categoría semántica dominante:")
    print(semantic_only['dominant_semantic_category'].value_counts())

    print("\n" + "="*80)
    print("Análisis preliminar completado")
    print("="*80)
