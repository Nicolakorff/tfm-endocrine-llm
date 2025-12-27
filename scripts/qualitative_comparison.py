"""
Análisis Cualitativo: 
Extrae ejemplos comparativos
"""

import pandas as pd
from pathlib import Path

# Cargar resultados
df = pd.read_csv("data/results/semantic_comparison/comparison_results.csv")

# Seleccionar prompts con mayor diferencia
print("Buscando ejemplos con mayor contraste...\n")

# Agrupar por prompt y calcular diferencia promedio en distinct_2
prompt_comparison = df.groupby(['prompt', 'condition'])['distinct_2'].mean().unstack()
prompt_comparison['difference'] = prompt_comparison['semantic_bias'] - prompt_comparison['simple_bias']
prompt_comparison = prompt_comparison.sort_values('difference', ascending=False)

# Top 5 prompts con mayor diferencia
top_prompts = prompt_comparison.head(5).index.tolist()

output_file = Path("data/results/semantic_comparison/qualitative_examples.txt")

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("ANÁLISIS CUALITATIVO: SESGO SIMPLE VS SESGO SEMÁNTICO\n")
    f.write("="*80 + "\n\n")

    f.write("Ejemplos seleccionados: Prompts con mayor diferencia en diversidad léxica\n\n")

    for prompt in top_prompts:
        f.write("\n" + "="*80 + "\n")
        f.write(f"PROMPT: {prompt}\n")
        f.write("="*80 + "\n\n")

        # Obtener ejemplos de cada condición
        simple_examples = df[
            (df['prompt'] == prompt) & 
            (df['condition'] == 'simple_bias')
        ].head(2)

        semantic_examples = df[
            (df['prompt'] == prompt) & 
            (df['condition'] == 'semantic_bias')
        ].head(2)

        # SIMPLE BIAS
        f.write("--- CONDICIÓN: SESGO SIMPLE (lista de tokens) ---\n\n")

        for idx, row in simple_examples.iterrows():
            f.write(f"Ejemplo {row['generation_idx'] + 1}:\n")
            f.write(f"{row['generated_text']}\n\n")
            f.write("Métricas:\n")
            f.write(f"  - Diversidad léxica: {row['distinct_2']:.3f}\n")
            f.write(f"  - Polaridad: {row['sentiment_polarity']:.3f}\n")
            f.write(f"  - Longitud: {row['length']} tokens\n\n")

        # SEMANTIC BIAS
        f.write("--- CONDICIÓN: SESGO SEMÁNTICO (embeddings) ---\n\n")

        for idx, row in semantic_examples.iterrows():
            f.write(f"Ejemplo {row['generation_idx'] + 1}:\n")
            f.write(f"{row['generated_text']}\n\n")
            f.write("Métricas:\n")
            f.write(f"  - Diversidad léxica: {row['distinct_2']:.3f}\n")
            f.write(f"  - Polaridad: {row['sentiment_polarity']:.3f}\n")
            f.write(f"  - Longitud: {row['length']} tokens\n")

            if 'semantic_activation_empathy' in row:
                f.write(f"  - Activación 'empathy': {row['semantic_activation_empathy']:.3f}\n")
                f.write(f"  - Categoría dominante: {row['dominant_semantic_category']}\n")

            f.write("\n")

        f.write("-"*80 + "\n\n")

print(f"Análisis cualitativo guardado: {output_file}")
