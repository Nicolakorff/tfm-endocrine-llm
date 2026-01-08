"""
Análisis Cualitativo para Experimento Semántico
==================================================

Extrae ejemplos representativos para inspección manual de:
- Mayor contraste entre condiciones
- Ejemplos de cada categoría semántica
- Casos de éxito y fracaso del sesgo semántico
"""

import pandas as pd
from pathlib import Path
import numpy as np

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

RESULTS_FILE = Path("data/results/semantic_comparison/comparison_results.csv")
OUTPUT_DIR = Path("data/results/semantic_comparison")
OUTPUT_FILE = OUTPUT_DIR / "qualitative_analysis.txt"

print("="*80)
print("ANÁLISIS CUALITATIVO: EXPERIMENTO SEMÁNTICO")
print("="*80)
print()

# ============================================================================
# CARGAR DATOS
# ============================================================================

print("Cargando resultados...")
df = pd.read_csv(RESULTS_FILE)

print(f"Total generaciones: {len(df)}")
print(f"Condiciones: {df['condition'].unique().tolist()}\n")

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def format_example(row, include_metrics=True):
    """Formatea un ejemplo para output de texto"""
    lines = []
    lines.append(f"Prompt: {row['prompt']}")
    lines.append(f"Texto generado:")
    lines.append(f"  {row['generated_text']}")
    
    if include_metrics:
        lines.append(f"\nMétricas:")
        lines.append(f"  • Diversidad léxica (Distinct-2): {row['distinct_2']:.3f}")
        lines.append(f"  • Polaridad sentimiento: {row['sentiment_polarity']:.3f}")
        lines.append(f"  • Tasa repetición: {row['repetition_rate']:.3f}")
        lines.append(f"  • Longitud: {row['length']} tokens")
        
        # Activación semántica si existe
        if 'semantic_activation_empathy' in row and pd.notna(row['semantic_activation_empathy']):
            lines.append(f"\nActivación semántica:")
            lines.append(f"  • Empathy: {row['semantic_activation_empathy']:.3f}")
            lines.append(f"  • Creativity: {row['semantic_activation_creativity']:.3f}")
            lines.append(f"  • Caution: {row['semantic_activation_caution']:.3f}")
            lines.append(f"  • Categoría dominante: {row['dominant_semantic_category']}")
        
        # Coherencia semántica si existe
        if 'semantic_coherence' in row and pd.notna(row['semantic_coherence']):
            lines.append(f"  • Coherencia con objetivo: {row['semantic_coherence']:.3f}")
    
    return "\n".join(lines)


def get_contrast_examples(df, metric='distinct_2', n=3):
    """Encuentra prompts con mayor diferencia entre condiciones"""
    
    # Calcular diferencia promedio entre semantic y lexical
    comparison = df[df['condition'].isin(['lexical_empathy', 'semantic_empathy'])]
    
    prompt_diffs = []
    for prompt in comparison['prompt'].unique():
        prompt_data = comparison[comparison['prompt'] == prompt]
        
        lexical_vals = prompt_data[prompt_data['condition'] == 'lexical_empathy'][metric]
        semantic_vals = prompt_data[prompt_data['condition'] == 'semantic_empathy'][metric]
        
        if len(lexical_vals) > 0 and len(semantic_vals) > 0:
            diff = semantic_vals.mean() - lexical_vals.mean()
            prompt_diffs.append({
                'prompt': prompt,
                'difference': abs(diff),
                'direction': 'higher' if diff > 0 else 'lower'
            })
    
    # Ordenar por diferencia
    prompt_diffs_df = pd.DataFrame(prompt_diffs).sort_values('difference', ascending=False)
    
    return prompt_diffs_df.head(n)


# ============================================================================
# GENERAR ANÁLISIS
# ============================================================================

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    
    # HEADER
    f.write("="*80 + "\n")
    f.write("ANÁLISIS CUALITATIVO: EXPERIMENTO SEMÁNTICO V2\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Total generaciones: {len(df)}\n")
    f.write(f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # ========================================================================
    # SECCIÓN 1: COMPARACIÓN BASELINE vs LEXICAL
    # ========================================================================
    
    f.write("\n" + "="*80 + "\n")
    f.write("1. EFECTO DEL SESGO LÉXICO (Baseline vs Lexical)\n")
    f.write("="*80 + "\n\n")
    
    baseline = df[df['condition'] == 'baseline']
    lexical = df[df['condition'] == 'lexical_empathy']
    
    if len(baseline) > 0 and len(lexical) > 0:
        # Seleccionar 3 ejemplos aleatorios con mismo prompt
        common_prompts = set(baseline['prompt'].unique()) & set(lexical['prompt'].unique())
        if common_prompts:
            sample_prompts = list(common_prompts)[:3]
            
            for i, prompt in enumerate(sample_prompts, 1):
                f.write(f"\n{'-'*80}\n")
                f.write(f"Ejemplo {i}\n")
                f.write(f"{'-'*80}\n\n")
                
                # Baseline
                f.write("BASELINE (sin modulación):\n")
                f.write("-" * 40 + "\n")
                baseline_example = baseline[baseline['prompt'] == prompt].iloc[0]
                f.write(format_example(baseline_example) + "\n\n")
                
                # Lexical
                f.write("LÉXICO (perfil empático):\n")
                f.write("-" * 40 + "\n")
                lexical_example = lexical[lexical['prompt'] == prompt].iloc[0]
                f.write(format_example(lexical_example) + "\n\n")
                
                # Comparación
                f.write("COMPARACIÓN:\n")
                diff_distinct = lexical_example['distinct_2'] - baseline_example['distinct_2']
                f.write(f"  • Δ Distinct-2: {diff_distinct:+.3f}\n")
                diff_sentiment = lexical_example['sentiment_polarity'] - baseline_example['sentiment_polarity']
                f.write(f"  • Δ Sentimiento: {diff_sentiment:+.3f}\n\n")
    else:
        f.write("Datos insuficientes para esta comparación\n\n")
    
    # ========================================================================
    # SECCIÓN 2: COMPARACIÓN LEXICAL vs SEMANTIC-EMPATHY
    # ========================================================================
    
    f.write("\n" + "="*80 + "\n")
    f.write("2. EFECTO DEL SESGO SEMÁNTICO (Lexical vs Semantic-Empathy)\n")
    f.write("="*80 + "\n\n")
    
    f.write("Criterio de selección: Prompts con MAYOR CONTRASTE en Distinct-2\n\n")
    
    contrast_prompts = get_contrast_examples(df, metric='distinct_2', n=5)
    
    for idx, row in contrast_prompts.iterrows():
        prompt = row['prompt']
        
        f.write(f"\n{'-'*80}\n")
        f.write(f"Ejemplo con contraste alto (diferencia: {row['difference']:.3f})\n")
        f.write(f"{'-'*80}\n\n")
        
        # Léxico
        f.write("LÉXICO (solo modulación hormonal):\n")
        f.write("-" * 40 + "\n")
        lexical_examples = lexical[lexical['prompt'] == prompt].head(2)
        for i, lex_row in lexical_examples.iterrows():
            f.write(format_example(lex_row, include_metrics=True) + "\n\n")
        
        # Semántico
        f.write("SEMÁNTICO (hormonal + embeddings):\n")
        f.write("-" * 40 + "\n")
        semantic_examples = df[
            (df['condition'] == 'semantic_empathy') & 
            (df['prompt'] == prompt)
        ].head(2)
        for i, sem_row in semantic_examples.iterrows():
            f.write(format_example(sem_row, include_metrics=True) + "\n\n")
    
    # ========================================================================
    # SECCIÓN 3: COMPARACIÓN ENTRE CATEGORÍAS SEMÁNTICAS
    # ========================================================================
    
    f.write("\n" + "="*80 + "\n")
    f.write("3. COMPARACIÓN ENTRE CATEGORÍAS SEMÁNTICAS\n")
    f.write("="*80 + "\n\n")
    
    semantic_conditions = ['semantic_empathy', 'semantic_creativity', 'semantic_caution']
    
    # Seleccionar un prompt que esté en todas las condiciones
    semantic_data = df[df['condition'].isin(semantic_conditions)]
    
    if len(semantic_data) > 0:
        # Encontrar prompts presentes en todas las categorías
        prompt_counts = semantic_data.groupby('prompt')['condition'].nunique()
        complete_prompts = prompt_counts[prompt_counts == len(semantic_conditions)].index.tolist()
        
        if complete_prompts:
            sample_prompt = complete_prompts[0]  # Tomar el primero
            
            f.write(f"Prompt seleccionado: {sample_prompt}\n\n")
            
            for condition in semantic_conditions:
                category_name = condition.replace('semantic_', '').upper()
                
                f.write(f"\n{'-'*80}\n")
                f.write(f"{category_name}\n")
                f.write(f"{'-'*80}\n\n")
                
                examples = semantic_data[
                    (semantic_data['condition'] == condition) & 
                    (semantic_data['prompt'] == sample_prompt)
                ].head(2)
                
                for i, ex_row in examples.iterrows():
                    f.write(format_example(ex_row, include_metrics=True) + "\n\n")
        else:
            f.write("No hay prompts presentes en todas las categorías\n\n")
    
    # ========================================================================
    # SECCIÓN 4: CASOS DE ALTA Y BAJA COHERENCIA SEMÁNTICA
    # ========================================================================
    
    f.write("\n" + "="*80 + "\n")
    f.write("4. ANÁLISIS DE COHERENCIA SEMÁNTICA\n")
    f.write("="*80 + "\n\n")
    
    semantic_with_coherence = df[df['semantic_coherence'].notna()]
    
    if len(semantic_with_coherence) > 0:
        # Top 3 con mayor coherencia
        f.write("TOP 3: MAYOR COHERENCIA SEMÁNTICA\n")
        f.write("-" * 80 + "\n\n")
        
        top_coherence = semantic_with_coherence.nlargest(3, 'semantic_coherence')
        for i, row in top_coherence.iterrows():
            f.write(f"Coherencia: {row['semantic_coherence']:.3f} | Condición: {row['condition']}\n")
            f.write(format_example(row, include_metrics=True))
            f.write("\n" + "-"*40 + "\n\n")
        
        # Bottom 3 con menor coherencia
        f.write("\nBOTTOM 3: MENOR COHERENCIA SEMÁNTICA\n")
        f.write("-" * 80 + "\n\n")
        
        bottom_coherence = semantic_with_coherence.nsmallest(3, 'semantic_coherence')
        for i, row in bottom_coherence.iterrows():
            f.write(f"Coherencia: {row['semantic_coherence']:.3f} | Condición: {row['condition']}\n")
            f.write(format_example(row, include_metrics=True))
            f.write("\n" + "-"*40 + "\n\n")
    else:
        f.write("⚠ No hay datos de coherencia semántica\n\n")
    
    # ========================================================================
    # SECCIÓN 5: ESTADÍSTICAS RESUMEN
    # ========================================================================
    
    f.write("\n" + "="*80 + "\n")
    f.write("5. ESTADÍSTICAS RESUMEN\n")
    f.write("="*80 + "\n\n")
    
    summary = df.groupby('condition').agg({
        'distinct_2': ['mean', 'std'],
        'sentiment_polarity': ['mean', 'std'],
        'repetition_rate': ['mean', 'std'],
        'semantic_coherence': ['mean', 'std'] if 'semantic_coherence' in df.columns else lambda x: None
    })
    
    f.write("Métricas promedio por condición:\n\n")
    f.write(summary.to_string())
    f.write("\n\n")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    f.write("\n" + "="*80 + "\n")
    f.write("FIN DEL ANÁLISIS CUALITATIVO\n")
    f.write("="*80 + "\n")

print(f"Análisis cualitativo guardado en: {OUTPUT_FILE}")
print(f"\nRevisión recomendada:")
print("  1. Leer ejemplos de Sección 2 (mayor contraste)")
print("  2. Verificar coherencia semántica en Sección 4")
print("  3. Comparar estadísticas en Sección 5")