"""
Experimento Fase 3: Comparación Léxico vs Semántico
==============================================================

"""

from endocrine_llm import (
    EndocrineModulatedLLM,
    HORMONE_PROFILES,
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
import numpy as np

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

OUTPUT_DIR = Path("data/results/semantic_comparison_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Generaciones por condición
N_PER_CONDITION = 250  # Balanceado

# Configuración experimental
SEMANTIC_STRENGTH = 3.0  # Aumentado de 1.5 a 3.0
MAX_NEW_TOKENS = 60

print("="*80)
print("EXPERIMENTO MEJORADO: LÉXICO VS SEMÁNTICO")
print("="*80)
print(f"\nN por condición: {N_PER_CONDITION}")
print(f"Semantic strength: {SEMANTIC_STRENGTH}")
print(f"Total generaciones esperadas: {N_PER_CONDITION * 5}")
print()

# ============================================================================
# INICIALIZACIÓN
# ============================================================================

print("Inicializando modelo...")
model = EndocrineModulatedLLM("gpt2")

print("Inicializando SemanticBiasManager...")
semantic_manager = SemanticBiasManager(model.tokenizer, device=model.device)

# ============================================================================
# CARGAR PROMPTS
# ============================================================================

print("Cargando prompts...")
prompts_df = pd.read_csv("data/prompts/prompts_dataset.csv")

# Usar todas las categorías disponibles
print(f"Total prompts disponibles: {len(prompts_df)}")
print(f"Categorías: {prompts_df['category'].unique().tolist()}")

# ============================================================================
# DISEÑO EXPERIMENTAL: 5 CONDICIONES
# ============================================================================

experimental_conditions = [
    {
        'name': 'baseline',
        'description': 'Sin modulación hormonal ni semántica',
        'hormone_profile': None,
        'semantic_category': None,
        'semantic_strength': 0.0
    },
    {
        'name': 'lexical_empathy',
        'description': 'Solo sesgo léxico con perfil empático',
        'hormone_profile': HORMONE_PROFILES["empathic"],
        'semantic_category': None,
        'semantic_strength': 0.0
    },
    {
        'name': 'semantic_empathy',
        'description': 'Sesgo semántico empathy (coherente con hormonal)',
        'hormone_profile': HORMONE_PROFILES["empathic"],
        'semantic_category': 'empathy',
        'semantic_strength': SEMANTIC_STRENGTH
    },
    {
        'name': 'semantic_creativity',
        'description': 'Sesgo semántico creativity (parcialmente coherente)',
        'hormone_profile': HORMONE_PROFILES["empathic"],  # conflicto intencional
        'semantic_category': 'creativity',
        'semantic_strength': SEMANTIC_STRENGTH
    },
    {
        'name': 'semantic_caution',
        'description': 'Sesgo semántico caution (opuesto a empático)',
        'hormone_profile': HORMONE_PROFILES["empathic"],  # conflicto alto
        'semantic_category': 'caution',
        'semantic_strength': SEMANTIC_STRENGTH
    }
]

print("\nCondiciones experimentales:")
for i, cond in enumerate(experimental_conditions, 1):
    print(f"{i}. {cond['name']}: {cond['description']}")

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def compute_semantic_coherence(text: str, target_category: str, 
                               semantic_manager: SemanticBiasManager) -> float:
    """
    Métrica nueva: coherencia semántica con categoría objetivo
    
    Returns:
        float: similitud coseno con categoría [0, 1]
    """
    similarities = semantic_manager.compare_categories(text)
    return similarities.get(target_category, 0.0)


def generate_single(model, prompt, condition):
    """Genera un texto bajo una condición específica"""
    
    if condition['hormone_profile'] is None:
        # Baseline: generación vanilla
        inputs = model.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        outputs = model.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=model.tokenizer.pad_token_id,
            eos_token_id=model.tokenizer.eos_token_id,
        )
        
        return model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    elif condition['semantic_category'] is None:
        # Solo léxico
        texts = model.generate_with_hormones(
            prompt=prompt,
            hormone_profile=condition['hormone_profile'],
            max_new_tokens=MAX_NEW_TOKENS,
            num_return_sequences=1
        )
        return texts[0]
    
    else:
        # Léxico + semántico
        texts = model.generate_with_semantic_bias(
            prompt=prompt,
            hormone_profile=condition['hormone_profile'],
            semantic_category=condition['semantic_category'],
            semantic_strength=condition['semantic_strength'],
            max_new_tokens=MAX_NEW_TOKENS,
            num_return_sequences=1
        )
        return texts[0]


# ============================================================================
# EJECUTAR EXPERIMENTO
# ============================================================================

results = []
total_generations = N_PER_CONDITION * len(experimental_conditions)

print(f"\n{'='*80}")
print("EJECUTANDO EXPERIMENTO")
print(f"{'='*80}\n")

with tqdm(total=total_generations, desc="Generaciones totales") as pbar:
    
    for condition in experimental_conditions:
        condition_name = condition['name']
        
        print(f"\n Condición: {condition_name}")
        print(f"   {condition['description']}")
        
        # Generar N_PER_CONDITION textos
        for i in range(N_PER_CONDITION):
            
            # Seleccionar prompt aleatorio
            prompt_row = prompts_df.sample(n=1).iloc[0]
            prompt = prompt_row['prompt']
            prompt_category = prompt_row['category']
            
            try:
                # Generar texto
                generated_text = generate_single(model, prompt, condition)
                
                # Métricas básicas
                metrics = TextMetrics.compute_all(generated_text)
                
                # Análisis semántico (para todas las condiciones)
                semantic_analysis = analyze_semantic_activation(
                    generated_text,
                    semantic_manager
                )
                
                # Coherencia semántica (si aplica)
                semantic_coherence = None
                if condition['semantic_category'] is not None:
                    semantic_coherence = compute_semantic_coherence(
                        generated_text,
                        condition['semantic_category'],
                        semantic_manager
                    )
                
                # Guardar resultado
                result = {
                    'condition': condition_name,
                    'condition_type': (
                        'baseline' if condition['hormone_profile'] is None 
                        else 'lexical_only' if condition['semantic_category'] is None
                        else 'semantic'
                    ),
                    'hormone_profile_name': (
                        None if condition['hormone_profile'] is None
                        else 'empathic'
                    ),
                    'semantic_category': condition['semantic_category'],
                    'semantic_strength': condition['semantic_strength'],
                    'prompt': prompt,
                    'prompt_category': prompt_category,
                    'generated_text': generated_text,
                    'generation_idx': i,
                    
                    # Métricas básicas
                    **metrics,
                    
                    # Análisis semántico
                    'semantic_activation_empathy': semantic_analysis['similarities']['empathy'],
                    'semantic_activation_creativity': semantic_analysis['similarities']['creativity'],
                    'semantic_activation_caution': semantic_analysis['similarities']['caution'],
                    'semantic_activation_factual': semantic_analysis['similarities']['factual'],
                    'semantic_activation_enthusiasm': semantic_analysis['similarities']['enthusiasm'],
                    'dominant_semantic_category': semantic_analysis['dominant_category'],
                    'dominant_semantic_score': semantic_analysis['dominant_score'],
                    'semantic_coherence': semantic_coherence,
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"\nError en generación {i}: {e}")
                continue
            
            pbar.update(1)

print(f"\n\n Experimento completado: {len(results)} generaciones")

# ============================================================================
# GUARDAR RESULTADOS
# ============================================================================

print("\n Guardando resultados...")

# JSON
with open(OUTPUT_DIR / "results.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# CSV
df = pd.DataFrame(results)
df.to_csv(OUTPUT_DIR / "results.csv", index=False)

print(f"✓ JSON: {OUTPUT_DIR / 'results.json'}")
print(f"✓ CSV: {OUTPUT_DIR / 'results.csv'}")

# ============================================================================
# ANÁLISIS PRELIMINAR
# ============================================================================

print(f"\n{'='*80}")
print("ANÁLISIS PRELIMINAR")
print(f"{'='*80}\n")

# 1. Distribución de generaciones
print("1. Generaciones por condición:")
print(df['condition'].value_counts().sort_index())
print()

# 2. Métricas por condición
print("2. Métricas promedio por condición:")
metrics_cols = ['distinct_2', 'sentiment_polarity', 'repetition_rate', 'length', 'perplexity']
summary = df.groupby('condition')[metrics_cols].agg(['mean', 'std'])
print(summary.round(4))
print()

# 3. Activación semántica
print("3. Activación semántica por condición:")
semantic_cols = [
    'semantic_activation_empathy',
    'semantic_activation_creativity', 
    'semantic_activation_caution'
]
semantic_summary = df.groupby('condition')[semantic_cols].mean()
print(semantic_summary.round(3))
print()

# 4. Coherencia semántica (solo condiciones semánticas)
semantic_conditions = df[df['semantic_coherence'].notna()]
if len(semantic_conditions) > 0:
    print("4. Coherencia semántica (similitud con categoría objetivo):")
    coherence = semantic_conditions.groupby('condition')['semantic_coherence'].agg(['mean', 'std', 'min', 'max'])
    print(coherence.round(3))
    print()

# 5. Tests estadísticos
print("5. Comparaciones estadísticas:")
print("-" * 60)

from scipy import stats

# Comparar lexical vs semantic_empathy
lexical = df[df['condition'] == 'lexical_empathy']
semantic_emp = df[df['condition'] == 'semantic_empathy']

for metric in ['distinct_2', 'repetition_rate', 'sentiment_polarity', 'perplexity']:
    lex_vals = lexical[metric].dropna()
    sem_vals = semantic_emp[metric].dropna()
    
    if len(lex_vals) > 0 and len(sem_vals) > 0:
        t_stat, p_value = stats.ttest_ind(lex_vals, sem_vals)
        cohen_d = (sem_vals.mean() - lex_vals.mean()) / np.sqrt(
            (lex_vals.std()**2 + sem_vals.std()**2) / 2
        )
        
        print(f"\n{metric} (Lexical vs Semantic-Empathy):")
        print(f"  Lexical:  {lex_vals.mean():.4f} (SD={lex_vals.std():.4f})")
        print(f"  Semantic: {sem_vals.mean():.4f} (SD={sem_vals.std():.4f})")
        print(f"  Diferencia: {sem_vals.mean() - lex_vals.mean():+.4f}")
        print(f"  t={t_stat:.3f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
        print(f"  Cohen's d={cohen_d:.3f}")

print(f"\n{'='*80}")
print("ANÁLISIS COMPLETADO")
print(f"{'='*80}")
print(f"\nResultados guardados en: {OUTPUT_DIR}")
print("\nPróximos pasos:")
print("  1. Revisar CSV para análisis detallado")
print("  2. Ejecutar ANOVA para comparar todas las condiciones")
print("  3. Crear visualizaciones comparativas")