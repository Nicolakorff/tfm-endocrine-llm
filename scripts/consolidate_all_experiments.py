"""
Consolida todos los experimentos en un único dataset para análisis estadístico.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

print("="*80)
print(" CONSOLIDACIÓN DE TODOS LOS EXPERIMENTOS")
print("="*80 + "\n")

# Directorios
DATA_DIR = Path("data/results")
OUTPUT_DIR = DATA_DIR / "consolidated"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. CARGAR TODOS LOS DATASETS
print("1. Cargando datasets...")

datasets = {}

# Fase 1: Hormonas individuales
if (DATA_DIR / "phase1_results.csv").exists():
    df_phase1 = pd.read_csv(DATA_DIR / "phase1_results.csv")
    df_phase1['experiment'] = 'phase1'
    datasets['phase1'] = df_phase1
    print(f"   Fase 1: {len(df_phase1)} generaciones")
else:
    print("   Fase 1 no encontrada")

# Fase 2: Perfiles combinados
if (DATA_DIR / "phase2_results.csv").exists():
    df_phase2 = pd.read_csv(DATA_DIR / "phase2_results.csv")
    df_phase2['experiment'] = 'phase2'
    datasets['phase2'] = df_phase2
    print(f"   Fase 2: {len(df_phase2)} generaciones")
else:
    print("   Fase 2 no encontrada")

# Comparación semántica
if (DATA_DIR / "semantic_comparison/comparison_results.csv").exists():
    df_semantic = pd.read_csv(DATA_DIR / "semantic_comparison/comparison_results.csv")
    df_semantic['experiment'] = 'semantic_comparison'
    datasets['semantic'] = df_semantic
    print(f"   Semántico: {len(df_semantic)} generaciones")
else:
    print("   Experimento semántico no encontrado")

if not datasets:
    print("\n No se encontraron datasets. Ejecuta los experimentos primero.")
    exit(1)

# 2. ESTANDARIZAR COLUMNAS
print("\n2. Estandarizando columnas...")

# Identificar columnas comunes
common_columns = set.intersection(*[set(df.columns) for df in datasets.values()])
print(f"   Columnas comunes: {len(common_columns)}")

# Columnas clave que deben estar
required_columns = [
    'prompt', 'profile_name', 'generated_text', 
    'distinct_2', 'sentiment_polarity', 'repetition_rate', 'length'
]

for col in required_columns:
    if col not in common_columns:
        print(f"   Columna '{col}' no está en todos los datasets")

# Seleccionar columnas relevantes
selected_columns = [
    'experiment',
    'prompt',
    'prompt_category',  # Si existe
    'profile_name',
    'generated_text',
    'generation_idx',
    'length',
    'distinct_1',
    'distinct_2',
    'distinct_3',
    'repetition_rate',
    'sentiment_polarity',
    'sentiment_subjectivity',
]

# Columnas opcionales
optional_columns = [
    'perplexity',
    'rouge_l',
    'entropy',
    'empathy_word_score',
    'semantic_activation_empathy',
    'condition'  # Para experimento semántico
]

# 3. CONSOLIDAR
print("\n3. Consolidando datasets...")

consolidated_dfs = []

for name, df in datasets.items():
    # Seleccionar columnas disponibles
    available_cols = [col for col in selected_columns if col in df.columns]
    df_subset = df[available_cols].copy()

    # Añadir columnas opcionales si existen
    for col in optional_columns:
        if col in df.columns:
            df_subset[col] = df[col]

    consolidated_dfs.append(df_subset)

# Combinar todos
df_all = pd.concat(consolidated_dfs, ignore_index=True)

print(f"   Total generaciones consolidadas: {len(df_all)}")
print(f"   Columnas finales: {len(df_all.columns)}")

# 4. AÑADIR METADATOS HORMONALES
print("\n4. Extrayendo metadatos hormonales...")

# Si existe columna hormone_profile como string JSON, parsear
if 'hormone_profile' in df_all.columns:
    try:
        # Intentar parsear JSON
        hormone_dicts = df_all['hormone_profile'].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )

        # Extraer niveles hormonales
        df_all['hormone_dopamine'] = hormone_dicts.apply(
            lambda x: x.get('dopamine', np.nan) if isinstance(x, dict) else np.nan
        )
        df_all['hormone_cortisol'] = hormone_dicts.apply(
            lambda x: x.get('cortisol', np.nan) if isinstance(x, dict) else np.nan
        )
        df_all['hormone_oxytocin'] = hormone_dicts.apply(
            lambda x: x.get('oxytocin', np.nan) if isinstance(x, dict) else np.nan
        )
        df_all['hormone_adrenaline'] = hormone_dicts.apply(
            lambda x: x.get('adrenaline', np.nan) if isinstance(x, dict) else np.nan
        )
        df_all['hormone_serotonin'] = hormone_dicts.apply(
            lambda x: x.get('serotonin', np.nan) if isinstance(x, dict) else np.nan
        )

        print("   Niveles hormonales extraídos")
    except Exception as e:
        print(f"   No se pudieron extraer niveles hormonales: {e}")

# 5. LIMPIAR Y VALIDAR
print("\n5. Limpiando datos...")

# Remover filas con datos faltantes críticos
initial_count = len(df_all)
df_all = df_all.dropna(subset=['profile_name', 'generated_text', 'distinct_2'])
final_count = len(df_all)

if initial_count > final_count:
    print(f"   Removidas {initial_count - final_count} filas con datos faltantes")

# Validar rangos
invalid_distinct = (df_all['distinct_2'] < 0) | (df_all['distinct_2'] > 1)
if invalid_distinct.any():
    print(f"   {invalid_distinct.sum()} valores de distinct_2 fuera de rango [0,1]")
    df_all = df_all[~invalid_distinct]

# 6. ESTADÍSTICAS DESCRIPTIVAS
print("\n6. Estadísticas del dataset consolidado:")
print("-" * 80)

print(f"\nTotal generaciones: {len(df_all)}")
print("\nExperimentos:")
for exp in df_all['experiment'].unique():
    count = len(df_all[df_all['experiment'] == exp])
    print(f"   {exp}: {count} ({count/len(df_all)*100:.1f}%)")

print("\nPerfiles hormonales:")
for profile in df_all['profile_name'].value_counts().head(10).items():
    print(f"   {profile[0]}: {profile[1]}")

if 'prompt_category' in df_all.columns:
    print("\nCategorías de prompts:")
    for cat in df_all['prompt_category'].value_counts().items():
        print(f"   {cat[0]}: {cat[1]}")

# 7. GUARDAR
print("\n7. Guardando dataset consolidado...")

# CSV principal
df_all.to_csv(OUTPUT_DIR / "all_experiments_consolidated.csv", index=False)
print(f"   CSV: {OUTPUT_DIR / 'all_experiments_consolidated.csv'}")

# Pickle para preservar tipos de datos
df_all.to_pickle(OUTPUT_DIR / "all_experiments_consolidated.pkl")
print(f"   Pickle: {OUTPUT_DIR / 'all_experiments_consolidated.pkl'}")

# Metadatos
metadata = {
    'total_generations': len(df_all),
    'experiments': df_all['experiment'].value_counts().to_dict(),
    'profiles': df_all['profile_name'].value_counts().to_dict(),
    'columns': list(df_all.columns),
    'date_created': pd.Timestamp.now().isoformat()
}

with open(OUTPUT_DIR / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"   Metadata: {OUTPUT_DIR / 'metadata.json'}")

# Estadísticas descriptivas
desc_stats = df_all.describe(include='all').T
desc_stats.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")
print(f"   Estadísticas: {OUTPUT_DIR / 'descriptive_statistics.csv'}")

print("\n" + "="*80)
print(" CONSOLIDACIÓN COMPLETADA")
print("="*80)
print("\nDataset final:")
print(f"   Filas: {len(df_all)}")
print(f"   Columnas: {len(df_all.columns)}")
print(f"   Tamaño: {df_all.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
