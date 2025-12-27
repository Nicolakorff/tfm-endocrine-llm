"""
Consolida todos los experimentos en un único dataset para análisis estadístico.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

print("="*80)
print("CONSOLIDACIÓN DE TODOS LOS EXPERIMENTOS")
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
    print(f"Fase 1: {len(df_phase1)} generaciones")
else:
    print("Fase 1 no encontrada")

# Fase 2: Perfiles combinados
if (DATA_DIR / "phase2_results.csv").exists():
    df_phase2 = pd.read_csv(DATA_DIR / "phase2_results.csv")
    df_phase2['experiment'] = 'phase2'
    datasets['phase2'] = df_phase2
    print(f"Fase 2: {len(df_phase2)} generaciones")
else:
    print("Fase 2 no encontrada")

# Fase 3: Sistema dinámico (CORREGIDO)
if (DATA_DIR / "phase3_dynamic_results.csv").exists():
    df_phase3 = pd.read_csv(DATA_DIR / "phase3_dynamic_results.csv")
    df_phase3['experiment'] = 'phase3_dynamic'
    datasets['phase3'] = df_phase3
    print(f"Fase 3 (Dinámico): {len(df_phase3)} generaciones")
else:
    print("Fase 3 no encontrada")

# Comparación semántica
if (DATA_DIR / "semantic_comparison/comparison_results.csv").exists():
    df_semantic = pd.read_csv(DATA_DIR / "semantic_comparison/comparison_results.csv")
    df_semantic['experiment'] = 'semantic_comparison'
    datasets['semantic'] = df_semantic
    print(f"Semántico: {len(df_semantic)} generaciones")
else:
    print("Experimento semántico no encontrado")

if not datasets:
    print("\n No se encontraron datasets. Ejecuta los experimentos primero.")
    exit(1)

# 2. ESTANDARIZAR COLUMNAS
print("\n2. Estandarizando columnas...")

# Identificar columnas comunes
common_columns = set.intersection(*[set(df.columns) for df in datasets.values()])
print(f"Columnas comunes: {len(common_columns)}")

# Columnas clave que deben estar
required_columns = [
    'prompt', 'profile_name', 'generated_text', 
    'distinct_2', 'sentiment_polarity', 'repetition_rate', 'length'
]

missing_required = []
for col in required_columns:
    if col not in common_columns:
        missing_required.append(col)

if missing_required:
    print(f"Columnas faltantes en algunos datasets: {missing_required}")

# Seleccionar columnas relevantes (comunes a la mayoría)
selected_columns = [
    'experiment',
    'prompt',
    'profile_name',
    'generated_text',
    'length',
    'distinct_1',
    'distinct_2',
    'distinct_3',
    'repetition_rate',
    'sentiment_polarity',
    'sentiment_subjectivity',
]

# Columnas opcionales (pueden no estar en todos)
optional_columns = [
    'prompt_category',
    'category',  # Fase 3 usa 'category'
    'generation_idx',
    'repetition',  # Fase 3 usa 'repetition'
    'perplexity',
    'rouge_l',
    'entropy',
    'empathy_word_score',
    'semantic_activation_empathy',
    'condition',
    # Columnas específicas de Fase 3
    'is_dynamic',
    'learning_rate',
    'init_dopamine',
    'init_cortisol',
    'init_oxytocin',
    'init_adrenaline',
    'init_serotonin',
    'final_dopamine',
    'final_cortisol',
    'final_oxytocin',
    'final_adrenaline',
    'final_serotonin',
    'delta_dopamine',
    'delta_cortisol',
    'delta_oxytocin',
    'delta_adrenaline',
    'delta_serotonin',
    'total_hormone_change',
]

# 3. CONSOLIDAR
print("\n3. Consolidando datasets...")

consolidated_dfs = []

for name, df in datasets.items():
    print(f"\n Procesando {name}...")

    # Crear DataFrame base con columnas seleccionadas
    df_subset = pd.DataFrame()

    # Añadir columnas seleccionadas que existan
    for col in selected_columns:
        if col in df.columns:
            df_subset[col] = df[col]
        elif col == 'experiment':
            df_subset[col] = df[col]
        else:
            # Columna faltante - llenar con NaN
            df_subset[col] = np.nan

    # Añadir columnas opcionales si existen
    for col in optional_columns:
        if col in df.columns:
            df_subset[col] = df[col]

    # Normalizar nombres de columnas específicas
    # 'category' → 'prompt_category'
    if 'category' in df_subset.columns and 'prompt_category' not in df_subset.columns:
        df_subset['prompt_category'] = df_subset['category']
        df_subset = df_subset.drop('category', axis=1)

    # 'repetition' → 'generation_idx'
    if 'repetition' in df_subset.columns and 'generation_idx' not in df_subset.columns:
        df_subset['generation_idx'] = df_subset['repetition']
        df_subset = df_subset.drop('repetition', axis=1)

    print(f"Columnas: {len(df_subset.columns)}")
    print(f"Filas: {len(df_subset)}")

    consolidated_dfs.append(df_subset)

# Combinar todos
print("\nCombinando datasets...")
df_all = pd.concat(consolidated_dfs, ignore_index=True, sort=False)

print(f"Total generaciones consolidadas: {len(df_all)}")
print(f"Columnas finales: {len(df_all.columns)}")

# 4. AÑADIR METADATOS HORMONALES
print("\n4. Extrayendo metadatos hormonales...")

# Si existe columna hormone_profile como string JSON, parsear
if 'hormone_profile' in df_all.columns:
    try:
        # Contar cuántas filas tienen hormone_profile
        has_hormone = df_all['hormone_profile'].notna().sum()
        print(f" {has_hormone} filas tienen hormone_profile")

        # Intentar parsear JSON
        def safe_parse_hormone(x):
            if pd.isna(x):
                return {}
            if isinstance(x, dict):
                return x
            if isinstance(x, str):
                try:
                    return json.loads(x)
                except:
                    return {}
            return {}

        hormone_dicts = df_all['hormone_profile'].apply(safe_parse_hormone)

        # Extraer niveles hormonales (solo si no existen ya)
        for hormone in ['dopamine', 'cortisol', 'oxytocin', 'adrenaline', 'serotonin']:
            col_name = f'hormone_{hormone}'
            # Solo extraer si la columna no existe (evitar sobrescribir datos de Fase 3)
            if col_name not in df_all.columns:
                df_all[col_name] = hormone_dicts.apply(
                    lambda x: x.get(hormone, np.nan)
                )

        print("Niveles hormonales extraídos")
    except Exception as e:
        print(f"No se pudieron extraer niveles hormonales: {e}")

# 5. LIMPIAR Y VALIDAR
print("\n5. Limpiando datos...")

# Remover filas con datos faltantes críticos
initial_count = len(df_all)
df_all = df_all.dropna(subset=['profile_name', 'generated_text'])
final_count = len(df_all)

if initial_count > final_count:
    print(f"Removidas {initial_count - final_count} filas con datos faltantes críticos")

# Validar rangos de distinct_2 (solo donde existe)
if 'distinct_2' in df_all.columns:
    distinct_available = df_all['distinct_2'].notna()
    invalid_distinct = (df_all['distinct_2'] < 0) | (df_all['distinct_2'] > 1)
    invalid_distinct = invalid_distinct & distinct_available

    if invalid_distinct.any():
        print(f"   {invalid_distinct.sum()} valores de distinct_2 fuera de rango [0,1]")
        df_all = df_all[~invalid_distinct]

# 6. ESTADÍSTICAS DESCRIPTIVAS
print("\n6. Estadísticas del dataset consolidado:")
print("-" * 80)

print(f"\nTotal generaciones: {len(df_all)}")

print("\nExperimentos:")
for exp in df_all['experiment'].value_counts().items():
    print(f"   {exp[0]:25s}: {exp[1]:5d} ({exp[1]/len(df_all)*100:5.1f}%)")

print("\n Perfiles hormonales (top 15):")
for profile in df_all['profile_name'].value_counts().head(15).items():
    print(f"   {profile[0]:25s}: {profile[1]:5d}")

if 'prompt_category' in df_all.columns:
    print("\n Categorías de prompts:")
    for cat in df_all['prompt_category'].value_counts().items():
        print(f"   {cat[0]:20s}: {cat[1]:5d}")

# Estadísticas de Fase 3
if 'is_dynamic' in df_all.columns:
    print("\n Sistema Dinámico (Fase 3):")
    dynamic_count = (df_all['is_dynamic'] == True).sum() if df_all['is_dynamic'].notna().any() else 0
    static_count = (df_all['is_dynamic'] == False).sum() if df_all['is_dynamic'].notna().any() else 0
    print(f"Dinámico: {dynamic_count}")
    print(f"Estático: {static_count}")

    if 'total_hormone_change' in df_all.columns:
        avg_change = df_all[df_all['is_dynamic'] == True]['total_hormone_change'].mean()
        if not pd.isna(avg_change):
            print(f"Cambio hormonal promedio: {avg_change:.4f}")

# 7. GUARDAR
print("\n7. Guardando dataset consolidado...")

# CSV principal
output_csv = OUTPUT_DIR / "all_experiments_consolidated.csv"
df_all.to_csv(output_csv, index=False)
print(f"CSV: {output_csv}")
print(f"Tamaño: {output_csv.stat().st_size / 1024**2:.2f} MB")

# Pickle para preservar tipos de datos
output_pkl = OUTPUT_DIR / "all_experiments_consolidated.pkl"
df_all.to_pickle(output_pkl)
print(f"Pickle: {output_pkl}")

# Metadatos
metadata = {
    'total_generations': len(df_all),
    'experiments': df_all['experiment'].value_counts().to_dict(),
    'profiles': df_all['profile_name'].value_counts().to_dict(),
    'columns': list(df_all.columns),
    'date_created': pd.Timestamp.now().isoformat(),
    'datasets_included': list(datasets.keys()),
}

# Añadir info de fase 3 si existe
if 'is_dynamic' in df_all.columns:
    metadata['phase3_info'] = {
        'dynamic_count': int((df_all['is_dynamic'] == True).sum()) if df_all['is_dynamic'].notna().any() else 0,
        'static_count': int((df_all['is_dynamic'] == False).sum()) if df_all['is_dynamic'].notna().any() else 0,
    }

output_meta = OUTPUT_DIR / "metadata.json"
with open(output_meta, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata: {output_meta}")

# Estadísticas descriptivas
desc_stats = df_all.describe(include='all').T
output_stats = OUTPUT_DIR / "descriptive_statistics.csv"
desc_stats.to_csv(output_stats)
print(f"Estadísticas: {output_stats}")

# Guardar también información de columnas
column_info = pd.DataFrame({
    'column': df_all.columns,
    'dtype': df_all.dtypes.values,
    'non_null_count': df_all.count().values,
    'null_count': df_all.isnull().sum().values,
    'coverage_pct': (df_all.count() / len(df_all) * 100).values
})
output_cols = OUTPUT_DIR / "column_info.csv"
column_info.to_csv(output_cols, index=False)
print(f"Info columnas: {output_cols}")

print("\n" + "="*80)
print(" CONSOLIDACIÓN COMPLETADA")
print("="*80)
print("\n Dataset final:")
print(f"Filas: {len(df_all):,}")
print(f"Columnas: {len(df_all.columns)}")
print(f"Memoria: {df_all.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"Experimentos: {len(df_all['experiment'].unique())}")
print(f"Perfiles únicos: {len(df_all['profile_name'].unique())}")

if 'prompt_category' in df_all.columns:
    print(f"Categorías: {df_all['prompt_category'].nunique()}")

print(f"\n Archivos guardados en: {OUTPUT_DIR}")
