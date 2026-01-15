"""
Funciones de utilidad para análisis experimental.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from scipy.stats import tukey_hsd
import torch
import random

# Configurar estilo de visualizaciones
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def print_section(title: str, char: str = "=", width: int = 80):
    """Imprime un título de sección formateado"""
    print("\n" + char * width)
    print(f" {title}")
    print(char * width + "\n")


def display_stats(df: pd.DataFrame, metric: str, groupby: str = "profile"):
    """
    Muestra estadísticas descriptivas de una métrica por perfil.

    Args:
        df: DataFrame con resultados
        metric: Nombre de la métrica a analizar
        groupby: Columna para agrupar (default: 'profile')
    """
    stats = df.groupby(groupby)[metric].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('median', 'median')
    ]).round(4)

    print(f"\nEstadísticas de {metric} por {groupby}:")
    print(stats)
    return stats


def plot_metric_comparison(df: pd.DataFrame,
                          metric: str,
                          groupby: str = "profile",
                          title: Optional[str] = None):
    """
    Crea un boxplot comparando una métrica entre perfiles.

    Args:
        df: DataFrame con resultados
        metric: Métrica a visualizar
        groupby: Columna para agrupar
        title: Título del gráfico (opcional)
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=groupby, y=metric, palette="Set2")
    plt.xticks(rotation=45, ha='right')
    plt.title(title or f"Distribución de {metric} por {groupby}")
    plt.ylabel(metric)
    plt.xlabel(groupby.capitalize())
    plt.tight_layout()
    plt.show()


def save_results_safely(df: pd.DataFrame, filepath: str):
    """
    Guarda resultados con manejo de errores.

    Args:
        df: DataFrame a guardar
        filepath: Ruta del archivo de salida
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"✓ Resultados guardados: {filepath}")
        print(f"  Total filas: {len(df):,}")
        return True
    except Exception as e:
        print(f"✗ Error guardando resultados: {e}")
        return False


def display_sample_generations(df: pd.DataFrame,
                               profile: str,
                               n: int = 3):
    """
    Muestra ejemplos de texto generado para un perfil.

    Args:
        df: DataFrame con resultados
        profile: Nombre del perfil a mostrar
        n: Número de ejemplos
    """
    samples = df[df['profile'] == profile].head(n)

    print(f"\nEjemplos de generación - Perfil: {profile}")
    print("=" * 80)

    for idx, row in samples.iterrows():
        print(f"\n[Ejemplo {idx + 1}]")
        print(f"Prompt: {row['prompt'][:80]}..." if len(row['prompt']) > 80 else f"Prompt: {row['prompt']}")
        print(f"\nGenerado: {row['generated_text']}")
        print("-" * 80)


def check_data_quality(df: pd.DataFrame):
    """
    Verifica la calidad de los datos experimentales.

    Args:
        df: DataFrame con resultados
    """
    print("\nVERIFICACIÓN DE CALIDAD DE DATOS")
    print("=" * 50)

    # Valores faltantes
    missing = df.isnull().sum()
    if missing.any():
        print("\nValores faltantes encontrados:")
        print(missing[missing > 0])
    else:
        print("\nNo hay valores faltantes")

    # Distribución de perfiles
    print("\nDistribución de perfiles:")
    print(df['profile'].value_counts())

    # Distribución de categorías (si existe)
    if 'category' in df.columns:
        print("\nDistribución de categorías:")
        print(df['category'].value_counts())

    print("\n✓ Verificación completada")


def get_phase_paths(phase_num: int, base_dir: str = "data"):
    """
    Retorna diccionario con paths estándar para una fase experimental.
    
    Args:
        phase_num: Número de fase (1-4)
        base_dir: Directorio base (default: "data")
    
    Returns:
        dict con keys: results_dir, csv, json, plots_dir
    
    Example:
        >>> paths = get_phase_paths(1)
        >>> OUTPUT_CSV = paths['csv']
    """
    base = Path(base_dir)
    results = base / "results"
    results.mkdir(parents=True, exist_ok=True)
    
    return {
        'results_dir': results,
        'csv': results / f"phase{phase_num}_results.csv",
        'json': results / f"phase{phase_num}_results.json",
        'plots_dir': results / f"phase{phase_num}_plots"
    }


def set_random_seeds(seed: int = 42):
    """
    Fija seeds para reproducibilidad completa.
    
    Args:
        seed: Semilla para generadores aleatorios
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"✓ Seeds fijadas a {seed} para reproducibilidad")


def run_statistical_pipeline(df: pd.DataFrame, 
                             metric_col: str, 
                             group_col: str = 'profile_name'):
    """
    Pipeline estadístico estándar: ANOVA + Tukey HSD.
    
    Args:
        df: DataFrame con resultados
        metric_col: Nombre de la columna con la métrica
        group_col: Nombre de la columna de agrupación
        
    Returns:
        dict con resultados de tests
        
    Example:
        >>> results = run_statistical_pipeline(df, 'distinct_2')
        >>> print(results['anova']['p_value'])
    """
    # Preparar grupos
    groups = [df[df[group_col] == g][metric_col].dropna().values 
              for g in df[group_col].unique()]
    
    # ANOVA
    f_stat, p_val = f_oneway(*groups)
    
    # Tukey HSD
    tukey_result = tukey_hsd(*groups)
    
    return {
        'anova': {
            'f_statistic': f_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        },
        'tukey': tukey_result,
        'group_names': df[group_col].unique().tolist(),
        'n_groups': len(groups),
        'group_sizes': [len(g) for g in groups]
    }


def experiment_health_check(results_df: pd.DataFrame, 
                           expected_total: int,
                           profile_col: str = 'profile_name'):
    """
    Verifica integridad de resultados experimentales.
    
    Args:
        results_df: DataFrame con resultados
        expected_total: Número esperado de generaciones totales
        profile_col: Nombre de la columna de perfiles
    """
    print("="*60)
    print("VERIFICACIÓN DE SALUD DEL EXPERIMENTO")
    print("="*60)
    
    # Check 1: Número total
    actual = len(results_df)
    print(f"\nGeneraciones totales: {actual}/{expected_total}")
    if actual != expected_total:
        print(f"Diferencia: {abs(actual - expected_total)}")
    
    # Check 2: Por perfil
    print(f"\nGeneraciones por perfil:")
    profile_counts = results_df[profile_col].value_counts()
    expected_per_profile = expected_total // len(profile_counts)
    
    for profile, count in profile_counts.items():
        status = "PASS" if count == expected_per_profile else "FAIL"
        print(f"{status} {profile}: {count}")
    
    # Check 3: Missing values
    print(f"\nValores faltantes:")
    missing = results_df.isnull().sum()
    if missing.sum() == 0:
        print("No hay valores faltantes")
    else:
        for col, n_missing in missing[missing > 0].items():
            pct = n_missing/len(results_df)*100
            print(f"{col}: {n_missing} ({pct:.1f}%)")
    
    # Check 4: Outliers extremos
    print(f"\n✓ Outliers extremos (>3σ):")
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    has_outliers = False
    
    for col in numeric_cols:
        if col in ['length', 'distinct_2', 'repetition_rate', 'perplexity']:
            mean = results_df[col].mean()
            std = results_df[col].std()
            outliers = results_df[
                (results_df[col] > mean + 3*std) | 
                (results_df[col] < mean - 3*std)
            ]
            if len(outliers) > 0:
                print(f"{col}: {len(outliers)} outliers")
                has_outliers = True
    
    if not has_outliers:
        print("No hay outliers extremos")
    
    print("\n" + "="*60)