"""
Anova analysis for phase1 and phase2 results
"""

# Anova phase1_results
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from pathlib import Path

# Obtener el directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).parent.parent

# Load the dataset
data = pd.read_csv(PROJECT_ROOT / 'data/results/phase1/phase1_results.csv')

# Definir métricas para analizar
metrics = ['distinct_2', 'sentiment_polarity', 'perplexity', 'repetition_rate']

# Realizar ANOVA para cada métrica
print("="*60)
print("ANOVA PHASE 1 - Análisis por perfil")
print("="*60)
anova_results_list = []
for metric in metrics:
    print(f"\n--- {metric.upper()} ---")
    model = ols(f'{metric} ~ C(profile_name)', data=data).fit()
    anova_result = anova_lm(model)
    anova_result['metric'] = metric
    anova_results_list.append(anova_result)
    print(anova_result)

# Guardar todos los resultados
combined_results = pd.concat(anova_results_list)
combined_results.to_csv(PROJECT_ROOT / 'data/results/phase1/anova_phase1_results.csv')

# Anova phase2_results
# Load the dataset
data2 = pd.read_csv(PROJECT_ROOT / 'data/results/phase2/phase2_results.csv')

print("\n" + "="*60)
print("ANOVA PHASE 2 - Análisis por perfil")
print("="*60)
anova_results_list2 = []
for metric in metrics:
    print(f"\n--- {metric.upper()} ---")
    model = ols(f'{metric} ~ C(profile_name)', data=data2).fit()
    anova_result = anova_lm(model)
    anova_result['metric'] = metric
    anova_results_list2.append(anova_result)
    print(anova_result)

# Guardar todos los resultados
combined_results2 = pd.concat(anova_results_list2)
combined_results2.to_csv(PROJECT_ROOT / 'data/results/phase2/anova_phase2_results.csv')
