# Verificación rápida
import pandas as pd

df = pd.read_csv("data/prompts/prompts_dataset.csv")
print(df.head())
print(f"\nTotal prompts: {len(df)}")
print(f"Categorías: {df['category'].unique()}")
