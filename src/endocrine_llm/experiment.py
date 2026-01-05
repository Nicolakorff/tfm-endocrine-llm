"""
Ejecuta experimentos sistemáticos con múltiples perfiles hormonales
y prompts, guardando todos los resultados para análisis posterior.
"""

import json
import pandas as pd
from typing import List, Dict
from tqdm.auto import tqdm
from pathlib import Path

from .core import EndocrineModulatedLLM, HormoneProfile
from .metrics import TextMetrics, AdvancedMetrics


class ExperimentRunner:
    """
    Ejecuta experimentos sistemáticos con diferentes configuraciones.

    Funcionalidad:
    - Genera texto con múltiples perfiles hormonales
    - Calcula métricas automáticas
    - Guarda resultados en JSON y CSV
    - Genera estadísticas resumidas

    Args:
        model: Sistema EndocrineModulatedLLM
        compute_advanced_metrics: Si calcular métricas avanzadas (más lento)

    Ejemplo:
        >>> from endocrine_llm import EndocrineModulatedLLM, HORMONE_PROFILES
        >>> model = EndocrineModulatedLLM("gpt2")
        >>> runner = ExperimentRunner(model)
        >>> runner.run_experiment(
        ...     prompts=["Hello, I am"],
        ...     profiles={"baseline": HORMONE_PROFILES["baseline"]},
        ...     num_generations=3
        ... )
        >>> df = runner.get_dataframe()
    """

    def __init__(
        self,
        model: EndocrineModulatedLLM,
        compute_advanced_metrics: bool = True
    ):
        self.model = model
        self.compute_advanced = compute_advanced_metrics
        self.results = []

        # Inicializar calculadores de métricas
        if compute_advanced_metrics:
            self.advanced_metrics = AdvancedMetrics(
                model.model,
                model.tokenizer,
                model.device
            )
        else:
            self.advanced_metrics = None

    def run_experiment(
        self,
        prompts: List[str],
        profiles: Dict[str, HormoneProfile],
        num_generations: int = 3,
        max_new_tokens: int = 50,
        save_every: int = 50
    ):
        """
        Ejecuta experimento completo.

        Args:
            prompts: Lista de prompts a probar
            profiles: Diccionario {nombre: HormoneProfile}
            num_generations: Generaciones por combinación prompt-perfil
            max_new_tokens: Longitud máxima de generación
            save_every: Guardar resultados cada N generaciones (0 = no guardar)

        Example:
            >>> prompts = ["I'm feeling anxious.", "Write a story."]
            >>> profiles = {
            ...     "baseline": HORMONE_PROFILES["baseline"],
            ...     "empathic": HORMONE_PROFILES["empathic"]
            ... }
            >>> runner.run_experiment(prompts, profiles, num_generations=5)
        """
        total = len(prompts) * len(profiles) * num_generations

        print("Ejecutando experimento:")
        print(f"Prompts: {len(prompts)}")
        print(f"Perfiles: {len(profiles)}")
        print(f"Generaciones/combinación: {num_generations}")
        print(f"Total generaciones: {total}")
        print(f"{'Métricas avanzadas activadas' if self.compute_advanced else 'Solo métricas básicas'}\n")

        generation_count = 0

        with tqdm(total=total, desc="Progreso") as pbar:
            for prompt_idx, prompt in enumerate(prompts):
                for profile_name, profile in profiles.items():
                    for gen_idx in range(num_generations):
                        # Generar texto
                        try:
                            texts = self.model.generate_with_hormones(
                                prompt=prompt,
                                hormone_profile=profile,
                                max_new_tokens=max_new_tokens,
                                num_return_sequences=1
                            )

                            generated_text = texts[0]

                            # Resultado base
                            result = {
                                'prompt_idx': prompt_idx,
                                'prompt': prompt,
                                'profile_name': profile_name,
                                'hormone_profile': profile.to_dict(),
                                'generation_idx': gen_idx,
                                'generated_text': generated_text,
                            }

                            # Calcular métricas básicas
                            basic_metrics = TextMetrics.compute_all(generated_text)
                            result.update(basic_metrics)

                            # Calcular métricas avanzadas
                            if self.compute_advanced and self.advanced_metrics:
                                advanced = {
                                    'perplexity': self.advanced_metrics.compute_perplexity(generated_text),
                                    'rouge_l': self.advanced_metrics.compute_rouge_l(prompt, generated_text),
                                    'entropy': self.advanced_metrics.compute_entropy(generated_text)
                                }
                                result.update(advanced)

                            self.results.append(result)
                            generation_count += 1

                            # Guardar periódicamente
                            if save_every > 0 and generation_count % save_every == 0:
                                self._save_checkpoint(f"checkpoint_{generation_count}.json")

                        except Exception as e:
                            print(f"\nError en generación: {e}")
                            print(f"Prompt: {prompt[:50]}...")
                            print(f"Perfil: {profile_name}")
                            continue

                        pbar.update(1)

        print(f"\n Experimento completado: {len(self.results)} resultados guardados")

    def run_dynamic_experiment(
        self,
        prompts_df: pd.DataFrame,
        num_generations: int = 2,
        max_new_tokens: int = 50,
        update_interval: int = 5,
        save_path: str = None
    ) -> pd.DataFrame:
        """
        Ejecuta experimento con perfiles dinámicos.

        Args:
            prompts_df: DataFrame con columnas 'prompt' y 'category'
            num_generations: Repeticiones por configuración
            max_new_tokens: Tokens máximos a generar
            update_interval: Cada cuántos tokens actualizar hormonas
            save_path: Ruta opcional para guardar resultados

        Returns:
            DataFrame con resultados del experimento dinámico

        Example:
            >>> prompts_df = pd.read_csv("prompts.csv")
            >>> df_results = runner.run_dynamic_experiment(
            ...     prompts_df,
            ...     num_generations=3,
            ...     save_path="phase3_dynamic_results.csv"
            ... )
        """
        print("\n" + "="*80)
        print(" FASE 3: SISTEMA HORMONAL DINÁMICO")
        print("="*80 + "\n")

        # Definir perfiles dinámicos
        dynamic_profiles = {
            'neutral_dynamic': HormoneProfile(0.5, 0.5, 0.5, 0.5, 0.5, dynamic=True, learning_rate=0.15),
            'creative_dynamic': HormoneProfile(0.9, 0.3, 0.5, 0.6, 0.5, dynamic=True, learning_rate=0.12),
            'empathic_dynamic': HormoneProfile(0.6, 0.4, 0.9, 0.4, 0.7, dynamic=True, learning_rate=0.12),
        }

        # Comparar con versiones estáticas equivalentes
        static_equivalents = {
            'neutral_static': HormoneProfile(0.5, 0.5, 0.5, 0.5, 0.5, dynamic=False),
            'creative_static': HormoneProfile(0.9, 0.3, 0.5, 0.6, 0.5, dynamic=False),
            'empathic_static': HormoneProfile(0.6, 0.4, 0.9, 0.4, 0.7, dynamic=False),
        }

        all_profiles = {**dynamic_profiles, **static_equivalents}

        # Usar subset del dataset (10-15 prompts por categoría)
        if 'category' in prompts_df.columns:
            sample_prompts = prompts_df.groupby('category').apply(
                lambda x: x.sample(min(10, len(x)), random_state=42)
            ).reset_index(drop=True)
        else:
            # Si no hay categorías, sample simple
            sample_prompts = prompts_df.sample(min(50, len(prompts_df)), random_state=42)

        print(f"Usando {len(sample_prompts)} prompts para experimento dinámico")
        print(f"Perfiles: {len(all_profiles)} (3 dinámicos + 3 estáticos)")
        print(f"Total generaciones: {len(sample_prompts) * len(all_profiles) * num_generations}\n")

        dynamic_results = []
        total_iterations = len(sample_prompts) * len(all_profiles) * num_generations

        with tqdm(total=total_iterations, desc="Progreso dinámico") as pbar:
            for _, prompt_row in sample_prompts.iterrows():
                prompt = prompt_row['prompt']
                category = prompt_row.get('category', 'general')

                for profile_name, profile in all_profiles.items():
                    for rep in range(num_generations):

                        try:
                            if profile.dynamic:
                                # Generación dinámica
                                result = self.model.generate_with_dynamic_hormones(
                                    prompt=prompt,
                                    initial_profile=profile.clone(),
                                    max_new_tokens=max_new_tokens,
                                    update_interval=update_interval,
                                    return_hormone_trajectory=True
                                )

                                generated_text = result['generated_text']
                                final_profile = result['final_hormone_profile']

                            else:
                                # Generación estática
                                texts = self.model.generate_with_hormones(
                                    prompt=prompt,
                                    hormone_profile=profile,
                                    max_new_tokens=max_new_tokens,
                                    num_return_sequences=1
                                )
                                generated_text = texts[0]
                                final_profile = profile.to_dict()

                            # Calcular métricas
                            metrics = TextMetrics.compute_all(generated_text)
                            
                            # Añadir métricas avanzadas si están habilitadas
                            if self.compute_advanced and self.advanced_metrics:
                                try:
                                    metrics['perplexity'] = self.advanced_metrics.compute_perplexity(generated_text)
                                except Exception as e:
                                    print(f"\n  Error calculando perplexity: {e}")
                                    metrics['perplexity'] = float('nan')

                            # Guardar resultado
                            dynamic_results.append({
                                'prompt': prompt,
                                'category': category,
                                'profile_name': profile_name,
                                'is_dynamic': profile.dynamic,
                                'learning_rate': profile.learning_rate if profile.dynamic else 0.0,
                                'repetition': rep,
                                'generated_text': generated_text,

                                # Perfil inicial
                                'init_dopamine': profile.dopamine,
                                'init_cortisol': profile.cortisol,
                                'init_oxytocin': profile.oxytocin,
                                'init_adrenaline': profile.adrenaline,
                                'init_serotonin': profile.serotonin,

                                # Perfil final
                                'final_dopamine': final_profile['dopamine'],
                                'final_cortisol': final_profile['cortisol'],
                                'final_oxytocin': final_profile['oxytocin'],
                                'final_adrenaline': final_profile['adrenaline'],
                                'final_serotonin': final_profile['serotonin'],

                                # Cambios
                                'delta_dopamine': final_profile['dopamine'] - profile.dopamine,
                                'delta_cortisol': final_profile['cortisol'] - profile.cortisol,
                                'delta_oxytocin': final_profile['oxytocin'] - profile.oxytocin,
                                'delta_adrenaline': final_profile['adrenaline'] - profile.adrenaline,
                                'delta_serotonin': final_profile['serotonin'] - profile.serotonin,

                                'total_hormone_change': sum([
                                    abs(final_profile['dopamine'] - profile.dopamine),
                                    abs(final_profile['cortisol'] - profile.cortisol),
                                    abs(final_profile['oxytocin'] - profile.oxytocin),
                                    abs(final_profile['adrenaline'] - profile.adrenaline),
                                    abs(final_profile['serotonin'] - profile.serotonin),
                                ]),

                                **metrics
                            })

                        except Exception as e:
                            print(f"\n Error en generación dinámica: {e}")
                            print(f"Prompt: {prompt[:50]}...")
                            print(f"Perfil: {profile_name}")
                            continue

                        pbar.update(1)

        # Convertir a DataFrame
        df_dynamic = pd.DataFrame(dynamic_results)

        # Guardar si se especifica ruta
        if save_path:
            df_dynamic.to_csv(save_path, index=False)
            print(f"\n Fase 3 completada: {save_path}")
            print(f"Total registros: {len(df_dynamic)}")

        return df_dynamic

    def _save_checkpoint(self, filename: str):
        """Guarda checkpoint intermedio"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error guardando checkpoint: {e}")

    def save_results(
        self,
        json_path: str = "results.json",
        csv_path: str = None
    ):
        """
        Guarda resultados en JSON y opcionalmente CSV.

        Args:
            json_path: Ruta para guardar JSON
            csv_path: Ruta opcional para guardar CSV

        Example:
            >>> runner.save_results("experiment_results.json", "experiment_results.csv")
        """
        # Guardar JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"Resultados guardados: {json_path}")

        # Guardar CSV
        if csv_path:
            df = self.get_dataframe()
            df.to_csv(csv_path, index=False)
            print(f"CSV guardado: {csv_path}")

    def load_results(self, json_path: str):
        """
        Carga resultados desde JSON.

        Args:
            json_path: Ruta del archivo JSON
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        print(f"Resultados cargados: {len(self.results)} registros")

    def get_dataframe(self) -> pd.DataFrame:
        """
        Convierte resultados a DataFrame de pandas.

        Returns:
            DataFrame con todos los resultados

        Example:
            >>> df = runner.get_dataframe()
            >>> print(df['profile_name'].value_counts())
        """
        if not self.results:
            return pd.DataFrame()

        # Aplanar hormone_profile si es dict
        flat_results = []
        for result in self.results:
            flat_result = result.copy()

            # Convertir hormone_profile dict a columnas individuales
            if 'hormone_profile' in flat_result and isinstance(flat_result['hormone_profile'], dict):
                for hormone, value in flat_result['hormone_profile'].items():
                    flat_result[f'hormone_{hormone}'] = value
                del flat_result['hormone_profile']

            flat_results.append(flat_result)

        return pd.DataFrame(flat_results)

    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Calcula estadísticas resumidas por perfil hormonal.

        Returns:
            DataFrame con media y desviación estándar por perfil

        Example:
            >>> summary = runner.get_summary_statistics()
            >>> print(summary)
        """
        df = self.get_dataframe()

        if df.empty:
            return pd.DataFrame()

        # Seleccionar columnas numéricas (métricas)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        metric_cols = [col for col in numeric_cols
                      if col not in ['prompt_idx', 'generation_idx']]

        if not metric_cols:
            return pd.DataFrame()

        # Agrupar por perfil y calcular estadísticas
        summary = df.groupby('profile_name')[metric_cols].agg(['mean', 'std']).round(3)

        return summary

    def compare_profiles(
        self,
        metric: str,
        profiles: List[str] = None
    ) -> pd.DataFrame:
        """
        Compara un métrica específica entre perfiles.

        Args:
            metric: Nombre de la métrica a comparar
            profiles: Lista de perfiles a incluir (None = todos)

        Returns:
            DataFrame con estadísticas por perfil

        Example:
            >>> comparison = runner.compare_profiles('distinct_2', ['baseline', 'empathic'])
            >>> print(comparison)
        """
        df = self.get_dataframe()

        if df.empty or metric not in df.columns:
            print(f"Métrica '{metric}' no encontrada")
            return pd.DataFrame()

        # Filtrar perfiles si se especifican
        if profiles:
            df = df[df['profile_name'].isin(profiles)]

        # Calcular estadísticas
        comparison = df.groupby('profile_name')[metric].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(3)

        return comparison.sort_values('mean', ascending=False)

    def export_examples(
        self,
        output_path: str,
        profile_name: str = None,
        num_examples: int = 5
    ):
        """
        Exporta ejemplos de texto generado a archivo.

        Args:
            output_path: Ruta del archivo de salida
            profile_name: Perfil específico (None = todos)
            num_examples: Número de ejemplos por perfil

        Example:
            >>> runner.export_examples("examples.txt", "empathic", 10)
        """
        df = self.get_dataframe()

        if df.empty:
            print(" No hay resultados para exportar")
            return

        # Filtrar por perfil si se especifica
        if profile_name:
            df = df[df['profile_name'] == profile_name]

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("EJEMPLOS DE TEXTO GENERADO\n")
            f.write("="*80 + "\n\n")

            for profile in df['profile_name'].unique():
                profile_df = df[df['profile_name'] == profile]
                samples = profile_df.sample(min(num_examples, len(profile_df)))

                f.write(f"\n{'='*80}\n")
                f.write(f"PERFIL: {profile.upper()}\n")
                f.write(f"{'='*80}\n\n")

                for idx, row in samples.iterrows():
                    f.write(f"Prompt: {row['prompt']}\n")
                    f.write(f"{'-'*80}\n")
                    f.write(f"{row['generated_text']}\n")
                    f.write(f"{'-'*80}\n")
                    f.write(f"Diversidad: {row.get('distinct_2', 0):.3f} | ")
                    f.write(f"Polaridad: {row.get('sentiment_polarity', 0):.3f}\n\n")

        print(f"Ejemplos exportados: {output_path}")
