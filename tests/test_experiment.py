"""Tests para experiment.py"""
import pytest
from endocrine_llm import EndocrineModulatedLLM, HORMONE_PROFILES, ExperimentRunner


def test_experiment_runner_initialization():
    """Test que ExperimentRunner se inicializa correctamente"""
    model = EndocrineModulatedLLM("gpt2")
    runner = ExperimentRunner(model, compute_advanced_metrics=False)

    assert runner.model is not None
    assert runner.results == []


def test_run_small_experiment():
    """Test de experimento pequeño"""
    model = EndocrineModulatedLLM("gpt2")
    runner = ExperimentRunner(model, compute_advanced_metrics=False)

    prompts = ["Hello"]
    profiles = {"baseline": HORMONE_PROFILES["baseline"]}

    runner.run_experiment(
        prompts=prompts,
        profiles=profiles,
        num_generations=2,
        max_new_tokens=10
    )

    # Debe haber 2 resultados (1 prompt × 1 perfil × 2 generaciones)
    assert len(runner.results) == 2
    assert runner.results[0]['profile_name'] == 'baseline'


def test_get_dataframe():
    """Test de conversión a DataFrame"""
    model = EndocrineModulatedLLM("gpt2")
    runner = ExperimentRunner(model, compute_advanced_metrics=False)

    prompts = ["Test"]
    profiles = {"baseline": HORMONE_PROFILES["baseline"]}

    runner.run_experiment(prompts, profiles, num_generations=1, max_new_tokens=10)

    df = runner.get_dataframe()
    assert not df.empty
    assert 'profile_name' in df.columns
    assert 'generated_text' in df.columns


def test_save_and_load_results(tmp_path):
    """Test de guardado y carga de resultados"""
    model = EndocrineModulatedLLM("gpt2")
    runner = ExperimentRunner(model, compute_advanced_metrics=False)

    prompts = ["Test"]
    profiles = {"baseline": HORMONE_PROFILES["baseline"]}
    runner.run_experiment(prompts, profiles, num_generations=1, max_new_tokens=10)

    # Guardar
    json_path = tmp_path / "results.json"
    runner.save_results(str(json_path))
    assert json_path.exists()

    # Cargar en nuevo runner
    runner2 = ExperimentRunner(model)
    runner2.load_results(str(json_path))
    assert len(runner2.results) == len(runner.results)
