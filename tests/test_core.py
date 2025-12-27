"""Tests para core.py"""
import pytest
from endocrine_llm.core import HormoneProfile, EndocrineModulatedLLM, HORMONE_PROFILES


def test_hormone_profile_validation():
    """Test que valores hormonales están en [0,1]"""
    # Debe funcionar
    profile = HormoneProfile(dopamine=0.5)
    assert 0 <= profile.dopamine <= 1

    # Debe fallar
    with pytest.raises(ValueError):
        HormoneProfile(dopamine=1.5)


def test_hormone_profiles_exist():
    """Test que perfiles predefinidos existen"""
    assert "baseline" in HORMONE_PROFILES
    assert "empathic" in HORMONE_PROFILES
    assert len(HORMONE_PROFILES) >= 6


def test_model_initialization():
    """Test que modelo se inicializa correctamente"""
    model = EndocrineModulatedLLM("gpt2")
    assert model.model is not None
    assert model.tokenizer is not None
    assert len(model.empathetic_tokens) > 0


def test_generation_basic():
    """Test que generación funciona"""
    model = EndocrineModulatedLLM("gpt2")
    texts = model.generate_with_hormones(
        "Hello",
        HORMONE_PROFILES["baseline"],
        max_new_tokens=10
    )
    assert len(texts) == 1
    assert len(texts[0]) > len("Hello")
