"""Tests para semantic.py"""
import pytest
import torch
from endocrine_llm.semantic import (
    SemanticBiasManager,
    SemanticLogitsProcessor,
    analyze_semantic_activation
)
from endocrine_llm import EndocrineModulatedLLM


@pytest.fixture
def tokenizer():
    """Fixture de tokenizer"""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("gpt2")


def test_semantic_manager_initialization(tokenizer):
    """Test que SemanticBiasManager se inicializa"""
    manager = SemanticBiasManager(tokenizer, device="cpu")

    assert manager.sbert is not None
    assert len(manager.categories) > 0
    assert 'empathy' in manager.categories
    assert 'creativity' in manager.categories


def test_add_custom_category(tokenizer):
    """Test de añadir categoría personalizada"""
    manager = SemanticBiasManager(tokenizer, device="cpu")

    manager.add_custom_category(
        "technical",
        [
            "algorithm complexity analysis",
            "data structure implementation",
            "computational optimization"
        ]
    )

    assert 'technical' in manager.categories


def test_compute_semantic_bias(tokenizer):
    """Test de computación de sesgo semántico"""
    manager = SemanticBiasManager(tokenizer, device="cpu")

    vocab_size = 1000  # Vocabulario reducido para test
    bias = manager.compute_semantic_bias(
        vocab_size=vocab_size,
        category="empathy",
        strength=1.0,
        sample_size=100
    )

    assert bias.shape == (vocab_size,)
    assert bias.dtype == torch.float32
    assert bias.min() >= 0  # Sesgos deben ser no negativos
    assert bias.max() > 0   # Al menos algunos tokens tienen sesgo


def test_compare_categories(tokenizer):
    """Test de comparación de categorías"""
    manager = SemanticBiasManager(tokenizer, device="cpu")

    empathic_text = "I understand how you feel and I'm here to support you"
    creative_text = "Imagine a fantastical world full of creative wonders"

    scores_empathic = manager.compare_categories(empathic_text)
    scores_creative = manager.compare_categories(creative_text)

    # Texto empático debe puntuar más alto en empatía
    assert scores_empathic['empathy'] > scores_empathic['creativity']

    # Texto creativo debe puntuar más alto en creatividad
    assert scores_creative['creativity'] > scores_creative['empathy']


def test_semantic_logits_processor(tokenizer):
    """Test del procesador de logits semántico"""
    manager = SemanticBiasManager(tokenizer, device="cpu")

    processor = SemanticLogitsProcessor(
        semantic_manager=manager,
        category="empathy",
        strength=1.0
    )

    # Simular logits
    batch_size = 1
    vocab_size = 1000
    input_ids = torch.tensor([[1, 2, 3]])
    scores = torch.randn(batch_size, vocab_size)

    # Aplicar procesador
    modulated_scores = processor(input_ids, scores)

    assert modulated_scores.shape == scores.shape
    # Debe haber cambiado al menos algunos valores
    assert not torch.allclose(modulated_scores, scores)


def test_analyze_semantic_activation(tokenizer):
    """Test de análisis de activación semántica"""
    manager = SemanticBiasManager(tokenizer, device="cpu")

    text = "I really understand and care about your feelings"
    analysis = analyze_semantic_activation(text, manager)

    assert 'similarities' in analysis
    assert 'dominant_category' in analysis
    assert 'dominant_score' in analysis
    assert analysis['dominant_category'] == 'empathy'
