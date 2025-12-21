"""Tests para metrics.py"""
import pytest
from endocrine_llm.metrics import TextMetrics, EmpathyMetrics


def test_distinct_n():
    """Test de Distinct-n"""
    text = "hello world hello world"

    # Distinct-1 = 2 únicos / 4 totales = 0.5
    distinct_1 = TextMetrics.compute_distinct_n(text, 1)
    assert 0.4 < distinct_1 < 0.6

    # Distinct-2 = 3 únicos / 3 totales = 1.0
    distinct_2 = TextMetrics.compute_distinct_n(text, 2)
    assert distinct_2 > 0.6


def test_sentiment():
    """Test de análisis de sentimiento"""
    positive_text = "I am very happy and excited!"
    negative_text = "I am sad and disappointed."

    pos_sentiment = TextMetrics.compute_sentiment(positive_text)
    neg_sentiment = TextMetrics.compute_sentiment(negative_text)

    assert pos_sentiment['polarity'] > 0
    assert neg_sentiment['polarity'] < 0


def test_repetition_rate():
    """Test de tasa de repetición"""
    repetitive = "hello hello hello hello"
    diverse = "the quick brown fox jumps"

    rep_rate_high = TextMetrics.compute_repetition_rate(repetitive, 2)
    rep_rate_low = TextMetrics.compute_repetition_rate(diverse, 2)

    assert rep_rate_high > rep_rate_low


def test_compute_all():
    """Test que compute_all retorna todas las métricas"""
    text = "This is a test sentence with some words."
    metrics = TextMetrics.compute_all(text)

    required_keys = [
        'length', 'distinct_1', 'distinct_2', 'distinct_3',
        'repetition_rate', 'sentiment_polarity', 'sentiment_subjectivity'
    ]

    for key in required_keys:
        assert key in metrics
        assert isinstance(metrics[key], (int, float))


def test_empathy_metrics():
    """Test de métricas de empatía"""
    empathy_metrics = EmpathyMetrics(use_classifier=False)

    empathic_text = "I understand how you feel and I'm here to support you."
    neutral_text = "The weather is nice today."

    score_empathic = empathy_metrics.compute_empathy_score(empathic_text)
    score_neutral = empathy_metrics.compute_empathy_score(neutral_text)

    assert score_empathic > score_neutral
    assert 0 <= score_empathic <= 1
