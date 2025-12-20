"""Métricas de texto (stubs) para pruebas y desarrollo.

Provee `TextMetrics` y `AdvancedMetrics` con implementaciones simples
que no requieren dependencias pesadas.
"""
from typing import Dict


class TextMetrics:
    """Cálculos básicos de métricas de texto.

    Métodos mínimos usados por la librería y tests.
    """

    @staticmethod
    def length(text: str) -> int:
        return len(text)

    @staticmethod
    def token_count(text: str) -> int:
        # En ausencia de un tokenizer, separar por espacios
        return len(text.split())

    @staticmethod
    def summary(text: str) -> Dict[str, int]:
        return {"length": TextMetrics.length(text), "tokens": TextMetrics.token_count(text)}


class AdvancedMetrics(TextMetrics):
    """Extiende `TextMetrics` con métricas adicionales (stub).

    Mantener simple para evitar dependencias en tiempo de testing.
    """

    @staticmethod
    def lexical_diversity(text: str) -> float:
        tokens = text.split()
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    @staticmethod
    def all_metrics(text: str) -> Dict[str, object]:
        base = TextMetrics.summary(text)
        base.update({"lexical_diversity": AdvancedMetrics.lexical_diversity(text)})
        return base


__all__ = ["TextMetrics", "AdvancedMetrics"]
