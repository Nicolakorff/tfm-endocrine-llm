"""
metrics.py - Sistema de Métricas de Evaluación

Implementa métricas automáticas para evaluar calidad de texto generado.
Incluye métricas básicas (diversidad) y avanzadas (perplexity, ROUGE).
"""

import torch
import torch.nn.functional as F
import numpy as np
import nltk
from textblob import TextBlob
from collections import Counter
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Descargar recursos NLTK si es necesario
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class TextMetrics:
    """
    Calcula métricas básicas de texto.

    Métricas incluidas:
    - Longitud (número de tokens)
    - Diversidad léxica (Distinct-1, Distinct-2)
    - Sentimiento (polaridad y subjetividad)
    - Tasa de repetición

    Ejemplo:
        >>> metrics = TextMetrics()
        >>> result = metrics.compute_all("I am happy and excited!")
        >>> print(result['sentiment_polarity'])  # Valor positivo
        0.65
    """

    @staticmethod
    def compute_distinct_n(text: str, n: int) -> float:
        """
        Calcula Distinct-n (diversidad de n-gramas).

        Args:
            text: Texto a analizar
            n: Tamaño del n-grama (1=unigrama, 2=bigrama, etc.)

        Returns:
            Ratio de n-gramas únicos [0,1]
        """
        tokens = text.lower().split()

        if len(tokens) < n:
            return 0.0

        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

        if len(ngrams) == 0:
            return 0.0

        return len(set(ngrams)) / len(ngrams)

    @staticmethod
    def compute_sentiment(text: str) -> Dict[str, float]:
        """
        Calcula sentimiento usando TextBlob.

        Args:
            text: Texto a analizar

        Returns:
            Diccionario con 'polarity' [-1,1] y 'subjectivity' [0,1]
        """
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception:
            return {'polarity': 0.0, 'subjectivity': 0.0}

    @staticmethod
    def compute_repetition_rate(text: str, n: int = 3) -> float:
        """
        Calcula tasa de repetición de n-gramas.

        Args:
            text: Texto a analizar
            n: Tamaño del n-grama

        Returns:
            Tasa de repetición [0,1] (0=sin repetición, 1=todo repetido)
        """
        tokens = text.lower().split()

        if len(tokens) < n:
            return 0.0

        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

        if len(ngrams) == 0:
            return 0.0

        unique_ngrams = len(set(ngrams))
        total_ngrams = len(ngrams)

        return 1 - (unique_ngrams / total_ngrams)

    @staticmethod
    def compute_all(text: str) -> Dict[str, float]:
        """
        Calcula todas las métricas básicas.

        Args:
            text: Texto a analizar

        Returns:
            Diccionario con todas las métricas
        """
        tokens = text.split()
        sentiment = TextMetrics.compute_sentiment(text)

        return {
            'length': len(tokens),
            'distinct_1': TextMetrics.compute_distinct_n(text, 1),
            'distinct_2': TextMetrics.compute_distinct_n(text, 2),
            'distinct_3': TextMetrics.compute_distinct_n(text, 3),
            'repetition_rate': TextMetrics.compute_repetition_rate(text, 3),
            'sentiment_polarity': sentiment['polarity'],
            'sentiment_subjectivity': sentiment['subjectivity']
        }


class AdvancedMetrics:
    """
    Calcula métricas avanzadas que requieren modelo base.

    Métricas incluidas:
    - Perplexity (usando modelo base)
    - ROUGE-L (coherencia con prompt)
    - Entropía promedio

    Args:
        model: Modelo de lenguaje base
        tokenizer: Tokenizer del modelo
        device: Dispositivo ('cuda' o 'cpu')

    Ejemplo:
        >>> from endocrine_llm import EndocrineModulatedLLM
        >>> model = EndocrineModulatedLLM("gpt2")
        >>> metrics = AdvancedMetrics(model.model, model.tokenizer, model.device)
        >>> result = metrics.compute_perplexity("This is a test.")
        >>> print(result)
        15.3
    """

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Cargar ROUGE scorer
        try:
            from rouge_score import rouge_scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        except ImportError:
            print(" rouge-score no instalado. ROUGE-L no estará disponible.")
            self.rouge_scorer = None

    def compute_perplexity(self, text: str) -> float:
        """
        Calcula perplexity del texto usando el modelo base.

        Perplexity mide qué tan "sorprendido" está el modelo del texto.
        Valores más bajos = texto más probable según el modelo.

        Args:
            text: Texto a evaluar

        Returns:
            Perplexity (típicamente entre 10-200 para texto coherente)
        """
        try:
            enc = self.tokenizer(text, return_tensors="pt", truncation=True)
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                outputs = self.model(**enc, labels=enc["input_ids"])
                loss = outputs.loss

            return float(torch.exp(loss))
        except Exception as e:
            print(f" Error calculando perplexity: {e}")
            return float('inf')

    def compute_rouge_l(self, reference: str, hypothesis: str) -> float:
        """
        Calcula ROUGE-L entre referencia e hipótesis.

        ROUGE-L mide la subsecuencia común más larga.
        Útil para evaluar coherencia con el prompt original.

        Args:
            reference: Texto de referencia (típicamente el prompt)
            hypothesis: Texto generado

        Returns:
            F-measure de ROUGE-L [0,1]
        """
        if self.rouge_scorer is None:
            return 0.0

        try:
            score = self.rouge_scorer.score(reference, hypothesis)
            return score['rougeL'].fmeasure
        except Exception as e:
            print(f" Error calculando ROUGE-L: {e}")
            return 0.0

    def compute_entropy(self, text: str) -> float:
        """
        Calcula entropía promedio de la distribución de tokens.

        Args:
            text: Texto a evaluar

        Returns:
            Entropía promedio (típicamente 2-8)
        """
        try:
            enc = self.tokenizer(text, return_tensors="pt", truncation=True)
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                outputs = self.model(**enc)
                logits = outputs.logits

            # Calcular entropía promedio
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)

            return float(entropy.mean())
        except Exception as e:
            print(f" Error calculando entropía: {e}")
            return 0.0

    def compute_all(self, prompt: str, generated_text: str) -> Dict[str, float]:
        """
        Calcula todas las métricas avanzadas.

        Args:
            prompt: Prompt original
            generated_text: Texto generado completo

        Returns:
            Diccionario con todas las métricas
        """
        # Métricas básicas
        basic = TextMetrics.compute_all(generated_text)

        # Métricas avanzadas
        advanced = {
            'perplexity': self.compute_perplexity(generated_text),
            'rouge_l': self.compute_rouge_l(prompt, generated_text),
            'entropy': self.compute_entropy(generated_text)
        }

        # Combinar
        return {**basic, **advanced}


class EmpathyMetrics:
    """
    Métricas específicas para evaluar empatía en texto.

    Usa dos enfoques:
    1. Conteo de palabras empáticas
    2. (Opcional) Clasificador RoBERTa si está disponible

    Ejemplo:
        >>> metrics = EmpathyMetrics()
        >>> score = metrics.compute_empathy_score("I understand your feelings.")
        >>> print(score)
        0.45
    """

    def __init__(self, use_classifier: bool = False):
        """
        Args:
            use_classifier: Si True, intenta cargar RoBERTa classifier
        """
        # Palabras empáticas de referencia
        self.empathy_words = [
            'understand', 'feel', 'care', 'support', 'help',
            'listen', 'here', 'together', 'comfort', 'safe',
            'sorry', 'empathy', 'compassion', 'sympathy'
        ]

        # Clasificador opcional
        self.classifier = None
        if use_classifier:
            try:
                from transformers import pipeline
                self.classifier = pipeline(
                    "text-classification",
                    model="cardiffnlp/twitter-roberta-base-emotion",
                    return_all_scores=True
                )
                print(" Clasificador de empatía cargado")
            except Exception as e:
                print(f" No se pudo cargar clasificador: {e}")

    def compute_empathy_score(self, text: str) -> float:
        """
        Calcula score de empatía basado en palabras clave.

        Args:
            text: Texto a evaluar

        Returns:
            Score de empatía [0,1]
        """
        text_lower = text.lower()
        words = text_lower.split()

        if len(words) == 0:
            return 0.0

        # Contar palabras empáticas
        empathy_count = sum(1 for word in self.empathy_words if word in text_lower)

        # Normalizar por longitud
        return min(1.0, empathy_count / max(len(words) * 0.1, 1.0))

    def compute_empathy_classifier(self, text: str) -> float:
        """
        Calcula empatía usando clasificador RoBERTa.

        Args:
            text: Texto a evaluar

        Returns:
            Score de empatía [0,1]
        """
        if self.classifier is None:
            return 0.0

        try:
            empathy_labels = {"joy", "trust", "optimism", "love"}
            scores = self.classifier(text[:512])[0]
            emp_score = sum(s["score"] for s in scores
                           if s["label"].lower() in empathy_labels)
            return float(emp_score)
        except Exception as e:
            print(f" Error en clasificador: {e}")
            return 0.0

    def compute_all(self, text: str) -> Dict[str, float]:
        """Calcula todas las métricas de empatía"""
        result = {
            'empathy_word_score': self.compute_empathy_score(text)
        }

        if self.classifier is not None:
            result['empathy_classifier_score'] = self.compute_empathy_classifier(text)

        return result
