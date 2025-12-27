"""
Modulación rica usando similitud semántica con Sentence-BERT
en lugar de listas simples de tokens.
"""

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SemanticCategory:
    """
    Define una categoría semántica con ejemplos de referencia.

    Attributes:
        name: Nombre de la categoría
        examples: Frases de ejemplo que definen la categoría
        embedding: Embedding promedio (calculado automáticamente)
    """
    name: str
    examples: List[str]
    embedding: Optional[torch.Tensor] = None


class SemanticBiasManager:
    """
    Gestiona sesgos semánticos usando embeddings de Sentence-BERT.

    En lugar de una lista fija de tokens, calcula similitud semántica
    entre cada token del vocabulario y categorías objetivo.

    Args:
        tokenizer: Tokenizer del modelo
        model_name: Modelo de Sentence-BERT (default: "all-MiniLM-L6-v2")
        device: Dispositivo para embeddings

    Ejemplo:
        >>> manager = SemanticBiasManager(tokenizer)
        >>> bias = manager.compute_semantic_bias(50257, "empathy", strength=1.0)
        >>> print(bias.shape)
        torch.Size([50257])
    """

    def __init__(
        self,
        tokenizer,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        self.tokenizer = tokenizer
        self.device = device

        print(f" Cargando modelo de embeddings: {model_name}...")
        self.sbert = SentenceTransformer(model_name)
        self.sbert.to(device)
        self.sbert.eval()

        # Definir categorías semánticas predefinidas
        self.categories = self._build_default_categories()

        # Pre-computar embeddings de categorías
        self._precompute_category_embeddings()

        # Cache de embeddings de tokens (para eficiencia)
        self.token_embedding_cache = {}

        print(f" SemanticBiasManager inicializado")
        print(f"   Categorías disponibles: {list(self.categories.keys())}")

    def _build_default_categories(self) -> Dict[str, SemanticCategory]:
        """Construye categorías semánticas predefinidas"""

        categories = {
            'empathy': SemanticCategory(
                name='empathy',
                examples=[
                    "I understand how you feel and I'm here for you",
                    "That must be really difficult for you",
                    "I can sense your pain and want to help",
                    "You're not alone in this, I care about you",
                    "It's completely valid to feel that way"
                ]
            ),

            'creativity': SemanticCategory(
                name='creativity',
                examples=[
                    "imagine a fantastical world full of wonder",
                    "let's think outside the box with innovative ideas",
                    "picture something original and artistic",
                    "envision creative possibilities and unique concepts"
                ]
            ),

            'factual': SemanticCategory(
                name='factual',
                examples=[
                    "according to scientific research and evidence",
                    "the factual data shows and proves",
                    "objectively speaking based on empirical studies",
                    "documented findings demonstrate clearly",
                    "verified information indicates precisely"
                ]
            ),

            'caution': SemanticCategory(
                name='caution',
                examples=[
                    "however, we should carefully consider potential risks",
                    "it's important to proceed thoughtfully and cautiously",
                    "perhaps we might want to think about this more",
                    "potentially there could be concerns to address",
                    "careful consideration suggests we evaluate thoroughly"
                ]
            ),

            'enthusiasm': SemanticCategory(
                name='enthusiasm',
                examples=[
                    "this is absolutely amazing and fantastic",
                    "I'm so excited and thrilled about this",
                    "wonderful, incredible, and truly extraordinary",
                    "brilliantly spectacular and remarkably awesome",
                    "enthusiastically positive and energetically great"
                ]
            )
        }

        return categories

    def _precompute_category_embeddings(self):
        """Pre-computa embeddings de todas las categorías"""
        print("   Computing category embeddings...")

        with torch.no_grad():
            for category_name, category in self.categories.items():
                # Codificar ejemplos
                embeddings = self.sbert.encode(
                    category.examples,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )

                # Promedio de embeddings
                category.embedding = embeddings.mean(dim=0)

    def add_custom_category(
        self,
        name: str,
        examples: List[str]
    ):
        """
        Añade una categoría semántica personalizada.

        Args:
            name: Nombre de la categoría
            examples: Lista de frases de ejemplo (mínimo 3)

        Example:
            >>> manager.add_custom_category(
            ...     "technical",
            ...     ["algorithm complexity analysis",
            ...      "data structure implementation",
            ...      "computational efficiency optimization"]
            ... )
        """
        if len(examples) < 3:
            raise ValueError("Se necesitan al menos 3 ejemplos")

        category = SemanticCategory(name=name, examples=examples)

        # Computar embedding

        with torch.no_grad():
            embeddings = self.sbert.encode(
                examples,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            category.embedding = embeddings.mean(dim=0)

        self.categories[name] = category
        print(f" Categoría '{name}' añadida")

    def get_token_embedding(self, token_id: int) -> torch.Tensor:
        """
        Obtiene embedding de un token (con cache).

        Args:
            token_id: ID del token

        Returns:
            Embedding del token normalizado
        """
        if token_id in self.token_embedding_cache:
            return self.token_embedding_cache[token_id]

        # Decodificar token
        token_text = self.tokenizer.decode([token_id])

        # Obtener embedding
        with torch.no_grad():
            embedding = self.sbert.encode(
                token_text,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )

        self.token_embedding_cache[token_id] = embedding
        return embedding

    def compute_semantic_bias(
        self,
        vocab_size: int,
        category: str,
        strength: float = 1.0,
        sample_size: int = 1000
    ) -> torch.Tensor:
        """
        Computa sesgo semántico para todo el vocabulario.

        Por razones de eficiencia, muestrea el vocabulario y asigna
        sesgo solo a los tokens más similares.

        Args:
            vocab_size: Tamaño del vocabulario
            category: Nombre de la categoría semántica
            strength: Multiplicador del sesgo [0, inf]
            sample_size: Número de tokens a muestrear (default: 1000)

        Returns:
            Tensor de sesgos [vocab_size]

        Example:
            >>> bias = manager.compute_semantic_bias(50257, "empathy", 1.5)
        """
        if category not in self.categories:
            raise ValueError(f"Categoría '{category}' no encontrada. "
                           f"Disponibles: {list(self.categories.keys())}")

        # Embedding objetivo
        target_embedding = self.categories[category].embedding.to(self.device)

        # Inicializar tensor de sesgos
        bias = torch.zeros(vocab_size, device=self.device)

        # Muestrear vocabulario (tokens más comunes + aleatorios)
        # Tokens comunes (0-999): más probables de aparecer
        common_ids = list(range(min(1000, vocab_size)))

        # Tokens aleatorios del resto del vocabulario
        if vocab_size > 1000:
            random_ids = torch.randperm(vocab_size - 1000)[:sample_size - 1000].tolist()
            random_ids = [i + 1000 for i in random_ids]
            sample_ids = common_ids + random_ids
        else:
            sample_ids = common_ids

        # Batch processing para eficiencia
        batch_size = 100

        for i in range(0, len(sample_ids), batch_size):
            batch_ids = sample_ids[i:i+batch_size]

            # Decodificar tokens
            batch_texts = [self.tokenizer.decode([tid]) for tid in batch_ids]

            # Obtener embeddings
            with torch.no_grad():
                batch_embeddings = self.sbert.encode(
                    batch_texts,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=batch_size
                ).to(self.device)

            # Calcular similitud coseno
            similarities = torch.nn.functional.cosine_similarity(
                batch_embeddings,
                target_embedding.unsqueeze(0),
                dim=1
            )

            # Asignar sesgos (solo valores positivos)
            similarities = torch.clamp(similarities, min=0.0)

            for j, token_id in enumerate(batch_ids):
                bias[token_id] = similarities[j] * strength

        return bias

    def compare_categories(
        self,
        text: str
    ) -> Dict[str, float]:
        """
        Calcula similitud de un texto con todas las categorías.

        Útil para análisis de qué categoría se activa más.

        Args:
            text: Texto a analizar

        Returns:
            Diccionario {categoria: similitud}

        Example:
            >>> scores = manager.compare_categories("I understand your feelings")
            >>> print(scores)
            {'empathy': 0.85, 'creativity': 0.23, ...}
        """
        # Embedding del texto
        with torch.no_grad():
            text_embedding = self.sbert.encode(
                text,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
            ).to(self.device)

        # Calcular similitud con cada categoría
        similarities = {}
        for name, category in self.categories.items():
            sim = torch.nn.functional.cosine_similarity(
                text_embedding.unsqueeze(0),
                category.embedding.unsqueeze(0),
                dim=1
            )
            similarities[name] = float(sim)

        return similarities


class SemanticLogitsProcessor:
    """
    LogitsProcessor que usa sesgos semánticos en lugar de listas de tokens.

    Compatible con la API de HormonalLogitsProcessor pero usa
    SemanticBiasManager internamente.

    Args:
        semantic_manager: Gestor de sesgos semánticos
        category: Categoría semántica a aplicar
        strength: Fuerza del sesgo
        threshold: Umbral mínimo para activar sesgo

    Example:
        >>> from transformers import LogitsProcessorList
        >>> processor = SemanticLogitsProcessor(manager, "empathy", strength=1.5)
        >>> processor_list = LogitsProcessorList([processor])
    """

    def __init__(
        self,
        semantic_manager: SemanticBiasManager,
        category: str,
        strength: float = 1.0,
        threshold: float = 0.0
    ):
        self.manager = semantic_manager
        self.category = category
        self.strength = strength
        self.threshold = threshold

        # Pre-computar bias si la categoría existe
        if category in self.manager.categories:
            print(f"Pre-computing semantic bias for '{category}'...")
            # Por ahora, dejamos que se compute on-demand
            self.bias_cache = None
        else:
            raise ValueError(f"Categoría '{category}' no encontrada")

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Aplica sesgo semántico a los logits.

        Args:
            input_ids: Tokens generados [batch_size, seq_len]
            scores: Logits [batch_size, vocab_size]

        Returns:
            Logits con sesgo semántico aplicado
        """
        batch_size, vocab_size = scores.shape

        # Computar o recuperar bias
        if self.bias_cache is None:
            self.bias_cache = self.manager.compute_semantic_bias(
                vocab_size=vocab_size,
                category=self.category,
                strength=self.strength
            )

        # Aplicar bias
        bias = self.bias_cache.to(scores.device)

        # Solo aplicar a valores sobre el umbral
        if self.threshold > 0:
            mask = bias >= self.threshold
            scores = scores + (bias * mask.float())
        else:
            scores = scores + bias

        return scores


def analyze_semantic_activation(
    generated_text: str,
    semantic_manager: SemanticBiasManager
) -> Dict[str, any]:
    """
    Analiza qué categorías semánticas se activaron en el texto generado.

    Args:
        generated_text: Texto a analizar
        semantic_manager: Gestor de sesgos semánticos

    Returns:
        Diccionario con análisis de activación

    Example:
        >>> analysis = analyze_semantic_activation(text, manager)
        >>> print(f"Categoría dominante: {analysis['dominant_category']}")
    """
    # Comparar con todas las categorías
    similarities = semantic_manager.compare_categories(generated_text)

    # Encontrar dominante
    dominant = max(similarities.items(), key=lambda x: x[1])

    # Estadísticas
    values = list(similarities.values())

    return {
        'similarities': similarities,
        'dominant_category': dominant[0],
        'dominant_score': dominant[1],
        'mean_similarity': np.mean(values),
        'std_similarity': np.std(values),
        'activation_spread': max(values) - min(values)
    }
