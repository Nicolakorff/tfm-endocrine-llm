"""
core.py - Sistema de Neuromodulación Endocrina Base

Este módulo implementa el sistema principal de modulación hormonal
para modelos de lenguaje usando LogitsProcessor de HuggingFace.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    NoRepeatNGramLogitsProcessor,
)


@dataclass
class HormoneProfile:
    """
    Perfil hormonal para modulación de generación de texto.

    Atributos:
        dopamine (float): Exploración/recompensa [0,1]
        cortisol (float): Cautela/precisión [0,1]
        oxytocin (float): Prosocialidad/empatía [0,1]
        adrenaline (float): Activación/intensidad [0,1]
        serotonin (float): Estabilidad/suavizado [0,1]

    Ejemplo:
        >>> profile = HormoneProfile(dopamine=0.8, cortisol=0.3)
        >>> print(profile)
        H[d=0.80, c=0.30, o=0.50, a=0.50, s=0.50]
    """
    dopamine: float = 0.5
    cortisol: float = 0.5
    oxytocin: float = 0.5
    adrenaline: float = 0.5
    serotonin: float = 0.5

    def __post_init__(self):
        """Valida que todos los valores estén en [0,1]"""
        for field in ['dopamine', 'cortisol', 'oxytocin', 'adrenaline', 'serotonin']:
            value = getattr(self, field)
            if not 0 <= value <= 1:
                raise ValueError(f"{field} debe estar en [0,1], recibido: {value}")

    def to_dict(self) -> Dict[str, float]:
        """Convierte el perfil a diccionario"""
        return asdict(self)

    def __repr__(self) -> str:
        return (f"H[d={self.dopamine:.2f}, c={self.cortisol:.2f}, "
                f"o={self.oxytocin:.2f}, a={self.adrenaline:.2f}, s={self.serotonin:.2f}]")


# Perfiles hormonales predefinidos
HORMONE_PROFILES = {
    "baseline": HormoneProfile(0.5, 0.5, 0.5, 0.5, 0.5),

    # Hormonas individuales
    "high_dopamine": HormoneProfile(0.9, 0.5, 0.5, 0.5, 0.5),
    "high_cortisol": HormoneProfile(0.5, 0.9, 0.5, 0.5, 0.5),
    "high_oxytocin": HormoneProfile(0.5, 0.5, 0.9, 0.5, 0.5),
    "high_adrenaline": HormoneProfile(0.5, 0.5, 0.5, 0.9, 0.5),
    "high_serotonin": HormoneProfile(0.5, 0.5, 0.5, 0.5, 0.9),

    # Perfiles combinados
    "euphoric": HormoneProfile(0.9, 0.2, 0.6, 0.5, 0.8),
    "stressed": HormoneProfile(0.3, 0.9, 0.4, 0.8, 0.3),
    "empathic": HormoneProfile(0.6, 0.4, 0.9, 0.4, 0.7),
    "cautious": HormoneProfile(0.3, 0.8, 0.5, 0.4, 0.6),
    "creative": HormoneProfile(0.9, 0.3, 0.5, 0.6, 0.5),
    "stable": HormoneProfile(0.5, 0.5, 0.5, 0.3, 0.9),
}


class HormonalLogitsProcessor(LogitsProcessor):
    """
    Procesador de logits que implementa modulación hormonal.

    Aplica tres transformaciones principales:
    1. Temperatura adaptativa según dopamina y cortisol
    2. Moldeado de distribución con serotonina y adrenalina
    3. Sesgo prosocial con oxitocina

    Args:
        hormone_profile: Configuración hormonal
        empathetic_tokens: Lista de IDs de tokens empáticos
        base_temperature: Temperatura base (default: 1.0)
        alpha: Peso de dopamina en temperatura (default: 0.3)
        beta: Peso de cortisol en temperatura (default: 0.3)
        gamma: Fuerza de Top-K dinámico (default: 0.2)
        delta: Fuerza de suavizado (default: 0.2)
        epsilon: Fuerza de sesgo prosocial (default: 0.5)
    """

    def __init__(
        self,
        hormone_profile: HormoneProfile,
        empathetic_tokens: List[int] = None,
        base_temperature: float = 1.0,
        alpha: float = 0.3,
        beta: float = 0.3,
        gamma: float = 0.2,
        delta: float = 0.2,
        epsilon: float = 0.5,
    ):
        self.H = hormone_profile
        self.empathetic_tokens = empathetic_tokens or []
        self.T_base = base_temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Modula logits según perfil hormonal.

        Args:
            input_ids: Tokens generados [batch_size, seq_len]
            scores: Logits del modelo [batch_size, vocab_size]

        Returns:
            Logits modulados
        """
        # 1. TEMPERATURA ADAPTATIVA
        T_adaptive = self.T_base + self.alpha * self.H.dopamine - self.beta * self.H.cortisol
        T_adaptive = max(0.1, min(2.0, float(T_adaptive)))
        scores = scores / T_adaptive

        # 2a. SEROTONINA: Suavizado
        if self.H.serotonin > 0:
            mean_scores = scores.mean(dim=-1, keepdim=True)
            smoothing_factor = max(0.0, min(1.0, self.delta * self.H.serotonin))
            scores = mean_scores + (scores - mean_scores) * (1.0 - smoothing_factor)

        # 2b. ADRENALINA: Top-K dinámico
        if self.H.adrenaline > 0.6:
            batch_size, vocab_size = scores.shape
            max_k = min(50, vocab_size)
            min_k = min(5, max_k)

            a_norm = (self.H.adrenaline - 0.6) / 0.4
            a_norm = max(0.0, min(1.0, a_norm))

            k = int(max_k - a_norm * (max_k - min_k))
            k = max(min_k, min(max_k, k))

            topk_scores, topk_indices = torch.topk(scores, k, dim=-1)
            mask = torch.full_like(scores, float("-inf"))
            mask.scatter_(1, topk_indices, topk_scores)
            scores = mask

        # 3. OXITOCINA: Sesgo prosocial
        if self.H.oxytocin > 0.6 and len(self.empathetic_tokens) > 0:
            for token_id in self.empathetic_tokens:
                if token_id < scores.shape[-1]:
                    scores[:, token_id] += self.epsilon * self.H.oxytocin

        return scores


class EndocrineModulatedLLM:
    """
    Sistema completo de neuromodulación endocrina para LLMs.

    Integra modelo de lenguaje base con modulación hormonal.

    Args:
        model_name: Nombre del modelo (default: "gpt2")
        device: Dispositivo ("cuda" o "cpu", auto-detecta si None)

    Ejemplo:
        >>> model = EndocrineModulatedLLM("gpt2")
        >>> texts = model.generate_with_hormones(
        ...     "Hello, I am",
        ...     HORMONE_PROFILES["empathic"],
        ...     max_new_tokens=30
        ... )
        >>> print(texts[0])
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = None
    ):
        print("Inicializando EndocrineModulatedLLM")
        print(f"Modelo: {model_name}")

        # Dispositivo
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"   Dispositivo: {self.device}")

        # Cargar modelo y tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Configurar pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id

        # Construir lexicón empático básico
        self.empathetic_tokens = self._build_empathetic_tokens()

        print("Sistema inicializado")
        print(f"Parámetros: {self.model.num_parameters() / 1e6:.1f}M")
        print(f"Tokens empáticos: {len(self.empathetic_tokens)}")

    def _build_empathetic_tokens(self) -> List[int]:
        """Construye lista de tokens empáticos"""
        words = [
            "care", "support", "help", "understand", "listen",
            "comfort", "safe", "together", "here", "feel",
            "sorry", "hope", "wish", "love", "kind"
        ]

        token_ids = set()
        for word in words:
            token_ids.update(self.tokenizer.encode(" " + word, add_special_tokens=False))
            token_ids.update(self.tokenizer.encode(word, add_special_tokens=False))

        return list(token_ids)

    def generate_with_hormones(
        self,
        prompt: str,
        hormone_profile: HormoneProfile,
        max_new_tokens: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> List[str]:
        """
        Genera texto con modulación hormonal.

        Args:
            prompt: Texto de entrada
            hormone_profile: Configuración hormonal
            max_new_tokens: Tokens máximos a generar
            num_return_sequences: Número de secuencias
            do_sample: Si usar muestreo o greedy
            top_k: Parámetro top-k
            top_p: Nucleus sampling

        Returns:
            Lista de textos generados
        """
        # Construir procesador hormonal
        hormonal_processor = HormonalLogitsProcessor(
            hormone_profile=hormone_profile,
            empathetic_tokens=self.empathetic_tokens
        )

        processors = [hormonal_processor]

        # Añadir NoRepeat si cortisol alto
        if hormone_profile.cortisol >= 0.6:
            processors.append(NoRepeatNGramLogitsProcessor(3))

        logits_processor = LogitsProcessorList(processors)

        # Preparar inputs
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generar
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            logits_processor=logits_processor,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decodificar
        return [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

    def generate_baseline(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        num_return_sequences: int = 1
    ) -> List[str]:
        """Generación baseline sin modulación"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
