"""
Módulo principal del sistema de modulación hormonal
para modelos de lenguaje usando LogitsProcessor de HuggingFace.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np
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
        dynamic (bool): Si el perfil se adapta durante la generación
        learning_rate (float): Tasa de aprendizaje para perfiles dinámicos

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
    dynamic: bool = False
    learning_rate: float = 0.1

    def __post_init__(self):
        """Valida que todos los valores estén en [0,1]"""
        for field in ['dopamine', 'cortisol', 'oxytocin', 'adrenaline', 'serotonin']:
            value = getattr(self, field)
            if not 0 <= value <= 1:
                raise ValueError(f"{field} debe estar en [0,1], recibido: {value}")

    def update(self, feedback: Dict[str, float]):
        """
        Actualiza niveles hormonales basándose en feedback de generación

        Args:
            feedback: Diccionario con señales como:
                - 'confidence': confianza del modelo (max prob) [0,1]
                - 'entropy': entropía de la distribución [0,10]
                - 'sentiment': sentimiento del texto [-1,1]
                - 'repetition': tasa de repetición [0,1]
        """
        if not self.dynamic:
            return

        lr = self.learning_rate

        # Dopamina: aumenta con alta confianza (recompensa)
        if 'confidence' in feedback:
            confidence = feedback['confidence']
            if confidence > 0.8:
                self.dopamine = min(1.0, self.dopamine + lr * 0.05)
            elif confidence < 0.3:
                self.dopamine = max(0.0, self.dopamine - lr * 0.03)

        # Cortisol: aumenta con alta entropía (incertidumbre/estrés)
        if 'entropy' in feedback:
            entropy = feedback['entropy']
            normalized_entropy = entropy / 10.0  # Normalizar
            if normalized_entropy > 0.7:
                self.cortisol = min(1.0, self.cortisol + lr * 0.1)
            elif normalized_entropy < 0.3:
                self.cortisol = max(0.0, self.cortisol - lr * 0.05)

        # Oxitocina: aumenta con sentimiento positivo
        if 'sentiment' in feedback:
            sentiment = feedback['sentiment']  # [-1, 1]
            if sentiment > 0.3:
                self.oxytocin = min(1.0, self.oxytocin + lr * 0.05)
            elif sentiment < -0.3:
                self.oxytocin = max(0.0, self.oxytocin - lr * 0.03)

        # Serotonina: disminuye con alta repetición
        if 'repetition' in feedback:
            repetition = feedback['repetition']
            if repetition > 0.5:
                self.serotonin = max(0.0, self.serotonin - lr * 0.08)
            elif repetition < 0.2:
                self.serotonin = min(1.0, self.serotonin + lr * 0.04)

        # Adrenalina: aumenta si cortisol alto y dopamina baja (estrés sin recompensa)
        if self.cortisol > 0.7 and self.dopamine < 0.4:
            self.adrenaline = min(1.0, self.adrenaline + lr * 0.06)
        elif self.cortisol < 0.4 and self.dopamine > 0.6:
            self.adrenaline = max(0.0, self.adrenaline - lr * 0.04)

    def to_dict(self) -> Dict[str, float]:
        """Convierte el perfil a diccionario"""
        return asdict(self)

    def clone(self) -> 'HormoneProfile':
        """Crea una copia del perfil hormonal"""
        return HormoneProfile(
            dopamine=self.dopamine,
            cortisol=self.cortisol,
            oxytocin=self.oxytocin,
            adrenaline=self.adrenaline,
            serotonin=self.serotonin,
            dynamic=self.dynamic,
            learning_rate=self.learning_rate
        )

    def __repr__(self) -> str:
        dyn = " [DYNAMIC]" if self.dynamic else ""
        return (f"H[d={self.dopamine:.2f}, c={self.cortisol:.2f}, "
                f"o={self.oxytocin:.2f}, a={self.adrenaline:.2f}, s={self.serotonin:.2f}]{dyn}")


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

    # Perfiles dinámicos
    "dynamic_neutral": HormoneProfile(0.5, 0.5, 0.5, 0.5, 0.5, dynamic=True, learning_rate=0.1),
    "dynamic_adaptive": HormoneProfile(0.5, 0.5, 0.5, 0.5, 0.5, dynamic=True, learning_rate=0.2),
    "dynamic_conservative": HormoneProfile(0.5, 0.5, 0.5, 0.5, 0.5, dynamic=True, learning_rate=0.05),
}


class HormonalLogitsProcessor(LogitsProcessor):
    """
    Procesador de logits que implementa modulación hormonal.
    Con tracking para feedback dinámico.

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

        # Tracking para feedback dinámico
        self.token_history = []
        self.confidence_history = deque(maxlen=10)
        self.entropy_history = deque(maxlen=10)

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

        # 4. TRACKING PARA FEEDBACK DINÁMICO
        if self.H.dynamic:
            self._track_generation_metrics(scores)

        return scores

    def _track_generation_metrics(self, scores: torch.FloatTensor):
        """Rastrea métricas para actualización hormonal"""
        # Calcular confianza (probabilidad del token más probable)
        probs = F.softmax(scores, dim=-1)
        max_prob = probs.max().item()
        self.confidence_history.append(max_prob)

        # Calcular entropía
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
        self.entropy_history.append(entropy)

    def get_feedback(self) -> Dict[str, float]:
        """
        Obtiene feedback acumulado para actualización hormonal

        Returns:
            Diccionario con métricas promedio
        """
        feedback = {}

        if self.confidence_history:
            feedback['confidence'] = float(np.mean(self.confidence_history))

        if self.entropy_history:
            feedback['entropy'] = float(np.mean(self.entropy_history))

        # Repetición simple (últimos 5 tokens)
        if len(self.token_history) >= 5:
            recent = self.token_history[-5:]
            unique_ratio = len(set(recent)) / len(recent)
            feedback['repetition'] = 1.0 - unique_ratio

        return feedback

    def add_token(self, token_id: int):
        """Añade token al historial"""
        self.token_history.append(token_id)


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

        print(f"Dispositivo: {self.device}")

        # Cargar modelo y tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )   
        self.model = self.model.to(self.device)
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

    def generate_with_dynamic_hormones(
        self,
        prompt: str,
        initial_profile: HormoneProfile,
        max_new_tokens: int = 50,
        update_interval: int = 5,
        return_hormone_trajectory: bool = False,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> Dict:
        """
        Genera texto con sistema hormonal dinámico

        Args:
            prompt: Texto de entrada
            initial_profile: Perfil hormonal inicial (debe tener dynamic=True)
            max_new_tokens: Tokens máximos a generar
            update_interval: Cada cuántos tokens actualizar hormonas
            return_hormone_trajectory: Si retornar evolución de hormonas
            do_sample: Si usar muestreo o greedy
            top_k: Parámetro top-k
            top_p: Nucleus sampling

        Returns:
            Diccionario con:
                - 'generated_text': texto generado completo
                - 'num_tokens': número de tokens generados
                - 'final_hormone_profile': perfil hormonal final
                - 'hormone_trajectory': (opcional) evolución de hormonas
        """
        if not initial_profile.dynamic:
            print(" Perfil no dinámico. Activando dynamic=True")
            initial_profile.dynamic = True

        # Preparar inputs
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        # Copiar perfil para no modificar el original
        current_profile = initial_profile.clone()

        # Tracking
        generated_tokens = []
        hormone_trajectory = []

        # Crear procesador hormonal
        processor = HormonalLogitsProcessor(
            hormone_profile=current_profile,
            empathetic_tokens=self.empathetic_tokens
        )

        # Añadir NoRepeat si cortisol alto
        processors = [processor]
        if current_profile.cortisol >= 0.6:
            processors.append(NoRepeatNGramLogitsProcessor(3))

        logits_processor = LogitsProcessorList(processors)

        # Generación token por token con actualización hormonal
        for step in range(max_new_tokens):
            # Generar siguiente token
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=inputs.get("attention_mask")
                )
                next_token_logits = outputs.logits[:, -1, :]

                # Aplicar procesador hormonal
                modulated_logits = logits_processor(input_ids, next_token_logits)

                # Muestreo
                if do_sample:
                    probs = F.softmax(modulated_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = modulated_logits.argmax(dim=-1, keepdim=True)

            # Añadir token al historial del procesador
            processor.add_token(next_token.item())
            generated_tokens.append(next_token.item())

            # Actualizar input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Actualizar attention_mask si existe
            if "attention_mask" in inputs:
                inputs["attention_mask"] = torch.cat([
                    inputs["attention_mask"],
                    torch.ones((inputs["attention_mask"].shape[0], 1), 
                              device=self.device, dtype=inputs["attention_mask"].dtype)
                ], dim=1)

            # Guardar estado hormonal
            if return_hormone_trajectory:
                hormone_trajectory.append(current_profile.to_dict())

            # Actualizar hormonas cada N tokens
            if (step + 1) % update_interval == 0:
                feedback = processor.get_feedback()
                current_profile.update(feedback)
                processor.H = current_profile  # Actualizar referencia

                # Actualizar también la lista de procesadores si cortisol cambió
                if current_profile.cortisol >= 0.6 and len(processors) == 1:
                    processors.append(NoRepeatNGramLogitsProcessor(3))
                    logits_processor = LogitsProcessorList(processors)
                elif current_profile.cortisol < 0.6 and len(processors) > 1:
                    processors = [processor]
                    logits_processor = LogitsProcessorList(processors)

            # Stop si EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        # Decodificar texto completo
        full_ids = input_ids[0].cpu().tolist()
        generated_text = self.tokenizer.decode(full_ids, skip_special_tokens=True)

        result = {
            'generated_text': generated_text,
            'num_tokens': len(generated_tokens),
            'final_hormone_profile': current_profile.to_dict(),
        }

        if return_hormone_trajectory:
            result['hormone_trajectory'] = hormone_trajectory

        return result

    def generate_with_semantic_bias(
        self,
        prompt: str,
        hormone_profile: HormoneProfile,
        semantic_category: str,
        semantic_strength: float = 1.0,
        max_new_tokens: int = 50,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Genera texto con modulación hormonal + sesgo semántico.

        Args:
            prompt: Texto de entrada
            hormone_profile: Configuración hormonal
            semantic_category: Categoría semántica ('empathy', 'creativity', etc.)
            semantic_strength: Fuerza del sesgo semántico
            max_new_tokens: Tokens máximos
            num_return_sequences: Número de secuencias

        Returns:
            Lista de textos generados

        Example:
            >>> texts = model.generate_with_semantic_bias(
            ...     "I'm feeling sad",
            ...     HORMONE_PROFILES["empathic"],
            ...     semantic_category="empathy",
            ...     semantic_strength=1.5
            ... )
        """
        # Lazy import para evitar dependencia obligatoria
        from .semantic import SemanticBiasManager, SemanticLogitsProcessor

        # Crear semantic manager si no existe
        if not hasattr(self, 'semantic_manager'):
            print("Inicializando SemanticBiasManager...")
            self.semantic_manager = SemanticBiasManager(
                self.tokenizer,
                device=self.device
            )

        # Construir procesadores
        hormonal_processor = HormonalLogitsProcessor(
            hormone_profile=hormone_profile,
            empathetic_tokens=self.empathetic_tokens
        )

        semantic_processor = SemanticLogitsProcessor(
            semantic_manager=self.semantic_manager,
            category=semantic_category,
            strength=semantic_strength
        )

        processors = [hormonal_processor, semantic_processor]

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
            do_sample=True,
            top_k=50,
            top_p=0.95,
            logits_processor=logits_processor,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        return [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
