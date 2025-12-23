"""
Sistema de Neuromodulación Endocrina para LLMs
"""

# __version__ = "0.1.0"

# from .core import (
#     HormoneProfile,
#     EndocrineModulatedLLM,
#    HORMONE_PROFILES
# )
# from .metrics import TextMetrics, AdvancedMetrics
# from .experiment import ExperimentRunner
# from .visualization import ResultVisualizer

# __all__ = [
#    'HormoneProfile',
#    'EndocrineModulatedLLM',
#    'HORMONE_PROFILES',
#    'TextMetrics',
#    'AdvancedMetrics',
#    'ExperimentRunner',
#    'ResultVisualizer',
#]

__version__ = "0.2.0"

from .core import (
    HormoneProfile,
    EndocrineModulatedLLM,
    HORMONE_PROFILES
)
from .metrics import TextMetrics, AdvancedMetrics, EmpathyMetrics
from .experiment import ExperimentRunner

__all__ = [
    'HormoneProfile',
    'EndocrineModulatedLLM',
    'HORMONE_PROFILES',
    'TextMetrics',
    'AdvancedMetrics',
    'EmpathyMetrics',
    'ExperimentRunner',
]

__version__ = "0.3.0"

__version__ = "0.4.0"

# Añadir imports semánticos
from .semantic import (
    SemanticBiasManager,
    SemanticLogitsProcessor,
    analyze_semantic_activation
)

__all__ = [
    # ... existentes ...
    'SemanticBiasManager',
    'SemanticLogitsProcessor',
    'analyze_semantic_activation',
]