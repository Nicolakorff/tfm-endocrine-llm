"""
Sistema de Neuromodulación Endocrina para LLMs

Sistema completo de modulación hormonal artificial para modelos de lenguaje,
incluyendo perfiles hormonales estáticos y dinámicos, sesgos semánticos,
y herramientas de experimentación y análisis.

Componentes principales:
- core: Sistema base de modulación hormonal
- metrics: Cálculo de métricas de evaluación
- experiment: Framework de experimentación
- semantic: Sesgos semánticos con embeddings
"""

__version__ = "0.5.0"

# Imports principales
from .core import (
    HormoneProfile,
    EndocrineModulatedLLM,
    HormonalLogitsProcessor,
    HORMONE_PROFILES
)

from .metrics import (
    TextMetrics,
    AdvancedMetrics,
    EmpathyMetrics
)

from .experiment import ExperimentRunner

# Imports opcionales (pueden no estar disponibles)
try:
    from .semantic import (
        SemanticBiasManager,
        SemanticLogitsProcessor,
        analyze_semantic_activation
    )
    _has_semantic = True
except ImportError:
    _has_semantic = False
    SemanticBiasManager = None
    SemanticLogitsProcessor = None
    analyze_semantic_activation = None

# Lista de exports
__all__ = [
    # Core
    'HormoneProfile',
    'EndocrineModulatedLLM',
    'HormonalLogitsProcessor',
    'HORMONE_PROFILES',

    # Metrics
    'TextMetrics',
    'AdvancedMetrics',
    'EmpathyMetrics',

    # Experiment
    'ExperimentRunner',
]

# Añadir semantic solo si está disponible
if _has_semantic:
    __all__.extend([
        'SemanticBiasManager',
        'SemanticLogitsProcessor',
        'analyze_semantic_activation',
    ])

# Metadata del paquete
__author__ = "Tu Nombre"
__description__ = "Sistema de Neuromodulación Endocrina para LLMs"
__url__ = "https://github.com/Nicolakorff/tfm-endocrine-llm"

# Información de versión
def get_version_info():
    """Retorna información detallada de la versión"""
    return {
        'version': __version__,
        'semantic_module': _has_semantic,
        'components': {
            'core': True,
            'metrics': True,
            'experiment': True,
            'semantic': _has_semantic,
        }
    }

# Banner de bienvenida
def print_info():
    """Imprime información del sistema"""
    print("="*60)
    print(f"Sistema de Neuromodulación Endocrina v{__version__}")
    print("="*60)
    print("Componentes disponibles:")
    print("Core (perfiles hormonales)")
    print("Metrics (evaluación)")
    print("Experiment (framework)")
    print(f"{'✓' if _has_semantic else '✗'} Semantic (sesgos semánticos)")
    print("="*60)
