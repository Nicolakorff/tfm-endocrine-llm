"""Runner de experimentos (stub).

Clase ligera que permite ejecutar experimentos de manera sintética
durante tests sin dependencias externas.
"""
from typing import Any, Dict


class ExperimentRunner:
    """Runner mínimo para coordinar experimentos.

    Los métodos son intencionalmente simples para no requerir I/O.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def run(self) -> Dict[str, Any]:
        # Devuelve un resultado sintético
        return {"status": "ok", "config": self.config}


__all__ = ["ExperimentRunner"]
