"""
Recurrent-Depth Transformer (Mytho) — full module index.

Core architecture modules are imported by default.
Agent scaffolding (ReAct, Reflexion) is kept separate and imported on demand:
    from mytho_model.react import ReActController
    from mytho_model.reflexion import ReflexionController
"""
from .config import MythoConfig
from .model import MythoModel, UncertaintyDrivenACT
from .scratchpad import LatentScratchpad
from .verifier import VerifierHead
from .branching import BranchingController
from .memory import MemoryManager
from .quantized_cache import HierarchicalQuantizedCache
from .self_consistency import SelfConsistencyDecoder
from .uncertainty import MCDropoutEstimator, EnsembleHead
from .experts import SwitchMoELayer
from .expert_growth import ExpertMetrics, DynamicExpertGrowth

__all__ = [
    # Core
    "MythoConfig", "MythoModel", "UncertaintyDrivenACT",
    "LatentScratchpad", "VerifierHead", "BranchingController",
    "MemoryManager", "HierarchicalQuantizedCache",
    "SelfConsistencyDecoder",
    "MCDropoutEstimator", "EnsembleHead",
    "SwitchMoELayer", "ExpertMetrics", "DynamicExpertGrowth",
    # Agent scaffolding (import from submodule directly):
    #   from mytho_model.react import ReActController
    #   from mytho_model.reflexion import ReflexionController
]

