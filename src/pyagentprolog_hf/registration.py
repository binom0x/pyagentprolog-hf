"""Register PyAgentProlog with AutoConfig / AutoModel / Pipeline.

Call explicitly:
    from pyagentprolog_hf import register_with_transformers
    register_with_transformers()

Importing the package itself has no side effects — this matters for
lazy loading and to ensure `import pyagentprolog_hf` does not
require transformers to be installed.
"""
from __future__ import annotations


_REGISTERED = False


def register_with_transformers(force: bool = False) -> bool:
    """Register PyAgentProlog with AutoConfig / AutoModel / Pipeline.

    Returns:
        True if registration was performed; False if transformers
        is unavailable.
    """
    global _REGISTERED
    if _REGISTERED and not force:
        return True

    try:
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForCausalLM,
        )
    except ImportError:
        return False

    from .configuration_pyagentprolog import PyAgentPrologConfig
    from .modeling_pyagentprolog import PyAgentPrologModel

    AutoConfig.register(
        PyAgentPrologConfig.model_type,
        PyAgentPrologConfig,
        exist_ok=True,
    )
    AutoModel.register(
        PyAgentPrologConfig,
        PyAgentPrologModel,
        exist_ok=True,
    )
    AutoModelForCausalLM.register(
        PyAgentPrologConfig,
        PyAgentPrologModel,
        exist_ok=True,
    )

    # Pipeline (task="prolog")
    try:
        from transformers.pipelines import PIPELINE_REGISTRY
        from transformers import AutoModel as _AutoModel
        from .pipeline_pyagentprolog import PyAgentPrologPipeline

        PIPELINE_REGISTRY.register_pipeline(
            "prolog",
            pipeline_class=PyAgentPrologPipeline,
            pt_model=_AutoModel,
        )
    except Exception as e:
        print(f"[PyAgentProlog] pipeline registration skipped: {e}")

    _REGISTERED = True
    return True


__all__ = ['register_with_transformers']