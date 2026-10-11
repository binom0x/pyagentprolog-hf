"""Регистрация PyAgentProlog в AutoConfig / AutoModel / Pipeline.

Вызывать явно:
    from pyagentprolog_hf import register_with_transformers
    register_with_transformers()

Импорт самого пакета не имеет сайд-эффектов — это важно для
ленивой загрузки и для того, чтобы `import pyagentprolog_hf` не
требовал установленного transformers.
"""
from __future__ import annotations


_REGISTERED = False


def register_with_transformers(force: bool = False) -> bool:
    """Регистрирует PyAgentProlog в AutoConfig / AutoModel / Pipeline.

    Returns:
        True, если регистрация выполнена; False, если transformers
        недоступен.
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