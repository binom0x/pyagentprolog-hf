"""PyAgentProlog — HF-compatible wrapper.

Lets you use PyAgentProlog as a regular HF model:
    * `from_pretrained` / `save_pretrained`;
    * `AutoModel.from_pretrained(..., trust_remote_code=True)`;
    * `pipeline("prolog", model=...)`;
    * locally with `local_files_only=True` or via HF Hub.

The package does NOT change the existing PyAgentProlog — it is only
an add-on.
"""
from .configuration_pyagentprolog import PyAgentPrologConfig
from .modeling_pyagentprolog import PyAgentPrologModel, PyAgentPrologOutput
from .pipeline_pyagentprolog import PyAgentPrologPipeline
from .hub_utils import save_for_hub, push_to_hub, copy_custom_code
from .registration import register_with_transformers

__version__ = "0.2.0"

__all__ = [
    "PyAgentPrologConfig",
    "PyAgentPrologModel",
    "PyAgentPrologOutput",
    "PyAgentPrologPipeline",
    "save_for_hub",
    "push_to_hub",
    "copy_custom_code",
    "register_with_transformers",
]