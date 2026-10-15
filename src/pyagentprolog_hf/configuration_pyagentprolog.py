"""PyAgentProlog — HF-compatible configuration.

Stores:
    * the Prolog program (rules_text);
    * semantic unification settings;
    * KB path/format;
    * other execution settings.

No neural-network hyperparameters — only symbolic ones.
"""
from __future__ import annotations

from typing import Optional

try:
    from transformers import PretrainedConfig
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

    class PretrainedConfig:  # minimal stub
        model_type = "pyagentprolog"

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            import json, os
            cfg_path = os.path.join(path, "config.json")
            with open(cfg_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # auto_map and other service keys are not passed to __init__
            data.pop("auto_map", None)
            data.pop("architectures", None)
            return cls(**data)

        def save_pretrained(self, path):
            import json, os
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "config.json"),
                      'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        def to_dict(self):
            return {k: v for k, v in self.__dict__.items()
                    if not k.startswith('_')}


class PyAgentPrologConfig(PretrainedConfig):
    """PyAgentProlog configuration for HF.

    Args:
        program:
            Text of the Prolog program (rules + directives). May be empty
            if all the logic lives in a separate KB.
        kb_file:
            Name of the KB file inside the model directory (e.g. "kb.pl"
            or "kb.bin"). Loaded on `__init__` if `load_kb=True`.
        kb_format:
            'auto' | 'text' | 'binary'. 'auto' → decided by extension.
        load_kb:
            Whether to load the KB immediately when the model is created.
        semantic_enabled:
            Whether to enable semantic unification.
        semantic_model:
            Name of the embedding model (HF ID or short alias).
        semantic_backend:
            'st' | 'ollama' | 'custom'.
        semantic_threshold, soft_cos_min, lev_max, lev_typo_max, max_cache:
            Parameters of EmbeddingUnifier (see prolog/embedding_unify.py).
        max_solutions:
            Limit on the number of solutions per query (0 = no limit).
        with_explanation:
            Whether to return the proof tree by default.
        default_main:
            Name of the goal predicate for "program → result" mode.
            If None — main/0 is not looked up, and only an explicit
            query is executed.
    """
    model_type = "pyagentprolog"

    def __init__(
        self,
        # --- Program and KB ---
        program: str = "",
        kb_file: Optional[str] = None,
        kb_format: str = "auto",
        load_kb: bool = True,
        # --- Semantic unification ---
        semantic_enabled: bool = False,
        semantic_model: str = "intfloat/multilingual-e5-large",
        semantic_backend: str = "st",
        semantic_threshold: float = 0.82,
        soft_cos_min: float = 0.55,
        lev_max: float = 0.35,
        lev_typo_max: float = 0.20,
        max_cache: int = 10000,
        # --- Execution ---
        max_solutions: int = 0,
        with_explanation: bool = False,
        default_main: Optional[str] = None,
        **kwargs,
    ):
        if _HF_AVAILABLE:
            super().__init__(**kwargs)
        else:
            super().__init__()

        # Program and KB
        self.program = program or ""
        self.kb_file = kb_file
        self.kb_format = kb_format
        self.load_kb = bool(load_kb)

        # Semantics
        self.semantic_enabled = bool(semantic_enabled)
        self.semantic_model = semantic_model
        self.semantic_backend = semantic_backend
        self.semantic_threshold = float(semantic_threshold)
        self.soft_cos_min = float(soft_cos_min)
        self.lev_max = float(lev_max)
        self.lev_typo_max = float(lev_typo_max)
        self.max_cache = int(max_cache)

        # Execution
        self.max_solutions = int(max_solutions)
        self.with_explanation = bool(with_explanation)
        self.default_main = default_main

    def save_pretrained(self, save_directory, **kwargs):
        """Saves config.json with auto_map for trust_remote_code."""
        import json
        import os

        super().save_pretrained(save_directory)

        cfg_path = os.path.join(save_directory, "config.json")
        with open(cfg_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Keys HF needs for trust_remote_code
        data["architectures"] = ["PyAgentPrologModel"]
        data["auto_map"] = {
            "AutoConfig":
                "configuration_pyagentprolog.PyAgentPrologConfig",
            "AutoModel":
                "modeling_pyagentprolog.PyAgentPrologModel",
            "AutoModelForCausalLM":
                "modeling_pyagentprolog.PyAgentPrologModel",
        }

        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


__all__ = ['PyAgentPrologConfig']