"""PyAgentProlog — HF-совместимая конфигурация.

Хранит:
    * Prolog-программу (rules_text);
    * настройки семантической унификации;
    * путь/формат KB;
    * прочие настройки исполнения.

Никаких нейросетевых гиперпараметров — только символьные.
"""
from __future__ import annotations

from typing import Optional

try:
    from transformers import PretrainedConfig
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

    class PretrainedConfig:  # минимальная заглушка
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
            # auto_map и прочие служебные ключи не идут в __init__
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
    """Конфигурация PyAgentProlog для HF.

    Args:
        program:
            Текст Prolog-программы (правила + директивы). Может быть пустым,
            если вся логика — в отдельной KB.
        kb_file:
            Имя файла KB в директории модели (например, "kb.pl" или
            "kb.bin"). Загружается при `__init__`, если `load_kb=True`.
        kb_format:
            'auto' | 'text' | 'binary'. 'auto' → по расширению.
        load_kb:
            Загружать ли KB сразу при создании модели.
        semantic_enabled:
            Включать ли семантическую унификацию.
        semantic_model:
            Имя модели эмбеддингов (HF ID или короткий алиас).
        semantic_backend:
            'st' | 'ollama' | 'custom'.
        semantic_threshold, soft_cos_min, lev_max, lev_typo_max, max_cache:
            Параметры EmbeddingUnifier (см. prolog/embedding_unify.py).
        max_solutions:
            Ограничение на число решений в одном запросе (0 = без лимита).
        with_explanation:
            Возвращать ли дерево доказательства по умолчанию.
        default_main:
            Имя предиката-цели для режима "программа → результат".
            Если None — main/0 не ищется, исполняется только явный запрос.
    """
    model_type = "pyagentprolog"

    def __init__(
        self,
        # --- Программа и KB ---
        program: str = "",
        kb_file: Optional[str] = None,
        kb_format: str = "auto",
        load_kb: bool = True,
        # --- Семантическая унификация ---
        semantic_enabled: bool = False,
        semantic_model: str = "intfloat/multilingual-e5-large",
        semantic_backend: str = "st",
        semantic_threshold: float = 0.82,
        soft_cos_min: float = 0.55,
        lev_max: float = 0.35,
        lev_typo_max: float = 0.20,
        max_cache: int = 10000,
        # --- Исполнение ---
        max_solutions: int = 0,
        with_explanation: bool = False,
        default_main: Optional[str] = None,
        **kwargs,
    ):
        if _HF_AVAILABLE:
            super().__init__(**kwargs)
        else:
            super().__init__()

        # Программа и KB
        self.program = program or ""
        self.kb_file = kb_file
        self.kb_format = kb_format
        self.load_kb = bool(load_kb)

        # Семантика
        self.semantic_enabled = bool(semantic_enabled)
        self.semantic_model = semantic_model
        self.semantic_backend = semantic_backend
        self.semantic_threshold = float(semantic_threshold)
        self.soft_cos_min = float(soft_cos_min)
        self.lev_max = float(lev_max)
        self.lev_typo_max = float(lev_typo_max)
        self.max_cache = int(max_cache)

        # Исполнение
        self.max_solutions = int(max_solutions)
        self.with_explanation = bool(with_explanation)
        self.default_main = default_main

    def save_pretrained(self, save_directory, **kwargs):
        """Сохраняет config.json с auto_map для trust_remote_code."""
        import json
        import os

        super().save_pretrained(save_directory)

        cfg_path = os.path.join(save_directory, "config.json")
        with open(cfg_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Ключи, которые нужны HF для trust_remote_code
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