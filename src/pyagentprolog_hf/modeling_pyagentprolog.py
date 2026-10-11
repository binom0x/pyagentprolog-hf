"""PyAgentProlog — HF-совместимая модель.

Оборачивает Solver/Agent из PyAgentProlog в PreTrainedModel.
Весов нет — вся логика символьная. `forward`/`generate` принимают
Prolog-запросы и возвращают решения.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

try:
    import torch
    import torch.nn as nn
    from transformers import PreTrainedModel
    from transformers.utils import ModelOutput
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    torch = None

    class nn:  # заглушка
        class Module:
            def __init__(self, *a, **kw): pass
        class Parameter:
            def __init__(self, *a, **kw): pass

    class PreTrainedModel:
        config_class = None
        base_model_prefix = "pyagentprolog"
        main_input_name = "queries"

        def __init__(self, config):
            self.config = config

        def save_pretrained(self, path, **kwargs):
            import os
            os.makedirs(path, exist_ok=True)
            self.config.save_pretrained(path)

    class ModelOutput(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            for k, v in kwargs.items():
                setattr(self, k, v)

        def to_tuple(self):
            return tuple(self.values())


from .configuration_pyagentprolog import PyAgentPrologConfig


# =====================================================================
# Выход модели
# =====================================================================
@dataclass
class PyAgentPrologOutput(ModelOutput):
    """Результат выполнения Prolog-запроса(ов).

    Поля:
        solutions:  список решений. Каждое — dict {var: value} или True/False.
        explanations: список tracer-объектов (или None).
        queries:    список исходных запросов.
        count:      число решений (суммарно).
        errors:     список сообщений об ошибках (None на позиции — успех).
    """
    solutions: List[Any] = field(default_factory=list)
    explanations: List[Any] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    count: int = 0
    errors: List[Optional[str]] = field(default_factory=list)

    def __post_init__(self):
        if _HF_AVAILABLE:
            super().__init__(
                solutions=self.solutions,
                explanations=self.explanations,
                queries=self.queries,
                count=self.count,
                errors=self.errors,
            )


# =====================================================================
# Основная модель
# =====================================================================
class PyAgentPrologModel(PreTrainedModel):
    """HF-совместимая модель PyAgentProlog.

    Использование:

        from pyagentprolog_hf import (
            PyAgentPrologModel, PyAgentPrologConfig,
        )

        config = PyAgentPrologConfig(
            program="родитель(иван, алексей). родитель(алексей, петр).",
            semantic_enabled=False,
        )
        model = PyAgentPrologModel(config)
        out = model.generate(["родитель(иван, X)."])
        print(out.solutions)   # [[{'X': 'алексей'}]]
    """

    config_class = PyAgentPrologConfig
    base_model_prefix = "pyagentprolog"
    main_input_name = "queries"
    _supports_gradient_checkpointing = False

    def __init__(self, config: PyAgentPrologConfig):
        if _HF_AVAILABLE:
            super().__init__(config)
        else:
            super().__init__(config)

        self.config = config

        # --- Ленивая инициализация ядра ---
        self._solver = None
        self._agent = None
        self._unifier = None
        self._initialized = False

        # --- Обратная связь с HF Hub: имя директории/репо ---
        self.name_or_path: Optional[str] = None
        self._model_dir: Optional[str] = None

        # --- Dummy-параметр: HF ожидает непустой state_dict ---
        # Реальных весов нет, но 1 float не мешает.
        if _HF_AVAILABLE:
            self._dummy = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    # Ядро
    # ------------------------------------------------------------------
    def _ensure_solver(self):
        """Создаёт Solver/Agent, применяет директивы, загружает KB."""
        if self._initialized:
            return

        from prolog.solver import Solver
        from prolog.agent import Agent

        program = self.config.program or ""
        # Agent сам парсит директивы через Solver
        self._agent = Agent(program=program, context={})
        self._solver = self._agent.solver

        # Семантическая унификация (опционально)
        if self.config.semantic_enabled:
            self._install_semantic()

        # Загрузка KB из директории модели
        if self.config.load_kb and self.config.kb_file:
            kb_path = self._resolve_kb_path(self.config.kb_file)
            if kb_path and os.path.exists(kb_path):
                self._load_kb_file(kb_path, self.config.kb_format)
            else:
                print(
                    f"[PyAgentProlog] KB file not found: "
                    f"{self.config.kb_file!r} (resolved: {kb_path!r})"
                )

        self._initialized = True

    def _install_semantic(self):
        """Устанавливает EmbeddingUnifier согласно config."""
        try:
            from prolog.embedding_unify import semantic_on
        except ImportError as e:
            print(f"[PyAgentProlog] embedding_unify unavailable: {e}")
            return

        try:
            semantic_on(
                threshold=self.config.semantic_threshold,
                model=self.config.semantic_model,
                max_cache=self.config.max_cache,
                soft_cos_min=self.config.soft_cos_min,
                lev_max=self.config.lev_max,
                lev_typo_max=self.config.lev_typo_max,
                backend=self.config.semantic_backend,
            )
        except Exception as e:
            print(f"[PyAgentProlog] semantic_on failed: {e}")

    def _resolve_kb_path(self, kb_file: str) -> Optional[str]:
        """Разрешает путь к KB относительно директории модели или cwd."""
        if not kb_file:
            return None
        if os.path.isabs(kb_file):
            return kb_file
        # 1) директория, откуда загружена модель
        if self._model_dir and os.path.isdir(self._model_dir):
            cand = os.path.join(self._model_dir, kb_file)
            if os.path.exists(cand):
                return cand
        # 2) name_or_path как директория
        if (self.name_or_path
                and os.path.isdir(str(self.name_or_path))):
            cand = os.path.join(str(self.name_or_path), kb_file)
            if os.path.exists(cand):
                return cand
        # 3) как есть (cwd)
        return kb_file

    def _load_kb_file(self, path: str, kb_format: str = "auto"):
        """Загрузка KB в текущую базу знаний."""
        from prolog.parser import Parser
        import pickle

        if kb_format == "auto":
            ext = os.path.splitext(path)[1].lower()
            kb_format = "binary" if ext in (
                ".bin", ".pkl", ".pickle"
            ) else "text"

        if kb_format == "text":
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            rules, directives = Parser(text).parse_rules(
                return_directives=True
            )
            self._solver._apply_directives(directives)
            for r in rules:
                self._agent.solver.database.add_rule(r)
        elif kb_format == "binary":
            with open(path, 'rb') as f:
                data = pickle.load(f)
            # Поддержка двух форматов:
            #   list[Rule]                      (KG-builder save_binary)
            #   {'version': 1, 'rules': [...]}  (Agent.save_kb)
            if isinstance(data, dict) and 'rules' in data:
                rules = data['rules']
            elif isinstance(data, list):
                rules = data
            else:
                raise ValueError(
                    f"Unsupported binary KB format: "
                    f"{type(data).__name__}"
                )
            for r in rules:
                self._agent.solver.database.add_rule(r)
        else:
            raise ValueError(f"Unknown kb_format: {kb_format!r}")

    # ------------------------------------------------------------------
    # HF API: forward / generate
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        queries: Optional[Union[str, List[str]]] = None,
        query: Optional[str] = None,
        with_explanation: Optional[bool] = None,
        **kwargs,
    ) -> PyAgentPrologOutput:
        """Выполняет один или несколько Prolog-запросов.

        Приоритет источников запроса:
            query      > queries > input_ids (строка или список строк)

        Returns:
            PyAgentPrologOutput
        """
        self._ensure_solver()

        # --- Нормализация входа ---
        if query is not None:
            q_list = [query]
        elif queries is not None:
            if isinstance(queries, str):
                q_list = [queries]
            else:
                q_list = list(queries)
        elif input_ids is not None:
            if isinstance(input_ids, str):
                q_list = [input_ids]
            elif isinstance(input_ids, (list, tuple)):
                q_list = list(input_ids)
            else:
                # Тензорный ввод — не поддерживаем без токенизатора
                q_list = []
        else:
            q_list = []

        if with_explanation is None:
            with_explanation = self.config.with_explanation

        solutions: List[Any] = []
        explanations: List[Any] = []
        errors: List[Optional[str]] = []
        total = 0

        for q in q_list:
            try:
                sol, expl = self._run_query(q, with_explanation)
                solutions.append(sol)
                explanations.append(expl)
                errors.append(None)
                # Считаем решения
                if isinstance(sol, list):
                    total += len(sol)
                elif sol is True:
                    total += 1
                elif isinstance(sol, dict):
                    total += 1
            except Exception as e:
                solutions.append(None)
                explanations.append(None)
                errors.append(f"{type(e).__name__}: {e}")

        return PyAgentPrologOutput(
            solutions=solutions,
            explanations=explanations,
            queries=q_list,
            count=total,
            errors=errors,
        )

    def generate(
        self,
        inputs: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> PyAgentPrologOutput:
        """HF-совместимый generate.

        Если `inputs` — строка/список строк, интерпретируется как
        Prolog-запрос(ы). Возвращает PyAgentPrologOutput.
        """
        if inputs is None:
            inputs = kwargs.pop("queries", None)
        return self.forward(queries=inputs, **kwargs)

    def __call__(self, *args, **kwargs):
        """nn.Module.__call__ переопределён, чтобы вести на forward."""
        return self.forward(*args, **kwargs)

    # ------------------------------------------------------------------
    # Выполнение запроса
    # ------------------------------------------------------------------
    def _run_query(self, query_text: str, with_explanation: bool):
        """Выполняет запрос через Agent.query или через main-предикат.

        Если query_text пустой и задан config.default_main, выполняет
        `main.` (или main/N с анонимными аргументами).
        """
        if not query_text or not query_text.strip():
            if self.config.default_main:
                query_text = self._format_main_query()
            else:
                return [], None

        agent = self._agent
        result = agent.query(
            query_text, with_explanation=with_explanation
        )

        # Ограничение на число решений
        if self.config.max_solutions > 0:
            result = self._truncate_solutions(
                result, self.config.max_solutions
            )

        if with_explanation:
            parsed, explanation = result
            return self._extract_solutions(parsed), explanation
        else:
            return self._extract_solutions(result), None

    def _format_main_query(self) -> str:
        """Формирует запрос к config.default_main.

        Ищет правила с головой main/N в текущей БД и подставляет
        анонимные переменные нужной арности.
        """
        name = self.config.default_main
        db = self._agent.solver.database
        # Ищем максимальную арность main/N
        candidates = [(n, a) for (n, a) in db.rule_index.keys() if n == name]
        if not candidates:
            return f"{name}."
        max_arity = max(a for _, a in candidates)
        if max_arity == 0:
            return f"{name}."
        args = ", ".join(["_"] * max_arity)
        return f"{name}({args})."

    @staticmethod
    def _truncate_solutions(result, limit: int):
        """Обрезает список решений до limit.

        Поддерживает оба формата результата Agent.query:
        * dict с ключом 'solution' (list/dict/True/False);
        * список словарей.
        """
        if isinstance(result, list):
            return result[:limit]
        if isinstance(result, dict):
            sol = result.get('solution')
            if isinstance(sol, list):
                result['solution'] = sol[:limit]
                if 'count' in result:
                    result['count'] = len(result['solution'])
            return result
        return result

    @staticmethod
    def _extract_solutions(parsed):
        """Извлекает решения из результата Agent.query.

        Возвращает:
            list[dict] — если есть решения со связками;
            list[True] — если успех без переменных;
            []         — если нет решений.
        """
        if parsed is True:
            return [True]
        if parsed is False:
            return []
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            sol = parsed.get('solution')
            if sol is True:
                return [True]
            if sol is False:
                return []
            if isinstance(sol, list):
                return sol
            if isinstance(sol, dict):
                return [sol]
        return []

    # ------------------------------------------------------------------
    # HF API: from_pretrained / save_pretrained
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config: Optional[PyAgentPrologConfig] = None,
        **kwargs,
    ):
        """Загрузка без весов.

        Веса не загружаются — модель символьная. Директория/репо может
        содержать:
            * config.json  (обязательно)
            * kb.pl / kb.bin  (опционально)
            * custom code (при trust_remote_code=True)
        """
        # Убираем HF-специфичные kwargs, которые не нужны
        kwargs.pop("ignore_mismatched_sizes", None)
        kwargs.pop("low_cpu_mem_usage", None)
        kwargs.pop("torch_dtype", None)
        kwargs.pop("device_map", None)

        if config is None:
            try:
                config = PyAgentPrologConfig.from_pretrained(
                    pretrained_model_name_or_path, **kwargs
                )
            except Exception:
                config = PyAgentPrologConfig()

        model = cls(config)

        # Запоминаем, откуда загружены — для resolve kb_file
        if isinstance(pretrained_model_name_or_path, str):
            model.name_or_path = pretrained_model_name_or_path
            if os.path.isdir(pretrained_model_name_or_path):
                model._model_dir = pretrained_model_name_or_path

        # Пробуем инициализировать ядро сразу (KB загрузится)
        try:
            model._ensure_solver()
        except Exception as e:
            print(f"[PyAgentProlog] lazy init deferred: {e}")

        return model

    def save_pretrained(self, save_directory, **kwargs):
        """Сохраняет config.json + dummy-веса + KB (если есть)."""
        import shutil

        os.makedirs(save_directory, exist_ok=True)

        # 1) config.json с auto_map
        self.config.save_pretrained(save_directory)

        # 2) dummy-веса (1 float), чтобы HF не ругался
        if _HF_AVAILABLE:
            import torch as _torch
            state = {"_dummy": _torch.zeros(1)}
            _torch.save(state,
                        os.path.join(save_directory, "pytorch_model.bin"))
        else:
            import pickle
            with open(os.path.join(save_directory, "pytorch_model.bin"),
                      'wb') as f:
                pickle.dump({"_dummy": 0.0}, f)

        # 3) KB-файл: копируем из директории модели, если он там есть
        if self.config.kb_file:
            src = self._resolve_kb_path(self.config.kb_file)
            if src and os.path.exists(src) and os.path.isfile(src):
                dst = os.path.join(
                    save_directory,
                    os.path.basename(self.config.kb_file),
                )
                if os.path.abspath(src) != os.path.abspath(dst):
                    shutil.copy2(src, dst)

        # 4) Заглушка generation_config.json (HF-совместимость)
        import json
        gen_cfg = {
            "max_new_tokens": self.config.max_solutions or 0,
            "do_sample": False,
            "num_return_sequences": 1,
        }
        with open(os.path.join(save_directory, "generation_config.json"),
                  'w', encoding='utf-8') as f:
            json.dump(gen_cfg, f, indent=2)

    # ------------------------------------------------------------------
    # Публичные утилиты
    # ------------------------------------------------------------------
    @property
    def agent(self):
        self._ensure_solver()
        return self._agent

    @property
    def solver(self):
        self._ensure_solver()
        return self._solver

    @property
    def database(self):
        self._ensure_solver()
        return self._agent.solver.database

    def add_fact(self, fact_text: str):
        """Добавляет факт(ы) в базу знаний."""
        self._ensure_solver()
        self._agent.add_fact(fact_text)

    def add_facts(self, facts_text: str):
        """Добавляет пачку фактов/правил за один вызов парсера."""
        self._ensure_solver()
        self._agent.add_facts(facts_text)

    def reset_kb(self):
        """Очищает базу знаний (программу и KB), сохраняя builtins."""
        self._ensure_solver()
        self._agent.clear_kb()

    def reload_kb(self):
        """Перезагружает KB из config.kb_file."""
        self._ensure_solver()
        if not self.config.kb_file:
            return
        self._agent.clear_kb()
        # Переинициализируем программу
        from prolog.parser import Parser
        rules, directives = Parser(self.config.program).parse_rules(
            return_directives=True
        )
        self._solver._apply_directives(directives)
        for r in rules:
            self._agent.solver.database.add_rule(r)
        # И KB
        path = self._resolve_kb_path(self.config.kb_file)
        if path and os.path.exists(path):
            self._load_kb_file(path, self.config.kb_format)


__all__ = ['PyAgentPrologModel', 'PyAgentPrologOutput']