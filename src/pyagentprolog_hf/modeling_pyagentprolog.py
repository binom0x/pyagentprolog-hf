"""PyAgentProlog — HF-compatible model.

Wraps Solver/Agent from PyAgentProlog into a PreTrainedModel.
There are no weights — all the logic is symbolic. `forward`/`generate`
accept Prolog queries and return solutions.
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

    class nn:  # stub
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
# Model output
# =====================================================================
@dataclass
class PyAgentPrologOutput(ModelOutput):
    """Result of executing Prolog query(ies).

    Fields:
        solutions:  list of solutions. Each is a dict {var: value} or True/False.
        explanations: list of tracer objects (or None).
        queries:    list of the original queries.
        count:      number of solutions (total).
        errors:     list of error messages (None at a position — success).
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
# Main model
# =====================================================================
class PyAgentPrologModel(PreTrainedModel):
    """HF-compatible PyAgentProlog model.

    Usage:

        from pyagentprolog_hf import (
            PyAgentPrologModel, PyAgentPrologConfig,
        )

        config = PyAgentPrologConfig(
            program="parent(ivan, alexey). parent(alexey, peter).",
            semantic_enabled=False,
        )
        model = PyAgentPrologModel(config)
        out = model.generate(["parent(ivan, X)."])
        print(out.solutions)   # [[{'X': 'alexey'}]]
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

        # --- Lazy initialization of the core ---
        self._solver = None
        self._agent = None
        self._unifier = None
        self._initialized = False

        # --- Feedback to HF Hub: directory/repo name ---
        self.name_or_path: Optional[str] = None
        self._model_dir: Optional[str] = None

        # --- Dummy parameter: HF expects a non-empty state_dict ---
        # There are no real weights, but 1 float does not hurt.
        if _HF_AVAILABLE:
            self._dummy = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    def _ensure_solver(self):
        """Creates Solver/Agent, applies directives, loads the KB."""
        if self._initialized:
            return

        from prolog.solver import Solver
        from prolog.agent import Agent

        program = self.config.program or ""
        # Agent parses directives itself via Solver
        self._agent = Agent(program=program, context={})
        self._solver = self._agent.solver

        # Semantic unification (optional)
        if self.config.semantic_enabled:
            self._install_semantic()

        # Load KB from the model directory
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
        """Installs EmbeddingUnifier according to config."""
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
        """Resolves the KB path relative to the model directory or cwd."""
        if not kb_file:
            return None
        if os.path.isabs(kb_file):
            return kb_file
        # 1) directory the model was loaded from
        if self._model_dir and os.path.isdir(self._model_dir):
            cand = os.path.join(self._model_dir, kb_file)
            if os.path.exists(cand):
                return cand
        # 2) name_or_path as a directory
        if (self.name_or_path
                and os.path.isdir(str(self.name_or_path))):
            cand = os.path.join(str(self.name_or_path), kb_file)
            if os.path.exists(cand):
                return cand
        # 3) as-is (cwd)
        return kb_file

    def _load_kb_file(self, path: str, kb_format: str = "auto"):
        """Loads KB into the current knowledge base."""
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
            # Support two formats:
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
        """Executes one or more Prolog queries.

        Priority of query sources:
            query      > queries > input_ids (string or list of strings)

        Returns:
            PyAgentPrologOutput
        """
        self._ensure_solver()

        # --- Input normalization ---
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
                # Tensor input — not supported without a tokenizer
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
                # Count solutions
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
        """HF-compatible generate.

        If `inputs` is a string/list of strings, it is interpreted as
        Prolog query(ies). Returns PyAgentPrologOutput.
        """
        if inputs is None:
            inputs = kwargs.pop("queries", None)
        return self.forward(queries=inputs, **kwargs)

    def __call__(self, *args, **kwargs):
        """nn.Module.__call__ overridden to route to forward."""
        return self.forward(*args, **kwargs)

    # ------------------------------------------------------------------
    # Executing a query
    # ------------------------------------------------------------------
    def _run_query(self, query_text: str, with_explanation: bool):
        """Executes a query via Agent.query or via the main predicate.

        If query_text is empty and config.default_main is set, executes
        `main.` (or main/N with anonymous arguments).
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

        # Limit on the number of solutions
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
        """Builds a query to config.default_main.

        Looks for rules with the main/N head in the current DB and
        substitutes the required number of anonymous variables.
        """
        name = self.config.default_main
        db = self._agent.solver.database
        # Look for the maximum arity of main/N
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
        """Truncates the solution list to limit.

        Supports both result formats of Agent.query:
        * dict with the 'solution' key (list/dict/True/False);
        * list of dicts.
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
        """Extracts solutions from the Agent.query result.

        Returns:
            list[dict] — if there are solutions with bindings;
            list[True] — if success without variables;
            []         — if there are no solutions.
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
        """Load without weights.

        No weights are loaded — the model is symbolic. The directory/repo
        may contain:
            * config.json  (required)
            * kb.pl / kb.bin  (optional)
            * custom code (when trust_remote_code=True)
        """
        # Strip HF-specific kwargs that are not needed
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

        # Remember where it was loaded from — for resolving kb_file
        if isinstance(pretrained_model_name_or_path, str):
            model.name_or_path = pretrained_model_name_or_path
            if os.path.isdir(pretrained_model_name_or_path):
                model._model_dir = pretrained_model_name_or_path

        # Try to initialize the core right away (KB will be loaded)
        try:
            model._ensure_solver()
        except Exception as e:
            print(f"[PyAgentProlog] lazy init deferred: {e}")

        return model

    def save_pretrained(self, save_directory, **kwargs):
        """Saves config.json + dummy weights + KB (if any)."""
        import shutil

        os.makedirs(save_directory, exist_ok=True)

        # 1) config.json with auto_map
        self.config.save_pretrained(save_directory)

        # 2) Dummy weights (1 float) so that HF does not complain
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

        # 3) KB file: copy from the model directory if it is there
        if self.config.kb_file:
            src = self._resolve_kb_path(self.config.kb_file)
            if src and os.path.exists(src) and os.path.isfile(src):
                dst = os.path.join(
                    save_directory,
                    os.path.basename(self.config.kb_file),
                )
                if os.path.abspath(src) != os.path.abspath(dst):
                    shutil.copy2(src, dst)

        # 4) generation_config.json stub (HF compatibility)
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
    # Public utilities
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
        """Adds fact(s) to the knowledge base."""
        self._ensure_solver()
        self._agent.add_fact(fact_text)

    def add_facts(self, facts_text: str):
        """Adds a batch of facts/rules in a single parser call."""
        self._ensure_solver()
        self._agent.add_facts(facts_text)

    def reset_kb(self):
        """Clears the knowledge base (program and KB), keeping builtins."""
        self._ensure_solver()
        self._agent.clear_kb()

    def reload_kb(self):
        """Reloads the KB from config.kb_file."""
        self._ensure_solver()
        if not self.config.kb_file:
            return
        self._agent.clear_kb()
        # Reinitialize the program
        from prolog.parser import Parser
        rules, directives = Parser(self.config.program).parse_rules(
            return_directives=True
        )
        self._solver._apply_directives(directives)
        for r in rules:
            self._agent.solver.database.add_rule(r)
        # And the KB
        path = self._resolve_kb_path(self.config.kb_file)
        if path and os.path.exists(path):
            self._load_kb_file(path, self.config.kb_format)


__all__ = ['PyAgentPrologModel', 'PyAgentPrologOutput']