"""HF Pipeline для PyAgentProlog.

Регистрируется под task="prolog":

    from transformers import pipeline
    pipe = pipeline("prolog", model=model)
    result = pipe("родитель(иван, X).")
"""
from __future__ import annotations

from typing import Any, List, Union

try:
    from transformers import Pipeline
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

    class Pipeline:
        def __init__(self, model=None, **kwargs):
            self.model = model


class PyAgentPrologPipeline(Pipeline):
    """Пайплайн: список Prolog-запросов → список решений."""

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        forward_kwargs = {}
        postprocess_kwargs = {}

        if "with_explanation" in kwargs:
            forward_kwargs["with_explanation"] = kwargs["with_explanation"]

        if "return_all" in kwargs:
            postprocess_kwargs["return_all"] = kwargs["return_all"]

        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, inputs, **kwargs):
        """Приводит вход к списку запросов."""
        if isinstance(inputs, str):
            return [inputs]
        if isinstance(inputs, (list, tuple)):
            return list(inputs)
        # Одиночный dict {'query': ...}
        if isinstance(inputs, dict) and 'query' in inputs:
            return [inputs['query']]
        return [str(inputs)]

    def _forward(self, model_inputs, with_explanation=None):
        """Вызывает модель. Возвращает PyAgentPrologOutput."""
        if with_explanation is None:
            with_explanation = self.model.config.with_explanation
        return self.model(queries=model_inputs,
                          with_explanation=with_explanation)

    def postprocess(self, model_outputs, return_all: bool = False):
        """Формирует вывод.

        По умолчанию:
            * один запрос → список решений;
            * несколько запросов → список списков.

        return_all=True → возвращает полный PyAgentPrologOutput.
        """
        if return_all:
            return model_outputs

        solutions = model_outputs.solutions
        if len(solutions) == 1:
            return solutions[0]
        return solutions


__all__ = ['PyAgentPrologPipeline']