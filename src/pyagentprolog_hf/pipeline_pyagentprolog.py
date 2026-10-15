"""HF Pipeline for PyAgentProlog.

Registered under task="prolog":

    from transformers import pipeline
    pipe = pipeline("prolog", model=model)
    result = pipe("parent(ivan, X).")
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
    """Pipeline: list of Prolog queries → list of solutions."""

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
        """Normalizes the input into a list of queries."""
        if isinstance(inputs, str):
            return [inputs]
        if isinstance(inputs, (list, tuple)):
            return list(inputs)
        # Single dict {'query': ...}
        if isinstance(inputs, dict) and 'query' in inputs:
            return [inputs['query']]
        return [str(inputs)]

    def _forward(self, model_inputs, with_explanation=None):
        """Calls the model. Returns PyAgentPrologOutput."""
        if with_explanation is None:
            with_explanation = self.model.config.with_explanation
        return self.model(queries=model_inputs,
                          with_explanation=with_explanation)

    def postprocess(self, model_outputs, return_all: bool = False):
        """Builds the output.

        By default:
            * one query → list of solutions;
            * multiple queries → list of lists.

        return_all=True → returns the full PyAgentPrologOutput.
        """
        if return_all:
            return model_outputs

        solutions = model_outputs.solutions
        if len(solutions) == 1:
            return solutions[0]
        return solutions


__all__ = ['PyAgentPrologPipeline']