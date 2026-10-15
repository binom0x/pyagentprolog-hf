"""Shared fixtures for PyAgentProlog-HF tests."""
import pytest


@pytest.fixture(scope="session")
def simple_program() -> str:
    return """
    parent(ivan, alexey).
    parent(ivan, maria).
    parent(alexey, peter).
    parent(alexey, anna).

    grandfather(G, S) :- parent(G, P), parent(P, S).
    """


@pytest.fixture
def config(simple_program):
    from pyagentprolog_hf import PyAgentPrologConfig
    return PyAgentPrologConfig(
        program=simple_program,
        semantic_enabled=False,
    )


@pytest.fixture
def model(config):
    from pyagentprolog_hf import PyAgentPrologModel
    return PyAgentPrologModel(config)