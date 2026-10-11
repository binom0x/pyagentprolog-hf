"""Общие фикстуры для тестов PyAgentProlog-HF."""
import pytest


@pytest.fixture(scope="session")
def simple_program() -> str:
    return """
    родитель(иван, алексей).
    родитель(иван, мария).
    родитель(алексей, петр).
    родитель(алексей, анна).

    дедушка(Д, В) :- родитель(Д, Р), родитель(Р, В).
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