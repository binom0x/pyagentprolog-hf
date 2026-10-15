"""Smoke tests for PyAgentPrologModel."""
import pytest

from pyagentprolog_hf import PyAgentPrologModel, PyAgentPrologOutput


def test_basic_generate(model):
    out = model.generate(["grandfather(ivan, X)."])
    assert isinstance(out, PyAgentPrologOutput)
    assert out.count == 2
    assert out.queries == ["grandfather(ivan, X)."]
    assert out.errors == [None]
    # The order of solutions is not guaranteed, so compare as sets
    xs = {sol["X"] for sol in out.solutions[0]}
    assert xs == {"peter", "anna"}


def test_failing_query(model):
    out = model.generate(["grandfather(peter, X)."])
    assert out.count == 0
    assert out.solutions == [[]]


def test_multiple_queries(model):
    out = model.generate([
        "parent(ivan, X).",
        "parent(alexey, Y).",
    ])
    assert len(out.solutions) == 2
    assert out.count >= 3


def test_save_load_roundtrip(model, tmp_path):
    model.save_pretrained(str(tmp_path))
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "configuration_pyagentprolog.py").exists() is False  # save_pretrained does not copy them

    from pyagentprolog_hf import PyAgentPrologModel as M
    m2 = M.from_pretrained(str(tmp_path), local_files_only=True)
    out = m2.generate(["grandfather(ivan, X)."])
    assert out.count == 2


def test_save_for_hub_copies_custom_code(model, tmp_path):
    from pyagentprolog_hf import save_for_hub
    save_for_hub(model, str(tmp_path))
    assert (tmp_path / "configuration_pyagentprolog.py").exists()
    assert (tmp_path / "modeling_pyagentprolog.py").exists()
    assert (tmp_path / "config.json").exists()