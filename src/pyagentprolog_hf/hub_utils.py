"""Utilities for publishing a PyAgentProlog model to HF Hub.

The main difficulty: `save_pretrained` saves only config.json
and the weights. For `trust_remote_code=True` you also need to put
the following files into the repository:
    * configuration_pyagentprolog.py
    * modeling_pyagentprolog.py

This utility copies them from the installed package into the model
directory and uploads everything to the Hub.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from typing import Optional


# Which files are needed in the repo for trust_remote_code=True
_CUSTOM_CODE_FILES = (
    "configuration_pyagentprolog.py",
    "modeling_pyagentprolog.py",
)


def _package_dir() -> str:
    """Directory of the installed pyagentprolog_hf package."""
    return os.path.dirname(os.path.abspath(__file__))


def copy_custom_code(dest_dir: str) -> list:
    """Copies configuration_*.py and modeling_*.py into dest_dir.

    Returns:
        List of names of the copied files.
    """
    os.makedirs(dest_dir, exist_ok=True)
    src_dir = _package_dir()
    copied = []
    for fname in _CUSTOM_CODE_FILES:
        src = os.path.join(src_dir, fname)
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"Custom code file not found in package: {src}"
            )
        dst = os.path.join(dest_dir, fname)
        shutil.copy2(src, dst)
        copied.append(fname)
    return copied


def save_for_hub(
    model,
    save_directory: str,
    include_custom_code: bool = True,
    persist_runtime_kb: bool = False,      # ← new option
    runtime_kb_name: str = "kb_auto.pl",
) -> list:
    """Saves the model in a format ready for upload to HF Hub.

    Args:
        model: PyAgentPrologModel.
        save_directory: target directory.
        include_custom_code: whether to copy configuration_*.py and
            modeling_*.py (needed for trust_remote_code=True).
        persist_runtime_kb: if True — before saving, dumps the current
            contents of model.database.rules to a KB file and writes
            its name into config.kb_file.

    Returns:
        List of paths to saved files (relative to the directory).
    """
    os.makedirs(save_directory, exist_ok=True)

    if persist_runtime_kb:
        kb_path = os.path.join(save_directory, runtime_kb_name)
        model.agent.save_kb(kb_path, format="prolog")
        model.config.kb_file = runtime_kb_name
        model.config.kb_format = "text"
        model.config.load_kb = True

    model.save_pretrained(save_directory)

    if include_custom_code:
        copy_custom_code(save_directory)

    return sorted(os.listdir(save_directory))


def push_to_hub(
    model,
    repo_id: str,
    *,
    private: bool = False,
    commit_message: str = "Upload PyAgentProlog model",
    include_custom_code: bool = True,
    token: Optional[str] = None,
) -> str:
    """Publishes the model to HF Hub.

    Args:
        model: PyAgentPrologModel.
        repo_id: "username/repo-name".
        private: private repository.
        commit_message: commit message.
        include_custom_code: whether to include configuration_*.py,
            modeling_*.py for trust_remote_code=True.
        token: HF token (or taken from env HUGGINGFACE_HUB_TOKEN).

    Returns:
        Repository URL.
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as e:
        raise ImportError(
            "push_to_hub requires huggingface_hub: "
            "pip install huggingface_hub"
        ) from e

    api = HfApi(token=token)

    # 1) Create the repo (idempotent)
    url = create_repo(
        repo_id=repo_id,
        token=token,
        private=private,
        exist_ok=True,
        repo_type="model",
    )

    # 2) Save into a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        save_for_hub(
            model,
            tmpdir,
            include_custom_code=include_custom_code,
        )

        # 3) Upload
        api.upload_folder(
            folder_path=tmpdir,
            repo_id=repo_id,
            token=token,
            commit_message=commit_message,
            repo_type="model",
        )

    return url if isinstance(url, str) else getattr(url, "repo_id", repo_id)


__all__ = ['save_for_hub', 'push_to_hub', 'copy_custom_code']