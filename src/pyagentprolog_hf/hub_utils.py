"""Утилиты для публикации PyAgentProlog-модели на HF Hub.

Главная сложность: `save_pretrained` сохраняет только config.json
и веса. Для `trust_remote_code=True` нужно дополнительно положить
в репозиторий файлы:
    * configuration_pyagentprolog.py
    * modeling_pyagentprolog.py

Эта утилита копирует их из установленного пакета в директорию
модели и загружает всё на Hub.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from typing import Optional


# Какие файлы нужны в репо для trust_remote_code=True
_CUSTOM_CODE_FILES = (
    "configuration_pyagentprolog.py",
    "modeling_pyagentprolog.py",
)


def _package_dir() -> str:
    """Директория установленного пакета pyagentprolog_hf."""
    return os.path.dirname(os.path.abspath(__file__))


def copy_custom_code(dest_dir: str) -> list:
    """Копирует configuration_*.py и modeling_*.py в dest_dir.

    Returns:
        Список имён скопированных файлов.
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
    persist_runtime_kb: bool = False,      # ← новая опция
    runtime_kb_name: str = "kb_auto.pl",
) -> list:
    """Сохраняет модель в формате, готовом к upload на HF Hub.
    Args:
        model: PyAgentPrologModel.
        save_directory: целевая директория.
        include_custom_code: копировать ли configuration_*.py и
            modeling_*.py (нужно для trust_remote_code=True).
        persist_runtime_kb: если True — перед сохранением выгружает
            текущее содержимое model.database.rules в KB-файл и
            прописывает его в config.kb_file.

    Returns:
        Список путей к сохранённым файлам (относительно директории).
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
    """Публикует модель на HF Hub.

    Args:
        model: PyAgentPrologModel.
        repo_id: "username/repo-name".
        private: приватный репозиторий.
        commit_message: сообщение коммита.
        include_custom_code: включать ли configuration_*.py,
            modeling_*.py для trust_remote_code=True.
        token: HF-токен (или из env HUGGINGFACE_HUB_TOKEN).

    Returns:
        URL репозитория.
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as e:
        raise ImportError(
            "Для push_to_hub требуется huggingface_hub: "
            "pip install huggingface_hub"
        ) from e

    api = HfApi(token=token)

    # 1) Создаём репо (idempotent)
    url = create_repo(
        repo_id=repo_id,
        token=token,
        private=private,
        exist_ok=True,
        repo_type="model",
    )

    # 2) Сохраняем во временную директорию
    with tempfile.TemporaryDirectory() as tmpdir:
        save_for_hub(
            model,
            tmpdir,
            include_custom_code=include_custom_code,
        )

        # 3) Загружаем
        api.upload_folder(
            folder_path=tmpdir,
            repo_id=repo_id,
            token=token,
            commit_message=commit_message,
            repo_type="model",
        )

    return url if isinstance(url, str) else getattr(url, "repo_id", repo_id)


__all__ = ['save_for_hub', 'push_to_hub', 'copy_custom_code']