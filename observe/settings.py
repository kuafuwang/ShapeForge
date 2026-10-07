"""The dashboard's model picker. Lists model ids from the credentials file and writes the chosen one into ``taic.toml``.

The credentials file holds API keys. This module never returns those keys, the base URL, or prices. A running search keeps the model it
started with; the next search that reads ``taic.toml`` uses the new one, including the next task a live experiment queue has not started yet.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from pathlib import Path

from observe.actions import Refused

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "taic.toml"
_SAFE = re.compile(r"[A-Za-z0-9._:-]+")


def _config_path() -> Path:
    override = os.environ.get("TAIC_CONFIG")
    return Path(override) if override else CONFIG_PATH


def _read_config() -> tuple[Path, str, dict]:
    path = _config_path()
    try:
        text = path.read_text(encoding="utf-8")
        doc = tomllib.loads(text)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise Refused(500, f"cannot read {path}: {e}") from e
    llm = doc.get("llm") if isinstance(doc, dict) else None
    if not isinstance(llm, dict):
        raise Refused(500, f"{path} has no [llm] section")
    return path, text, llm


def _credentials(llm: dict) -> dict:
    raw = llm.get("credentials_file") or "~/.pi/agent/models.json"
    path = Path(os.path.expanduser(str(raw)))
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(500, f"cannot read the model list from {path}: {e}") from e
    providers = doc.get("providers") if isinstance(doc, dict) else None
    if not isinstance(providers, dict):
        raise Refused(500, f"{path} has no providers")
    return providers


def model_choices() -> dict:
    """The current default and every model id in the credentials file. No keys, URLs, or prices."""
    _path, _text, llm = _read_config()
    choices = []
    for provider, prov in _credentials(llm).items():
        if not isinstance(prov, dict):
            continue
        for model in prov.get("models") or []:
            if not isinstance(model, dict) or not model.get("id"):
                continue
            choices.append({"provider": str(provider), "id": str(model["id"]), "name": str(model.get("name") or model["id"])})
    return {"provider": str(llm.get("provider") or ""), "model": str(llm.get("model") or ""), "choices": choices}


def _replace_in_llm(text: str, key: str, value: str) -> tuple[str, bool]:
    """Replace the first ``key = "..."`` line inside ``[llm]``, before the next section."""
    lines = text.splitlines(keepends=True)
    in_llm = False
    replaced = False
    out = []
    pattern = re.compile(rf'^(\s*{key}\s*=\s*")[^"]*(")')
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_llm = stripped == "[llm]"
        if in_llm and not replaced and pattern.match(line):
            out.append(pattern.sub(rf"\g<1>{value}\2", line, count=1))
            replaced = True
            continue
        out.append(line)
    return "".join(out), replaced


def set_default_model(provider: str, model: str) -> dict:
    """Write ``provider`` and ``model`` into ``taic.toml``. Both must already be listed in the credentials file."""
    if not isinstance(provider, str) or not isinstance(model, str) or not _SAFE.fullmatch(provider) or not _SAFE.fullmatch(model):
        raise Refused(400, "provider and model must be the ids from the list")
    current = model_choices()
    if not any(row["provider"] == provider and row["id"] == model for row in current["choices"]):
        raise Refused(400, f"unknown model {provider}/{model}")
    path, text, _llm = _read_config()
    new, n_provider = _replace_in_llm(text, "provider", provider)
    new, n_model = _replace_in_llm(new, "model", model)
    if not n_provider or not n_model:
        raise Refused(500, f"{path} has no provider or model line under [llm]")
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(new, encoding="utf-8")
    os.replace(tmp, path)
    return {"provider": provider, "model": model}
