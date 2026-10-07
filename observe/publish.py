"""The dashboard's subscription to :mod:`tcenv.events`.

Importing this module subscribes the file log. The page reads those files; it does not import the search loop. Search and evaluation emit events without importing this package.
"""

from __future__ import annotations

from tcenv.events import (
    append_jsonl,
    cache_root,
    gpu_dir,
    gpu_events_path,
    install_file_log,
    job_state,
    note_gpu,
    observe_dir,
    publish,
    tcenv_live_path,
)

install_file_log()

__all__ = [
    "append_jsonl",
    "cache_root",
    "gpu_dir",
    "gpu_events_path",
    "install_file_log",
    "job_state",
    "note_gpu",
    "observe_dir",
    "publish",
    "tcenv_live_path",
]
