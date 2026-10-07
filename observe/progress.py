"""Stage updates, re-exported for readers that already import this module.

The worker binds a path through :mod:`tcenv.events`. This import subscribes the file log. The page reads the file the parent copied beside the run; it does not import the worker.
"""

from __future__ import annotations

from tcenv.events import bind_progress as bind
from tcenv.events import install_file_log, update_progress as update

install_file_log()

__all__ = ["bind", "update", "install_file_log"]
