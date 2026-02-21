"""Centralised logging utilities for legend components."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

_LOGGER_NAME = "block_model_viewer.legend"
_INITIALISED = False
_LOG_FILE: Optional[Path] = None


def _ensure_initialized() -> None:
    """Ensure a dedicated file handler is attached to the legend logger."""

    global _INITIALISED, _LOG_FILE
    if _INITIALISED:
        return

    log_dir = Path.home() / ".block_model_viewer" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = log_dir / "legend_debug.log"

    handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
    )

    legend_logger = logging.getLogger(_LOGGER_NAME)
    legend_logger.setLevel(logging.DEBUG)
    legend_logger.propagate = False

    if not any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", None) == str(_LOG_FILE)
        for h in legend_logger.handlers
    ):
        legend_logger.addHandler(handler)

    _INITIALISED = True


def get_legend_logger(component: str = "general") -> logging.Logger:
    """Return a logger scoped to the legend system.

    Args:
        component: Logical component name (e.g. "manager", "widget").

    Returns:
        Configured logger instance.
    """

    _ensure_initialized()
    return logging.getLogger(f"{_LOGGER_NAME}.{component}")


__all__ = ["get_legend_logger"]
