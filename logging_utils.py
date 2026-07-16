"""
Minimal logging utilities for the DesignLoRA server.

Creates one directory per session (under ``DESIGNLORA_LOGS_DIR``) that holds the
setup metadata; the server writes per-interaction artifacts (panel images,
checkpoints) into it. Console diagnostics go through the standard ``logging``
module.

This is a slimmed-down replacement for the original research logger, which also
dumped LoRA weights, VLM query traces, and composite PNG grids on every step.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class InteractionLogger:
    """Create per-session log directories and forward events to ``logging``."""

    def __init__(self, base_dir: str = "./designlora_logs"):
        self.base_dir = Path(base_dir)
        self.sessions_dir = self.base_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def create_session_dir(self, user_id: str, timestamp: Optional[str] = None) -> Path:
        timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.sessions_dir / f"{user_id}_{timestamp}"
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def log_setup(self, user_id: str, setup_data: Dict[str, Any]) -> Path:
        """Create a session directory, persist the setup metadata, and return it."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.create_session_dir(user_id, timestamp)
        with open(session_dir / "setup.json", "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "user_id": user_id,
                    "setup_data": setup_data,
                    "session_dir": str(session_dir),
                },
                f,
                indent=2,
            )
        logging.info("Session started: %s", session_dir)
        return session_dir

    def log_system_event(
        self,
        event_type: str,
        user_id: str,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Forward a system event to the standard logger (no disk artifacts)."""
        if error:
            logging.error("[%s] user=%s: %s", event_type, user_id, error)
        else:
            logging.info("[%s] user=%s", event_type, user_id)


_logger: Optional[InteractionLogger] = None


def get_logger(base_dir: Optional[str] = None) -> InteractionLogger:
    """Return the process-wide InteractionLogger, creating it on first use."""
    global _logger
    if _logger is None or base_dir is not None:
        _logger = InteractionLogger(base_dir or "./designlora_logs")
    return _logger
