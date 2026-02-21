"""
Process History Tracker
======================

Tracks executed processes/tasks during a session for project history display.
Allows users to see what analyses and operations have been performed when opening saved projects.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import threading

logger = logging.getLogger(__name__)


@dataclass
class ProcessExecution:
    """
    Records a single process execution.

    Attributes:
        task_name: Name of the task (e.g., "kriging", "sgsim", "variogram")
        timestamp: When the process was executed
        parameters: Parameters used (sanitized for display)
        status: "success", "failed", or "running"
        duration_seconds: How long it took (None if still running)
        error_message: Error message if failed
        result_summary: Brief summary of results (e.g., "Generated 150,000 blocks")
    """
    task_name: str
    timestamp: datetime
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: str = "running"  # "running", "success", "failed"
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    result_summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_name": self.task_name,
            "timestamp": self.timestamp.isoformat(),
            "parameters": self._sanitize_parameters(self.parameters),
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "result_summary": self.result_summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessExecution":
        """Create from dictionary (for deserialization)."""
        return cls(
            task_name=data["task_name"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            parameters=data.get("parameters", {}),
            status=data.get("status", "success"),
            duration_seconds=data.get("duration_seconds"),
            error_message=data.get("error_message"),
            result_summary=data.get("result_summary"),
        )

    @staticmethod
    def _sanitize_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize parameters for safe storage/display."""
        sanitized = {}
        for key, value in params.items():
            # Skip sensitive/private parameters
            if key.startswith("_") or key in ["password", "token", "api_key"]:
                continue

            # Convert complex objects to strings
            if hasattr(value, "__name__"):
                sanitized[key] = value.__name__
            elif hasattr(value, "__class__"):
                sanitized[key] = f"{value.__class__.__name__} instance"
            else:
                try:
                    # Try to convert to string representation
                    str_val = str(value)
                    # Limit length to prevent huge parameter storage
                    if len(str_val) > 500:
                        sanitized[key] = str_val[:500] + "..."
                    else:
                        sanitized[key] = value
                except Exception:
                    sanitized[key] = f"<{type(value).__name__}>"
        return sanitized


class ProcessHistoryTracker:
    """
    Thread-safe tracker for process execution history.

    Maintains a chronological list of all processes executed during a session.
    Can be serialized/deserialized for project files.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._history: List[ProcessExecution] = []
        self._active_processes: Dict[str, ProcessExecution] = {}
        logger.info("ProcessHistoryTracker initialized")

    def start_process(self, task_name: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Record the start of a process execution.

        Args:
            task_name: Name of the task being executed
            parameters: Parameters for the process

        Returns:
            Process ID for tracking completion
        """
        with self._lock:
            execution = ProcessExecution(
                task_name=task_name,
                timestamp=datetime.now(),
                parameters=parameters or {},
                status="running"
            )

            process_id = f"{task_name}_{execution.timestamp.isoformat()}"
            self._active_processes[process_id] = execution
            self._history.append(execution)

            logger.info(f"Started process: {task_name}")
            return process_id

    def complete_process(self, process_id: str, success: bool = True,
                        result_summary: Optional[str] = None,
                        error_message: Optional[str] = None) -> None:
        """
        Record the completion of a process.

        Args:
            process_id: Process ID returned from start_process
            success: Whether the process completed successfully
            result_summary: Brief summary of results
            error_message: Error message if failed
        """
        with self._lock:
            if process_id not in self._active_processes:
                logger.warning(f"Unknown process ID: {process_id}")
                return

            execution = self._active_processes.pop(process_id)

            # Calculate duration
            duration = (datetime.now() - execution.timestamp).total_seconds()
            execution.duration_seconds = duration

            if success:
                execution.status = "success"
                execution.result_summary = result_summary
                logger.info(f"Completed process: {execution.task_name} ({duration:.1f}s)")
            else:
                execution.status = "failed"
                execution.error_message = error_message
                logger.warning(f"Failed process: {execution.task_name} ({duration:.1f}s): {error_message}")

    def get_history(self, limit: Optional[int] = None) -> List[ProcessExecution]:
        """
        Get the process execution history.

        Args:
            limit: Maximum number of recent executions to return

        Returns:
            List of process executions (most recent first)
        """
        with self._lock:
            history = self._history.copy()
            if limit:
                history = history[-limit:]
            return history

    def get_process_count(self) -> int:
        """Get total number of completed processes."""
        with self._lock:
            return len([p for p in self._history if p.status in ["success", "failed"]])

    def get_successful_processes(self) -> List[ProcessExecution]:
        """Get all successfully completed processes."""
        with self._lock:
            return [p for p in self._history if p.status == "success"]

    def get_failed_processes(self) -> List[ProcessExecution]:
        """Get all failed processes."""
        with self._lock:
            return [p for p in self._history if p.status == "failed"]

    def clear_history(self) -> None:
        """Clear all process history."""
        with self._lock:
            self._history.clear()
            self._active_processes.clear()
            logger.info("Process history cleared")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        with self._lock:
            return {
                "version": 1,
                "history": [execution.to_dict() for execution in self._history]
            }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load from dictionary (for deserialization)."""
        with self._lock:
            version = data.get("version", 1)
            if version == 1:
                self._history = [
                    ProcessExecution.from_dict(item)
                    for item in data.get("history", [])
                ]
                logger.info(f"Loaded {len(self._history)} process executions from saved project")
            else:
                logger.warning(f"Unknown process history version: {version}")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the process history."""
        with self._lock:
            total = len(self._history)
            successful = len([p for p in self._history if p.status == "success"])
            failed = len([p for p in self._history if p.status == "failed"])
            running = len([p for p in self._history if p.status == "running"])

            # Calculate total runtime for successful processes
            total_runtime = sum(
                p.duration_seconds or 0
                for p in self._history
                if p.status == "success" and p.duration_seconds is not None
            )

            return {
                "total_processes": total,
                "successful": successful,
                "failed": failed,
                "running": running,
                "success_rate": successful / max(total, 1) * 100,
                "total_runtime_seconds": total_runtime,
                "most_recent": self._history[-1].task_name if self._history else None
            }


# Global singleton instance
_instance: Optional[ProcessHistoryTracker] = None
_instance_lock = threading.Lock()


def get_process_history_tracker() -> ProcessHistoryTracker:
    """Get the global process history tracker instance."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ProcessHistoryTracker()
    return _instance
