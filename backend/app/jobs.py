import threading
from typing import Callable


class Job:
    """Tracks progress of a long-running batch operation run in a background thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self.status = "idle"  # idle | running | done
        self.total = 0
        self.done = 0
        self.failed = 0

    def start(self, run_fn: Callable[["Job"], None]) -> bool:
        """Start run_fn(self) in a background thread. Returns False if already running."""
        with self._lock:
            if self.status == "running":
                return False
            self.status = "running"
            self.total = 0
            self.done = 0
            self.failed = 0

        def _run():
            try:
                run_fn(self)
            finally:
                with self._lock:
                    self.status = "done"

        threading.Thread(target=_run, daemon=True).start()
        return True

    def set_total(self, total: int) -> None:
        with self._lock:
            self.total = total

    def increment(self, *, failed: bool = False) -> None:
        with self._lock:
            self.done += 1
            if failed:
                self.failed += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "total": self.total,
                "done": self.done,
                "failed": self.failed,
            }
