"""Cross-workbench locking for physical robot rig (arm + cameras).

Only one process on a given host may control the physical hardware at a time.
The lock is a JSON file in /tmp keyed by hostname, containing the holder's PID,
working directory, and timestamp.  Stale locks (dead PIDs) are automatically
reclaimed.

Usage as context manager
------------------------
    from rig_lock import RigLock

    with RigLock() as lock:
        robot = connect_robot(config)
        camera = create_camera(config)
        ...  # exclusive hardware access

Usage with explicit acquire/release
------------------------------------
    lock = RigLock()
    lock.acquire()        # raises RigLockError if held by another live process
    try:
        ...
    finally:
        lock.release()

CLI
---
    python -m rig_lock status   # show current lock state
    python -m rig_lock release  # force-release (e.g. after crash)
"""

from __future__ import annotations

import json
import os
import signal
import socket
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lock file location: /tmp/robot_rig_<hostname>.lock
# Using /tmp ensures visibility across all workbenches on the same host.
# ---------------------------------------------------------------------------
_LOCK_DIR = Path("/tmp")


def _lock_path(host: Optional[str] = None) -> Path:
    """Return the lock file path for the given (or current) hostname."""
    host = host or socket.gethostname()
    return _LOCK_DIR / f"robot_rig_{host}.lock"


def _pid_alive(pid: int) -> bool:
    """Check whether a process with the given PID is still running."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)  # signal 0 = existence check, no actual signal
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it (different user) — treat as alive.
        return True


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class LockInfo:
    """Metadata stored inside the lock file."""

    pid: int
    cwd: str
    hostname: str
    timestamp: float
    holder: str = ""  # optional human-readable label (e.g. Forge task ID)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "LockInfo":
        return cls(**json.loads(text))

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def is_alive(self) -> bool:
        """True if the holder process is still running on this host."""
        if self.hostname != socket.gethostname():
            # Different host — can't check PID; assume alive.
            return True
        return _pid_alive(self.pid)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class RigLockError(RuntimeError):
    """Raised when the rig lock cannot be acquired."""

    def __init__(self, info: LockInfo):
        self.info = info
        age = int(info.age_seconds)
        super().__init__(
            f"Robot rig is locked by PID {info.pid} "
            f"(cwd={info.cwd}, holder={info.holder!r}, age={age}s). "
            f"Lock file: {_lock_path(info.hostname)}"
        )


# ---------------------------------------------------------------------------
# Main lock class
# ---------------------------------------------------------------------------
class RigLock:
    """File-based advisory lock for the physical robot rig.

    Args:
        holder: Optional label identifying who holds the lock
                (e.g. Forge task ID, user name).
        host:   Override hostname (for testing).
    """

    def __init__(self, holder: str = "", host: Optional[str] = None):
        self.holder = holder or os.environ.get("FORGE_TASK_ID", "")
        self._host = host or socket.gethostname()
        self._path = _lock_path(self._host)
        self._owned = False

    # -- public API ---------------------------------------------------------

    def acquire(self, force: bool = False) -> LockInfo:
        """Acquire the rig lock.

        Args:
            force: If True, steal the lock even if the holder is alive.

        Returns:
            LockInfo for the new lock.

        Raises:
            RigLockError: If the lock is held by another live process
                          and force is False.
        """
        existing = self.read()
        if existing is not None:
            if existing.pid == os.getpid():
                log.debug("Lock already held by this process (re-entrant).")
                self._owned = True
                return existing
            if existing.is_alive() and not force:
                raise RigLockError(existing)
            else:
                reason = "forced" if force else "stale (PID dead)"
                log.warning(
                    "Reclaiming %s lock from PID %d (cwd=%s, age=%ds).",
                    reason,
                    existing.pid,
                    existing.cwd,
                    int(existing.age_seconds),
                )

        info = LockInfo(
            pid=os.getpid(),
            cwd=os.getcwd(),
            hostname=self._host,
            timestamp=time.time(),
            holder=self.holder,
        )
        self._path.write_text(info.to_json())
        self._owned = True
        log.info("Rig lock acquired: PID %d, holder=%r", info.pid, info.holder)
        return info

    def release(self) -> bool:
        """Release the lock if we own it.

        Returns:
            True if the lock was released, False if we didn't own it.
        """
        if not self._owned:
            return False
        existing = self.read()
        if existing is not None and existing.pid == os.getpid():
            try:
                self._path.unlink()
                log.info("Rig lock released.")
            except FileNotFoundError:
                pass
        self._owned = False
        return True

    def read(self) -> Optional[LockInfo]:
        """Read the current lock state without modifying it.

        Returns:
            LockInfo if a lock file exists, None otherwise.
        """
        try:
            text = self._path.read_text()
            return LockInfo.from_json(text)
        except (FileNotFoundError, json.JSONDecodeError, TypeError, KeyError):
            return None

    @property
    def is_locked(self) -> bool:
        """True if the rig is currently locked by a live process."""
        info = self.read()
        return info is not None and info.is_alive()

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> "RigLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    # -- class-level helpers ------------------------------------------------

    @classmethod
    def status(cls, host: Optional[str] = None) -> str:
        """Return a human-readable status string."""
        lock = cls(host=host)
        info = lock.read()
        if info is None:
            return f"Rig is UNLOCKED (no lock file at {lock._path})"
        alive = info.is_alive()
        state = "LOCKED (holder alive)" if alive else "STALE (holder dead)"
        age = int(info.age_seconds)
        return (
            f"Rig is {state}\n"
            f"  PID:       {info.pid}\n"
            f"  CWD:       {info.cwd}\n"
            f"  Holder:    {info.holder or '(none)'}\n"
            f"  Hostname:  {info.hostname}\n"
            f"  Age:       {age}s\n"
            f"  Lock file: {lock._path}"
        )

    @classmethod
    def force_release(cls, host: Optional[str] = None) -> str:
        """Force-remove the lock file regardless of holder state."""
        path = _lock_path(host)
        try:
            info_text = path.read_text()
            path.unlink()
            return f"Lock file removed: {path}\nPrevious contents:\n{info_text}"
        except FileNotFoundError:
            return f"No lock file found at {path}"


# ---------------------------------------------------------------------------
# CLI entry point: python -m rig_lock {status|release}
# ---------------------------------------------------------------------------
def _cli():
    import sys

    usage = "Usage: python -m rig_lock {status|release}"
    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "status":
        print(RigLock.status())
    elif cmd == "release":
        print(RigLock.force_release())
    else:
        print(f"Unknown command: {cmd!r}\n{usage}")
        sys.exit(1)


if __name__ == "__main__":
    _cli()
