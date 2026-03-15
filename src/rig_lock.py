"""Cross-workbench locking for physical robot rig (arm + cameras).

Only one process on a given host may control the physical hardware at a time.
The lock is a JSON file in /tmp keyed by hostname, containing the holder's PID,
working directory, and timestamp.  Stale locks (dead PIDs) are automatically
reclaimed.

Optional heartbeat mechanism
-----------------------------
For long-running agents that might hang without dying, a heartbeat thread can
periodically update the lock timestamp. Other agents detect staleness if:
  1. The PID is alive, AND
  2. The lock hasn't been heartbeated in N seconds (heartbeat_timeout)

Usage as context manager with heartbeat
----------------------------------------
    from rig_lock import RigLock

    with RigLock(enable_heartbeat=True, heartbeat_interval=30) as lock:
        robot = connect_robot(config)
        camera = create_camera(config)
        ...  # exclusive hardware access
        # Heartbeat is automatically started and stopped

Usage with explicit acquire/release
------------------------------------
    lock = RigLock()
    lock.acquire()        # raises RigLockError if held by another live process
    lock.start_heartbeat(interval=30)  # optional: start heartbeat thread
    try:
        ...
    finally:
        lock.stop_heartbeat()  # clean shutdown of heartbeat
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
import threading
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
    heartbeat_timeout: float = 0.0  # seconds; 0 = no heartbeat monitoring

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "LockInfo":
        data = json.loads(text)
        # Handle backward compatibility: old locks won't have heartbeat_timeout
        if "heartbeat_timeout" not in data:
            data["heartbeat_timeout"] = 0.0
        return cls(**data)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def is_alive(self) -> bool:
        """True if the holder process is still running on this host."""
        if self.hostname != socket.gethostname():
            # Different host — can't check PID; assume alive.
            return True
        return _pid_alive(self.pid)

    def is_heartbeat_stale(self) -> bool:
        """True if heartbeat has timed out (PID alive but no recent heartbeat).

        Returns False if:
          - No heartbeat timeout configured (heartbeat_timeout == 0)
          - Age is less than timeout (heartbeat is fresh)

        Returns True if:
          - Heartbeat timeout is configured, AND
          - Lock age exceeds the timeout
        """
        if self.heartbeat_timeout <= 0:
            return False
        return self.age_seconds > self.heartbeat_timeout


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
        enable_heartbeat: If True, start a heartbeat thread in __enter__.
        heartbeat_interval: Seconds between heartbeat updates (default: 30).
        heartbeat_timeout: Seconds before lock is considered stale if no heartbeat
                          (default: 300, i.e. 5 minutes).
    """

    def __init__(
        self,
        holder: str = "",
        host: Optional[str] = None,
        enable_heartbeat: bool = False,
        heartbeat_interval: float = 30.0,
        heartbeat_timeout: float = 300.0,
    ):
        self.holder = holder or os.environ.get("FORGE_TASK_ID", "")
        self._host = host or socket.gethostname()
        self._path = _lock_path(self._host)
        self._owned = False
        self._enable_heartbeat = enable_heartbeat
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop = threading.Event()

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
            # Check both: PID alive AND heartbeat not stale
            pid_alive = existing.is_alive()
            hb_stale = existing.is_heartbeat_stale()
            if pid_alive and hb_stale and not force:
                # PID is alive but heartbeat has timed out
                raise RigLockError(
                    LockInfo(
                        pid=existing.pid,
                        cwd=existing.cwd,
                        hostname=existing.hostname,
                        timestamp=existing.timestamp,
                        holder=existing.holder,
                        heartbeat_timeout=existing.heartbeat_timeout,
                    )
                )
            if pid_alive and not hb_stale and not force:
                # PID is alive and heartbeat is fresh (or no heartbeat configured)
                raise RigLockError(
                    LockInfo(
                        pid=existing.pid,
                        cwd=existing.cwd,
                        hostname=existing.hostname,
                        timestamp=existing.timestamp,
                        holder=existing.holder,
                        heartbeat_timeout=existing.heartbeat_timeout,
                    )
                )
            else:
                if hb_stale:
                    reason = "forced" if force else "stale (heartbeat timed out)"
                else:
                    reason = "forced" if force else "stale (PID dead)"
                log.warning(
                    "Reclaiming %s lock from PID %d (cwd=%s, age=%ds).",
                    reason,
                    existing.pid,
                    existing.cwd,
                    int(existing.age_seconds),
                )

        # Create new lock with heartbeat timeout if enabled
        info = LockInfo(
            pid=os.getpid(),
            cwd=os.getcwd(),
            hostname=self._host,
            timestamp=time.time(),
            holder=self.holder,
            heartbeat_timeout=self._heartbeat_timeout if self._enable_heartbeat else 0.0,
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
        # Stop heartbeat first
        self.stop_heartbeat()

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

    def start_heartbeat(self, interval: Optional[float] = None) -> None:
        """Start a background thread that periodically updates the lock timestamp.

        This allows other agents to detect if this process is hung (PID alive but
        no heartbeat) after heartbeat_timeout seconds.

        Args:
            interval: Override the heartbeat interval (seconds). If None, uses
                     the value from __init__.
        """
        if self._heartbeat_thread is not None:
            log.debug("Heartbeat already running.")
            return
        if not self._owned:
            log.warning("Cannot start heartbeat: lock not acquired.")
            return

        interval = interval or self._heartbeat_interval
        self._heartbeat_stop.clear()

        def _heartbeat_worker():
            """Background worker that periodically updates the lock timestamp."""
            while not self._heartbeat_stop.wait(interval):
                try:
                    existing = self.read()
                    if existing is None or existing.pid != os.getpid():
                        # Lock was released or stolen; stop heartbeat
                        log.warning("Heartbeat: lock not owned by this process, stopping.")
                        break
                    # Update timestamp to signal we're still alive
                    existing.timestamp = time.time()
                    self._path.write_text(existing.to_json())
                    log.debug("Heartbeat updated: age=0s")
                except Exception as e:
                    log.error("Heartbeat worker error: %s", e)

        self._heartbeat_thread = threading.Thread(
            target=_heartbeat_worker, daemon=False, name="RigLockHeartbeat"
        )
        self._heartbeat_thread.start()
        log.info("Heartbeat started (interval=%.1fs, timeout=%.1fs).",
                 interval, self._heartbeat_timeout)

    def stop_heartbeat(self) -> None:
        """Stop the heartbeat thread if it's running.

        Safe to call even if heartbeat was never started.
        Waits up to 2 seconds for the thread to shut down cleanly.
        """
        if self._heartbeat_thread is None:
            return
        self._heartbeat_stop.set()
        self._heartbeat_thread.join(timeout=2.0)
        if self._heartbeat_thread.is_alive():
            log.warning("Heartbeat thread did not shut down cleanly.")
        else:
            log.debug("Heartbeat stopped.")
        self._heartbeat_thread = None

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
        if self._enable_heartbeat:
            self.start_heartbeat()
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
