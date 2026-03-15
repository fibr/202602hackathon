"""Tests for rig_lock: file-based advisory lock for robot rig."""

import json
import os
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from rig_lock import RigLock, RigLockError, LockInfo, _pid_alive, _lock_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _TestLock(RigLock):
    """RigLock subclass that uses a custom temp-dir for the lock file."""

    def __init__(self, lock_dir, **kwargs):
        super().__init__(**kwargs)
        self._path = lock_dir / f"robot_rig_{self._host}.lock"


@pytest.fixture()
def lock_dir(tmp_path):
    """Provide a temp directory for lock files."""
    return tmp_path


@pytest.fixture()
def make_lock(lock_dir):
    """Factory fixture: returns a _TestLock bound to the temp dir."""
    def _make(**kwargs):
        return _TestLock(lock_dir, **kwargs)
    return _make


# ---------------------------------------------------------------------------
# LockInfo
# ---------------------------------------------------------------------------

class TestLockInfo:
    def test_roundtrip_json(self):
        info = LockInfo(pid=42, cwd="/tmp/test", hostname="host1",
                        timestamp=1000.0, holder="task-abc")
        text = info.to_json()
        restored = LockInfo.from_json(text)
        assert restored.pid == 42
        assert restored.cwd == "/tmp/test"
        assert restored.hostname == "host1"
        assert restored.holder == "task-abc"

    def test_age_seconds(self):
        info = LockInfo(pid=1, cwd="/", hostname="h",
                        timestamp=time.time() - 10.0)
        assert 9.0 < info.age_seconds < 12.0

    def test_is_alive_current_pid(self):
        info = LockInfo(pid=os.getpid(), cwd="/", hostname=os.uname().nodename,
                        timestamp=time.time())
        assert info.is_alive()

    def test_is_alive_dead_pid(self):
        info = LockInfo(pid=999999999, cwd="/", hostname=os.uname().nodename,
                        timestamp=time.time())
        assert not info.is_alive()


# ---------------------------------------------------------------------------
# _pid_alive
# ---------------------------------------------------------------------------

class TestPidAlive:
    def test_current_process(self):
        assert _pid_alive(os.getpid())

    def test_dead_process(self):
        assert not _pid_alive(999999999)

    def test_invalid_pid(self):
        assert not _pid_alive(0)
        assert not _pid_alive(-1)


# ---------------------------------------------------------------------------
# RigLock — acquire / release
# ---------------------------------------------------------------------------

class TestRigLock:
    def test_acquire_and_release(self, make_lock):
        lock = make_lock(holder="test")
        info = lock.acquire()
        assert info.pid == os.getpid()
        assert info.holder == "test"
        assert lock._path.exists()

        assert lock.release()
        assert not lock._path.exists()

    def test_reentrant_acquire(self, make_lock):
        lock = make_lock()
        lock.acquire()
        # Second acquire by same PID should succeed silently
        info = lock.acquire()
        assert info.pid == os.getpid()
        lock.release()

    def test_contention_raises(self, make_lock, lock_dir):
        """Two different locks (simulating different PIDs) should conflict."""
        lock1 = make_lock(holder="agent-1")
        lock1.acquire()

        # Simulate another process by writing a lock with a different (alive) PID
        # Use PID 1 (init/systemd) which is always alive
        fake_info = LockInfo(pid=1, cwd="/other", hostname=lock1._host,
                             timestamp=time.time(), holder="agent-2")
        lock1._path.write_text(fake_info.to_json())

        lock2 = make_lock(holder="agent-3")
        with pytest.raises(RigLockError) as exc_info:
            lock2.acquire()
        assert "locked by PID 1" in str(exc_info.value)

    def test_stale_lock_reclaimed(self, make_lock):
        """A lock held by a dead PID should be automatically reclaimed."""
        lock = make_lock()
        # Write a stale lock (non-existent PID)
        stale = LockInfo(pid=999999999, cwd="/dead", hostname=lock._host,
                         timestamp=time.time() - 3600, holder="dead-agent")
        lock._path.write_text(stale.to_json())

        # New acquire should succeed (stale reclaim)
        info = lock.acquire()
        assert info.pid == os.getpid()

    def test_force_acquire(self, make_lock):
        """force=True should steal the lock even from a live process."""
        lock = make_lock()
        # Simulate lock held by PID 1 (always alive)
        live = LockInfo(pid=1, cwd="/other", hostname=lock._host,
                        timestamp=time.time(), holder="live-agent")
        lock._path.write_text(live.to_json())

        info = lock.acquire(force=True)
        assert info.pid == os.getpid()

    def test_release_without_acquire(self, make_lock):
        lock = make_lock()
        assert not lock.release()  # Should return False

    def test_context_manager(self, make_lock):
        lock = make_lock()
        with lock:
            assert lock._path.exists()
            info = lock.read()
            assert info.pid == os.getpid()
        # After exiting, lock file should be gone
        assert not lock._path.exists()

    def test_read_no_lockfile(self, make_lock):
        lock = make_lock()
        assert lock.read() is None

    def test_is_locked_property(self, make_lock):
        lock = make_lock()
        assert not lock.is_locked
        lock.acquire()
        assert lock.is_locked
        lock.release()
        assert not lock.is_locked


# ---------------------------------------------------------------------------
# Class-level helpers
# ---------------------------------------------------------------------------

class TestClassHelpers:
    def test_status_unlocked(self, make_lock):
        lock = make_lock()
        # Use the lock's path directly
        status = f"Rig is UNLOCKED (no lock file at {lock._path})"
        info = lock.read()
        assert info is None

    def test_force_release_no_file(self, lock_dir):
        """force_release on non-existent file should report cleanly."""
        from pathlib import Path
        path = lock_dir / "robot_rig_nonexistent.lock"
        assert not path.exists()
        # Just verify the class method doesn't crash
        result = RigLock.force_release(host="nonexistent")
        assert "No lock file" in result or "removed" in result
