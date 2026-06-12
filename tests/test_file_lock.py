"""
Tests for inter-process file locking around shelve.open and compact.

dbm.open() determines the database type by reading the file with no locking
at all (dbm.whichdb), so a concurrent reader can observe a half-created
database -- e.g. a 0-byte file between gdbm's open(O_CREAT) and its header
write, or the window where compact() has renamed the file away and is
recreating it -- and fail with "db type could not be determined".

memoshelve guards against this by taking an exclusive flock on a sidecar
``<filename>.lock`` file around every shelve.open and for the duration of
compact()/upgrade().  These tests simulate the "creation in progress" state
deterministically by holding that lock from another process.
"""

import multiprocessing
import os
import shelve
import sys
import time

import pytest

from memoshelve import compact, lazy_shelve_open

if sys.platform != "win32":
    import fcntl

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="flock-based locking is POSIX-only"
)


def _lock_path(filename):
    return str(filename) + ".lock"


def _hold_lock_with_half_created_db(filename, hold_seconds, started):
    """Simulate a database mid-creation: hold the sidecar lock while the db
    file exists but is 0 bytes (no header yet), then finish creating a valid
    db and release the lock."""
    fd = os.open(_lock_path(filename), os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    with open(filename, "wb"):
        pass  # 0 bytes: the state whichdb cannot identify
    started.set()
    time.sleep(hold_seconds)
    os.remove(filename)
    with shelve.open(filename) as db:
        db["seeded"] = True
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def _hold_lock(filename, hold_seconds, started):
    fd = os.open(_lock_path(filename), os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    started.set()
    time.sleep(hold_seconds)
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def _hold_db_open(filename, hold_seconds, started):
    with shelve.open(filename) as db:
        db["held"] = True
        started.set()
        time.sleep(hold_seconds)


@pytest.mark.parametrize("eager", [False, True])
def test_open_waits_for_in_progress_creation(tmp_path, eager):
    """Opening must wait for the lock instead of reading a half-created file
    and dying with 'db type could not be determined'."""
    filename = str(tmp_path / "cache.shelve")
    started = multiprocessing.Event()
    p = multiprocessing.Process(
        target=_hold_lock_with_half_created_db, args=(filename, 1.0, started)
    )
    p.start()
    try:
        assert started.wait(10)
        t0 = time.monotonic()
        with lazy_shelve_open(filename, eager=eager) as get_db:
            with get_db() as db:
                assert db["seeded"] is True
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.4, "open should have blocked on the lock"
    finally:
        p.join()


def test_compact_waits_for_lock(tmp_path):
    """compact() must hold the inter-process lock for its remove-and-recreate,
    and therefore wait if someone else holds it."""
    filename = str(tmp_path / "cache.shelve")
    with shelve.open(filename) as db:
        db["k"] = 1
    started = multiprocessing.Event()
    p = multiprocessing.Process(target=_hold_lock, args=(filename, 1.0, started))
    p.start()
    try:
        assert started.wait(10)
        t0 = time.monotonic()
        compact(filename)
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.4, "compact should have blocked on the lock"
    finally:
        p.join()
    with shelve.open(filename) as db:
        assert db["k"] == 1


def test_no_deadlock_when_db_held_open(tmp_path):
    """The lock covers only the open itself, so a process that keeps the
    shelve open (holding gdbm's own write lock) must not deadlock others."""
    filename = str(tmp_path / "cache.shelve")
    with shelve.open(filename) as db:
        db["k"] = 1
    started = multiprocessing.Event()
    p = multiprocessing.Process(target=_hold_db_open, args=(filename, 1.0, started))
    p.start()
    try:
        assert started.wait(10)
        with lazy_shelve_open(filename) as get_db:
            with get_db() as db:
                assert db["k"] == 1
    finally:
        p.join()
