try:
    import _gdbm
except ImportError:
    _gdbm = None

import inspect
import logging
import os
import shelve
import time
import traceback
from contextlib import asynccontextmanager, contextmanager
from functools import partial, wraps
from pathlib import Path
from pickle import UnpicklingError
from typing import Any, Callable, Collection, Literal, Optional, TypeVar

logger = logging.getLogger(__name__)

try:
    import dill

    # monkeypatch shelve as per https://stackoverflow.com/q/52927236/377022
    shelve.Pickler = dill.Pickler  # type: ignore
    shelve.Unpickler = dill.Unpickler  # type: ignore
except ImportError as e:
    logger.warning(
        f"Dill not found; some arguments may raise errors when passed to cached functions: {e}"
    )
    dill = None

try:
    import stablehash
except ImportError as e:
    logger.warning(f"stablehash not found, falling back to repr: {e}")
    stablehash = None

__all__ = [
    "compact",
    "memoshelve",
    "uncache",
    "cache",
    "CacheStatus",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_HASH",
    "__version__",
]


def hash_via_stablehash(obj: object) -> str:
    assert stablehash is not None
    return stablehash.stablehash(obj).hexdigest()


__version__ = "1.0.3"


class _gdbm_dummy_error(Exception):
    pass


memoshelve_cache: dict[str, dict[str, Any]] = {}
T = TypeVar("T")

DEFAULT_PRINT_MEM_CACHE_MISS = False
DEFAULT_PRINT_MEM_CACHE_HIT = False
DEFAULT_PRINT_DISK_CACHE_MISS = True
DEFAULT_PRINT_DISK_CACHE_HIT = False
DEFAULT_PRINT_CACHE_MISS_FN = logger.warning
DEFAULT_PRINT_CACHE_HIT_FN = logger.debug
DEFAULT_CACHE_DIR = (
    Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "memoshelve"
)
DEFAULT_HASH = hash_via_stablehash if stablehash is not None else repr  # type: ignore


def next_backup_ext(ext: str, strip_suffix: bool | None = None) -> tuple[str, bool]:
    """Generate the next backup extension by incrementing the number if present."""
    if len(ext) > 1 and ext[1:].isdigit():
        return ext[0] + str(int(ext[1:]) + 1), True
    elif strip_suffix:
        return ext + ext, strip_suffix
    else:
        return ext, strip_suffix or False


def backup_file(
    filename: str | Path, ext: str = ".bak", *, strip_suffix: bool | None = None
) -> Optional[Path]:
    filename = Path(filename)
    assert ext != ""
    if strip_suffix is None:
        strip_suffix = ext[1:].isdigit()
    new_suffix = ext if strip_suffix else filename.suffix + ext
    backup_name = filename.with_suffix(new_suffix)
    assert (
        backup_name != filename
    ), f"backup_file({filename!r}, ext={ext!r}, strip_suffix={strip_suffix!r})"
    if filename.exists():
        if backup_name.exists():
            next_ext, strip_suffix = next_backup_ext(ext, strip_suffix=strip_suffix)
            backup_file(backup_name, ext=next_ext, strip_suffix=strip_suffix)
            assert not backup_name.exists()
        filename.rename(backup_name)
        return backup_name
    return None


def compact(
    filename: Path | str, backup: bool = True, *, remove_on_unknown_type: bool = False
):
    """Compact a shelve database by removing corrupted entries.

    This function reads all entries from a shelve database, backs up the original
    file (if requested), recreates the database with only valid entries, and
    removes the backup if successful.

    Args:
        filename: Path to the shelve database file to compact
        backup: Whether to create a backup before compacting (default: True)

    Raises:
        UnpicklingError: Logged as warning for corrupted entries (entries are skipped)
        Various IO errors: From file operations
    """
    save_backup = False
    entries = {}
    try:
        with shelve.open(filename) as db:
            for k in db.keys():
                try:
                    entries[k] = db[k]
                except UnpicklingError:
                    logger.warning(f"UnpicklingError for {k} in {filename}")
    except getattr(_gdbm, "error", _gdbm_dummy_error) as e:  # handle recovery
        if not (
            e.args
            and e.args[0] == "db type could not be determined"
            and remove_on_unknown_type
        ):
            raise e
        backup = True
        save_backup = True
        logger.warning(
            f"DB type could not be determined ({e}), removing {filename} and creating a new one"
        )
    if backup:
        backup_name = backup_file(filename)
    else:
        backup_name = None
        os.remove(filename)
    with shelve.open(filename) as db:
        for k in entries.keys():
            db[k] = entries[k]
    if backup_name and not save_backup:
        assert backup_name != filename, backup_name
        os.remove(backup_name)


def upgrade_value(value: Any) -> Optional[dict]:
    if isinstance(value, tuple) and len(value) == 3:
        result, args, kwargs = value
        return {
            "result": result,
            "version": __version__,
            "args": args,
            "kwargs": kwargs,
        }
    elif (
        isinstance(value, dict)
        and "result" in value
        and "args" in value
        and "kwargs" in value
    ):
        return value
    else:
        return None


def upgrade_value_or_raise(value: Any, exn: Exception = KeyError()) -> dict[str, Any]:
    new_value = upgrade_value(value)
    if new_value is not None:
        return new_value
    else:
        raise exn


def upgrade(
    filename: Path | str,
    new_hash: Callable | None = None,
    *,
    backup: bool = True,
    remove_backup_on_failure: bool = False,
):
    """Upgrade a shelve database by rehashing keys with a new hash function.

    This function reads all entries from a shelve database, computes new keys using
    the provided hash function based on stored args and kwargs, backs up the original
    file (if requested), recreates the database with updated keys, and removes the
    backup if successful. Entries without stored args/kwargs are skipped.

    Also upgrades from previous formats (e.g., tuple) to new format (dict).

    Args:
        filename: Path to the shelve database file to upgrade
        new_hash: The new hash function to use for generating keys
        backup: Whether to create a backup before upgrading (default: True)

    Raises:
        UnpicklingError: Logged as warning for corrupted entries (entries are skipped)
        Various IO errors: From file operations
    """
    filename = Path(filename)
    with shelve.open(filename) as db:
        old_entries = dict(db.items())

    entries = {}
    remove_backup = True
    for k, stored in old_entries.items():
        new_value = upgrade_value(stored)
        if new_value is not None:
            new_key = (
                str(new_hash((stored["args"], stored["kwargs"]))) if new_hash else k
            )
            entries[new_key] = new_value
        else:
            remove_backup = remove_backup_on_failure
            logger.warning(
                f"Skipping non-upgradable entry {k} with value {stored} in {filename}"
            )

    if backup:
        backup_name = backup_file(filename)
    else:
        backup_name = None
        if filename.exists():
            os.remove(filename)
    with shelve.open(filename) as db:
        for k, v in entries.items():
            db[k] = v
    if backup_name and remove_backup:
        os.remove(backup_name)


@contextmanager
def lazy_shelve_open(filename: Path | str, *, eager: bool = False):
    """Context manager for lazy shelve database opening with retry logic.

    Provides a context manager that returns a function to open shelve databases.
    In eager mode, opens the database immediately. In lazy mode (default),
    opens the database only when needed and retries on temporary failures.

    Args:
        filename: Path to the shelve database file
        eager: If True, opens the database immediately. If False (default),
               opens lazily with retry logic for temporary failures.

    Yields:
        A context manager function that yields the opened shelve database

    Example:
        ```python
        with lazy_shelve_open("cache.db") as get_db:
            with get_db() as db:
                db["key"] = "value"
        ```
    """
    if eager:
        with shelve.open(filename) as db:

            @contextmanager
            def get_db():
                yield db

            get_db.eager = eager
            yield get_db
    else:

        @contextmanager
        def get_db():
            sh = None
            while sh is None:
                try:
                    sh = shelve.open(filename)
                except Exception as e:
                    if e.args == (11, "Resource temporarily unavailable"):
                        time.sleep(0.1)
                    else:
                        if len(e.args) == 1 and isinstance(e.args[0], str):
                            e.args = (e.args[0] + f" ({filename})",)
                        else:
                            e.args = (*e.args, filename)
                        raise e
            with sh as db:
                yield db

        get_db.eager = eager
        yield get_db


CacheStatus = Literal["cached (mem)", "cached (disk)", "miss"]


class MemoshelveMetadata:
    def __init__(
        self,
        filename: Path | str,
        *,
        get_hash: Callable,
        get_hash_mem: Callable,
        get_db: Callable,
        mem_db: dict[str, dict[str, Any]],
    ):
        self.filename = filename
        self.get_hash = get_hash
        self.get_hash_mem = get_hash_mem
        self._get_db = get_db
        self._mem_db = mem_db

    def disk_keys(self) -> set[str]:
        with self._get_db() as db:
            return set(db.keys())

    def disk_raw_items(self):
        with self._get_db() as db:
            return list(db.items())

    def disk_raw_values(self):
        with self._get_db() as db:
            return list(db.values())

    def disk_items(self):
        return [(k, v["result"]) for k, v in self.disk_raw_items()]

    def disk_values(self):
        return [v["result"] for v in self.disk_raw_values()]

    def mem_keys(self) -> set[str]:
        return set(self._mem_db.keys())

    def mem_raw_items(self):
        return self._mem_db.items()

    def mem_raw_values(self):
        return self._mem_db.values()

    def mem_values(self):
        return [v["result"] for v in self.mem_raw_values()]

    def mem_items(self):
        return [(k, v["result"]) for k, v in self.mem_raw_items()]

    def keys(self):
        return set(self.disk_keys()) | set(self.mem_keys())

    def raw_items(self):
        return (dict(self.disk_raw_items()) | self._mem_db).items()

    def raw_values(self):
        return (dict(self.disk_raw_values()) | self._mem_db).values()

    def items(self):
        return (dict(self.disk_items()) | dict(self.mem_items())).items()

    def values(self):
        return (dict(self.disk_items()) | dict(self.mem_items())).values()

    def compact(self, backup: bool = True):
        compact(self.filename, backup=backup)

    def upgrade(
        self,
        new_hash: Callable,
        backup: bool = True,
        remove_backup_on_failure: bool = False,
    ):
        upgrade(
            self.filename,
            new_hash,
            backup=backup,
            remove_backup_on_failure=remove_backup_on_failure,
        )


class MemoCacheMetadata:
    def __init__(self, filename: Path | str, f, disable: bool = False):
        self.disabled = disable
        self._f = f
        self.filename = filename

    def get_hash(self, obj):
        if self.disabled:
            return "disabled"
        else:
            with self._f() as f:
                return f.memoshelve.get_hash(obj)

    def get_hash_mem(self, obj):
        if self.disabled:
            return "disabled"
        else:
            with self._f() as f:
                return f.memoshelve.get_hash_mem(obj)

    def disk_keys(self):
        if self.disabled:
            return set()
        else:
            with self._f() as f:
                return f.memoshelve.disk_keys()

    def disk_raw_items(self):
        if self.disabled:
            return []
        else:
            with self._f() as f:
                return f.memoshelve.disk_raw_items()

    def disk_raw_values(self):
        if self.disabled:
            return []
        else:
            with self._f() as f:
                return f.memoshelve.disk_raw_values()

    def disk_items(self):
        if self.disabled:
            return []
        else:
            with self._f() as f:
                return f.memoshelve.disk_items()

    def disk_values(self):
        if self.disabled:
            return []
        else:
            with self._f() as f:
                return f.memoshelve.disk_values()

    def mem_keys(self):
        if self.disabled:
            return set()
        else:
            with self._f() as f:
                return f.memoshelve.mem_keys()

    def mem_raw_items(self):
        if self.disabled:
            return []
        else:
            with self._f() as f:
                return f.memoshelve.mem_raw_items()

    def mem_raw_values(self):
        if self.disabled:
            return []
        else:
            with self._f() as f:
                return f.memoshelve.mem_raw_values()

    def mem_items(self):
        if self.disabled:
            return []
        else:
            with self._f() as f:
                return f.memoshelve.mem_items()

    def mem_values(self):
        if self.disabled:
            return []
        else:
            with self._f() as f:
                return f.memoshelve.mem_values()

    def keys(self):
        if self.disabled:
            return set()
        else:
            with self._f() as f:
                return f.memoshelve.keys()

    def raw_items(self):
        if self.disabled:
            return []
        else:
            with self._f() as f:
                return f.memoshelve.raw_items()

    def raw_values(self):
        if self.disabled:
            return []
        else:
            with self._f() as f:
                return f.memoshelve.raw_values()

    def items(self):
        if self.disabled:
            return []
        else:
            with self._f() as f:
                return f.memoshelve.items()

    def values(self):
        if self.disabled:
            return []
        else:
            with self._f() as f:
                return f.memoshelve.values()

    def compact(self, backup: bool = True):
        if self.disabled:
            return
        else:
            with self._f() as f:
                f.memoshelve.compact(backup=backup)

    def upgrade(
        self,
        new_hash: Callable | None = None,
        *,
        backup: bool = True,
        remove_backup_on_failure: bool = False,
    ):
        upgrade(
            self.filename,
            new_hash=new_hash,
            backup=backup,
            remove_backup_on_failure=remove_backup_on_failure,
        )


class MemoCacheAsyncMetadata:
    def __init__(self, filename: Path | str, f, disable: bool = False):
        self.disabled = disable
        self._f = f
        self.filename = filename

    async def get_hash(self, obj):
        if self.disabled:
            return "disabled"
        else:
            async with self._f() as f:
                return f.memoshelve.get_hash(obj)

    async def get_hash_mem(self, obj):
        if self.disabled:
            return "disabled"
        else:
            async with self._f() as f:
                return f.memoshelve.get_hash_mem(obj)

    async def disk_keys(self):
        if self.disabled:
            return set()
        else:
            async with self._f() as f:
                return f.memoshelve.disk_keys()

    async def disk_raw_items(self):
        if self.disabled:
            return []
        else:
            async with self._f() as f:
                return f.memoshelve.disk_raw_items()

    async def disk_raw_values(self):
        if self.disabled:
            return []
        else:
            async with self._f() as f:
                return f.memoshelve.disk_raw_values()

    async def disk_items(self):
        if self.disabled:
            return []
        else:
            async with self._f() as f:
                return f.memoshelve.disk_items()

    async def disk_values(self):
        if self.disabled:
            return []
        else:
            async with self._f() as f:
                return f.memoshelve.disk_values()

    async def mem_keys(self):
        if self.disabled:
            return set()
        else:
            async with self._f() as f:
                return f.memoshelve.mem_keys()

    async def mem_raw_items(self):
        if self.disabled:
            return []
        else:
            async with self._f() as f:
                return f.memoshelve.mem_raw_items()

    async def mem_raw_values(self):
        if self.disabled:
            return []
        else:
            async with self._f() as f:
                return f.memoshelve.mem_raw_values()

    async def mem_items(self):
        if self.disabled:
            return []
        else:
            async with self._f() as f:
                return f.memoshelve.mem_items()

    async def mem_values(self):
        if self.disabled:
            return []
        else:
            async with self._f() as f:
                return f.memoshelve.mem_values()

    async def keys(self):
        if self.disabled:
            return set()
        else:
            async with self._f() as f:
                return f.memoshelve.keys()

    async def raw_items(self):
        if self.disabled:
            return []
        else:
            async with self._f() as f:
                return f.memoshelve.raw_items()

    async def raw_values(self):
        if self.disabled:
            return []
        else:
            async with self._f() as f:
                return f.memoshelve.raw_values()

    async def items(self):
        if self.disabled:
            return []
        else:
            async with self._f() as f:
                return f.memoshelve.items()

    async def values(self):
        if self.disabled:
            return []
        else:
            async with self._f() as f:
                return f.memoshelve.values()

    async def compact(self, backup: bool = True):
        if self.disabled:
            return
        else:
            async with self._f() as f:
                f.memoshelve.compact(backup=backup)

    async def upgrade(
        self,
        new_hash: Callable | None = None,
        *,
        backup: bool = True,
        remove_backup_on_failure: bool = False,
    ):
        if self.disabled:
            return
        else:
            async with self._f() as f:
                f.memoshelve.upgrade(
                    new_hash=new_hash,
                    backup=backup,
                    remove_backup_on_failure=remove_backup_on_failure,
                )


def make_compute_cache_tuple(
    value: Callable[..., T] | None,
    ignore: Collection[str],
    always_bind: bool,
) -> Callable[..., tuple[tuple, dict[str, Any]]]:
    sig = inspect.signature(value or (lambda *_args, **_kwargs: None))

    def compute_cache_tuple(*args, **kwargs) -> tuple[tuple, dict[str, Any]]:
        if not always_bind and not ignore:
            return (args, kwargs)
        assert (
            value is not None
        ), "value must be provided if always_bind is True or ignore is not empty"
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
        except TypeError as e:
            if always_bind:
                # For always_bind, try partial binding and apply defaults
                try:
                    bound = sig.bind_partial(*args, **kwargs)
                    bound.apply_defaults()
                    filtered = {
                        k: v for k, v in bound.arguments.items() if k not in ignore
                    }
                    return ((), filtered)
                except TypeError:
                    pass

            # Fallback when binding fails and always_bind is False
            if ignore and any(param in kwargs for param in ignore):
                logging.error(
                    f"Attempted to bind args to ignore {ignore} in {value!r} with signature {sig} but got {e}"
                )

            return (args, {k: v for k, v in kwargs.items() if k not in ignore})
        filtered = {k: v for k, v in bound.arguments.items() if k not in ignore}
        return ((), filtered)

    return compute_cache_tuple


def copy_result(copy: Callable[[T], T], result: T, **kwargs) -> dict:
    return {"result": copy(result), **kwargs}


def make_make_get_raw(
    value: Callable[..., T],
    filename: Path | str,
    cache: dict[str, dict[str, Any]] = memoshelve_cache,
    *,
    get_hash: Callable | None = None,
    get_hash_mem: Callable | None = None,
    print_cache_miss: bool | None = None,
    print_cache_hit: bool | None = None,
    print_disk_cache_miss: bool | Callable[[str], None] | None = None,
    print_disk_cache_hit: bool | Callable[[str], None] | None = None,
    print_mem_cache_miss: bool | Callable[[str], None] | None = None,
    print_mem_cache_hit: bool | Callable[[str], None] | None = None,
    print_extended_cache_miss_disk: bool = False,
    copy: Callable[[T], T] = lambda x: x,
    allow_race: bool = True,
    ignore: Collection[str] = (),
    always_bind: bool = False,
):
    """Create a memoized version of a function using shelve + in-memory cache.

    This function provides a two-tier caching system: an in-memory cache for fastest
    access and a persistent shelve-based disk cache for longer-term storage.

    Args:
        value: The function to memoize
        filename: Path to the shelve database file for persistent cache
        cache: In-memory cache dictionary (default: global memoshelve_cache)
        get_hash: Function to compute hash for disk cache keys (default: stablehash.stablehash(-).hexdigest() or repr)
        get_hash_mem: Function to compute hash for memory cache keys (default: same as get_hash)
        print_cache_miss: Global setting for cache miss logger
        print_cache_hit: Global setting for cache hit logger
        print_disk_cache_miss: Disk cache miss logger setting or function
        print_disk_cache_hit: Disk cache hit logger setting or function
        print_mem_cache_miss: Memory cache miss logger setting or function
        print_mem_cache_hit: Memory cache hit logger setting or function
        print_extended_cache_miss_disk: Include extended traceback info in disk cache miss logs
        copy: Function to copy cached values (default: identity function)
        allow_race: Allow race conditions in cache updates (default: True)

    Returns:
        A context manager that yields a memoized function with additional methods:
        - __call_with_status__: Call function and return (result, cache_status)
        - get_with_status: Get cached value and return (result, cache_status)
        - get: Get cached value without status
        - __contains__: Check if arguments are cached
        - put: Manually store a value in cache

    Example:
        ```python
        with memoshelve(expensive_function, "cache.db") as cached_fn:
            result = cached_fn(arg1, arg2)
            # Or get with status:
            result, status = cached_fn.__call_with_status__(arg1, arg2)
        ```
    """

    def set_print_fn(
        print_val: bool | Callable[[str], None] | None,
        print_gen_val: bool | None,
        default_val: bool,
        default_fn: Callable[[str], None],
    ) -> Callable[[str], None]:
        if print_val is None:
            print_val = default_val if print_gen_val is None else print_gen_val
        if print_val is True:
            return default_fn
        elif print_val is False:
            return lambda _: None
        else:
            return print_val

    print_mem_cache_miss = set_print_fn(
        print_mem_cache_miss,
        print_cache_miss,
        DEFAULT_PRINT_MEM_CACHE_MISS,
        DEFAULT_PRINT_CACHE_MISS_FN,
    )
    print_mem_cache_hit = set_print_fn(
        print_mem_cache_hit,
        print_cache_hit,
        DEFAULT_PRINT_MEM_CACHE_HIT,
        DEFAULT_PRINT_CACHE_HIT_FN,
    )
    print_disk_cache_miss = set_print_fn(
        print_disk_cache_miss,
        print_cache_miss,
        DEFAULT_PRINT_DISK_CACHE_MISS,
        DEFAULT_PRINT_CACHE_MISS_FN,
    )
    print_disk_cache_hit = set_print_fn(
        print_disk_cache_hit,
        print_cache_hit,
        DEFAULT_PRINT_DISK_CACHE_HIT,
        DEFAULT_PRINT_CACHE_HIT_FN,
    )

    filename = str(Path(filename).absolute())
    mem_db: dict[str, dict[str, Any]] = cache.setdefault(filename, {})
    if get_hash is None:
        get_hash = DEFAULT_HASH
    if get_hash_mem is None:
        get_hash_mem = get_hash

    compute_cache_tuple = make_compute_cache_tuple(
        value, ignore=ignore, always_bind=always_bind
    )

    def make_get_raw(get_db):
        def get_raw_dict(
            *args, **kwargs
        ) -> tuple[dict, Literal["cached (mem)", "cached (disk)", "miss"]]:
            cache_tuple = compute_cache_tuple(*args, **kwargs)
            mkey = get_hash_mem(cache_tuple)
            try:
                result = mem_db[mkey]
                assert isinstance(result, dict), result
                result = copy_result(copy, **result)
                print_mem_cache_hit(f"Cache hit (mem): {mkey}")
                return result, "cached (mem)"
            except KeyError:
                print_mem_cache_miss(f"Cache miss (mem): {mkey}")
                key = str(get_hash(cache_tuple))
                try:
                    with get_db() as db:
                        mem_db[mkey] = upgrade_value_or_raise(db[key])
                    print_disk_cache_hit(f"Cache hit (disk: {filename}): {key}")
                    result = mem_db[mkey]
                    assert isinstance(result, dict), result
                    result = copy_result(copy, **result)
                    return result, "cached (disk)"
                except Exception as e:
                    if isinstance(e, KeyError):
                        frames = traceback.extract_stack()
                        # Remove the current frame and the memoshelve internal frames
                        frames = [
                            f
                            for f in frames
                            if not f.filename.endswith("memoshelve/__init__.py")
                        ]
                        print_disk_cache_miss(
                            f"Cache miss (disk: {filename}): {key} ({value.__name__ if hasattr(value, '__name__') else 'anonymous'})"
                            + (
                                f" ({[f.filename + ':' + f.name for f in frames]})"
                                if print_extended_cache_miss_disk
                                else ""
                            )
                        )
                    elif isinstance(e, (KeyboardInterrupt, SystemExit)):
                        raise e
                    else:
                        logger.error(f"Error {e} in {filename} with key {key}")
                    if not isinstance(e, (KeyError, AttributeError, UnpicklingError)):
                        raise e
                return {
                    "result": (mkey, key),
                    "version": __version__,
                    "args": cache_tuple[0],
                    "kwargs": cache_tuple[1],
                }, "miss"

        def get_raw(
            *args, **kwargs
        ) -> (
            tuple[T, Literal["cached (mem)", "cached (disk)"]]
            | tuple[tuple[str, str], Literal["miss"]]
        ):
            result, status = get_raw_dict(*args, **kwargs)
            return result["result"], status  # type: ignore

        return get_raw_dict, get_raw

    return (
        filename,
        get_hash,
        get_hash_mem,
        mem_db,
        make_get_raw,
        compute_cache_tuple,
        partial(
            MemoshelveMetadata,
            filename,
            get_hash=get_hash,
            get_hash_mem=get_hash_mem,
            mem_db=mem_db,
        ),
    )


def memoshelve(
    value: Callable,
    filename: Path | str,
    cache: dict[str, dict[str, Any]] = memoshelve_cache,
    *,
    get_hash: Callable | None = None,
    get_hash_mem: Callable | None = None,
    print_cache_miss: bool | None = None,
    print_cache_hit: bool | None = None,
    print_disk_cache_miss: bool | Callable[[str], None] | None = None,
    print_disk_cache_hit: bool | Callable[[str], None] | None = None,
    print_mem_cache_miss: bool | Callable[[str], None] | None = None,
    print_mem_cache_hit: bool | Callable[[str], None] | None = None,
    print_extended_cache_miss_disk: bool = False,
    copy: Callable[[T], T] = lambda x: x,
    allow_race: bool = True,
    ignore: Collection[str] = (),
    always_bind: bool = False,
):
    """Create a memoized version of a function using shelve + in-memory cache.

    This function provides a two-tier caching system: an in-memory cache for fastest
    access and a persistent shelve-based disk cache for longer-term storage.

    Args:
        value: The function to memoize
        filename: Path to the shelve database file for persistent cache
        cache: In-memory cache dictionary (default: global memoshelve_cache)
        get_hash: Function to compute hash for disk cache keys (default: stablehash.stablehash(-).hexdigest() or repr)
        get_hash_mem: Function to compute hash for memory cache keys (default: same as get_hash)
        print_cache_miss: Global setting for cache miss logger
        print_cache_hit: Global setting for cache hit logger
        print_disk_cache_miss: Disk cache miss logger setting or function
        print_disk_cache_hit: Disk cache hit logger setting or function
        print_mem_cache_miss: Memory cache miss logger setting or function
        print_mem_cache_hit: Memory cache hit logger setting or function
        print_extended_cache_miss_disk: Include extended traceback info in disk cache miss logs
        copy: Function to copy cached values (default: identity function)
        allow_race: Allow race conditions in cache updates (default: True)
        ignore: Collection of argument names to ignore in cache key computation
        always_bind: Always bind arguments to the function signature for cache key computation (default: False)

    Returns:
        A context manager that yields a memoized function with additional methods:
        - __call_with_status__: Call function and return (result, cache_status)
        - get_with_status: Get cached value and return (result, cache_status)
        - get: Get cached value without status
        - __contains__: Check if arguments are cached
        - put: Manually store a value in cache

    Example:
        ```python
        with memoshelve(expensive_function, "cache.db") as cached_fn:
            result = cached_fn(arg1, arg2)
            # Or get with status:
            result, status = cached_fn.__call_with_status__(arg1, arg2)
        ```
    """
    (
        filename,
        get_hash,
        get_hash_mem,
        mem_db,
        make_get_raw,
        compute_cache_tuple,
        make_metadata,
    ) = make_make_get_raw(
        value,
        filename=filename,
        cache=cache,
        get_hash=get_hash,
        get_hash_mem=get_hash_mem,
        print_cache_miss=print_cache_miss,
        print_cache_hit=print_cache_hit,
        print_disk_cache_miss=print_disk_cache_miss,
        print_disk_cache_hit=print_disk_cache_hit,
        print_mem_cache_miss=print_mem_cache_miss,
        print_mem_cache_hit=print_mem_cache_hit,
        print_extended_cache_miss_disk=print_extended_cache_miss_disk,
        copy=copy,
        allow_race=allow_race,
        ignore=ignore,
        always_bind=always_bind,
    )

    # filename = str(Path(filename).absolute())
    # mem_db = cache.setdefault(filename, {})
    # if get_hash_mem is None:
    #     get_hash_mem = get_hash

    @contextmanager
    def open_db():
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        pending_compact = False
        with lazy_shelve_open(filename, eager=not allow_race) as get_db:
            _get_raw_dict, get_raw = make_get_raw(get_db)
            metadata = make_metadata(get_db=get_db)

            def get(*args, **kwargs):
                result, status = get_raw(*args, **kwargs)
                return result

            def contains(*args, **kwargs):
                _, status = get_raw(*args, **kwargs)
                return status != "miss"

            def put_raw_via_key(value, key, mkey, *args, **kwargs):
                nonlocal pending_compact
                cache_tuple = compute_cache_tuple(*args, **kwargs)
                if mkey is None:
                    mkey = get_hash_mem(cache_tuple)
                if key is None:
                    key = str(get_hash(cache_tuple))
                cache_value = {
                    "result": value,
                    "version": __version__,
                    "args": cache_tuple[0],
                    "kwargs": cache_tuple[1],
                }
                try:
                    with get_db() as db:
                        db[key] = mem_db[mkey] = cache_value
                except getattr(
                    _gdbm, "error", _gdbm_dummy_error
                ) as e:  # handle recovery
                    if get_db.eager:
                        logging.error(
                            f"Error writing to {filename}, queueing compact: {e}"
                        )
                        pending_compact = True
                        raise e
                    else:
                        logging.warning(
                            f"Error writing to {filename}, attempting compact: {e}"
                        )
                        compact(filename, remove_on_unknown_type=True)
                        with get_db() as db:
                            db[key] = mem_db[mkey] = cache_value
                return mem_db[mkey]["result"]

            def put(value, *args, **kwargs):
                put_raw_via_key(value, None, None, *args, **kwargs)

            def delegate_raw(*args, **kwargs):
                result, status = get_raw(*args, **kwargs)
                if status == "miss":
                    assert isinstance(result, tuple), result
                    mkey, key = result
                    result = put_raw_via_key(
                        copy(value(*args, **kwargs)), key, mkey, *args, **kwargs
                    )
                    return result, "miss"
                else:
                    return result, status

            def delegate(*args, **kwargs):
                result, _status = delegate_raw(*args, **kwargs)
                return result

            delegate.__call_with_status__ = delegate_raw
            delegate.get_with_status = get_raw
            delegate.get = get
            delegate.__contains__ = contains
            delegate.put = put
            delegate.memoshelve = metadata

            yield delegate

            if pending_compact:
                logging.warning(f"Compacting {filename} after use due to error")
                compact(filename, remove_on_unknown_type=True)

    return open_db


def async_memoshelve(
    value: Callable,
    filename: Path | str,
    cache: dict[str, dict[str, Any]] = memoshelve_cache,
    get_hash: Callable | None = None,
    get_hash_mem: Callable | None = None,
    print_cache_miss: bool | None = None,
    print_cache_hit: bool | None = None,
    print_disk_cache_miss: bool | Callable[[str], None] | None = None,
    print_disk_cache_hit: bool | Callable[[str], None] | None = None,
    print_mem_cache_miss: bool | Callable[[str], None] | None = None,
    print_mem_cache_hit: bool | Callable[[str], None] | None = None,
    print_extended_cache_miss_disk: bool = False,
    copy: Callable[[T], T] = lambda x: x,
    allow_race: bool = True,
    ignore: Collection[str] = (),
    always_bind: bool = False,
):
    """Create a memoized version of a function using shelve + in-memory cache.

    This function provides a two-tier caching system: an in-memory cache for fastest
    access and a persistent shelve-based disk cache for longer-term storage.

    Args:
        value: The function to memoize
        filename: Path to the shelve database file for persistent cache
        cache: In-memory cache dictionary (default: global memoshelve_cache)
        get_hash: Function to compute hash for disk cache keys (default: get_hash_ascii)
        get_hash_mem: Function to compute hash for memory cache keys (default: same as get_hash)
        print_cache_miss: Global setting for cache miss logger
        print_cache_hit: Global setting for cache hit logger
        print_disk_cache_miss: Disk cache miss logger setting or function
        print_disk_cache_hit: Disk cache hit logger setting or function
        print_mem_cache_miss: Memory cache miss logger setting or function
        print_mem_cache_hit: Memory cache hit logger setting or function
        print_extended_cache_miss_disk: Include extended traceback info in disk cache miss logs
        copy: Function to copy cached values (default: identity function)
        allow_race: Allow race conditions in cache updates (default: True)
        ignore: Collection of argument names to ignore in cache key computation
        always_bind: Always bind arguments to the function signature for cache key computation (default: False)

    Returns:
        A context manager that yields a memoized function with additional methods:
        - __call_with_status__: Call function and return (result, cache_status)
        - get_with_status: Get cached value and return (result, cache_status)
        - get: Get cached value without status
        - __contains__: Check if arguments are cached
        - put: Manually store a value in cache

    Example:
        ```python
        with memoshelve(expensive_function, "cache.db") as cached_fn:
            result = cached_fn(arg1, arg2)
            # Or get with status:
            result, status = cached_fn.__call_with_status__(arg1, arg2)
        ```
    """
    (
        filename,
        get_hash,
        get_hash_mem,
        mem_db,
        make_get_raw,
        compute_cache_tuple,
        make_metadata,
    ) = make_make_get_raw(
        value,
        filename,
        cache,
        get_hash=get_hash,
        get_hash_mem=get_hash_mem,
        print_cache_miss=print_cache_miss,
        print_cache_hit=print_cache_hit,
        print_disk_cache_miss=print_disk_cache_miss,
        print_disk_cache_hit=print_disk_cache_hit,
        print_mem_cache_miss=print_mem_cache_miss,
        print_mem_cache_hit=print_mem_cache_hit,
        print_extended_cache_miss_disk=print_extended_cache_miss_disk,
        copy=copy,
        allow_race=allow_race,
        ignore=ignore,
        always_bind=always_bind,
    )

    # filename = str(Path(filename).absolute())
    # mem_db = cache.setdefault(filename, {})
    # if get_hash_mem is None:
    #     get_hash_mem = get_hash

    @asynccontextmanager
    async def open_db():
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        pending_compact = False
        with lazy_shelve_open(filename, eager=not allow_race) as get_db:
            _get_raw_dict, get_raw = make_get_raw(get_db)
            metadata = make_metadata(get_db=get_db)

            def get(*args, **kwargs):
                result, status = get_raw(*args, **kwargs)
                return result

            def contains(*args, **kwargs):
                _, status = get_raw(*args, **kwargs)
                return status != "miss"

            def put_raw_via_key(value, key, mkey, *args, **kwargs):
                nonlocal pending_compact
                cache_tuple = compute_cache_tuple(*args, **kwargs)
                if mkey is None:
                    mkey = get_hash_mem(cache_tuple)
                if key is None:
                    key = str(get_hash(cache_tuple))
                cache_value = {
                    "result": value,
                    "version": __version__,
                    "args": cache_tuple[0],
                    "kwargs": cache_tuple[1],
                }
                try:
                    with get_db() as db:
                        db[key] = mem_db[mkey] = cache_value
                except getattr(
                    _gdbm, "error", _gdbm_dummy_error
                ) as e:  # handle recovery
                    if get_db.eager:
                        logging.error(
                            f"Error writing to {filename}, queueing compact: {e}"
                        )
                        pending_compact = True
                        raise e
                    else:
                        logging.warning(
                            f"Error writing to {filename}, attempting compact: {e}"
                        )
                        compact(filename)
                        with get_db() as db:
                            db[key] = mem_db[mkey] = cache_value
                return mem_db[mkey]["result"]

            def put(value, *args, **kwargs):
                put_raw_via_key(value, None, None, *args, **kwargs)

            async def delegate_raw(*args, **kwargs):
                result, status = get_raw(*args, **kwargs)
                if status == "miss":
                    assert isinstance(result, tuple), result
                    mkey, key = result
                    result = put_raw_via_key(
                        copy(await value(*args, **kwargs)), key, mkey, *args, **kwargs
                    )
                    return result, "miss"
                else:
                    return result, status

            async def delegate(*args, **kwargs):
                result, _status = await delegate_raw(*args, **kwargs)
                return result

            delegate.__call_with_status__ = delegate_raw
            delegate.get_with_status = get_raw
            delegate.get = get
            delegate.__contains__ = contains
            delegate.put = put
            delegate.memoshelve = metadata

            yield delegate

    return open_db


def uncache(
    *args,
    filename: Path | str,
    cache: dict[str, dict[str, Any]] = memoshelve_cache,
    get_hash: Callable | None = None,
    get_hash_mem: Callable | None = None,
    ignore: Collection[str] = (),
    always_bind: bool = False,
    value: Callable | None = None,
    **kwargs,
):
    """Remove cached entries for specific arguments from both memory and disk cache.

    This function removes cached results for the given arguments from both the
    in-memory cache and the persistent shelve database.

    Args:
        *args: Positional arguments that were passed to the cached function
        filename: Path to the shelve database file
        cache: In-memory cache dictionary (default: global memoshelve_cache)
        get_hash: Function to compute hash for disk cache keys (default: stablehash.stablehash(-).hexdigest() or repr)
        get_hash_mem: Function to compute hash for memory cache keys (default: same as get_hash)
        **kwargs: Keyword arguments that were passed to the cached function

    Example:
        ```python
        # Remove cached result for specific arguments
        uncache(arg1, arg2, filename="cache.db", kwarg1="value")
        ```
    """
    filename = str(Path(filename).absolute())
    mem_db = cache.setdefault(filename, {})
    if get_hash is None:
        get_hash = DEFAULT_HASH
    if get_hash_mem is None:
        get_hash_mem = get_hash

    cache_tuple = make_compute_cache_tuple(
        value, ignore=ignore, always_bind=always_bind
    )(*args, **kwargs)

    with shelve.open(filename) as db:
        mkey = get_hash_mem(cache_tuple)
        if mkey in mem_db:
            del mem_db[mkey]
        key = str(get_hash(cache_tuple))
        if key in db:
            del db[key]


# for decorators
def sync_cache(
    filename: Path | str | None = None,
    cache: dict[str, dict[str, Any]] = memoshelve_cache,
    *,
    get_hash: Callable | None = None,
    get_hash_mem: Callable | None = None,
    print_cache_miss: bool | None = None,
    print_cache_hit: bool | None = None,
    print_disk_cache_miss: bool | Callable[[str], None] | None = None,
    print_disk_cache_hit: bool | Callable[[str], None] | None = None,
    print_mem_cache_miss: bool | Callable[[str], None] | None = None,
    print_mem_cache_hit: bool | Callable[[str], None] | None = None,
    print_extended_cache_miss_disk: bool = False,
    disable: bool = False,
    copy: Callable[[T], T] = lambda x: x,
    allow_race: bool = True,
    ignore: Collection[str] = (),
    always_bind: bool = False,
):
    """Decorator for memoizing functions with two-tier caching (memory + disk).

    This decorator provides persistent memoization using both in-memory and disk-based
    caching via shelve. The decorated function gains additional methods for cache
    inspection and manipulation.

    Args:
        filename: Path to shelve cache file. If None, uses DEFAULT_CACHE_DIR/function_name.shelve
        cache: In-memory cache dictionary (default: global memoshelve_cache)
        get_hash: Function to compute hash for disk cache keys (default: stablehash.stablehash(-).hexdigest() or repr)
        get_hash_mem: Function to compute hash for memory cache keys (default: same as get_hash)
        print_cache_miss: Global setting for cache miss logger
        print_cache_hit: Global setting for cache hit logger
        print_disk_cache_miss: Disk cache miss logger setting or function
        print_disk_cache_hit: Disk cache hit logger setting or function
        print_mem_cache_miss: Memory cache miss logger setting or function
        print_mem_cache_hit: Memory cache hit logger setting or function
        print_extended_cache_miss_disk: Include extended traceback info in disk cache miss logs
        disable: Disable caching entirely (default: False)
        copy: Function to copy cached values (default: identity function)
        allow_race: Allow race conditions in cache updates (default: True)
        ignore: Collection of argument names to ignore in cache key computation
        always_bind: Always bind arguments to the function signature for cache key computation (default: False)

    Returns:
        A decorator function that wraps the target function with caching capabilities.
        The wrapped function gains these additional methods:
        - __call_with_status__: Call and return (result, cache_status)
        - get_with_status: Get cached value and return (result, cache_status)
        - get: Get cached value without status
        - __contains__: Check if arguments are cached
        - put: Manually store a value in cache
        - uncache: Remove specific cached entries

    Example:
        ```python
        @cache(filename="my_cache.db")
        def expensive_function(x, y):
            return x * y

        result = expensive_function(5, 10)  # Computed and cached
        result = expensive_function(5, 10)  # Retrieved from cache

        # Check cache status
        result, status = expensive_function.__call_with_status__(5, 10)
        print(status)  # "cached (mem)" or "cached (disk)" or "miss"

        # Manual cache operations
        if expensive_function.__contains__(5, 10):
            cached_result = expensive_function.get(5, 10)
        expensive_function.uncache(5, 10)  # Remove from cache
        ```
    """

    def wrap(value):
        path = (
            Path(filename)
            if filename
            else DEFAULT_CACHE_DIR / f"{value.__name__}.shelve"
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        memo = memoshelve(
            value,
            filename=path,
            cache=cache,
            get_hash=get_hash,
            get_hash_mem=get_hash_mem,
            print_cache_miss=print_cache_miss,
            print_cache_hit=print_cache_hit,
            print_disk_cache_miss=print_disk_cache_miss,
            print_disk_cache_hit=print_disk_cache_hit,
            print_mem_cache_miss=print_mem_cache_miss,
            print_mem_cache_hit=print_mem_cache_hit,
            print_extended_cache_miss_disk=print_extended_cache_miss_disk,
            copy=copy,
            allow_race=allow_race,
            ignore=ignore,
            always_bind=always_bind,
        )
        if disable:

            def wrapper_with_status(*args, **kwargs):  # type: ignore
                return value(*args, **kwargs), False

            def wrapper_get_with_status(*args, **kwargs):  # type: ignore
                return None, "miss"

            def wrapper_get(*args, **kwargs):  # type: ignore
                return None

            def wrapper_contains(*args, **kwargs):  # type: ignore
                return False

            def wrapper_put(val, *args, **kwargs):  # type: ignore
                return
        else:

            def wrapper_with_status(*args, **kwargs):
                with memo() as f:
                    return f.__call_with_status__(*args, **kwargs)

            def wrapper_get_with_status(*args, **kwargs):
                with memo() as f:
                    return f.get_with_status(*args, **kwargs)

            def wrapper_get(*args, **kwargs):
                with memo() as f:
                    return f.get(*args, **kwargs)

            def wrapper_contains(*args, **kwargs):
                with memo() as f:
                    return f.__contains__(*args, **kwargs)

            def wrapper_put(val, *args, **kwargs):
                with memo() as f:
                    f.put(val, *args, **kwargs)

        value.__call_with_status__ = wrapper_with_status
        value.get_with_status = wrapper_get_with_status
        value.get = wrapper_get
        value.__contains__ = wrapper_contains
        value.put = wrapper_put
        value.uncache = partial(
            uncache,
            filename=path,
            cache=cache,
            get_hash=get_hash,
            get_hash_mem=get_hash_mem,
            ignore=ignore,
            always_bind=always_bind,
            value=value,
        )
        value.memoshelve = MemoCacheMetadata(
            str(path.absolute()), memo, disable=disable
        )

        @wraps(value)
        def wrapper(*args, **kwargs):
            result, _status = wrapper_with_status(*args, **kwargs)
            return result

        return wrapper

    return wrap


def async_cache(
    filename: Path | str | None = None,
    cache: dict[str, dict[str, Any]] = memoshelve_cache,
    *,
    get_hash: Callable | None = None,
    get_hash_mem: Callable | None = None,
    print_cache_miss: bool | None = None,
    print_cache_hit: bool | None = None,
    print_disk_cache_miss: bool | Callable[[str], None] | None = None,
    print_disk_cache_hit: bool | Callable[[str], None] | None = None,
    print_mem_cache_miss: bool | Callable[[str], None] | None = None,
    print_mem_cache_hit: bool | Callable[[str], None] | None = None,
    print_extended_cache_miss_disk: bool = False,
    disable: bool = False,
    copy: Callable[[T], T] = lambda x: x,
    allow_race: bool = True,
    ignore: Collection[str] = (),
    always_bind: bool = False,
):
    """Decorator for memoizing async functions with two-tier caching (memory + disk).

    Similar to sync_cache, but for asynchronous functions.
    """

    def wrap(value):
        path = (
            Path(filename)
            if filename
            else DEFAULT_CACHE_DIR / f"{value.__name__}.shelve"
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        memo = async_memoshelve(
            value,
            filename=path,
            cache=cache,
            get_hash=get_hash,
            get_hash_mem=get_hash_mem,
            print_cache_miss=print_cache_miss,
            print_cache_hit=print_cache_hit,
            print_disk_cache_miss=print_disk_cache_miss,
            print_disk_cache_hit=print_disk_cache_hit,
            print_mem_cache_miss=print_mem_cache_miss,
            print_mem_cache_hit=print_mem_cache_hit,
            print_extended_cache_miss_disk=print_extended_cache_miss_disk,
            copy=copy,
            allow_race=allow_race,
            ignore=ignore,
            always_bind=always_bind,
        )
        if disable:

            async def wrapper_with_status(*args, **kwargs):  # type: ignore
                return await value(*args, **kwargs), "miss"

            async def wrapper_get_with_status(*args, **kwargs):  # type: ignore
                return None, "miss"

            async def wrapper_get(*args, **kwargs):  # type: ignore
                return None

            async def wrapper_contains(*args, **kwargs):  # type: ignore
                return False

            async def wrapper_put(val, *args, **kwargs):  # type: ignore
                return
        else:

            async def wrapper_with_status(*args, **kwargs):
                async with memo() as f:
                    return await f.__call_with_status__(*args, **kwargs)

            async def wrapper_get_with_status(*args, **kwargs):
                async with memo() as f:
                    return f.get_with_status(*args, **kwargs)

            async def wrapper_get(*args, **kwargs):
                async with memo() as f:
                    return f.get(*args, **kwargs)

            async def wrapper_contains(*args, **kwargs):
                async with memo() as f:
                    return f.__contains__(*args, **kwargs)

            async def wrapper_put(val, *args, **kwargs):
                async with memo() as f:
                    f.put(val, *args, **kwargs)

        value.__call_with_status__ = wrapper_with_status
        value.get_with_status = wrapper_get_with_status
        value.get = wrapper_get
        value.__contains__ = wrapper_contains
        value.put = wrapper_put
        value.uncache = partial(
            uncache,
            filename=path,
            cache=cache,
            get_hash=get_hash,
            get_hash_mem=get_hash_mem,
            ignore=ignore,
            always_bind=always_bind,
            value=value,
        )
        value.memoshelve = MemoCacheAsyncMetadata(
            str(path.absolute()), memo, disable=disable
        )

        @wraps(value)
        async def wrapper(*args, **kwargs):
            result, _status = await wrapper_with_status(*args, **kwargs)
            return result

        return wrapper

    return wrap


def cache(
    filename: Path | str | None = None,
    cache: dict[str, dict[str, Any]] = memoshelve_cache,
    *,
    get_hash: Callable | None = None,
    get_hash_mem: Callable | None = None,
    print_cache_miss: bool | None = None,
    print_cache_hit: bool | None = None,
    print_disk_cache_miss: bool | Callable[[str], None] | None = None,
    print_disk_cache_hit: bool | Callable[[str], None] | None = None,
    print_mem_cache_miss: bool | Callable[[str], None] | None = None,
    print_mem_cache_hit: bool | Callable[[str], None] | None = None,
    print_extended_cache_miss_disk: bool = False,
    disable: bool = False,
    copy: Callable[[T], T] = lambda x: x,
    allow_race: bool = True,
    ignore: Collection[str] = (),
    always_bind: bool = False,
):
    """Decorator for memoizing functions with two-tier caching, choosing sync or async based on the function type."""

    def decorator(func):
        if inspect.iscoroutinefunction(func):
            return async_cache(
                filename=filename,
                cache=cache,
                get_hash=get_hash,
                get_hash_mem=get_hash_mem,
                print_cache_miss=print_cache_miss,
                print_cache_hit=print_cache_hit,
                print_disk_cache_miss=print_disk_cache_miss,
                print_disk_cache_hit=print_disk_cache_hit,
                print_mem_cache_miss=print_mem_cache_miss,
                print_mem_cache_hit=print_mem_cache_hit,
                print_extended_cache_miss_disk=print_extended_cache_miss_disk,
                disable=disable,
                copy=copy,
                allow_race=allow_race,
                ignore=ignore,
                always_bind=always_bind,
            )(func)
        else:
            return sync_cache(
                filename=filename,
                cache=cache,
                get_hash=get_hash,
                get_hash_mem=get_hash_mem,
                print_cache_miss=print_cache_miss,
                print_cache_hit=print_cache_hit,
                print_disk_cache_miss=print_disk_cache_miss,
                print_disk_cache_hit=print_disk_cache_hit,
                print_mem_cache_miss=print_mem_cache_miss,
                print_mem_cache_hit=print_mem_cache_hit,
                print_extended_cache_miss_disk=print_extended_cache_miss_disk,
                disable=disable,
                copy=copy,
                allow_race=allow_race,
                ignore=ignore,
                always_bind=always_bind,
            )(func)

    return decorator
