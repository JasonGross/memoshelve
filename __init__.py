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
from typing import Any, Callable, Literal, Optional, TypeVar

logger = logging.getLogger(__name__)

try:
    import dill
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

if dill is not None:
    # monkeypatch shelve as per https://stackoverflow.com/q/52927236/377022
    shelve.Pickler = dill.Pickler  # type: ignore
    shelve.Unpickler = dill.Unpickler  # type: ignore


def hash_via_stablehash(obj: object) -> str:
    assert stablehash is not None
    return stablehash.stablehash(obj).hexdigest()


__version__ = "0.1.0"

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


def compact(filename: Path | str, backup: bool = True):
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
    entries = {}
    with shelve.open(filename) as db:
        for k in db.keys():
            try:
                entries[k] = db[k]
            except UnpicklingError:
                logger.warning(f"UnpicklingError for {k} in {filename}")
    if backup:
        backup_name = backup_file(filename)
    else:
        backup_name = None
        os.remove(filename)
    with shelve.open(filename) as db:
        for k in entries.keys():
            db[k] = entries[k]
    if backup_name:
        assert backup_name != filename, backup_name
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
                        raise e
            with sh as db:
                yield db

        yield get_db


CacheStatus = Literal["cached (mem)", "cached (disk)", "miss"]


def make_make_get_raw(
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
    mem_db = cache.setdefault(filename, {})
    if get_hash is None:
        get_hash = DEFAULT_HASH
    if get_hash_mem is None:
        get_hash_mem = get_hash

    def make_get_raw(get_db):
        def get_raw(*args, **kwargs):
            mkey = get_hash_mem((args, kwargs))
            try:
                result = mem_db[mkey]
                print_mem_cache_hit(f"Cache hit (mem): {mkey}")
                return result, "cached (mem)"
            except KeyError:
                print_mem_cache_miss(f"Cache miss (mem): {mkey}")
                key = str(get_hash((args, kwargs)))
                try:
                    with get_db() as db:
                        mem_db[mkey] = db[key]
                    print_disk_cache_hit(f"Cache hit (disk: {filename}): {key}")
                    return mem_db[mkey], "cached (disk)"
                except Exception as e:
                    if isinstance(e, KeyError):
                        frames = traceback.extract_stack()
                        # Remove the current frame and the memoshelve internal frames
                        frames = [
                            f for f in frames if "memoshelve.py" not in f.filename
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
                return (mkey, key), "miss"

        return get_raw

    def make_disk_keys_items_values(get_db):
        def disk_keys():
            with get_db() as db:
                return db.keys()

        def disk_items():
            with get_db() as db:
                return db.items()

        def disk_values():
            with get_db() as db:
                return db.values()

        return disk_keys, disk_items, disk_values

    def make_keys_items_values(get_db):
        disk_keys, disk_items, disk_values = make_disk_keys_items_values(get_db)

        def keys():
            return set(disk_keys()) | set(mem_db.keys())

        def items():
            return (dict(disk_items()) | mem_db).items()

        def values():
            return (dict(disk_items()) | mem_db).values()

        return keys, items, values

    return (
        filename,
        get_hash,
        get_hash_mem,
        mem_db,
        make_get_raw,
        make_keys_items_values,
        make_disk_keys_items_values,
    )


def memoshelve(
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
    (
        filename,
        get_hash,
        get_hash_mem,
        mem_db,
        make_get_raw,
        make_keys_items_values,
        make_disk_keys_items_values,
    ) = make_make_get_raw(
        value,
        filename,
        cache,
        get_hash,
        get_hash_mem,
        print_cache_miss,
        print_cache_hit,
        print_disk_cache_miss,
        print_disk_cache_hit,
        print_mem_cache_miss,
        print_mem_cache_hit,
        print_extended_cache_miss_disk,
        copy,
        allow_race,
    )

    # filename = str(Path(filename).absolute())
    # mem_db = cache.setdefault(filename, {})
    # if get_hash_mem is None:
    #     get_hash_mem = get_hash

    @contextmanager
    def open_db():
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with lazy_shelve_open(filename, eager=not allow_race) as get_db:
            get_raw = make_get_raw(get_db)
            mem_keys, mem_items, mem_values = mem_db.keys, mem_db.items, mem_db.values
            keys, items, values = make_keys_items_values(get_db)
            disk_keys, disk_items, disk_values = make_disk_keys_items_values(get_db)

            def get(*args, **kwargs):
                result, status = get_raw(*args, **kwargs)
                return result

            def contains(*args, **kwargs):
                _, status = get_raw(*args, **kwargs)
                return status != "miss"

            def put(value, *args, **kwargs):
                mkey = get_hash_mem((args, kwargs))
                key = str(get_hash((args, kwargs)))
                with get_db() as db:
                    mem_db[mkey] = db[key] = value

            def delegate_raw(*args, **kwargs):
                result, status = get_raw(*args, **kwargs)
                if status == "miss":
                    mkey, key = result
                    mem_db[mkey] = copy(value(*args, **kwargs))
                    with get_db() as db:
                        db[key] = mem_db[mkey]
                    return mem_db[mkey], "miss"
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
            delegate.memoshelve_mem_keys = mem_keys
            delegate.memoshelve_mem_items = mem_items
            delegate.memoshelve_mem_values = mem_values
            delegate.memoshelve_keys = keys
            delegate.memoshelve_items = items
            delegate.memoshelve_values = values
            delegate.memoshelve_disk_keys = disk_keys
            delegate.memoshelve_disk_items = disk_items
            delegate.memoshelve_disk_values = disk_values

            yield delegate

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
        make_keys_items_values,
        make_disk_keys_items_values,
    ) = make_make_get_raw(
        value,
        filename,
        cache,
        get_hash,
        get_hash_mem,
        print_cache_miss,
        print_cache_hit,
        print_disk_cache_miss,
        print_disk_cache_hit,
        print_mem_cache_miss,
        print_mem_cache_hit,
        print_extended_cache_miss_disk,
        copy,
        allow_race,
    )

    # filename = str(Path(filename).absolute())
    # mem_db = cache.setdefault(filename, {})
    # if get_hash_mem is None:
    #     get_hash_mem = get_hash

    @asynccontextmanager
    async def open_db():
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with lazy_shelve_open(filename, eager=not allow_race) as get_db:
            get_raw = make_get_raw(get_db)
            mem_keys, mem_items, mem_values = mem_db.keys, mem_db.items, mem_db.values
            keys, items, values = make_keys_items_values(get_db)
            disk_keys, disk_items, disk_values = make_disk_keys_items_values(get_db)

            def get(*args, **kwargs):
                result, status = get_raw(*args, **kwargs)
                return result

            def contains(*args, **kwargs):
                _, status = get_raw(*args, **kwargs)
                return status != "miss"

            def put(value, *args, **kwargs):
                mkey = get_hash_mem((args, kwargs))
                key = str(get_hash((args, kwargs)))
                with get_db() as db:
                    mem_db[mkey] = db[key] = value

            async def delegate_raw(*args, **kwargs):
                result, status = get_raw(*args, **kwargs)
                if status == "miss":
                    mkey, key = result
                    mem_db[mkey] = copy(await value(*args, **kwargs))
                    with get_db() as db:
                        db[key] = mem_db[mkey]
                    return mem_db[mkey], "miss"
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
            delegate.memoshelve_mem_keys = mem_keys
            delegate.memoshelve_mem_items = mem_items
            delegate.memoshelve_mem_values = mem_values
            delegate.memoshelve_keys = keys
            delegate.memoshelve_items = items
            delegate.memoshelve_values = values
            delegate.memoshelve_disk_keys = disk_keys
            delegate.memoshelve_disk_items = disk_items
            delegate.memoshelve_disk_values = disk_values

            yield delegate

    return open_db


def uncache(
    *args,
    filename: Path | str,
    cache: dict[str, dict[str, Any]] = memoshelve_cache,
    get_hash: Callable | None = None,
    get_hash_mem: Callable | None = None,
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

    with shelve.open(filename) as db:
        mkey = get_hash_mem((args, kwargs))
        if mkey in mem_db:
            del mem_db[mkey]
        key = get_hash((args, kwargs))
        if key in db:
            del db[key]


# for decorators
def sync_cache(
    filename: Path | str | None = None,
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
    disable: bool = False,
    copy: Callable[[T], T] = lambda x: x,
    allow_race: bool = True,
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
        )

        def wrapper_with_status(*args, **kwargs):
            if disable:
                return value(*args, **kwargs), False
            else:
                with memo() as f:
                    return f.__call_with_status__(*args, **kwargs)

        def wrapper_get_with_status(*args, **kwargs):
            if disable:
                return None, "miss"
            else:
                with memo() as f:
                    return f.get_with_status(*args, **kwargs)

        def wrapper_get(*args, **kwargs):
            if disable:
                return None
            else:
                with memo() as f:
                    return f.get(*args, **kwargs)

        def wrapper_contains(*args, **kwargs):
            if disable:
                return False
            else:
                with memo() as f:
                    return f.__contains__(*args, **kwargs)

        def wrapper_put(val, *args, **kwargs):
            if disable:
                return
            else:
                with memo() as f:
                    f.put(val, *args, **kwargs)

        def wrapper_memoshelve_mem_keys():
            if disable:
                return set()
            else:
                with memo() as f:
                    return f.memoshelve_mem_keys()

        def wrapper_memoshelve_mem_items():
            if disable:
                return ()
            else:
                with memo() as f:
                    return f.memoshelve_mem_items()

        def wrapper_memoshelve_mem_values():
            if disable:
                return ()
            else:
                with memo() as f:
                    return f.memoshelve_mem_values()

        def wrapper_memoshelve_keys():
            if disable:
                return ()
            else:
                with memo() as f:
                    return f.memoshelve_keys()

        def wrapper_memoshelve_items():
            if disable:
                return ()
            else:
                with memo() as f:
                    return f.memoshelve_items()

        def wrapper_memoshelve_values():
            if disable:
                return ()
            else:
                with memo() as f:
                    return f.memoshelve_values()

        def wrapper_memoshelve_disk_keys():
            if disable:
                return ()
            else:
                with memo() as f:
                    return f.memoshelve_disk_keys()

        def wrapper_memoshelve_disk_items():
            if disable:
                return ()
            else:
                with memo() as f:
                    return f.memoshelve_disk_items()

        def wrapper_memoshelve_disk_values():
            if disable:
                return ()
            else:
                with memo() as f:
                    return f.memoshelve_disk_values()

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
        )
        value.memoshelve_mem_keys = wrapper_memoshelve_mem_keys
        value.memoshelve_mem_items = wrapper_memoshelve_mem_items
        value.memoshelve_mem_values = wrapper_memoshelve_mem_values
        value.memoshelve_keys = wrapper_memoshelve_keys
        value.memoshelve_items = wrapper_memoshelve_items
        value.memoshelve_values = wrapper_memoshelve_values
        value.memoshelve_disk_keys = wrapper_memoshelve_disk_keys
        value.memoshelve_disk_items = wrapper_memoshelve_disk_items
        value.memoshelve_disk_values = wrapper_memoshelve_disk_values

        @wraps(value)
        def wrapper(*args, **kwargs):
            result, _status = wrapper_with_status(*args, **kwargs)
            return result

        return wrapper

    return wrap


def async_cache(
    filename: Path | str | None = None,
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
    disable: bool = False,
    copy: Callable[[T], T] = lambda x: x,
    allow_race: bool = True,
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
        )

        async def wrapper_with_status(*args, **kwargs):
            if disable:
                return await value(*args, **kwargs), "miss"
            else:
                async with memo() as f:
                    return await f.__call_with_status__(*args, **kwargs)

        async def wrapper_get_with_status(*args, **kwargs):
            if disable:
                return None, "miss"
            else:
                async with memo() as f:
                    return f.get_with_status(*args, **kwargs)

        async def wrapper_get(*args, **kwargs):
            if disable:
                return None
            else:
                async with memo() as f:
                    return f.get(*args, **kwargs)

        async def wrapper_contains(*args, **kwargs):
            if disable:
                return False
            else:
                async with memo() as f:
                    return f.__contains__(*args, **kwargs)

        async def wrapper_put(val, *args, **kwargs):
            if disable:
                return
            else:
                async with memo() as f:
                    f.put(val, *args, **kwargs)

        async def wrapper_memoshelve_mem_keys():
            if disable:
                return set()
            else:
                async with memo() as f:
                    return f.memoshelve_mem_keys()

        async def wrapper_memoshelve_mem_items():
            if disable:
                return ()
            else:
                async with memo() as f:
                    return f.memoshelve_mem_items()

        async def wrapper_memoshelve_mem_values():
            if disable:
                return ()
            else:
                async with memo() as f:
                    return f.memoshelve_mem_values()

        async def wrapper_memoshelve_keys():
            if disable:
                return ()
            else:
                async with memo() as f:
                    return f.memoshelve_keys()

        async def wrapper_memoshelve_items():
            if disable:
                return ()
            else:
                async with memo() as f:
                    return f.memoshelve_items()

        async def wrapper_memoshelve_values():
            if disable:
                return ()
            else:
                async with memo() as f:
                    return f.memoshelve_values()

        async def wrapper_memoshelve_disk_keys():
            if disable:
                return ()
            else:
                async with memo() as f:
                    return f.memoshelve_disk_keys()

        async def wrapper_memoshelve_disk_items():
            if disable:
                return ()
            else:
                async with memo() as f:
                    return f.memoshelve_disk_items()

        async def wrapper_memoshelve_disk_values():
            if disable:
                return ()
            else:
                async with memo() as f:
                    return f.memoshelve_disk_values()

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
        )
        value.memoshelve_mem_keys = wrapper_memoshelve_mem_keys
        value.memoshelve_mem_items = wrapper_memoshelve_mem_items
        value.memoshelve_mem_values = wrapper_memoshelve_mem_values
        value.memoshelve_keys = wrapper_memoshelve_keys
        value.memoshelve_items = wrapper_memoshelve_items
        value.memoshelve_values = wrapper_memoshelve_values
        value.memoshelve_disk_keys = wrapper_memoshelve_disk_keys
        value.memoshelve_disk_items = wrapper_memoshelve_disk_items
        value.memoshelve_disk_values = wrapper_memoshelve_disk_values

        @wraps(value)
        async def wrapper(*args, **kwargs):
            result, _status = await wrapper_with_status(*args, **kwargs)
            return result

        return wrapper

    return wrap


def cache(
    filename: Path | str | None = None,
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
    disable: bool = False,
    copy: Callable[[T], T] = lambda x: x,
    allow_race: bool = True,
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
            )(func)

    return decorator
