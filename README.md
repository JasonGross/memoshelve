# memoshelve

A persistent memoization decorator using Python's `shelve` with two-tier caching (memory + disk).

## Features

- Two-tier caching: in-memory + persistent disk storage
- Async and sync function support
- Cache inspection and management
- Optional enhanced serialization with `dill` and `stablehash`

## Installation

```bash
pip install memoshelve

# For enhanced serialization
pip install memoshelve[robust]
```

## Usage

### Basic Decorator

```python
from memoshelve import cache

@cache(filename="cache.db")
def expensive_function(x, y):
    return x * y + 42

result = expensive_function(10, 20)  # Computed and cached
result = expensive_function(10, 20)  # Retrieved from cache
```

### Context Manager

```python
from memoshelve import memoshelve

with memoshelve(expensive_function, "cache.db") as cached_fn:
    result = cached_fn(10, 20)
```

### Async Functions

```python
@cache(filename="async_cache.db")
async def async_function(data):
    return len(data) * 42
```

## API

### Cache Methods

```python
@cache(filename="example.db")
def compute(x, y):
    return x ** y

# Check if cached
compute.__contains__(2, 3)

# Get without computing
compute.get(2, 3)  # Raises KeyError if not cached

# Get with status
result, status = compute.__call_with_status__(2, 3)
# status: "cached (mem)", "cached (disk)", or "miss"

# Manual operations
compute.put(2, 3, 8)      # Store value
compute.uncache(2, 3)     # Remove from cache
```

### Configuration

```python
@cache(
    filename="cache.db",
    ignore=["debug"],         # Ignore parameters in cache key
    get_hash=custom_hash,     # Custom hash function
    disable=False,            # Toggle caching
    print_cache_miss=True,    # Log cache misses
)
def my_function(data, debug=False):
    return process(data)
```

### Cache Management

```python
from memoshelve import compact

# Compact cache file
compact("cache.db", backup=True)

# Access metadata
metadata = my_function.memoshelve
metadata.disk_keys()     # Keys in disk cache
metadata.mem_keys()      # Keys in memory cache
metadata.compact()       # Compact this cache
```

## Storage

Default cache location: `~/.cache/memoshelve/` (configurable via `XDG_CACHE_HOME`)

Cache files use Python's `shelve` module and may create multiple files (`.db`, `.dir`, `.dat`).

## License

MIT License
