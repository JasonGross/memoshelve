# memoshelve

[![PyPI version](https://badge.fury.io/py/memoshelve.svg)](https://badge.fury.io/py/memoshelve)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent on-disk caching decorator based on Python's `shelve` module, featuring a two-tier caching system (memory + disk) for optimal performance.

## ✨ Features

- **Two-tier caching**: Lightning-fast in-memory cache with persistent disk storage
- **Async/sync support**: Works seamlessly with both synchronous and asynchronous functions
- **Robust serialization**: Uses `dill` for advanced pickling and `stablehash` for consistent hashing
- **Cache inspection**: Rich API for examining and manipulating cache contents
- **Flexible configuration**: Extensive customization options for logging, hashing, and storage
- **Race condition handling**: Built-in support for concurrent access patterns
- **Cache management**: Automatic compaction, backup, and upgrade utilities

## 🚀 Quick Start

### Installation

```bash
pip install memoshelve
```

For enhanced functionality with robust serialization:

```bash
pip install memoshelve[robust]
```

### Basic Usage

```python
from memoshelve import cache

@cache(filename="my_cache.db")
def expensive_computation(x, y):
    # Simulate expensive work
    import time
    time.sleep(1)
    return x * y + 42

# First call: computed and cached
result = expensive_computation(10, 20)  # Takes ~1 second

# Second call: retrieved from cache
result = expensive_computation(10, 20)  # Nearly instant!
```

### Context Manager

```python
from memoshelve import memoshelve

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

with memoshelve(fibonacci, "fib_cache.db") as cached_fib:
    result = cached_fib(30)  # Cached across calls
    print(f"fib(30) = {result}")
```

### Async Support

```python
import asyncio
from memoshelve import cache

@cache(filename="async_cache.db")
async def async_expensive_task(data):
    await asyncio.sleep(1)  # Simulate async I/O
    return len(data) * 42

async def main():
    result = await async_expensive_task("hello")  # Cached automatically
    print(result)

asyncio.run(main())
```

## 📖 API Reference

### Decorator: `@cache`

The main decorator that automatically chooses between sync and async caching:

```python
@cache(
    filename="cache.db",          # Cache file path
    get_hash=None,                # Custom hash function
    disable=False,                # Disable caching entirely
    ignore=(),                    # Ignore specific parameters
    copy=lambda x: x,             # Custom copy function
    allow_race=True,              # Allow concurrent access
    print_cache_miss=None,        # Log cache misses
    print_cache_hit=None,         # Log cache hits
)
def my_function(arg1, arg2):
    return expensive_operation(arg1, arg2)
```

### Context Managers

#### Synchronous Context Manager

```python
from memoshelve import memoshelve

with memoshelve(my_function, "cache.db") as cached_fn:
    result = cached_fn(args)
```

#### Asynchronous Context Manager

```python
from memoshelve import async_memoshelve

async with async_memoshelve(my_async_function, "cache.db") as cached_fn:
    result = await cached_fn(args)
```

### Enhanced Function Methods

Cached functions gain additional methods for inspection and control:

```python
@cache(filename="example.db")
def compute(x, y):
    return x ** y

# Check if arguments are cached
if compute.__contains__(2, 3):
    print("Result is cached!")

# Get cached result without computing
try:
    result = compute.get(2, 3)
except KeyError:
    print("Not in cache")

# Get result with cache status
result, status = compute.__call_with_status__(2, 3)
print(f"Result: {result}, Status: {status}")
# Status can be: "cached (mem)", "cached (disk)", or "miss"

# Manually store in cache
compute.put(2, 3, 8)

# Remove from cache
compute.uncache(2, 3)

# Access cache metadata
print(f"Cache keys: {list(compute.memoshelve.keys())}")
print(f"Cache values: {list(compute.memoshelve.values())}")
```

## 🔧 Advanced Usage

### Custom Hashing

```python
import hashlib
from memoshelve import cache

def custom_hash(obj):
    return hashlib.md5(str(obj).encode()).hexdigest()

@cache(filename="custom_hash.db", get_hash=custom_hash)
def my_function(data):
    return process_data(data)
```

### Ignoring Parameters

```python
@cache(filename="ignore_params.db", ignore=["timestamp", "debug"])
def process_data(data, timestamp=None, debug=False):
    # timestamp and debug won't affect cache key
    return expensive_processing(data)
```

### Cache Management

```python
from memoshelve import compact

# Compact cache file to remove unused space
compact("my_cache.db", backup=True)

# Access cache metadata directly
@cache(filename="metadata_example.db")
def example_func(x):
    return x * 2

# Inspect cache contents
metadata = example_func.memoshelve
print(f"Disk keys: {metadata.disk_keys()}")
print(f"Memory keys: {metadata.mem_keys()}")
print(f"All items: {dict(metadata.items())}")

# Compact the cache
metadata.compact(backup=True)
```

### Error Handling and Logging

```python
import logging
from memoshelve import cache

# Configure logging
logging.basicConfig(level=logging.INFO)

@cache(
    filename="logged_cache.db",
    print_cache_miss=True,        # Log cache misses
    print_cache_hit=True,         # Log cache hits
    print_disk_cache_miss=True,   # Log disk cache misses
    print_mem_cache_miss=True,    # Log memory cache misses
)
def logged_function(x):
    return expensive_computation(x)
```

## 🏎️ Performance Tips

1. **Use `stablehash`**: Install with `pip install stablehash` for consistent, fast hashing
2. **Configure memory limits**: Monitor memory usage for large cache sizes
3. **Regular compaction**: Use `compact()` to optimize disk cache performance
4. **Strategic `ignore` parameters**: Exclude non-essential parameters from cache keys
5. **Consider `copy` function**: Use appropriate copying for mutable return values

## 🛠️ Cache Storage

### Default Locations

- **Linux/macOS**: `~/.cache/memoshelve/`
- **Configurable**: Set `XDG_CACHE_HOME` environment variable

### File Formats

Cache files use Python's `shelve` module, which may create multiple files:
- `cache.db` (main file)
- Platform-specific extensions (`.dir`, `.dat`, `.bak`)

## 🤝 Contributing

We welcome contributions! Please check out our [contributing guidelines](CONTRIBUTING.md).

### Development Setup

```bash
git clone https://github.com/jxnl/memoshelve.git
cd memoshelve
pip install -e ".[dev]"
pre-commit install
```

### Running Tests

```bash
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on Python's robust `shelve` module
- Enhanced with `dill` for advanced serialization
- Uses `stablehash` for consistent hashing across runs
- Inspired by the need for persistent, high-performance caching solutions

## 📚 Related Projects

- [`functools.lru_cache`](https://docs.python.org/3/library/functools.html#functools.lru_cache) - In-memory caching
- [`joblib.Memory`](https://joblib.readthedocs.io/en/latest/memory.html) - Disk-based caching for scientific computing
- [`diskcache`](https://pypi.org/project/diskcache/) - Alternative disk caching solution
