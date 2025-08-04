"""
Pytest test harness for memoshelve module.

Requirements:
    - pytest
    - pytest-asyncio

Optional (for full functionality):
    - dill
    - stablehash

Note: Some tests may need adjustment based on the platform's shelve implementation.
Different platforms may use different database backends (.db, .dat, .dir files).
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

# Import the module to test
import memoshelve
from memoshelve import (
    cache,
    compact,
    memoshelve as memoshelve_context,
    async_memoshelve,
    backup_file,
    next_backup_ext,
)


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestBackupFunctions:
    """Test backup utility functions."""

    def test_next_backup_ext(self):
        """Test backup extension generation."""
        # Only increments if everything after first char is digits
        assert next_backup_ext(".bak") == (".bak", False)
        assert next_backup_ext(".1") == (".2", True)
        assert next_backup_ext(".99") == (".100", True)
        assert next_backup_ext(".bak1") == (".bak1", False)  # Not all digits
        assert next_backup_ext(".backup", strip_suffix=True) == (".backup.backup", True)

    def test_backup_file(self, tmp_path):
        """Test file backup functionality."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("original content")

        # First backup
        backup1 = backup_file(test_file)
        assert backup1 == test_file.with_suffix(".txt.bak")
        assert backup1.exists()
        assert backup1.read_text() == "original content"
        assert not test_file.exists()

        # Restore original file and create backup again
        test_file.write_text("new content")
        backup2 = backup_file(test_file)
        # Since .bak already exists, it should create a different backup
        assert backup2 == test_file.with_suffix(".txt.bak")
        # The previous backup should have been renamed (appending .bak)
        expected_old_backup = backup1.parent / (backup1.name + ".bak")
        assert expected_old_backup.exists()

        # Test with non-existent file
        result = backup_file(tmp_path / "nonexistent.txt")
        assert result is None


class TestBasicCaching:
    """Test basic caching functionality."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def cache_file(self, temp_cache_dir):
        """Create a cache file path."""
        return temp_cache_dir / "test_cache.shelve"

    def test_simple_cache_decorator(self, cache_file):
        """Test basic cache decorator functionality."""
        call_count = 0

        @cache(filename=cache_file)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y

        # First call - cache miss
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1

        # Second call - cache hit
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Function not called again

        # Different arguments - cache miss
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2

    def test_cache_with_status(self, cache_file):
        """Test cache status tracking."""

        @cache(filename=cache_file)
        def add(x, y):
            return x + y

        # First call - miss
        result, status = add.__call_with_status__(1, 2)
        assert result == 3
        assert status == "miss"

        # Second call - memory cache hit
        result, status = add.__call_with_status__(1, 2)
        assert result == 3
        assert status == "cached (mem)"

        # Clear the specific memory cache for this file
        cache_key = str(cache_file.absolute())
        if cache_key in memoshelve.memoshelve_cache:
            memoshelve.memoshelve_cache[cache_key].clear()

        # Now should hit disk cache
        result, status = add.__call_with_status__(1, 2)
        assert result == 3
        assert status == "cached (disk)"

    def test_cache_operations(self, cache_file):
        """Test various cache operations."""

        @cache(filename=cache_file)
        def multiply(x, y):
            return x * y

        # Test contains
        assert not multiply.__contains__(3, 4)
        multiply(3, 4)
        assert multiply.__contains__(3, 4)

        # Test get - for uncached values, get returns the key tuple
        uncached_result = multiply.get(5, 6)
        # When not cached, it returns a tuple of keys
        assert isinstance(uncached_result, tuple)

        # Cache a value and then get it
        multiply(5, 6)
        assert multiply.get(5, 6) == 30

        # Test put
        multiply.put(100, 7, 8)
        assert multiply.get(7, 8) == 100

        # Test uncache
        multiply.uncache(3, 4)
        assert not multiply.__contains__(3, 4)

    def test_context_manager_usage(self, cache_file):
        """Test using memoshelve as a context manager."""

        def expensive_calc(x):
            return x**2

        # memoshelve returns a context manager factory
        memo_factory = memoshelve_context(expensive_calc, cache_file)

        with memo_factory() as cached_fn:
            # First call
            result1, status1 = cached_fn.__call_with_status__(5)
            assert result1 == 25
            assert status1 == "miss"

            # Second call
            result2, status2 = cached_fn.__call_with_status__(5)
            assert result2 == 25
            assert status2 == "cached (mem)"

    def test_disabled_cache(self, cache_file):
        """Test disabled cache functionality."""
        call_count = 0

        @cache(filename=cache_file, disable=True)
        def increment_counter(x):
            nonlocal call_count
            call_count += 1
            return x + 1

        # Cache is disabled, function should be called every time
        result1 = increment_counter(5)
        result2 = increment_counter(5)
        assert result1 == result2 == 6
        assert call_count == 2


class TestAsyncCaching:
    """Test async function caching."""

    @pytest.mark.asyncio
    async def test_async_cache_decorator(self, temp_cache_dir):
        """Test async cache decorator."""
        cache_file = temp_cache_dir / "async_cache.shelve"
        call_count = 0

        @cache(filename=cache_file)
        async def async_expensive(x, y):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate async work
            return x * y

        # First call - miss
        result1 = await async_expensive(3, 4)
        assert result1 == 12
        assert call_count == 1

        # Second call - hit
        result2 = await async_expensive(3, 4)
        assert result2 == 12
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_context_manager(self, temp_cache_dir):
        """Test async context manager usage."""
        cache_file = temp_cache_dir / "async_context.shelve"

        async def async_calc(x):
            await asyncio.sleep(0.01)
            return x**3

        # async_memoshelve also returns a context manager factory
        memo_factory = async_memoshelve(async_calc, cache_file)

        async with memo_factory() as cached_fn:
            result1, status1 = await cached_fn.__call_with_status__(4)
            assert result1 == 64
            assert status1 == "miss"

            result2, status2 = await cached_fn.__call_with_status__(4)
            assert result2 == 64
            assert status2 == "cached (mem)"


class TestHashFunctions:
    """Test custom hash functions."""

    def test_custom_hash_function(self, temp_cache_dir):
        """Test using custom hash functions."""
        cache_file = temp_cache_dir / "custom_hash.shelve"

        def simple_hash(args_kwargs):
            args, kwargs = args_kwargs
            return f"custom_{args}_{kwargs}"

        @cache(filename=cache_file, get_hash=simple_hash)
        def add(x, y):
            return x + y

        result = add(1, 2)
        assert result == 3

        # Check that custom hash was used
        memo_factory = memoshelve_context(add, cache_file, get_hash=simple_hash)
        with memo_factory() as cached_fn:
            assert cached_fn.__contains__(1, 2)

    def test_separate_mem_disk_hash(self, temp_cache_dir):
        """Test using different hash functions for memory and disk."""
        cache_file = temp_cache_dir / "dual_hash.shelve"

        def mem_hash(args_kwargs):
            return f"mem_{id(args_kwargs)}"

        def disk_hash(args_kwargs):
            args, _ = args_kwargs
            return f"disk_{args}"

        @cache(filename=cache_file, get_hash=disk_hash, get_hash_mem=mem_hash)
        def multiply(x, y):
            return x * y

        result = multiply(3, 4)
        assert result == 12


class TestPrintingAndLogging:
    """Test cache miss/hit printing functionality."""

    def test_print_configuration(self, temp_cache_dir, capsys):
        """Test various print configuration options."""
        cache_file = temp_cache_dir / "print_test.shelve"

        # Custom print function
        printed_messages = []

        def custom_print(msg):
            printed_messages.append(msg)

        # Temporarily change the default cache hit function to print
        import memoshelve

        original_default = memoshelve.DEFAULT_PRINT_CACHE_HIT_FN
        memoshelve.DEFAULT_PRINT_CACHE_HIT_FN = print

        try:

            @cache(
                filename=cache_file,
                print_disk_cache_miss=custom_print,
                print_mem_cache_hit=True,
            )
            def compute(x):
                return x * 2

            # First call - disk cache miss
            compute(5)
            assert len(printed_messages) == 1
            assert "Cache miss" in printed_messages[0]

            # Second call - mem cache hit (should print to stdout)
            compute(5)
            captured = capsys.readouterr()
            assert (
                "Cache hit (mem)" in captured.out or "Cache hit (mem)" in captured.err
            )
        finally:
            # Reset the default
            memoshelve.DEFAULT_PRINT_CACHE_HIT_FN = original_default


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_corrupted_cache_compact(self, temp_cache_dir):
        """Test compacting corrupted cache."""
        import shelve

        cache_file = str(temp_cache_dir / "corrupted.shelve")

        # Create a cache with some entries
        with shelve.open(cache_file) as db:
            db["good_key"] = "good_value"
            # In real scenario, corruption would happen differently
            # This is just to test the compact function logic

        # Test compact function
        compact(cache_file, backup=True)

        # Verify the cache still works
        with shelve.open(cache_file) as db:
            assert db.get("good_key") == "good_value"

    def test_missing_dependencies(self, temp_cache_dir):
        """Test behavior with missing optional dependencies."""
        cache_file = temp_cache_dir / "no_deps.shelve"

        # Test with missing stablehash (falls back to repr)
        with patch.dict("sys.modules", {"stablehash": None}):

            @cache(filename=cache_file)
            def add(x, y):
                return x + y

            result = add(1, 2)
            assert result == 3

    def test_concurrent_access(self, temp_cache_dir):
        """Test concurrent access handling."""
        cache_file = temp_cache_dir / "concurrent.shelve"

        @cache(filename=cache_file, allow_race=True)
        def slow_function(x):
            return x**2

        # This would test race conditions in real concurrent scenario
        result = slow_function(4)
        assert result == 16

    def test_kwargs_caching(self, temp_cache_dir):
        """Test caching with keyword arguments."""
        cache_file = temp_cache_dir / "kwargs.shelve"

        @cache(filename=cache_file)
        def func_with_kwargs(a, b=10, c=20):
            return a + b + c

        # Different ways to call should cache separately
        result1 = func_with_kwargs(1)
        result2 = func_with_kwargs(1, b=10)
        result3 = func_with_kwargs(1, b=10, c=20)

        assert result1 == result2 == result3 == 31

        # But with different values
        result4 = func_with_kwargs(1, b=20)
        assert result4 == 41


class TestCopyFunction:
    """Test copy function functionality."""

    def test_copy_function(self, temp_cache_dir):
        """Test using a custom copy function."""
        cache_file = temp_cache_dir / "copy_test.shelve"

        class MutableResult:
            def __init__(self, value):
                self.value = value

            def copy(self):
                return MutableResult(self.value)

        @cache(filename=cache_file, copy=lambda x: x.copy())
        def create_mutable(x):
            return MutableResult(x)

        # Get cached result
        result1 = create_mutable(5)
        result2 = create_mutable(5)

        # Modify one shouldn't affect the other (due to copy)
        result1.value = 10
        assert result2.value == 5


class TestCacheInspection:
    """Test cache inspection methods."""

    def test_cache_keys_values_items(self, temp_cache_dir):
        """Test inspecting cache contents."""
        cache_file = temp_cache_dir / "inspect.shelve"

        @cache(filename=cache_file)
        def square(x):
            return x**2

        # Add some values
        square(2)
        square(3)
        square(4)

        # Test inspection methods
        keys = square.memoshelve.keys()
        assert len(keys) >= 3

        values = list(square.memoshelve.values())
        assert 4 in values
        assert 9 in values
        assert 16 in values

        # Test disk-specific methods
        disk_keys = square.memoshelve.disk_keys()
        assert len(list(disk_keys)) >= 3


class TestDefaultCacheDir:
    """Test default cache directory behavior."""

    def test_default_cache_dir(self):
        """Test that default cache dir is created and used."""
        # Save original cache dir
        original_cache_dir = memoshelve.DEFAULT_CACHE_DIR

        try:
            # Use temp dir as default
            with tempfile.TemporaryDirectory() as tmpdir:
                memoshelve.DEFAULT_CACHE_DIR = Path(tmpdir)

                @cache()  # No filename specified
                def auto_cached(x):
                    return x + 1

                result = auto_cached(5)
                assert result == 6

                # Check that cache file was created (shelve may create multiple files)
                _expected_file = Path(tmpdir) / "auto_cached.shelve"
                # Check for any shelve-related files
                cache_files = list(Path(tmpdir).glob("auto_cached.shelve*"))
                assert len(cache_files) > 0, "No cache files were created"
        finally:
            # Restore original
            memoshelve.DEFAULT_CACHE_DIR = original_cache_dir


class TestExtendedFeatures:
    """Test extended features and configurations."""

    def test_extended_traceback_info(self, temp_cache_dir, capsys):
        """Test extended cache miss information."""
        cache_file = temp_cache_dir / "extended.shelve"
        printed = []

        @cache(
            filename=cache_file,
            print_extended_cache_miss_disk=True,
            print_disk_cache_miss=lambda x: printed.append(x),
        )
        def traced_func(x):
            return x * 3

        traced_func(7)
        assert len(printed) == 1
        assert "test_memoshelve.py" in printed[0]  # Should contain traceback info

    def test_global_cache_dict(self, temp_cache_dir):
        """Test using a custom global cache dictionary."""
        cache_file = temp_cache_dir / "custom_dict.shelve"
        custom_cache = {}

        @cache(filename=cache_file, cache=custom_cache)
        def custom_cached(x):
            return x - 1

        result = custom_cached(10)
        assert result == 9
        assert str(cache_file) in custom_cache


class TestIgnoreParameter:
    """Test the ignore parameter functionality."""

    def test_basic_ignore(self, temp_cache_dir):
        """Test basic ignore functionality with positional arguments."""
        cache_file = temp_cache_dir / "ignore_basic.shelve"
        call_count = 0

        @cache(filename=cache_file, ignore=["y"])
        def compute(x, y, z):
            nonlocal call_count
            call_count += 1
            return x + y + z

        # First call
        result1 = compute(1, 2, 3)
        assert result1 == 6
        assert call_count == 1

        # Second call with different y - should hit cache since y is ignored
        result2 = compute(1, 99, 3)
        assert result2 == 6  # Returns cached value
        assert call_count == 1  # Function not called again

        # Third call with different x - should miss cache
        result3 = compute(2, 2, 3)
        assert result3 == 7
        assert call_count == 2

    def test_ignore_kwargs(self, temp_cache_dir):
        """Test ignoring keyword arguments."""
        cache_file = temp_cache_dir / "ignore_kwargs.shelve"
        call_count = 0

        @cache(filename=cache_file, ignore=["debug", "verbose"])
        def process_data(data, debug=False, verbose=False, mode="fast"):
            nonlocal call_count
            call_count += 1
            if debug:
                print(f"Processing {data}")
            return len(data) * (2 if mode == "fast" else 3)

        # First call
        result1 = process_data("hello", debug=True, verbose=True)
        assert result1 == 10
        assert call_count == 1

        # Second call with different ignored kwargs - should hit cache
        result2 = process_data("hello", debug=False, verbose=False)
        assert result2 == 10
        assert call_count == 1

        # Third call with different non-ignored kwarg - should miss cache
        result3 = process_data("hello", mode="slow")
        assert result3 == 15
        assert call_count == 2

    def test_ignore_mixed_args(self, temp_cache_dir):
        """Test ignoring with mixed positional and keyword arguments."""
        cache_file = temp_cache_dir / "ignore_mixed.shelve"
        call_count = 0

        @cache(filename=cache_file, ignore=["b", "d"])
        def complex_func(a, b, c=10, d=20):
            nonlocal call_count
            call_count += 1
            return a * b + c * d

        # Test various calling patterns
        result1 = complex_func(2, 3, c=10, d=20)
        assert result1 == 206
        assert call_count == 1

        # Different b and d - should hit cache
        result2 = complex_func(2, 99, c=10, d=99)
        assert result2 == 206
        assert call_count == 1

        # Different a - should miss cache
        result3 = complex_func(3, 3, c=10, d=20)
        assert result3 == 209
        assert call_count == 2

        # Different c - should miss cache
        result4 = complex_func(2, 3, c=20, d=20)
        assert result4 == 406
        assert call_count == 3

    def test_ignore_all_args(self, temp_cache_dir):
        """Test ignoring all arguments (edge case)."""
        cache_file = temp_cache_dir / "ignore_all.shelve"
        call_count = 0

        @cache(filename=cache_file, ignore=["x", "y"])
        def always_cached(x, y):
            nonlocal call_count
            call_count += 1
            return call_count  # Return call count to verify caching

        # All calls should return the same cached value
        assert always_cached(1, 2) == 1
        assert always_cached(3, 4) == 1
        assert always_cached(99, 99) == 1
        assert call_count == 1

    def test_ignore_nonexistent_arg(self, temp_cache_dir):
        """Test ignoring argument names that don't exist."""
        cache_file = temp_cache_dir / "ignore_nonexistent.shelve"

        @cache(filename=cache_file, ignore=["nonexistent", "y"])
        def func(x, y):
            return x + y

        # Should work normally, ignoring the nonexistent parameter name
        result1 = func(1, 2)
        assert result1 == 3

        # Different y should hit cache
        result2 = func(1, 99)
        assert result2 == 3

    def test_always_bind_parameter(self, temp_cache_dir):
        """Test the always_bind parameter."""
        cache_file = temp_cache_dir / "always_bind.shelve"
        call_count = 0

        @cache(filename=cache_file, always_bind=True)
        def func_with_defaults(a, b=10, c=20):
            nonlocal call_count
            call_count += 1
            return a + b + c

        # These should all be treated as the same call due to always_bind
        result1 = func_with_defaults(1)
        assert result1 == 31
        assert call_count == 1

        result2 = func_with_defaults(1, 10)
        assert result2 == 31
        assert call_count == 1  # Should hit cache

        result3 = func_with_defaults(1, b=10, c=20)
        assert result3 == 31
        assert call_count == 1  # Should hit cache

        # Different actual values should miss cache
        result4 = func_with_defaults(1, 20)
        assert result4 == 41
        assert call_count == 2

    def test_ignore_with_always_bind(self, temp_cache_dir):
        """Test using ignore and always_bind together."""
        cache_file = temp_cache_dir / "ignore_always_bind.shelve"
        call_count = 0

        @cache(filename=cache_file, ignore=["debug"], always_bind=True)
        def process(data, mode="fast", debug=False):
            nonlocal call_count
            call_count += 1
            return f"{data}_{mode}"

        # All these should hit the same cache entry
        result1 = process("test", debug=True)
        assert result1 == "test_fast"
        assert call_count == 1

        result2 = process("test", "fast", False)
        assert result2 == "test_fast"
        assert call_count == 1

        result3 = process("test", mode="fast", debug=False)
        assert result3 == "test_fast"
        assert call_count == 1

        # Different mode should miss cache
        result4 = process("test", mode="slow", debug=True)
        assert result4 == "test_slow"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_ignore(self, temp_cache_dir):
        """Test ignore parameter with async functions."""
        cache_file = temp_cache_dir / "async_ignore.shelve"
        call_count = 0

        @cache(filename=cache_file, ignore=["delay"])
        async def async_compute(x, y, delay=0.01):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(delay)
            return x * y

        # First call
        result1 = await async_compute(3, 4, delay=0.01)
        assert result1 == 12
        assert call_count == 1

        # Second call with different delay - should hit cache
        result2 = await async_compute(3, 4, delay=1.0)
        assert result2 == 12
        assert call_count == 1

        # Third call with different x - should miss cache
        result3 = await async_compute(4, 4, delay=0.01)
        assert result3 == 16
        assert call_count == 2

    def test_ignore_with_varargs(self, temp_cache_dir):
        """Test ignore parameter with *args."""
        cache_file = temp_cache_dir / "ignore_varargs.shelve"
        call_count = 0

        @cache(filename=cache_file, ignore=["b"])
        def func_varargs(a, b, *args):
            nonlocal call_count
            call_count += 1
            return a + b + sum(args)

        # First call
        result1 = func_varargs(1, 2, 3, 4)
        assert result1 == 10
        assert call_count == 1

        # Different b - should hit cache
        result2 = func_varargs(1, 99, 3, 4)
        assert result2 == 10
        assert call_count == 1

        # Different varargs - should miss cache
        result3 = func_varargs(1, 2, 5, 6)
        assert result3 == 14
        assert call_count == 2

    def test_ignore_with_status(self, temp_cache_dir):
        """Test that ignore works correctly with status tracking."""
        cache_file = temp_cache_dir / "ignore_status.shelve"

        @cache(filename=cache_file, ignore=["timestamp"])
        def log_event(event_type, message, timestamp=None):
            return f"{event_type}: {message}"

        # First call
        result1, status1 = log_event.__call_with_status__(
            "INFO", "Started", timestamp=12345
        )
        assert result1 == "INFO: Started"
        assert status1 == "miss"

        # Second call with different timestamp - should hit cache
        result2, status2 = log_event.__call_with_status__(
            "INFO", "Started", timestamp=67890
        )
        assert result2 == "INFO: Started"
        assert status2 == "cached (mem)"

    def test_ignore_with_put_operation(self, temp_cache_dir):
        """Test that put operation works correctly with ignore."""
        cache_file = temp_cache_dir / "ignore_put.shelve"

        @cache(filename=cache_file, ignore=["version"])
        def compute(x, version=1):
            return x * 2

        # Manually put a value
        compute.put(100, 5, version=1)

        # Should retrieve the put value even with different version
        result = compute(5, version=99)
        assert result == 100

    def test_ignore_type_error_handling(self, temp_cache_dir):
        """Test handling of TypeError when binding fails."""
        cache_file = temp_cache_dir / "ignore_type_error.shelve"
        call_count = 0

        @cache(filename=cache_file, ignore=["extra"], always_bind=True)
        def strict_func(a, b):
            nonlocal call_count
            call_count += 1
            return a + b

        # This should raise a TypeError when trying to bind
        # extra arguments that the function doesn't accept
        with pytest.raises(TypeError):
            strict_func(1, 2, extra=99)  # Extra kwarg not in signature

        # Should still cache properly for valid calls
        result2 = strict_func(1, 2)
        assert result2 == 3
        assert call_count == 1  # Should hit cache


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
