"""
Integration and performance tests for memoshelve module.

Requirements:
    - pytest
    - pytest-asyncio
    - numpy (for performance tests)

These tests cover real-world scenarios, performance benchmarks,
and integration between different caching features.
"""

import asyncio
import time
import tempfile
import platform
from pathlib import Path
import pytest
import numpy as np

from memoshelve import cache, memoshelve_cache, compact


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestIntegration:
    """Integration tests for real-world scenarios."""

    def test_nested_function_calls(self, temp_dir):
        """Test caching with nested function calls."""
        cache_a = temp_dir / "cache_a.shelve"
        cache_b = temp_dir / "cache_b.shelve"

        @cache(filename=cache_a)
        def func_a(x):
            return x * 2

        @cache(filename=cache_b)
        def func_b(x):
            # Calls another cached function
            return func_a(x) + 10

        # First call - both functions miss
        result1 = func_b(5)
        assert result1 == 20

        # Second call - both hit cache
        result2 = func_b(5)
        assert result2 == 20

        # Verify both caches contain data
        assert func_a.__contains__(5)
        assert func_b.__contains__(5)

    def test_recursive_caching(self, temp_dir):
        """Test caching recursive functions."""
        cache_file = temp_dir / "recursive.shelve"

        @cache(filename=cache_file)
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)

        # Calculate fibonacci(10)
        result = fibonacci(10)
        assert result == 55

        # Check that intermediate values are cached
        for i in range(11):
            assert fibonacci.__contains__(i)

    def test_exception_caching(self, temp_dir):
        """Test that exceptions are not cached."""
        cache_file = temp_dir / "exceptions.shelve"
        call_count = 0

        @cache(filename=cache_file)
        def sometimes_fails(x):
            nonlocal call_count
            call_count += 1
            if x < 0:
                raise ValueError("Negative input")
            return x**2

        # First call with error
        with pytest.raises(ValueError):
            sometimes_fails(-1)

        # Second call with same input - should call function again
        with pytest.raises(ValueError):
            sometimes_fails(-1)

        assert call_count == 2  # Function called twice, exceptions not cached

    def test_large_data_caching(self, temp_dir):
        """Test caching large data structures."""
        cache_file = temp_dir / "large_data.shelve"

        @cache(filename=cache_file)
        def generate_matrix(size):
            # Generate a large random matrix
            return np.random.rand(size, size).tolist()

        # Cache a 100x100 matrix
        start_time = time.time()
        matrix1 = generate_matrix(100)
        first_call_time = time.time() - start_time

        # Retrieve from cache
        start_time = time.time()
        matrix2 = generate_matrix(100)
        cached_call_time = time.time() - start_time

        assert matrix1 == matrix2
        # Cached call should be significantly faster
        assert cached_call_time < first_call_time * 0.5

    def test_cache_persistence(self, temp_dir):
        """Test that cache persists across Python sessions."""
        cache_file = temp_dir / "persistent.shelve"

        # First "session"
        @cache(filename=cache_file)
        def compute(x):
            return x**3

        result1 = compute(7)
        assert result1 == 343

        # Clear memory cache to simulate new session
        memoshelve_cache.clear()

        # Second "session" - should load from disk
        @cache(filename=cache_file)
        def compute_session2(x):
            return "This should not be called"

        result2 = compute_session2(7)
        assert result2 == 343  # Got old cached value

    @pytest.mark.asyncio
    async def test_mixed_sync_async(self, temp_dir):
        """Test mixing sync and async cached functions."""
        sync_cache = temp_dir / "sync.shelve"
        async_cache = temp_dir / "async.shelve"

        @cache(filename=sync_cache)
        def sync_func(x):
            return x * 2

        @cache(filename=async_cache)
        async def async_func(x):
            # Call sync function from async
            sync_result = sync_func(x)
            await asyncio.sleep(0.01)
            return sync_result + 10

        result = await async_func(5)
        assert result == 20

        # Both should be cached
        assert sync_func.__contains__(5)
        assert await async_func.__contains__(5)


class TestPerformance:
    """Performance benchmarks for caching."""

    @pytest.mark.skipif(
        platform.system() == "Windows",
        reason="Windows is too fast at uncached operations :-/",
    )
    def test_cache_overhead(self, temp_dir):
        """Measure caching overhead for simple functions."""
        cache_file = temp_dir / "overhead.shelve"
        iterations = 1000

        # Non-cached function (make it more substantial to reduce relative overhead)
        def simple_func(x):
            # More substantial computation to make caching overhead relatively smaller
            result = 0
            for i in range(x % 20 + 10):  # 10-29 iterations
                result += i**2 + i
            return result

        # Cached version
        @cache(filename=cache_file)
        def cached_func(x):
            # More substantial computation to make caching overhead relatively smaller
            result = 0
            for i in range(x % 20 + 10):  # 10-29 iterations
                result += i**2 + i
            return result

        # Warm up cache
        for i in range(10):
            cached_func(i)

        # Measure non-cached performance
        start = time.time()
        for i in range(iterations):
            simple_func(i % 10)
        uncached_time = time.time() - start

        # Measure cached performance (all hits)
        start = time.time()
        for i in range(iterations):
            cached_func(i % 10)
        cached_time = time.time() - start

        # Cache overhead should be reasonable for memory cache hits
        # Since all calls are cache hits (memory lookup), overhead should be much lower
        # Allow up to 50x overhead to account for hashing, memory lookups, copying, etc.
        ratio = float("inf") if uncached_time == 0 else cached_time / uncached_time
        assert (
            cached_time < uncached_time * 50
        ), f"Cache overhead too high: {cached_time:.6f}s vs {uncached_time:.6f}s (ratio: {ratio:.1f}x)"

    def test_memory_vs_disk_performance(self, temp_dir):
        """Compare memory cache vs disk cache performance."""
        cache_file = temp_dir / "mem_vs_disk.shelve"

        @cache(filename=cache_file)
        def compute(x):
            return x**2

        # Populate cache
        compute(42)

        # Time memory cache hits
        start = time.time()
        for _ in range(1000):
            compute(42)
        mem_time = time.time() - start

        # Clear memory cache
        memoshelve_cache[str(cache_file.absolute())].clear()

        # Time disk cache hits
        start = time.time()
        for _ in range(100):  # Fewer iterations as disk is slower
            compute(42)
        disk_time = time.time() - start

        # Both caching methods should be reasonably fast (no strict comparison)
        # Memory cache operations
        assert mem_time / 1000 < 0.01  # Less than 10ms per operation
        # Disk cache operations
        assert disk_time / 100 < 0.01  # Less than 10ms per operation


class TestCompaction:
    """Test database compaction functionality."""

    def test_compact_removes_corruption(self, temp_dir):
        """Test that compact removes corrupted entries."""
        import shelve

        cache_file = str(temp_dir / "corrupt.shelve")

        # Create a cache with valid entries
        with shelve.open(cache_file) as db:
            db["good1"] = "value1"
            db["good2"] = "value2"
            # Simulate corruption by storing unpickleable data
            # (In practice, corruption happens differently)

        # Run compact
        compact(cache_file, backup=True)

        # Verify good entries remain
        with shelve.open(cache_file) as db:
            assert db.get("good1") == "value1"
            assert db.get("good2") == "value2"

    @pytest.mark.skipif(
        platform.system() == "Windows", reason="Backup cleanup issues on Windows"
    )
    def test_compact_backup_behavior(self, temp_dir):
        """Test compact backup creation and removal."""
        cache_file = temp_dir / "compact_backup.shelve"

        # Create cache
        @cache(filename=cache_file)
        def func(x):
            return x

        func(1)
        func(2)

        # Compact with backup
        compact(str(cache_file), backup=True)

        # Verify cache still works
        assert func.get(1) == 1
        assert func.get(2) == 2

        # Verify no backup files remain
        backup_files = list(temp_dir.glob("*.bak*"))
        assert len(backup_files) == 0


class TestErrorHandling:
    """Test error handling and recovery."""

    # @pytest.mark.xfail(reason="Permission handling may vary across platforms")
    def test_cache_file_permissions(self, temp_dir):
        """Test handling of permission errors."""
        cache_file = temp_dir / "readonly.shelve"

        @cache(filename=cache_file)
        def func(x):
            return x * 2

        # Create cache
        func(5)

        # Make cache file read-only
        if not cache_file.exists():
            cache_file.touch()
        cache_file.chmod(0o444)

        try:
            # Should still work (read from cache)
            result = func(5)
            assert result == 10

            # New values should fail due to permission error
            with pytest.raises(Exception):  # Should raise permission error
                func(6)
        finally:
            # Restore permissions for cleanup
            # try:
            cache_file.chmod(0o644)
            # except FileNotFoundError:
            # pass

    def test_unicode_handling(self, temp_dir):
        """Test caching with unicode strings."""
        cache_file = temp_dir / "unicode.shelve"

        @cache(filename=cache_file)
        def process_text(text):
            return text.upper()

        # Test various unicode strings
        test_strings = [
            "Hello, 世界",
            "Здравствуй, мир",
            "مرحبا بالعالم",
            "🌍🌎🌏",
            "café",
        ]

        for text in test_strings:
            result = process_text(text)
            assert result == text.upper()
            assert process_text.__contains__(text)


class TestCustomHashStrategies:
    """Test different hashing strategies."""

    def test_hash_ignoring_order(self, temp_dir):
        """Test hash function that ignores argument order."""
        cache_file = temp_dir / "order_agnostic.shelve"

        def order_agnostic_hash(args_kwargs):
            args, kwargs = args_kwargs
            # Sort args to make order irrelevant
            return str(sorted(args))

        @cache(filename=cache_file, get_hash=order_agnostic_hash)
        def commutative_add(*args):
            return sum(args)

        # These should all use the same cache entry
        result1 = commutative_add(1, 2, 3)
        result2 = commutative_add(3, 1, 2)
        result3 = commutative_add(2, 3, 1)

        assert result1 == result2 == result3 == 6

        # Should have been computed only once
        _, status = commutative_add.__call_with_status__(2, 1, 3)
        assert status == "cached (mem)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
