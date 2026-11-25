"""Small helper utilities (includes a decorator)."""
import functools
import time


def timer(func):
    """Decorator that prints how long a function took to run."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[timer] {func.__qualname__} finished in {elapsed:.2f}s")
        return result

    return wrapper
