import asyncio
import threading
import queue
import sys
from typing import Any, Callable, Coroutine, Optional, TypeVar, Union
from functools import wraps, partial

T = TypeVar('T')

class AsyncExecutor:
    """A singleton class to manage async execution in a separate thread."""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized') or not self._initialized:
            self._loop = asyncio.new_event_loop()
            self._queue = queue.Queue()
            self._stop_event = threading.Event()
            self._initialized = True
            
            # Start the event loop thread
            self._thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
                name="AsyncExecutorThread"
            )
            self._thread.start()
    
    def _run_loop(self):
        """Run the event loop in a separate thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
    
    def submit(self, coro: Coroutine[Any, Any, T]) -> 'asyncio.Future[T]':
        """Submit a coroutine to be executed in the async executor."""
        if not self._loop.is_running():
            raise RuntimeError("Event loop is not running")
            
        # Create a future in the event loop thread
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future
    
    def run_async(self, coro: Coroutine[Any, Any, T], timeout: Optional[float] = 60) -> T:
        """Run a coroutine and return its result synchronously."""
        future = self.submit(coro)
        try:
            return future.result(timeout=timeout)
        except Exception as e:
            # If there was an error, propagate it
            raise e from None
    
    def shutdown(self):
        """Shutdown the executor cleanly."""
        if hasattr(self, '_stop_event'):
            self._stop_event.set()
            
        if hasattr(self, '_loop') and self._loop.is_running():
            # Schedule the loop to stop
            self._loop.call_soon_threadsafe(self._loop.stop)
            
        if hasattr(self, '_thread') and self._thread.is_alive():
            self._thread.join(timeout=5.0)

# Global executor instance
_executor = None

def get_executor() -> AsyncExecutor:
    """Get or create the global async executor."""
    global _executor
    if _executor is None:
        _executor = AsyncExecutor()
    return _executor

def run_async(coro: Coroutine[Any, Any, T], timeout: Optional[float] = 60) -> T:
    """Run a coroutine in the global async executor.
    
    Args:
        coro: The coroutine to run
        timeout: Maximum time to wait for the coroutine to complete
        
    Returns:
        The result of the coroutine
    """
    return get_executor().run_async(coro, timeout=timeout)

def async_to_sync(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
    """Decorator to convert an async function to a synchronous one.
    
    Example:
        @async_to_sync
        async def my_async_func():
            return await some_async_operation()
            
        # Can be called synchronously
        result = my_async_func()
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return run_async(func(*args, **kwargs))
    return wrapper

# Register cleanup on interpreter shutdown
import atexit
atexit.register(lambda: get_executor().shutdown() if _executor else None)
