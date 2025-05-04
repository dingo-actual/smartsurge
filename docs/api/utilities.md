# Utilities API Reference

This page documents the utility functions and classes in SmartSurge.

## `SmartSurgeTimer`

```python
class SmartSurgeTimer:
"""
Context manager for timing code execution and logging the results.
"""
```

### Constructor

```python
def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None)
```

Initializes a SmartSurgeTimer.

**Parameters:**

- `operation_name` - Name of the operation being timed
- `logger` - Logger to use for output

### Methods

```python
def __enter__(self)
```

Starts the timer.

```python
def __exit__(self, exc_type, exc_val, exc_tb)
```

Stops the timer and logs the elapsed time.

## `log_context`

```python
@contextmanager
def log_context(context_name: str, logger: Optional[logging.Logger] = None,
                level: int = logging.DEBUG, include_id: bool = True) -> None
```

A context manager for logging the start and end of an operation.

**Parameters:**

- `context_name` - Name of the context for logging
- `logger` - Logger to use
- `level` - Logging level
- `include_id` - Whether to include a correlation ID

## `async_request_with_history`

```python
async def async_request_with_history(
    request_func: Callable[..., Coroutine[Any, Any, Tuple[T, RequestHistory]]],
    endpoints: List[str],
    method: RequestMethod,
    max_concurrent: int = 5,
    min_time_period: float = 1.0,
    max_time_period: float = 3600.0,
    confidence_threshold: float = 0.9,
    **kwargs
) -> Tuple[List[T], Dict[str, RequestHistory]]
```

Makes multiple async requests and consolidates histories for the same endpoint.

**Parameters:**

- `request_func` - Async function to make the request
- `endpoints` - List of endpoints to request
- `method` - HTTP method to use
- `max_concurrent` - Maximum number of concurrent requests
- `min_time_period` - Minimum time period for rate limiting
- `max_time_period` - Maximum time period for rate limiting
- `confidence_threshold` - Confidence threshold for rate limit estimation
- `**kwargs` - Additional arguments to pass to request_func

**Returns:**

- A tuple containing:
  - List of responses
  - Dictionary of consolidated `RequestHistory` objects by endpoint

## `merge_histories`

```python
def merge_histories(histories: List[RequestHistory]) -> Dict[Tuple[str, RequestMethod], RequestHistory]
```

Merges multiple `RequestHistory` objects by endpoint/method.

**Parameters:**

- `histories` - List of `RequestHistory` objects

**Returns:**

- Dictionary of merged `RequestHistory` objects keyed by (endpoint, method)
