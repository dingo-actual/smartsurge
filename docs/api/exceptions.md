# Exceptions API Reference

SmartSurge provides a hierarchy of exceptions for different error cases.

## `EnhancedRequestsException`

```python
class EnhancedRequestsException(Exception):
    """Base exception for the SmartSurge library."""
```

The base exception class for all SmartSurge exceptions.

## `RateLimitExceeded`

```python
class RateLimitExceeded(EnhancedRequestsException):
    """
    Exception raised when a rate limit is exceeded.
    """
```

### Attributes

- `endpoint` - The endpoint that was rate limited
- `method` - The HTTP method that was rate limited
- `retry_after` - Optional retry-after time in seconds
- `message` - The exception message

### Constructor

```python
def __init__(self, message: str, endpoint: Optional[str] = None,
            method: Optional[Union[str, RequestMethod]] = None,
            retry_after: Optional[int] = None)
```

## `StreamingError`

```python
class StreamingError(EnhancedRequestsException):
    """
    Exception raised when a streaming request fails.
    """
```

### Attributes

- `endpoint` - The endpoint that was requested
- `position` - The position in the stream where the error occurred
- `message` - The exception message

### Constructor

```python
def __init__(self, message: str, endpoint: Optional[str] = None, position: Optional[int] = None)
```

## `ResumeError`

```python
class ResumeError(EnhancedRequestsException):
    """
    Exception raised when resuming a request fails.
    """
```

### Attributes

- `state_file` - The state file that failed to load
- `message` - The exception message

### Constructor

```python
def __init__(self, message: str, state_file: Optional[str] = None)
```

## ValidationError

```python
class ValidationError(EnhancedRequestsException):
    """
    Exception raised when validation fails.
    """
```

### Attributes

- `field` - The field that failed validation
- `message` - The exception message

### Constructor

```python
def __init__(self, message: str, field: Optional[str] = None)
```

## `ConfigurationError`

```python
class ConfigurationError(EnhancedRequestsException):
    """
    Exception raised when configuration is invalid.
    """
```

### Attributes

- `parameter` - The parameter that is invalid
- `message` - The exception message

### Constructor

```python
def __init__(self, message: str, parameter: Optional[str] = None)
```
