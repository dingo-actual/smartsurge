# Exceptions API Reference

SmartSurge provides a comprehensive hierarchy of exceptions for precise error handling. All exceptions inherit from `SmartSurgeException` and include detailed context information with automatic logging.

## Exception Hierarchy

```
SmartSurgeException (base)
├── RateLimitExceeded
├── StreamingError
├── ResumeError
├── ValidationError
├── ConfigurationError
├── ConnectionError
├── TimeoutError
├── ServerError
└── ClientError
```

## Base Exception

### SmartSurgeException

```python
from smartsurge.exceptions import SmartSurgeException
```

Base exception class for all SmartSurge exceptions. Provides automatic logging and context tracking.

#### Constructor

```python
SmartSurgeException(
    message: str,
    **kwargs
)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `message` | `str` | The exception message |
| `**kwargs` | `Any` | Additional context information stored in `context` attribute |

#### Attributes

- `message`: The exception message
- `context`: Dictionary containing additional context information

#### Features

- Automatic error logging with context
- Safe string conversion for all attributes
- Graceful handling of logging failures

## Request Exceptions

### RateLimitExceeded

```python
from smartsurge.exceptions import RateLimitExceeded
```

Raised when a rate limit is exceeded. Includes retry information when available.

#### Constructor

```python
RateLimitExceeded(
    message: str,
    endpoint: Optional[str] = None,
    method: Optional[Union[str, RequestMethod]] = None,
    retry_after: Optional[int] = None,
    response_headers: Optional[Dict[str, str]] = None
)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `message` | `str` | The exception message |
| `endpoint` | `Optional[str]` | The endpoint that was rate limited |
| `method` | `Optional[Union[str, RequestMethod]]` | The HTTP method used |
| `retry_after` | `Optional[int]` | Retry delay in seconds from server |
| `response_headers` | `Optional[Dict[str, str]]` | Response headers with rate limit info |

#### Attributes

All constructor parameters are available as instance attributes.

#### Example

```python
from smartsurge import Client
from smartsurge.exceptions import RateLimitExceeded

client = Client()

try:
    response = client.get("/api/data")
except RateLimitExceeded as e:
    print(f"Rate limited on {e.endpoint}")
    if e.retry_after:
        print(f"Retry after {e.retry_after} seconds")
        time.sleep(e.retry_after)
```

## Streaming Exceptions

### StreamingError

```python
from smartsurge.exceptions import StreamingError
```

Raised when a streaming request fails.

#### Constructor

```python
StreamingError(
    message: str,
    endpoint: Optional[str] = None,
    position: Optional[int] = None,
    response: Optional[Any] = None
)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `message` | `str` | The exception message |
| `endpoint` | `Optional[str]` | The endpoint being streamed |
| `position` | `Optional[int]` | Position in stream where error occurred |
| `response` | `Optional[Any]` | The response object if available |

#### Attributes

All constructor parameters plus:
- `response_status`: Extracted status code from response (if available)

### ResumeError

```python
from smartsurge.exceptions import ResumeError
```

Raised when resuming a streaming request fails.

#### Constructor

```python
ResumeError(
    message: str,
    state_file: Optional[str] = None,
    original_error: Optional[Exception] = None
)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `message` | `str` | The exception message |
| `state_file` | `Optional[str]` | Path to the state file that failed |
| `original_error` | `Optional[Exception]` | The underlying exception |

#### Attributes

All constructor parameters plus:
- `traceback`: Formatted traceback of original error (if available)

#### Example

```python
from smartsurge.streaming import JSONStreamingRequest
from smartsurge.exceptions import ResumeError

try:
    result = client.stream_request(
        JSONStreamingRequest,
        "/api/large-file",
        state_file="download.state"
    )
except ResumeError as e:
    print(f"Failed to resume: {e.message}")
    print(f"State file: {e.state_file}")
    # Start fresh download
    os.remove(e.state_file)
```

## Validation Exceptions

### ValidationError

```python
from smartsurge.exceptions import ValidationError
```

Raised when data validation fails.

#### Constructor

```python
ValidationError(
    message: str,
    field: Optional[str] = None,
    value: Optional[Any] = None
)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `message` | `str` | The exception message |
| `field` | `Optional[str]` | The field that failed validation |
| `value` | `Optional[Any]` | The invalid value |

#### Attributes

All constructor parameters are available as instance attributes with safe string conversion.

### ConfigurationError

```python
from smartsurge.exceptions import ConfigurationError
```

Raised when configuration is invalid.

#### Constructor

```python
ConfigurationError(
    message: str,
    parameter: Optional[str] = None,
    value: Optional[Any] = None
)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `message` | `str` | The exception message |
| `parameter` | `Optional[str]` | The invalid parameter name |
| `value` | `Optional[Any]` | The invalid parameter value |

#### Example

```python
from smartsurge.exceptions import ConfigurationError

try:
    client = Client(timeout=-1)
except ConfigurationError as e:
    print(f"Invalid config: {e.parameter} = {e.value}")
```

## Connection Exceptions

### ConnectionError

```python
from smartsurge.exceptions import ConnectionError
```

Raised when a connection error occurs.

#### Constructor

```python
ConnectionError(
    message: str,
    endpoint: Optional[str] = None,
    original_error: Optional[Exception] = None
)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `message` | `str` | The exception message |
| `endpoint` | `Optional[str]` | The endpoint that failed to connect |
| `original_error` | `Optional[Exception]` | The underlying connection error |

#### Attributes

All constructor parameters plus:
- `traceback`: Formatted traceback of original error (if available)

### TimeoutError

```python
from smartsurge.exceptions import TimeoutError
```

Raised when a request times out.

#### Constructor

```python
TimeoutError(
    message: str,
    endpoint: Optional[str] = None,
    timeout: Optional[Union[float, int]] = None
)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `message` | `str` | The exception message |
| `endpoint` | `Optional[str]` | The endpoint that timed out |
| `timeout` | `Optional[Union[float, int]]` | The timeout value in seconds |

## HTTP Status Exceptions

### ServerError

```python
from smartsurge.exceptions import ServerError
```

Raised when a server error occurs (5xx status codes).

#### Constructor

```python
ServerError(
    message: str,
    endpoint: Optional[str] = None,
    status_code: Optional[int] = None,
    response: Optional[Any] = None
)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `message` | `str` | The exception message |
| `endpoint` | `Optional[str]` | The endpoint that returned an error |
| `status_code` | `Optional[int]` | The HTTP status code (5xx) |
| `response` | `Optional[Any]` | The response object if available |

#### Attributes

All constructor parameters plus:
- `response_text`: First 200 characters of response text (if available)

### ClientError

```python
from smartsurge.exceptions import ClientError
```

Raised when a client error occurs (4xx status codes).

#### Constructor

```python
ClientError(
    message: str,
    endpoint: Optional[str] = None,
    status_code: Optional[int] = None,
    response: Optional[Any] = None
)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `message` | `str` | The exception message |
| `endpoint` | `Optional[str]` | The endpoint that returned an error |
| `status_code` | `Optional[int]` | The HTTP status code (4xx) |
| `response` | `Optional[Any]` | The response object if available |

#### Attributes

All constructor parameters plus:
- `response_text`: First 200 characters of response text (if available)

## Error Handling Patterns

### Comprehensive Error Handling

```python
from smartsurge import Client
from smartsurge.exceptions import (
    RateLimitExceeded,
    StreamingError,
    ConnectionError,
    TimeoutError,
    ServerError,
    ClientError
)

client = Client()

try:
    response = client.get("/api/data")
except RateLimitExceeded as e:
    # Handle rate limiting
    if e.retry_after:
        time.sleep(e.retry_after)
        # Retry request
except TimeoutError as e:
    # Handle timeout
    print(f"Request timed out after {e.timeout}s")
except ConnectionError as e:
    # Handle connection issues
    print(f"Connection failed to {e.endpoint}")
except ServerError as e:
    # Handle server errors (5xx)
    print(f"Server error {e.status_code}: {e.response_text}")
except ClientError as e:
    # Handle client errors (4xx)
    print(f"Client error {e.status_code}: {e.response_text}")
except Exception as e:
    # Handle unexpected errors
    print(f"Unexpected error: {e}")
```

### Streaming Error Recovery

```python
from smartsurge.streaming import JSONStreamingRequest
from smartsurge.exceptions import StreamingError, ResumeError

def download_with_retry(client, url, max_retries=3):
    state_file = "download.state"
    
    for attempt in range(max_retries):
        try:
            return client.stream_request(
                JSONStreamingRequest,
                url,
                state_file=state_file
            )
        except ResumeError as e:
            print(f"Resume failed: {e.message}")
            if os.path.exists(state_file):
                os.remove(state_file)
            # Will retry fresh
        except StreamingError as e:
            if e.position:
                print(f"Stream failed at position {e.position}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

### Custom Exception Handling

```python
from smartsurge.exceptions import SmartSurgeException

class MyCustomError(SmartSurgeException):
    """Custom error for my application."""
    pass

try:
    # Your code here
    if not valid_data:
        raise MyCustomError(
            "Invalid data format",
            data=data,
            validation_rules=rules
        )
except MyCustomError as e:
    # Access context
    print(f"Error: {e.message}")
    print(f"Context: {e.context}")
```

## Exception Safety Features

All SmartSurge exceptions include:

1. **Safe string conversion**: Handles unprintable objects gracefully
2. **Automatic logging**: Errors are logged with full context
3. **Traceback capture**: Original errors include formatted tracebacks
4. **Context preservation**: All kwargs are stored in the `context` attribute
5. **Type safety**: Proper handling of RequestMethod enums and other types

These features ensure robust error handling even in edge cases where exception data might be malformed or unprintable.