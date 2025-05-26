# Client API Reference

The `Client` class (internally `SmartSurgeClient`) is the core interface for making HTTP requests with SmartSurge. It provides intelligent rate limit detection, automatic retries, and comprehensive request history tracking.

## SmartSurgeClient

```python
from smartsurge import SmartSurgeClient
# or
from smartsurge import Client  # alias
```

### Overview

The SmartSurgeClient enhances the standard `requests` library (and `aiohttp` for async operations) with:

- **Hierarchical rate limiting**: User-defined → Server-provided → Adaptive detection
- **Automatic retry logic** with exponential backoff
- **Request history tracking** per endpoint/method combination
- **Both synchronous and asynchronous** operation modes
- **Resumable streaming** for large file downloads
- **Comprehensive logging** with request correlation IDs

### Constructor

```python
SmartSurgeClient(
    base_url: Optional[str] = None,
    timeout: Union[float, Tuple[float, float]] = (10.0, 30.0),
    rate_limit: Optional[Dict[str, Union[int, float]]] = None,
    max_retries: int = 3,
    backoff_factor: float = 0.3,
    verify_ssl: bool = True,
    min_time_period: float = 1.0,
    max_time_period: float = 3600.0,
    refit_every: int = 20,
    logger: Optional[logging.Logger] = None,
    **kwargs
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | `Optional[str]` | `None` | Base URL prepended to all relative endpoints |
| `timeout` | `Union[float, Tuple[float, float]]` | `(10.0, 30.0)` | Request timeout in seconds. Single value or (connect, read) tuple |
| `rate_limit` | `Optional[Dict[str, Union[int, float]]]` | `None` | Manual rate limit: `{'requests': int, 'period': float}` |
| `max_retries` | `int` | `3` | Maximum retry attempts for failed requests (0-10) |
| `backoff_factor` | `float` | `0.3` | Exponential backoff multiplier between retries (0.0-10.0) |
| `verify_ssl` | `bool` | `True` | Whether to verify SSL certificates |
| `min_time_period` | `float` | `1.0` | Minimum seconds for rate limit detection window |
| `max_time_period` | `float` | `3600.0` | Maximum seconds for rate limit detection window |
| `refit_every` | `int` | `20` | Number of responses after which HMM should be refit |
| `logger` | `Optional[logging.Logger]` | `None` | Custom logger instance |
| `**kwargs` | `Any` | - | Additional options passed to `ClientConfig` |

### Usage Examples

```python
from smartsurge import Client

# Basic usage
client = Client()
response = client.get("https://api.example.com/users")

# With configuration
client = Client(
    base_url="https://api.example.com",
    max_retries=5,
    rate_limit={'requests': 100, 'period': 60}  # 100 requests per minute
)

# Using context manager
with Client() as client:
    response = client.get("/users")
    # Session automatically closed on exit

# With request history
response, history = client.get("/users", return_history=True)
print(f"Success rate: {history.success_rate():.2%}")
print(f"Total requests: {len(history)}")
```

## Core Methods

### request()

The primary method for making HTTP requests with automatic rate limit handling.

```python
def request(
    method: Union[str, RequestMethod],
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Union[Dict[str, Any], str, bytes]] = None,
    json: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    cookies: Optional[Dict[str, str]] = None,
    files: Optional[Dict[str, Any]] = None,
    auth: Optional[Any] = None,
    timeout: Optional[Union[float, Tuple[float, float]]] = None,
    allow_redirects: bool = True,
    verify: Optional[bool] = None,
    stream: bool = False,
    cert: Optional[Union[str, Tuple[str, str]]] = None,
    num_async: int = 1,
    request_history: Optional[RequestHistory] = None,
    return_history: bool = False
) -> Union[Tuple[requests.Response, RequestHistory], requests.Response]
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `method` | `Union[str, RequestMethod]` | HTTP method (GET, POST, etc.) |
| `endpoint` | `str` | URL endpoint (appended to base_url if relative) |
| `params` | `Optional[Dict[str, Any]]` | URL query parameters |
| `data` | `Optional[Union[Dict, str, bytes]]` | Request body data |
| `json` | `Optional[Dict[str, Any]]` | JSON request body |
| `headers` | `Optional[Dict[str, str]]` | HTTP headers |
| `cookies` | `Optional[Dict[str, str]]` | Cookies to send |
| `files` | `Optional[Dict[str, Any]]` | Files for multipart upload |
| `auth` | `Optional[Any]` | Authentication handler |
| `timeout` | `Optional[Union[float, Tuple]]` | Override default timeout |
| `allow_redirects` | `bool` | Follow redirects |
| `verify` | `Optional[bool]` | Override SSL verification |
| `stream` | `bool` | Stream the response |
| `cert` | `Optional[Union[str, Tuple]]` | Client certificate |
| `num_async` | `int` | Number of parallel requests (for testing) |
| `request_history` | `Optional[RequestHistory]` | Use specific history object |
| `return_history` | `bool` | Return tuple with history if True |

#### Returns

- If `return_history=False` (default): `requests.Response`
- If `return_history=True`: `Tuple[requests.Response, RequestHistory]`

#### Raises

- `RateLimitExceeded`: Rate limit exceeded
- `requests.RequestException`: Other request failures

#### Example

```python
# Basic request
response = client.request("GET", "/api/users")

# With parameters and headers
response = client.request(
    "POST",
    "/api/users",
    json={"name": "John", "email": "john@example.com"},
    headers={"X-API-Key": "secret"},
    timeout=60.0
)

# With request history
response, history = client.request(
    "GET", 
    "/api/data",
    return_history=True
)
print(f"Total requests: {len(history)}")
print(f"Success rate: {history.success_rate():.2%}")
```

### Convenience Methods

SmartSurge provides convenience methods for all standard HTTP verbs:

#### get()
```python
def get(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    num_async: int = 1,
    return_history: bool = False,
    **kwargs
) -> Union[Tuple[requests.Response, RequestHistory], requests.Response]
```

#### post()
```python
def post(
    endpoint: str,
    data: Optional[Union[Dict[str, Any], str, bytes]] = None,
    json: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    num_async: int = 1,
    return_history: bool = False,
    **kwargs
) -> Union[Tuple[requests.Response, RequestHistory], requests.Response]
```

#### put()
```python
def put(
    endpoint: str,
    data: Optional[Union[Dict[str, Any], str, bytes]] = None,
    json: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    num_async: int = 1,
    return_history: bool = False,
    **kwargs
) -> Union[Tuple[requests.Response, RequestHistory], requests.Response]
```

#### delete()
```python
def delete(
    endpoint: str,
    headers: Optional[Dict[str, str]] = None,
    num_async: int = 1,
    return_history: bool = False,
    **kwargs
) -> Union[Tuple[requests.Response, RequestHistory], requests.Response]
```

#### patch()
```python
def patch(
    endpoint: str,
    data: Optional[Union[Dict[str, Any], str, bytes]] = None,
    json: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    return_history: bool = False,
    **kwargs
) -> Union[Tuple[requests.Response, RequestHistory], requests.Response]
```

#### head()
```python
def head(
    endpoint: str,
    headers: Optional[Dict[str, str]] = None,
    return_history: bool = False,
    **kwargs
) -> Union[Tuple[requests.Response, RequestHistory], requests.Response]
```

#### options()
```python
def options(
    endpoint: str,
    headers: Optional[Dict[str, str]] = None,
    return_history: bool = False,
    **kwargs
) -> Union[Tuple[requests.Response, RequestHistory], requests.Response]
```

All convenience methods accept the same optional parameters as `request()` through `**kwargs`.

### Example Usage

```python
# GET request
users = client.get("/users", params={"page": 1})

# POST request with JSON
new_user = client.post(
    "/users",
    json={"name": "Alice", "email": "alice@example.com"}
)

# PUT request with form data
updated = client.put(
    "/users/123",
    data={"name": "Alice Smith"}
)

# DELETE request
client.delete("/users/123")

# With request history
response, history = client.get("/api/data", return_history=True)
```

## Rate Limit Management

### set_rate_limit()

Manually configure a rate limit for a specific endpoint and method.

```python
def set_rate_limit(
    endpoint: str,
    method: Union[str, RequestMethod],
    max_requests: int,
    time_period: float,
    source: str = "manual"
) -> None
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `endpoint` | `str` | API endpoint URL |
| `method` | `Union[str, RequestMethod]` | HTTP method |
| `max_requests` | `int` | Maximum requests allowed |
| `time_period` | `float` | Time window in seconds |
| `source` | `str` | Source identifier (default: "manual") |

#### Example

```python
# Set rate limit: 100 requests per minute
client.set_rate_limit(
    endpoint="/api/users",
    method="GET",
    max_requests=100,
    time_period=60.0
)
```

### get_rate_limit()

Retrieve the current rate limit for an endpoint/method combination.

```python
def get_rate_limit(
    endpoint: str,
    method: Union[str, RequestMethod]
) -> Optional[RateLimit]
```

#### Returns

`Optional[RateLimit]`: Current rate limit or None if not set

#### Example

```python
limit = client.get_rate_limit("/api/users", "GET")
if limit:
    print(f"Rate limit: {limit.max_requests} per {limit.time_period}s")
    print(f"Source: {limit.source}")
```

### list_rate_limits()

Get all configured rate limits.

```python
def list_rate_limits() -> Dict[Tuple[str, str], Optional[RateLimit]]
```

#### Returns

`Dict[Tuple[str, str], Optional[RateLimit]]`: Dictionary mapping (endpoint, method) to rate limits

#### Example

```python
limits = client.list_rate_limits()
for (endpoint, method), limit in limits.items():
    if limit:
        print(f"{method} {endpoint}: {limit.max_requests} per {limit.time_period}s")
```

### reset_rate_limit()

Remove a specific rate limit.

```python
def reset_rate_limit(
    endpoint: str,
    method: Union[str, RequestMethod]
) -> None
```

#### Example

```python
# Remove rate limit
client.reset_rate_limit("/api/users", "GET")
```

### reset_all_rate_limits()

Clear all rate limits and request histories.

```python
def reset_all_rate_limits() -> None
```

#### Example

```python
# Start fresh
client.reset_all_rate_limits()
```

## Request History

### get_request_history()

Get the request history for a specific endpoint/method.

```python
def get_request_history(
    endpoint: str,
    method: Union[str, RequestMethod]
) -> Optional[RequestHistory]
```

#### Returns

`Optional[RequestHistory]`: Request history or None if no requests made

#### Example

```python
history = client.get_request_history("/api/users", "GET")
if history:
    print(f"Total requests: {len(history)}")
    print(f"Success rate: {history.success_rate():.2%}")
    print(f"Average response time: {history.avg_response_time():.3f}s")
```

## Streaming

### stream()

Download large files with automatic resume capability.

```python
def stream(
    endpoint: str,
    output_path: str,
    method: Union[str, RequestMethod] = RequestMethod.GET,
    chunk_size: int = 8192,
    resume: bool = True,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    timeout: Optional[Union[float, Tuple[float, float]]] = None,
    **kwargs
) -> str
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endpoint` | `str` | - | URL to stream from |
| `output_path` | `str` | - | Local file path to save to |
| `method` | `Union[str, RequestMethod]` | `GET` | HTTP method |
| `chunk_size` | `int` | `8192` | Download chunk size in bytes |
| `resume` | `bool` | `True` | Resume partial downloads |
| `headers` | `Optional[Dict[str, str]]` | `None` | Additional headers |
| `params` | `Optional[Dict[str, Any]]` | `None` | Query parameters |
| `max_retries` | `int` | `3` | Maximum retry attempts |
| `timeout` | `Optional[Union[float, Tuple]]` | `None` | Override timeout |
| `**kwargs` | `Any` | - | Additional arguments |

#### Returns

`str`: Path to the downloaded file

#### Example

```python
# Download with resume support
file_path = client.stream(
    "https://example.com/large-file.zip",
    "downloads/large-file.zip",
    chunk_size=1024*1024  # 1MB chunks
)

# Download without resume
file_path = client.stream(
    "https://example.com/data.csv",
    "data.csv",
    resume=False
)
```

## Session Management

### close()

Close the underlying HTTP session.

```python
def close() -> None
```

#### Example

```python
client = Client()
# ... make requests ...
client.close()
```

### Context Manager

The client supports context manager protocol for automatic cleanup:

```python
with Client() as client:
    response = client.get("/api/users")
    # Session automatically closed on exit
```

## Configuration

### ClientConfig

The `ClientConfig` class centralizes all configuration options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `base_url` | `Optional[str]` | `None` | Base URL for requests |
| `timeout` | `Tuple[float, float]` | `(10.0, 30.0)` | (connect, read) timeouts |
| `max_retries` | `int` | `3` | Maximum retry attempts |
| `backoff_factor` | `float` | `0.3` | Exponential backoff factor |
| `verify_ssl` | `bool` | `True` | Verify SSL certificates |
| `min_time_period` | `float` | `1.0` | Min rate limit window |
| `max_time_period` | `float` | `3600.0` | Max rate limit window |
| `user_agent` | `str` | `"SmartSurge/0.0.5"` | User agent string |
| `max_connections` | `int` | `10` | Max connections |
| `keep_alive` | `bool` | `True` | Keep connections alive |
| `max_pool_size` | `int` | `10` | Connection pool size |
| `log_level` | `int` | `logging.INFO` | Logging level |

### Example with Custom Config

```python
from smartsurge import ClientConfig, Client

config = ClientConfig(
    base_url="https://api.example.com",
    timeout=(5.0, 60.0),
    max_retries=5,
    user_agent="MyApp/1.0"
)

client = Client(**config.model_dump())
```

## Error Handling

SmartSurge defines several custom exceptions:

```python
from smartsurge import (
    RateLimitExceeded,
    StreamingError,
    ResumeError,
    ValidationError,
    ConfigurationError
)

try:
    response = client.get("/api/resource")
except RateLimitExceeded as e:
    print(f"Rate limit hit: {e.message}")
    if e.retry_after:
        print(f"Retry after {e.retry_after} seconds")
except StreamingError as e:
    print(f"Streaming failed: {e.message}")
except ValidationError as e:
    print(f"Invalid parameters: {e}")
```

## Logging

SmartSurge uses hierarchical logging under the `smartsurge` namespace:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or configure specific loggers
logging.getLogger("smartsurge.client").setLevel(logging.DEBUG)
logging.getLogger("smartsurge.hmm").setLevel(logging.INFO)
```

Each request is assigned a unique request ID for correlation:

```
INFO:smartsurge.client:[req-abc123] Making GET request to /api/users
DEBUG:smartsurge.client:[req-abc123] Rate limit check passed
INFO:smartsurge.client:[req-abc123] Request successful: 200 OK
```

## Thread Safety

The SmartSurgeClient is thread-safe. Multiple threads can share the same client instance:

```python
import threading

client = Client()

def worker(user_id):
    response = client.get(f"/users/{user_id}")
    print(f"User {user_id}: {response.status_code}")

threads = []
for i in range(10):
    t = threading.Thread(target=worker, args=(i,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
```

## Performance Tips

1. **Reuse client instances**: Create one client and reuse it
2. **Use connection pooling**: Enabled by default
3. **Set appropriate timeouts**: Balance reliability and performance
4. **Configure refit_every**: Higher values reduce computation overhead
5. **Use streaming for large files**: Reduces memory usage
6. **Enable keep-alive**: Reduces connection overhead

## Complete Example

```python
from smartsurge import Client, RateLimitExceeded
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create client with configuration
client = Client(
    base_url="https://api.github.com",
    max_retries=5,
    refit_every=20
)

try:
    # Set known rate limit
    client.set_rate_limit(
        endpoint="/users",
        method="GET",
        max_requests=60,
        time_period=3600  # GitHub's unauthenticated rate limit
    )
    
    # Make requests
    for username in ["torvalds", "gvanrossum", "dhh"]:
        response, history = client.get(
            f"/users/{username}",
            return_history=True
        )
        
        user = response.json()
        print(f"{user['name']} - {user['public_repos']} repos")
        
        # Check rate limit status
        if history.rate_limit:
            remaining = history.rate_limit.max_requests - len(history)
            print(f"  Rate limit remaining: {remaining}")
    
    # Download a file
    client.stream(
        "https://api.github.com/repos/python/cpython/zipball/master",
        "cpython-master.zip"
    )
    
except RateLimitExceeded as e:
    print(f"Rate limit exceeded: {e.message}")
    if e.retry_after:
        print(f"Retry after {e.retry_after} seconds")
        
finally:
    client.close()
```