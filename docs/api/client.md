# SmartSurgeClient API Reference

The `SmartSurgeClient` class is the main entry point for the SmartSurge library, providing methods for making HTTP requests with adaptive rate limit detection and enforcement.

## `SmartSurgeClient`

```python
class SmartSurgeClient:
"""
A wrapper around requests library with adaptive rate limiting and resumable streaming.
"""
```

### Constructor

```python
def __init__(self,
            base_url: Optional[str] = None,
            timeout: Union[float, Tuple[float, float]] = (10.0, 30.0),
            max_retries: int = 3,
            backoff_factor: float = 0.3,
            verify_ssl: bool = True,
            min_time_period: float = 1.0,
            max_time_period: float = 3600.0,
            confidence_threshold: float = 0.9,
            logger: Optional[logging.Logger] = None,
            **kwargs)
```

Creates a new SmartSurge client.

**Parameters:**

- `base_url` - Base URL for all requests
- `timeout` - Request timeout in seconds, either a single value or a tuple of (connect_timeout, read_timeout)
- `max_retries` - Maximum number of retries for failed requests
- `backoff_factor` - Backoff factor for retries
- `verify_ssl` - Whether to verify SSL certificates
- `min_time_period` - Minimum time period to consider for rate limiting (seconds)
- `max_time_period` - Maximum time period to consider for rate limiting (seconds)
- `confidence_threshold` - Confidence threshold for rate limit estimation (0.0-1.0)
- `logger` - Optional custom logger to use
- `**kwargs` - Additional configuration options for SmartSurgeClientConfig

### Methods

#### `request`

```python
def request(self,
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
            confidence_threshold: Optional[float] = None,
            request_history: Optional[RequestHistory] = None) -> Tuple[requests.Response, RequestHistory]
```

Makes an HTTP request with adaptive rate limiting.

**Parameters:**

- `method` - HTTP method to use
- `endpoint` - Endpoint to request
- `params` - Query parameters
- `data` - Form data
- `json` - JSON data
- `headers` - HTTP headers
- `cookies` - Cookies to send
- `files` - Files to upload
- `auth` - Authentication
- `timeout` - Request timeout
- `allow_redirects` - Whether to follow redirects
- `verify` - Whether to verify SSL certificates
- `stream` - Whether to stream the response
- `cert` - SSL client certificate
- `num_async` - Number of asynchronous requests
- `confidence_threshold` - Confidence threshold for rate limit estimation
- `request_history` - Explicit RequestHistory to use

**Returns:**

- A tuple containing:
  - The HTTP response
  - The RequestHistory used for this request

**Raises:**

- `RateLimitExceeded` - If the rate limit has been exceeded
- `requests.RequestException` - For other request failures

#### `stream_request`

```python
def stream_request(self,
                    streaming_class: Type[AbstractStreamingRequest],
                    endpoint: str,
                    params: Optional[Dict[str, Any]] = None,
                    headers: Optional[Dict[str, str]] = None,
                    state_file: Optional[str] = None,
                    chunk_size: int = 8192,
                    num_async: int = 1,
                    confidence_threshold: Optional[float] = None) -> Tuple[Any, RequestHistory]
```

Makes a streaming request with resumable functionality.

**Parameters:**

- `streaming_class` - Class to handle streaming
- `endpoint` - Endpoint to request
- `params` - Query parameters
- `headers` - HTTP headers
- `state_file` - File to save state for resumption
- `chunk_size` - Size of chunks to process
- `num_async` - Number of asynchronous requests
- `confidence_threshold` - Confidence threshold for rate limit estimation

**Returns:**

- A tuple containing:
  - The result of the streaming request
  - The RequestHistory used for this request

**Raises:**

- `StreamingError` - If the streaming request fails

#### HTTP Method Convenience Methods

```python
def get(self, endpoint, params=None, headers=None, num_async=1, confidence_threshold=None, **kwargs)
def post(self, endpoint, data=None, json=None, headers=None, num_async=1, confidence_threshold=None, **kwargs)
def put(self, endpoint, data=None, json=None, headers=None, num_async=1, confidence_threshold=None, **kwargs)
def delete(self, endpoint, params=None, headers=None, num_async=1, confidence_threshold=None, **kwargs)
def patch(self, endpoint, data=None, json=None, headers=None, num_async=1, confidence_threshold=None, **kwargs)
```

Convenience methods for making requests with specific HTTP methods.

#### Async HTTP Methods

```python
async def async_get(self, endpoint, params=None, headers=None, num_async=1, confidence_threshold=None, **kwargs)
async def async_post(self, endpoint, data=None, json=None, headers=None, num_async=1, confidence_threshold=None, **kwargs)
async def async_request(self, method, endpoint, params=None, data=None, json=None, headers=None, **kwargs)
```

Asynchronous versions of the request methods.

#### Utility Methods

```python
def close(self)
```

Closes the client and releases resources.

```python
def __enter__(self) -> 'SmartSurgeClient'
def __exit__(self, exc_type, exc_val, exc_tb) -> None
```

Context manager support for the client.

## `SmartSurgeClientConfig`

```python
class SmartSurgeClientConfig(BaseModel):
    """
    Configuration for the SmartSurge client.
    """
```

A Pydantic model that centralizes all configuration options for the client.

### Attributes

- `base_url` - Base URL for all requests
- `timeout` - Default timeout for requests in seconds
- `max_retries` - Maximum number of retries for failed requests
- `backoff_factor` - Backoff factor for retries
- `verify_ssl` - Whether to verify SSL certificates
- `min_time_period` - Minimum time period to consider for rate limiting (seconds)
- `max_time_period` - Maximum time period to consider for rate limiting (seconds)
- `confidence_threshold` - Confidence threshold for rate limit estimation
- `user_agent` - User agent string for requests
- `max_connections` - Maximum number of connections to keep alive
- `keep_alive` - Whether to keep connections alive
- `max_pool_size` - Maximum size of the connection pool
- `log_level` - Log level for the client
