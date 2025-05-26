# Streaming API Reference

The Streaming module provides classes for handling large HTTP responses with support for resumable downloads and efficient chunk processing. It includes security features like automatic authentication header purging when saving state.

## Overview

SmartSurge's streaming functionality:
- **Resumable downloads**: Save and restore download state
- **Chunk processing**: Handle large responses efficiently
- **Security**: Automatic purging of authentication headers from saved state
- **Progress tracking**: Monitor download progress with configurable logging
- **Error recovery**: Automatic state saving on errors

## StreamingState

```python
from smartsurge.streaming import StreamingState
```

State of a streaming request for resumption.

### Constructor

```python
StreamingState(
    endpoint: str,
    method: str,
    headers: Dict[str, str],
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    accumulated_data: bytes,
    last_position: int,
    total_size: Optional[int] = None,
    etag: Optional[str] = None,
    last_updated: datetime = <now>,
    request_id: str = <auto>
)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `endpoint` | `str` | The endpoint being requested (min length: 1) |
| `method` | `str` | HTTP method being used |
| `headers` | `Dict[str, str]` | HTTP headers (auth headers are purged) |
| `params` | `Optional[Dict[str, Any]]` | Query parameters |
| `data` | `Optional[Dict[str, Any]]` | Request body data |
| `accumulated_data` | `bytes` | Data accumulated so far |
| `last_position` | `int` | Last position in stream (≥ 0) |
| `total_size` | `Optional[int]` | Total size if known |
| `etag` | `Optional[str]` | ETag for resource validation |
| `last_updated` | `datetime` | When state was last updated (UTC) |
| `request_id` | `str` | Unique request identifier |

### Data Handling

- `accumulated_data` is automatically base64 encoded/decoded for JSON serialization
- Validates base64 format and handles encoding errors gracefully
- Supports both string and bytes input

## AbstractStreamingRequest

```python
from smartsurge.streaming import AbstractStreamingRequest
```

Abstract base class for resumable streaming requests.

### Constructor

```python
AbstractStreamingRequest(
    endpoint: str,
    headers: Dict[str, str],
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    chunk_size: int = 8192,
    state_file: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    request_id: Optional[str] = None
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endpoint` | `str` | - | The endpoint to request |
| `headers` | `Dict[str, str]` | - | HTTP headers for the request |
| `params` | `Optional[Dict[str, Any]]` | `None` | Query parameters |
| `data` | `Optional[Dict[str, Any]]` | `None` | Request body data |
| `chunk_size` | `int` | `8192` | Size of chunks to process |
| `state_file` | `Optional[str]` | `None` | File path to save state |
| `logger` | `Optional[logging.Logger]` | `None` | Custom logger instance |
| `request_id` | `Optional[str]` | `None` | Request ID for tracking |

### Security Features

#### Authentication Header Purging

The following headers are automatically removed when saving state:

- `authorization`, `x-api-key`, `x-auth-token`, `api-key`
- `x-access-token`, `x-token`, `x-session-token`, `cookie`
- `x-csrf-token`, `x-client-secret`, `proxy-authorization`
- `x-amz-security-token`, `x-goog-api-key`, `apikey`
- `auth-token`, `authentication`, `x-authentication`
- `x-authorization`, `access-token`, `secret-key`
- `private-key`, `x-secret-key`, `x-private-key`
- `bearer`, `oauth-token`, `x-oauth-token`

### Abstract Methods

Subclasses must implement:

#### start()
```python
@abstractmethod
def start() -> None
```
Start the streaming request.

#### resume()
```python
@abstractmethod
def resume() -> None
```
Resume from saved state.

#### process_chunk()
```python
@abstractmethod
def process_chunk(chunk: bytes) -> None
```
Process a data chunk.

#### get_result()
```python
@abstractmethod
def get_result() -> Any
```
Get the final result.

### Concrete Methods

#### save_state()
```python
def save_state() -> None
```
Save current state to file with authentication headers purged.

#### load_state()
```python
def load_state() -> Optional[StreamingState]
```
Load state from file and restore instance variables.

#### _purge_auth_headers()
```python
def _purge_auth_headers(headers: Dict[str, str]) -> Dict[str, str]
```
Remove authentication headers for security.

## JSONStreamingRequest

```python
from smartsurge.streaming import JSONStreamingRequest
```

Streaming request implementation for JSON data.

### Constructor

```python
JSONStreamingRequest(
    endpoint: str,
    headers: Dict[str, str],
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    chunk_size: int = 8192,
    state_file: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    request_id: Optional[str] = None
)
```

Parameters are identical to `AbstractStreamingRequest`.

### Features

- **HTTP session management** with retry strategy:
  - 3 retries with exponential backoff
  - Retries on 500, 502, 503, 504 status codes
  - Connection and read timeouts (10s, 30s)

- **Resume support**:
  - Uses `Range` header for partial content
  - Validates resource with `ETag`/`If-Match` headers
  - Automatic position tracking

- **Progress tracking**:
  - Logs progress every ~10 chunks for large downloads
  - Periodic state saving during download

### Methods

#### start()
```python
def start() -> None
```

Start streaming request with:
- Automatic retry handling
- Range header support for resumption
- Content-Length and ETag extraction
- Chunk-by-chunk processing

Raises `StreamingError` on failure.

#### resume()
```python
def resume() -> None
```

Resume from saved state:
- Loads previous state
- Sets Range header from last position
- Validates with ETag if available

Raises `ResumeError` on failure.

#### process_chunk()
```python
def process_chunk(chunk: bytes) -> None
```

Process data chunks:
- Accumulates data
- Updates position
- Logs progress for large files
- Periodic state saving

#### get_result()
```python
def get_result() -> Any
```

Parse accumulated JSON data.

Raises `StreamingError` if:
- Streaming not completed
- JSON parsing fails

## Complete Examples

### Basic Streaming

```python
from smartsurge import Client
from smartsurge.streaming import JSONStreamingRequest

client = Client()

# Stream JSON data
result = client.stream_request(
    JSONStreamingRequest,
    "https://api.example.com/large-dataset.json"
)

print(f"Received {len(result)} items")
```

### Resumable Download

```python
from smartsurge import Client
from smartsurge.streaming import JSONStreamingRequest
from smartsurge.exceptions import StreamingError, ResumeError
import os

client = Client()
state_file = "download_state.json"

try:
    # Automatically resumes if state file exists
    result = client.stream_request(
        JSONStreamingRequest,
        "https://api.example.com/huge-dataset.json",
        state_file=state_file,
        chunk_size=1024 * 1024  # 1MB chunks
    )
    
    # Success - clean up state file
    if os.path.exists(state_file):
        os.remove(state_file)
        
    print(f"Download complete: {len(result)} records")
    
except KeyboardInterrupt:
    print("Download paused. Run again to resume.")
except ResumeError as e:
    print(f"Failed to resume: {e}")
    # Optionally delete corrupt state and retry fresh
    if os.path.exists(state_file):
        os.remove(state_file)
except StreamingError as e:
    print(f"Streaming failed: {e}")
```

### Custom Streaming Implementation

```python
from smartsurge.streaming import AbstractStreamingRequest
import csv
import io

class CSVStreamingRequest(AbstractStreamingRequest):
    """Stream and parse CSV data."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = []
        
    def start(self):
        """Start streaming CSV data."""
        # Similar to JSONStreamingRequest.start()
        # but process CSV instead
        pass
        
    def process_chunk(self, chunk: bytes):
        """Process CSV chunks."""
        # Accumulate data
        self.accumulated_data.extend(chunk)
        self.position += len(chunk)
        
        # Parse complete lines
        data_str = self.accumulated_data.decode('utf-8', errors='ignore')
        lines = data_str.split('\n')
        
        # Keep incomplete last line
        if not data_str.endswith('\n'):
            self.accumulated_data = bytearray(lines[-1].encode('utf-8'))
            lines = lines[:-1]
        else:
            self.accumulated_data = bytearray()
            
        # Parse complete lines
        for line in lines:
            if line.strip():
                self.rows.append(line.split(','))
                
    def get_result(self):
        """Return parsed CSV rows."""
        if not self.completed:
            raise StreamingError("Streaming not completed")
        return self.rows
```

### With Authentication

```python
from smartsurge import Client
from smartsurge.streaming import JSONStreamingRequest

client = Client()

# Authentication headers are automatically purged from saved state
headers = {
    "Authorization": "Bearer secret-token",
    "X-API-Key": "my-api-key"
}

result = client.stream_request(
    JSONStreamingRequest,
    "https://api.example.com/secure-data.json",
    headers=headers,
    state_file="secure_download.json"
)

# If resumed later, you'll need to provide auth headers again
```

## Integration with Client

The `Client.stream_request()` method handles:

```python
client = Client()

# With automatic rate limiting
result, history = client.stream_request(
    JSONStreamingRequest,
    "/api/large-file",
    state_file="download.json",
    return_history=True
)

# Check request history
print(f"Requests made: {len(history)}")
if history.rate_limit:
    print(f"Rate limit detected: {history.rate_limit}")
```

## Error Handling

```python
from smartsurge.exceptions import StreamingError, ResumeError

try:
    result = client.stream_request(
        JSONStreamingRequest,
        endpoint,
        state_file="download.json"
    )
except ResumeError as e:
    # Handle resume failures
    print(f"Resume failed: {e}")
    if e.state_file:
        # Clean up corrupt state
        os.remove(e.state_file)
except StreamingError as e:
    # Handle streaming errors
    print(f"Streaming error: {e}")
    if hasattr(e, 'response'):
        print(f"Status code: {e.response.status_code}")
```

## Performance Considerations

1. **Chunk Size**: Larger chunks (1MB+) for fast connections, smaller (8KB) for slow/unstable
2. **State Saving**: Saves every ~10 chunks by default, adjust based on stability needs
3. **Memory Usage**: Data accumulates in memory, consider streaming parsers for huge files
4. **Progress Logging**: Logs every ~10 chunks for files >10x chunk size

## Security Best Practices

1. **State Files**: Store in secure location with appropriate permissions
2. **Authentication**: Re-provide auth headers when resuming (they're purged from state)
3. **Validation**: Use ETag headers to ensure resource hasn't changed
4. **Cleanup**: Always remove state files after successful completion