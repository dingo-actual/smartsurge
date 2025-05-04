# Streaming API Reference

This page documents the streaming functionality in SmartSurge.

## `StreamingState`

```python
class StreamingState(BaseModel):
    """
    State of a streaming request for resumption.
    """
```

### Attributes

- `endpoint` - The endpoint being requested
- `method` - The HTTP method being used
- `headers` - HTTP headers for the request
- `params` - Optional query parameters
- `data` - Optional request body data
- `accumulated_data` - Data accumulated so far
- `last_position` - Last position in the stream
- `total_size` - Total size of the stream if known
- `etag` - ETag of the resource if available
- `last_updated` - Timestamp of when this state was last updated
- `request_id` - Unique identifier for the request (for tracking)

## `AbstractStreamingRequest`

```python
class AbstractStreamingRequest(ABC):
"""
Abstract base class for resumable streaming requests.
"""
```

### Constructor

```python
def __init__(self,
            endpoint: str,
            headers: Dict[str, str],
            chunk_size: int = 8192,
            state_file: Optional[str] = None,
            logger: Optional[logging.Logger] = None,
            request_id: Optional[str] = None)
```

Initializes a streaming request.

**Parameters:**

- `endpoint` - The endpoint to request
- `headers` - HTTP headers for the request
- `chunk_size` - Size of chunks to process
- `state_file` - File to save state for resumption
- `logger` - Optional custom logger to use
- `request_id` - Optional request ID for tracking and correlation

### Methods

```python
@abstractmethod
def start(self) -> None
```

Starts the streaming request.

```python
@abstractmethod
def resume(self) -> None
```

Resumes the streaming request from saved state.

```python
@abstractmethod
def process_chunk(self, chunk: bytes) -> None
```

Processes a chunk of data.

```python
@abstractmethod
def get_result(self) -> Any
```

Gets the final result after all chunks have been processed.

```python
def save_state(self) -> None
```

Saves the current state for resumption.

```python
def load_state(self) -> Optional[StreamingState]
```

Loads saved state for resumption.

## `JSONStreamingRequest`

```python
class JSONStreamingRequest(AbstractStreamingRequest):
    """
    A streaming request implementation that accumulates JSON data.
    """
```

### Constructor

```python
def __init__(self,
            endpoint: str,
            headers: Dict[str, str],
            params: Optional[Dict[str, Any]] = None,
            chunk_size: int = 8192,
            state_file: Optional[str] = None,
            logger: Optional[logging.Logger] = None,
            request_id: Optional[str] = None)
```

Initializes a JSON streaming request.

**Parameters:**

- `endpoint` - The endpoint to request
- `headers` - HTTP headers for the request
- `params` - Query parameters
- `chunk_size` - Size of chunks to process
- `state_file` - File to save state for resumption
- `logger` - Optional custom logger to use
- `request_id` - Optional request ID for tracking and correlation

### Methods

```python
def start(self) -> None
```

Starts the streaming request.

```python
def resume(self) -> None
```

Resumes the streaming request from saved state.

```python
def process_chunk(self, chunk: bytes) -> None
```

Processes a chunk of data.

```python
def get_result(self) -> Any
```

Gets the final result after all chunks have been processed.
