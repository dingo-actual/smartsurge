# SmartSurge API Reference

Welcome to the SmartSurge API documentation. SmartSurge enhances the requests library with intelligent rate limit detection using Hidden Markov Models and provides resumable streaming capabilities for robust HTTP operations.

## Installation

```bash
pip install smartsurge

# With benchmark extras
pip install smartsurge[benchmark]
```

## Core Components

### [Client](client.md)
The `SmartSurgeClient` class is your primary interface to SmartSurge. It wraps standard HTTP libraries (requests/aiohttp) with automatic rate limit detection and retry logic.

```python
from smartsurge import SmartSurgeClient

client = SmartSurgeClient()
response = client.get("https://api.example.com/data")
```

**Key Classes:**
- `SmartSurgeClient`: Main client class
- `ClientConfig`: Configuration options

### [Streaming](streaming.md)
Provides resumable download capabilities for large files with automatic state management and security features.

```python
from smartsurge import JSONStreamingRequest

result = client.stream_request(
    JSONStreamingRequest,
    "https://api.example.com/large-file.json",
    state_file="download.state"
)
```

**Key Classes:**
- `AbstractStreamingRequest`: Base class for streaming implementations
- `JSONStreamingRequest`: JSON-specific streaming handler
- `StreamingState`: State management for resumption

### [HMM Rate Detection](hmm.md)
Hidden Markov Model implementation that statistically analyzes request patterns to detect rate limits without API-specific configuration.

```python
from smartsurge.hmm import HMM, HMMParams

hmm = HMM()
hmm.fit_mle(observations)
max_requests, time_period = hmm.predict_rate_limit(observations)
```

**Key Classes:**
- `HMM`: Hidden Markov Model implementation
- `HMMParams`: HMM configuration parameters

### [Models](models.md)
Core data models and enumerations for structured data representation throughout the library.

```python
from smartsurge.models import RequestMethod, RequestEntry, RequestHistory

history = RequestHistory(
    endpoint="/api/users",
    method=RequestMethod.GET
)
```

**Key Classes:**
- `RequestMethod`: HTTP method enumeration
- `SearchStatus`: Rate limit search status
- `RequestEntry`: Individual request metadata
- `RequestHistory`: Request collection with HMM integration
- `RateLimit`: Rate limit information

### [Exceptions](exceptions.md)
Comprehensive exception hierarchy for precise error handling with automatic logging and context preservation.

```python
from smartsurge import (
    RateLimitExceeded,
    StreamingError,
    ResumeError,
    ValidationError,
    ConfigurationError
)
```

**Exception Hierarchy:**
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

### [Utilities](utilities.md)
Helper functions and utilities for enhanced functionality.

```python
from smartsurge import (
    SmartSurgeTimer,
    log_context,
    merge_histories,
    async_request_with_history,
    configure_logging
)
```

**Key Utilities:**
- `SmartSurgeTimer`: Performance timing context manager
- `log_context`: Contextual logging
- `merge_histories`: Combine request histories
- `async_request_with_history`: Async request wrapper
- `configure_logging`: Logging configuration

## Architecture Overview

```
┌─────────────────────┐
│  SmartSurgeClient   │ ← Main entry point
├─────────────────────┤
│  Request History    │ ← Per-endpoint tracking
├─────────────────────┤
│   HMM Analyzer      │ ← Statistical detection
├─────────────────────┤
│  Retry Strategy     │ ← Exponential backoff
└─────────────────────┘
         ↓
┌─────────────────────┐
│ Streaming Handler   │ ← Resumable downloads
└─────────────────────┘
```

## Key Features

### Three-Tier Rate Limiting

SmartSurge employs a hierarchical approach:

1. **User-defined**: Explicit rate limits via `set_rate_limit()`
2. **Server-provided**: HTTP 429 `Retry-After` headers
3. **Adaptive detection**: HMM-based statistical analysis

### Request History Tracking

- Isolated per endpoint + HTTP method combination
- Limited to `max_observations` (default: 50) for memory efficiency
- Tracks success rates, latencies, and status codes
- Automatic HMM retraining every N responses

### Resumable Streaming

- Automatic state persistence with security
- Authentication header purging from saved state
- Range header support for partial content
- ETag validation for resource integrity

### Statistical Analysis

- Advanced rate limit detection
- Parameter optimization using MLE and EM algorithms
- Automatic model fitting and prediction

## Quick Start Examples

### Basic Usage

```python
from smartsurge import SmartSurgeClient

# Simple requests
client = SmartSurgeClient()
response = client.get("https://api.example.com/data")
print(response.json())
```

### With Configuration

```python
client = SmartSurgeClient(
    base_url="https://api.example.com",
    max_retries=5,
    timeout=(10.0, 30.0)         # (connect, read) timeouts
)

# Manual rate limit
client.set_rate_limit("/api/users", "GET", max_requests=100, time_period=60.0)
```

### Async Operations

```python
import asyncio

async def fetch_data():
    async with SmartSurgeClient() as client:
        response = await client.async_get("/api/data")
        return await response.json()

data = asyncio.run(fetch_data())
```

### Streaming with Resume

```python
from smartsurge import SmartSurgeClient, JSONStreamingRequest

client = SmartSurgeClient()

try:
    result = client.stream_request(
        JSONStreamingRequest,
        "https://api.example.com/huge-file.json",
        state_file="download.state",
        chunk_size=1024 * 1024  # 1MB chunks
    )
except KeyboardInterrupt:
    print("Download paused. Run again to resume.")
```

### Error Handling

```python
from smartsurge import SmartSurgeClient, RateLimitExceeded

client = SmartSurgeClient()

try:
    response = client.get("/api/data")
except RateLimitExceeded as e:
    print(f"Rate limited: {e.endpoint}")
    if e.retry_after:
        print(f"Retry after {e.retry_after} seconds")
```

## Benchmark Support

Check if benchmarks are available:

```python
import smartsurge

if smartsurge.has_benchmarks():
    from smartsurge import create_benchmark_server
    server = create_benchmark_server()
else:
    print("Install with: pip install smartsurge[benchmark]")
```

## Public API Exports

The following are available directly from the `smartsurge` module:

```python
# Core client
from smartsurge import SmartSurgeClient, ClientConfig

# Exceptions
from smartsurge import (
    RateLimitExceeded,
    StreamingError,
    ResumeError,
    ValidationError,
    ConfigurationError
)

# Streaming
from smartsurge import (
    StreamingState,
    AbstractStreamingRequest,
    JSONStreamingRequest
)

# Utilities
from smartsurge import (
    SmartSurgeTimer,
    log_context,
    merge_histories,
    async_request_with_history,
    configure_logging,
    has_benchmarks
)

# Version
from smartsurge import __version__
```

Note: Models (`RequestMethod`, `RequestHistory`, etc.) and HMM classes must be imported from their respective submodules.

## Performance Considerations

1. **Memory**: Each endpoint/method maintains up to `max_observations` entries
2. **HMM Training**: Initial training after `min_data_points` observations
3. **Retraining**: Occurs every `refit_every` responses (default: 10)
4. **State Files**: Streaming state files should be stored securely

## Next Steps

- Explore the [Client API](client.md) for detailed HTTP operations
- Learn about [Streaming](streaming.md) for large file downloads
- Understand [HMM](hmm.md) statistical rate detection
- Review [Models](models.md) for data structures
- Handle [Exceptions](exceptions.md) properly
- Use [Utilities](utilities.md) for enhanced functionality