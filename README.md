# SmartSurge: Intelligent HTTP Client with Adaptive Rate Limit Detection

<img src="images/smartsurge.png" alt="SmartSurge logo" width="1280" height="640">
<br>
<br>

[![PyPI version](https://img.shields.io/pypi/v/smartsurge.svg)](https://pypi.org/project/smartsurge/)
[![Python versions](https://img.shields.io/pypi/pyversions/smartsurge.svg)](https://pypi.org/project/smartsurge/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

SmartSurge is an intelligent HTTP client library that enhances the popular `requests` library with automatic rate limit detection, handling, and request throttling. Using statistical analysis and machine learning techniques, SmartSurge can automatically detect and adapt to rate limits without requiring any API-specific configuration.

## 🚀 Key Features

- **⚡ Drop-in Replacement**: Works seamlessly with your existing requests-based code
- **🔄 Resumable Downloads**: Stream large files with automatic resume capability on connection failures
- **🔀 Async Support**: Full support for asynchronous operations using aiohttp
- **📊 Multi-Tiered Rate Limit Detection**: Supports manual, server-provided, and HMM-based rate limit detection
- **🧠 Intelligent Rate Limit Detection**: Automatically detects rate limits using Hidden Markov Models (HMM)
- **🛡️ Automatic Retries**: Built-in retry logic with exponential backoff

> **IMPORTANT NOTE**: The Hidden Markov Model component is currently under development and is not yet production-ready.

## 📦 Installation

```bash
pip install smartsurge
```

For development or additional features:

```bash
# Install with development tools
pip install smartsurge[dev]

# Install with benchmarking tools
pip install smartsurge[benchmark]

# Install with documentation tools
pip install smartsurge[docs]

# Install everything
pip install smartsurge[dev,benchmark,docs]
```

## 🎯 Quick Start

### Basic Usage

```python
from smartsurge import SmartSurgeClient

# Create a client - it works just like requests!
client = SmartSurgeClient()

# Make requests without worrying about rate limits
response = client.get("https://api.example.com/data")
print(response.json())

# Get response with request history for insights
response, history = client.get("https://api.example.com/data", return_history=True)

# Check detected rate limits
limits = client.list_rate_limits()
for (endpoint, method), limit in limits.items():
    if limit:
        print(f"{endpoint} {method}: {limit.max_requests} requests per {limit.time_period}s")
```

### Configuration

```python
from smartsurge import SmartSurgeClient

# Create a client with custom configuration
client = SmartSurgeClient(
    base_url="https://api.example.com",
    timeout=(10.0, 30.0),      # (connect, read) timeouts
    max_retries=3,             # Maximum retry attempts
    backoff_factor=0.3,        # Exponential backoff multiplier
    refit_every=20,            # Refit HMM every N requests
)

# All requests will use the base URL
response = client.get("/users")  # Requests https://api.example.com/users
```

### Manual Rate Limit Configuration

```python
# Set a known rate limit
client.set_rate_limit(
    endpoint="https://api.example.com/users",
    method="GET",
    max_requests=100,
    time_period=60.0  # 100 requests per minute
)

# Check current rate limit
rate_limit = client.get_rate_limit("https://api.example.com/users", "GET")
if rate_limit:
    print(f"Rate limit: {rate_limit.max_requests} per {rate_limit.time_period}s")
    print(f"Source: {rate_limit.source}")  # 'manual', 'header', or 'estimated'
```

### Streaming Large Files

```python
from smartsurge import JSONStreamingRequest

# Stream large files with automatic resume capability
result = client.stream_request(
    streaming_class=JSONStreamingRequest,
    endpoint="https://example.com/large-dataset.json",
    state_file="download_state.json",  # Automatically saves progress
    chunk_size=1024 * 1024  # 1MB chunks
)

# If interrupted, the download will resume from where it left off
```

### Async Support

```python
import asyncio

async def fetch_data():
    client = SmartSurgeClient()
    
    # Make async requests
    response = await client.async_get("https://api.example.com/data")
    data = await response.json()
    
    # Parallel requests
    tasks = [
        client.async_get(f"https://api.example.com/items/{i}")
        for i in range(10)
    ]
    responses = await asyncio.gather(*tasks)
    
    return responses

# Run the async function
results = asyncio.run(fetch_data())
```

## 🧠 How It Works

SmartSurge uses a three-tier approach to rate limiting:

1. **Manual Rate Limits** (Highest Priority)
   - Explicitly set rate limits using `set_rate_limit()`
   - Useful when you know the API's limits

2. **Server-Provided Limits**
   - Automatically reads standard rate limit headers:
     - `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`
     - `Retry-After` headers from 429 responses

3. **HMM-Based Learning** (Automatic)
   - Uses a 3-state Hidden Markov Model to detect patterns:
     - **State 0**: Normal operation
     - **State 1**: Approaching rate limit
     - **State 2**: Rate limited
   - Continuously adapts to changing patterns

## 📊 Advanced Features

### Request History Analysis

```python
# Get detailed request history
response, history = client.get("/api/data", return_history=True)

# Analyze the history
print(f"Total requests: {len(history.requests)}")

# Check if rate limit was detected
if history.rate_limit:
    print(f"Detected limit: {history.rate_limit}"))
```

### Custom Configuration

```python
from smartsurge import ClientConfig

# Create reusable configuration
config = ClientConfig(
    base_url="https://api.example.com",
    timeout=(5.0, 30.0),
    max_retries=5,
    backoff_factor=0.5,
    
    # HMM Configuration
    min_time_period=1.0,       # Minimum rate limit window (seconds)
    max_time_period=3600.0,    # Maximum rate limit window (1 hour)
    refit_every=20,            # Refit HMM every N requests
    
    # Other options
    verify_ssl=True,
    user_agent="MyApp/1.0 (SmartSurge)"
)

client = SmartSurgeClient(**config.model_dump())
```

### Error Handling

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
except Exception as e:
    print(f"Unexpected error: {e}")
```

## 🔧 Utilities

SmartSurge includes helpful utilities:

```python
from smartsurge import (
    SmartSurgeTimer,
    configure_logging,
    merge_histories,
    async_request_with_history
)

# Configure logging
configure_logging(level=logging.DEBUG)

# Time your requests
with SmartSurgeTimer() as timer:
    response = client.get("/api/data")
print(f"Request took {timer.duration:.3f} seconds")

# Merge multiple histories
history1 = client.get("/api/users", return_history=True)[1]
history2 = client.get("/api/posts", return_history=True)[1]
combined = merge_histories([history1, history2])
```

## 📈 Benchmarking

When installed with benchmark extras, you can test rate limit detection:

```python
from smartsurge import create_benchmark_server, SmartSurgeClient

# Create a mock server with known rate limits
server = create_benchmark_server(
    max_requests=10,
    time_period=1.0  # 10 requests per second
)

# Test SmartSurge against it
client = SmartSurgeClient()
# ... run your tests
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING) for details.

## 📄 License

SmartSurge is released under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Documentation**: [Full documentation](src/smartsurge/docs/index.md)
- **PyPI**: [smartsurge](https://pypi.org/project/smartsurge/)
- **GitHub**: [dingo-actual/smartsurge](https://github.com/dingo-actual/smartsurge)
- **Issues**: [Report bugs or request features](https://github.com/dingo-actual/smartsurge/issues)

## 📧 Contact

- **Author**: Ryan Taylor
- **Email**: [ryan@beta-reduce.net](mailto:ryan@beta-reduce.net)

## 🏆 Credits

SmartSurge uses Hidden Markov Models for intelligent rate limit detection, leveraging the power of statistical learning to provide a truly adaptive HTTP client experience.
