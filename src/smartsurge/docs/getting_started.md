# Getting Started with SmartSurge

This guide will help you get up and running with SmartSurge in just a few minutes.

## Installation

### Basic Installation

Install SmartSurge using pip:

```bash
pip install smartsurge
```

### Installation with Extras

SmartSurge offers several optional extras for additional functionality:

```bash
# Install with development tools (testing, linting, etc.)
pip install smartsurge[dev]

# Install with benchmarking tools
pip install smartsurge[benchmark]

# Install with documentation tools
pip install smartsurge[docs]

# Install all extras
pip install smartsurge[dev,benchmark,docs]
```

### Requirements

- Python 3.9 or higher
- Dependencies are automatically installed:
  - `requests>=2.0.0` - HTTP library
  - `aiohttp>=3.0.0` - Async HTTP support
  - `pydantic>=2.0.0` - Data validation
  - `scipy>=1.0.0` - Statistical computations for HMM

## Quick Start

### Your First Request

Here's the simplest way to get started:

```python
from smartsurge import SmartSurgeClient

# Create a client - it works just like requests!
client = SmartSurgeClient()

# Make a simple GET request
response = client.get("https://api.github.com/users/github")

# Use the response exactly like requests
print(response.status_code)
print(response.json())
```

### Automatic Rate Limit Detection

SmartSurge automatically handles rate limiting for you:

```python
from smartsurge import SmartSurgeClient

client = SmartSurgeClient()

# Make many requests - SmartSurge will handle rate limits automatically
for i in range(100):
    response = client.get("https://api.example.com/data")
    print(f"Request {i+1}: {response.status_code}")
```

### Getting Rate Limit Information

Want to see what SmartSurge learned? Request the history:

```python
# Get response with history
response, history = client.get("https://api.example.com/data", return_history=True)

# Check for detected rate limits
limits = client.list_rate_limits()
for (endpoint, method), limit in limits.items():
    if limit:
        print(f"{endpoint} {method}: {limit.max_requests}/{limit.time_period}s")
        print(f"  Source: {limit.source}")
```

## Configuration

### Basic Configuration

Configure SmartSurge behavior during initialization:

```python
from smartsurge import SmartSurgeClient

client = SmartSurgeClient(
    timeout=30.0,              # Request timeout in seconds
    max_retries=5,             # Maximum retry attempts
)
```

### Advanced Configuration

Fine-tune SmartSurge for your specific needs:

```python
from smartsurge import SmartSurgeClient

client = SmartSurgeClient(
    # Base URL for all requests
    base_url="https://api.example.com",
    
    # Timeout settings
    timeout=(5.0, 30.0),       # (connect, read) timeouts
    
    # Retry settings
    max_retries=10,
    backoff_factor=0.3,        # Exponential backoff multiplier
    
    # Rate limit detection
    min_time_period=1.0,       # Minimum rate limit window (seconds)
    max_time_period=3600.0,    # Maximum rate limit window (seconds)
    
    # HMM settings
    refit_every=20,            # Refit HMM after N responses
    
    # SSL and verification
    verify_ssl=True,
)
```

### Using ClientConfig

For reusable configurations:

```python
from smartsurge import SmartSurgeClient, ClientConfig

# Create configuration
config = ClientConfig(
    base_url="https://api.example.com",
    timeout=(10.0, 30.0),
    max_retries=3,
    user_agent="MyApp/1.0 (SmartSurge)"
)

# Create multiple clients with same config
client1 = SmartSurgeClient(**config.model_dump())
client2 = SmartSurgeClient(**config.model_dump())
```

### Setting Headers and Authentication

```python
# Pass headers in requests
response = client.get(
    "https://api.example.com/data",
    headers={
        "Authorization": "Bearer your-token",
        "User-Agent": "MyApp/1.0"
    }
)

# Use basic auth
response = client.get(
    "https://api.example.com/data",
    auth=("username", "password")
)
```

## Async Support

SmartSurge fully supports asynchronous operations:

```python
import asyncio
from smartsurge import SmartSurgeClient

async def fetch_data():
    client = SmartSurgeClient()
    
    # Make async requests
    tasks = []
    for i in range(10):
        task = client.async_get(f"https://api.example.com/data/{i}")
        tasks.append(task)
    
    # Wait for all requests
    responses = await asyncio.gather(*tasks)
    
    # Process responses
    for response in responses:
        data = await response.json()
        print(f"Status: {response.status}")

# Run the async function
asyncio.run(fetch_data())
```

## Streaming Large Files

Download large files with the streaming API:

```python
from smartsurge import SmartSurgeClient, JSONStreamingRequest

client = SmartSurgeClient()

# Basic streaming
response = client.get(
    "https://example.com/large-file.zip",
    stream=True
)

# Save to file
with open("large-file.zip", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)

# Advanced streaming with resume support
result = client.stream_request(
    streaming_class=JSONStreamingRequest,
    endpoint="https://example.com/large-data.json",
    state_file="download.state",  # Automatically saves/resumes
    chunk_size=1024 * 1024  # 1MB chunks
)
```

## Understanding Rate Limit Detection

SmartSurge uses a three-tier hierarchy for rate limiting:

### 1. Manual Rate Limits (Highest Priority)

Set explicit rate limits when you know them:

```python
client = SmartSurgeClient()

# Set rate limit for a specific endpoint and method
client.set_rate_limit(
    endpoint="https://api.example.com/users",
    method="GET",
    max_requests=100,
    time_period=60  # 100 requests per minute
)

# Set rate limit for all methods on an endpoint
client.set_rate_limit(
    endpoint="https://api.example.com/data",
    method="*",  # Applies to all HTTP methods
    max_requests=1000,
    time_period=3600  # 1000 requests per hour
)
```

### 2. Server-Provided Rate Limits

SmartSurge automatically reads standard rate limit headers:
- `X-RateLimit-Limit`
- `X-RateLimit-Remaining`
- `X-RateLimit-Reset`
- `Retry-After`

### 3. HMM-Based Detection (Automatic)

When no rate limit information is available, SmartSurge learns:

```python
# SmartSurge learns automatically as you make requests
for i in range(50):
    response = client.get("https://api.example.com/data")
    
# Check what was learned
limits = client.list_rate_limits()
for (endpoint, method), limit in limits.items():
    if limit and limit.source == "estimated":
        print(f"Learned: {endpoint} allows {limit.max_requests} requests per {limit.time_period}s")
```

## Error Handling

SmartSurge provides specific exceptions for different scenarios:

```python
from smartsurge import (
    SmartSurgeClient,
    RateLimitExceeded,
    ConnectionError,
    TimeoutError,
    ServerError,
    ClientError,
    ValidationError
)

client = SmartSurgeClient()

try:
    response = client.get("https://api.example.com/data")
except RateLimitExceeded as e:
    print(f"Rate limit hit: {e.message}")
    if e.retry_after:
        print(f"Retry after: {e.retry_after} seconds")
except ConnectionError as e:
    print(f"Connection failed: {e.message}")
except TimeoutError as e:
    print(f"Request timed out: {e.message}")
except ServerError as e:
    print(f"Server error {e.status_code}: {e.message}")
except ClientError as e:
    print(f"Client error {e.status_code}: {e.message}")
```

## Logging and Debugging

Enable detailed logging to see what SmartSurge is doing:

```python
import logging
from smartsurge import SmartSurgeClient, configure_logging

# Configure SmartSurge logging
configure_logging(level=logging.DEBUG)

# Create client
client = SmartSurgeClient()

# Make requests - you'll see detailed logs
response = client.get("https://api.example.com/data")
```

## Best Practices

1. **Reuse Client Instances**: Create one client and reuse it for multiple requests
2. **Set Known Rate Limits**: If you know the API's rate limits, set them manually
3. **Handle Errors Gracefully**: Always wrap requests in try-except blocks
4. **Use Async for Bulk Requests**: Use async methods when making many concurrent requests
5. **Enable Logging in Development**: Use logging to understand SmartSurge's behavior

## Next Steps

Now that you're up and running:

1. **[Basic Usage](basic_usage.md)** - Learn common patterns and use cases
2. **[Advanced Usage](advanced_usage.md)** - Explore advanced features
3. **[Examples](examples.md)** - See real-world examples
4. **[API Reference](api/index.md)** - Detailed API documentation

## Troubleshooting

### Connection Issues

```python
# Increase timeout for slow connections
client = SmartSurgeClient(timeout=60.0)

# Disable SSL verification for testing (not for production!)
client = SmartSurgeClient(verify_ssl=False)
```

### Rate Limit Detection Issues

```python
# Adjust refit frequency if detection needs more data
client = SmartSurgeClient(refit_every=50)  # Less frequent refitting

# Or manually set rate limits
client.set_rate_limit(
    endpoint="https://api.example.com/data",
    method="GET",
    max_requests=10,
    time_period=1  # 10 requests per second
)
```

### Memory Issues with Large Downloads

```python
# Use streaming for large files
response = client.get(url, stream=True)
for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
    process_chunk(chunk)
```

## Getting Help

- **Documentation**: You're reading it!
- **GitHub Issues**: [Report bugs or request features](https://github.com/dingo-actual/smartsurge/issues)
- **Examples**: Check the [examples directory](examples.md)
- **Email**: ryan@beta-reduce.net

## Version Information

Check your SmartSurge version:

```python
import smartsurge
print(smartsurge.__version__)
```

Check if benchmark extras are installed:

```python
import smartsurge
if smartsurge.has_benchmarks():
    print("Benchmarks available!")
else:
    print("Install with: pip install smartsurge[benchmark]")
```