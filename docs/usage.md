# Usage

This guide provides a comprehensive overview of SmartSurge's functionality.

## Basic Usage

```python
from smartsurge import SmartSurgeClient

# Create a client

client = SmartSurgeClient(base_url="https://api.example.com")

# Make a GET request

response, history = client.get("/endpoint")

# Print response

print(response.status_code)
print(response.json())

# Access the detected rate limit

if history.rate_limit:
    print(f"Rate limit: {history.rate_limit.max_requests} requests per {history.rate_limit.time_period} seconds")

```

## SmartSurgeSmartSurgeClient Configuration

The SmartSurgeClient can be configured with various options:

```python
from smartsurge import SmartSurgeClient, configure_logging

# Configure logging

logger = configure_logging(level="DEBUG", output_file="smartsurge.log")

# Create a client with custom configuration

client = SmartSurgeClient(
    base_url="https://api.example.com",
    timeout=(5.0, 30.0),  # (connect timeout, read timeout)
    max_retries=5,
    backoff_factor=0.5,
    verify_ssl=True,
    min_time_period=0.5,  # Minimum period to consider for rate limiting (seconds)
    max_time_period=7200.0,  # Maximum period to consider for rate limiting (seconds)
    confidence_threshold=0.8,  # Confidence threshold for rate limit estimation
    logger=logger
)
```

## Making Requests

SmartSurge provides methods for all standard HTTP methods:

```python
# GET request

response, history = client.get("/endpoint", params={"param": "value"})

# POST request

response, history = client.post("/endpoint",
json={"key": "value"},
headers={"Custom-Header": "Value"})

# PUT request

response, history = client.put("/endpoint", data={"key": "value"})

# DELETE request

response, history = client.delete("/endpoint")

# PATCH request

response, history = client.patch("/endpoint", json={"key": "new_value"})
```

## Streaming Requests

SmartSurge supports resumable streaming requests:

```python
from smartsurge import JSONStreamingRequest

# Stream a large JSON response

result, history = client.stream_request(
streaming_class=JSONStreamingRequest,
endpoint="/large-dataset",
params={"limit": 10000},
state_file="download_state.json"  # For resumability
)

# Work with the parsed JSON result

for item in result:
process_item(item)
```

## Asynchronous Requests

For high-throughput applications, SmartSurge provides async request methods:

```python
import asyncio
from smartsurge import SmartSurgeClient

async def fetch_data():
client = SmartSurgeClient(base_url="https://api.example.com")

    # Make async requests
    response, history = await client.async_get("/endpoint")
    
    # Process the response
    data = await response.json()
    return data
    
# Run the async function

data = asyncio.run(fetch_data())
```

## Multiple Async Requests

For making multiple requests to the same endpoint pattern:

```python
import asyncio
from smartsurge import SmartSurgeClient, async_request_with_history

async def fetch_multiple():
client = SmartSurgeClient(base_url="https://api.example.com")

    endpoints = [f"/items/{i}" for i in range(1, 11)]
    
    # Make multiple async requests
    responses, histories = await async_request_with_history(
        client.async_get,
        endpoints=endpoints,
        method="GET",
        max_concurrent=3  # Limit concurrency
    )
    
    # Process responses
    results = []
    for response in responses:
        if not isinstance(response, Exception):
            data = await response.json()
            results.append(data)
    
    return results
    
# Run the async function

results = asyncio.run(fetch_multiple())
```

## Error Handling

SmartSurge provides specific exceptions for different error cases:

```python
from smartsurge import SmartSurgeClient, RateLimitExceeded, StreamingError

client = SmartSurgeClient(base_url="https://api.example.com")

try:
    response, history = client.get("/endpoint")
except RateLimitExceeded as e:
    print(f"Rate limit exceeded: {e}")
    print(f"Retry after: {e.retry_after} seconds")
except StreamingError as e:
    print(f"Streaming error: {e}")
    print(f"Position: {e.position}")
except Exception as e:
    print(f"Other error: {e}")
```

## Context Manager

SmartSurge's SmartSurgeClient can be used as a context manager:

```python
from smartsurge import SmartSurgeClient

with SmartSurgeClient(base_url="https://api.example.com") as client:
    response, history = client.get("/endpoint")
# SmartSurgeClient is automatically closed when exiting the context
```

See the [API Reference](api/client.md) for detailed documentation of all available methods and options.
