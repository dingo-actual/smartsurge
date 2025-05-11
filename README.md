# SmartSurge: Enhanced Requests Library with Adaptive Rate Limit Estimation

SmartSurge is a Python library that enhances the functionality of the requests library with automatic rate limit detection and enforcement, using a statistically rigorous approach.

## Features

- **Automatic Rate Limit Detection**: Automatically detects rate limits through a principled search procedure.
- **Resumable Streaming Requests**: Supports resumable streaming for large downloads.
- **Comprehensive Logging**: Detailed logging of request history and rate limit estimation.
- **Async Support**: Includes asynchronous request methods using aiohttp.

## Installation

```bash
pip install smartsurge

```

## Quick Start

```python
from smartsurge import SmartSurgeClient

# Create a client

client = SmartSurgeClient(base_url="https://api.example.com")

# Make requests - SmartSurge will automatically detect and respect rate limits

response, history = client.get("/endpoint")

# Get the detected rate limit

rate_limit = history.rate_limit
print(f"Detected rate limit: {rate_limit}")

# Streaming requests with resumability

from smartsurge import JSONStreamingRequest


result, history = client.stream_request(
    streaming_class=JSONStreamingRequest,
    endpoint="/large-dataset",
    state_file="download_state.json"  # For resumability
)
```

## Documentation

> In progress.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
