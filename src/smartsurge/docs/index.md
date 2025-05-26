# SmartSurge Documentation

Welcome to SmartSurge, an intelligent HTTP client library that enhances the popular `requests` library with automatic rate limit detection and resumable streaming downloads.

## Overview

SmartSurge provides a drop-in replacement for the `requests` library that automatically handles rate limiting without requiring API-specific configuration. Using statistical analysis powered by Hidden Markov Models (HMM), SmartSurge learns and adapts to each API's rate limiting patterns in real-time.

### Key Features

- **🧠 Intelligent Rate Limit Detection** - Automatically detects and respects rate limits using HMM-based statistical analysis
- **📊 Adaptive Learning** - Continuously learns from API responses to improve rate limit predictions
- **⚡ Drop-in Replacement** - Works seamlessly with your existing requests-based code
- **🔄 Resumable Downloads** - Stream large files with automatic resume capability on connection failures
- **🔀 Async Support** - Full support for asynchronous operations using aiohttp
- **📈 Statistical Analysis** - Advanced rate limit detection using Hidden Markov Models
- **🎯 Multi-tier Rate Limiting** - Respects manual, server-provided, and learned rate limits

## Why SmartSurge?

Traditional approaches to rate limiting require:
- Manual configuration for each API
- Parsing API-specific headers
- Hardcoding rate limit values
- Maintaining per-API configurations

SmartSurge eliminates this complexity by:
- Learning rate limits automatically
- Adapting to changing API behaviors
- Providing accurate rate limit predictions
- Working with any HTTP API

## Quick Example

```python
from smartsurge import SmartSurgeClient

# Create a client - works just like requests!
client = SmartSurgeClient()

# Make requests without worrying about rate limits
for i in range(100):
    response = client.get("https://api.example.com/data")
    print(response.json())
```

SmartSurge will automatically detect when you're approaching rate limits and throttle requests accordingly.

## Documentation Structure

- **[Getting Started](getting_started.md)** - Installation, setup, and your first request
- **[Basic Usage](basic_usage.md)** - Common use cases and patterns
- **[Advanced Usage](advanced_usage.md)** - Configuration, customization, and advanced features
- **[Examples](examples.md)** - Real-world examples and best practices
- **[Benchmark Usage](benchmark_usage.md)** - Performance testing and benchmarking tools
- **[API Reference](api/)** - Complete API documentation
  - [Client](api/client.md) - SmartSurgeClient API
  - [Models](api/models.md) - Data models and types
  - [Exceptions](api/exceptions.md) - Error handling
  - [Streaming](api/streaming.md) - Streaming downloads
  - [HMM](api/hmm.md) - Rate limit detection internals

## How It Works

SmartSurge uses a three-tier approach to rate limiting:

1. **Manual Rate Limits** - If you know an API's rate limits, you can set them manually
2. **Server-Provided Limits** - Automatically reads rate limit headers from API responses
3. **HMM-Based Learning** - Uses statistical analysis to detect rate limits when not explicitly provided

The Hidden Markov Model analyzes the timing between your requests and the API's responses to identify patterns that indicate rate limiting behavior. This allows SmartSurge to proactively throttle requests before hitting hard rate limits.

## Installation

```bash
pip install smartsurge
```

For development or to run benchmarks:

```bash
# Install with development tools
pip install smartsurge[dev]

# Install with benchmarking support
pip install smartsurge[benchmark]

# Install with documentation tools
pip install smartsurge[docs]

# Install everything
pip install smartsurge[dev,benchmark,docs]
```

## Architecture

SmartSurge's architecture consists of several key components:

```
┌─────────────────┐
│ SmartSurgeClient│ ◄── Your Application
└────────┬────────┘
         │
    ┌────▼────┐     ┌──────────────┐
    │ Request │────►│ HMM Analyzer │
    │ History │     └──────────────┘
    └────┬────┘
         │
   ┌─────▼─────┐    ┌──────────────┐
   │   HTTP    │───►│  Streaming   │
   │  Handler  │    │   Handler    │
   └───────────┘    └──────────────┘
```

### Core Components

- **SmartSurgeClient**: Enhanced HTTP client that wraps requests/aiohttp
- **Request History**: Tracks all requests for pattern analysis
- **HMM Analyzer**: Statistical rate limit detection using Hidden Markov Models
- **HTTP Handler**: Manages request execution with retries and backoff
- **Streaming Handler**: Handles large downloads with resume capability

### Key Classes

- `SmartSurgeClient` - Main client class (aliased as `Client`)
- `ClientConfig` - Configuration management
- `RateLimit` - Rate limit representation
- `RequestHistory` - Request tracking and analysis
- `StreamingState` - Download state management
- `AbstractStreamingRequest` - Base class for custom streaming

## Performance

SmartSurge adds minimal overhead to your requests:
- Rate limit detection runs asynchronously
- HMM analysis is optimized with scipy
- Request history uses efficient circular buffers
- Streaming supports configurable chunk sizes

## Benchmarking

SmartSurge includes comprehensive benchmarking tools to test rate limit detection:

```python
from smartsurge import create_benchmark_server, SmartSurgeClient

# Create a mock server with known rate limits
server = create_benchmark_server(
    max_requests=10,
    time_period=1.0
)

# Test SmartSurge detection
client = SmartSurgeClient()
# ... run your tests
```

See the [Benchmark Usage](benchmark_usage.md) guide for more details.

## License

SmartSurge is released under the Apache License 2.0. See [LICENSE](https://github.com/dingo-actual/smartsurge/blob/main/LICENSE) for details.

## Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/dingo-actual/smartsurge/blob/main/CONTRIBUTING) for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/dingo-actual/smartsurge/issues)
- **Email**: ryan@beta-reduce.net
- **Source**: [GitHub Repository](https://github.com/dingo-actual/smartsurge)

## Version

Current version: 0.0.5

Check your installed version:
```python
import smartsurge
print(smartsurge.__version__)
```