# Using SmartSurge Benchmarks

SmartSurge provides comprehensive benchmarking tools as an optional feature. The benchmarks are designed to demonstrate SmartSurge's HMM-based rate limit detection advantages over traditional exponential backoff approaches.

## Installation

Install SmartSurge with benchmark support:

```bash
pip install smartsurge[benchmark]
```

This adds the following dependencies:
- `pyperf>=2.0.0` - Statistical benchmarking framework
- `matplotlib>=3.0.0` - Visualization tools
- `numpy>=2.0.0` - Numerical computations
- `flask>=3.0.0` and `werkzeug>=3.0.0` - Mock server implementation

Core package without benchmarks:

```bash
pip install smartsurge
```

## Checking Availability

Check if benchmarks are available at runtime:

```python
import smartsurge

if smartsurge.has_benchmarks():
    print("Benchmarks are available!")
    from smartsurge import create_benchmark_server
else:
    print("Benchmarks not installed. Install with: pip install smartsurge[benchmark]")
```

## Available Components

### Mock Server Components

The benchmark package provides a sophisticated mock server for testing rate limiting scenarios:

```python
from smartsurge import (
    BenchmarkMockServer,      # Main mock server class
    RateLimitConfig,          # Configuration dataclass
    RateLimitStrategy,        # Enum of rate limiting strategies
    create_benchmark_server,  # Factory function
)
```

### Rate Limiting Strategies

The `RateLimitStrategy` enum includes:
- `FIXED_WINDOW` - Traditional fixed time window limits
- `SLIDING_WINDOW` - Rolling window rate limits
- `TOKEN_BUCKET` - Token bucket algorithm
- `ADAPTIVE` - Adjusts limits based on usage patterns
- `DYNAMIC` - Time-varying rate limits
- `LOAD_DEPENDENT` - Server load-based limits
- `NONE` - No rate limiting (for baseline comparisons)

### Configuration Presets

Pre-configured rate limit scenarios:

```python
from smartsurge import (
    get_strict_rate_limit_config,  # Fixed window, strict enforcement
    get_token_bucket_config,       # Token bucket with burst capacity
    get_noisy_config,              # Adds random noise to responses
    get_adaptive_config,           # Adapts to usage patterns
    get_dynamic_config,            # Time-varying limits
    get_load_dependent_config,     # Server load-based limits
)
```

### Baseline Client Components

Reference implementation for comparison:

```python
from smartsurge import (
    BaselineAsyncClient,      # Traditional exponential backoff client
    BaselineMetrics,          # Metrics collection for baseline
    collect_baseline_metrics, # Helper for metrics collection
)
```

## Command Line Interface

The `smartsurge-benchmark` command provides several modes:

```bash
# Quick demonstration
smartsurge-benchmark --mode demo

# Comprehensive benchmark suite (default)
smartsurge-benchmark --mode comprehensive

# Full benchmark suite with all scenarios
smartsurge-benchmark --mode full

# Visualize existing results
smartsurge-benchmark --mode visualize

# Additional options
smartsurge-benchmark --comprehensive --pyperf --no-visualize
```

Options:
- `--mode`: Choose benchmark mode (demo, comprehensive, full, visualize)
- `--pyperf`: Use pyperf for statistical rigor (recommended)
- `--no-visualize`: Skip visualization after benchmarks
- `--comprehensive`: Shorthand for comprehensive mode

## Example Usage

### Basic Mock Server Setup

```python
import smartsurge

if smartsurge.has_benchmarks():
    # Create a mock server with strict rate limiting
    config = smartsurge.get_strict_rate_limit_config()
    server = smartsurge.create_benchmark_server(config)
    
    # Start the server (runs in background thread)
    server_url = server.start()
    
    # Use with SmartSurge client
    client = smartsurge.SmartSurgeClient()
    response = client.get(f"{server_url}/api/endpoint")
    
    # Check server metrics
    metrics = server.get_metrics()
    print(f"Total requests: {metrics['total_requests']}")
    print(f"Rate limited: {metrics['rate_limited_requests']}")
    
    # Stop the server
    server.stop()
```

### Custom Rate Limit Configuration

```python
from smartsurge import RateLimitConfig, RateLimitStrategy

custom_config = RateLimitConfig(
    strategy=RateLimitStrategy.TOKEN_BUCKET,
    requests_per_window=100,
    window_size_seconds=60,
    burst_capacity=20,
    refill_rate=2.0,  # tokens per second
)

server = smartsurge.create_benchmark_server(custom_config)
```

### Running Effectiveness Benchmarks

```python
# Programmatically run benchmark scenarios
from smartsurge.benchmarks.benchmark_hmm_effectiveness import run_scenario

results = run_scenario(
    scenario_name="strict_rate_limit",
    config=smartsurge.get_strict_rate_limit_config(),
    duration=30,
    use_pyperf=True
)
```

## Benchmark Architecture

The benchmarking system consists of:

1. **Mock Server** (`mock_server.py`): Configurable server implementing various rate limiting strategies
2. **Baseline Client** (`baseline_client.py`): Traditional exponential backoff implementation
3. **Effectiveness Benchmarks** (`benchmark_hmm_effectiveness.py`): Compares SmartSurge vs baseline
4. **Performance Benchmarks** (`benchmark_hmm_performance.py`): HMM algorithm performance analysis
5. **Benchmark Suite** (`run_benchmark_suite.py`): Orchestrates complete benchmark runs
6. **Visualization** (`visualize_results.py`): ASCII charts and comparison tables

## Conditional Import Pattern

For libraries that optionally use benchmarks:

```python
import smartsurge

HAS_BENCHMARKS = smartsurge.has_benchmarks()

def run_performance_test():
    if not HAS_BENCHMARKS:
        raise RuntimeError(
            "Benchmarks not available. "
            "Install with: pip install smartsurge[benchmark]"
        )
    
    from smartsurge import create_benchmark_server, get_adaptive_config
    
    config = get_adaptive_config()
    server = create_benchmark_server(config)
    # ... run tests
```

## Error Handling

Handle missing benchmark functionality gracefully:

```python
try:
    from smartsurge import create_benchmark_server
except AttributeError:
    print("Benchmarks not installed. Install with: pip install smartsurge[benchmark]")
```

Or check availability before import:

```python
import smartsurge

if hasattr(smartsurge, 'create_benchmark_server'):
    server = smartsurge.create_benchmark_server()
else:
    print("Benchmark functionality not available")
```
