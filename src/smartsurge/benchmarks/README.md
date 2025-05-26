# SmartSurge Benchmarks

Comprehensive benchmarking tools for evaluating SmartSurge's HMM-based rate limit detection against traditional exponential backoff approaches.

## Quick Start

```bash
# Interactive benchmark suite
python -m smartsurge.benchmarks.run_benchmark_suite

# Run specific benchmark mode
python -m smartsurge.benchmarks.run_benchmark_suite --mode demo
python -m smartsurge.benchmarks.run_benchmark_suite --mode comprehensive
python -m smartsurge.benchmarks.run_benchmark_suite --mode full
```

## Directory Structure

- **`run_benchmark_suite.py`** - Main orchestrator for running benchmarks
- **`benchmark_hmm_effectiveness.py`** - Core benchmark comparing SmartSurge vs baseline
- **`benchmark_hmm_performance.py`** - HMM algorithm performance measurements
- **`mock_server.py`** - Configurable server with multiple rate limiting strategies
- **`baseline_client.py`** - Reference implementation using exponential backoff
- **`run_hmm_benchmark.sh`** - Shell script helper for HMM benchmarks

## Key Components

### 1. Benchmark Suite Runner

The main entry point for running benchmarks:

```bash
# Interactive mode (presents menu)
python -m smartsurge.benchmarks.run_benchmark_suite

# Command-line options
python -m smartsurge.benchmarks.run_benchmark_suite --mode demo --log-level DEBUG
```

**Modes:**
- `demo` - Quick demonstration (~30 seconds)
- `comprehensive` - All rate limiting strategies (~2 minutes)
- `full` - Comprehensive + pyperf statistical analysis (~5 minutes)

### 2. HMM Effectiveness Benchmark

Compares SmartSurge against baseline across various scenarios:

```bash
# Single scenario demo
python -m smartsurge.benchmarks.benchmark_hmm_effectiveness --mode demo

# All scenarios
python -m smartsurge.benchmarks.benchmark_hmm_effectiveness --mode comprehensive

# Custom parameters
python -m smartsurge.benchmarks.benchmark_hmm_effectiveness \
    --requests 200 \
    --delay 0.05 \
    --output results.json \
    --log-level DEBUG
```

### 3. HMM Performance Benchmark

Measures computational performance of the HMM fitting process:

```bash
# Run performance benchmarks
python -m smartsurge.benchmarks.benchmark_hmm_performance

# Or use the shell script
./src/smartsurge/benchmarks/run_hmm_benchmark.sh
```

## Rate Limiting Strategies

The mock server implements six different strategies:

### Strict Fixed Window
- Hard rate limit enforcement
- Resets at window boundaries
- Config: `get_strict_rate_limit_config()`

### Token Bucket
- Allows bursts up to bucket size
- Steady refill rate
- Config: `get_token_bucket_config()`

### Adaptive
- Adjusts limits based on client behavior
- Simulates smart rate limiting
- Config: `get_adaptive_config()`

### Dynamic
- Changes limits over time
- Tests adaptation capabilities
- Config: `get_dynamic_config()`

### Noisy
- Random delays and false positives
- Real-world network simulation
- Config: `get_noisy_config()`

### Load-Dependent
- Sinusoidal load patterns
- Smooth limit transitions
- Config: `get_load_dependent_config()`

## Metrics Collected

### Detection Metrics
- **Time to Detection**: When HMM identifies the rate limit
- **Detection Accuracy**: Error percentage vs actual limit
- **Detection Request Number**: How many requests before detection

### Performance Metrics
- **Elapsed Time**: Total benchmark duration
- **Throughput**: Successful requests per second
- **Request Times**: Average, P95, P99 latencies
- **Success Rate**: Percentage of successful requests

### Comparison Metrics
- **Time Improvement**: Percentage faster than baseline
- **Throughput Improvement**: Higher request rate achieved
- **Wait Time Reduction**: Backoff time saved

### HMM Performance Metrics
- **Fitting Time**: Time to train the model
- **Memory Usage**: Peak and allocated memory
- **Convergence**: Iterations and final likelihood

## Example Output

```
============================================================
Scenario: Token Bucket
============================================================
Configuration:
  Strategy: token_bucket
  Rate limit: 20 per 10s
  Expected rate: 2.00 req/s

1. Testing SmartSurge (HMM Detection)
----------------------------------------
  Completed: 100/100 successful
  Time: 43.28s
  Throughput: 2.31 req/s
  HMM Detection: 2.35s (request #24)

2. Testing Baseline (Exponential Backoff)
----------------------------------------
  Completed: 100/100 successful
  Time: 48.23s
  Throughput: 2.07 req/s
  Total wait time: 41.00s

3. Improvement Summary
----------------------------------------
  Time reduction: 10.3%
  Throughput increase: 11.6%
  Success rate delta: 0.0%
```

## Using with Pyperf

For statistically rigorous results:

```bash
# Install pyperf
pip install pyperf

# Run benchmark with pyperf
python -m pyperf run \
    --warmups 1 \
    --values 3 \
    -o results.json \
    -m smartsurge.benchmarks.benchmark_hmm_effectiveness \
    -- --mode demo

# Compare results
python -m pyperf compare_to baseline.json improved.json --table
```

## Configuration

### Logging Levels

Control output verbosity:

```bash
# Minimal output
--log-level ERROR

# Default
--log-level WARNING

# Detailed HMM information
--log-level DEBUG
```

### Custom Scenarios

Create custom benchmarks:

```python
from smartsurge.benchmarks import (
    RateLimitConfig,
    RateLimitStrategy,
    HMMEffectivenessBenchmark,
)

# Configure custom rate limit
config = RateLimitConfig(
    strategy=RateLimitStrategy.STRICT,
    requests_per_window=100,
    window_seconds=60,
)

# Run benchmark
benchmark = HMMEffectivenessBenchmark()
result = await benchmark.run_scenario(
    "Custom Test",
    config,
    num_requests=200,
)
```

## Dependencies

Install benchmark dependencies:

```bash
pip install smartsurge[benchmark]
```

Required packages:
- `flask` - Mock server
- `werkzeug` - Server utilities
- `pyperf` - Statistical benchmarking (optional)
- `matplotlib` - Visualization (optional)
- `numpy` - Numerical operations

## Troubleshooting

### Port Conflicts
Default ports start at 8765. Kill conflicting processes or modify base port in code.

### Import Errors
Ensure you're in the project root or have installed the package:
```bash
pip install -e .
```

### Performance Issues
- Reduce request count for faster tests
- Use `--mode demo` for quick runs
- Close other applications during benchmarking

## Best Practices

1. **Warm up** - Run benchmarks multiple times
2. **Isolate** - Close unnecessary applications
3. **Be consistent** - Use same parameters for comparisons
4. **Use pyperf** - For statistically significant results
5. **Test realistically** - Match your actual use case

## Results Interpretation

### When SmartSurge Excels
- Undocumented rate limits
- Complex limiting strategies
- High-throughput requirements
- Cost-sensitive APIs

### When Baseline Suffices
- Clear Retry-After headers
- Low request volumes
- Simple rate limiting
- Infrequent limit encounters