# SmartSurge Benchmarking Guide

This directory contains comprehensive benchmarking tools for evaluating SmartSurge's HMM-based rate limit detection against traditional exponential backoff approaches.

## Overview

The benchmark suite provides:
- **Mock Server**: Configurable rate limiting server with multiple strategies
- **Baseline Client**: Reference implementation using exponential backoff
- **HMM Effectiveness Benchmarks**: Comprehensive suite comparing HMM detection vs baseline
- **HMM Performance Benchmarks**: Algorithm performance metrics (fitting time, memory usage)
- **Benchmark Runner**: Orchestrator for running the complete suite

## Quick Start

### 1. Run Complete Suite

The easiest way to run benchmarks is using the suite runner:

```bash
# Run with interactive menu
python -m smartsurge.benchmarks.run_benchmark_suite

# Or with command-line options
python -m smartsurge.benchmarks.run_benchmark_suite --mode demo
python -m smartsurge.benchmarks.run_benchmark_suite --mode comprehensive
python -m smartsurge.benchmarks.run_benchmark_suite --mode full
```

### 2. Run Specific Benchmarks

#### HMM Effectiveness Benchmark

Compare SmartSurge against baseline exponential backoff:

```bash
# Demo mode - single scenario
python -m smartsurge.benchmarks.benchmark_hmm_effectiveness --mode demo

# Comprehensive - all scenarios
python -m smartsurge.benchmarks.benchmark_hmm_effectiveness --mode comprehensive

# Custom parameters
python -m smartsurge.benchmarks.benchmark_hmm_effectiveness \
    --mode demo \
    --requests 200 \
    --delay 0.05 \
    --log-level DEBUG \
    --output results.json
```

#### HMM Performance Benchmark

Measure HMM fitting performance:

```bash
# Run performance benchmarks
python -m smartsurge.benchmarks.benchmark_hmm_performance

# Use the shell script helper
./src/smartsurge/benchmarks/run_hmm_benchmark.sh
```

## Rate Limiting Strategies

The mock server supports six different rate limiting strategies:

### 1. Strict Fixed Window
- Fixed rate limit with hard cutoffs
- Best for testing basic detection capabilities
- Configuration: `get_strict_rate_limit_config()`

### 2. Token Bucket
- Allows bursts up to bucket capacity
- Refills at a constant rate
- Tests burst handling and recovery
- Configuration: `get_token_bucket_config()`

### 3. Adaptive
- Dynamically adjusts limits based on request patterns
- Simulates APIs that adapt to client behavior
- Configuration: `get_adaptive_config()`

### 4. Dynamic
- Time-based rate limit changes
- Tests adaptation to changing conditions
- Configuration: `get_dynamic_config()`

### 5. Noisy
- Includes random delays and false positives
- Simulates real-world network conditions
- Configuration: `get_noisy_config()`

### 6. Load-Dependent
- Smoothly transitions limits based on simulated load
- Uses sinusoidal patterns with noise
- Configuration: `get_load_dependent_config()`

## Benchmark Metrics

### Core Metrics Collected

1. **Detection Metrics**
   - Time to first detection
   - Detection accuracy (error vs actual rate)
   - Number of requests before detection

2. **Performance Metrics**
   - Total elapsed time
   - Request throughput (successful req/s)
   - Average, P95, and P99 request times

3. **Behavioral Metrics**
   - Consecutive successful runs
   - Total wait time (backoff)
   - Success rate

4. **Improvement Metrics** (vs baseline)
   - Time reduction percentage
   - Throughput improvement
   - Wait time reduction

### HMM Performance Metrics

- Fitting time (seconds)
- Peak memory usage (MB)
- Memory allocated (MB)
- Number of iterations to convergence
- Final log-likelihood
- Convergence status

## Using with Pyperf

For statistically rigorous benchmarking:

```bash
# Install pyperf first
pip install pyperf

# Run with pyperf
python -m pyperf run \
    --warmups 1 \
    --values 3 \
    -o results.json \
    -m smartsurge.benchmarks.benchmark_hmm_effectiveness \
    -- --mode demo

# Compare results
python -m pyperf compare_to baseline.json improved.json --table

# Get statistics
python -m pyperf stats results.json
```

## Interpreting Results

### Benchmark Output Example

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

### Key Performance Indicators

1. **Detection Speed**: How quickly SmartSurge identifies the rate limit
2. **Accuracy**: How close the detected rate is to the actual rate
3. **Efficiency**: Less total time and fewer 429 responses
4. **Adaptability**: Performance across different rate limiting strategies

## Logging Configuration

Control log verbosity for debugging:

```bash
# Minimal output
python -m smartsurge.benchmarks.run_benchmark_suite --log-level ERROR

# Standard output
python -m smartsurge.benchmarks.run_benchmark_suite --log-level WARNING

# Detailed HMM information
python -m smartsurge.benchmarks.run_benchmark_suite --log-level DEBUG
```

## Custom Benchmarks

Create custom benchmark scenarios:

```python
from smartsurge.benchmarks import (
    RateLimitConfig,
    RateLimitStrategy,
    create_benchmark_server,
)
from smartsurge.benchmarks.benchmark_hmm_effectiveness import HMMEffectivenessBenchmark

# Custom rate limit configuration
config = RateLimitConfig(
    strategy=RateLimitStrategy.STRICT,
    requests_per_window=50,
    window_seconds=60,
    burst_size=10,
)

# Run benchmark
benchmark = HMMEffectivenessBenchmark()
result = await benchmark.run_scenario(
    scenario_name="Custom Test",
    config=config,
    num_requests=200,
    request_delay=0.05,
)
```

## Architecture

### Components

1. **Mock Server** (`mock_server.py`)
   - Flask-based server with configurable rate limiting
   - Supports multiple strategies via strategy pattern
   - Thread-safe request tracking

2. **Baseline Client** (`baseline_client.py`)
   - Async HTTP client with exponential backoff
   - Respects Retry-After headers
   - Collects same metrics as SmartSurge for comparison

3. **Benchmark Framework** (`benchmark_hmm_effectiveness.py`)
   - Orchestrates test scenarios
   - Manages server lifecycle
   - Collects and compares metrics

4. **Performance Benchmarks** (`benchmark_hmm_performance.py`)
   - Measures HMM fitting performance
   - Memory profiling with tracemalloc
   - Multiple scenario testing

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   - Default ports: 8765+ for effectiveness, 6200 for performance
   - Solution: Kill existing processes or change base port

2. **Import Errors**
   - Ensure benchmark dependencies are installed: `pip install smartsurge[benchmark]`
   - Run from project root or install package

3. **Slow Performance**
   - Reduce number of requests for quick tests
   - Use `--mode demo` for faster runs
   - Close other applications during benchmarking

4. **Memory Issues**
   - Reduce `max_observations` in RequestHistory
   - Use fewer concurrent requests
   - Monitor system resources during runs

## Best Practices

1. **Consistency**: Use same parameters when comparing results
2. **Isolation**: Close unnecessary applications during benchmarks
3. **Warm-up**: Run benchmarks multiple times, discard first results
4. **Statistical Significance**: Use pyperf for reliable measurements
5. **Realistic Scenarios**: Test with parameters matching your use case

## Future Enhancements

Planned improvements:
- WebSocket rate limit testing
- Distributed client testing
- Real API testing mode
- Performance regression detection
- CI/CD integration templates