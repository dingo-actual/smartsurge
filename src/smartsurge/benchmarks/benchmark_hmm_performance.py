"""
Benchmark HMM fitting performance (time and memory) under various rate-limiting scenarios.

This module measures:
1. Time to fit HMM model
2. Memory usage during fitting
3. Model quality metrics
"""

import asyncio
import gc
import psutil
import threading
import time
import tracemalloc
from dataclasses import dataclass
from typing import List

import pyperf

from smartsurge import SmartSurgeClient
from smartsurge.models import RequestEntry, RequestMethod
from smartsurge.hmm import HMM, HMMParams

from .mock_server import (
    RateLimitConfig,
    create_benchmark_server,
    get_adaptive_config,
    get_dynamic_config,
    get_noisy_config,
    get_strict_rate_limit_config,
    get_load_dependent_config,
    get_token_bucket_config,
)


@dataclass
class HMMFittingMetrics:
    """Metrics from HMM fitting process."""
    fitting_time: float  # seconds
    peak_memory: float  # MB
    memory_allocated: float  # MB
    num_observations: int
    num_iterations: int
    final_log_likelihood: float
    converged: bool


class HMMFittingBenchmark:
    """Benchmark HMM fitting under various scenarios."""
    
    def __init__(self, config: RateLimitConfig, server_port: int = 6200):
        self.config = config
        self.server_port = server_port
        self.base_url = f"http://127.0.0.1:{server_port}"
    
    async def collect_request_data(self, num_requests: int = 100) -> List[RequestEntry]:
        """Collect request data by making actual requests to the mock server."""
        # Start server
        server = create_benchmark_server(self.config)
        server_thread = threading.Thread(
            target=lambda: server.start(port=self.server_port)
        )
        server_thread.daemon = True
        server_thread.start()
        
        # Wait for server
        await asyncio.sleep(0.5)
        
        try:
            client = SmartSurgeClient(
                base_url=self.base_url,
                timeout=30,
                max_retries=0  # No retries to get raw data
            )
            
            entries = []
            
            for i in range(num_requests):
                try:
                    response, history = await client.async_get("/api/test")
                    
                    # Extract the last entry
                    if history.entries:
                        entries.append(history.entries[-1])
                    
                    # Small delay to create realistic patterns
                    await asyncio.sleep(0.05)
                    
                except Exception:
                    # Still try to get the entry from failed requests
                    if hasattr(client, 'histories'):
                        key = ("/api/test", RequestMethod.GET)
                        if key in client.histories:
                            hist = client.histories[key]
                            if hist.entries:
                                entries.append(hist.entries[-1])
            
            return entries
            
        finally:
            server.stop()
    
    def measure_hmm_fitting(self, entries: List[RequestEntry]) -> HMMFittingMetrics:
        """Measure time and memory for fitting HMM to request entries."""
        # Prepare observations for HMM
        observations = []
        for i, entry in enumerate(entries):
            if i == 0:
                continue
                
            prev_entry = entries[i-1]
            
            # Create observation tuple
            obs = (
                1 if entry.success else 0,  # Current success
                1 if prev_entry.success else 0,  # Previous success
                entry.response_time,  # Response time
                entry.timestamp.timestamp() - prev_entry.timestamp.timestamp()  # Time delta
            )
            observations.append(obs)
        
        # Initialize HMM
        hmm = HMM(
            params=HMMParams(
                n_states=3
            )
        )
        
        # Measure memory before fitting
        gc.collect()
        tracemalloc.start()
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Measure fitting time
        start_time = time.perf_counter()
        
        try:
            # Fit the model
            log_likelihood = hmm.fit(observations)
            converged = hmm.converged
            num_iterations = hmm.n_iter_
            
            fitting_time = time.perf_counter() - start_time
            
            # Measure memory after fitting
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            peak_memory_mb = peak / 1024 / 1024
            allocated_memory_mb = (memory_after - memory_before)
            
            return HMMFittingMetrics(
                fitting_time=fitting_time,
                peak_memory=peak_memory_mb,
                memory_allocated=allocated_memory_mb,
                num_observations=len(observations),
                num_iterations=num_iterations,
                final_log_likelihood=log_likelihood,
                converged=converged
            )
            
        except Exception as e:
            tracemalloc.stop()
            print(f"Error fitting HMM: {e}")
            return HMMFittingMetrics(
                fitting_time=0,
                peak_memory=0,
                memory_allocated=0,
                num_observations=len(observations),
                num_iterations=0,
                final_log_likelihood=float('-inf'),
                converged=False
            )


# Pyperf benchmark functions

def bench_hmm_fitting_time(loops: int, config: RateLimitConfig, server_port: int) -> float:
    """Benchmark HMM fitting time."""
    benchmark = HMMFittingBenchmark(config, server_port)
    
    # Collect data once (reuse for all loops)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    entries = loop.run_until_complete(benchmark.collect_request_data(100))
    loop.close()
    
    if not entries:
        return float('inf')
    
    total_time = 0
    
    for _ in range(loops):
        metrics = benchmark.measure_hmm_fitting(entries)
        total_time += metrics.fitting_time
    
    return total_time


def bench_hmm_memory_usage(loops: int, config: RateLimitConfig, server_port: int) -> float:
    """Benchmark HMM memory usage."""
    benchmark = HMMFittingBenchmark(config, server_port)
    
    # Collect data once
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    entries = loop.run_until_complete(benchmark.collect_request_data(100))
    loop.close()
    
    if not entries:
        return 0.0
    
    total_memory = 0
    
    for _ in range(loops):
        gc.collect()  # Clean before each measurement
        metrics = benchmark.measure_hmm_fitting(entries)
        total_memory += metrics.peak_memory
    
    return total_memory / loops  # Return average memory usage


def bench_hmm_fitting_quality(loops: int, config: RateLimitConfig, server_port: int) -> float:
    """Benchmark HMM fitting quality (convergence rate and likelihood)."""
    benchmark = HMMFittingBenchmark(config, server_port)
    
    # Collect data once
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    entries = loop.run_until_complete(benchmark.collect_request_data(100))
    loop.close()
    
    if not entries:
        return float('inf')
    
    total_quality_score = 0
    
    for _ in range(loops):
        metrics = benchmark.measure_hmm_fitting(entries)
        
        # Quality score: lower is better
        # Combines convergence speed and final likelihood
        if metrics.converged and metrics.final_log_likelihood > -float('inf'):
            quality_score = metrics.num_iterations / abs(metrics.final_log_likelihood)
        else:
            quality_score = float('inf')
        
        total_quality_score += quality_score
    
    return total_quality_score


# Standalone benchmark runner

async def run_hmm_fitting_analysis():
    """Run detailed HMM fitting analysis for all scenarios."""
    configs = {
        'strict': get_strict_rate_limit_config(),
        'token_bucket': get_token_bucket_config(),
        'noisy': get_noisy_config(),
        'adaptive': get_adaptive_config(),
        'dynamic': get_dynamic_config(),
        'load_dependent': get_load_dependent_config(),
    }
    
    results = {}
    port = 6300
    
    print("HMM Fitting Performance Analysis")
    print("=" * 60)
    
    for name, config in configs.items():
        print(f"\nTesting {name} configuration...")
        
        benchmark = HMMFittingBenchmark(config, port)
        port += 1
        
        # Collect varying amounts of data
        data_sizes = [50, 100, 200]
        scenario_results = []
        
        for size in data_sizes:
            print(f"  Collecting {size} requests...")
            entries = await benchmark.collect_request_data(size)
            
            if entries:
                print(f"  Fitting HMM...")
                metrics = benchmark.measure_hmm_fitting(entries)
                
                scenario_results.append({
                    'data_size': size,
                    'metrics': metrics
                })
                
                print(f"    Time: {metrics.fitting_time:.4f}s")
                print(f"    Memory: {metrics.peak_memory:.2f}MB")
                print(f"    Iterations: {metrics.num_iterations}")
                print(f"    Converged: {metrics.converged}")
                print(f"    Log-likelihood: {metrics.final_log_likelihood:.2f}")
        
        results[name] = scenario_results
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, scenario_results in results.items():
        print(f"\n{name.upper()}:")
        
        avg_time = sum(r['metrics'].fitting_time for r in scenario_results) / len(scenario_results)
        avg_memory = sum(r['metrics'].peak_memory for r in scenario_results) / len(scenario_results)
        convergence_rate = sum(1 for r in scenario_results if r['metrics'].converged) / len(scenario_results)
        
        print(f"  Average fitting time: {avg_time:.4f}s")
        print(f"  Average memory usage: {avg_memory:.2f}MB")
        print(f"  Convergence rate: {convergence_rate*100:.0f}%")
    
    return results


# Pyperf integration

def add_hmm_fitting_benchmarks(runner: pyperf.Runner):
    """Add HMM fitting benchmarks to pyperf runner."""
    configs = {
        'strict': get_strict_rate_limit_config(),
        'token_bucket': get_token_bucket_config(),
        'noisy': get_noisy_config(),
        'adaptive': get_adaptive_config(),
        'dynamic': get_dynamic_config(),
    }
    
    port = 6400
    
    for config_name, config in configs.items():
        # Time benchmark
        runner.bench_func(
            f'hmm_fitting_time_{config_name}',
            bench_hmm_fitting_time,
            config,
            port
        )
        port += 1
        
        # Memory benchmark
        runner.bench_func(
            f'hmm_memory_usage_{config_name}',
            bench_hmm_memory_usage,
            config,
            port
        )
        port += 1
        
        # Quality benchmark
        runner.bench_func(
            f'hmm_fitting_quality_{config_name}',
            bench_hmm_fitting_quality,
            config,
            port
        )
        port += 1


def main():
    """Main entry point for standalone execution."""
    import sys
    
    if '--pyperf' in sys.argv:
        # Run as pyperf benchmark
        runner = pyperf.Runner()
        runner.metadata['description'] = 'HMM fitting performance benchmarks'
        add_hmm_fitting_benchmarks(runner)
    else:
        # Run detailed analysis
        import asyncio
        asyncio.run(run_hmm_fitting_analysis())


if __name__ == '__main__':
    import threading
    main()