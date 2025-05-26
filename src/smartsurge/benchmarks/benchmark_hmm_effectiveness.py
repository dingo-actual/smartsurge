"""
Comprehensive benchmark suite for HMM rate limit detection effectiveness.

This module provides a complete benchmarking framework for evaluating SmartSurge's
HMM-based rate limit detection against baseline approaches. It includes:

- Statistical benchmarking using pyperf for accurate measurements
- Multiple rate limiting scenarios (fixed, token bucket, dynamic, etc.)
- Comparative analysis against exponential backoff baseline
- Detailed metrics collection and reporting
- Support for both quick demos and comprehensive benchmarks
"""

import asyncio
import json
import time
import statistics
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from multiprocessing import Process
from datetime import datetime

import pyperf
from .baseline_client import collect_baseline_metrics
from .mock_server import (
    get_strict_rate_limit_config, 
    get_token_bucket_config,
    get_adaptive_config,
    get_dynamic_config,
    get_noisy_config,
    get_load_dependent_config,
    create_benchmark_server,
    RateLimitConfig
)
from smartsurge import SmartSurgeClient
from smartsurge.models import SearchStatus


@dataclass
class BenchmarkMetrics:
    """Comprehensive metrics for a benchmark run."""
    total_requests: int
    successful_requests: int
    rate_limited_requests: int
    elapsed_time: float
    throughput: float
    detection_time: Optional[float]
    detection_request_num: Optional[int]
    detected_rate: Optional[float]
    consecutive_runs: List[int]
    avg_consecutive: float
    max_consecutive: int
    avg_request_time: float
    p95_request_time: float
    p99_request_time: float
    success_rate: float
    total_wait_time: float = 0.0
    max_backoff_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


@dataclass
class ScenarioResult:
    """Results from a single benchmark scenario."""
    scenario_name: str
    rate_limit_config: Dict[str, Any]
    smartsurge_metrics: BenchmarkMetrics
    baseline_metrics: Optional[BenchmarkMetrics]
    improvement_metrics: Optional[Dict[str, float]]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class HMMEffectivenessBenchmark:
    """Main benchmark class for HMM effectiveness testing."""
    
    def __init__(self, base_port: int = 8765):
        self.base_port = base_port
        self.current_port = base_port
        
    def get_next_port(self) -> int:
        """Get next available port for server."""
        port = self.current_port
        self.current_port += 1
        return port
        
    async def collect_smartsurge_metrics(
        self,
        client: SmartSurgeClient,
        num_requests: int,
        request_delay: float = 0.01
    ) -> BenchmarkMetrics:
        """Collect metrics for SmartSurge client."""
        successful_requests = 0
        rate_limited_requests = 0
        request_times = []
        detection_time = None
        detection_request_num = None
        detected_rate = None
        consecutive_runs = []
        current_run = 0
        total_wait_time = 0
        
        start_time = time.perf_counter()
        
        for i in range(num_requests):
            request_start = time.perf_counter()
            
            try:
                response, history = client.request(
                    method="GET",
                    endpoint="/api/test",
                    return_history=True
                )
                
                request_time = time.perf_counter() - request_start
                request_times.append(request_time)
                
                if response.status_code == 200:
                    successful_requests += 1
                    current_run += 1
                elif response.status_code == 429:
                    rate_limited_requests += 1
                    if current_run > 0:
                        consecutive_runs.append(current_run)
                        current_run = 0
                
                # Track detection
                if detection_time is None and history.search_status == SearchStatus.COMPLETED:
                    detection_time = time.perf_counter() - start_time
                    detection_request_num = i + 1
                    if history.estimated_rate_limit:
                        detected_rate = history.estimated_rate_limit.max_requests / history.estimated_rate_limit.time_period
                        
            except Exception as e:
                rate_limited_requests += 1
                if current_run > 0:
                    consecutive_runs.append(current_run)
                    current_run = 0
            
            await asyncio.sleep(request_delay)
        
        # Record final run
        if current_run > 0:
            consecutive_runs.append(current_run)
        
        elapsed_time = time.perf_counter() - start_time
        throughput = successful_requests / elapsed_time if elapsed_time > 0 else 0
        
        # Calculate statistics
        avg_request_time = statistics.mean(request_times) if request_times else 0
        p95_request_time = sorted(request_times)[int(len(request_times) * 0.95)] if request_times else 0
        p99_request_time = sorted(request_times)[int(len(request_times) * 0.99)] if request_times else 0
        
        return BenchmarkMetrics(
            total_requests=num_requests,
            successful_requests=successful_requests,
            rate_limited_requests=rate_limited_requests,
            elapsed_time=elapsed_time,
            throughput=throughput,
            detection_time=detection_time,
            detection_request_num=detection_request_num,
            detected_rate=detected_rate,
            consecutive_runs=consecutive_runs,
            avg_consecutive=statistics.mean(consecutive_runs) if consecutive_runs else 0,
            max_consecutive=max(consecutive_runs) if consecutive_runs else 0,
            avg_request_time=avg_request_time,
            p95_request_time=p95_request_time,
            p99_request_time=p99_request_time,
            success_rate=successful_requests / num_requests if num_requests > 0 else 0,
            total_wait_time=total_wait_time
        )
    
    async def collect_baseline_metrics_enhanced(
        self,
        base_url: str,
        num_requests: int,
        request_delay: float = 0.01
    ) -> BenchmarkMetrics:
        """Collect enhanced metrics for baseline client."""
        baseline_metrics = await collect_baseline_metrics(base_url, num_requests, request_delay)
        
        # Convert BaselineMetrics to BenchmarkMetrics
        # Calculate statistics from consecutive_runs
        avg_consecutive = statistics.mean(baseline_metrics.consecutive_runs) if baseline_metrics.consecutive_runs else 0
        max_consecutive = max(baseline_metrics.consecutive_runs) if baseline_metrics.consecutive_runs else 0
        
        # Calculate request time statistics
        request_times = baseline_metrics.request_times
        avg_request_time = statistics.mean(request_times) if request_times else 0
        p95_request_time = sorted(request_times)[int(len(request_times) * 0.95)] if request_times else 0
        p99_request_time = sorted(request_times)[int(len(request_times) * 0.99)] if request_times else 0
        
        return BenchmarkMetrics(
            total_requests=baseline_metrics.total_requests,
            successful_requests=baseline_metrics.successful_requests,
            rate_limited_requests=baseline_metrics.rate_limited_requests,
            elapsed_time=baseline_metrics.elapsed_time,
            throughput=baseline_metrics.throughput,
            detection_time=None,  # Baseline doesn't detect
            detection_request_num=None,
            detected_rate=None,
            consecutive_runs=baseline_metrics.consecutive_runs,
            avg_consecutive=avg_consecutive,
            max_consecutive=max_consecutive,
            avg_request_time=avg_request_time,
            p95_request_time=p95_request_time,
            p99_request_time=p99_request_time,
            success_rate=baseline_metrics.successful_requests / baseline_metrics.total_requests if baseline_metrics.total_requests > 0 else 0,
            total_wait_time=baseline_metrics.total_wait_time,
            max_backoff_time=baseline_metrics.max_backoff_time
        )
    
    async def run_scenario(
        self,
        scenario_name: str,
        config: RateLimitConfig,
        num_requests: int = 100,
        request_delay: float = 0.01,
        compare_baseline: bool = True,
        verbose: bool = True
    ) -> ScenarioResult:
        """Run a single benchmark scenario."""
        server_port = self.get_next_port()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Scenario: {scenario_name}")
            print(f"{'='*60}")
            print(f"Configuration:")
            print(f"  Strategy: {config.strategy.value}")
            print(f"  Rate limit: {config.requests_per_window} per {config.window_seconds}s")
            print(f"  Expected rate: {config.requests_per_window/config.window_seconds:.2f} req/s")
        
        # Start server
        server = create_benchmark_server(config)
        
        def run_server():
            server.start(port=server_port)
        
        server_process = Process(target=run_server)
        server_process.daemon = True
        server_process.start()
        
        # Wait for server startup
        await asyncio.sleep(2)
        
        base_url = f"http://localhost:{server_port}"
        
        try:
            # Test SmartSurge
            if verbose:
                print(f"\n1. Testing SmartSurge (HMM Detection)")
                print("-" * 40)
            
            # Create a logger for the client that respects the configured log level
            client_logger = logging.getLogger("smartsurge.benchmark.client")
            # Ensure the client logger inherits the root logger's level
            client_logger.setLevel(logging.getLogger().getEffectiveLevel())
            smartsurge_client = SmartSurgeClient(base_url, timeout=30, max_retries=2, logger=client_logger)
            smartsurge_metrics = await self.collect_smartsurge_metrics(
                smartsurge_client, num_requests, request_delay
            )
            
            if verbose:
                print(f"  Completed: {smartsurge_metrics.successful_requests}/{num_requests} successful")
                print(f"  Time: {smartsurge_metrics.elapsed_time:.2f}s")
                print(f"  Throughput: {smartsurge_metrics.throughput:.2f} req/s")
                if smartsurge_metrics.detection_time:
                    print(f"  HMM Detection: {smartsurge_metrics.detection_time:.2f}s (request #{smartsurge_metrics.detection_request_num})")
            
            baseline_metrics = None
            improvement_metrics = None
            
            if compare_baseline:
                # Reset server state
                await asyncio.sleep(config.window_seconds + 1)
                
                # Test Baseline
                if verbose:
                    print(f"\n2. Testing Baseline (Exponential Backoff)")
                    print("-" * 40)
                
                baseline_metrics = await self.collect_baseline_metrics_enhanced(
                    base_url, num_requests, request_delay
                )
                
                if verbose:
                    print(f"  Completed: {baseline_metrics.successful_requests}/{num_requests} successful")
                    print(f"  Time: {baseline_metrics.elapsed_time:.2f}s")
                    print(f"  Throughput: {baseline_metrics.throughput:.2f} req/s")
                    print(f"  Total wait time: {baseline_metrics.total_wait_time:.2f}s")
                
                # Calculate improvements
                improvement_metrics = self.calculate_improvements(smartsurge_metrics, baseline_metrics)
                
                if verbose:
                    print(f"\n3. Improvement Summary")
                    print("-" * 40)
                    print(f"  Time reduction: {improvement_metrics['time_improvement']:.1f}%")
                    print(f"  Throughput increase: {improvement_metrics['throughput_improvement']:.1f}%")
                    print(f"  Success rate delta: {improvement_metrics['success_rate_delta']:.1f}%")
            
            return ScenarioResult(
                scenario_name=scenario_name,
                rate_limit_config=asdict(config),
                smartsurge_metrics=smartsurge_metrics,
                baseline_metrics=baseline_metrics,
                improvement_metrics=improvement_metrics
            )
            
        finally:
            server_process.terminate()
            server_process.join(timeout=5)
    
    def calculate_improvements(
        self,
        smartsurge: BenchmarkMetrics,
        baseline: BenchmarkMetrics
    ) -> Dict[str, float]:
        """Calculate improvement metrics."""
        time_improvement = (baseline.elapsed_time - smartsurge.elapsed_time) / baseline.elapsed_time * 100
        throughput_improvement = (smartsurge.throughput - baseline.throughput) / baseline.throughput * 100
        success_rate_delta = (smartsurge.success_rate - baseline.success_rate) * 100
        
        return {
            'time_improvement': time_improvement,
            'throughput_improvement': throughput_improvement,
            'success_rate_delta': success_rate_delta,
            'wait_time_reduction': baseline.total_wait_time - smartsurge.total_wait_time,
            'detection_efficiency': smartsurge.detection_time / smartsurge.elapsed_time * 100 if smartsurge.detection_time else 0
        }
    
    async def run_comprehensive_benchmark(
        self,
        num_requests: int = 100,
        request_delay: float = 0.01
    ) -> List[ScenarioResult]:
        """Run comprehensive benchmark across multiple scenarios."""
        scenarios = [
            ("Strict Fixed Window", get_strict_rate_limit_config()),
            ("Token Bucket", get_token_bucket_config()),
            ("Adaptive Strategy", get_adaptive_config()),
            ("Dynamic Limits", get_dynamic_config()),
            ("Noisy Environment", get_noisy_config()),
            ("Load Dependent", get_load_dependent_config()),
        ]
        
        results = []
        for name, config in scenarios:
            result = await self.run_scenario(name, config, num_requests, request_delay)
            results.append(result)
        
        return results
    
    def run_pyperf_benchmark(
        self,
        runner: pyperf.Runner,
        scenario_name: str,
        config: RateLimitConfig,
        num_requests: int = 50
    ):
        """Run a pyperf benchmark for a specific scenario."""
        async def bench_func():
            result = await self.run_scenario(
                scenario_name,
                config,
                num_requests=num_requests,
                request_delay=0.01,
                compare_baseline=False,
                verbose=False
            )
            return result.smartsurge_metrics.elapsed_time
        
        def sync_bench():
            return asyncio.run(bench_func())
        
        runner.bench_func(f"hmm_detection_{scenario_name.lower().replace(' ', '_')}", sync_bench)


def create_pyperf_suite():
    """Create a pyperf benchmark suite."""
    runner = pyperf.Runner()
    runner.metadata['description'] = 'SmartSurge HMM rate limit detection benchmarks'
    
    benchmark = HMMEffectivenessBenchmark()
    
    # Add various scenarios
    scenarios = [
        ("strict_fixed", get_strict_rate_limit_config()),
        ("token_bucket", get_token_bucket_config()),
        ("adaptive", get_adaptive_config()),
        ("noisy", get_noisy_config()),
    ]
    
    for name, config in scenarios:
        benchmark.run_pyperf_benchmark(runner, name, config)
    
    return runner


async def main():
    """Main entry point for standalone execution."""
    import argparse
    import logging
    
    parser = argparse.ArgumentParser(description='HMM Rate Limit Detection Effectiveness Benchmark')
    parser.add_argument('--mode', choices=['demo', 'comprehensive', 'pyperf'], 
                       default='demo', help='Benchmark mode')
    parser.add_argument('--requests', type=int, default=100, 
                       help='Number of requests per scenario')
    parser.add_argument('--delay', type=float, default=0.01,
                       help='Delay between requests in seconds')
    parser.add_argument('--output', help='Output file for results (JSON)')
    parser.add_argument('--log-level', '-l',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       default='WARNING',
                       help='Set the logging level (default: WARNING)')
    args = parser.parse_args()
    
    # Configure logging based on the provided level
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    benchmark = HMMEffectivenessBenchmark()
    
    if args.mode == 'pyperf':
        runner = create_pyperf_suite()
        runner.parse_args()
    else:
        if args.mode == 'demo':
            # Run single scenario for demo
            result = await benchmark.run_scenario(
                "Demo - Token Bucket",
                get_token_bucket_config(),
                num_requests=args.requests,
                request_delay=args.delay
            )
            results = [result]
        else:
            # Run comprehensive benchmark
            results = await benchmark.run_comprehensive_benchmark(
                num_requests=args.requests,
                request_delay=args.delay
            )
        
        # Save results if requested
        if args.output:
            output_data = {
                'timestamp': datetime.now().isoformat(),
                'mode': args.mode,
                'parameters': {
                    'num_requests': args.requests,
                    'request_delay': args.delay
                },
                'results': [
                    {
                        'scenario': r.scenario_name,
                        'config': r.rate_limit_config,
                        'smartsurge': r.smartsurge_metrics.to_dict(),
                        'baseline': r.baseline_metrics.to_dict() if r.baseline_metrics else None,
                        'improvements': r.improvement_metrics
                    }
                    for r in results
                ]
            }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    asyncio.run(main())