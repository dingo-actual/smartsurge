"""
Run the complete benchmark suite comparing SmartSurge vs baseline.

This script runs all benchmarks and generates a comprehensive report.
"""

import asyncio
import subprocess
import sys
import os
import json
import time
import argparse
import logging

# Check if benchmark dependencies are available
try:
    import pyperf
    import matplotlib
    import numpy
    import flask
    import werkzeug
    BENCHMARK_DEPS_AVAILABLE = True
except ImportError as e:
    BENCHMARK_DEPS_AVAILABLE = False
    MISSING_DEPS = str(e)

# No longer need sys.path manipulation when running as a module


def run_comprehensive_benchmark(log_level=None):
    """Run the comprehensive comparison benchmark."""
    print("=" * 80)
    print("Running Comprehensive SmartSurge vs Baseline Benchmark")
    print("=" * 80)
    print("\nThis will test both approaches under various scenarios...")
    print("Please wait, this may take a few minutes.\n")
    
    # Run the comprehensive benchmark
    cmd = [
        sys.executable, "-m",
        "smartsurge.benchmarks.benchmark_hmm_effectiveness",
        "--mode", "comprehensive"
    ]
    if log_level:
        cmd.extend(["--log-level", log_level])
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running comprehensive benchmark: {e}")
        return False
    
    return True


def run_pyperf_benchmarks(log_level=None):
    """Run pyperf benchmarks for statistical rigor."""
    print("\n" + "=" * 80)
    print("Running Pyperf Statistical Benchmarks")
    print("=" * 80)
    print("\nTesting different rate limiting strategies with statistical analysis...\n")
    
    strategies = ["strict", "token_bucket", "adaptive", "dynamic", "noisy", "load_dependent"]
    results_files = []
    
    for strategy in strategies:
        print(f"\nBenchmarking {strategy} strategy...")
        output_file = f"pyperf_results_{strategy}.json"
        
        cmd = [
            sys.executable, "-m", "pyperf", "run",
            "--quiet",
            "--warmups", "1",
            "--values", "3",
            "-o", output_file,
            "-m", "smartsurge.benchmarks.benchmark_hmm_effectiveness",
            "--",
            "--benchmark", "comprehensive",
            "--strategy", strategy,
            "--include-baseline"
        ]
        if log_level:
            cmd.extend(["--log-level", log_level])
        
        try:
            subprocess.run(cmd, check=True)
            results_files.append(output_file)
            print(f"✓ Completed {strategy} benchmark")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error running {strategy} benchmark: {e}")
    
    # Compare results
    if len(results_files) > 1:
        print("\n" + "=" * 80)
        print("Pyperf Comparison Results")
        print("=" * 80)
        
        try:
            subprocess.run([
                sys.executable, "-m", "pyperf", "compare_to",
                results_files[0], *results_files[1:],
                "--table"
            ])
        except subprocess.CalledProcessError:
            print("Could not generate comparison table")
    
    return results_files


def visualize_results():
    """Visualize the benchmark results."""
    print("\n" + "=" * 80)
    print("Generating Benchmark Visualizations")
    print("=" * 80)
    
    try:
        subprocess.run([
            sys.executable, "-m",
            "smartsurge.benchmarks.visualize_benchmark_results"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error visualizing results: {e}")
        return False
    
    return True


def run_quick_demo(log_level=None):
    """Run a quick demonstration of the baseline comparison."""
    print("\n" + "=" * 80)
    print("Quick Demonstration: SmartSurge vs Baseline")
    print("=" * 80)
    
    cmd = [
        sys.executable, "-m",
        "smartsurge.benchmarks.benchmark_hmm_effectiveness",
        "--mode", "demo"
    ]
    if log_level:
        cmd.extend(["--log-level", log_level])
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running demo: {e}")
        return False
    
    return True


def generate_summary_report():
    """Generate a final summary report."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUITE SUMMARY")
    print("=" * 80)
    
    # Check if results exist
    if os.path.exists("benchmark_results.json"):
        with open("benchmark_results.json", "r") as f:
            results = json.load(f)
        
        print("\nKey Findings:")
        print(f"• Average performance improvement: {results['summary']['avg_time_improvement']:.1f}%")
        print(f"• Average throughput improvement: {results['summary']['avg_throughput_improvement']:.1f}%")
        print(f"• Total backoff time avoided: {results['summary']['total_avoided_wait_time']:.1f}s")
        print(f"• HMM detection success rate: {results['summary']['detection_success_rate']:.0f}%")
        
        print("\nScenario Performance:")
        for scenario in results['scenarios']:
            print(f"• {scenario['name']}: {scenario['improvements']['time_improvement']:.1f}% faster")
    
    print("\nFiles Generated:")
    files = [
        "benchmark_results.json",
        "pyperf_results_strict.json",
        "pyperf_results_token_bucket.json",
        "pyperf_results_adaptive.json",
        "pyperf_results_dynamic.json",
        "pyperf_results_noisy.json",
        "pyperf_results_load_dependent.json",
    ]
    
    for file in files:
        if os.path.exists(file):
            print(f"• {file}")
    
    print("\nRecommendations:")
    print("• SmartSurge provides significant advantages for APIs with:")
    print("  - Undocumented rate limits")
    print("  - Complex rate limiting strategies")
    print("  - High-throughput requirements")
    print("• Traditional exponential backoff may suffice for:")
    print("  - Simple APIs with clear Retry-After headers")
    print("  - Low-volume applications")


def main():
    """Run the complete benchmark suite."""
    # Check if benchmark dependencies are available
    if not BENCHMARK_DEPS_AVAILABLE:
        print("Error: Benchmark dependencies are not installed.")
        print(f"Missing dependency: {MISSING_DEPS}")
        print("\nTo install benchmark dependencies, run:")
        print("  pip install smartsurge[benchmark]")
        print("\nOr if installing from source:")
        print("  pip install -e '.[benchmark]'")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(
        description="SmartSurge Benchmark Suite - Compare HMM-based approach against traditional exponential backoff",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick demo
  python benchmarks/run_benchmark_suite.py --mode demo
  
  # Run comprehensive benchmark
  python benchmarks/run_benchmark_suite.py --mode comprehensive
  
  # Run full suite with pyperf
  python benchmarks/run_benchmark_suite.py --mode full
  
  # Visualize existing results only
  python benchmarks/run_benchmark_suite.py --mode visualize
  
  # Run specific benchmarks
  python benchmarks/run_benchmark_suite.py --comprehensive --pyperf
  
  # Run with debug logging to see HMM details
  python benchmarks/run_benchmark_suite.py --mode demo --log-level DEBUG
  
  # Run with minimal logging
  python benchmarks/run_benchmark_suite.py --mode full --log-level ERROR
        """
    )
    
    # Add mutually exclusive group for mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--mode', '-m',
        choices=['demo', 'comprehensive', 'full', 'visualize'],
        help='Benchmark mode to run'
    )
    
    # Add individual options
    mode_group.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Run quick demo (fastest, ~30 seconds)'
    )
    mode_group.add_argument(
        '--comprehensive', '-c',
        action='store_true',
        help='Run comprehensive benchmark (detailed, ~2 minutes)'
    )
    mode_group.add_argument(
        '--full', '-f',
        action='store_true',
        help='Run full suite with pyperf (most rigorous, ~5 minutes)'
    )
    mode_group.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Visualize existing results only'
    )
    
    # Additional options
    parser.add_argument(
        '--pyperf', '-p',
        action='store_true',
        help='Include pyperf benchmarks (can be combined with --comprehensive)'
    )
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Skip visualization step'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output verbosity'
    )
    parser.add_argument(
        '--log-level', '-l',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='WARNING',
        help='Set the logging level for SmartSurge components (default: WARNING)'
    )
    
    args = parser.parse_args()
    
    # Determine mode from arguments
    if args.mode:
        mode = args.mode
    elif args.demo:
        mode = 'demo'
    elif args.comprehensive:
        mode = 'comprehensive'
    elif args.full:
        mode = 'full'
    elif args.visualize:
        mode = 'visualize'
    else:
        # Default to interactive mode if no arguments provided
        print("SmartSurge Benchmark Suite")
        print("=" * 80)
        print("This suite compares SmartSurge's HMM-based approach against")
        print("traditional exponential backoff across various scenarios.")
        print("=" * 80)
        
        print("\nSelect benchmark mode:")
        print("1. Quick demo (fastest, ~30 seconds)")
        print("2. Comprehensive benchmark (detailed, ~2 minutes)")
        print("3. Full suite with pyperf (most rigorous, ~5 minutes)")
        print("4. Visualize existing results")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        mode_map = {
            '1': 'demo',
            '2': 'comprehensive',
            '3': 'full',
            '4': 'visualize'
        }
        
        mode = mode_map.get(choice)
        if not mode:
            print("Invalid choice")
            return
        
        # Set default log level for interactive mode if not already set
        if not hasattr(args, 'log_level'):
            args.log_level = 'WARNING'
    
    # Print header unless quiet mode
    if not args.quiet:
        print("SmartSurge Benchmark Suite")
        print("=" * 80)
        print("This suite compares SmartSurge's HMM-based approach against")
        print("traditional exponential backoff across various scenarios.")
        print("=" * 80)
    
    start_time = time.time()
    
    # Execute based on mode
    if mode == 'demo':
        run_quick_demo(args.log_level)
    elif mode == 'comprehensive':
        if run_comprehensive_benchmark(args.log_level):
            if not args.no_visualize:
                visualize_results()
        # Add pyperf if requested
        if args.pyperf:
            run_pyperf_benchmarks(args.log_level)
    elif mode == 'full':
        if run_comprehensive_benchmark(args.log_level):
            run_pyperf_benchmarks(args.log_level)
            if not args.no_visualize:
                visualize_results()
        generate_summary_report()
    elif mode == 'visualize':
        visualize_results()
    
    elapsed = time.time() - start_time
    if not args.quiet:
        print(f"\nBenchmark suite completed in {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()