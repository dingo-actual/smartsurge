#!/bin/bash
#
# Run HMM fitting benchmarks for SmartSurge
#

echo "SmartSurge HMM Fitting Benchmarks"
echo "================================="
echo

# Check if psutil is installed (required for memory measurements)
python -c "import psutil" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: psutil not installed. Memory measurements will be unavailable."
    echo "Install with: pip install psutil"
    echo
fi

# Option 1: Quick test
echo "1. Running quick HMM performance test..."
python -m smartsurge.benchmarks.benchmark_hmm_performance --quick
echo

# Option 2: Detailed analysis
echo "2. Running detailed HMM performance analysis..."
python -m smartsurge.benchmarks.benchmark_hmm_performance
echo

# Option 3: Pyperf benchmarks
if command -v pyperf &> /dev/null; then
    echo "3. Running pyperf HMM benchmarks..."
    python -m pyperf run --fast -m smartsurge.benchmarks.benchmark_hmm_performance -- --pyperf -o hmm_performance_results.json
    
    echo
    echo "Results saved to hmm_performance_results.json"
    echo "View with: python -m pyperf stats hmm_performance_results.json"
else
    echo "3. Skipping pyperf benchmarks (pyperf not installed)"
    echo "   Install with: pip install pyperf"
fi

echo
echo "Benchmark complete!"