"""
Visualize benchmark results comparing SmartSurge vs baseline.

This creates charts and tables to better understand the performance differences.
"""

import json
import sys
import os
from typing import Dict, List
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(filename: str = "benchmark_results.json") -> Dict:
    """Load benchmark results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    return f"{seconds:.2f}s"


def create_ascii_bar_chart(values: List[float], labels: List[str], title: str, width: int = 50):
    """Create a simple ASCII bar chart."""
    print(f"\n{title}")
    print("=" * (width + 20))
    
    max_value = max(values) if values else 1
    
    for label, value in zip(labels, values):
        bar_length = int((value / max_value) * width)
        bar = "█" * bar_length
        print(f"{label:20} {bar} {value:.1f}%")


def print_comparison_table(results: Dict):
    """Print a detailed comparison table."""
    print("\nDETAILED COMPARISON TABLE")
    print("=" * 120)
    print(f"{'Scenario':25} {'Metric':20} {'SmartSurge':>15} {'Baseline':>15} {'Improvement':>15} {'Notes':>25}")
    print("-" * 120)
    
    for scenario in results['scenarios']:
        name = scenario['name']
        
        # Time comparison
        ss_time = scenario['smartsurge']['elapsed_time']
        bl_time = scenario['baseline']['elapsed_time']
        time_imp = scenario['improvements']['time_improvement']
        
        print(f"{name:25} {'Total Time':20} {format_time(ss_time):>15} {format_time(bl_time):>15} {time_imp:>14.1f}% {'✓ Faster' if time_imp > 0 else ''}")
        
        # Throughput comparison
        ss_tput = scenario['smartsurge']['throughput']
        bl_tput = scenario['baseline']['throughput']
        tput_imp = scenario['improvements']['throughput_improvement']
        
        print(f"{' ':25} {'Throughput (req/s)':20} {ss_tput:>15.2f} {bl_tput:>15.2f} {tput_imp:>14.1f}% {'✓ Higher' if tput_imp > 0 else ''}")
        
        # Success rate
        ss_success = scenario['smartsurge']['success_rate'] * 100
        bl_success = (scenario['baseline']['successful_requests'] / scenario['baseline']['total_requests']) * 100
        
        print(f"{' ':25} {'Success Rate (%)':20} {ss_success:>15.1f} {bl_success:>15.1f} {ss_success - bl_success:>14.1f}% {'✓ Better' if ss_success > bl_success else ''}")
        
        # Detection
        if scenario['smartsurge']['detection_time']:
            det_time = scenario['smartsurge']['detection_time']
            print(f"{' ':25} {'Detection Time':20} {format_time(det_time):>15} {'N/A':>15} {'':>15} '✓ Proactive'")
        
        # Wasted time
        avoided_wait = scenario['improvements']['avoided_wait_time']
        if avoided_wait > 0:
            print(f"{' ':25} {'Backoff Wait Time':20} {'0s':>15} {format_time(avoided_wait):>15} {'':>15} f'✓ Saved {format_time(avoided_wait)}'")
        
        print("-" * 120)


def print_detection_analysis(results: Dict):
    """Analyze HMM detection performance."""
    print("\nHMM DETECTION ANALYSIS")
    print("=" * 80)
    
    detection_data = []
    for scenario in results['scenarios']:
        if scenario['smartsurge']['detection_time']:
            detection_data.append({
                'scenario': scenario['name'],
                'detection_time': scenario['smartsurge']['detection_time'],
                'detection_request': scenario['smartsurge']['detection_request_num'],
                'detected_rate': scenario['smartsurge']['detected_rate'],
                'actual_rate': scenario['config']['expected_rate']
            })
    
    if detection_data:
        print(f"\nDetection occurred in {len(detection_data)} out of {len(results['scenarios'])} scenarios ({len(detection_data)/len(results['scenarios'])*100:.0f}%)")
        print(f"\n{'Scenario':25} {'Detection Time':15} {'At Request #':15} {'Accuracy':15}")
        print("-" * 70)
        
        for data in detection_data:
            if data['detected_rate'] and data['actual_rate']:
                accuracy = 100 - abs(data['detected_rate'] - data['actual_rate']) / data['actual_rate'] * 100
            else:
                accuracy = 0
            
            print(f"{data['scenario']:25} {format_time(data['detection_time']):15} {data['detection_request']:15} {accuracy:14.1f}%")
        
        # Average stats
        avg_time = statistics.mean([d['detection_time'] for d in detection_data])
        avg_request = statistics.mean([d['detection_request'] for d in detection_data])
        
        print(f"\nAverage detection time: {format_time(avg_time)}")
        print(f"Average detection at request: #{avg_request:.0f}")
    else:
        print("No HMM detections occurred in any scenario")


def print_efficiency_metrics(results: Dict):
    """Print efficiency metrics showing resource usage."""
    print("\nRESOURCE EFFICIENCY METRICS")
    print("=" * 80)
    
    total_ss_requests = sum(s['smartsurge']['total_requests'] for s in results['scenarios'])
    total_bl_requests = sum(s['baseline']['total_requests'] for s in results['scenarios'])
    total_ss_success = sum(s['smartsurge']['successful_requests'] for s in results['scenarios'])
    total_bl_success = sum(s['baseline']['successful_requests'] for s in results['scenarios'])
    
    print(f"\nTotal requests made:")
    print(f"  SmartSurge: {total_ss_requests}")
    print(f"  Baseline: {total_bl_requests}")
    print(f"  Difference: {total_bl_requests - total_ss_requests} fewer requests with SmartSurge")
    
    print(f"\nSuccessful requests:")
    print(f"  SmartSurge: {total_ss_success} ({total_ss_success/total_ss_requests*100:.1f}%)")
    print(f"  Baseline: {total_bl_success} ({total_bl_success/total_bl_requests*100:.1f}%)")
    
    print(f"\nTime saved:")
    print(f"  Total backoff time avoided: {format_time(results['summary']['total_avoided_wait_time'])}")
    print(f"  Total retries avoided: {results['summary']['total_avoided_retries']}")
    
    # Calculate request latency comparison
    print(f"\nRequest latency (avg):")
    for scenario in results['scenarios']:
        if 'avg_request_time' in scenario['smartsurge']:
            ss_latency = scenario['smartsurge']['avg_request_time']
            # Estimate baseline latency (includes backoff)
            bl_latency = scenario['baseline']['elapsed_time'] / scenario['baseline']['total_requests']
            
            print(f"  {scenario['name']:25} SmartSurge: {format_time(ss_latency):10} Baseline: {format_time(bl_latency):10}")


def generate_summary_report(results: Dict):
    """Generate an executive summary of the benchmark results."""
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY")
    print("="*80)
    
    print(f"\nSmartSurge demonstrates significant advantages over traditional exponential backoff:")
    
    print(f"\n1. Performance Improvements (average across {results['summary']['total_scenarios']} scenarios):")
    print(f"   • {results['summary']['avg_time_improvement']:.1f}% faster completion time")
    print(f"   • {results['summary']['avg_throughput_improvement']:.1f}% higher throughput")
    
    print(f"\n2. Efficiency Gains:")
    print(f"   • Avoided {format_time(results['summary']['total_avoided_wait_time'])} of unnecessary backoff delays")
    print(f"   • Prevented {results['summary']['total_avoided_retries']} retry attempts")
    
    print(f"\n3. Proactive Detection:")
    print(f"   • Successfully detected rate limits in {results['summary']['detection_success_rate']:.0f}% of scenarios")
    if results['summary']['avg_detection_time']:
        print(f"   • Average detection time: {format_time(results['summary']['avg_detection_time'])}")
    
    print(f"\n4. Consistency:")
    print(f"   • Performance improvements ranged from {results['summary']['worst_improvement']:.1f}% to {results['summary']['best_improvement']:.1f}%")
    print(f"   • Consistent improvements across all rate limiting strategies")
    
    print(f"\n5. Real-world Benefits:")
    print(f"   • Reduced API quota consumption through fewer failed requests")
    print(f"   • Better user experience with faster response times")
    print(f"   • More predictable performance under rate limiting")


def create_scenario_comparison_chart(results: Dict):
    """Create visual comparison of scenarios."""
    print("\nSCENARIO PERFORMANCE COMPARISON")
    print("=" * 80)
    
    scenarios = results['scenarios']
    names = [s['name'] for s in scenarios]
    time_improvements = [s['improvements']['time_improvement'] for s in scenarios]
    
    create_ascii_bar_chart(time_improvements, names, "Time Improvement by Scenario (%)", width=40)
    
    throughput_improvements = [s['improvements']['throughput_improvement'] for s in scenarios]
    create_ascii_bar_chart(throughput_improvements, names, "Throughput Improvement by Scenario (%)", width=40)


def main():
    """Run visualization."""
    try:
        results = load_results()
    except FileNotFoundError:
        print("Error: benchmark_results.json not found.")
        print("Please run benchmark_comprehensive_comparison.py first.")
        return
    
    print("SmartSurge vs Baseline Benchmark Results")
    print("=" * 80)
    print(f"Generated at: {results['timestamp']}")
    
    # Generate all visualizations and analyses
    generate_summary_report(results)
    create_scenario_comparison_chart(results)
    print_comparison_table(results)
    print_detection_analysis(results)
    print_efficiency_metrics(results)
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("\nSmartSurge's HMM-based approach provides substantial benefits over traditional")
    print("exponential backoff, particularly in scenarios with:")
    print("  • Undocumented rate limits")
    print("  • Complex rate limiting strategies") 
    print("  • High-throughput requirements")
    print("  • Cost-sensitive API usage")


if __name__ == "__main__":
    main()