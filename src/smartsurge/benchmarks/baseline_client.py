"""
Baseline implementation using requests library with exponential backoff.

This provides a comparison point for SmartSurge benchmarks, showing
how a traditional approach handles rate limiting.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import aiohttp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class BaselineMetrics:
    """Metrics collected from baseline client runs."""
    total_requests: int
    successful_requests: int
    rate_limited_requests: int
    total_retries: int
    elapsed_time: float
    throughput: float
    total_wait_time: float  # Time spent in backoff
    max_backoff_time: float
    consecutive_runs: List[int]
    request_times: List[float]  # Time for each request


class BaselineRequestsClient:
    """Standard requests library with exponential backoff."""
    
    def __init__(
        self,
        base_url: str,
        max_retries: int = 3,
        backoff_factor: float = 0.3,
        status_forcelist: Optional[List[int]] = None
    ):
        self.base_url = base_url
        self.session = requests.Session()
        
        # Configure retry strategy
        if status_forcelist is None:
            status_forcelist = [429, 500, 502, 503, 504]
            
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=status_forcelist,
            backoff_factor=backoff_factor,
            allowed_methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "TRACE"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Metrics tracking
        self.total_wait_time = 0
        self.max_backoff_time = 0
        self.total_retries = 0
        
    def get(self, endpoint: str, **kwargs) -> requests.Response:
        """Make a GET request with exponential backoff."""
        url = f"{self.base_url}{endpoint}"
        
        # Track retries manually for metrics
        retries = 0
        backoff = 0.3  # Initial backoff
        
        while retries <= 3:
            try:
                response = self.session.get(url, **kwargs)
                
                if response.status_code == 429:
                    self.total_retries += 1
                    
                    # Check for Retry-After header
                    retry_after = response.headers.get('Retry-After')
                    if retry_after:
                        wait_time = float(retry_after)
                    else:
                        # Exponential backoff
                        wait_time = backoff * (2 ** retries)
                    
                    self.total_wait_time += wait_time
                    self.max_backoff_time = max(self.max_backoff_time, wait_time)
                    
                    time.sleep(wait_time)
                    retries += 1
                    continue
                    
                return response
                
            except requests.exceptions.RequestException as e:
                if retries >= 3:
                    raise
                retries += 1
                wait_time = backoff * (2 ** retries)
                time.sleep(wait_time)
        
        # Should not reach here
        raise requests.exceptions.RetryError("Max retries exceeded")
    
    def get_metrics(self) -> Dict[str, float]:
        """Return metrics for baseline client."""
        return {
            'total_retries': self.total_retries,
            'total_wait_time': self.total_wait_time,
            'max_backoff_time': self.max_backoff_time
        }


class BaselineAsyncClient:
    """Async baseline client using aiohttp with exponential backoff."""
    
    def __init__(
        self,
        base_url: str,
        max_retries: int = 3,
        backoff_factor: float = 0.3
    ):
        self.base_url = base_url
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.session = None
        
        # Metrics
        self.total_wait_time = 0
        self.max_backoff_time = 0
        self.total_retries = 0
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def get(self, endpoint: str, **kwargs) -> aiohttp.ClientResponse:
        """Make an async GET request with exponential backoff."""
        url = f"{self.base_url}{endpoint}"
        
        retries = 0
        backoff = self.backoff_factor
        
        while retries <= self.max_retries:
            try:
                response = await self.session.get(url, **kwargs)
                
                if response.status == 429:
                    self.total_retries += 1
                    
                    # Check for Retry-After header
                    retry_after = response.headers.get('Retry-After')
                    if retry_after:
                        wait_time = float(retry_after)
                    else:
                        # Exponential backoff
                        wait_time = backoff * (2 ** retries)
                    
                    self.total_wait_time += wait_time
                    self.max_backoff_time = max(self.max_backoff_time, wait_time)
                    
                    await asyncio.sleep(wait_time)
                    retries += 1
                    continue
                    
                return response
                
            except aiohttp.ClientError as e:
                if retries >= self.max_retries:
                    raise
                retries += 1
                wait_time = backoff * (2 ** retries)
                await asyncio.sleep(wait_time)
        
        raise aiohttp.ClientError("Max retries exceeded")


async def collect_baseline_metrics(
    base_url: str,
    num_requests: int = 100,
    request_delay: float = 0.05
) -> BaselineMetrics:
    """Collect metrics using baseline client with exponential backoff."""
    
    async with BaselineAsyncClient(base_url) as client:
        start_time = time.perf_counter()
        
        # Track metrics
        total_requests = 0
        successful_requests = 0
        rate_limited_requests = 0
        consecutive_runs = []
        current_run = 0
        request_times = []
        
        for i in range(num_requests):
            request_start = time.perf_counter()
            
            try:
                response = await client.get("/api/test")
                total_requests += 1
                
                if response.status == 429:
                    rate_limited_requests += 1
                    current_run += 1
                else:
                    successful_requests += 1
                    if current_run > 0:
                        consecutive_runs.append(current_run)
                        current_run = 0
                
                request_times.append(time.perf_counter() - request_start)
                
            except Exception:
                total_requests += 1
                if current_run > 0:
                    consecutive_runs.append(current_run)
                    current_run = 0
            
            await asyncio.sleep(request_delay)
        
        # Record final run
        if current_run > 0:
            consecutive_runs.append(current_run)
        
        elapsed_time = time.perf_counter() - start_time
        throughput = successful_requests / elapsed_time if elapsed_time > 0 else 0
        
        return BaselineMetrics(
            total_requests=total_requests,
            successful_requests=successful_requests,
            rate_limited_requests=rate_limited_requests,
            total_retries=client.total_retries,
            elapsed_time=elapsed_time,
            throughput=throughput,
            total_wait_time=client.total_wait_time,
            max_backoff_time=client.max_backoff_time,
            consecutive_runs=consecutive_runs,
            request_times=request_times
        )


def calculate_baseline_score(metrics: BaselineMetrics) -> float:
    """Calculate a score for baseline metrics (for comparison with SmartSurge)."""
    score = 0.0
    
    # Time component (includes backoff wait time)
    total_time_normalized = metrics.elapsed_time / 10.0  # Normalize to ~1.0
    score += total_time_normalized * 0.25
    
    # No detection capability (maximum penalty)
    score += 1.0 * 0.25
    
    # Consecutive patterns (baseline doesn't prevent them)
    if metrics.consecutive_runs:
        avg_consecutive = sum(metrics.consecutive_runs) / len(metrics.consecutive_runs)
        pattern_score = min(avg_consecutive / 10.0, 2.0)
    else:
        pattern_score = 0.0
    score += pattern_score * 0.2
    
    # Throughput
    if metrics.throughput > 0:
        throughput_score = 10.0 / metrics.throughput
    else:
        throughput_score = 2.0
    score += throughput_score * 0.2
    
    # No early detection (maximum penalty)
    score += 2.0 * 0.1
    
    return score