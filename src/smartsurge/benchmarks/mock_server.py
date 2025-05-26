"""
Mock web server for benchmarking SmartSurge HMM effectiveness.

This server provides configurable rate limiting strategies to test how well
the HMM can detect and adapt to different rate limiting patterns.

Key Features:
- Multiple rate limiting strategies (fixed window, sliding window, token bucket, etc.)
- Configurable response patterns and headers
- Metrics collection for analysis
- Noise injection for realistic scenarios
- Dynamic rate limit changes
"""

import logging
import math
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request
from werkzeug.serving import make_server

# Configure logging - will be overridden if run as a module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Available rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    ADAPTIVE = "adaptive"
    TIERED = "tiered"
    DYNAMIC = "dynamic"
    LOAD_DEPENDENT = "load_dependent"  # Smooth transitions based on load
    COMBINED = "combined"  # Combines multiple strategies


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting behavior."""
    
    # Basic configuration
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    requests_per_window: int = 100
    window_seconds: int = 60
    
    # Burst configuration (for token/leaky bucket)
    burst_size: Optional[int] = None
    refill_rate: float = 1.0  # tokens per second
    
    # Response configuration
    response_code: int = 429
    response_message: str = "Rate limit exceeded"
    include_retry_after: bool = True
    retry_after_seconds: Optional[int] = None
    include_rate_headers: bool = True
    custom_headers: Dict[str, str] = field(default_factory=dict)
    
    # Advanced features
    per_endpoint: bool = False
    per_ip: bool = True
    
    # Adaptive rate limiting
    adaptive_threshold: float = 0.8  # Start adapting at 80% capacity
    adaptive_decrease_factor: float = 0.9  # Reduce limit by 10%
    adaptive_increase_factor: float = 1.1  # Increase limit by 10%
    
    # Tiered rate limiting
    tiers: Dict[str, int] = field(default_factory=lambda: {
        "free": 100,
        "basic": 1000,
        "premium": 10000
    })
    
    # Dynamic rate limiting (time-based changes)
    dynamic_schedule: Dict[int, int] = field(default_factory=dict)  # hour -> limit
    
    # Noise and variations
    noise_enabled: bool = False
    noise_probability: float = 0.1  # Probability of injecting noise
    random_delay_ms: Tuple[int, int] = (0, 100)  # Min/max random delay
    false_positive_rate: float = 0.0  # Occasionally trigger rate limit early
    false_negative_rate: float = 0.0  # Occasionally don't trigger rate limit
    
    # Metrics collection
    collect_metrics: bool = True
    metrics_window: int = 300  # Keep metrics for last 5 minutes
    
    # Load-dependent rate limiting
    load_min_limit: int = 10  # Minimum rate limit at high load
    load_max_limit: int = 1000  # Maximum rate limit at low load
    load_transition_period: float = 60.0  # Seconds for full transition
    load_oscillation_period: float = 300.0  # Seconds for load cycle
    load_noise_factor: float = 0.1  # Random noise in load calculation


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    timestamp: datetime
    endpoint: str
    method: str
    ip: str
    status_code: int
    response_time_ms: float
    rate_limited: bool
    rate_limit_remaining: Optional[int] = None
    strategy_used: Optional[str] = None


class RateLimiter(ABC):
    """Abstract base class for rate limiting strategies."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.lock = Lock()
    
    @abstractmethod
    def check_rate_limit(self, key: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed under rate limit.
        
        Returns:
            Tuple of (is_allowed, rate_info)
        """
        pass
    
    @abstractmethod
    def reset(self, key: Optional[str] = None):
        """Reset rate limiter state."""
        pass
    
    def get_key(self, endpoint: str, ip: str) -> str:
        """Generate rate limit key based on configuration."""
        parts = []
        if self.config.per_ip:
            parts.append(ip)
        if self.config.per_endpoint:
            parts.append(endpoint)
        return ":".join(parts) if parts else "global"


class FixedWindowRateLimiter(RateLimiter):
    """Fixed window rate limiting strategy."""
    
    def __init__(self, config: RateLimitConfig):
        super().__init__(config)
        self.windows: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "window_start": time.time()
        })
    
    def check_rate_limit(self, key: str) -> Tuple[bool, Dict[str, Any]]:
        with self.lock:
            now = time.time()
            window = self.windows[key]
            
            # Check if we need to reset the window
            window_start = window["window_start"]
            if now - window_start >= self.config.window_seconds:
                window["count"] = 0
                window["window_start"] = now
                window_start = now
            
            # Check rate limit
            count = window["count"]
            limit = self.config.requests_per_window
            
            rate_info = {
                "limit": limit,
                "remaining": max(0, limit - count),
                "reset": int(window_start + self.config.window_seconds),
                "window_start": int(window_start)
            }
            
            if count >= limit:
                if self.config.include_retry_after:
                    if self.config.retry_after_seconds is not None:
                        rate_info["retry_after"] = self.config.retry_after_seconds
                    else:
                        rate_info["retry_after"] = rate_info["reset"] - int(now)
                return False, rate_info
            
            # Increment counter
            window["count"] += 1
            rate_info["remaining"] = max(0, limit - window["count"])
            
            return True, rate_info
    
    def reset(self, key: Optional[str] = None):
        with self.lock:
            if key:
                self.windows.pop(key, None)
            else:
                self.windows.clear()


class SlidingWindowRateLimiter(RateLimiter):
    """Sliding window rate limiting strategy."""
    
    def __init__(self, config: RateLimitConfig):
        super().__init__(config)
        self.requests: Dict[str, deque] = defaultdict(deque)
    
    def check_rate_limit(self, key: str) -> Tuple[bool, Dict[str, Any]]:
        with self.lock:
            now = time.time()
            window_start = now - self.config.window_seconds
            
            # Clean old requests
            while self.requests[key] and self.requests[key][0] < window_start:
                self.requests[key].popleft()
            
            count = len(self.requests[key])
            limit = self.config.requests_per_window
            
            rate_info = {
                "limit": limit,
                "remaining": max(0, limit - count),
                "reset": int(now + self.config.window_seconds)
            }
            
            if count >= limit:
                if self.config.include_retry_after:
                    if self.config.retry_after_seconds is not None:
                        rate_info["retry_after"] = self.config.retry_after_seconds
                    else:
                        # Calculate when the oldest request will expire
                        oldest = self.requests[key][0]
                        rate_info["retry_after"] = int(oldest + self.config.window_seconds - now) + 1
                return False, rate_info
            
            # Add request
            self.requests[key].append(now)
            rate_info["remaining"] = max(0, limit - len(self.requests[key]))
            
            return True, rate_info
    
    def reset(self, key: Optional[str] = None):
        with self.lock:
            if key:
                self.requests.pop(key, None)
            else:
                self.requests.clear()


class TokenBucketRateLimiter(RateLimiter):
    """Token bucket rate limiting strategy."""
    
    def __init__(self, config: RateLimitConfig):
        super().__init__(config)
        self.buckets: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "tokens": config.burst_size or config.requests_per_window,
            "last_refill": time.time()
        })
    
    def check_rate_limit(self, key: str) -> Tuple[bool, Dict[str, Any]]:
        with self.lock:
            now = time.time()
            bucket = self.buckets[key]
            
            # Refill tokens
            time_passed = now - bucket["last_refill"]
            tokens_to_add = time_passed * self.config.refill_rate
            max_tokens = self.config.burst_size or self.config.requests_per_window
            
            bucket["tokens"] = min(max_tokens, bucket["tokens"] + tokens_to_add)
            bucket["last_refill"] = now
            
            rate_info = {
                "limit": max_tokens,
                "remaining": int(bucket["tokens"]),
                "refill_rate": self.config.refill_rate,
                "burst_size": max_tokens
            }
            
            if bucket["tokens"] < 1:
                if self.config.include_retry_after:
                    # Calculate time until we have 1 token
                    time_until_token = (1 - bucket["tokens"]) / self.config.refill_rate
                    rate_info["retry_after"] = int(time_until_token) + 1
                return False, rate_info
            
            # Consume token
            bucket["tokens"] -= 1
            rate_info["remaining"] = int(bucket["tokens"])
            
            return True, rate_info
    
    def reset(self, key: Optional[str] = None):
        with self.lock:
            if key:
                self.buckets.pop(key, None)
            else:
                self.buckets.clear()


class AdaptiveRateLimiter(RateLimiter):
    """Adaptive rate limiting that adjusts limits based on load."""
    
    def __init__(self, config: RateLimitConfig):
        super().__init__(config)
        self.base_limiter = SlidingWindowRateLimiter(config)
        self.current_limits: Dict[str, int] = defaultdict(
            lambda: config.requests_per_window
        )
        self.load_history: deque = deque(maxlen=10)  # Track last 10 windows
        self.last_adaptation = time.time()
    
    def check_rate_limit(self, key: str) -> Tuple[bool, Dict[str, Any]]:
        # Adapt limits periodically
        now = time.time()
        if now - self.last_adaptation > self.config.window_seconds:
            self._adapt_limits()
            self.last_adaptation = now
        
        # Use adjusted limit
        original_limit = self.config.requests_per_window
        self.config.requests_per_window = self.current_limits[key]
        
        allowed, rate_info = self.base_limiter.check_rate_limit(key)
        
        # Restore original limit
        self.config.requests_per_window = original_limit
        
        # Add adaptive info
        rate_info["adaptive"] = True
        rate_info["original_limit"] = original_limit
        
        return allowed, rate_info
    
    def _adapt_limits(self):
        """Adjust limits based on current load."""
        # This is a simplified adaptation logic
        # In practice, you'd want more sophisticated algorithms
        
        for key in list(self.current_limits.keys()):
            current_limit = self.current_limits[key]
            
            # Random adjustment for demo purposes
            if random.random() < 0.3:
                # Decrease limit
                new_limit = int(current_limit * self.config.adaptive_decrease_factor)
            elif random.random() < 0.3:
                # Increase limit
                new_limit = int(current_limit * self.config.adaptive_increase_factor)
            else:
                # Keep current
                new_limit = current_limit
            
            # Enforce bounds
            original_limit = self.config.requests_per_window
            new_limit = max(int(original_limit * 0.1), new_limit)  # Min 10% of original
            new_limit = min(int(original_limit * 2), new_limit)    # Max 200% of original
            
            self.current_limits[key] = new_limit
    
    def reset(self, key: Optional[str] = None):
        self.base_limiter.reset(key)
        with self.lock:
            if key:
                self.current_limits.pop(key, None)
            else:
                self.current_limits.clear()


class DynamicRateLimiter(RateLimiter):
    """Dynamic rate limiting that changes limits based on time of day."""
    
    def __init__(self, config: RateLimitConfig):
        super().__init__(config)
        self.base_limiter = SlidingWindowRateLimiter(config)
        
        # Set up default schedule if none provided
        if not config.dynamic_schedule:
            # Example: Lower limits during business hours
            for hour in range(24):
                if 9 <= hour <= 17:  # Business hours
                    config.dynamic_schedule[hour] = config.requests_per_window // 2
                else:
                    config.dynamic_schedule[hour] = config.requests_per_window
    
    def check_rate_limit(self, key: str) -> Tuple[bool, Dict[str, Any]]:
        # Get current hour and adjust limit
        current_hour = datetime.now().hour
        original_limit = self.config.requests_per_window
        
        if current_hour in self.config.dynamic_schedule:
            self.config.requests_per_window = self.config.dynamic_schedule[current_hour]
        
        allowed, rate_info = self.base_limiter.check_rate_limit(key)
        
        # Restore original limit
        self.config.requests_per_window = original_limit
        
        # Add dynamic info
        rate_info["dynamic"] = True
        rate_info["current_hour"] = current_hour
        rate_info["schedule"] = self.config.dynamic_schedule
        
        return allowed, rate_info
    
    def reset(self, key: Optional[str] = None):
        self.base_limiter.reset(key)


class LoadDependentRateLimiter(RateLimiter):
    """Load-dependent rate limiting with smooth transitions."""
    
    def __init__(self, config: RateLimitConfig):
        super().__init__(config)
        self.base_limiter = SlidingWindowRateLimiter(config)
        self.start_time = time.time()
        self.last_load_update = time.time()
        self.current_load = 0.5  # Start at medium load
        self.target_load = 0.5
        self.load_velocity = 0.0  # Rate of change
        
    def _calculate_load(self) -> float:
        """Calculate current load with smooth transitions and oscillation."""
        now = time.time()
        elapsed = now - self.start_time
        
        # Base oscillation using sine wave
        oscillation = math.sin(2 * math.pi * elapsed / self.config.load_oscillation_period)
        
        # Map oscillation to 0-1 range
        base_load = (oscillation + 1) / 2
        
        # Add some randomness for more realistic behavior
        noise = random.gauss(0, self.config.load_noise_factor)
        
        # Smooth transitions
        dt = now - self.last_load_update
        self.last_load_update = now
        
        # Update target with some randomness
        if random.random() < 0.1:  # 10% chance to change direction
            self.target_load = max(0, min(1, base_load + noise))
        
        # Smooth transition to target
        load_diff = self.target_load - self.current_load
        transition_rate = 1.0 / self.config.load_transition_period
        self.current_load += load_diff * min(1.0, dt * transition_rate)
        
        # Ensure load stays in bounds
        self.current_load = max(0, min(1, self.current_load))
        
        return self.current_load
    
    def _get_current_limit(self) -> int:
        """Calculate current rate limit based on load."""
        load = self._calculate_load()
        
        # Inverse relationship: high load = low limit
        # Linear interpolation between min and max limits
        limit_range = self.config.load_max_limit - self.config.load_min_limit
        current_limit = int(self.config.load_max_limit - (load * limit_range))
        
        return current_limit
    
    def check_rate_limit(self, key: str) -> Tuple[bool, Dict[str, Any]]:
        # Update the base limiter's config with current limit
        original_limit = self.config.requests_per_window
        self.config.requests_per_window = self._get_current_limit()
        
        allowed, rate_info = self.base_limiter.check_rate_limit(key)
        
        # Restore original limit
        self.config.requests_per_window = original_limit
        
        # Add load-dependent info
        rate_info["load_dependent"] = True
        rate_info["current_load"] = round(self.current_load, 3)
        rate_info["load_min_limit"] = self.config.load_min_limit
        rate_info["load_max_limit"] = self.config.load_max_limit
        
        return allowed, rate_info
    
    def reset(self, key: Optional[str] = None):
        self.base_limiter.reset(key)
        if key is None:
            # Reset load tracking
            self.start_time = time.time()
            self.last_load_update = time.time()
            self.current_load = 0.5
            self.target_load = 0.5


class CombinedRateLimiter(RateLimiter):
    """Combines multiple rate limiting strategies."""
    
    def __init__(self, config: RateLimitConfig, strategies: List[RateLimiter]):
        super().__init__(config)
        self.strategies = strategies
    
    def check_rate_limit(self, key: str) -> Tuple[bool, Dict[str, Any]]:
        """All strategies must allow the request."""
        combined_info = {
            "combined": True,
            "strategies": {}
        }
        
        for i, strategy in enumerate(self.strategies):
            allowed, rate_info = strategy.check_rate_limit(key)
            strategy_name = strategy.__class__.__name__
            combined_info["strategies"][strategy_name] = rate_info
            
            if not allowed:
                combined_info.update(rate_info)  # Use info from limiting strategy
                return False, combined_info
        
        # All strategies allowed
        # Use the most restrictive remaining count
        min_remaining = float('inf')
        for strategy_info in combined_info["strategies"].values():
            if "remaining" in strategy_info:
                min_remaining = min(min_remaining, strategy_info["remaining"])
        
        combined_info["remaining"] = int(min_remaining) if min_remaining != float('inf') else 0
        
        return True, combined_info
    
    def reset(self, key: Optional[str] = None):
        for strategy in self.strategies:
            strategy.reset(key)


class BenchmarkMockServer:
    """Mock server for benchmarking SmartSurge."""
    
    def __init__(self, config: RateLimitConfig):
        self.app = Flask(__name__)
        self.config = config
        self.rate_limiter = self._create_rate_limiter()
        self.server = None
        
        # Metrics collection
        self.metrics: deque = deque(maxlen=10000)  # Keep last 10k requests
        self.metrics_lock = Lock()
        
        # Request counters
        self.total_requests = 0
        self.rate_limited_requests = 0
        self.endpoint_counts = defaultdict(int)
        
        self._setup_routes()
        
        # Start metrics cleanup thread
        if config.collect_metrics:
            self._start_metrics_cleanup()
    
    def _create_rate_limiter(self) -> RateLimiter:
        """Create rate limiter based on strategy."""
        strategy = self.config.strategy
        
        if strategy == RateLimitStrategy.FIXED_WINDOW:
            return FixedWindowRateLimiter(self.config)
        elif strategy == RateLimitStrategy.SLIDING_WINDOW:
            return SlidingWindowRateLimiter(self.config)
        elif strategy == RateLimitStrategy.TOKEN_BUCKET:
            return TokenBucketRateLimiter(self.config)
        elif strategy == RateLimitStrategy.ADAPTIVE:
            return AdaptiveRateLimiter(self.config)
        elif strategy == RateLimitStrategy.DYNAMIC:
            return DynamicRateLimiter(self.config)
        elif strategy == RateLimitStrategy.LOAD_DEPENDENT:
            return LoadDependentRateLimiter(self.config)
        elif strategy == RateLimitStrategy.COMBINED:
            # Example: Combine sliding window with token bucket
            strategies = [
                SlidingWindowRateLimiter(self.config),
                TokenBucketRateLimiter(self.config)
            ]
            return CombinedRateLimiter(self.config, strategies)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _inject_noise(self) -> Optional[Tuple[int, Dict[str, Any]]]:
        """Inject noise into responses if configured."""
        if not self.config.noise_enabled:
            return None
        
        if random.random() > self.config.noise_probability:
            return None
        
        # Random delay
        if self.config.random_delay_ms[1] > 0:
            delay_ms = random.randint(*self.config.random_delay_ms)
            time.sleep(delay_ms / 1000.0)
        
        # False positive (rate limit when shouldn't)
        if self.config.false_positive_rate > 0 and random.random() < self.config.false_positive_rate:
            return self.config.response_code, {
                "error": "False positive rate limit",
                "noise": True
            }
        
        return None
    
    def _rate_limit_decorator(self, f: Callable) -> Callable:
        """Decorator to apply rate limiting to routes."""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            start_time = time.time()
            
            # Get request info
            endpoint = request.endpoint or "unknown"
            ip = request.remote_addr or "unknown"
            method = request.method
            
            # Inject noise (early return)
            noise_response = self._inject_noise()
            if noise_response:
                status_code, data = noise_response
                self._record_metric(
                    endpoint, method, ip, status_code,
                    time.time() - start_time, True, None
                )
                return jsonify(data), status_code, self.config.custom_headers
            
            # Check rate limit
            key = self.rate_limiter.get_key(endpoint, ip)
            allowed, rate_info = self.rate_limiter.check_rate_limit(key)
            
            # False negative (don't rate limit when should)
            if (not allowed and self.config.false_negative_rate > 0 and 
                random.random() < self.config.false_negative_rate):
                allowed = True
                rate_info["false_negative"] = True
            
            # Build headers
            headers = dict(self.config.custom_headers)
            
            if self.config.include_rate_headers:
                headers.update({
                    "X-RateLimit-Limit": str(rate_info.get("limit", self.config.requests_per_window)),
                    "X-RateLimit-Remaining": str(rate_info.get("remaining", 0)),
                    "X-RateLimit-Reset": str(rate_info.get("reset", 0))
                })
                
                # Add strategy-specific headers
                if "burst_size" in rate_info:
                    headers["X-RateLimit-Burst"] = str(rate_info["burst_size"])
                if "refill_rate" in rate_info:
                    headers["X-RateLimit-Refill"] = str(rate_info["refill_rate"])
                if "adaptive" in rate_info:
                    headers["X-RateLimit-Adaptive"] = "true"
                if "dynamic" in rate_info:
                    headers["X-RateLimit-Dynamic"] = "true"
                if "load_dependent" in rate_info:
                    headers["X-RateLimit-LoadDependent"] = "true"
                    headers["X-RateLimit-CurrentLoad"] = str(rate_info.get("current_load", 0))
            
            if not allowed:
                # Rate limited
                self.rate_limited_requests += 1
                
                if self.config.include_retry_after and "retry_after" in rate_info:
                    headers["Retry-After"] = str(rate_info["retry_after"])
                
                response_data = {
                    "error": self.config.response_message,
                    "code": "RATE_LIMITED"
                }
                
                if "retry_after" in rate_info:
                    response_data["retry_after"] = rate_info["retry_after"]
                
                # Record metric
                self._record_metric(
                    endpoint, method, ip, self.config.response_code,
                    time.time() - start_time, True, rate_info.get("remaining", 0)
                )
                
                return jsonify(response_data), self.config.response_code, headers
            
            # Execute the actual route
            self.total_requests += 1
            self.endpoint_counts[endpoint] += 1
            
            response = f(*args, **kwargs)
            
            # Add headers to successful response
            if isinstance(response, tuple):
                data, status = response[:2]
                if len(response) > 2:
                    resp_headers = response[2]
                    resp_headers.update(headers)
                else:
                    resp_headers = headers
                
                # Record metric
                self._record_metric(
                    endpoint, method, ip, status,
                    time.time() - start_time, False, rate_info.get("remaining", 0)
                )
                
                return data, status, resp_headers
            else:
                # Record metric
                self._record_metric(
                    endpoint, method, ip, 200,
                    time.time() - start_time, False, rate_info.get("remaining", 0)
                )
                
                return response, 200, headers
        
        return decorated_function
    
    def _record_metric(self, endpoint: str, method: str, ip: str,
                      status_code: int, response_time: float,
                      rate_limited: bool, remaining: Optional[int]):
        """Record request metrics."""
        if not self.config.collect_metrics:
            return
        
        metric = RequestMetrics(
            timestamp=datetime.now(),
            endpoint=endpoint,
            method=method,
            ip=ip,
            status_code=status_code,
            response_time_ms=response_time * 1000,
            rate_limited=rate_limited,
            rate_limit_remaining=remaining,
            strategy_used=self.config.strategy.value
        )
        
        with self.metrics_lock:
            self.metrics.append(metric)
    
    def _start_metrics_cleanup(self):
        """Start background thread to clean old metrics."""
        def cleanup():
            while True:
                time.sleep(60)  # Check every minute
                cutoff = datetime.now() - timedelta(seconds=self.config.metrics_window)
                
                with self.metrics_lock:
                    # Remove old metrics
                    while self.metrics and self.metrics[0].timestamp < cutoff:
                        self.metrics.popleft()
        
        thread = Thread(target=cleanup, daemon=True)
        thread.start()
    
    def _setup_routes(self):
        """Set up Flask routes."""
        
        @self.app.route("/api/test", methods=["GET", "POST", "PUT", "DELETE"])
        @self._rate_limit_decorator
        def test_endpoint():
            """Basic test endpoint."""
            return jsonify({
                "message": "Success",
                "timestamp": datetime.now().isoformat(),
                "method": request.method
            })
        
        @self.app.route("/api/data/<int:item_id>", methods=["GET"])
        @self._rate_limit_decorator
        def get_data(item_id):
            """Parameterized endpoint."""
            return jsonify({
                "id": item_id,
                "data": f"Item {item_id}",
                "timestamp": datetime.now().isoformat()
            })
        
        @self.app.route("/api/search", methods=["GET"])
        @self._rate_limit_decorator
        def search():
            """Search endpoint with query parameters."""
            query = request.args.get("q", "")
            page = request.args.get("page", 1, type=int)
            
            return jsonify({
                "query": query,
                "page": page,
                "results": [f"Result {i} for '{query}'" for i in range(10)],
                "timestamp": datetime.now().isoformat()
            })
        
        @self.app.route("/api/heavy", methods=["GET"])
        @self._rate_limit_decorator
        def heavy_endpoint():
            """Endpoint that simulates heavy processing."""
            # Simulate processing time
            time.sleep(random.uniform(0.1, 0.5))
            
            return jsonify({
                "message": "Heavy processing complete",
                "timestamp": datetime.now().isoformat()
            })
        
        # Admin endpoints (no rate limiting)
        
        @self.app.route("/admin/metrics", methods=["GET"])
        def get_metrics():
            """Get server metrics."""
            with self.metrics_lock:
                recent_metrics = list(self.metrics)[-100:]  # Last 100 requests
            
            # Calculate statistics
            total = len(recent_metrics)
            rate_limited = sum(1 for m in recent_metrics if m.rate_limited)
            avg_response_time = (
                sum(m.response_time_ms for m in recent_metrics) / total
                if total > 0 else 0
            )
            
            return jsonify({
                "total_requests": self.total_requests,
                "rate_limited_requests": self.rate_limited_requests,
                "rate_limit_percentage": (
                    self.rate_limited_requests / self.total_requests * 100
                    if self.total_requests > 0 else 0
                ),
                "endpoint_counts": dict(self.endpoint_counts),
                "recent_requests": total,
                "recent_rate_limited": rate_limited,
                "avg_response_time_ms": avg_response_time,
                "strategy": self.config.strategy.value,
                "config": {
                    "requests_per_window": self.config.requests_per_window,
                    "window_seconds": self.config.window_seconds,
                    "noise_enabled": self.config.noise_enabled
                }
            })
        
        @self.app.route("/admin/reset", methods=["POST"])
        def reset():
            """Reset rate limiter and metrics."""
            self.rate_limiter.reset()
            
            with self.metrics_lock:
                self.metrics.clear()
            
            self.total_requests = 0
            self.rate_limited_requests = 0
            self.endpoint_counts.clear()
            
            return jsonify({"message": "Server reset complete"})
        
        @self.app.route("/admin/config", methods=["GET", "POST"])
        def config():
            """Get or update server configuration."""
            if request.method == "GET":
                return jsonify({
                    "strategy": self.config.strategy.value,
                    "requests_per_window": self.config.requests_per_window,
                    "window_seconds": self.config.window_seconds,
                    "noise_enabled": self.config.noise_enabled,
                    "noise_probability": self.config.noise_probability
                })
            else:
                # Update configuration
                data = request.json
                
                # Only allow updating certain fields
                if "requests_per_window" in data:
                    self.config.requests_per_window = int(data["requests_per_window"])
                if "window_seconds" in data:
                    self.config.window_seconds = int(data["window_seconds"])
                if "noise_enabled" in data:
                    self.config.noise_enabled = bool(data["noise_enabled"])
                if "noise_probability" in data:
                    self.config.noise_probability = float(data["noise_probability"])
                
                # Recreate rate limiter with new config
                self.rate_limiter = self._create_rate_limiter()
                
                return jsonify({"message": "Configuration updated"})
        
        @self.app.route("/health", methods=["GET"])
        def health():
            """Health check endpoint (no rate limiting)."""
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat()
            })
    
    def start(self, host: str = "127.0.0.1", port: int = 5000):
        """Start the mock server."""
        logger.info(f"Starting benchmark server on {host}:{port}")
        logger.info(f"Rate limit strategy: {self.config.strategy.value}")
        logger.info(f"Rate limit: {self.config.requests_per_window} requests per {self.config.window_seconds} seconds")
        
        self.server = make_server(host, port, self.app, threaded=True)
        self.server.serve_forever()
    
    def stop(self):
        """Stop the mock server."""
        if self.server:
            logger.info("Stopping benchmark server")
            self.server.shutdown()


def create_benchmark_server(config: Optional[RateLimitConfig] = None) -> BenchmarkMockServer:
    """Factory function to create a benchmark server."""
    if config is None:
        config = RateLimitConfig()
    
    return BenchmarkMockServer(config)


# Example configurations for different benchmark scenarios

def get_strict_rate_limit_config() -> RateLimitConfig:
    """Strict rate limiting for testing detection."""
    return RateLimitConfig(
        strategy=RateLimitStrategy.SLIDING_WINDOW,
        requests_per_window=10,
        window_seconds=10,  # Shorter window for faster benchmarking
        include_retry_after=True,
        retry_after_seconds=2  # Much shorter retry for benchmarking
    )


def get_token_bucket_config() -> RateLimitConfig:
    """Token bucket for testing burst handling."""
    return RateLimitConfig(
        strategy=RateLimitStrategy.TOKEN_BUCKET,
        requests_per_window=20,
        window_seconds=10,
        burst_size=10,
        refill_rate=20 / 10,  # 20 tokens per 10 seconds
        retry_after_seconds=2
    )


def get_noisy_config() -> RateLimitConfig:
    """Configuration with noise for realistic testing."""
    return RateLimitConfig(
        strategy=RateLimitStrategy.SLIDING_WINDOW,
        requests_per_window=15,
        window_seconds=10,
        noise_enabled=True,
        noise_probability=0.2,
        random_delay_ms=(0, 200),
        false_positive_rate=0.05,
        false_negative_rate=0.05,
        retry_after_seconds=2
    )


def get_adaptive_config() -> RateLimitConfig:
    """Adaptive rate limiting that changes over time."""
    return RateLimitConfig(
        strategy=RateLimitStrategy.ADAPTIVE,
        requests_per_window=20,
        window_seconds=10,
        adaptive_threshold=0.8,
        adaptive_decrease_factor=0.8,
        adaptive_increase_factor=1.2,
        retry_after_seconds=2
    )


def get_dynamic_config() -> RateLimitConfig:
    """Time-based dynamic rate limiting."""
    config = RateLimitConfig(
        strategy=RateLimitStrategy.DYNAMIC,
        requests_per_window=20,
        window_seconds=10,
        retry_after_seconds=2
    )
    
    # Custom schedule: lower limits during peak hours
    for hour in range(24):
        if 8 <= hour <= 10 or 12 <= hour <= 14 or 17 <= hour <= 19:
            # Peak hours
            config.dynamic_schedule[hour] = 10
        elif 0 <= hour <= 6:
            # Night hours
            config.dynamic_schedule[hour] = 30
        else:
            # Normal hours
            config.dynamic_schedule[hour] = 20
    
    return config


def get_load_dependent_config() -> RateLimitConfig:
    """Load-dependent rate limiting with smooth transitions."""
    return RateLimitConfig(
        strategy=RateLimitStrategy.LOAD_DEPENDENT,
        requests_per_window=20,  # Base limit (not used directly)
        window_seconds=10,
        load_min_limit=5,  # Minimum rate limit at high load
        load_max_limit=50,  # Maximum rate limit at low load
        load_transition_period=10.0,  # 10 seconds for smooth transitions
        load_oscillation_period=60.0,  # 1-minute load cycles
        load_noise_factor=0.15,  # 15% random variation
        include_retry_after=True,
        include_rate_headers=True,
        retry_after_seconds=2
    )


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    strategy = sys.argv[1] if len(sys.argv) > 1 else "sliding_window"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    
    # Select configuration based on strategy
    configs = {
        "strict": get_strict_rate_limit_config(),
        "token_bucket": get_token_bucket_config(),
        "noisy": get_noisy_config(),
        "adaptive": get_adaptive_config(),
        "dynamic": get_dynamic_config(),
        "load_dependent": get_load_dependent_config()
    }
    
    config = configs.get(strategy, RateLimitConfig())
    
    # Create and start server
    server = create_benchmark_server(config)
    
    try:
        server.start(port=port)
    except KeyboardInterrupt:
        server.stop()