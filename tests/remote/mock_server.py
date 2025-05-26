"""Mock Flask server with configurable rate-limiting for testing SmartSurge."""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from threading import Lock
from typing import Any, Callable, Dict, Optional, Tuple

import pytest
from flask import Flask, Response, jsonify, request, stream_with_context
from werkzeug.serving import make_server


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting behavior."""
    
    requests_per_window: int = 10
    window_seconds: int = 60
    burst_size: Optional[int] = None  # If set, allows burst requests
    retry_after_seconds: Optional[int] = None  # If set, adds Retry-After header
    response_code: int = 429
    response_message: str = "Rate limit exceeded"
    use_sliding_window: bool = True  # True for sliding, False for fixed window
    per_endpoint: bool = False  # If True, limits are per endpoint
    per_ip: bool = True  # If True, limits are per IP address
    
    # Advanced configurations
    gradual_backoff: bool = False  # Increases retry time on repeated violations
    soft_limit: Optional[int] = None  # Warn before hard limit
    custom_headers: Dict[str, str] = field(default_factory=dict)


class RateLimiter:
    """Implements various rate limiting strategies."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.violations: Dict[str, int] = defaultdict(int)
        self.lock = Lock()
    
    def _get_key(self, endpoint: Optional[str] = None) -> str:
        """Generate rate limit key based on configuration."""
        parts = []
        
        if self.config.per_ip:
            parts.append(request.remote_addr or "unknown")
        
        if self.config.per_endpoint and endpoint:
            parts.append(endpoint)
        
        return ":".join(parts) if parts else "global"
    
    def _clean_old_requests(self, key: str, now: float) -> None:
        """Remove requests outside the current window."""
        window_start = now - self.config.window_seconds
        
        while self.requests[key] and self.requests[key][0] < window_start:
            self.requests[key].popleft()
    
    def check_rate_limit(self, endpoint: Optional[str] = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if request is allowed under rate limit.
        
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        with self.lock:
            now = time.time()
            key = self._get_key(endpoint)
            
            if self.config.use_sliding_window:
                self._clean_old_requests(key, now)
            else:
                # Fixed window - reset if window expired
                if self.requests[key] and len(self.requests[key]) > 0:
                    window_start = (int(now) // self.config.window_seconds) * self.config.window_seconds
                    if self.requests[key][0] < window_start:
                        self.requests[key].clear()
            
            current_count = len(self.requests[key])
            
            # Check burst allowance
            effective_limit = self.config.requests_per_window
            if self.config.burst_size and current_count < self.config.burst_size:
                effective_limit = self.config.burst_size
            
            # Soft limit warning
            if self.config.soft_limit and current_count >= self.config.soft_limit:
                headers = {"X-RateLimit-Warning": "Approaching rate limit"}
            else:
                headers = {}
            
            if current_count >= effective_limit:
                # Rate limited
                self.violations[key] += 1
                
                retry_after = self.config.retry_after_seconds
                if retry_after and self.config.gradual_backoff:
                    # Exponential backoff based on violations
                    retry_after = retry_after * (2 ** min(self.violations[key] - 1, 5))
                
                rate_info = {
                    "limit": self.config.requests_per_window,
                    "remaining": 0,
                    "reset": int(now + self.config.window_seconds),
                    "headers": headers,
                    "retry_after": retry_after,
                    "violations": self.violations[key]
                }
                
                return False, rate_info
            
            # Request allowed - add to queue first
            self.requests[key].append(now)
            self.violations[key] = 0  # Reset violations on successful request
            
            # Calculate remaining after adding current request
            rate_info = {
                "limit": self.config.requests_per_window,
                "remaining": max(0, effective_limit - len(self.requests[key])),
                "reset": int(now + self.config.window_seconds),
                "headers": headers
            }
            
            return True, rate_info


class MockServer:
    """Flask-based mock server with configurable rate limiting."""
    
    def __init__(self, config: RateLimitConfig = None):
        self.app = Flask(__name__)
        self.config = config or RateLimitConfig()
        self.rate_limiter = RateLimiter(self.config)
        self.server = None
        self.request_count = 0
        self.endpoint_counts = defaultdict(int)
        
        self._setup_routes()
    
    def _rate_limit_decorator(self, f: Callable) -> Callable:
        """Decorator to apply rate limiting to routes."""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            endpoint = request.endpoint
            allowed, rate_info = self.rate_limiter.check_rate_limit(endpoint)
            
            # Add rate limit headers
            headers = {
                "X-RateLimit-Limit": str(self.config.requests_per_window),
                "X-RateLimit-Remaining": str(rate_info["remaining"]),
                "X-RateLimit-Reset": str(rate_info["reset"]),
            }
            
            # Add custom headers
            headers.update(self.config.custom_headers)
            headers.update(rate_info.get("headers", {}))
            
            if not allowed:
                # Rate limited response
                if rate_info.get("retry_after"):
                    headers["Retry-After"] = str(rate_info["retry_after"])
                
                response_data = {
                    "error": self.config.response_message,
                    "retry_after": rate_info.get("retry_after"),
                }
                
                return jsonify(response_data), self.config.response_code, headers
            
            # Execute the actual route
            response = f(*args, **kwargs)
            
            # Add headers to successful response
            if isinstance(response, tuple):
                data, status, resp_headers = response[0], response[1], response[2] if len(response) > 2 else {}
                resp_headers.update(headers)
                return data, status, resp_headers
            elif isinstance(response, Response):
                # For Response objects, update headers directly
                response.headers.update(headers)
                return response
            else:
                return response, 200, headers
        
        return decorated_function
    
    def _setup_routes(self):
        """Set up Flask routes with rate limiting."""
        
        @self.app.before_request
        def count_requests():
            self.request_count += 1
            self.endpoint_counts[request.endpoint or "unknown"] += 1
        
        @self.app.route("/api/test", methods=["GET", "POST"])
        @self._rate_limit_decorator
        def test_endpoint():
            return jsonify({
                "message": "Success",
                "request_count": self.request_count,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        @self.app.route("/api/data/<int:item_id>", methods=["GET"])
        @self._rate_limit_decorator
        def get_data(item_id):
            return jsonify({
                "id": item_id,
                "data": f"Item {item_id} data",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        @self.app.route("/api/stream", methods=["GET"])
        @self._rate_limit_decorator
        def stream_data():
            """Endpoint for testing streaming with rate limits."""
            def generate():
                for i in range(10):
                    yield f"data: Chunk {i}\n\n"
                    time.sleep(0.1)
            
            return Response(
                stream_with_context(generate()),
                mimetype="text/event-stream"
            )
        
        @self.app.route("/api/download", methods=["GET"])
        @self._rate_limit_decorator
        def download_file():
            """Endpoint for testing resumable downloads."""
            total_size = 1024 * 1024  # 1MB
            chunk_size = 1024  # 1KB chunks
            
            range_header = request.headers.get("Range")
            start = 0
            end = total_size - 1
            
            if range_header:
                # Parse range header
                range_match = range_header.replace("bytes=", "").split("-")
                start = int(range_match[0])
                if range_match[1]:
                    end = int(range_match[1])
            
            def generate():
                current = start
                while current <= end:
                    chunk_end = min(current + chunk_size - 1, end)
                    yield b"X" * (chunk_end - current + 1)
                    current = chunk_end + 1
            
            headers = {
                "Content-Type": "application/octet-stream",
                "Content-Length": str(end - start + 1),
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes {start}-{end}/{total_size}"
            }
            
            return Response(
                stream_with_context(generate()),
                status=206 if range_header else 200,
                headers=headers
            )
        
        @self.app.route("/api/status", methods=["GET"])
        def status():
            """Status endpoint without rate limiting."""
            return jsonify({
                "status": "ok",
                "total_requests": self.request_count,
                "endpoint_counts": dict(self.endpoint_counts),
                "rate_limit_config": {
                    "requests_per_window": self.config.requests_per_window,
                    "window_seconds": self.config.window_seconds,
                    "sliding_window": self.config.use_sliding_window
                }
            })
        
        @self.app.route("/api/reset", methods=["POST"])
        def reset():
            """Reset rate limiter state."""
            with self.rate_limiter.lock:
                self.rate_limiter.requests.clear()
                self.rate_limiter.violations.clear()
            self.request_count = 0
            self.endpoint_counts.clear()
            return jsonify({"message": "Rate limiter reset"})
    
    def start(self, port: int = 5000, threaded: bool = True):
        """Start the mock server."""
        self.server = make_server("127.0.0.1", port, self.app, threaded=threaded)
        self.server.serve_forever()
    
    def stop(self):
        """Stop the mock server."""
        if self.server:
            self.server.shutdown()


@pytest.fixture(scope="function")
def mock_server():
    """Pytest fixture that provides a mock server instance."""
    import threading
    
    server = MockServer()
    thread = threading.Thread(target=lambda: server.start(port=5555))
    thread.daemon = True
    thread.start()
    
    # Wait for server to start
    time.sleep(0.5)
    
    yield server
    
    server.stop()


@pytest.fixture(scope="function")
def custom_rate_limit_server():
    """Pytest fixture that provides a customizable mock server."""
    import threading
    
    servers = []
    
    def _create_server(config: RateLimitConfig, port: int = 5556) -> MockServer:
        server = MockServer(config)
        thread = threading.Thread(target=lambda: server.start(port=port))
        thread.daemon = True
        thread.start()
        
        # Wait for server to start
        time.sleep(0.5)
        
        servers.append(server)
        return server
    
    yield _create_server
    
    # Cleanup all servers
    for server in servers:
        server.stop()


if __name__ == "__main__":
    # Example usage
    config = RateLimitConfig(
        requests_per_window=5,
        window_seconds=10,
        retry_after_seconds=10,
        gradual_backoff=True
    )
    
    server = MockServer(config)
    print("Starting mock server on http://127.0.0.1:5000")
    print("Rate limit: 5 requests per 10 seconds")
    server.start()