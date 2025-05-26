"""Tests for the mock server implementation."""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from .mock_server import RateLimitConfig, custom_rate_limit_server, mock_server


class TestMockServer:
    """Test the mock server functionality."""
    
    def test_basic_rate_limiting(self, mock_server):
        """Test basic rate limiting with default configuration."""
        base_url = "http://127.0.0.1:5555"
        
        # Use requests directly to test the mock server behavior
        # Default config allows 10 requests per 60 seconds
        responses = []
        for i in range(12):
            resp = requests.get(f"{base_url}/api/test")
            responses.append(resp)
        
        # First 10 should succeed
        assert all(r.status_code == 200 for r in responses[:10])
        # 11th and 12th should be rate limited
        assert all(r.status_code == 429 for r in responses[10:])
    
    def test_custom_rate_limit(self, custom_rate_limit_server):
        """Test with custom rate limit configuration."""
        config = RateLimitConfig(
            requests_per_window=3,
            window_seconds=5,
            retry_after_seconds=5
        )
        
        server = custom_rate_limit_server(config, port=5557)
        base_url = "http://127.0.0.1:5557"
        
        # Make 5 requests
        responses = []
        for i in range(5):
            resp = requests.get(f"{base_url}/api/test")
            responses.append(resp)
        
        # First 3 should succeed
        assert all(r.status_code == 200 for r in responses[:3])
        # 4th and 5th should be rate limited
        assert all(r.status_code == 429 for r in responses[3:])
        
        # Check Retry-After header
        assert responses[3].headers.get("Retry-After") == "5"
    
    def test_rate_limit_headers(self, mock_server):
        """Test rate limit headers in responses."""
        base_url = "http://127.0.0.1:5555"
        
        resp = requests.get(f"{base_url}/api/test")
        assert resp.status_code == 200
        
        # Check rate limit headers
        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Remaining" in resp.headers
        assert "X-RateLimit-Reset" in resp.headers
        
        assert resp.headers["X-RateLimit-Limit"] == "10"
        assert int(resp.headers["X-RateLimit-Remaining"]) == 9
    
    def test_sliding_window(self, custom_rate_limit_server):
        """Test sliding window rate limiting."""
        config = RateLimitConfig(
            requests_per_window=3,
            window_seconds=3,
            use_sliding_window=True
        )
        
        server = custom_rate_limit_server(config, port=5558)
        base_url = "http://127.0.0.1:5558"
        
        # Make 3 requests
        for _ in range(3):
            resp = requests.get(f"{base_url}/api/test")
            assert resp.status_code == 200
        
        # 4th request should fail
        resp = requests.get(f"{base_url}/api/test")
        assert resp.status_code == 429
        
        # Wait for 1 request to expire
        time.sleep(3.1)
        
        # Should be able to make 1 more request
        resp = requests.get(f"{base_url}/api/test")
        assert resp.status_code == 200
    
    def test_burst_allowance(self, custom_rate_limit_server):
        """Test burst size allowance."""
        config = RateLimitConfig(
            requests_per_window=5,
            window_seconds=10,
            burst_size=8
        )
        
        server = custom_rate_limit_server(config, port=5559)
        base_url = "http://127.0.0.1:5559"
        
        # Should allow burst of 8 requests
        responses = []
        for _ in range(10):
            resp = requests.get(f"{base_url}/api/test")
            responses.append(resp)
        
        # First 8 should succeed (burst)
        assert all(r.status_code == 200 for r in responses[:8])
        # 9th and 10th should fail
        assert all(r.status_code == 429 for r in responses[8:])
    
    def test_gradual_backoff(self, custom_rate_limit_server):
        """Test gradual backoff on repeated violations."""
        config = RateLimitConfig(
            requests_per_window=2,
            window_seconds=10,
            retry_after_seconds=1,
            gradual_backoff=True
        )
        
        server = custom_rate_limit_server(config, port=5560)
        base_url = "http://127.0.0.1:5560"
        
        # Exhaust rate limit
        for _ in range(2):
            requests.get(f"{base_url}/api/test")
        
        # Check increasing retry times
        retry_times = []
        for _ in range(3):
            resp = requests.get(f"{base_url}/api/test")
            assert resp.status_code == 429
            retry_times.append(int(resp.headers.get("Retry-After", 0)))
        
        # Retry times should increase: 1, 2, 4
        assert retry_times == [1, 2, 4]
    
    def test_per_endpoint_limiting(self, custom_rate_limit_server):
        """Test per-endpoint rate limiting."""
        config = RateLimitConfig(
            requests_per_window=2,
            window_seconds=10,
            per_endpoint=True
        )
        
        server = custom_rate_limit_server(config, port=5561)
        base_url = "http://127.0.0.1:5561"
        
        # Make 2 requests to /api/test
        for _ in range(2):
            resp = requests.get(f"{base_url}/api/test")
            assert resp.status_code == 200
        
        # 3rd request to /api/test should fail
        resp = requests.get(f"{base_url}/api/test")
        assert resp.status_code == 429
        
        # But /api/data/1 should still work
        resp = requests.get(f"{base_url}/api/data/1")
        assert resp.status_code == 200
    
    def test_streaming_endpoint(self, mock_server):
        """Test streaming endpoint with rate limiting."""
        base_url = "http://127.0.0.1:5555"
        
        resp = requests.get(f"{base_url}/api/stream", stream=True)
        assert resp.status_code == 200
        
        chunks = []
        for chunk in resp.iter_content(chunk_size=1024):
            if chunk:
                chunks.append(chunk)
        
        assert len(chunks) > 0
    
    def test_resumable_download(self, mock_server):
        """Test resumable download endpoint."""
        base_url = "http://127.0.0.1:5555"
        
        # Initial request
        resp = requests.get(f"{base_url}/api/download")
        assert resp.status_code == 200
        assert "Accept-Ranges" in resp.headers
        
        # Range request
        headers = {"Range": "bytes=1024-2047"}
        resp = requests.get(f"{base_url}/api/download", headers=headers)
        assert resp.status_code == 206
        assert "Content-Range" in resp.headers
        assert len(resp.content) == 1024
    
    def test_reset_endpoint(self, mock_server):
        """Test rate limiter reset functionality."""
        base_url = "http://127.0.0.1:5555"
        
        # Exhaust rate limit
        for _ in range(10):
            requests.get(f"{base_url}/api/test")
        
        # Should be rate limited
        resp = requests.get(f"{base_url}/api/test")
        assert resp.status_code == 429
        
        # Reset rate limiter
        resp = requests.post(f"{base_url}/api/reset")
        assert resp.status_code == 200
        
        # Should work again
        resp = requests.get(f"{base_url}/api/test")
        assert resp.status_code == 200
    
    def test_concurrent_requests(self, custom_rate_limit_server):
        """Test rate limiting with concurrent requests."""
        config = RateLimitConfig(
            requests_per_window=5,
            window_seconds=5
        )
        
        server = custom_rate_limit_server(config, port=5562)
        base_url = "http://127.0.0.1:5562"
        
        def make_request():
            return requests.get(f"{base_url}/api/test")
        
        # Make 10 concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [f.result() for f in futures]
        
        # Exactly 5 should succeed
        success_count = sum(1 for r in responses if r.status_code == 200)
        assert success_count == 5
        
        # Exactly 5 should be rate limited
        limited_count = sum(1 for r in responses if r.status_code == 429)
        assert limited_count == 5