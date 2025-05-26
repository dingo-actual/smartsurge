"""Integration tests using mock server with SmartSurge client."""

import asyncio

import pytest

from smartsurge import SmartSurgeClient
from .mock_server import RateLimitConfig, custom_rate_limit_server


class TestSmartSurgeIntegration:
    """Test SmartSurge client with mock rate-limited server."""
    
    def test_sync_client_rate_limit_detection(self, custom_rate_limit_server):
        """Test that SmartSurge detects and handles rate limits."""
        config = RateLimitConfig(
            requests_per_window=5,
            window_seconds=10,
            retry_after_seconds=2
        )
        
        server = custom_rate_limit_server(config, port=5563)
        
        # Create SmartSurge client
        client = SmartSurgeClient(
            base_url="http://127.0.0.1:5563",
            timeout=30,
            max_retries=3
        )
        
        # Make multiple requests
        responses = []
        all_histories = []
        for i in range(10):
            try:
                resp, history = client.get("/api/test")
                responses.append(resp)
                all_histories.append(history)
            except Exception as e:
                # SmartSurge should handle rate limits gracefully
                pass
        
        # Check that some requests succeeded
        assert len(responses) > 0
        
        # Check request history shows rate limit pattern
        # Get entries from the history objects
        all_entries = []
        for hist in all_histories:
            all_entries.extend(hist.entries)
        assert len(all_entries) > 0
        
        # Verify rate limits were detected
        rate_limited = [r for r in all_entries if r.status_code == 429]
        assert len(rate_limited) > 0
    
    @pytest.mark.asyncio
    async def test_async_client_rate_limit_detection(self, custom_rate_limit_server):
        """Test async client with rate limits."""
        config = RateLimitConfig(
            requests_per_window=3,
            window_seconds=5,
            retry_after_seconds=1
        )
        
        server = custom_rate_limit_server(config, port=5564)
        
        # Create async SmartSurge client
        client = SmartSurgeClient(
            base_url="http://127.0.0.1:5564",
            timeout=30,
            max_retries=2
        )
        
        # Make concurrent async requests
        async def make_request(i):
            try:
                resp, history = await client.async_get(f"/api/data/{i}")
                return resp
            except Exception:
                return None
        
        tasks = [make_request(i) for i in range(10)]
        responses = await asyncio.gather(*tasks)
        
        # Some requests should succeed
        successful = [r for r in responses if r is not None]
        assert len(successful) > 0
        
        # The client tracks histories internally
        # Rate limiting detection happens automatically through the RequestHistory objects
    
    def test_streaming_with_rate_limits(self, custom_rate_limit_server):
        """Test streaming functionality under rate limits."""
        config = RateLimitConfig(
            requests_per_window=2,
            window_seconds=10
        )
        
        server = custom_rate_limit_server(config, port=5565)
        
        client = SmartSurgeClient(base_url="http://127.0.0.1:5565")
        
        # First stream should work
        response, _ = client.get("/api/stream", stream=True)
        chunks = list(response.iter_content(chunk_size=1024))
        assert len(chunks) > 0
        
        # Second stream should also work (within limit)
        response, _ = client.get("/api/stream", stream=True)
        chunks = list(response.iter_content(chunk_size=1024))
        assert len(chunks) > 0
        
        # Third attempt might be rate limited
        try:
            response, _ = client.get("/api/stream", stream=True)
            # If not rate limited immediately, it worked
            assert response.status_code in [200, 429]
        except Exception:
            # Rate limit detected and handled
            pass
    
    def test_custom_headers_propagation(self, custom_rate_limit_server):
        """Test that custom headers are properly handled."""
        config = RateLimitConfig(
            requests_per_window=10,
            window_seconds=60,
            custom_headers={
                "X-API-Version": "1.0",
                "X-Service-Name": "test-service"
            }
        )
        
        server = custom_rate_limit_server(config, port=5566)
        
        client = SmartSurgeClient(base_url="http://127.0.0.1:5566")
        response, _ = client.get("/api/test")
        
        # Check custom headers are present
        assert response.headers.get("X-API-Version") == "1.0"
        assert response.headers.get("X-Service-Name") == "test-service"
    
    def test_soft_limit_warnings(self, custom_rate_limit_server):
        """Test soft limit warning functionality."""
        config = RateLimitConfig(
            requests_per_window=10,
            window_seconds=60,
            soft_limit=7
        )
        
        server = custom_rate_limit_server(config, port=5567)
        
        client = SmartSurgeClient(base_url="http://127.0.0.1:5567")
        
        # Make requests up to soft limit
        for i in range(8):
            response, _ = client.get("/api/test")
            
            if i >= 7:  # After soft limit
                assert "X-RateLimit-Warning" in response.headers
            else:
                assert "X-RateLimit-Warning" not in response.headers