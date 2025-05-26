# Advanced Usage

This guide explores SmartSurge's advanced features for power users who need fine-grained control over rate limiting, performance optimization, and complex scenarios.

## Table of Contents

- [Asynchronous Operations](#asynchronous-operations)
- [Rate Limit Management](#rate-limit-management)
- [Custom Configuration](#custom-configuration)
- [HMM-Based Detection](#hmm-based-detection)
- [Resumable Streaming](#resumable-streaming)
- [Request History and Analytics](#request-history-and-analytics)
- [Error Handling Patterns](#error-handling-patterns)
- [Performance Optimization](#performance-optimization)

## Asynchronous Operations

SmartSurge provides full async/await support through its async request methods, enabling high-performance concurrent operations.

### Basic Async Usage

```python
import asyncio
from smartsurge import SmartSurgeClient

async def fetch_data():
    """Basic async request example"""
    client = SmartSurgeClient()
    
    # Async GET request
    response = await client.async_get("https://api.example.com/data")
    print(f"Status: {response.status_code}")
    print(f"Data: {response.json()}")
    
    # Async POST request
    response = await client.async_post(
        "https://api.example.com/users",
        json={"name": "John", "email": "john@example.com"}
    )
    return response.json()

# Run the async function
result = asyncio.run(fetch_data())
```

### Concurrent Requests with Rate Limit Protection

Execute multiple requests concurrently while SmartSurge handles rate limiting:

```python
import asyncio
from smartsurge import SmartSurgeClient

async def fetch_multiple_items(item_ids):
    """Fetch multiple items concurrently"""
    client = SmartSurgeClient()
    
    # Create coroutines for all requests
    tasks = [
        client.async_get(f"https://api.example.com/items/{item_id}")
        for item_id in item_ids
    ]
    
    # Execute concurrently - SmartSurge handles rate limiting automatically
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    results = []
    for item_id, response in zip(item_ids, responses):
        if isinstance(response, Exception):
            print(f"Failed to fetch item {item_id}: {response}")
        else:
            results.append(response.json())
    
    return results

# Fetch 50 items concurrently
item_ids = range(1, 51)
items = asyncio.run(fetch_multiple_items(item_ids))
```

### Async Requests with History Tracking

Use the `async_request_with_history` utility for batch operations with history:

```python
from smartsurge import SmartSurgeClient
from smartsurge.utilities import async_request_with_history
import asyncio

async def analyze_endpoints():
    """Analyze multiple endpoints and get request history"""
    client = SmartSurgeClient()
    
    # Define requests to make
    requests = [
        {"method": "GET", "url": "https://api.example.com/users"},
        {"method": "GET", "url": "https://api.example.com/products"},
        {"method": "POST", "url": "https://api.example.com/analytics", 
         "json": {"event": "page_view"}}
    ]
    
    # Make requests and get history
    responses, history = await async_request_with_history(client, requests)
    
    # Analyze the history
    print(f"Total requests made: {len(history.requests)}")
    
    # Check rate limit estimation
    rate_limit = history.estimate_rate_limit()
    if rate_limit:
        print(f"Estimated rate limit: {rate_limit.max} requests per {rate_limit.time_period}s")
        print(f"Confidence: {rate_limit.confidence:.2f}")
    
    return responses

# Run the analysis
responses = asyncio.run(analyze_endpoints())
```

### Context Manager Pattern

SmartSurgeClient supports context manager for proper resource cleanup:

```python
import asyncio
from smartsurge import SmartSurgeClient

async def context_manager_example():
    """Using context manager with async operations"""
    async with SmartSurgeClient() as client:
        # Client is properly initialized
        response = await client.async_get("https://api.example.com/data")
        data = response.json()
        
        # Make multiple requests
        tasks = [
            client.async_get(f"https://api.example.com/items/{i}")
            for i in range(10)
        ]
        responses = await asyncio.gather(*tasks)
        
        return data
    # Client is automatically cleaned up

result = asyncio.run(context_manager_example())
```

## Rate Limit Management

SmartSurge provides sophisticated rate limit management with multiple priority levels and detection methods.

### Manual Rate Limit Setting

Set explicit rate limits for specific endpoints:

```python
from smartsurge import SmartSurgeClient

client = SmartSurgeClient()

# Set rate limit for specific endpoint and method
client.set_rate_limit(
    endpoint="https://api.example.com/users",
    method="GET",
    max_requests=100,
    time_period=60  # 100 requests per 60 seconds
)

# Set rate limit for all methods on an endpoint
client.set_rate_limit(
    endpoint="https://api.example.com/products",
    method="*",  # Applies to all HTTP methods
    max_requests=500,
    time_period=3600  # 500 requests per hour
)

# Make requests - SmartSurge will respect the limits
for i in range(150):
    response = client.get(f"https://api.example.com/users/{i}")
    print(f"Request {i+1}: Status {response.status_code}")
```

### Rate Limit Priority System

SmartSurge uses a three-tier priority system:

1. **User-defined limits** (highest priority)
2. **Server-provided limits** (from response headers)
3. **HMM-estimated limits** (automatic detection)

```python
# Check current rate limit for an endpoint
rate_limit = client.get_rate_limit(
    endpoint="https://api.example.com/users",
    method="GET"
)

if rate_limit:
    print(f"Rate limit: {rate_limit.max} requests per {rate_limit.time_period}s")
    print(f"Source: {rate_limit.source}")  # 'user', 'header', or 'hmm'
    print(f"Confidence: {rate_limit.confidence:.2f}")

# List all known rate limits
all_limits = client.list_rate_limits()
for key, limit in all_limits.items():
    print(f"{key}: {limit.max}/{limit.time_period}s (source: {limit.source})")

# Clear a specific rate limit
client.clear_rate_limit(
    endpoint="https://api.example.com/users",
    method="GET"
)
```

### Server Header Detection

SmartSurge automatically extracts rate limits from standard headers:

```python
# Headers automatically detected:
# - X-RateLimit-Limit
# - X-RateLimit-Remaining
# - X-RateLimit-Reset
# - RateLimit-Limit (draft-ietf-httpapi-ratelimit)
# - Retry-After (for 429 responses)

response = client.get("https://api.example.com/data")

# Access the extracted rate limit info
if response.headers.get('X-RateLimit-Limit'):
    print(f"Server limit: {response.headers['X-RateLimit-Limit']}")
    print(f"Remaining: {response.headers.get('X-RateLimit-Remaining', 'unknown')}")
    print(f"Reset: {response.headers.get('X-RateLimit-Reset', 'unknown')}")
```

## Custom Configuration

Fine-tune SmartSurge's behavior with comprehensive configuration options.

### Client Configuration

```python
from smartsurge import SmartSurgeClient, ClientConfig

config = ClientConfig(
    # Base settings
    base_url="https://api.example.com",  # Optional base URL for all requests
    timeout=30.0,                         # Request timeout in seconds
    
    # Retry configuration
    max_retries=3,                        # Maximum retry attempts
    backoff_factor=2.0,                   # Exponential backoff multiplier
    
    # SSL and security
    verify_ssl=True,                      # SSL certificate verification
    
    # Custom headers
    user_agent="MyApp/1.0 (SmartSurge)", # Custom user agent
    
    # Connection pooling (for underlying requests/aiohttp)
    pool_connections=10,                  # Connection pool size
    pool_maxsize=20,                      # Maximum pool size
    
    # HMM parameters
    min_time_period=1,                    # Minimum rate limit period to consider
    max_time_period=3600,                 # Maximum rate limit period (1 hour)
    confidence_threshold=0.85,            # HMM confidence threshold
    n_bootstrap=100                       # Bootstrap iterations for confidence
)

client = SmartSurgeClient(config=config)
```

### Environment-Based Configuration

Load configuration from environment variables:

```python
import os
from smartsurge import SmartSurgeClient, ClientConfig

def create_client_from_env():
    """Create client with environment-based configuration"""
    config = ClientConfig(
        base_url=os.getenv("API_BASE_URL"),
        timeout=float(os.getenv("API_TIMEOUT", "30")),
        max_retries=int(os.getenv("API_MAX_RETRIES", "3")),
        verify_ssl=os.getenv("API_VERIFY_SSL", "true").lower() == "true",
        user_agent=os.getenv("API_USER_AGENT", "SmartSurge/1.0")
    )
    
    return SmartSurgeClient(config=config)

# Use environment configuration
client = create_client_from_env()
```

## HMM-Based Detection

SmartSurge uses a sophisticated Hidden Markov Model to automatically detect rate limits when they're not explicitly provided.

### Understanding the HMM System

The HMM models three states:
1. **Normal**: Regular operation, no rate limiting
2. **Approaching Limit**: Nearing rate limit threshold
3. **Rate Limited**: Actively being rate limited

```python
from smartsurge import SmartSurgeClient

client = SmartSurgeClient()

# Make several requests to build history
for i in range(20):
    response = client.get(f"https://api.example.com/data/{i}")
    print(f"Request {i+1}: Status {response.status_code}")

# Check if HMM has detected a rate limit
history = client._get_request_history(
    endpoint="https://api.example.com/data",
    method="GET"
)

if history:
    rate_limit = history.estimate_rate_limit()
    if rate_limit:
        print(f"\nHMM detected rate limit: {rate_limit.max} per {rate_limit.time_period}s")
        print(f"Confidence: {rate_limit.confidence:.2%}")
        print(f"Source: {rate_limit.source}")
```

### HMM Configuration

Fine-tune the HMM parameters for your use case:

```python
from smartsurge import SmartSurgeClient, ClientConfig

config = ClientConfig(
    # HMM time period search range
    min_time_period=1,        # Minimum period to consider (seconds)
    max_time_period=3600,     # Maximum period to consider (1 hour)
    
    # Confidence threshold for rate limit detection
    confidence_threshold=0.85, # Require 85% confidence
    
    # Bootstrap iterations for confidence calculation
    n_bootstrap=100           # More iterations = more accurate confidence
)

client = SmartSurgeClient(config=config)
```

### Accessing HMM Results

```python
# Get request history with HMM analysis
client = SmartSurgeClient()

# Make requests
for i in range(50):
    client.get("https://api.example.com/users")

# Access history and rate limit estimation
history = client._get_request_history(
    endpoint="https://api.example.com/users",
    method="GET"
)

# Analyze the history
print(f"Total requests tracked: {len(history.requests)}")
print(f"Successful requests: {history.successes}")
print(f"Failed requests (429s): {history.failures}")

# Get HMM's rate limit estimation
estimated_limit = history.estimate_rate_limit()
if estimated_limit:
    print(f"\nEstimated rate limit:")
    print(f"  Max requests: {estimated_limit.max}")
    print(f"  Time period: {estimated_limit.time_period} seconds")
    print(f"  Confidence: {estimated_limit.confidence:.2%}")
    print(f"  Source: {estimated_limit.source}")
else:
    print("\nNo rate limit detected yet (insufficient data or confidence)")
```

### Working with Request History

```python
# Return history with requests
response, history = client.get(
    "https://api.example.com/data",
    return_history=True
)

# Analyze the returned history
print(f"Request successful: {response.status_code == 200}")
print(f"History contains {len(history.requests)} requests")

# Check for patterns
consecutive_429s = 0
for req in history.requests:
    if req.response_code == 429:
        consecutive_429s += 1
    else:
        if consecutive_429s > 0:
            print(f"Found {consecutive_429s} consecutive 429s")
        consecutive_429s = 0
```

## Resumable Streaming

SmartSurge provides built-in support for resumable downloads through its streaming classes.

### Basic Streaming Usage

```python
from smartsurge import SmartSurgeClient

client = SmartSurgeClient()

# Stream a large file
response = client.stream_request(
    method="GET",
    url="https://example.com/large-file.zip",
    output_file="downloaded-file.zip"
)

print(f"Downloaded: {response['downloaded_bytes']} bytes")
print(f"Status: {response['status']}")
```

### Resumable Downloads with JSONStreamingRequest

For JSON API responses that support range requests:

```python
from smartsurge import SmartSurgeClient, JSONStreamingRequest
from smartsurge.streaming import StreamingState
import json

# Create a streaming request
streaming_request = JSONStreamingRequest(
    url="https://api.example.com/large-dataset.json",
    method="GET",
    headers={"Accept": "application/json"},
    chunk_size=1024 * 1024  # 1MB chunks
)

client = SmartSurgeClient()

# Start or resume download
state_file = "download_state.json"
try:
    # Load previous state if exists
    with open(state_file, 'r') as f:
        state_dict = json.load(f)
        streaming_request.state = StreamingState(**state_dict)
        print(f"Resuming download from byte {streaming_request.state.downloaded_bytes}")
except FileNotFoundError:
    print("Starting new download")

# Perform the download
result = client.stream_request(
    streaming_request=streaming_request,
    output_file="large_dataset.json"
)

# Save state for potential resume
with open(state_file, 'w') as f:
    json.dump(streaming_request.state.dict(), f)

if result['status'] == 'completed':
    print("Download completed successfully")
    # Clean up state file
    import os
    os.remove(state_file)
else:
    print(f"Download status: {result['status']}")
    print(f"Downloaded: {result['downloaded_bytes']} bytes")
```

### Custom Streaming Implementation

Create custom streaming requests for specific protocols:

```python
from smartsurge.streaming import AbstractStreamingRequest, StreamingState
from typing import Dict, Optional
import time

class ChunkedAPIRequest(AbstractStreamingRequest):
    """Custom streaming for APIs that provide data in chunks"""
    
    def __init__(self, url: str, chunk_endpoint: str, **kwargs):
        super().__init__(url=url, **kwargs)
        self.chunk_endpoint = chunk_endpoint
        self.current_chunk = 0
    
    def prepare_request(self) -> Dict:
        """Prepare request for next chunk"""
        # Request specific chunk
        chunk_url = f"{self.url}/{self.chunk_endpoint}/{self.current_chunk}"
        
        request_dict = {
            'method': self.method,
            'url': chunk_url,
            'headers': self.get_headers(),
        }
        
        # Add any additional parameters
        if self.params:
            request_dict['params'] = self.params
            
        return request_dict
    
    def update_state(self, response) -> None:
        """Update state after receiving chunk"""
        super().update_state(response)
        
        # Move to next chunk
        self.current_chunk += 1
        
        # Check if more chunks available
        if response.headers.get('X-More-Chunks') == 'false':
            self.state.completed = True
    
    def is_resumable(self, response) -> bool:
        """Check if this API supports resume"""
        return response.headers.get('X-Supports-Resume', 'false') == 'true'

# Use custom streaming
client = SmartSurgeClient()
custom_request = ChunkedAPIRequest(
    url="https://api.example.com",
    chunk_endpoint="data/chunks",
    method="GET"
)

result = client.stream_request(
    streaming_request=custom_request,
    output_file="chunked_data.json"
)
```

### Streaming with Progress Tracking

```python
from smartsurge import SmartSurgeClient
import sys

def download_with_progress(url: str, output_file: str):
    """Download file with progress bar"""
    client = SmartSurgeClient()
    
    # First, get file size
    response = client.head(url)
    total_size = int(response.headers.get('Content-Length', 0))
    
    # Stream download
    downloaded = 0
    with open(output_file, 'wb') as f:
        response = client.get(url, stream=True)
        
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                
                # Update progress
                if total_size > 0:
                    progress = downloaded / total_size
                    bar_length = 40
                    filled = int(bar_length * progress)
                    bar = '█' * filled + '-' * (bar_length - filled)
                    sys.stdout.write(f'\r[{bar}] {progress:.1%}')
                    sys.stdout.flush()
    
    print(f"\nDownload complete: {output_file}")

# Download with progress
download_with_progress(
    "https://example.com/large-file.zip",
    "output.zip"
)
```

## Request History and Analytics

SmartSurge maintains detailed request history for analytics and rate limit detection.

### Accessing Request History

```python
from smartsurge import SmartSurgeClient
from smartsurge.utilities import merge_histories

client = SmartSurgeClient()

# Make some requests
for i in range(20):
    client.get(f"https://api.example.com/items/{i}")

# Get history for specific endpoint
history = client._get_request_history(
    endpoint="https://api.example.com/items",
    method="GET"
)

# Analyze the history
print(f"Total requests: {len(history.requests)}")
print(f"Success rate: {history.successes / len(history.requests):.2%}")
print(f"Average response time: {history.avg_response_time:.3f}s")

# Get all histories
all_histories = client._request_histories
for key, hist in all_histories.items():
    print(f"\n{key}:")
    print(f"  Requests: {len(hist.requests)}")
    print(f"  Success rate: {hist.success_rate:.2%}")
    if hist.estimate_rate_limit():
        limit = hist.estimate_rate_limit()
        print(f"  Rate limit: {limit.max}/{limit.time_period}s")
```

### Merging Request Histories

Combine histories from multiple clients or sessions:

```python
from smartsurge import SmartSurgeClient
from smartsurge.utilities import merge_histories

# Create multiple clients
client1 = SmartSurgeClient()
client2 = SmartSurgeClient()

# Make requests with different clients
for i in range(10):
    client1.get(f"https://api.example.com/users/{i}")
    client2.get(f"https://api.example.com/users/{i+10}")

# Get histories
history1 = client1._get_request_history(
    endpoint="https://api.example.com/users",
    method="GET"
)
history2 = client2._get_request_history(
    endpoint="https://api.example.com/users",
    method="GET"
)

# Merge histories
merged = merge_histories([history1, history2])
print(f"Merged history contains {len(merged.requests)} requests")

# Estimate rate limit from combined data
if merged.estimate_rate_limit():
    limit = merged.estimate_rate_limit()
    print(f"Combined rate limit estimate: {limit.max}/{limit.time_period}s")
```

### Request History Analytics

```python
from datetime import datetime, timedelta
from collections import Counter

def analyze_request_patterns(client: SmartSurgeClient, endpoint: str):
    """Analyze request patterns for an endpoint"""
    history = client._get_request_history(endpoint, "GET")
    
    if not history.requests:
        print("No request history available")
        return
    
    # Time-based analysis
    requests = history.requests
    first_request = min(req.timestamp for req in requests)
    last_request = max(req.timestamp for req in requests)
    duration = last_request - first_request
    
    print(f"Analysis for {endpoint}:")
    print(f"Time span: {duration:.1f} seconds")
    print(f"Total requests: {len(requests)}")
    print(f"Average rate: {len(requests) / duration:.2f} req/s")
    
    # Status code distribution
    status_codes = Counter(req.response_code for req in requests)
    print("\nStatus code distribution:")
    for code, count in status_codes.most_common():
        print(f"  {code}: {count} ({count/len(requests):.1%})")
    
    # Response time analysis
    response_times = [req.response_time for req in requests if req.response_time]
    if response_times:
        print(f"\nResponse times:")
        print(f"  Min: {min(response_times):.3f}s")
        print(f"  Max: {max(response_times):.3f}s")
        print(f"  Avg: {sum(response_times)/len(response_times):.3f}s")
    
    # Burst detection
    print("\nBurst analysis:")
    for window in [1, 5, 10, 60]:  # Different time windows
        bursts = []
        for i in range(len(requests)):
            count = sum(1 for req in requests 
                       if requests[i].timestamp - req.timestamp <= window)
            bursts.append(count)
        max_burst = max(bursts)
        print(f"  Max requests in {window}s window: {max_burst}")

# Analyze patterns
client = SmartSurgeClient()
# Make requests...
analyze_request_patterns(client, "https://api.example.com/data")
```

## Performance Optimization

Tips for maximizing SmartSurge performance in production environments.

### Connection Pooling

Configure connection pools for high-throughput scenarios:

```python
from smartsurge import SmartSurgeClient, ClientConfig
import concurrent.futures

# Optimized configuration for parallel requests
config = ClientConfig(
    # Connection pool settings
    pool_connections=50,      # Number of connection pools
    pool_maxsize=100,         # Max connections per pool
    
    # Timeouts for fast APIs
    timeout=10.0,             # Total request timeout
    
    # Retry settings
    max_retries=2,            # Limit retries for speed
    backoff_factor=1.5        # Faster backoff
)

client = SmartSurgeClient(config=config)

def fetch_many_urls(urls: list, max_workers: int = 50):
    """Fetch multiple URLs in parallel"""
    def fetch_one(url):
        try:
            response = client.get(url)
            return {'url': url, 'status': response.status_code}
        except Exception as e:
            return {'url': url, 'error': str(e)}
    
    # Use thread pool for parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(fetch_one, urls))
    
    successful = sum(1 for r in results if 'status' in r)
    print(f"Successfully fetched {successful}/{len(urls)} URLs")
    
    return results

# Fetch many URLs efficiently
urls = [f"https://api.example.com/item/{i}" for i in range(100)]
results = fetch_many_urls(urls)
```

### Caching with SmartSurge

Implement caching to reduce API calls:

```python
from smartsurge import SmartSurgeClient
from functools import lru_cache
import hashlib
import json

class CachedSmartSurgeClient(SmartSurgeClient):
    """SmartSurge client with built-in caching"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}
    
    def _cache_key(self, method: str, url: str, **kwargs) -> str:
        """Generate cache key from request parameters"""
        # Create a stable key from request details
        key_parts = [method, url]
        if params := kwargs.get('params'):
            key_parts.append(json.dumps(params, sort_keys=True))
        if json_data := kwargs.get('json'):
            key_parts.append(json.dumps(json_data, sort_keys=True))
        
        key_str = '|'.join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def cached_get(self, url: str, cache_time: int = 300, **kwargs):
        """GET request with caching (default 5 minutes)"""
        import time
        
        cache_key = self._cache_key('GET', url, **kwargs)
        
        # Check cache
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < cache_time:
                return cached_data
        
        # Make request
        response = self.get(url, **kwargs)
        
        # Cache successful responses
        if response.status_code == 200:
            self._cache[cache_key] = (response, time.time())
        
        return response

# Use cached client
client = CachedSmartSurgeClient()

# First call makes the request
data1 = client.cached_get("https://api.example.com/data")

# Second call returns from cache (instant)
data2 = client.cached_get("https://api.example.com/data")
```

### Batch Request Optimization

Optimize batch requests with SmartSurge:

```python
from smartsurge import SmartSurgeClient
from smartsurge.utilities import async_request_with_history
import asyncio
import time

async def optimized_batch_fetch(endpoints: list):
    """Fetch multiple endpoints with optimization"""
    client = SmartSurgeClient()
    
    # Group requests by base URL for better rate limit handling
    grouped_requests = {}
    for endpoint in endpoints:
        base_url = endpoint.split('?')[0]  # Remove query params
        if base_url not in grouped_requests:
            grouped_requests[base_url] = []
        grouped_requests[base_url].append({
            'method': 'GET',
            'url': endpoint
        })
    
    all_responses = []
    
    # Process each group with rate limit awareness
    for base_url, requests in grouped_requests.items():
        print(f"Processing {len(requests)} requests for {base_url}")
        
        # Check current rate limit status
        rate_limit = client.get_rate_limit(base_url, "GET")
        if rate_limit:
            print(f"  Rate limit: {rate_limit.max}/{rate_limit.time_period}s")
        
        # Make requests with history tracking
        responses, history = await async_request_with_history(client, requests)
        all_responses.extend(responses)
        
        # Analyze results
        success_rate = sum(1 for r in responses if r.status_code == 200) / len(responses)
        print(f"  Success rate: {success_rate:.1%}")
        
        # If we're hitting rate limits, slow down
        if history.failures > 0:
            wait_time = min(5, history.failures * 0.5)
            print(f"  Rate limit detected, waiting {wait_time}s")
            await asyncio.sleep(wait_time)
    
    return all_responses

# Fetch many endpoints efficiently
endpoints = [
    f"https://api.example.com/users/{i}" for i in range(50)
] + [
    f"https://api.example.com/products/{i}" for i in range(50)
]

responses = asyncio.run(optimized_batch_fetch(endpoints))
print(f"\nFetched {len(responses)} total responses")
```

### Memory-Efficient Streaming

Handle large responses efficiently:

```python
from smartsurge import SmartSurgeClient
import json

def process_large_json_stream(url: str, chunk_size: int = 1024*1024):
    """Process large JSON responses in chunks"""
    client = SmartSurgeClient()
    
    # Stream the response
    with client.get(url, stream=True) as response:
        response.raise_for_status()
        
        # Process JSON array items as they arrive
        buffer = ""
        in_string = False
        bracket_count = 0
        current_item = ""
        
        for chunk in response.iter_content(chunk_size=chunk_size, decode_unicode=True):
            if not chunk:
                continue
                
            buffer += chunk
            
            # Simple JSON array parser
            for char in buffer:
                if char == '"' and buffer[len(current_item)-1] != '\\':
                    in_string = not in_string
                
                if not in_string:
                    if char == '{':
                        bracket_count += 1
                    elif char == '}':
                        bracket_count -= 1
                
                current_item += char
                
                # Complete JSON object found
                if bracket_count == 0 and current_item.strip().endswith('}'):
                    try:
                        item = json.loads(current_item.strip().rstrip(','))
                        yield item  # Process item without loading entire response
                    except json.JSONDecodeError:
                        pass
                    current_item = ""
            
            buffer = current_item  # Keep unparsed portion

# Process large JSON responses efficiently
for item in process_large_json_stream("https://api.example.com/large-dataset"):
    # Process each item as it arrives
    print(f"Processing item: {item.get('id', 'unknown')}")
```

## Error Handling Patterns

SmartSurge provides comprehensive exception handling for robust applications.

### Exception Hierarchy

```python
from smartsurge import SmartSurgeClient
from smartsurge.exceptions import (
    SmartSurgeException,      # Base exception
    RateLimitExceeded,        # 429 responses
    StreamingError,           # Streaming issues
    ResumeError,              # Resume failures
    ValidationError,          # Input validation
    ConfigurationError,       # Config problems
    ConnectionError,          # Network issues
    TimeoutError,             # Request timeouts
    ServerError,              # 5xx responses
    ClientError               # 4xx responses
)
```

### Comprehensive Error Handling

```python
from smartsurge import SmartSurgeClient
from smartsurge.exceptions import *
import logging
import time

logger = logging.getLogger(__name__)

def robust_api_call(client: SmartSurgeClient, url: str, max_attempts: int = 3):
    """Make API call with comprehensive error handling"""
    for attempt in range(max_attempts):
        try:
            response = client.get(url)
            return response.json()
            
        except RateLimitExceeded as e:
            # Handle rate limiting
            logger.warning(f"Rate limit hit: {e.message}")
            if e.retry_after:
                logger.info(f"Waiting {e.retry_after} seconds before retry")
                time.sleep(e.retry_after)
            else:
                # Use exponential backoff
                wait_time = 2 ** attempt
                logger.info(f"Waiting {wait_time} seconds (exponential backoff)")
                time.sleep(wait_time)
                
        except TimeoutError as e:
            # Handle timeouts
            logger.error(f"Request timed out: {e.message}")
            if attempt < max_attempts - 1:
                logger.info("Retrying with longer timeout")
                # Could increase timeout in client config here
            else:
                raise
                
        except ConnectionError as e:
            # Handle connection issues
            logger.error(f"Connection error: {e.message}")
            if attempt < max_attempts - 1:
                wait_time = 5 * (attempt + 1)
                logger.info(f"Waiting {wait_time} seconds before retry")
                time.sleep(wait_time)
            else:
                raise
                
        except ServerError as e:
            # Handle server errors (5xx)
            logger.error(f"Server error {e.status_code}: {e.message}")
            if e.status_code == 503:  # Service Unavailable
                time.sleep(10)  # Wait before retry
            else:
                raise
                
        except ClientError as e:
            # Handle client errors (4xx)
            logger.error(f"Client error {e.status_code}: {e.message}")
            if e.status_code == 401:
                # Could refresh authentication here
                pass
            raise  # Client errors usually shouldn't be retried
            
        except ValidationError as e:
            # Handle validation errors
            logger.error(f"Validation error: {e.message}")
            logger.error(f"Context: {e.context}")
            raise  # These indicate programming errors
            
        except SmartSurgeException as e:
            # Catch any other SmartSurge exceptions
            logger.error(f"SmartSurge error: {e.message}")
            raise
            
    raise Exception(f"Failed after {max_attempts} attempts")

# Use robust error handling
client = SmartSurgeClient()
try:
    data = robust_api_call(client, "https://api.example.com/data")
    print(f"Successfully retrieved {len(data)} items")
except Exception as e:
    print(f"Failed to retrieve data: {e}")
```

### Custom Error Recovery

```python
from smartsurge import SmartSurgeClient
from smartsurge.exceptions import RateLimitExceeded, ConnectionError
from typing import Optional, Dict, Any

class RecoverableClient:
    """Client with custom error recovery strategies"""
    
    def __init__(self, primary_url: str, backup_url: Optional[str] = None):
        self.client = SmartSurgeClient()
        self.primary_url = primary_url
        self.backup_url = backup_url
        self.cache = {}
    
    def get_with_fallback(self, endpoint: str, use_cache: bool = True) -> Dict[str, Any]:
        """GET request with multiple fallback strategies"""
        full_url = f"{self.primary_url}/{endpoint}"
        
        try:
            # Try primary endpoint
            response = self.client.get(full_url)
            data = response.json()
            
            # Update cache on success
            if use_cache:
                self.cache[endpoint] = data
            
            return data
            
        except RateLimitExceeded as e:
            print(f"Rate limit exceeded for primary API")
            
            # Try backup API if available
            if self.backup_url:
                try:
                    backup_url = f"{self.backup_url}/{endpoint}"
                    response = self.client.get(backup_url)
                    return response.json()
                except Exception as backup_error:
                    print(f"Backup API also failed: {backup_error}")
            
            # Fall back to cache
            if use_cache and endpoint in self.cache:
                print("Returning cached data")
                return self.cache[endpoint]
            
            raise
            
        except ConnectionError as e:
            print(f"Connection error: {e.message}")
            
            # Return cached data if available
            if use_cache and endpoint in self.cache:
                print("Network unavailable, using cached data")
                return self.cache[endpoint]
            
            raise
    
    def get_with_retry_strategies(self, endpoint: str) -> Dict[str, Any]:
        """Try different strategies for resilient data access"""
        strategies = [
            ("primary", lambda: self.client.get(f"{self.primary_url}/{endpoint}")),
            ("backup", lambda: self.client.get(f"{self.backup_url}/{endpoint}") if self.backup_url else None),
            ("cache", lambda: self.cache.get(endpoint)),
        ]
        
        errors = []
        for strategy_name, strategy_func in strategies:
            try:
                if strategy_func is None:
                    continue
                    
                result = strategy_func()
                if result is None:
                    continue
                    
                if hasattr(result, 'json'):
                    return result.json()
                else:
                    return result
                    
            except Exception as e:
                errors.append(f"{strategy_name}: {str(e)}")
                continue
        
        # All strategies failed
        raise Exception(f"All strategies failed: {'; '.join(errors)}")

# Use recoverable client
client = RecoverableClient(
    primary_url="https://api.example.com",
    backup_url="https://backup-api.example.com"
)

try:
    data = client.get_with_fallback("users/123")
    print(f"Retrieved user: {data}")
except Exception as e:
    print(f"Failed to retrieve user: {e}")
```

### Streaming Error Recovery

```python
from smartsurge import SmartSurgeClient
from smartsurge.exceptions import StreamingError, ResumeError
import time

def download_with_resume(client: SmartSurgeClient, url: str, 
                        output_file: str, max_retries: int = 3):
    """Download file with automatic resume on failure"""
    for attempt in range(max_retries):
        try:
            result = client.stream_request(
                method="GET",
                url=url,
                output_file=output_file
            )
            
            if result['status'] == 'completed':
                print(f"Download completed: {result['downloaded_bytes']} bytes")
                return result
            else:
                print(f"Download incomplete: {result['status']}")
                
        except StreamingError as e:
            print(f"Streaming error on attempt {attempt + 1}: {e.message}")
            
            if attempt < max_retries - 1:
                # Wait before retry
                wait_time = 5 * (attempt + 1)
                print(f"Waiting {wait_time} seconds before retry")
                time.sleep(wait_time)
            else:
                raise
                
        except ResumeError as e:
            print(f"Resume not supported: {e.message}")
            # Could implement fallback to full download
            raise
    
    raise Exception(f"Download failed after {max_retries} attempts")

# Download with automatic resume
client = SmartSurgeClient()
try:
    result = download_with_resume(
        client,
        "https://example.com/large-file.zip",
        "output.zip"
    )
except Exception as e:
    print(f"Download failed: {e}")
```

### Logging and Debugging

SmartSurge provides comprehensive logging for debugging and monitoring.

```python
from smartsurge import SmartSurgeClient, configure_logging
from smartsurge.utilities import log_context, SmartSurgeTimer
import logging

# Configure SmartSurge logging
configure_logging(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Get logger for your application
logger = logging.getLogger(__name__)

# Use logging context
with log_context("batch_processing", user_id="12345"):
    client = SmartSurgeClient()
    
    # All logs within this context will include the context info
    for i in range(10):
        with SmartSurgeTimer() as timer:
            response = client.get(f"https://api.example.com/items/{i}")
            logger.info(f"Fetched item {i} in {timer.elapsed:.3f}s")

# Debug rate limit detection
client = SmartSurgeClient()

# Enable debug logging for HMM
hmm_logger = logging.getLogger("smartsurge.hmm")
hmm_logger.setLevel(logging.DEBUG)

# Make requests and observe HMM decisions
for i in range(20):
    response = client.get("https://api.example.com/data")
    logger.debug(f"Request {i+1}: Status {response.status_code}")
```

### Monitoring Rate Limit Behavior

```python
from smartsurge import SmartSurgeClient
import time
import json

def monitor_rate_limits(client: SmartSurgeClient, duration: int = 60):
    """Monitor rate limit behavior over time"""
    start_time = time.time()
    observations = []
    
    while time.time() - start_time < duration:
        # Make a request
        try:
            response = client.get("https://api.example.com/data")
            status = response.status_code
        except Exception as e:
            status = None
        
        # Get current rate limit info
        rate_limit = client.get_rate_limit(
            "https://api.example.com/data", "GET"
        )
        
        # Record observation
        observation = {
            'timestamp': time.time() - start_time,
            'status_code': status,
            'rate_limit': {
                'max': rate_limit.max if rate_limit else None,
                'period': rate_limit.time_period if rate_limit else None,
                'source': rate_limit.source if rate_limit else None,
                'confidence': rate_limit.confidence if rate_limit else None
            } if rate_limit else None
        }
        observations.append(observation)
        
        # Wait a bit before next request
        time.sleep(1)
    
    # Analyze results
    print(f"\nMonitoring complete. {len(observations)} observations.")
    
    # Find when rate limit was detected
    for i, obs in enumerate(observations):
        if obs['rate_limit'] and obs['rate_limit']['source'] == 'hmm':
            print(f"\nHMM detected rate limit at {obs['timestamp']:.1f}s:")
            print(f"  Limit: {obs['rate_limit']['max']}/{obs['rate_limit']['period']}s")
            print(f"  Confidence: {obs['rate_limit']['confidence']:.2%}")
            break
    
    # Save for analysis
    with open('rate_limit_observations.json', 'w') as f:
        json.dump(observations, f, indent=2)
    
    return observations

# Monitor rate limits
client = SmartSurgeClient()
observations = monitor_rate_limits(client, duration=120)
```

## Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from smartsurge import SmartSurgeClient
from typing import Dict, Any
import asyncio

app = FastAPI()

# Create a shared client instance
client = SmartSurgeClient()

@app.get("/proxy/{path:path}")
async def proxy_request(path: str) -> Dict[str, Any]:
    """Proxy requests through SmartSurge with rate limit protection"""
    try:
        # Use async request
        response = await client.async_get(f"https://api.example.com/{path}")
        
        # Get rate limit info
        rate_limit = client.get_rate_limit(
            f"https://api.example.com/{path}", "GET"
        )
        
        return {
            "data": response.json(),
            "rate_limit": {
                "detected": rate_limit is not None,
                "source": rate_limit.source if rate_limit else None,
                "limit": f"{rate_limit.max}/{rate_limit.time_period}s" if rate_limit else None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
async def batch_request(urls: list[str]) -> Dict[str, Any]:
    """Process multiple URLs with SmartSurge"""
    # Create async tasks
    tasks = [client.async_get(url) for url in urls]
    
    # Execute concurrently
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    results = []
    for url, response in zip(urls, responses):
        if isinstance(response, Exception):
            results.append({
                "url": url,
                "success": False,
                "error": str(response)
            })
        else:
            results.append({
                "url": url,
                "success": True,
                "status": response.status_code,
                "size": len(response.content)
            })
    
    return {"results": results, "total": len(results)}
```

### Flask Integration

```python
from flask import Flask, jsonify, request
from smartsurge import SmartSurgeClient, ClientConfig
import os

app = Flask(__name__)

# Configure SmartSurge from environment
config = ClientConfig(
    base_url=os.getenv("API_BASE_URL", "https://api.example.com"),
    timeout=int(os.getenv("API_TIMEOUT", "30")),
    max_retries=int(os.getenv("API_MAX_RETRIES", "3"))
)

client = SmartSurgeClient(config=config)

@app.route("/api/<path:endpoint>", methods=["GET", "POST", "PUT", "DELETE"])
def api_proxy(endpoint):
    """Proxy API requests with rate limit protection"""
    try:
        # Forward the request
        response = client.request(
            method=request.method,
            url=f"{config.base_url}/{endpoint}",
            params=request.args,
            json=request.get_json() if request.is_json else None,
            headers={
                k: v for k, v in request.headers 
                if k.lower() not in ['host', 'content-length']
            }
        )
        
        # Return proxied response
        return jsonify({
            "data": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
            "status": response.status_code,
            "rate_limit_detected": client.get_rate_limit(f"{config.base_url}/{endpoint}", request.method) is not None
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health_check():
    """Check API health through SmartSurge"""
    try:
        response = client.get(f"{config.base_url}/health")
        return jsonify({
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "api_status": response.status_code,
            "rate_limits": [
                {"endpoint": k, "limit": f"{v.max}/{v.time_period}s"}
                for k, v in client.list_rate_limits().items()
            ]
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 503
```

### Requests Session Replacement

```python
import requests
from smartsurge import SmartSurgeClient

# Replace requests.Session with SmartSurge
class SmartSurgeSession:
    """Drop-in replacement for requests.Session with rate limit protection"""
    
    def __init__(self):
        self.client = SmartSurgeClient()
        self.headers = {}
        self.cookies = requests.cookies.RequestsCookieJar()
    
    def get(self, url, **kwargs):
        kwargs.setdefault('headers', {}).update(self.headers)
        kwargs.setdefault('cookies', self.cookies)
        response = self.client.get(url, **kwargs)
        self.cookies.update(response.cookies)
        return response
    
    def post(self, url, **kwargs):
        kwargs.setdefault('headers', {}).update(self.headers)
        kwargs.setdefault('cookies', self.cookies)
        response = self.client.post(url, **kwargs)
        self.cookies.update(response.cookies)
        return response
    
    def put(self, url, **kwargs):
        kwargs.setdefault('headers', {}).update(self.headers)
        kwargs.setdefault('cookies', self.cookies)
        response = self.client.put(url, **kwargs)
        self.cookies.update(response.cookies)
        return response
    
    def delete(self, url, **kwargs):
        kwargs.setdefault('headers', {}).update(self.headers)
        kwargs.setdefault('cookies', self.cookies)
        response = self.client.delete(url, **kwargs)
        self.cookies.update(response.cookies)
        return response
    
    def close(self):
        # SmartSurge handles cleanup automatically
        pass

# Use as drop-in replacement
session = SmartSurgeSession()
session.headers['Authorization'] = 'Bearer token'

# Make requests with automatic rate limit handling
response = session.get('https://api.example.com/data')
data = response.json()

# Check if rate limits were detected
rate_limit = session.client.get_rate_limit(
    'https://api.example.com/data', 'GET'
)
if rate_limit:
    print(f"Rate limit detected: {rate_limit.max} per {rate_limit.time_period}s")
```

### Scrapy Integration

```python
# smartsurge_middleware.py
from scrapy import signals
from scrapy.downloadermiddlewares.retry import RetryMiddleware
from smartsurge import SmartSurgeClient
from smartsurge.exceptions import RateLimitExceeded

class SmartSurgeDownloaderMiddleware:
    """Scrapy middleware using SmartSurge for rate limit handling"""
    
    def __init__(self, settings):
        self.client = SmartSurgeClient(
            config={
                'max_retries': settings.getint('SMARTSURGE_MAX_RETRIES', 3),
                'timeout': settings.getfloat('SMARTSURGE_TIMEOUT', 30)
            }
        )
    
    @classmethod
    def from_crawler(cls, crawler):
        middleware = cls(crawler.settings)
        crawler.signals.connect(
            middleware.spider_opened, signal=signals.spider_opened
        )
        return middleware
    
    def process_request(self, request, spider):
        """Process request through SmartSurge"""
        try:
            # Make request through SmartSurge
            response = self.client.request(
                method=request.method,
                url=request.url,
                headers=request.headers,
                body=request.body
            )
            
            # Convert to Scrapy response
            from scrapy.http import HtmlResponse
            return HtmlResponse(
                url=request.url,
                status=response.status_code,
                headers=response.headers,
                body=response.content,
                encoding='utf-8',
                request=request
            )
            
        except RateLimitExceeded as e:
            # Let Scrapy retry with delay
            spider.logger.warning(f"Rate limit hit: {e.message}")
            request.meta['retry_times'] = request.meta.get('retry_times', 0) + 1
            request.meta['download_delay'] = e.retry_after or 5
            raise
    
    def spider_opened(self, spider):
        spider.logger.info('SmartSurge middleware enabled')

# settings.py
DOWNLOADER_MIDDLEWARES = {
    'myproject.middlewares.SmartSurgeDownloaderMiddleware': 543,
}

SMARTSURGE_MAX_RETRIES = 5
SMARTSURGE_TIMEOUT = 30
```

## Best Practices

### Configuration Tips

1. **Set Appropriate Timeouts**: Balance between reliability and performance
   ```python
   config = ClientConfig(
       timeout=30.0,  # Reasonable default
       max_retries=3,  # Don't retry too many times
       backoff_factor=2.0  # Exponential backoff
   )
   ```

2. **Tune HMM Parameters**: Adjust based on your API's behavior
   ```python
   config = ClientConfig(
       min_time_period=1,  # For APIs with second-level limits
       max_time_period=3600,  # For APIs with hourly limits
       confidence_threshold=0.85,  # Balance accuracy vs speed
       n_bootstrap=100  # More iterations = better confidence
   )
   ```

3. **Use Manual Rate Limits**: When you know the limits
   ```python
   # Set known rate limits to avoid detection time
   client.set_rate_limit(
       endpoint="https://api.example.com/v1",
       method="*",
       max_requests=1000,
       time_period=3600
   )
   ```

### Performance Optimization

1. **Reuse Client Instances**: Don't create new clients for each request
2. **Use Connection Pooling**: Configure appropriate pool sizes
3. **Implement Caching**: Reduce unnecessary API calls
4. **Batch Requests**: Use async operations for concurrent requests

### Error Handling

1. **Handle Specific Exceptions**: Don't catch generic exceptions
2. **Implement Retry Logic**: Use exponential backoff
3. **Log Errors Appropriately**: Include context and request details
4. **Monitor Rate Limits**: Track detection and compliance

### Testing

1. **Use Mock Servers**: Test rate limit scenarios with benchmarks
   ```python
   if smartsurge.has_benchmarks():
       from smartsurge import create_benchmark_server
       server = create_benchmark_server()
   ```

2. **Test Edge Cases**: Network failures, timeouts, malformed responses
3. **Verify Rate Limit Compliance**: Ensure your app respects limits

## Summary

SmartSurge provides powerful features for handling rate-limited APIs:

- **Automatic Detection**: HMM-based rate limit discovery
- **Multi-Priority System**: User → Server → HMM priority
- **Async Support**: High-performance concurrent operations
- **Resumable Streaming**: Reliable large file transfers
- **Comprehensive Error Handling**: Robust exception hierarchy
- **Request History**: Analytics and pattern detection
- **Flexible Configuration**: Extensive customization options

For more information:
- See [API Reference](api/index.md) for detailed documentation
- Check [Examples](examples.md) for common use cases
- Use [Benchmarks](benchmark_usage.md) to test performance