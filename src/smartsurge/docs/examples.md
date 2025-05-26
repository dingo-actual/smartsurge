# SmartSurge Examples

This page provides practical examples of how to use SmartSurge in various scenarios. All examples have been cross-referenced with the codebase to ensure accuracy.

## Basic Usage Examples

### Simple GET Request

```python
from smartsurge import SmartSurgeClient  # or Client (alias)

# Create a client
client = SmartSurgeClient()

# Make a GET request
response = client.get("https://api.example.com/data")

# Print the response
print(f"Status code: {response.status_code}")
print(f"Response data: {response.json()}")

# Get request with history
response, history = client.get("https://api.example.com/data", return_history=True)
print(f"Request history contains {len(history.requests)} requests")
```

### Using Configuration

```python
from smartsurge import SmartSurgeClient

# Create a client with custom configuration
client = SmartSurgeClient(
    base_url="https://api.example.com",
    timeout=(10.0, 30.0),  # (connect, read) timeouts
    max_retries=3,
    backoff_factor=0.3,
    verify_ssl=True,
    n_bootstrap=10  # Bootstrap iterations
)

# Now all requests will use the base URL
response = client.get("/users")  # Goes to https://api.example.com/users
```

### Context Manager Usage

```python
from smartsurge import SmartSurgeClient

# SmartSurgeClient automatically handles session management
client = SmartSurgeClient()

# Make multiple requests
response1 = client.get("https://api.example.com/users")
response2 = client.get("https://api.example.com/products")

# Use the responses
print(f"Users: {response1.json()}")
print(f"Products: {response2.json()}")
```

## Rate Limit Examples

### Setting Manual Rate Limits

```python
from smartsurge import SmartSurgeClient
import time

# Create a client
client = SmartSurgeClient()

# Set rate limit for a specific endpoint
client.set_rate_limit(
    endpoint="https://api.example.com/items",
    method="GET",
    max_requests=5,
    time_period=1.0  # 5 requests per second
)

# Make multiple requests - will automatically respect rate limits
start_time = time.time()
for i in range(10):
    response = client.get(f"https://api.example.com/items/{i}")
    print(f"Request {i+1}: Status {response.status_code}")

end_time = time.time()
print(f"Total time: {end_time - start_time:.2f} seconds")
print("Should take ~2 seconds for 10 requests at 5 req/s")
```

### Automatic Rate Limit Detection

```python
from smartsurge import SmartSurgeClient
import time

# Create a client with custom HMM configuration
client = SmartSurgeClient(
    min_time_period=1.0,       # Min period to search (seconds)
    max_time_period=3600.0,    # Max period to search (1 hour)
    refit_every=20             # Refit HMM every 20 requests
)

# Make multiple requests to learn the rate limits
for i in range(30):
    try:
        # Get response with history to track learning
        response, history = client.get(
            "https://api.example.com/data",
            return_history=True
        )
        
        print(f"Request {i+1}: Status {response.status_code}")
        
    except Exception as e:
        print(f"Request {i+1} failed: {e}")
        
    # Small delay between requests
    time.sleep(0.1)

# Check all detected rate limits
all_limits = client.list_rate_limits()
print(f"\nDetected {len(all_limits)} rate limits:")
for (endpoint, method), limit in all_limits.items():
    if limit:
        print(f"  {endpoint} {method}: {limit.max_requests}/{limit.time_period}s")
        print(f"    Source: {limit.source}")
```

### Handling Rate Limit Errors

```python
from smartsurge import SmartSurgeClient, RateLimitExceeded
import time

client = SmartSurgeClient(max_retries=3)

# SmartSurge automatically retries on 429, but you can handle exceptions
for i in range(20):
    try:
        response = client.get("https://api.example.com/rate-limited")
        print(f"Request {i+1}: Success")
        
    except RateLimitExceeded as e:
        # This only happens after all retries are exhausted
        print(f"Rate limit exceeded: {e.message}")
        if e.retry_after:
            print(f"Server suggested waiting for {e.retry_after} seconds")
            time.sleep(e.retry_after)
    except Exception as e:
        print(f"Other error: {e}")
```

## Streaming Examples

### Basic Streaming Download

```python
from smartsurge import SmartSurgeClient, JSONStreamingRequest

client = SmartSurgeClient()

# Create a streaming request
streaming_request = JSONStreamingRequest(
    url="https://example.com/large-file.json",
    method="GET",
    chunk_size=1024 * 1024  # 1MB chunks
)

# Stream download
result = client.stream_request(
    streaming_class=JSONStreamingRequest,
    endpoint="https://example.com/large-file.json",
    chunk_size=1024 * 1024
)

print(f"Download completed: {result}")
```

### Resumable Download with State Management

```python
from smartsurge import SmartSurgeClient, JSONStreamingRequest, StreamingState
import json
import os

client = SmartSurgeClient()

# State file for resume support
state_file = "download_state.json"
url = "https://api.example.com/large-dataset.json"

# Create streaming request
streaming_request = JSONStreamingRequest(
    url=url,
    method="GET",
    chunk_size=1024 * 1024  # 1MB chunks
)

# Load previous state if resuming
if os.path.exists(state_file):
    with open(state_file, 'r') as f:
        state_data = json.load(f)
        streaming_request.state = StreamingState(**state_data)
    print(f"Resuming download from byte {streaming_request.state.downloaded_bytes}")

try:
    # Perform the streaming download
    result = client.stream_request(
        streaming_class=type(streaming_request),
        endpoint=url,
        state_file=state_file,  # Automatically saves state
        chunk_size=1024 * 1024
    )
    
    print(f"Download completed! Downloaded bytes: {result}")
    
    # Clean up state file on success
    if os.path.exists(state_file):
        os.remove(state_file)
        
except Exception as e:
    print(f"Error during download: {e}")
    print("State saved. You can resume the download later.")
```

### Custom Streaming Implementation

```python
from smartsurge import SmartSurgeClient, AbstractStreamingRequest
from smartsurge.streaming import StreamingState
import csv
import io

class CSVStreamingRequest(AbstractStreamingRequest):
    """Custom streaming request for CSV data"""
    
    def __init__(self, url: str, **kwargs):
        super().__init__(url=url, **kwargs)
        self.rows = []
        self.headers = None
    
    def prepare_request(self) -> dict:
        """Prepare the request parameters"""
        return {
            'method': self.method,
            'url': self.url,
            'headers': self.get_headers(),
            'params': self.params,
            'stream': True
        }
    
    def process_response(self, response):
        """Process the streaming response"""
        # Read CSV data
        text_stream = io.StringIO(response.text)
        csv_reader = csv.DictReader(text_stream)
        
        self.headers = csv_reader.fieldnames
        self.rows = list(csv_reader)
        
        # Update state
        self.state.downloaded_bytes = len(response.content)
        self.state.completed = True
        
        return self.rows

# Use the custom streaming request
client = SmartSurgeClient()

result = client.stream_request(
    streaming_class=CSVStreamingRequest,
    endpoint="https://example.com/data.csv"
)

print(f"Downloaded CSV data: {result}")
```

## HTTP Methods Examples

### POST, PUT, DELETE Requests

```python
from smartsurge import SmartSurgeClient

client = SmartSurgeClient()

# POST request with JSON data
response = client.post(
    "https://api.example.com/users",
    json={"name": "John Doe", "email": "john@example.com"}
)
print(f"Created user: {response.json()}")

# PUT request to update
response = client.put(
    "https://api.example.com/users/123",
    json={"name": "John Smith"}
)
print(f"Updated user: {response.json()}")

# DELETE request
response = client.delete("https://api.example.com/users/123")
print(f"Delete status: {response.status_code}")

# PATCH request
response = client.patch(
    "https://api.example.com/users/123",
    json={"status": "active"}
)
print(f"Patched user: {response.json()}")
```

### Request with Headers and Parameters

```python
from smartsurge import SmartSurgeClient

client = SmartSurgeClient()

# Request with headers and query parameters
response = client.get(
    "https://api.example.com/search",
    headers={
        "Authorization": "Bearer your-token",
        "X-API-Version": "2.0"
    },
    params={
        "q": "python",
        "sort": "relevance",
        "limit": 10
    }
)

print(f"Search results: {response.json()}")
```

## Async Examples

### Basic Async Request

```python
import asyncio
from smartsurge import SmartSurgeClient

async def fetch_data():
    client = SmartSurgeClient()
    
    # Make an async GET request
    response = await client.async_get(
        "https://api.example.com/data",
        headers={"Authorization": "Bearer token"}
    )
    
    print(f"Status: {response.status}")
    print(f"Data: {await response.json()}")
    
    # Get with history
    response, history = await client.async_get(
        "https://api.example.com/data",
        return_history=True
    )
    print(f"Made {len(history.requests)} total requests")

# Run the async function
asyncio.run(fetch_data())
```

### Multiple Concurrent Requests

```python
import asyncio
from smartsurge import SmartSurgeClient, async_request_with_history

async def fetch_multiple():
    client = SmartSurgeClient()
    
    # Method 1: Individual async requests
    tasks = [
        client.async_get(f"https://api.example.com/users/{i}") 
        for i in range(1, 6)
    ]
    
    # Execute all requests concurrently
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            print(f"Request {i+1} failed: {response}")
        else:
            data = await response.json()
            print(f"User {i+1}: {data}")
    
    # Method 2: Using async_request_with_history utility
    requests = [
        {"method": "GET", "url": f"https://api.example.com/products/{i}"}
        for i in range(1, 6)
    ]
    
    responses, history = await async_request_with_history(client, requests)
    print(f"\nFetched {len(responses)} products")
    print(f"Total requests made: {len(history.requests)}")

# Run the async function
asyncio.run(fetch_multiple())
```

## Error Handling Examples

### Comprehensive Error Handling

```python
from smartsurge import SmartSurgeClient
from smartsurge import (
    RateLimitExceeded,
    ConnectionError,
    TimeoutError,
    ServerError,
    ClientError,
    ValidationError,
    StreamingError,
    ResumeError
)
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_api_call(client, url, max_retries=3):
    """Make API call with comprehensive error handling"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} for {url}")
            response = client.get(url)
            return response
            
        except RateLimitExceeded as e:
            # SmartSurge already handles retries, this is after exhaustion
            logger.warning(f"Rate limit exceeded: {e.message}")
            if e.retry_after:
                logger.info(f"Waiting {e.retry_after}s as suggested by server")
                time.sleep(e.retry_after)
            else:
                # Exponential backoff
                wait = min(60, 2 ** attempt)
                logger.info(f"Waiting {wait}s before retry")
                time.sleep(wait)
                
        except TimeoutError as e:
            logger.error(f"Request timed out: {e.message}")
            if attempt < max_retries - 1:
                time.sleep(5)  # Brief pause before retry
            else:
                raise
                
        except ConnectionError as e:
            logger.error(f"Connection error: {e.message}")
            if attempt < max_retries - 1:
                time.sleep(10)  # Longer pause for connection issues
            else:
                raise
                
        except ServerError as e:
            logger.error(f"Server error {e.status_code}: {e.message}")
            if e.status_code == 503:  # Service Unavailable
                time.sleep(30)  # Wait longer for service recovery
            else:
                raise  # Don't retry other 5xx errors
                
        except ClientError as e:
            logger.error(f"Client error {e.status_code}: {e.message}")
            # Don't retry client errors (4xx)
            raise
            
        except ValidationError as e:
            logger.error(f"Validation error: {e.message}")
            logger.error(f"Context: {e.context}")
            # Don't retry validation errors
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error: {type(e).__name__}: {e}")
            raise
    
    raise Exception(f"Failed after {max_retries} attempts")

# Use the safe API call
client = SmartSurgeClient()

try:
    response = safe_api_call(client, "https://api.example.com/data")
    print(f"Success! Status: {response.status_code}")
    print(f"Data: {response.json()}")
except Exception as e:
    print(f"Failed to fetch data: {e}")
```

### Streaming Error Handling

```python
from smartsurge import SmartSurgeClient, JSONStreamingRequest
from smartsurge import StreamingError, ResumeError
import os
import json

def download_with_retry(url, output_file, max_attempts=3):
    """Download with automatic resume on failure"""
    client = SmartSurgeClient()
    state_file = f"{output_file}.state"
    
    for attempt in range(max_attempts):
        try:
            print(f"Download attempt {attempt + 1}/{max_attempts}")
            
            # Perform download with automatic state management
            result = client.stream_request(
                streaming_class=JSONStreamingRequest,
                endpoint=url,
                state_file=state_file,  # Automatic state persistence
                chunk_size=1024 * 1024  # 1MB chunks
            )
            
            print(f"Download completed: {result}")
            # Clean up state file on success
            if os.path.exists(state_file):
                os.remove(state_file)
            return result
                
        except StreamingError as e:
            print(f"Streaming error: {e.message}")
            # State is automatically saved
                
        except ResumeError as e:
            print(f"Resume not supported: {e.message}")
            # Start fresh download
            if os.path.exists(state_file):
                os.remove(state_file)
                
        except Exception as e:
            print(f"Unexpected error: {e}")
            
        if attempt < max_attempts - 1:
            print("Retrying in 5 seconds...")
            time.sleep(5)
    
    raise Exception(f"Download failed after {max_attempts} attempts")

# Use the download function
try:
    result = download_with_retry(
        "https://api.example.com/large-dataset.json",
        "dataset.json"
    )
    print("Download successful!")
except Exception as e:
    print(f"Download failed: {e}")
```

## Integration Examples

### Flask Web Application

```python
from flask import Flask, jsonify, request
from smartsurge import SmartSurgeClient
from functools import lru_cache
import os

app = Flask(__name__)

# Configure client with environment variables
client = SmartSurgeClient(
    base_url=os.getenv("API_BASE_URL", "https://api.example.com"),
    timeout=(5.0, 30.0),
    max_retries=3
)

# Set known rate limits
client.set_rate_limit(
    endpoint=client.base_url,
    method="GET",
    max_requests=100,
    time_period=60  # 100 requests per minute
)

# Cache results to reduce API calls
@lru_cache(maxsize=128)
def cached_get(endpoint: str):
    """Cached GET request"""
    response = client.get(endpoint)
    return response.json()

@app.route('/api/users')
def get_users():
    try:
        # Using the configured base_url
        response = client.get("/users")
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/users/<int:user_id>')
def get_user(user_id):
    try:
        response = client.get(f"/users/{user_id}")
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/users', methods=['POST'])
def create_user():
    try:
        response = client.post(
            "/users",
            json=request.json
        )
        return jsonify(response.json()), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/rate-limits')
def show_rate_limits():
    """Show current rate limit status"""
    limits = client.list_rate_limits()
    return jsonify({
        f"{endpoint} {method}": {
            "max": limit.max_requests if limit else None,
            "period": limit.time_period if limit else None,
            "source": limit.source if limit else None
        }
        for (endpoint, method), limit in limits.items()
    })

if __name__ == '__main__':
    app.run(debug=True)
```

### Data Processing Pipeline

```python
import pandas as pd
from smartsurge import SmartSurgeClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_and_process_data():
    """Fetch data from API and process with pandas"""
    client = SmartSurgeClient()
    
    # Fetch paginated data
    all_data = []
    page = 1
    per_page = 100
    
    while True:
        logger.info(f"Fetching page {page}")
        
        response = client.get(
            "https://api.example.com/data",
            params={"page": page, "per_page": per_page}
        )
        
        data = response.json()
        if not data.get('items'):
            break  # No more data
            
        all_data.extend(data['items'])
        
        # Check rate limit status
        limits = client.list_rate_limits()
        for (endpoint, method), limit in limits.items():
            if limit and "api.example.com/data" in endpoint:
                logger.info(f"Rate limit: {limit.max_requests}/{limit.time_period}s")
        
        page += 1
        
        # Be respectful of rate limits
        if page % 10 == 0:
            logger.info(f"Processed {len(all_data)} items so far")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    logger.info(f"Total records fetched: {len(df)}")
    
    # Perform data processing
    df['date'] = pd.to_datetime(df['timestamp'])
    df_daily = df.groupby(df['date'].dt.date).agg({
        'value': ['sum', 'mean', 'count'],
        'category': lambda x: x.value_counts().to_dict()
    })
    
    # Save results
    df_daily.to_csv('daily_summary.csv')
    df.to_parquet('raw_data.parquet')  # Efficient storage
    
    logger.info(f"Processed {len(df)} records into {len(df_daily)} daily summaries")
    return df_daily

# Run the pipeline with error handling
try:
    result = fetch_and_process_data()
    print("Pipeline completed successfully")
    print(result.head())
except Exception as e:
    logger.error(f"Pipeline failed: {e}")
    raise
```

### Command-Line Tool

```python
#!/usr/bin/env python3
"""
CLI tool using SmartSurge for API interactions
"""

import argparse
import json
import sys
from smartsurge import SmartSurgeClient
from smartsurge.exceptions import SmartSurgeException

def main():
    parser = argparse.ArgumentParser(description="API CLI with SmartSurge")
    parser.add_argument("action", choices=["get", "post", "list-limits"])
    parser.add_argument("endpoint", nargs="?", help="API endpoint")
    parser.add_argument("--data", help="JSON data for POST requests")
    parser.add_argument("--base-url", default="https://api.example.com")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON")
    
    args = parser.parse_args()
    
    # Configure client
    client = SmartSurgeClient(
        base_url=args.base_url,
        timeout=(10.0, float(args.timeout))
    )
    
    try:
        if args.action == "get":
            if not args.endpoint:
                parser.error("endpoint required for get action")
            
            response = client.get(args.endpoint)
            output = response.json()
            
        elif args.action == "post":
            if not args.endpoint or not args.data:
                parser.error("endpoint and --data required for post action")
            
            data = json.loads(args.data)
            response = client.post(args.endpoint, json=data)
            output = response.json()
            
        elif args.action == "list-limits":
            limits = client.list_rate_limits()
            output = {
                f"{endpoint} {method}": {
                    "max": limit.max_requests if limit else None,
                    "period": limit.time_period if limit else None,
                    "source": limit.source if limit else None
                }
                for (endpoint, method), limit in limits.items()
            }
        
        # Output results
        if args.pretty:
            print(json.dumps(output, indent=2))
        else:
            print(json.dumps(output))
            
    except SmartSurgeException as e:
        print(f"Error: {e.message}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Advanced Usage

### Custom User Agent and Headers

```python
from smartsurge import SmartSurgeClient

# Set custom user agent during initialization
client = SmartSurgeClient(
    user_agent="MyApp/1.0 (SmartSurge)"
)

# Or use headers in individual requests
response = client.get(
    "https://api.example.com/data",
    headers={
        "User-Agent": "MyApp/1.0",
        "X-Custom-Header": "value"
    }
)
```

### Request History Analysis

```python
from smartsurge import SmartSurgeClient

client = SmartSurgeClient()

# Make several requests
for i in range(10):
    response, history = client.get(
        f"https://api.example.com/items/{i}",
        return_history=True
    )
    
    # Analyze request patterns
    print(f"Request {i+1}:")
    print(f"  Total requests in history: {len(history.requests)}")
    print(f"  Success rate: {history.success_rate:.2%}")
    print(f"  Average response time: {history.avg_response_time:.3f}s")
    
    # Check for detected patterns
    if history.has_rate_limit_pattern:
        print(f"  Rate limit pattern detected!")
```

### Utility Functions

```python
from smartsurge import (
    SmartSurgeTimer,
    log_context,
    merge_histories,
    configure_logging
)
import logging

# Configure logging for SmartSurge
configure_logging(level=logging.DEBUG)

# Use the timer utility
with SmartSurgeTimer() as timer:
    client = SmartSurgeClient()
    response = client.get("https://api.example.com/data")
    
print(f"Request took {timer.duration:.3f} seconds")

# Use log context for better debugging
with log_context(request_id="123", user="john"):
    response = client.get("https://api.example.com/users/john")
    # Logs will include context information

# Merge request histories
history1 = client.get("/endpoint1", return_history=True)[1]
history2 = client.get("/endpoint2", return_history=True)[1]

merged_history = merge_histories([history1, history2])
print(f"Merged history has {len(merged_history.requests)} requests")
```

## Best Practices

1. **Always handle exceptions** - SmartSurge raises specific exceptions for different error types
2. **Use rate limit detection** - Let SmartSurge learn rate limits automatically
3. **Enable request history** - Use `return_history=True` to monitor patterns
4. **Configure timeouts appropriately** - Set both connect and read timeouts
5. **Use async for concurrent requests** - Better performance for multiple API calls
6. **Implement proper logging** - Use SmartSurge's logging utilities
7. **Cache responses when possible** - Reduce unnecessary API calls
8. **Use streaming for large downloads** - Automatic resume support
9. **Set a base URL** - Cleaner code when working with a single API
10. **Monitor rate limit status** - Use `list_rate_limits()` to check detected limits