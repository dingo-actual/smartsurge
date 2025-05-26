# Basic Usage

This guide covers the fundamental features of SmartSurge with practical examples.

## Installation

```bash
pip install smartsurge
```

## Creating a Client

### The Simplest Way

```python
from smartsurge import SmartSurgeClient

# Create a client - it works just like requests!
client = SmartSurgeClient()

# Make requests to any URL
response = client.get("https://api.github.com/users/github")
print(response.json())
```

### With a Base URL

```python
# Set a base URL to avoid repeating it
client = SmartSurgeClient(base_url="https://api.github.com")

# Now use relative paths
response = client.get("/users/github")
response = client.get("/repos/python/cpython")
```

### With Configuration

```python
# Configure client behavior
client = SmartSurgeClient(
    base_url="https://api.example.com",
    timeout=(10.0, 30.0),        # (connect, read) timeouts
    max_retries=5,               # Retry failed requests
    verify_ssl=True              # SSL certificate verification
)
```

## Making HTTP Requests

SmartSurge supports all standard HTTP methods. The interface is identical to `requests`, but with automatic rate limiting.

### GET Requests

```python
# Simple GET
response = client.get("https://api.example.com/users")

# With query parameters
response = client.get(
    "https://api.example.com/users",
    params={"page": 1, "limit": 10}
)

# With custom headers
response = client.get(
    "https://api.example.com/users/123",
    headers={"Authorization": "Bearer token123"}
)
```

### POST Requests

```python
# POST with JSON data
response = client.post(
    "https://api.example.com/users",
    json={"name": "John Doe", "email": "john@example.com"}
)

# POST with form data
response = client.post(
    "https://api.example.com/login",
    data={"username": "john", "password": "secret"}
)

# POST with files
with open("document.pdf", "rb") as f:
    response = client.post(
        "https://api.example.com/upload",
        files={"file": f}
    )
```

### Other HTTP Methods

```python
# PUT - Full update
response = client.put(
    "https://api.example.com/users/123",
    json={"name": "John Updated", "email": "john.updated@example.com"}
)

# PATCH - Partial update
response = client.patch(
    "https://api.example.com/users/123",
    json={"email": "new.email@example.com"}
)

# DELETE
response = client.delete("https://api.example.com/users/123")

# HEAD (not available in current implementation)
# OPTIONS (not available in current implementation)
```

## Working with Responses

Responses work exactly like in the `requests` library:

```python
response = client.get("https://api.example.com/users")

# Status code
print(response.status_code)  # 200

# Headers
print(response.headers["Content-Type"])  # application/json

# Different response formats
json_data = response.json()    # Parse JSON
text_data = response.text      # Get as string
binary_data = response.content # Get as bytes

# Response properties
print(response.ok)        # True if status < 400
print(response.url)       # Final URL after redirects
print(response.elapsed)   # Time taken for request
```

## Understanding Rate Limits

SmartSurge automatically handles rate limiting using Hidden Markov Models. You can access detailed information about detected limits.

### Getting Request History

```python
# Request with history
response, history = client.get("https://api.example.com/data", return_history=True)

# Access rate limit info
if history.rate_limit:
    print(f"Rate limit: {history.rate_limit.max_requests} requests per {history.rate_limit.time_period}s")
    print(f"Source: {history.rate_limit.source}")  # 'manual', 'headers', or 'estimated'

# Success rate and history
print(f"Success rate: {history.success_rate():.2%}")
print(f"Total requests: {len(history)}")
```

### Manual Rate Limit Configuration

```python
from smartsurge.models import RequestMethod

# Set a known rate limit for specific endpoint and method
client.set_rate_limit(
    endpoint="/api/users",
    method="GET",
    max_requests=100,
    time_period=60.0,  # 100 requests per 60 seconds
    cooldown=1.0       # Optional: 1 second between requests
)

# Check current rate limit
limit = client.get_rate_limit("/api/users", "GET")
if limit:
    print(f"Current limit: {limit}")
    print(f"Requests per second: {limit.get_requests_per_second():.2f}")

# Clear rate limit
client.clear_rate_limit("/api/users", "GET")

# List all rate limits
all_limits = client.list_rate_limits()
for (endpoint, method), limit in all_limits.items():
    if limit:
        print(f"{method} {endpoint}: {limit}")
```

## Error Handling

SmartSurge provides specific exceptions for different error scenarios:

```python
from smartsurge import (
    RateLimitExceeded,
    StreamingError,
    ResumeError,
    ValidationError,
    ConfigurationError
)
from smartsurge.exceptions import (
    ConnectionError,
    TimeoutError,
    ServerError,
    ClientError
)

try:
    response = client.get("https://api.example.com/data")
    response.raise_for_status()  # Raise exception for 4xx/5xx
    
except RateLimitExceeded as e:
    print(f"Rate limit hit on {e.endpoint}!")
    if e.retry_after:
        print(f"Retry after {e.retry_after} seconds")
    
except ConnectionError as e:
    print(f"Connection failed: {e}")
    
except TimeoutError as e:
    print(f"Request timed out after {e.timeout}s")
    
except ServerError as e:
    print(f"Server error {e.status_code}: {e.response_text}")
    
except ClientError as e:
    print(f"Client error {e.status_code}: {e.response_text}")
```

## Authentication Examples

### API Key Authentication

```python
# In header
client = SmartSurgeClient()
response = client.get(
    "https://api.example.com/data",
    headers={"X-API-Key": "your-api-key"}
)

# In query parameter
response = client.get(
    "https://api.example.com/data",
    params={"api_key": "your-api-key"}
)
```

### Bearer Token Authentication

```python
# Per request
response = client.get(
    "https://api.example.com/protected",
    headers={"Authorization": "Bearer your-token"}
)

# Or set default headers on client
client = SmartSurgeClient(
    base_url="https://api.example.com",
    # Headers would need to be passed to each request
)
```

### Basic Authentication

```python
# Using auth parameter
from requests.auth import HTTPBasicAuth

response = client.get(
    "https://api.example.com/protected",
    auth=HTTPBasicAuth("username", "password")
)

# Or using tuple
response = client.get(
    "https://api.example.com/protected",
    auth=("username", "password")
)
```

## Working with Sessions

SmartSurge automatically manages sessions for connection pooling:

```python
# Use as context manager (recommended)
with SmartSurgeClient() as client:
    response = client.get("https://api.example.com/data")
    # Session automatically closed when done

# Or manage manually
client = SmartSurgeClient()
response = client.get("https://api.example.com/data")
client.close()  # Don't forget to close!
```

## Timeouts and Retries

Configure how SmartSurge handles slow responses and failures:

```python
# Configure timeouts and retries
client = SmartSurgeClient(
    timeout=(10.0, 30.0),    # (connect, read) timeouts in seconds
    max_retries=5,           # Retry up to 5 times
    backoff_factor=0.3       # Wait 0.3 * (2 ** retry) seconds between retries
)

# Override timeout for specific request
response = client.get(
    "https://api.example.com/slow-endpoint",
    timeout=(10.0, 60.0)  # 60 seconds read timeout for this request
)
```

## Common Patterns

### Pagination

```python
# Fetch all pages automatically
def fetch_all_users(client):
    users = []
    page = 1
    
    while True:
        response = client.get(
            "/users",
            params={"page": page, "per_page": 100}
        )
        
        data = response.json()
        if not data:  # No more results
            break
            
        users.extend(data)
        page += 1
        
        # SmartSurge handles rate limiting automatically
        
    return users

# Usage
client = SmartSurgeClient(base_url="https://api.example.com")
all_users = fetch_all_users(client)
```

### Batch Processing

```python
# Process items with automatic rate limit protection
def process_items(client, item_ids):
    results = []
    errors = []
    
    for item_id in item_ids:
        try:
            response = client.get(f"/items/{item_id}")
            response.raise_for_status()
            results.append(response.json())
        except RateLimitExceeded as e:
            # SmartSurge already waited and retried
            # This only happens if retry limit exceeded
            errors.append({"id": item_id, "error": "rate_limit_exceeded"})
        except Exception as e:
            errors.append({"id": item_id, "error": str(e)})
            
    return {"results": results, "errors": errors}
```

### Monitoring API Health

```python
# Check API health with history
def check_api_health(client):
    try:
        response, history = client.get(
            "/health",
            timeout=5.0,
            return_history=True
        )
        
        health = {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "response_time": response.elapsed.total_seconds(),
            "success_rate": history.success_rate(),
            "rate_limit": str(history.rate_limit) if history.rate_limit else None
        }
        
        return health
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
```

### Async Operations (Preview)

```python
import asyncio

async def fetch_async():
    client = SmartSurgeClient()
    
    # Make async requests
    response = await client.async_get("https://api.example.com/data")
    data = await response.json()
    
    return data

# Run async function
data = asyncio.run(fetch_async())
```

## Best Practices

1. **Reuse Client Instances**
   ```python
   # Good - reuse client (connection pooling)
   client = SmartSurgeClient()
   for url in urls:
       response = client.get(url)
   
   # Bad - creates new client each time
   for url in urls:
       client = SmartSurgeClient()
       response = client.get(url)
   ```

2. **Use Context Managers**
   ```python
   # Good - automatic cleanup
   with SmartSurgeClient() as client:
       response = client.get("https://api.example.com/data")
   ```

3. **Handle Errors Gracefully**
   ```python
   # Always handle potential errors
   try:
       response = client.get("https://api.example.com/data")
       response.raise_for_status()
       data = response.json()
   except Exception as e:
       # Log error with context
       print(f"Request failed: {e}")
       # Fallback logic here
   ```

4. **Set Known Rate Limits**
   ```python
   # If you know the API limits, set them explicitly
   client.set_rate_limit("/api/v1/search", "GET", 
                        max_requests=100, time_period=900)  # 100 per 15 min
   ```

5. **Monitor Success Rates**
   ```python
   # Periodically check success rates
   _, history = client.get("/api/data", return_history=True)
   if history.success_rate() < 0.95:  # Less than 95% success
       print(f"Warning: Low success rate {history.success_rate():.2%}")
   ```

## Understanding HMM Detection

SmartSurge uses Hidden Markov Models to detect rate limits:

```python
# Control HMM detection behavior
client = SmartSurgeClient(
    refit_every=20            # Retrain HMM every 20 responses
)

# Check detection status
_, history = client.get("/api/data", return_history=True)
print(f"Search status: {history.search_status}")
# NOT_STARTED -> WAITING_TO_ESTIMATE -> COMPLETED
```

## Next Steps

- **[Advanced Usage](advanced_usage.md)** - Streaming, advanced configuration, and custom implementations
- **[Examples](examples.md)** - Real-world use cases and complete examples
- **[API Reference](api/index.md)** - Detailed API documentation
- **[Benchmarking](benchmark_usage.md)** - Test and measure rate limit detection