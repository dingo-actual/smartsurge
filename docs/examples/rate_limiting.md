# Rate Limiting Examples

## Observing Rate Limit Discovery

This example demonstrates how SmartSurge discovers and adapts to rate limits.

```python
import time
from smartsurge import Client, configure_logging

# Configure detailed logging to observe the rate limit discovery process

logger = configure_logging(level="DEBUG", output_file="rate_limit_discovery.log")

# Create a client with a short min_time_period to observe rate limit discovery faster

client = Client(
    base_url="https://api.example.com",
    min_time_period=0.1,  # 100ms minimum time period
    max_time_period=60.0,  # 1 minute maximum time period
    confidence_threshold=0.8,
    logger=logger
)

# Make a series of requests to observe rate limit discovery

for i in range(100):
print(f"Request {i+1}")
try:
    response, history = client.get(f"/endpoint?iteration={i}")
    print(f"Status: {response.status_code}")

    # Print rate limit info if available
    if history.rate_limit:
        print(f"Current rate limit estimate: {history.rate_limit}")
        print(f"Search status: {history.search_status}")
        print(f"Confidence: {history.rate_limit.confidence:.2f}")
    else:
        print("No rate limit detected yet")
        
    # Print recent entries count
    recent_count = len([e for e in history.entries if 
                        (time.time() - e.timestamp.timestamp()) < 60])
    print(f"Requests in last minute: {recent_count}")
    
    time.sleep(0.1)  # Small delay between requests
except Exception as e:
    print(f"Error: {e}")
    time.sleep(1)  # Longer delay after error
```

## Rate Limit Adaptation

This example shows how SmartSurge adapts when a rate limit changes.

```python
import time
from smartsurge import Client, configure_logging

logger = configure_logging(level="INFO")
client = Client(base_url="https://api.example.com")

# First phase: Make requests until rate limit is detected

print("Phase 1: Initial rate limit discovery")
for i in range(50):
    try:
        response, history = client.get("/endpoint")
        time.sleep(0.2)
        if history.rate_limit and history.search_status == "completed":
            print(f"Rate limit detected: {history.rate_limit}")
            break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(1)

# Second phase: Continue making requests with detected rate limit

print("\nPhase 2: Working with detected rate limit")
for i in range(20):
    try:
        response, history = client.get("/endpoint")
        print(f"Request successful with rate limit: {history.rate_limit}")
        time.sleep(0.5)
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(1)

# Rate limit likely changes here (e.g., different time of day or API tier)

print("\nPhase 3: Adaptation to changed rate limit")
for i in range(50):
    try:
        response, history = client.get("/endpoint")
        print(f"Request with potentially changed rate limit: {history.rate_limit}")
        print(f"Restart count: {history.restart_count}")
        time.sleep(0.5)
    except Exception as e:
        print(f"Error during adaptation: {e}")
        time.sleep(2)
```

## Handling Multiple Endpoints

```python
from smartsurge import Client

client = Client(base_url="https://api.example.com")

# Different endpoints may have different rate limits

endpoints = [
    "/users",
    "/products",
    "/orders",
    "/analytics"
]

# Make requests to each endpoint

for endpoint in endpoints:
    print(f"\nTesting endpoint: {endpoint}")
    for i in range(5):
        try:
            response, history = client.get(endpoint)
            status = "Completed" if history.search_status == "completed" else "In progress"
            if history.rate_limit:
                print(f"Request {i+1}: {status} - Rate limit: {history.rate_limit}")
            else:
                print(f"Request {i+1}: {status} - No rate limit detected yet")
        else:
            print(f"Request {i+1}: {status} - No rate limit detected yet")
        except Exception as e:
            print(f"Request {i+1}: Error - {e}")
```

## Using Different HTTP Methods

```python
from smartsurge import Client, RequestMethod

client = Client(base_url="https://api.example.com")

# Different HTTP methods may have different rate limits

methods = [
    RequestMethod.GET,
    RequestMethod.POST,
    RequestMethod.PUT,
    RequestMethod.DELETE
]

endpoint = "/resource"

# Test different methods on the same endpoint

for method in methods:
    print(f"\nTesting method: {method.value}")
    for i in range(5):
        try:
            if method == RequestMethod.GET:
                response, history = client.get(endpoint)
            elif method == RequestMethod.POST:
                response, history = client.post(endpoint, json={"test": True})
            elif method == RequestMethod.PUT:
                response, history = client.put(endpoint, json={"test": True})
            elif method == RequestMethod.DELETE:
                response, history = client.delete(endpoint)

            if history.rate_limit:
                print(f"Request {i+1}: Rate limit: {history.rate_limit}")
            else:
                print(f"Request {i+1}: No rate limit detected yet")
        except Exception as e:
            print(f"Request {i+1}: Error - {e}")
```
