# Basic Usage Examples

## Simple GET Request

```python
from smartsurge import Client

# Create a client

client = Client(base_url="https://api.example.com")

# Make a GET request

response, history = client.get("/users")

# Print response data

print(f"Status code: {response.status_code}")
print(f"Response JSON: {response.json()}")

# Check if rate limit has been detected

if history.rate_limit:
    print(f"Detected rate limit: {history.rate_limit}")
else:
    print("No rate limit detected yet")
```

## POST Request with JSON Data

```python
from smartsurge import Client

client = Client(base_url="https://api.example.com")

# Create a new user

user_data = {
    "name": "Jane Doe",
    "email": "jane@example.com",
    "role": "admin"
}

response, history = client.post(
    "/users",
    json=user_data,
    headers={"Content-Type": "application/json"}
)

# Check if request was successful

if response.ok:
print(f"User created successfully: {response.json()}")
else:
print(f"Failed to create user: {response.text}")
```

## Using Client as a Context Manager

```python
from smartsurge import Client

# Use context manager to ensure the client is properly closed

with Client(base_url="https://api.example.com") as client:
    # Make multiple requests
    users_response, users_history = client.get("/users")
    products_response, products_history = client.get("/products")

    # Process responses
    users = users_response.json()
    products = products_response.json()
    
    print(f"Found {len(users)} users and {len(products)} products")
```

## Error Handling

```python
from smartsurge import Client, RateLimitExceeded, StreamingError

client = Client(base_url="https://api.example.com")

try:
    response, history = client.get("/restricted-endpoint")
    print(f"Response: {response.json()}")
except RateLimitExceeded as e:
    print(f"Rate limit exceeded: {e}")
    print(f"Endpoint: {e.endpoint}")
    print(f"Method: {e.method}")
    print(f"Retry after: {e.retry_after} seconds")
except requests.RequestException as e:
    print(f"Request failed: {e}")
```

## Making Requests with Custom Parameters

```python
from smartsurge import Client

client = Client(base_url="https://api.example.com")

# GET request with parameters

response, history = client.get(
    "/search",
    params={"q": "python", "sort": "relevance"},
    headers={"Accept": "application/json"},
    timeout=10.0,
    verify=True
)

# Print search results

results = response.json()
print(f"Found {len(results)} results for 'python'")
for result in results[:5]:  # Print first 5 results
    print(f"- {result['title']}")
```
