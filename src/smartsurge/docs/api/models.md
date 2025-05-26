# Models API Reference

The Models module provides core data models and enumerations used throughout SmartSurge for structured representation of requests, rate limits, and search status.

## Enumerations

### RequestMethod

```python
from smartsurge.models import RequestMethod
```

HTTP request methods supported by the library.

#### Values

| Value | Description |
|-------|-------------|
| `GET` | HTTP GET method |
| `POST` | HTTP POST method |
| `PUT` | HTTP PUT method |
| `DELETE` | HTTP DELETE method |
| `HEAD` | HTTP HEAD method |
| `OPTIONS` | HTTP OPTIONS method |
| `PATCH` | HTTP PATCH method |

#### Usage

```python
from smartsurge.models import RequestMethod

# Use enum directly
method = RequestMethod.GET

# Convert from string
method_str = "post"
method = RequestMethod(method_str.upper())  # RequestMethod.POST

# String representation
print(method)  # "POST"
print(str(method))  # "POST"
```

### SearchStatus

```python
from smartsurge.models import SearchStatus
```

Status of the rate limit search process.

#### Values

| Value | Description |
|-------|-------------|
| `NOT_STARTED` | Rate limit search has not begun |
| `WAITING_TO_ESTIMATE` | Collecting data before estimation |
| `COMPLETED` | Rate limit has been estimated |

## Data Models

### RequestEntry

```python
from smartsurge.models import RequestEntry
```

A single request entry that records details of an HTTP request.

#### Constructor

```python
RequestEntry(
    endpoint: str,
    method: RequestMethod,
    timestamp: datetime = <now>,
    status_code: int,
    response_time: float,
    success: bool,
    max_requests: Optional[int] = None,
    max_request_period: Optional[float] = None,
    response_headers: Optional[Dict[str, Any]] = None
)
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `endpoint` | `str` | The API endpoint that was requested (min length: 1) |
| `method` | `RequestMethod` | HTTP method used for the request |
| `timestamp` | `datetime` | When the request was made (UTC, auto-set to now) |
| `status_code` | `int` | HTTP status code received (0-599) |
| `response_time` | `float` | Time taken to receive response in seconds (≥ 0) |
| `success` | `bool` | Whether the request was successful |
| `max_requests` | `Optional[int]` | Maximum requests allowed if specified (≥ 1) |
| `max_request_period` | `Optional[float]` | Period for max_requests in seconds (> 0) |
| `response_headers` | `Optional[Dict[str, Any]]` | Response headers for rate limit extraction |

#### Validation

- Warns if `success=True` but `status_code >= 400`
- Requires `max_request_period` if `max_requests` is set
- Automatically extracts rate limits from common headers:
  - `X-RateLimit-Limit`, `RateLimit-Limit`
  - `X-RateLimit-Remaining`, `RateLimit-Remaining`
  - `X-RateLimit-Reset`, `RateLimit-Reset`
  - `Retry-After`, `X-Rate-Limit`

#### Example

```python
from datetime import datetime, timezone
from smartsurge.models import RequestEntry, RequestMethod

# Basic entry
entry = RequestEntry(
    endpoint="/api/users",
    method=RequestMethod.GET,
    status_code=200,
    response_time=0.25,
    success=True
)

# Entry with rate limit info
entry = RequestEntry(
    endpoint="/api/data",
    method=RequestMethod.POST,
    status_code=429,
    response_time=0.1,
    success=False,
    max_requests=100,
    max_request_period=60.0,
    response_headers={"X-RateLimit-Limit": "100", "Retry-After": "30"}
)
```

### RateLimit

```python
from smartsurge.models import RateLimit
```

Rate limit information for an endpoint, estimated using HMM.

#### Constructor

```python
RateLimit(
    endpoint: str,
    method: RequestMethod,
    max_requests: int,
    time_period: float,
    last_updated: datetime = <now>,
    cooldown: Optional[float] = None,
    time_cooldown_set: Optional[datetime] = None,
    source: str = "estimated"
)
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `endpoint` | `str` | The API endpoint this rate limit applies to |
| `method` | `RequestMethod` | HTTP method this rate limit applies to |
| `max_requests` | `int` | Maximum requests allowed in time period (≥ 1) |
| `time_period` | `float` | Time period in seconds for rate limit window (> 0) |
| `last_updated` | `datetime` | When this rate limit was last updated (UTC) |
| `cooldown` | `Optional[float]` | Cooldown period in seconds before next request |
| `time_cooldown_set` | `Optional[datetime]` | Timestamp when cooldown was set |
| `source` | `str` | Source of rate limit: "estimated", "headers", "manual" |

#### Methods

##### get_requests_per_second()

Get the rate limit as requests per second for easier comparison.

```python
def get_requests_per_second() -> float
```

#### String Representation

```python
rate_limit = RateLimit(
    endpoint="/api/users",
    method=RequestMethod.GET,
    max_requests=100,
    time_period=60.0,
    cooldown=5.0
)
print(rate_limit)
# Output: RateLimit(100 requests per 60.00s, cooldown: 5.00s, source: estimated)
```

### RequestHistory

```python
from smartsurge.models import RequestHistory
```

Tracks request logs and estimates rate limits for a single endpoint/method combination using a Hidden Markov Model approach.

#### Constructor

```python
RequestHistory(
    endpoint: str,
    method: RequestMethod,
    entries: List[RequestEntry] = [],
    rate_limit: Optional[RateLimit] = None,
    search_status: SearchStatus = SearchStatus.NOT_STARTED,
    min_time_period: float = 1.0,
    max_time_period: float = 3600.0,
    min_data_points: int = 10,
    max_observations: int = 50,
    consecutive_refusals: int = 0,
    request_id: str = <auto>,
    hmm: Optional[HMM] = None,
    logger: Optional[logging.Logger] = None,
    refit_every: int = 10,
    responses_since_refit: int = 0
)
```

#### Key Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `endpoint` | `str` | - | The endpoint being tracked |
| `method` | `RequestMethod` | - | HTTP method being tracked |
| `entries` | `List[RequestEntry]` | `[]` | Request entries (limited to max_observations) |
| `rate_limit` | `Optional[RateLimit]` | `None` | Current estimated rate limit |
| `search_status` | `SearchStatus` | `NOT_STARTED` | Current rate limit search status |
| `min_time_period` | `float` | `1.0` | Min seconds for rate limit window |
| `max_time_period` | `float` | `3600.0` | Max seconds for rate limit window |
| `min_data_points` | `int` | `10` | Min observations before estimation (5-100) |
| `max_observations` | `int` | `50` | Max observations to store (20-1000) |
| `consecutive_refusals` | `int` | `0` | Count of consecutive 429 responses |
| `request_id` | `str` | auto | Unique ID for request tracking |
| `hmm` | `Optional[HMM]` | auto | Hidden Markov Model instance |
| `refit_every` | `int` | `10` | Responses between HMM refits (1-1000) |

#### Core Methods

##### add_request()

Add a request entry to the history.

```python
def add_request(entry: RequestEntry) -> None
```

Raises `ValueError` if entry's endpoint/method doesn't match.

##### has_minimum_observations()

Check if there are enough observations for estimation.

```python
def has_minimum_observations() -> bool
```

Returns `True` if:
- At least `min_data_points` observations exist
- At least one success and one failure

##### merge()

Merge another RequestHistory into this one.

```python
def merge(other: RequestHistory) -> None
```

Raises `ValueError` if endpoint/method doesn't match.

##### intercept_request()

Intercept a request to enforce rate limits (call before making request).

```python
def intercept_request() -> None
```

Enforces:
- Cooldown periods
- Rate limit throttling
- Search status transitions

##### log_response_and_update()

Log response and update search status/HMM (call after receiving response).

```python
def log_response_and_update(entry: RequestEntry) -> None
```

Handles:
- Consecutive refusal tracking with exponential backoff
- HMM retraining when needed
- Search status updates

#### Utility Methods

##### __len__()

Get number of entries.

```python
len(history)  # Returns number of entries
```

#### Private Methods

- `_update_hmm()`: Update HMM with current data
- `_enforce_rate_limit()`: Enforce estimated rate limit

## Complete Example

```python
from smartsurge.models import (
    RequestMethod, RequestEntry, RequestHistory, 
    RateLimit, SearchStatus
)
from datetime import datetime, timezone
import time

# Create request history for tracking
history = RequestHistory(
    endpoint="/api/users",
    method=RequestMethod.GET,
    min_data_points=5  # Need at least 5 observations
)

# Simulate API requests
for i in range(15):
    # Intercept before request
    history.intercept_request()
    
    # Simulate making request
    time.sleep(0.1)  # Simulate network delay
    
    # Create entry based on response
    entry = RequestEntry(
        endpoint="/api/users",
        method=RequestMethod.GET,
        status_code=200 if i < 10 else 429,
        response_time=0.1,
        success=i < 10
    )
    
    # Log response
    history.log_response_and_update(entry)
    
    # Check status
    print(f"Request {i+1}: Status={history.search_status}")
    
    if history.rate_limit:
        print(f"Rate limit: {history.rate_limit}")
        print(f"Requests/sec: {history.rate_limit.get_requests_per_second():.2f}")

# Check final status
print(f"Total requests: {len(history)}")
print(f"Search status: {history.search_status}")

# Manual rate limit setting
if history.rate_limit:
    history.rate_limit = RateLimit(
        endpoint="/api/users",
        method=RequestMethod.GET,
        max_requests=50,
        time_period=60.0,
        source="manual"
    )
```

## Integration with Client

The `RequestHistory` class is used internally by `SmartSurgeClient`:

```python
from smartsurge import Client

client = Client()

# Client manages RequestHistory instances per endpoint/method
response, history = client.get("/api/users", return_history=True)

# Access history details
print(f"Requests tracked: {len(history)}")
if history.rate_limit:
    print(f"Rate limit: {history.rate_limit}")
```

## Performance Considerations

1. **Memory**: Each `RequestHistory` stores up to `max_observations` entries
2. **HMM Refitting**: Occurs every `refit_every` responses after initial estimation
3. **Exponential Backoff**: Consecutive 429s trigger exponential cooldown (2^n seconds)

## Advanced Configuration

```python
# High-performance configuration
history = RequestHistory(
    endpoint="/api/critical",
    method=RequestMethod.POST,
    min_data_points=20,      # More data before estimation
    max_observations=100,    # Keep more history
    refit_every=50          # Less frequent refitting
)

# Quick detection configuration  
history = RequestHistory(
    endpoint="/api/test",
    method=RequestMethod.GET,
    min_data_points=5,       # Estimate with less data
    max_observations=20,     # Smaller memory footprint
    refit_every=5           # Frequent updates
)
```