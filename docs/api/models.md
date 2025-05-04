# Models API Reference

This page documents the core data models used in SmartSurge.

## `RequestMethod`

```python
class RequestMethod(str, Enum):
    """HTTP request methods supported by the library."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"
    PATCH = "PATCH"
```

An enumeration of HTTP methods supported by the library.

## `SearchStatus`

```python
class SearchStatus(str, Enum):
    """Status of the rate limit search process."""
    NOT_STARTED = "not_started"
    WAITING_FOR_FIRST_REFUSAL = "waiting_for_first_refusal"
    ESTIMATING = "estimating"
    COMPLETED = "completed"
```

An enumeration of the possible states in the rate limit search process.

## `RequestEntry`

```python
class RequestEntry(BaseModel):
    """
    A single request entry that records details of an HTTP request.
    """
```

### Attributes

- `endpoint` - The API endpoint that was requested
- `method` - The HTTP method used for the request
- `timestamp` - When the request was made (UTC)
- `status_code` - HTTP status code received
- `response_time` - Time taken to receive response in seconds
- `success` - Whether the request was successful (typically status < 400)

### Methods

```python
def validate_success_code_consistency(self) -> "RequestEntry"
```

Ensures success flag is consistent with status code.

## `RateLimit`

```python
class RateLimit(BaseModel):
    """
    Rate limit information for an endpoint.
    """
```

### Attributes

- `endpoint` - The API endpoint this rate limit applies to
- `method` - The HTTP method this rate limit applies to
- `max_requests` - Maximum number of requests allowed in the time period
- `time_period` - Time period in seconds for the rate limit window
- `confidence` - Confidence level in this rate limit estimation (0.0-1.0)
- `last_updated` - When this rate limit was last updated
- `time_period_lower` - Lower bound of credible interval for time_period
- `time_period_upper` - Upper bound of credible interval for time_period
- `max_requests_lower` - Lower bound of credible interval for max_requests
- `max_requests_upper` - Upper bound of credible interval for max_requests

### Methods

```python
def __str__(self) -> str
```

Returns a human-readable string representation of the rate limit.

## `RequestHistory`

```python
class RequestHistory(BaseModel):
    """
    Tracks request logs and estimates rate limits for a single endpoint and method combination
    using a statistically rigorous Bayesian approach.
    """
```

### Attributes

- `endpoint` - The endpoint being tracked
- `method` - The HTTP method being tracked
- `entries` - List of request entries
- `rate_limit` - The current estimated rate limit
- `search_status` - Current status of the rate limit search
- `min_time_period` - Minimum time period to consider for rate limiting (seconds)
- `max_time_period` - Maximum time period to consider for rate limiting (seconds)
- `confidence_threshold` - Confidence threshold for rate limit estimation
- `restart_count` - Number of times estimation has been restarted
- `poisson_alpha` - Shape parameter for Gamma distribution (time period rate prior)
- `poisson_beta` - Rate parameter for Gamma distribution (time period rate prior)
- `beta_a` - Alpha parameter for Beta distribution (success probability prior)
- `beta_b` - Beta parameter for Beta distribution (success probability prior)
- `common_rate_limits` - Dictionary of common API rate limits to inform priors
- `credible_interval_width` - Width of credible interval for confidence calculation
- `regularization_strength` - Regularization parameter to prevent numerical instability
- `logger` - Custom logger for this instance (not included in serialized output)
- `request_id` - Request tracking ID for better log correlation

### Methods

```python
def __init__(self, **data)
```

Initializes a RequestHistory instance with appropriate logger.

```python
def add_request(self, entry: RequestEntry) -> None
```

Adds a request entry to the history.

```python
def merge(self, other: 'RequestHistory') -> None
```

Merges another RequestHistory into this one, preserving sorting by timestamp.

```python
def intercept_request(self) -> None
```

Intercepts a request to enforce rate limit search procedure.

```python
def log_response_and_update(self, entry: RequestEntry) -> None
```

Logs the response and updates the search status and Bayesian estimates.

```python
def _reset_priors_for_restart(self) -> None
```

Resets or adjusts Bayesian priors when restarting the rate limit search.

```python
def _initialize_bayesian_priors(self) -> None
```

Initializes Bayesian priors based on domain knowledge and the current request history.

```python
def _update_bayesian_estimates(self, limit_exceeded: bool) -> None
```

Updates Bayesian estimates based on the latest request outcome.

```python
def _update_rate_limit_from_priors(self) -> None
```

Updates the rate limit based on current Bayesian priors.

```python
def _check_estimation_confidence(self) -> bool
```

Checks if the estimation confidence is high enough to consider it complete.

```python
def _enforce_rate_limit(self) -> None
```

Enforces the estimated rate limit by waiting if necessary.

```python
def _enforce_bayesian_estimate(self) -> None
```

Enforces rate limiting based on current Bayesian estimates during the ESTIMATING phase.
