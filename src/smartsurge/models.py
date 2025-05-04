"""
Core models and enums for the SmartSurge library.

This module defines the data models and enumerations used throughout the
SmartSurge library, providing structured representation of requests, 
rate limits, and search status.
"""

from typing import List, Optional, Dict, Any, Union, Tuple, Set
from datetime import datetime, timedelta, timezone
from enum import Enum
import logging
import math
import time
import uuid

from pydantic import BaseModel, Field, model_validator, ConfigDict
from scipy import stats


# Module-level logger
logger = logging.getLogger(__name__)

class RequestMethod(str, Enum):
    """HTTP request methods supported by the library."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"
    PATCH = "PATCH"

class SearchStatus(str, Enum):
    """Status of the rate limit search process."""
    NOT_STARTED = "not_started"
    WAITING_FOR_FIRST_REFUSAL = "waiting_for_first_refusal"
    ESTIMATING = "estimating"
    COMPLETED = "completed"

class RequestEntry(BaseModel):
    """
    A single request entry that records details of an HTTP request.
    
    Attributes:
        endpoint: The API endpoint that was requested
        method: The HTTP method used for the request
        timestamp: When the request was made (UTC)
        status_code: HTTP status code received
        response_time: Time taken to receive response in seconds
        success: Whether the request was successful (typically status < 400)
    """
    endpoint: str = Field(min_length=1, description="The endpoint that was requested")
    method: RequestMethod = Field(description="HTTP method used")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When the request was made (UTC)")
    status_code: int = Field(ge=100, le=599, description="HTTP status code received")
    response_time: float = Field(ge=0.0, description="Time taken to receive response in seconds")
    success: bool = Field(description="Whether the request was successful")
    
    @model_validator(mode="after")
    def validate_success_code_consistency(self) -> "RequestEntry":
        """Ensure success flag is consistent with status code."""
        if self.success and self.status_code >= 400:
            logger.warning(f"Inconsistent success flag: marked as success but status code is {self.status_code}")
        return self

class RateLimit(BaseModel):
    """
    Rate limit information for an endpoint.
    
    Attributes:
        endpoint: The API endpoint this rate limit applies to
        method: The HTTP method this rate limit applies to
        max_requests: Maximum number of requests allowed in the time period
        time_period: Time period in seconds for the rate limit window
        confidence: Confidence level in this rate limit estimation (0.0-1.0)
        last_updated: When this rate limit was last updated
        time_period_lower: Lower bound of credible interval for time_period
        time_period_upper: Upper bound of credible interval for time_period
        max_requests_lower: Lower bound of credible interval for max_requests
        max_requests_upper: Upper bound of credible interval for max_requests
    """
    endpoint: str = Field(..., min_length=1, description="The endpoint this rate limit applies to")
    method: RequestMethod = Field(..., description="HTTP method this applies to")
    max_requests: int = Field(..., ge=1, description="Maximum number of requests allowed in the time period")
    time_period: float = Field(..., gt=0.0, description="Time period in seconds for the rate limit window")
    confidence: float = Field(default=0.0, gt=0.0, le=0.99, description="Confidence level in this estimate (0.0-1.0)")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When this rate limit was last updated")
    time_period_lower: Optional[float] = Field(default=None, gt=0.0, description="Lower bound of credible interval for time_period")
    time_period_upper: Optional[float] = Field(default=None, gt=0.0, description="Upper bound of credible interval for time_period")
    max_requests_lower: Optional[int] = Field(default=None, ge=1, description="Lower bound of credible interval for max_requests")
    max_requests_upper: Optional[int] = Field(default=None, ge=1, description="Upper bound of credible interval for max_requests")
    
    def __str__(self) -> str:
        """Return a human-readable string representation of the rate limit."""
        return (f"RateLimit({self.max_requests} requests per {self.time_period:.2f}s, "
                f"confidence: {self.confidence:.2f})")

class RequestHistory(BaseModel):
    """
    Tracks request logs and estimates rate limits for a single endpoint and method combination
    using a statistically rigorous Bayesian approach.
    
    Features:
    - Gamma-Poisson model for time period estimation with domain-informed priors
    - Beta-Binomial model for request success probability estimation within time windows
    - Variance-based weighting for failed requests based on statistical principles
    - Credible intervals for confidence calculation and conservative rate limiting
    - Uses average inter-arrival time for principled max_requests estimation
    - Statistically grounded approach to uncertainty handling during restarts
    - Robust numerical stability with regularization and bounds checking
    
    Attributes:
        endpoint: The endpoint being tracked
        method: The HTTP method being tracked
        entries: List of request entries
        rate_limit: The current estimated rate limit
        search_status: Current status of the rate limit search
        min_time_period: Minimum time period to consider for rate limiting (seconds)
        max_time_period: Maximum time period to consider for rate limiting (seconds)
        confidence_threshold: Confidence threshold for rate limit estimation
        restart_count: Number of times estimation has been restarted
        poisson_alpha: Shape parameter for Gamma distribution (time period rate prior)
        poisson_beta: Rate parameter for Gamma distribution (time period rate prior)
        beta_a: Alpha parameter for Beta distribution (success probability prior)
        beta_b: Beta parameter for Beta distribution (success probability prior)
        common_rate_limits: Dictionary of common API rate limits to inform priors
        credible_interval_width: Width of credible interval for confidence calculation
        regularization_strength: Regularization parameter to prevent numerical instability
        logger: Custom logger for this instance (not included in serialized output)
    """
    endpoint: str = Field(..., min_length=1)
    method: RequestMethod
    entries: List[RequestEntry] = Field(default_factory=list)
    rate_limit: Optional[RateLimit] = None
    search_status: SearchStatus = SearchStatus.NOT_STARTED
    min_time_period: float = Field(default=1.0, gt=0.0)
    max_time_period: float = Field(default=3600.0, gt=0.0)
    confidence_threshold: float = Field(default=0.9, ge=0.0, le=0.99) # <1.0 to prevent numerical instabilities
    restart_count: int = Field(default=0, ge=0)
    
    # Bayesian parameters for Gamma-Poisson model (time period rate)
    poisson_alpha: Optional[float] = None  # Shape parameter for Gamma distribution
    poisson_beta: Optional[float] = None   # Rate parameter for Gamma distribution
    
    # Bayesian parameters for Beta-Binomial model (success probability)
    beta_a: Optional[float] = None  # Alpha parameter for Beta distribution
    beta_b: Optional[float] = None  # Beta parameter for Beta distribution
    
    # Common API rate limits to inform priors (requests, seconds)
    common_rate_limits: Dict[str, Tuple[int, float]] = Field(
        default_factory=lambda: {
            "default": (60, 60.0),        # 60 requests per minute
            "standard": (100, 3600.0),    # 100 requests per hour
            "aggressive": (10, 1.0),      # 10 requests per second
            "conservative": (1000, 86400.0)  # 1000 requests per day
        }
    )
    
    # Credible interval width for confidence calculation
    credible_interval_width: float = Field(default=0.95, ge=0.5, le=0.99)
    
    # Regularization parameter to prevent numerical instability
    regularization_strength: float = Field(default=0.01, gt=0.0, le=1.0)
    
    # Logger (not included in serialized output)
    logger: Optional[logging.Logger] = Field(default=None, exclude=True)
    
    # Request tracking ID for better log correlation
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], exclude=True)
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    
    def __init__(self, **data):
        """
        Initialize a RequestHistory instance with appropriate logger.
        
        Args:
            **data: Data to initialize the model with, may include an optional 'logger'.
        """
        super().__init__(**data)
        # Set up logger if not provided
        if self.logger is None:
            self.logger = logger.getChild(f"RequestHistory.{self.endpoint}.{self.method}")
        
        # Initialize Bayesian priors if not already initialized
        if self.poisson_alpha is None or self.poisson_beta is None or self.beta_a is None or self.beta_b is None:
            self._initialize_bayesian_priors()

    def add_request(self, entry: RequestEntry) -> None:
        """
        Add a request entry to the history.

        Args:
            entry (RequestEntry): The request entry to add.

        Raises:
            ValueError: If the entry's endpoint or method does not match this history.
        """
        self.logger.debug(f"[{self.request_id}] Adding request entry for {self.endpoint}/{self.method}: {entry}")
        if entry.endpoint != self.endpoint or entry.method != self.method:
            self.logger.error(f"[{self.request_id}] Entry endpoint/method mismatch: {entry.endpoint}/{entry.method} vs {self.endpoint}/{self.method}")
            raise ValueError("Entry endpoint and method must match RequestHistory")
        self.entries.append(entry)
        self.entries.sort(key=lambda e: e.timestamp)  # Keep sorted by timestamp

    def merge(self, other: 'RequestHistory') -> None:
        """
        Merge another RequestHistory into this one, preserving sorting by timestamp.

        Args:
            other (RequestHistory): Another RequestHistory object with the same endpoint and method.

        Raises:
            ValueError: If the other history has a different endpoint or method.
        """
        self.logger.debug(f"[{self.request_id}] Merging RequestHistory with {len(other.entries)} entries")
        if self.endpoint != other.endpoint or self.method != other.method:
            self.logger.error(f"[{self.request_id}] Cannot merge RequestHistory with different endpoint/method: {self.endpoint}/{self.method} vs {other.endpoint}/{other.method}")
            raise ValueError("Can only merge RequestHistory objects with the same endpoint and method")
        
        # Combine entries, ensuring they remain sorted by timestamp
        combined_entries = self.entries + other.entries
        combined_entries.sort(key=lambda e: e.timestamp)
        self.entries = combined_entries
        
        # Merge rate limits, preferring the more recent one
        if other.rate_limit:
            if (not self.rate_limit) or (other.rate_limit.last_updated > self.rate_limit.last_updated):
                self.rate_limit = other.rate_limit
                
        # Merge Bayesian parameters, preferring the more informed posteriors
        if (other.poisson_alpha is not None and other.poisson_beta is not None and 
            (self.poisson_alpha is None or other.poisson_alpha > self.poisson_alpha)):
            self.poisson_alpha = other.poisson_alpha
            self.poisson_beta = other.poisson_beta
            
        if (other.beta_a is not None and other.beta_b is not None and
            (self.beta_a is None or other.beta_a + other.beta_b > (self.beta_a or 0) + (self.beta_b or 0))):
            self.beta_a = other.beta_a
            self.beta_b = other.beta_b
            
        # Update search status to the furthest along
        status_order = {
            SearchStatus.NOT_STARTED: 0,
            SearchStatus.WAITING_FOR_FIRST_REFUSAL: 1,
            SearchStatus.ESTIMATING: 2,
            SearchStatus.COMPLETED: 3
        }
        if status_order[other.search_status] > status_order[self.search_status]:
            self.search_status = other.search_status
            
        # Take the higher restart count
        self.restart_count = max(self.restart_count, other.restart_count)
            
        # Use the more strict confidence threshold
        self.confidence_threshold = max(self.confidence_threshold, other.confidence_threshold)
        
        self.logger.info(f"[{self.request_id}] Merged histories for {self.endpoint} {self.method}, now with {len(self.entries)} entries and search status {self.search_status}")

    def intercept_request(self) -> None:
        """
        Intercept a request to enforce rate limit search procedure.
        
        This method should be called before making a request.
        It waits as necessary based on the current search status.
        """
        self.logger.debug(f"[{self.request_id}] Intercepting request with search status: {self.search_status}")
        
        if self.search_status == SearchStatus.COMPLETED:
            # Use the estimated rate limit for throttling
            if self.rate_limit:
                self._enforce_rate_limit()
            return
        
        if self.search_status == SearchStatus.NOT_STARTED:
            # Initialize the search status
            self.search_status = SearchStatus.WAITING_FOR_FIRST_REFUSAL
            self.logger.debug(f"[{self.request_id}] Search status set to WAITING_FOR_FIRST_REFUSAL")
            return
        
        if self.search_status == SearchStatus.WAITING_FOR_FIRST_REFUSAL:
            # No waiting needed, we're still looking for the first refusal
            return
        
        if self.search_status == SearchStatus.ESTIMATING:
            # Use current Bayesian estimates to determine wait time
            if self.poisson_alpha is not None and self.poisson_beta is not None:
                self._enforce_bayesian_estimate()
            return

    def log_response_and_update(self, entry: RequestEntry) -> None:
        """
        Log the response and update the search status and Bayesian estimates.
        
        This method should be called after receiving a response.
        
        Key feature: If we receive a rate limit error (429) after search is completed,
        we restart the estimation process to adapt to changing rate limits.
        
        Args:
            entry (RequestEntry): The request entry to log.
        """
        self.logger.debug(f"[{self.request_id}] Logging response and updating search status: {entry}")
        self.add_request(entry)
        
        # Handle rate limit error (HTTP 429) - special case when search is COMPLETED
        if entry.status_code == 429 and self.search_status == SearchStatus.COMPLETED:
            self.logger.warning(f"[{self.request_id}] Rate limit error received after estimation was completed! "
                          f"Restarting rate limit search for {self.endpoint} {self.method}")
            
            # Increment restart count
            self.restart_count += 1
            
            # Reset to estimation phase
            self.search_status = SearchStatus.ESTIMATING
            
            # Reset or adjust Bayesian priors with the new information
            self._reset_priors_for_restart()
            
            # Log restart details
            self.logger.info(f"[{self.request_id}] Restarted rate limit search (restart #{self.restart_count}) "
                       f"for {self.endpoint} {self.method}")
            return
        
        if self.search_status == SearchStatus.NOT_STARTED:
            if entry.status_code == 429:  # Rate limit exceeded on first request
                self.logger.warning(f"[{self.request_id}] Rate limit refusal on first request for {self.endpoint} {self.method}")
                self._update_bayesian_estimates(True)
                self.search_status = SearchStatus.ESTIMATING
            else:
                self.logger.info(f"[{self.request_id}] First request received, waiting for first rate limit refusal for {self.endpoint} {self.method}")
                self.search_status = SearchStatus.WAITING_FOR_FIRST_REFUSAL
            return
        
        if self.search_status == SearchStatus.WAITING_FOR_FIRST_REFUSAL:
            if entry.status_code == 429:  # Rate limit exceeded
                self.logger.info(f"[{self.request_id}] First rate limit refusal detected for {self.endpoint} {self.method}")
                self._update_bayesian_estimates(True) # Update estimates with first refusal
                self.search_status = SearchStatus.ESTIMATING
            return
        
        if self.search_status == SearchStatus.ESTIMATING:
            if entry.status_code == 429:  # Another rate limit exceeded
                self._update_bayesian_estimates(True)
            else:
                self._update_bayesian_estimates(False)
            
            # Check if we have enough confidence to mark as completed
            if self._check_estimation_confidence():
                self.search_status = SearchStatus.COMPLETED
                if self.rate_limit:
                    self.logger.info(
                        f"[{self.request_id}] Rate limit estimation completed for {self.endpoint} {self.method}: "
                        f"{self.rate_limit} (restart count: {self.restart_count})"
                    )
            
            return

    def _reset_priors_for_restart(self) -> None:
        """
        Reset or adjust Bayesian priors when restarting the rate limit search.
        
        Uses a statistically principled approach to increase uncertainty by reducing
        the effective sample size of the priors, reflecting the model's failure to predict
        the rate limit accurately.
        """
        if not self.rate_limit:
            # If we don't have a previous rate limit, just initialize normally
            self._initialize_bayesian_priors()
            return
        
        # Increase uncertainty by reducing effective sample size
        uncertainty_factor = 0.7  # Reduce by 30% to reflect increased uncertainty
        if self.poisson_alpha is not None and self.poisson_beta is not None:
            # Preserve the mean but increase variance by reducing effective sample size
            old_mean = self.poisson_alpha / self.poisson_beta if self.poisson_beta > 0 else self.max_time_period
            self.poisson_alpha = max(1.0, self.poisson_alpha * uncertainty_factor)
            self.poisson_beta = max(0.001, self.poisson_alpha / old_mean)
        
        if self.beta_a is not None and self.beta_b is not None:
            # Preserve mean but increase variance by reducing effective sample size
            total = self.beta_a + self.beta_b
            if total > 0:
                p_mean = self.beta_a / total
                new_total = total * uncertainty_factor
                self.beta_a = max(0.5, new_total * p_mean)
                self.beta_b = max(0.5, new_total - self.beta_a)
            else:
                self.beta_a = max(0.5, self.beta_a)
                self.beta_b = max(0.5, self.beta_b)
        
        self.logger.info(f"[{self.request_id}] Restarted priors with increased uncertainty (factor={uncertainty_factor})")
        # Update rate limit with adjusted priors
        self._update_rate_limit_from_priors()

    def _initialize_bayesian_priors(self) -> None:
        """
        Initialize Bayesian priors based on domain knowledge and the current request history.
        
        This method uses common API rate limit patterns to set informed priors for both
        time period (Gamma-Poisson) and success probability (Beta-Binomial) models.
        """
        if len(self.entries) < 2:
            self.logger.warning(f"[{self.request_id}] Not enough entries to initialize priors")
            return
        
        # Use domain knowledge - start with the "default" common rate limit
        default_requests, default_period = self.common_rate_limits["default"]
        implied_lambda = default_requests / default_period  # Implied rate from domain
        
        # Initialize Gamma prior for time period rate (lambda)
        self.poisson_alpha = 5.0  # Moderate prior strength
        self.poisson_beta = 5.0 / implied_lambda  # Scale to match implied rate
        
        # Initialize Beta prior for success probability in time window
        self.beta_a = default_requests  # Prior successes
        self.beta_b = 2.0  # Prior failures for slight conservatism
        
        self.logger.debug(f"[{self.request_id}] Initialized Bayesian priors with domain knowledge: "
                    f"Gamma({self.poisson_alpha:.4f}, {self.poisson_beta:.4f}), "
                    f"Beta({self.beta_a:.4f}, {self.beta_b:.4f})")
        
        # Update rate limit with initial estimates
        self._update_rate_limit_from_priors()

    def _update_bayesian_estimates(self, limit_exceeded: bool) -> None:
        """
        Update Bayesian estimates based on the latest request outcome.
        Incorporates both successful and failed requests in time period estimation
        with a principled variance-based weighting for failed requests.
        
        Args:
            limit_exceeded (bool): Whether the rate limit was exceeded.
        """
        if self.poisson_alpha is None or self.poisson_beta is None or \
           self.beta_a is None or self.beta_b is None:
            self.logger.warning(f"[{self.request_id}] Bayesian priors not initialized")
            return
        
        if len(self.entries) < 2:
            self.logger.warning(f"[{self.request_id}] Not enough data for Bayesian update: {len(self.entries)} entries")
            return
        
        # Calculate inter-arrival time for time period estimation
        latest = self.entries[-1].timestamp
        previous = self.entries[-2].timestamp
        time_diff = max(0.001, min((latest - previous).total_seconds(), self.max_time_period))
        
        # Determine weight for time period update based on variance ratio for failed requests
        weight = 1.0
        if limit_exceeded:
            # Extract recent inter-arrival times for successful and all requests (up to last 50 for efficiency)
            recent_entries = self.entries[-51:-1] if len(self.entries) > 51 else self.entries[:-1]
            if len(recent_entries) >= 5:  # Need sufficient data for variance estimation
                success_times = []
                all_times = []
                for ix in range(len(recent_entries) - 1):
                    t1 = recent_entries[ix].timestamp
                    t2 = recent_entries[ix+1].timestamp
                    diff = (t2 - t1).total_seconds()
                    all_times.append(diff)
                    if recent_entries[ix+1].success:
                        success_times.append(diff)
                
                # Calculate variances with regularization for stability
                if len(success_times) >= 3 and len(all_times) >= 3:
                    success_mean = sum(success_times) / len(success_times)
                    success_var = sum((t - success_mean)**2 for t in success_times) / len(success_times)
                    success_var += self.regularization_strength * (success_mean**2)  # Regularization
                    
                    all_mean = sum(all_times) / len(all_times)
                    all_var = sum((t - all_mean)**2 for t in all_times) / len(all_times)
                    all_var += self.regularization_strength * (all_mean**2)  # Regularization
                    
                    if all_var > 0:
                        weight = success_var / all_var
                        weight = max(0.3, min(0.7, weight))  # Bound weight to reasonable range
                    else:
                        weight = 0.5  # Fallback if variance calculation fails
                else:
                    weight = 0.5  # Fallback for insufficient data
            else:
                weight = 0.5  # Fallback for small sample size
            
            self.logger.debug(f"[{self.request_id}] Calculated weight for failed request: {weight:.3f} based on variance ratio")
        else:
            self.logger.debug(f"[{self.request_id}] Using full weight for successful request: {weight:.3f}")
        
        # Update Gamma distribution parameters for time period estimation
        self.poisson_alpha += weight
        self.poisson_beta += weight * time_diff
        
        # Update Beta distribution parameters for success probability (Beta-Binomial model)
        if limit_exceeded:
            self.beta_b += 1  # One more failure
        else:
            self.beta_a += 1  # One more success
        
        # Ensure numerical stability of parameters
        self.poisson_alpha = max(1.0, self.poisson_alpha)
        self.poisson_beta = max(1.0 / self.max_time_period, self.poisson_beta)
        self.beta_a = max(0.5, self.beta_a)
        self.beta_b = max(0.5, self.beta_b)
        
        self.logger.debug(f"[{self.request_id}] Updated Bayesian estimates: "
                    f"Gamma(alpha={self.poisson_alpha:.4f}, beta={self.poisson_beta:.4f}), "
                    f"Beta(a={self.beta_a:.4f}, b={self.beta_b:.4f})")
        
        # Update rate limit with new estimates
        self._update_rate_limit_from_priors()

    def _update_rate_limit_from_priors(self) -> None:
        """
        Update the rate limit based on current Bayesian priors.
        Uses posterior statistics and credible intervals for a principled
        calculation of confidence and conservative rate limiting bounds.
        """
        if self.poisson_alpha is None or self.poisson_beta is None or \
           self.beta_a is None or self.beta_b is None:
            return
        
        # Calculate credible intervals using proper statistical functions
        lower_p = 0.025  # 95% credible interval
        upper_p = 0.975
        
        # Time period estimation (inverse of rate)
        try:
            lambda_mean = self.poisson_alpha / self.poisson_beta if self.poisson_beta > 0 else 1.0
            lambda_lower = stats.gamma.ppf(lower_p, self.poisson_alpha, scale=1/self.poisson_beta)
            lambda_upper = stats.gamma.ppf(upper_p, self.poisson_alpha, scale=1/self.poisson_beta)
            
            # Convert rate to time period (inverse)
            time_period_mean = 1.0 / lambda_mean
            time_period_lower = 1.0 / lambda_upper if lambda_upper > 0 else self.min_time_period
            time_period_upper = 1.0 / lambda_lower if lambda_lower > 0 else self.max_time_period
            
            # Ensure values are within bounds
            time_period_lower = max(self.min_time_period, time_period_lower)
            time_period_upper = min(self.max_time_period, time_period_upper)
            time_period = max(self.min_time_period, min(self.max_time_period, time_period_mean))
        except (ValueError, ZeroDivisionError) as e:
            self.logger.warning(f"[{self.request_id}] Gamma credible interval calculation failed: {e}, using fallback")
            lambda_mean = self.poisson_alpha / self.poisson_beta if self.poisson_beta > 0 else 1.0
            time_period_mean = 1.0 / lambda_mean
            # Fallback approximation using standard deviation
            lambda_var = self.poisson_alpha / (self.poisson_beta ** 2) if self.poisson_beta > 0 else (lambda_mean / 2) ** 2
            lambda_std = math.sqrt(lambda_var)
            time_period_lower = max(self.min_time_period, 1.0 / (lambda_mean + 2 * lambda_std))
            time_period_upper = min(self.max_time_period, 1.0 / max(0.001, lambda_mean - 2 * lambda_std))
            time_period = time_period_mean
        
        # Max requests estimation using observed inter-arrival time
        try:
            p_mean = self.beta_a / (self.beta_a + self.beta_b)
            p_lower = stats.beta.ppf(lower_p, self.beta_a, self.beta_b)
            p_upper = stats.beta.ppf(upper_p, self.beta_a, self.beta_b)
            
            # Calculate average inter-arrival time from recent successful requests
            recent_successes = [e for e in self.entries[-20:] if e.success]
            if len(recent_successes) >= 2:
                timestamps = [e.timestamp for e in recent_successes]
                timestamps.sort()
                time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
                avg_delta_t = sum(time_diffs) / len(time_diffs) if time_diffs else time_period
            else:
                avg_delta_t = time_period
                
            # Estimate max requests as expected successes in time window
            max_requests_mean = math.floor(p_mean * (time_period / avg_delta_t)) if avg_delta_t > 0 else 1
            max_requests_lower = math.floor(p_lower * (time_period / avg_delta_t)) if avg_delta_t > 0 else 1
            max_requests_upper = math.floor(p_upper * (time_period / avg_delta_t)) if avg_delta_t > 0 else 1
            
            # Ensure values are within reasonable bounds
            max_requests = max(1, min(10000, max_requests_mean))
            max_requests_lower = max(1, max_requests_lower)
            max_requests_upper = max(max_requests_lower + 1, max_requests_upper)
        except (ValueError, ZeroDivisionError) as e:
            self.logger.warning(f"[{self.request_id}] Beta credible interval calculation failed: {e}, using fallback")
            p_mean = self.beta_a / (self.beta_a + self.beta_b) if (self.beta_a + self.beta_b) > 0 else 0.5
            p_var = (self.beta_a * self.beta_b) / (((self.beta_a + self.beta_b) ** 2) * (self.beta_a + self.beta_b + 1))
            p_std = math.sqrt(p_var) if p_var >= 0 else 0.1
            
            avg_delta_t = time_period  # Fallback to time period
            max_requests_mean = math.floor(p_mean * (time_period / avg_delta_t)) if avg_delta_t > 0 else 1
            max_requests_lower = max(1, math.floor((p_mean - 2 * p_std) * (time_period / avg_delta_t))) if avg_delta_t > 0 else 1
            max_requests_upper = max(max_requests_lower + 1, math.floor((p_mean + 2 * p_std) * (time_period / avg_delta_t))) if avg_delta_t > 0 else 1
            max_requests = max_requests_mean
        
        # Calculate confidence based on credible interval width using harmonic mean
        time_rel_width = (time_period_upper - time_period_lower) / time_period if time_period > 0 else 1.0
        req_rel_width = (max_requests_upper - max_requests_lower) / max_requests if max_requests > 0 else 1.0
        # Harmonic mean of inverse widths for balanced uncertainty measure
        confidence = 2.0 / (1.0 / (time_rel_width + 1e-10) + 1.0 / (req_rel_width + 1e-10))
        confidence = min(0.99, max(0.01, confidence))
        
        self.rate_limit = RateLimit(
            endpoint=self.endpoint,
            method=self.method,
            max_requests=max_requests,
            time_period=time_period,
            confidence=confidence,
            last_updated=datetime.now(timezone.utc),
            time_period_lower=time_period_lower,
            time_period_upper=time_period_upper,
            max_requests_lower=max_requests_lower,
            max_requests_upper=max_requests_upper
        )
        
        self.logger.debug(f"[{self.request_id}] Updated rate limit using credible intervals: "
                    f"time_period={time_period:.4f} [{time_period_lower:.4f}, {time_period_upper:.4f}], "
                    f"max_requests={max_requests} [{max_requests_lower}, {max_requests_upper}], "
                    f"confidence={confidence:.4f}")

    def _check_estimation_confidence(self) -> bool:
        """
        Check if the estimation confidence is high enough to consider it complete.
        Uses a statistically principled approach based on credible interval width
        and sample size (observation count).
        
        Returns:
            bool: True if estimation is complete, False otherwise.
        """
        if not self.rate_limit:
            return False
        
        # Calculate a principled adaptive threshold based on statistical considerations
        base_threshold = self.confidence_threshold
        
        # Adjust based on the number of observations (more observations => can require higher confidence)
        observation_count = 0
        if self.poisson_alpha is not None:
            # Poisson_alpha approximately tracks the effective number of observations
            observation_count = max(0, self.poisson_alpha - 1)
        
        # Adjust threshold based on sample size - statistical principle that
        # larger samples allow more precise estimation
        if observation_count < 5:
            # With very few observations, be more lenient to allow progress
            adaptive_threshold = max(0.6, base_threshold - 0.1)
        elif observation_count > 20:
            # With many observations, be more strict since we should have precise estimates
            adaptive_threshold = min(0.95, base_threshold + 0.05)
        else:
            # Linear interpolation for intermediate sample sizes
            adaptive_threshold = base_threshold
        
        # Further adjust based on restart history
        # With more restarts, require higher confidence before declaring completion
        restart_adjustment = min(0.1, 0.02 * self.restart_count)
        adaptive_threshold = min(0.98, adaptive_threshold + restart_adjustment)
        
        self.logger.debug(f"[{self.request_id}] Checking estimation confidence: current={self.rate_limit.confidence:.4f}, "
                    f"threshold={adaptive_threshold:.4f} (base={base_threshold:.4f}, "
                    f"observations={observation_count:.1f}, restarts={self.restart_count})")
        
        # Consider estimation complete if confidence is above threshold
        return self.rate_limit.confidence >= adaptive_threshold

    def _enforce_rate_limit(self) -> None:
        """
        Enforce the estimated rate limit by waiting if necessary.
        Uses the lower bound of max_requests for conservative rate limiting.
        """
        if not self.rate_limit or not self.entries:
            return
        
        # Calculate how many requests we've made in the last time_period
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=self.rate_limit.time_period)
        recent_requests = [e for e in self.entries if e.timestamp >= cutoff_time]
        
        # Use lower bound of max_requests for conservatism
        effective_max_requests = self.rate_limit.max_requests_lower if self.rate_limit.max_requests_lower else self.rate_limit.max_requests
        
        if len(recent_requests) >= effective_max_requests:
            # Need to wait until oldest request is outside the time period
            if recent_requests:
                oldest_time = min(e.timestamp for e in recent_requests)
                wait_time = (oldest_time + timedelta(seconds=self.rate_limit.time_period) - datetime.now(timezone.utc)).total_seconds() + 0.1
                if wait_time > 0:
                    self.logger.info(f"[{self.request_id}] Enforcing rate limit: waiting {wait_time:.2f} seconds (using conservative max_requests={effective_max_requests})")
                    time.sleep(wait_time)

    def _enforce_bayesian_estimate(self) -> None:
        """
        Enforce rate limiting based on current Bayesian estimates during the ESTIMATING phase.
        Uses the credible interval to determine a conservative wait time.
        """
        if self.poisson_alpha is None or self.poisson_beta is None or not self.entries:
            return
        
        # Use a conservative estimate based on the credible interval
        # Less uncertainty (higher alpha) => less conservatism needed
        
        # Determine conservatism level based on amount of data
        # With less data (lower alpha), be more conservative
        conservatism_level = max(0.7, 0.9 - min(0.2, (self.poisson_alpha - 1) / 20))
        
        # Calculate percentile to use (higher = more conservative)
        # With low alpha or many restarts, use a higher percentile
        percentile = conservatism_level
        
        try:
            # Use scipy for proper statistical calculation of conservative lambda (low percentile)
            conservative_lambda = stats.gamma.ppf(1.0 - percentile, self.poisson_alpha, scale=1/self.poisson_beta)
            conservative_time = 1.0 / conservative_lambda if conservative_lambda > 0 else self.max_time_period
        except Exception as e:
            # Fallback if statistical calculation fails
            self.logger.warning(f"[{self.request_id}] Statistical calculation for Bayesian estimate failed: {e}, using fallback")
            mean_lambda = self.poisson_alpha / self.poisson_beta if self.poisson_beta > 0 else 1.0
            lambda_var = self.poisson_alpha / (self.poisson_beta ** 2) if self.poisson_beta > 0 else mean_lambda / 4
            lambda_std = math.sqrt(lambda_var)
            conservative_lambda = max(0.0001, mean_lambda - 2 * lambda_std)  # 2 std dev lower
            conservative_time = 1.0 / conservative_lambda
        
        # Cap at max_time_period
        conservative_time = min(conservative_time, self.max_time_period)
        
        # Get time since last request
        last_request_time = self.entries[-1].timestamp
        time_since_last = (datetime.now(timezone.utc) - last_request_time).total_seconds()
        
        # Wait if needed, with a small buffer
        wait_time = max(0, conservative_time - time_since_last)
        if wait_time > 0:
            self.logger.info(f"[{self.request_id}] Waiting {wait_time:.2f} seconds based on Bayesian estimate "
                       f"(using {percentile:.2f} percentile of posterior)")
            time.sleep(wait_time)
