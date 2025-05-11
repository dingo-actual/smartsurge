"""
Core models and enums for the SmartSurge library.

This module defines the data models and enumerations used throughout the
SmartSurge library, providing structured representation of requests, 
rate limits, and search status.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone
from enum import Enum
import logging
import time
import uuid

from pydantic import BaseModel, Field, model_validator, ConfigDict

from .hmm import HMM


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
    
    def __str__(self) -> str:
        return self.value

class SearchStatus(str, Enum):
    """Status of the rate limit search process."""
    NOT_STARTED = "NOT_STARTED"
    WAITING_TO_ESTIMATE = "WAITING_TO_ESTIMATE"
    COMPLETED = "COMPLETED"
    
    def __str__(self) -> str:
        return self.value

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
        max_requests: Optional parameter indicating maximum requests allowed
        max_request_period: Optional parameter indicating the period for max_requests in seconds
    """
    endpoint: str = Field(min_length=1, description="The endpoint that was requested")
    method: RequestMethod = Field(description="HTTP method used")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When the request was made (UTC)")
    status_code: int = Field(ge=0, le=599, description="HTTP status code received")
    response_time: float = Field(ge=0.0, description="Time taken to receive response in seconds")
    success: bool = Field(description="Whether the request was successful")
    max_requests: Optional[int] = Field(default=None, ge=1, description="Maximum requests allowed if specified")
    max_request_period: Optional[float] = Field(default=None, gt=0.0, description="Period for max_requests in seconds")
    response_headers: Optional[Dict[str, Any]] = Field(default=None, description="Response headers for additional rate limit details")

    @model_validator(mode="after")
    def validate_success_code_consistency(self) -> "RequestEntry":
        """Ensure success flag is consistent with status code."""
        if self.success and self.status_code >= 400:
            logger.warning(f"Inconsistent success flag: marked as success but status code is {self.status_code}")
        
        # Ensure max_request_period is provided if max_requests is set
        if self.max_requests is not None and self.max_request_period is None:
            logger.warning(f"max_requests provided without max_request_period, ignoring rate limit info")
            self.max_requests = None
        
        # Try to extract rate limit info from headers if provided
        if self.response_headers and not self.max_requests:
            # Look for common rate limit headers
            rate_limit_headers = {
                'X-RateLimit-Limit': None,
                'X-RateLimit-Remaining': None,
                'X-RateLimit-Reset': None,
                'Retry-After': None,
                'X-Rate-Limit': None,
                'RateLimit-Limit': None,
                'RateLimit-Remaining': None,
                'RateLimit-Reset': None,
            }
            
            # Check if any rate limit headers are present
            for header in rate_limit_headers:
                if header.lower() in {k.lower() for k in self.response_headers.keys()}:
                    # Extract values using case-insensitive lookup
                    for rh in self.response_headers:
                        if rh.lower() == header.lower():
                            rate_limit_headers[header] = self.response_headers[rh]
            
            # Try to determine rate limits from headers
            if rate_limit_headers['X-RateLimit-Limit'] or rate_limit_headers['RateLimit-Limit']:
                try:
                    limit = int(rate_limit_headers['X-RateLimit-Limit'] or rate_limit_headers['RateLimit-Limit'])
                    # Assume a default period of 60 seconds if not specified
                    self.max_requests = limit
                    self.max_request_period = 60.0
                    logger.debug(f"Extracted rate limit from headers: {limit} requests per 60s")
                except (ValueError, TypeError):
                    pass
                    
        return self

class RateLimit(BaseModel):
    """
    Rate limit information for an endpoint, estimated using HMM.

    Attributes:
        endpoint: The API endpoint this rate limit applies to
        method: The HTTP method this rate limit applies to
        max_requests: Maximum number of requests allowed in the time period
        time_period: Time period in seconds for the rate limit window
        last_updated: When this rate limit was last updated
        cooldown: Optional cooldown period in seconds before next request
        time_cooldown_set: Optional timestamp when the cooldown was set
    """
    endpoint: str = Field(..., min_length=1, description="The endpoint this rate limit applies to")
    method: RequestMethod = Field(..., description="HTTP method this applies to")
    max_requests: int = Field(..., ge=1, description="Maximum number of requests allowed in the time period")
    time_period: float = Field(..., gt=0.0, description="Time period in seconds for the rate limit window")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When this rate limit was last updated")
    cooldown: Optional[float] = Field(default=None, gt=0.0, description="Cooldown period in seconds before next request")
    time_cooldown_set: Optional[datetime] = Field(default=None, description="Timestamp when the cooldown was set")
    source: str = Field(default="estimated", description="Source of the rate limit (estimated, headers, manual)")

    def __str__(self) -> str:
        """Return a human-readable string representation of the rate limit."""
        cooldown_info = f", cooldown: {self.cooldown:.2f}s" if self.cooldown is not None else ""
        return f"RateLimit({self.max_requests} requests per {self.time_period:.2f}s{cooldown_info}, source: {self.source})"

    def get_requests_per_second(self) -> float:
        """Get the rate limit as requests per second for easier comparison."""
        return self.max_requests / self.time_period if self.time_period > 0 else 0

class RequestHistory(BaseModel):
    """
    Tracks request logs and estimates rate limits for a single endpoint and method combination
    using a Hidden Markov Model approach.
    
    Features:
    - HMM with states representing different traffic levels
    - Dual emissions for request outcome and rate limits
    - Viterbi algorithm for state sequence decoding
    - Baum-Welch algorithm for parameter learning
    - Automatic rate limit estimation based on state analysis
    - Exponential backoff for successive refusals
    - Limited observation history to conserve memory
    
    Attributes:
        endpoint: The endpoint being tracked
        method: The HTTP method being tracked
        entries: List of request entries, limited to max_observations
        rate_limit: The current estimated rate limit
        search_status: Current status of the rate limit search
        min_time_period: Minimum time period to consider for rate limiting (seconds)
        max_time_period: Maximum time period to consider for rate limiting (seconds)
        confidence_threshold: Minimum confidence level required for estimation
        min_data_points: Minimum number of data points needed before estimation
        max_observations: Maximum number of observations to store in memory
        consecutive_refusals: Count of consecutive request refusals (for exponential backoff)
        request_id: Unique ID for tracking this history's requests
        hmm: The Hidden Markov Model used for estimation
        logger: Custom logger for this instance
    """
    endpoint: str = Field(..., min_length=1)
    method: RequestMethod
    entries: List[RequestEntry] = Field(default_factory=list)
    rate_limit: Optional[RateLimit] = None
    search_status: SearchStatus = SearchStatus.NOT_STARTED
    min_time_period: float = Field(default=1.0, gt=0.0)
    max_time_period: float = Field(default=3600.0, gt=0.0)
    confidence_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    min_data_points: int = Field(default=10, ge=5, le=100)
    max_observations: int = Field(default=50, ge=20, le=1000)
    consecutive_refusals: int = Field(default=0, ge=0)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], exclude=True)
    hmm: Optional[HMM] = None
    logger: Optional[logging.Logger] = Field(default=None, exclude=True)
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    
    def __init__(self, **data):
        """
        Initialize a RequestHistory instance with appropriate logger and HMM.
        
        Args:
            **data: Data to initialize the model with, may include an optional 'logger'.
        """
        super().__init__(**data)
        
        # Set up logger if not provided
        if self.logger is None:
            self.logger = logger.getChild(f"RequestHistory.{self.endpoint}.{self.method}")
        
        # Initialize HMM if not provided
        if self.hmm is None:
            self.hmm = HMM(logger=self.logger.getChild("HMM"))
    
    def add_request(self, entry: RequestEntry) -> None:
        """
        Add a request entry to the history, limiting to max_observations.
        
        Args:
            entry: The request entry to add
            
        Raises:
            ValueError: If the entry's endpoint or method does not match this history
        """
        self.logger.debug(f"[{self.request_id}] Adding request entry for {self.endpoint}/{self.method}: {entry}")
        
        if entry.endpoint != self.endpoint or entry.method != self.method:
            self.logger.error(f"[{self.request_id}] Entry endpoint/method mismatch: {entry.endpoint}/{entry.method} vs {self.endpoint}/{self.method}")
            raise ValueError("Entry endpoint and method must match RequestHistory")
        
        # Directly set rate limit if entry provides max_requests and max_request_period
        if entry.max_requests is not None and entry.max_request_period is not None:
            self.logger.info(f"[{self.request_id}] Setting rate limit directly from request entry: {entry.max_requests} requests per {entry.max_request_period}s")
            self.rate_limit = RateLimit(
                endpoint=self.endpoint,
                method=self.method,
                max_requests=entry.max_requests,
                time_period=entry.max_request_period,
                last_updated=datetime.now(timezone.utc),
                source="headers"
            )
            self.search_status = SearchStatus.COMPLETED
        
        # Add entry to history
        self.entries.append(entry)
        self.entries.sort(key=lambda e: e.timestamp)  # Keep sorted by timestamp
        
        # Limit the number of observations
        if len(self.entries) > self.max_observations:
            removed = len(self.entries) - self.max_observations
            self.entries = self.entries[removed:]
            self.logger.debug(f"[{self.request_id}] Removed {removed} oldest entries to maintain max_observations limit")
    
    def has_minimum_observations(self) -> bool:
        """
        Check if there are at least min_data_points observations with at least 
        one success and one failure.
        
        Returns:
            bool: True if minimum observation criteria are met
        """
        if len(self.entries) < self.min_data_points:
            return False
        
        has_success = any(entry.success for entry in self.entries)
        has_failure = any(not entry.success for entry in self.entries)
        
        return has_success and has_failure
    
    def merge(self, other: 'RequestHistory') -> None:
        """
        Merge another RequestHistory into this one, preserving sorting by timestamp.
        
        Args:
            other: Another RequestHistory object with the same endpoint and method
            
        Raises:
            ValueError: If the other history has a different endpoint or method
        """
        self.logger.debug(f"[{self.request_id}] Merging RequestHistory with {len(other.entries)} entries")
        
        if self.endpoint != other.endpoint or self.method != other.method:
            self.logger.error(f"[{self.request_id}] Cannot merge RequestHistory with different endpoint/method: {self.endpoint}/{self.method} vs {other.endpoint}/{other.method}")
            raise ValueError("Can only merge RequestHistory objects with the same endpoint and method")
        
        # Combine entries, ensuring they remain sorted by timestamp
        combined_entries = self.entries + other.entries
        combined_entries.sort(key=lambda e: e.timestamp)
        
        # Apply maximum observations limit
        if len(combined_entries) > self.max_observations:
            self.entries = combined_entries[-self.max_observations:]
            self.logger.debug(f"[{self.request_id}] Limited merged entries to {self.max_observations} most recent observations")
        else:
            self.entries = combined_entries
        
        # Merge rate limits, preferring the more recent one
        if other.rate_limit:
            if (not self.rate_limit) or (other.rate_limit.last_updated > self.rate_limit.last_updated):
                self.rate_limit = other.rate_limit
        
        # Update search status to the furthest along
        status_order = {
            SearchStatus.NOT_STARTED: 0,
            SearchStatus.WAITING_TO_ESTIMATE: 1,
            SearchStatus.COMPLETED: 2
        }
        
        if status_order[other.search_status] > status_order[self.search_status]:
            self.search_status = other.search_status
        
        # Take the higher consecutive refusals count
        self.consecutive_refusals = max(self.consecutive_refusals, other.consecutive_refusals)
        
        self.logger.info(f"[{self.request_id}] Merged histories for {self.endpoint} {self.method}, now with {len(self.entries)} entries and search status {self.search_status}")
    
    def intercept_request(self) -> None:
        """
        Intercept a request to enforce rate limit search procedure.
        
        This method should be called before making a request.
        It waits as necessary based on the current search status and rate limits.
        """
        self.logger.debug(f"[{self.request_id}] Intercepting request with search status: {self.search_status}")
        
        # Check if we need to enforce a cooldown period
        if self.rate_limit and self.rate_limit.cooldown is not None and self.rate_limit.time_cooldown_set is not None:
            time_since_cooldown = (datetime.now(timezone.utc) - self.rate_limit.time_cooldown_set).total_seconds()
            
            if time_since_cooldown < self.rate_limit.cooldown:
                wait_time = self.rate_limit.cooldown - time_since_cooldown
                self.logger.info(f"[{self.request_id}] Enforcing cooldown: waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
            
            # Reset cooldown after waiting
            self.rate_limit.cooldown = None
            self.rate_limit.time_cooldown_set = None
            self.logger.debug(f"[{self.request_id}] Cooldown period completed and reset")
        
        if self.search_status == SearchStatus.COMPLETED:
            # Use the estimated rate limit for throttling
            if self.rate_limit:
                self._enforce_rate_limit()
            return
        
        if self.search_status == SearchStatus.NOT_STARTED:
            # Initialize the search status
            self.search_status = SearchStatus.WAITING_TO_ESTIMATE
            self.logger.debug(f"[{self.request_id}] Search status set to WAITING_TO_ESTIMATE")
            return
        
        if self.search_status == SearchStatus.WAITING_TO_ESTIMATE:
            # No waiting needed, we're still collecting data
            return
    
    def log_response_and_update(self, entry: RequestEntry) -> None:
        """
        Log the response and update the search status and HMM estimates.
        
        This method should be called after receiving a response.
        
        Args:
            entry: The request entry to log
        """
        self.logger.debug(f"[{self.request_id}] Logging response and updating search status: {entry}")
        
        # Track consecutive refusals for exponential backoff
        if entry.status_code == 429:
            self.consecutive_refusals += 1
            self.logger.debug(f"[{self.request_id}] Consecutive refusal count: {self.consecutive_refusals}")
            
            # Set cooldown with exponential backoff if multiple consecutive refusals
            if self.consecutive_refusals > 1 and self.rate_limit:
                backoff_seconds = min(60.0, 2.0 ** (self.consecutive_refusals - 1))
                self.rate_limit.cooldown = backoff_seconds
                self.rate_limit.time_cooldown_set = datetime.now(timezone.utc)
                self.logger.warning(f"[{self.request_id}] Setting exponential backoff: {backoff_seconds:.2f} seconds after {self.consecutive_refusals} consecutive refusals")
        else:
            # Reset consecutive refusals counter
            if self.consecutive_refusals > 0:
                self.logger.debug(f"[{self.request_id}] Resetting consecutive refusal count from {self.consecutive_refusals} to 0")
                self.consecutive_refusals = 0
        
        self.add_request(entry)
        
        # Handle rate limit error (HTTP 429) - special case when search is COMPLETED
        if entry.status_code == 429 and self.search_status == SearchStatus.COMPLETED:
            if self.has_minimum_observations():
                self.logger.warning(f"[{self.request_id}] Rate limit error received after estimation was completed! "
                                   f"Recalculating HMM parameters for {self.endpoint} {self.method}")
                
                # Update the HMM with the new data
                self._update_hmm()
            else:
                self.logger.warning(f"[{self.request_id}] Rate limit error received but insufficient data for HMM estimation. "
                                   f"Transitioning to WAITING_TO_ESTIMATE for {self.endpoint} {self.method}")
                self.search_status = SearchStatus.WAITING_TO_ESTIMATE
            
            return
        
        if self.search_status == SearchStatus.NOT_STARTED:
            self.logger.info(f"[{self.request_id}] First request received, collecting data for {self.endpoint} {self.method}")
            self.search_status = SearchStatus.WAITING_TO_ESTIMATE
            return
        
        if self.search_status == SearchStatus.WAITING_TO_ESTIMATE:
            # Check if we have enough data to start estimation
            if self.has_minimum_observations():
                self.logger.info(f"[{self.request_id}] Collected {len(self.entries)} data points with required success/failure mix, starting HMM estimation for {self.endpoint} {self.method}")
                self._update_hmm()
                self.search_status = SearchStatus.COMPLETED
            
            return
    
    def _update_hmm(self) -> None:
        """
        Update the HMM with the current data and estimate a new rate limit.
        
        Requires at least min_data_points observations with at least one success and one failure.
        """
        try:
            if not self.has_minimum_observations():
                self.logger.debug(f"[{self.request_id}] Not enough data to update HMM: need {self.min_data_points} entries with at least one success and one failure")
                return
            
            # Prepare observations for the HMM
            observations = []
            
            # Extract rate information from the entries
            for i in range(1, len(self.entries)):
                # Calculate requests per second based on time difference
                prev_time = self.entries[i-1].timestamp
                curr_time = self.entries[i].timestamp
                time_diff = max(0.001, (curr_time - prev_time).total_seconds())
                
                # Rate is requests per second
                rate = int(1.0 / time_diff) if time_diff > 0 else 1
                
                # Add observation (success, rate)
                observations.append((self.entries[i].success, rate))
            
            if not observations:
                self.logger.warning(f"[{self.request_id}] No valid observations for HMM update")
                return
            
            self.logger.debug(f"[{self.request_id}] Updating HMM with {len(observations)} observations")
            
            # Train the HMM using Baum-Welch
            log_likelihood = self.hmm.baum_welch(observations, max_iter=20, tol=1e-3)
            
            self.logger.info(f"[{self.request_id}] HMM training completed with log-likelihood: {log_likelihood:.4f}")
            
            # Predict rate limit with confidence
            max_requests, time_period, confidence = self.hmm.predict_rate_limit(observations)
            
            # Only update if confidence meets threshold
            if confidence >= self.confidence_threshold:
                # Update the rate limit
                self.rate_limit = RateLimit(
                    endpoint=self.endpoint,
                    method=self.method,
                    max_requests=max_requests,
                    time_period=time_period,
                    last_updated=datetime.now(timezone.utc),
                    source="estimated"
                )
                self.logger.info(f"[{self.request_id}] Updated rate limit: {self.rate_limit} with confidence {confidence:.2f}")
            else:
                self.logger.warning(f"[{self.request_id}] Rate limit prediction confidence {confidence:.2f} below threshold {self.confidence_threshold}, "
                                   f"setting conservative default rate limit")
                # Set a conservative default rate limit
                self.rate_limit = RateLimit(
                    endpoint=self.endpoint,
                    method=self.method,
                    max_requests=1,
                    time_period=2.0,  # 1 request per 2 seconds (conservative)
                    last_updated=datetime.now(timezone.utc),
                    source="fallback"
                )
            
        except Exception as e:
            self.logger.error(f"[{self.request_id}] Error updating HMM: {e}")
            
            # Set a very conservative fallback rate limit
            self.rate_limit = RateLimit(
                endpoint=self.endpoint,
                method=self.method,
                max_requests=1,
                time_period=5.0,  # 1 request per 5 seconds (very conservative)
                last_updated=datetime.now(timezone.utc),
                source="error_fallback"
            )
            self.logger.warning(f"[{self.request_id}] Set fallback rate limit after error: {self.rate_limit}")
    
    def _enforce_rate_limit(self) -> None:
        """
        Enforce the estimated rate limit by waiting if necessary.
        """
        if not self.rate_limit or not self.entries:
            return
        
        # Calculate how many requests we've made in the last time_period
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=self.rate_limit.time_period)
        recent_requests = [e for e in self.entries if e.timestamp >= cutoff_time]
        
        if len(recent_requests) >= self.rate_limit.max_requests:
            # Need to wait until oldest request is outside the time period
            if recent_requests:
                oldest_time = min(e.timestamp for e in recent_requests)
                wait_time = (oldest_time + timedelta(seconds=self.rate_limit.time_period) - datetime.now(timezone.utc)).total_seconds() + 0.1
                
                if wait_time > 0:
                    self.logger.info(f"[{self.request_id}] Enforcing rate limit: waiting {wait_time:.2f} seconds (max_requests={self.rate_limit.max_requests}, time_period={self.rate_limit.time_period}s)")
                    time.sleep(wait_time)
