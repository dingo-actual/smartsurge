"""
Core models and enums for the SmartSurge library.

This module defines the data models and enumerations used throughout the
SmartSurge library, providing structured representation of requests, 
rate limits, and search status.
"""
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timedelta, timezone
from enum import Enum
import logging
import time
import uuid

import numpy as np
from scipy import stats
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


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

class HMMParams(BaseModel):
    """
    Parameters for the Hidden Markov Model used in rate limit estimation.
    
    Attributes:
        n_states: Number of hidden states in the model
        initial_probs: Initial state probabilities
        transition_matrix: State transition probability matrix
        success_probs: Bernoulli parameters for request outcome per state
        rate_lambdas: Poisson parameters for rate limit per state
    """
    n_states: int = Field(default=3, ge=2, le=10, description="Number of hidden states")
    initial_probs: Optional[np.ndarray] = Field(default=None, description="Initial state probabilities")
    transition_matrix: Optional[np.ndarray] = Field(default=None, description="State transition probability matrix")
    success_probs: Optional[np.ndarray] = Field(default=None, description="Bernoulli parameters for request outcome per state")
    rate_lambdas: Optional[np.ndarray] = Field(default=None, description="Poisson parameters for rate limit per state")
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    
    @field_validator('initial_probs', 'transition_matrix', 'success_probs', 'rate_lambdas', mode='before')
    def validate_array_dimensions(cls, v, info):
        """Validate array dimensions are consistent with n_states."""
        if v is None:
            return v
        
        n_states = info.data.get('n_states', 3)
        
        if isinstance(v, np.ndarray):
            if v.ndim == 1 and len(v) != n_states:
                raise ValueError(f"1D array must have length {n_states}, got {len(v)}")
            elif v.ndim == 2 and v.shape != (n_states, n_states):
                raise ValueError(f"2D array must have shape ({n_states}, {n_states}), got {v.shape}")
        
        return v
    
    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        """Custom model_dump to handle numpy arrays."""
        result = super().model_dump(*args, **kwargs)
        # Convert numpy arrays to lists for serialization
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
        return result

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

class HMM(BaseModel):
    """
    Hidden Markov Model for rate limit estimation.
    
    This class implements a Hidden Markov Model with:
    - Hidden states representing different traffic load levels (normal, approaching limit, rate limited)
    - Emissions consisting of request outcomes (success/failure) and rate limits
    - Request outcome emissions follow a Bernoulli distribution with parameter determined by state
    - Rate limit emissions follow a shifted Poisson distribution (rate_limit ~ 1 + Poisson(λ))
    
    Attributes:
        params: Parameters for the HMM
        logger: Logger instance for this HMM
    """
    params: HMMParams = Field(default_factory=HMMParams)
    logger: Optional[logging.Logger] = Field(default=None, exclude=True)
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    
    def __init__(self, **data):
        """
        Initialize an HMM instance with appropriate logger and parameters.
        
        Args:
            **data: Data to initialize the model with, may include an optional 'logger'.
        """
        super().__init__(**data)
        
        # Set up logger if not provided
        if self.logger is None:
            self.logger = logger.getChild(f"HMM.{id(self)}")
            
        # Initialize model parameters if not provided
        self._initialize_parameters()
    
    def _initialize_parameters(self) -> None:
        """
        Initialize HMM parameters with reasonable defaults if not already set.
        
        The states are conceptualized as:
        - State 0: Normal operation (high success probability)
        - State 1: Approaching rate limit (medium success probability)
        - State 2: Rate limited (low success probability)
        """
        try:
            n_states = self.params.n_states
            
            # Initialize initial state probabilities
            if self.params.initial_probs is None:
                self.params.initial_probs = np.random.dirichlet(alpha=n_states)
                self.logger.debug(f"Initialized initial state probabilities: {self.params.initial_probs}")
                
            # Initialize transition matrix with a tendency to stay in the same state
            if self.params.transition_matrix is None:
                self.params.transition_matrix = np.exp(np.random.normal(size=(n_states, n_states)))
                self.params.transition_matrix /= self.params.transition_matrix.sum(axis=1, keepdims=True)
                self.logger.debug(f"Initialized transition matrix shape: {self.params.transition_matrix.shape}")
                
            # Initialize success probabilities for each state
            if self.params.success_probs is None:
                self.params.success_probs = np.random.rand(n_states)
                self.logger.debug(f"Initialized success probabilities: {self.params.success_probs}")
                
            # Initialize rate limit Poisson parameters for each state
            if self.params.rate_lambdas is None:
                self.params.rate_lambdas = np.sort(np.random.exponential(size=n_states))[::-1]
                self.logger.debug(f"Initialized rate limit Poisson parameters: {self.params.rate_lambdas}")
                
            # Validate dimensions
            if len(self.params.initial_probs) != n_states:
                raise ValueError(f"Initial probabilities dimension {len(self.params.initial_probs)} does not match n_states {n_states}")
            
            if self.params.transition_matrix.shape != (n_states, n_states):
                raise ValueError(f"Transition matrix shape {self.params.transition_matrix.shape} does not match (n_states, n_states) ({n_states}, {n_states})")
            
            if len(self.params.success_probs) != n_states:
                raise ValueError(f"Success probabilities dimension {len(self.params.success_probs)} does not match n_states {n_states}")
            
            if len(self.params.rate_lambdas) != n_states:
                raise ValueError(f"Rate lambdas dimension {len(self.params.rate_lambdas)} does not match n_states {n_states}")
                
            # Ensure probabilities sum to 1
            self.params.initial_probs = self.params.initial_probs / np.sum(self.params.initial_probs)
            for i in range(n_states):
                self.params.transition_matrix[i] = self.params.transition_matrix[i] / np.sum(self.params.transition_matrix[i])
                
            # Ensure probabilities are in valid range
            self.params.success_probs = np.clip(self.params.success_probs, 1e-10, 1.0 - 1e-10)
            self.params.rate_lambdas = np.clip(self.params.rate_lambdas, 1e-3, 1e3)
            
        except Exception as e:
            self.logger.error(f"Error initializing HMM parameters: {e}")
            # Set safe defaults
            self.params = HMMParams(
                n_states=3,
                initial_probs=np.array([0.8, 0.15, 0.05]),
                transition_matrix=np.array([
                    [0.85, 0.14, 0.01],
                    [0.20, 0.75, 0.05],
                    [0.10, 0.30, 0.60],
                ]),
                success_probs=np.array([0.99, 0.70, 0.20]),
                rate_lambdas=np.array([5.0, 2.0, 0.5])
            )
            self.logger.warning(f"Using safe default parameters after initialization error")
    
    def emission_probability(self, outcome: bool, rate_limit: int, state: int) -> float:
        """
        Calculate the emission probability for a given observation and state.
        
        Args:
            outcome: Boolean indicating request success (True) or failure (False)
            rate_limit: Observed rate limit (requests per time interval)
            state: Hidden state index
            
        Returns:
            float: Emission probability P(observation | state)
        """
        try:
            # Ensure state is valid
            n_states = self.params.n_states
            if state < 0 or state >= n_states:
                self.logger.error(f"Invalid state index: {state}")
                return 1e-10  # Small non-zero probability to avoid numerical issues
            
            # Calculate Bernoulli probability for request outcome
            p_outcome = self.params.success_probs[state] if outcome else (1 - self.params.success_probs[state])
            
            # Calculate shifted Poisson probability for rate limit
            # rate_limit ~ 1 + Poisson(λ), so we shift by -1 for the Poisson calculation
            shifted_rate = max(0, rate_limit - 1)
            p_rate = stats.poisson.pmf(shifted_rate, self.params.rate_lambdas[state])
            
            # Combined emission probability (assuming conditional independence)
            emission_prob = p_outcome * p_rate
            
            # Avoid numerical underflow
            if emission_prob < 1e-10:
                emission_prob = 1e-10
                
            return emission_prob
            
        except Exception as e:
            self.logger.error(f"Error calculating emission probability: {e}")
            return 1e-10  # Return small non-zero probability to avoid numerical issues
    
    def forward_backward(self, observations: List[Tuple[bool, int]]) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Implement the forward-backward algorithm for HMM inference.
        
        Args:
            observations: List of (outcome, rate_limit) tuples
            
        Returns:
            Tuple containing:
                - alpha: Forward probabilities (T x n_states)
                - beta: Backward probabilities (T x n_states)
                - log_likelihood: Log-likelihood of the observations
        """
        try:
            T = len(observations)
            n_states = self.params.n_states
            
            if T == 0:
                self.logger.warning("Empty observation sequence provided to forward_backward")
                return np.zeros((0, n_states)), np.zeros((0, n_states)), -np.inf
            
            # Initialize forward and backward variables in log space
            log_alpha = np.zeros((T, n_states))
            log_beta = np.zeros((T, n_states))
            
            # Forward pass (alpha) in log space
            for j in range(n_states):
                log_alpha[0, j] = np.log(max(self.params.initial_probs[j], 1e-10)) + \
                                  np.log(max(self.emission_probability(
                                      observations[0][0], observations[0][1], j), 1e-10))
            
            # Recursive forward calculation
            for t in range(1, T):
                for j in range(n_states):
                    # Log sum exp trick for numerical stability
                    log_sum = np.log(sum(np.exp(log_alpha[t-1, i]) * self.params.transition_matrix[i, j] 
                                   for i in range(n_states)))
                    log_alpha[t, j] = log_sum + np.log(max(self.emission_probability(
                        observations[t][0], observations[t][1], j), 1e-10))
            
            # Initialize backward pass
            for j in range(n_states):
                log_beta[T-1, j] = 0  # log(1) = 0
            
            # Backward pass (beta) in log space
            for t in range(T-2, -1, -1):
                for i in range(n_states):
                    log_sum = np.log(sum(
                        self.params.transition_matrix[i, j] * 
                        self.emission_probability(observations[t+1][0], observations[t+1][1], j) * 
                        np.exp(log_beta[t+1, j])
                        for j in range(n_states)
                    ))
                    log_beta[t, i] = log_sum
            
            # Calculate log-likelihood from alpha values at the final step
            log_likelihood = np.log(sum(np.exp(log_alpha[T-1, j]) for j in range(n_states)))
            
            # Convert back from log space
            alpha = np.exp(log_alpha)
            beta = np.exp(log_beta)
            
            self.logger.debug(f"Forward-backward completed with log-likelihood: {log_likelihood:.4f}")
            return alpha, beta, log_likelihood
            
        except Exception as e:
            self.logger.error(f"Error in forward_backward algorithm: {e}")
            # Return safe values
            return (
                np.ones((max(1, T), n_states)) / n_states,
                np.ones((max(1, T), n_states)) / n_states,
                -np.inf
            )
    
    def viterbi(self, observations: List[Tuple[bool, int]]) -> List[int]:
        """
        Implement the Viterbi algorithm to find the most likely state sequence.
        
        Args:
            observations: List of (outcome, rate_limit) tuples
            
        Returns:
            List[int]: Most likely sequence of hidden states
        """
        try:
            T = len(observations)
            n_states = self.params.n_states
            
            if T == 0:
                self.logger.warning("Empty observation sequence provided to viterbi")
                return []
            
            # Initialize variables in log space for numerical stability
            log_delta = np.zeros((T, n_states))
            psi = np.zeros((T, n_states), dtype=int)
            
            # Initialize first step
            for j in range(n_states):
                log_delta[0, j] = np.log(max(self.params.initial_probs[j], 1e-10)) + np.log(max(
                    self.emission_probability(observations[0][0], observations[0][1], j), 1e-10
                ))
            
            # Recursion
            for t in range(1, T):
                for j in range(n_states):
                    # Find the most likely previous state
                    log_probs = log_delta[t-1] + np.log(np.maximum(self.params.transition_matrix[:, j], 1e-10))
                    psi[t, j] = np.argmax(log_probs)
                    log_delta[t, j] = log_probs[psi[t, j]] + np.log(max(
                        self.emission_probability(observations[t][0], observations[t][1], j), 1e-10
                    ))
            
            # Backtracking
            q_star = np.zeros(T, dtype=int)
            q_star[T-1] = np.argmax(log_delta[T-1])
            
            for t in range(T-2, -1, -1):
                q_star[t] = psi[t+1, q_star[t+1]]
            
            self.logger.debug(f"Viterbi algorithm completed, found most likely state sequence")
            return q_star.tolist()
            
        except Exception as e:
            self.logger.error(f"Error in Viterbi algorithm: {e}")
            # Return a safe default sequence
            return [0] * max(0, T)
    
    def baum_welch(self, observations: List[Tuple[bool, int]], max_iter: int = 100, tol: float = 1e-4) -> float:
        """
        Implement the Baum-Welch algorithm (EM for HMMs) to learn model parameters.
        
        Args:
            observations: List of (outcome, rate_limit) tuples
            max_iter: Maximum number of iterations
            tol: Convergence tolerance for log-likelihood
            
        Returns:
            float: Final log-likelihood
        """
        try:
            T = len(observations)
            n_states = self.params.n_states
            
            if T < 2:
                self.logger.warning("Insufficient data for Baum-Welch algorithm (need at least 2 observations)")
                return -np.inf
            
            self.logger.info(f"Starting Baum-Welch algorithm with {T} observations, max_iter={max_iter}, tol={tol}")
            
            prev_log_likelihood = -np.inf
            
            for iteration in range(max_iter):
                # E-step: Calculate forward-backward variables
                alpha, beta, log_likelihood = self.forward_backward(observations)
                
                # Check for convergence
                if abs(log_likelihood - prev_log_likelihood) < tol and iteration > 0:
                    self.logger.info(f"Baum-Welch converged after {iteration+1} iterations with log-likelihood {log_likelihood:.4f}")
                    return log_likelihood
                
                prev_log_likelihood = log_likelihood
                
                # Calculate state probabilities and transition counts
                gamma = alpha * beta
                # Avoid division by zero
                row_sums = np.sum(gamma, axis=1, keepdims=True)
                row_sums = np.maximum(row_sums, 1e-10)
                gamma = gamma / row_sums
                
                xi = np.zeros((T-1, n_states, n_states))
                for t in range(T-1):
                    denominator = 0.0
                    for i in range(n_states):
                        for j in range(n_states):
                            emission_prob = self.emission_probability(
                                observations[t+1][0], observations[t+1][1], j
                            )
                            xi[t, i, j] = alpha[t, i] * self.params.transition_matrix[i, j] * \
                                        emission_prob * beta[t+1, j]
                            denominator += xi[t, i, j]
                    
                    # Normalize xi to prevent numerical issues
                    if denominator > 1e-10:
                        xi[t] /= denominator
                
                # M-step: Update model parameters
                # Update initial state probabilities
                self.params.initial_probs = gamma[0]
                
                # Update transition matrix
                for i in range(n_states):
                    denominator = np.sum(gamma[:-1, i])
                    if denominator > 1e-10:
                        self.params.transition_matrix[i] = np.sum(xi[:, i, :], axis=0) / denominator
                
                # Update emission parameters
                for j in range(n_states):
                    # Update success probabilities (Bernoulli parameter)
                    success_indices = [t for t, obs in enumerate(observations) if obs[0]]
                    if success_indices:
                        success_gamma_sum = np.sum(gamma[success_indices, j])
                        total_gamma_sum = np.sum(gamma[:, j])
                        
                        if total_gamma_sum > 1e-10:
                            self.params.success_probs[j] = success_gamma_sum / total_gamma_sum
                    
                    # Update rate limit parameters (Poisson lambda)
                    weighted_sum = 0
                    weight_sum = 0
                    for t, (_, rate) in enumerate(observations):
                        shifted_rate = max(0, rate - 1)  # Shift back for Poisson
                        weighted_sum += gamma[t, j] * shifted_rate
                        weight_sum += gamma[t, j]
                    
                    if weight_sum > 1e-10:
                        self.params.rate_lambdas[j] = weighted_sum / weight_sum
                
                # Ensure parameters remain valid
                self.params.initial_probs = np.maximum(self.params.initial_probs, 1e-10)
                self.params.initial_probs = self.params.initial_probs / np.sum(self.params.initial_probs)
                
                for i in range(n_states):
                    self.params.transition_matrix[i] = np.maximum(self.params.transition_matrix[i], 1e-10)
                    self.params.transition_matrix[i] = self.params.transition_matrix[i] / np.sum(self.params.transition_matrix[i])
                
                self.params.success_probs = np.clip(self.params.success_probs, 0.01, 0.99)
                self.params.rate_lambdas = np.clip(self.params.rate_lambdas, 0.1, 100.0)
                
                self.logger.debug(f"Iteration {iteration+1}: log-likelihood={log_likelihood:.4f}")
            
            self.logger.info(f"Baum-Welch reached max iterations ({max_iter}) with log-likelihood {log_likelihood:.4f}")
            return log_likelihood
            
        except Exception as e:
            self.logger.error(f"Error in Baum-Welch algorithm: {e}")
            return -np.inf

    def predict_rate_limit(self, observations: List[Tuple[bool, int]]) -> Tuple[int, float, float]:
        """
        Predict the rate limit based on the current model and observations.
        
        Args:
            observations: List of (outcome, rate_limit) tuples
            
        Returns:
            Tuple containing:
            - max_requests: Maximum number of requests allowed in the time period
            - time_period: Time period in seconds
            - confidence: Confidence level in the prediction (0.0-1.0)
        """
        try:
            if not observations:
                self.logger.warning("Empty observation sequence for rate limit prediction")
                return 1, 1.0, 0.0
            
            # Find the most likely state sequence
            state_sequence = self.viterbi(observations)
            
            if not state_sequence:
                self.logger.warning("Empty state sequence from Viterbi algorithm")
                return 1, 1.0, 0.0
            
            # Analyze the state distribution in recent observations
            recent_states = state_sequence[-min(len(state_sequence), 10):]
            state_counts = np.zeros(self.params.n_states)
            for state in recent_states:
                state_counts[state] += 1
            
            # Normalize to get state probabilities
            state_probs = state_counts / len(recent_states)
            
            # Calculate confidence based on state entropy
            # Lower entropy = higher confidence
            entropy = -np.sum(state_probs * np.log2(np.maximum(state_probs, 1e-10)))
            max_entropy = np.log2(self.params.n_states)  # Maximum possible entropy
            confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.5
            
            # Calculate the expected rate limit based on state probabilities
            # For max_requests, we use a weighted average of the rate_lambdas
            expected_lambda = np.sum(state_probs * self.params.rate_lambdas)
            
            # Convert to max_requests (add 1 for the shift in the Poisson)
            max_requests = max(1, int(np.ceil(expected_lambda + 1)))
            
            # For time period, we use a fixed value of 1.0 second
            # This simplifies the model and makes the rate limit interpretable as "requests per second"
            time_period = 1.0
            
            self.logger.debug(f"Predicted rate limit: {max_requests} requests per {time_period:.2f}s with confidence {confidence:.2f}")
            return max_requests, time_period, confidence
            
        except Exception as e:
            self.logger.error(f"Error in rate limit prediction: {e}")
            return 1, 1.0, 0.0  # Safe default with low confidence

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
