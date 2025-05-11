from typing import List, Optional, Tuple, Dict, Any
import logging

import numpy as np
from scipy import stats
from pydantic import BaseModel, Field, field_validator, ConfigDict


# Module-level logger
logger = logging.getLogger(__name__)

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