from typing import List, Optional, Tuple, Dict, Any
import logging

import numpy as np
from scipy import stats
from scipy.optimize import minimize
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
    initial_probs: Optional[np.ndarray] = Field(
        default=None, description="Initial state probabilities"
    )
    transition_matrix: Optional[np.ndarray] = Field(
        default=None, description="State transition probability matrix"
    )
    success_probs: Optional[np.ndarray] = Field(
        default=None, description="Bernoulli parameters for request outcome per state"
    )
    rate_lambdas: Optional[np.ndarray] = Field(
        default=None, description="Poisson parameters for rate limit per state"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @field_validator(
        "initial_probs",
        "transition_matrix",
        "success_probs",
        "rate_lambdas",
        mode="before",
    )
    def validate_array_dimensions(cls, v, info):
        """Validate array dimensions are consistent with n_states."""
        if v is None:
            return v

        n_states = info.data.get("n_states", 3)

        if isinstance(v, np.ndarray):
            if v.ndim == 1 and len(v) != n_states:
                raise ValueError(f"1D array must have length {n_states}, got {len(v)}")
            elif v.ndim == 2 and v.shape != (n_states, n_states):
                raise ValueError(
                    f"2D array must have shape ({n_states}, {n_states}), got {v.shape}"
                )

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
    - MLE-based parameter fitting

    Attributes:
        params: Parameters for the HMM
        logger: Logger instance for this HMM
    """

    params: HMMParams = Field(default_factory=HMMParams)
    logger: Optional[logging.Logger] = Field(default=None, exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def __init__(
        self,
        params: Optional[HMMParams] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        """Initialize HMM with optional parameters and logger."""
        if params is None:
            params = HMMParams()

        if logger is None:
            logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        super().__init__(params=params, logger=logger, **kwargs)

        self.logger.debug(f"Initializing HMM with {self.params.n_states} states")
        self.logger.debug(f"Additional kwargs: {kwargs}")

        # Initialize parameters if not provided
        if self.params.initial_probs is None:
            self.logger.debug("No initial parameters provided, initializing randomly")
            self.initialize_parameters()
        else:
            self.logger.debug("Using provided parameters")
            self.logger.debug(
                f"Initial probs shape: {self.params.initial_probs.shape if self.params.initial_probs is not None else None}"
            )
            self.logger.debug(
                f"Transition matrix shape: {self.params.transition_matrix.shape if self.params.transition_matrix is not None else None}"
            )

    def initialize_parameters(self):
        """Initialize HMM parameters with random values."""
        n_states = self.params.n_states
        self.logger.debug(f"Initializing parameters for {n_states}-state HMM")

        pseudo_counts = np.ones(n_states)
        pseudo_counts[0] = 5
        self.logger.debug(f"Dirichlet pseudo-counts: {pseudo_counts}")

        # Initial state probabilities
        self.params.initial_probs = np.sort(np.random.dirichlet(pseudo_counts))[::-1]
        self.logger.debug(
            f"Generated initial probabilities: {self.params.initial_probs}"
        )

        # Transition matrix
        transition_probs = [
            np.sort(np.random.dirichlet(pseudo_counts))[::-1] for _ in range(n_states)
        ]
        self.logger.debug(f"Raw transition probabilities: {transition_probs}")

        transition_probs = [
            np.concatenate([vec[-ix:], vec[:-ix]])
            for ix, vec in enumerate(transition_probs)
        ]
        self.params.transition_matrix = np.concatenate(
            list(map(lambda vec: vec.reshape(1, -1), transition_probs)), axis=0
        )
        self.logger.debug(
            f"Transition matrix after circular shift:\n{self.params.transition_matrix}"
        )

        # Success probabilities per state
        beta_params = np.linspace(5.0, 0.25, n_states)
        self.logger.debug(f"Beta distribution parameters: {beta_params}")
        success_probs = np.array([np.random.beta(a, 1.0) for a in beta_params])
        # Clip to ensure we don't start with extreme values
        success_probs = np.clip(success_probs, 0.05, 0.95)
        self.params.success_probs = np.sort(success_probs)[::-1]
        self.logger.debug(f"Success probabilities: {self.params.success_probs}")

        # Rate lambdas for shifted Poisson (rate_limit ~ 1 + Poisson(λ))
        # Use a more reasonable range for initialization
        raw_lambdas = np.random.exponential(5.0, size=n_states) + 0.5
        # Clip to reasonable range
        raw_lambdas = np.clip(raw_lambdas, 0.1, 50.0)
        self.params.rate_lambdas = np.sort(raw_lambdas)[::-1]
        self.logger.debug(
            f"Rate lambdas (Poisson parameters): {self.params.rate_lambdas}"
        )

        # Validate parameters
        if (self.params.initial_probs <= 0).any() or not np.isclose(
            np.sum(self.params.initial_probs), 1.0
        ):
            self.logger.warning(
                f"Invalid initial probabilities: {self.params.initial_probs}. "
                f"Sum: {np.sum(self.params.initial_probs)}"
            )

        for i in range(n_states):
            row_sum = np.sum(self.params.transition_matrix[i])
            if not np.isclose(row_sum, 1.0):
                self.logger.warning(
                    f"Transition matrix row {i} does not sum to 1: {row_sum}"
                )

        self.logger.debug(
            f"Initialized parameters for {n_states}-state HMM:\n"
            f"  Initial probs: {self.params.initial_probs}\n"
            f"  Transition matrix:\n{self.params.transition_matrix}\n"
            f"  Success probs: {self.params.success_probs}\n"
            f"  Rate lambdas: {self.params.rate_lambdas}"
        )

    def forward_algorithm(
        self, observations: List[Tuple[bool, int]]
    ) -> Tuple[np.ndarray, float]:
        """
        Implement the forward algorithm to compute the likelihood of observations.

        Args:
            observations: List of (outcome, rate_limit) tuples

        Returns:
            Tuple of (alpha matrix, log-likelihood)
        """
        n_states = self.params.n_states
        n_obs = len(observations)

        self.logger.debug(f"Forward algorithm: {n_obs} observations, {n_states} states")
        self.logger.debug(f"First 5 observations: {observations[:5]}")

        # Initialize forward variables (in log space to avoid underflow)
        log_alpha = np.zeros((n_obs, n_states))

        # Initialize first time step
        for i in range(n_states):
            log_emission = self.log_emission_probability(observations[0], i)
            log_alpha[0, i] = np.log(self.params.initial_probs[i]) + log_emission

        self.logger.debug(f"Initial log_alpha[0]: {log_alpha[0]}")

        # Forward pass
        for t in range(1, n_obs):
            for j in range(n_states):
                # Compute log-sum-exp for numerical stability
                log_transition_probs = []
                for i in range(n_states):
                    log_trans = log_alpha[t - 1, i] + np.log(
                        self.params.transition_matrix[i, j]
                    )
                    log_transition_probs.append(log_trans)

                log_emission = self.log_emission_probability(observations[t], j)
                log_alpha[t, j] = self._log_sum_exp(log_transition_probs) + log_emission

            if t < 5 or t == n_obs - 1:
                self.logger.debug(f"log_alpha[{t}]: {log_alpha[t]}")

        # Compute log-likelihood
        log_likelihood = self._log_sum_exp(log_alpha[-1, :])

        # Check for numerical issues
        if np.isnan(log_likelihood) or log_likelihood == -np.inf:
            self.logger.warning(
                f"Forward algorithm produced invalid log-likelihood: {log_likelihood}. "
                f"Final alpha values: {log_alpha[-1, :]}"
            )

        return log_alpha, log_likelihood

    def backward_algorithm(self, observations: List[Tuple[bool, int]]) -> np.ndarray:
        """
        Implement the backward algorithm.

        Args:
            observations: List of (outcome, rate_limit) tuples

        Returns:
            Beta matrix (backward probabilities)
        """
        n_states = self.params.n_states
        n_obs = len(observations)

        self.logger.debug(
            f"Backward algorithm: {n_obs} observations, {n_states} states"
        )

        # Initialize backward variables (in log space)
        log_beta = np.zeros((n_obs, n_states))

        # Initialize last time step
        log_beta[-1, :] = 0  # log(1) = 0

        # Backward pass
        for t in range(n_obs - 2, -1, -1):
            for i in range(n_states):
                log_terms = []
                for j in range(n_states):
                    log_emission = self.log_emission_probability(observations[t + 1], j)
                    log_term = (
                        np.log(self.params.transition_matrix[i, j])
                        + log_emission
                        + log_beta[t + 1, j]
                    )
                    log_terms.append(log_term)

                log_beta[t, i] = self._log_sum_exp(log_terms)

        self.logger.debug(
            f"Backward algorithm complete. log_beta[0] range: "
            f"[{np.min(log_beta[0]):.4f}, {np.max(log_beta[0]):.4f}]"
        )

        return log_beta

    def viterbi_algorithm(
        self, observations: List[Tuple[bool, int]]
    ) -> Tuple[List[int], float]:
        """
        Implement the Viterbi algorithm to find the most likely state sequence.

        Args:
            observations: List of (outcome, rate_limit) tuples

        Returns:
            Tuple of (state_sequence, log_probability)
        """
        n_states = self.params.n_states
        n_obs = len(observations)

        self.logger.debug(f"Viterbi algorithm: {n_obs} observations, {n_states} states")

        # Initialize Viterbi variables (in log space)
        log_delta = np.zeros((n_obs, n_states))
        psi = np.zeros((n_obs, n_states), dtype=int)

        # Initialize first time step
        for i in range(n_states):
            log_emission = self.log_emission_probability(observations[0], i)
            log_delta[0, i] = np.log(self.params.initial_probs[i]) + log_emission

        self.logger.debug(f"Initial log_delta[0]: {log_delta[0]}")

        # Forward pass
        for t in range(1, n_obs):
            for j in range(n_states):
                # Find the maximum over all possible previous states
                log_candidates = []
                for i in range(n_states):
                    log_candidate = log_delta[t - 1, i] + np.log(
                        self.params.transition_matrix[i, j]
                    )
                    log_candidates.append(log_candidate)

                log_emission = self.log_emission_probability(observations[t], j)
                max_idx = np.argmax(log_candidates)
                log_delta[t, j] = log_candidates[max_idx] + log_emission
                psi[t, j] = max_idx

        # Backtrack to find the most likely state sequence
        state_sequence = np.zeros(n_obs, dtype=int)
        state_sequence[-1] = np.argmax(log_delta[-1, :])
        max_log_prob = log_delta[-1, state_sequence[-1]]

        self.logger.debug(f"Final log_delta: {log_delta[-1, :]}")
        self.logger.debug(f"Most likely final state: {state_sequence[-1]}")
        self.logger.debug(f"Max log probability: {max_log_prob}")

        for t in range(n_obs - 2, -1, -1):
            state_sequence[t] = psi[t + 1, state_sequence[t + 1]]

        self.logger.debug(f"Most likely state sequence: {state_sequence.tolist()}")

        # Count state occurrences
        state_counts = np.bincount(state_sequence, minlength=n_states)
        self.logger.debug(f"State counts: {dict(enumerate(state_counts))}")

        return state_sequence.tolist(), max_log_prob

    def log_emission_probability(
        self, observation: Tuple[bool, int], state: int
    ) -> float:
        """
        Compute log emission probability for an observation given a state.

        Args:
            observation: Tuple of (outcome, rate_limit) where:
                - outcome: True if request succeeded, False if rate limited
                - rate_limit: Estimated rate limit (requests per period)
            state: Hidden state index

        Returns:
            Log probability of the observation given the state
        """
        outcome, rate_limit = observation

        # Compute Bernoulli probability for request outcome
        success_prob = self.params.success_probs[state]

        # Defensive check (should not happen with our bounds)
        if success_prob <= 0 or success_prob >= 1:
            self.logger.debug(
                f"Edge case: success probability {success_prob} for state {state}. "
                f"Clamping to [0.001, 0.999]"
            )
            success_prob = np.clip(success_prob, 0.001, 0.999)
            self.params.success_probs[state] = success_prob

        if outcome:
            log_outcome_prob = np.log(success_prob)
        else:
            log_outcome_prob = np.log(1 - success_prob)

        # Compute shifted Poisson probability for rate limit
        # Model: rate_limit ~ 1 + Poisson(λ)
        # So we compute P(rate_limit - 1 | λ)
        lambda_param = self.params.rate_lambdas[state]

        # Defensive check (should not happen with our bounds)
        if lambda_param <= 0 or np.isnan(lambda_param):
            self.logger.debug(
                f"Edge case: lambda parameter {lambda_param} for state {state}. "
                f"Setting to 1.0"
            )
            lambda_param = 1.0
            self.params.rate_lambdas[state] = lambda_param

        shifted_rate = max(0, rate_limit - 1)  # Ensure non-negative

        # Use Poisson PMF
        log_rate_prob = stats.poisson.logpmf(shifted_rate, lambda_param)

        # Handle edge cases
        if np.isnan(log_rate_prob) or np.isinf(log_rate_prob):
            # For very large rate limits or lambda values, use a small probability
            self.logger.debug(
                f"Edge case in emission probability: state={state}, rate_limit={rate_limit}, "
                f"lambda={lambda_param}, log_rate_prob={log_rate_prob}"
            )
            log_rate_prob = -10.0  # log(0.0000454)

        total_log_prob = log_outcome_prob + log_rate_prob

        if (
            self.logger.isEnabledFor(logging.DEBUG) and np.random.random() < 0.01
        ):  # Log 1% of emissions to avoid spam
            self.logger.debug(
                f"Emission: obs=({outcome}, {rate_limit}), state={state}, "
                f"log_outcome={log_outcome_prob:.3f}, log_rate={log_rate_prob:.3f}, "
                f"total={total_log_prob:.3f}"
            )

        return total_log_prob

    def _log_sum_exp(self, log_values: List[float]) -> float:
        """
        Compute log(sum(exp(log_values))) in a numerically stable way.

        Args:
            log_values: List of log values

        Returns:
            log(sum(exp(log_values)))
        """
        if len(log_values) == 0:
            return -np.inf

        max_val = max(log_values)
        if max_val == -np.inf:
            return -np.inf

        sum_exp = sum(np.exp(log_val - max_val) for log_val in log_values)
        return max_val + np.log(sum_exp)

    def fit(
        self,
        observations: List[Tuple[bool, int]],
        max_iter: int = 1000,
        tol: float = 1e-6,
        n_restarts: int = 10,
        method: str = "baum_welch",
    ) -> float:
        """Fit HMM parameters

        Args:
            observations (List[Tuple[bool, int]]): List of (outcome, rate_limit) tuples
            max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
            tol (float, optional): Tolerance for convergence. Defaults to 1e-6.
            n_restarts (int, optional): Number of random restarts. Defaults to 10.
            method (str, optional): Method to use for fitting. Can be "baum_welch" or "l-bfgs-b". Defaults to "baum_welch".

        Returns:
            float: Log-likelihood of the best model after n_restarts random restarts
        """
        self.logger.debug(
            f"Fitting HMM with method={method}, max_iter={max_iter}, tol={tol}, n_restarts={n_restarts}"
        )
        self.logger.debug(f"Number of observations: {len(observations)}")

        if len(observations) < 2:
            self.logger.warning("Not enough observations for MLE fitting")
            return -np.inf

        # Analyze observations
        outcomes = [outcome for outcome, _ in observations]
        rate_limits = [rate for _, rate in observations]
        success_rate = sum(outcomes) / len(outcomes)

        self.logger.debug(f"Success rate: {success_rate:.2%}")
        self.logger.debug(f"Rate limit range: [{min(rate_limits)}, {max(rate_limits)}]")
        self.logger.debug(
            f"Rate limit mean: {np.mean(rate_limits):.2f}, std: {np.std(rate_limits):.2f}"
        )

        if all(outcome for outcome, _ in observations):
            self.logger.warning("All observations are successful, cannot fit HMM")
            return -np.inf

        if all(not outcome for outcome, _ in observations):
            self.logger.warning("All observations are rate limited, cannot fit HMM")
            return -np.inf

        if method not in ["baum_welch", "l-bfgs-b"]:
            raise ValueError("Invalid method. Use 'baum_welch' or 'l-bfgs-b'.")

        self.logger.info(f"Starting HMM fitting with {method} method")

        if method == "baum_welch":
            return self._fit_baum_welch(observations, max_iter, tol, n_restarts)
        elif method == "l-bfgs-b":
            return self._fit_l_bfgs_b(observations, max_iter, tol, n_restarts)
        else:
            raise ValueError("Invalid method. Use 'baum_welch' or 'l-bfgs-b'.")

    def _fit_l_bfgs_b(
        self,
        observations: List[Tuple[bool, int]],
        max_iter: int = 1000,
        tol: float = 1e-6,
        n_restarts: int = 10,
    ) -> float:
        """
        Fit HMM parameters using Maximum Likelihood Estimation (MLE).

        Uses multiple random restarts to avoid local optima.

        Args:
            observations: List of (outcome, rate_limit) tuples
            n_restarts: Number of random restarts

        Returns:
            Best log-likelihood achieved
        """
        best_log_likelihood = -np.inf
        best_params = None

        self.logger.info(
            f"Starting MLE fitting with {len(observations)} observations, {n_restarts} restarts"
        )

        for restart in range(n_restarts):
            try:
                # Random initialization for restarts > 0
                if restart > 0:
                    self.initialize_parameters()

                # Define the negative log-likelihood function to minimize
                def neg_log_likelihood(params_vector):
                    # Unpack parameters
                    self._unpack_parameters(params_vector)

                    # Compute forward algorithm
                    _, log_likelihood = self.forward_algorithm(observations)

                    # Return negative for minimization
                    return -log_likelihood

                # Pack current parameters into a vector
                initial_params = self._pack_parameters()

                # Set bounds for parameters
                bounds = self._get_parameter_bounds()

                # Minimize negative log-likelihood
                result = minimize(
                    neg_log_likelihood,
                    initial_params,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": max_iter, "ftol": tol},
                )

                if result.success:
                    # Unpack the optimized parameters
                    self._unpack_parameters(result.x)

                    # Compute final log-likelihood
                    _, log_likelihood = self.forward_algorithm(observations)

                    self.logger.debug(
                        f"Restart {restart + 1}: Optimization succeeded. "
                        f"Iterations: {result.nit}, Function evals: {result.nfev}, "
                        f"Final log-likelihood: {log_likelihood:.4f}"
                    )

                    if log_likelihood > best_log_likelihood:
                        best_log_likelihood = log_likelihood
                        best_params = self._pack_parameters()

                        self.logger.debug(
                            f"Restart {restart + 1}: New best log-likelihood = {log_likelihood:.4f}"
                        )
                        self.logger.debug(f"Best parameters updated")
                    else:
                        self.logger.debug(
                            f"Restart {restart + 1}: Log-likelihood {log_likelihood:.4f} not better than {best_log_likelihood:.4f}"
                        )
                else:
                    self.logger.warning(
                        f"Restart {restart + 1}: Optimization failed - {result.message}. "
                        f"Iterations: {result.nit}, Status: {result.status}"
                    )

            except Exception as e:
                self.logger.warning(f"Restart {restart + 1} failed with exception: {e}")
                continue

        if best_params is not None:
            # Set the best parameters found
            self._unpack_parameters(best_params)

            self.logger.info(
                f"MLE fitting completed with log-likelihood {best_log_likelihood:.4f}"
            )
            return best_log_likelihood
        else:
            self.logger.error(
                f"MLE fitting failed for all {n_restarts} restarts. "
                f"Best log-likelihood achieved: {best_log_likelihood}. "
                "Check warnings above for specific failure reasons."
            )
            return -np.inf

    def _fit_baum_welch(
        self,
        observations: List[Tuple[bool, int]],
        max_iter: int = 1000,
        tol: float = 1e-6,
        n_restarts: int = 5,
    ) -> float:
        """
        Implement the Baum-Welch algorithm (EM for HMMs) to learn model parameters.
        Uses multiple random restarts to avoid local optima.

        Args:
            observations: List of (outcome, rate_limit) tuples
            max_iter: Maximum number of iterations per restart
            tol: Convergence tolerance for log-likelihood
            n_restarts: Number of random restarts

        Returns:
            Best log-likelihood achieved across all restarts
        """
        n_states = self.params.n_states
        n_obs = len(observations)

        if n_obs < 2:
            self.logger.warning("Not enough observations for Baum-Welch")
            return -np.inf

        best_log_likelihood = -np.inf
        best_params = None

        self.logger.info(
            f"Starting Baum-Welch with {len(observations)} observations, {n_restarts} restarts"
        )

        for restart in range(n_restarts):
            # Random initialization for restarts > 0
            if restart > 0:
                self.initialize_parameters()

            # Store initial parameters for this restart
            initial_params = self._pack_parameters()
            prev_log_likelihood = -np.inf

            try:
                for iteration in range(max_iter):
                    # E-step: Compute forward and backward variables
                    log_alpha, log_likelihood = self.forward_algorithm(observations)
                    log_beta = self.backward_algorithm(observations)

                    # Check for convergence
                    improvement = log_likelihood - prev_log_likelihood
                    if abs(improvement) < tol:
                        self.logger.debug(
                            f"Restart {restart + 1}: Baum-Welch converged after {iteration + 1} iterations. "
                            f"Final improvement: {improvement:.6f}"
                        )
                        break

                    if iteration % 10 == 0:
                        self.logger.debug(
                            f"Restart {restart + 1}, Iteration {iteration}: "
                            f"log-likelihood = {log_likelihood:.4f}, improvement = {improvement:.6f}"
                        )

                    prev_log_likelihood = log_likelihood

                    # Compute gamma (state posteriors)
                    log_gamma = np.zeros((n_obs, n_states))
                    for t in range(n_obs):
                        for i in range(n_states):
                            log_gamma[t, i] = (
                                log_alpha[t, i] + log_beta[t, i] - log_likelihood
                            )

                    # Convert from log space
                    gamma = np.exp(log_gamma)

                    # Compute xi (transition posteriors)
                    xi = np.zeros((n_obs - 1, n_states, n_states))
                    for t in range(n_obs - 1):
                        for i in range(n_states):
                            for j in range(n_states):
                                log_emission = self.log_emission_probability(
                                    observations[t + 1], j
                                )
                                log_xi = (
                                    log_alpha[t, i]
                                    + np.log(self.params.transition_matrix[i, j])
                                    + log_emission
                                    + log_beta[t + 1, j]
                                    - log_likelihood
                                )
                                xi[t, i, j] = np.exp(log_xi)

                    # M-step: Update parameters
                    # Update initial probabilities
                    self.params.initial_probs = np.clip(gamma[0], 1e-6, 1.0 - 1e-6)

                    # Update transition matrix
                    for i in range(n_states):
                        for j in range(n_states):
                            numerator = np.sum(xi[:, i, j])
                            denominator = np.sum(gamma[:-1, i])
                            if denominator > 0:
                                self.params.transition_matrix[i, j] = (
                                    numerator / denominator
                                )
                            else:
                                self.params.transition_matrix[i, j] = 1.0 / n_states

                    # Clip to avoid exact 0 or 1 probabilities
                    self.params.transition_matrix = np.clip(
                        self.params.transition_matrix, 1e-6, 1.0 - 1e-6
                    )

                    # Update emission parameters
                    for i in range(n_states):
                        # Update success probability
                        state_weight = np.sum(gamma[:, i])
                        if state_weight > 0:
                            success_weight = sum(
                                gamma[t, i] for t in range(n_obs) if observations[t][0]
                            )
                            # Clip to avoid exact 0 or 1 probabilities
                            raw_prob = success_weight / state_weight
                            self.params.success_probs[i] = np.clip(raw_prob, 0.01, 0.99)

                            # Update rate lambda (using weighted mean of shifted rates)
                            rate_sum = sum(
                                gamma[t, i] * max(0, observations[t][1] - 1)
                                for t in range(n_obs)
                            )
                            # Ensure lambda is positive and reasonable
                            raw_lambda = rate_sum / state_weight
                            self.params.rate_lambdas[i] = max(
                                0.1, min(1000.0, raw_lambda)
                            )

                # Check if this restart achieved better likelihood
                if log_likelihood > best_log_likelihood:
                    best_log_likelihood = log_likelihood
                    best_params = self._pack_parameters()
                    self.logger.debug(
                        f"Restart {restart + 1}: New best log-likelihood = {log_likelihood:.4f}"
                    )
                else:
                    self.logger.debug(
                        f"Restart {restart + 1}: Log-likelihood {log_likelihood:.4f} not better than {best_log_likelihood:.4f}"
                    )

            except Exception as e:
                self.logger.warning(f"Restart {restart + 1} failed with exception: {e}")
                # Restore initial parameters for this restart
                self._unpack_parameters(initial_params)
                continue

        # Set the best parameters found
        if best_params is not None:
            self._unpack_parameters(best_params)
            self.logger.info(
                f"Baum-Welch completed with best log-likelihood {best_log_likelihood:.4f}"
            )
            return best_log_likelihood
        else:
            self.logger.error(
                f"Baum-Welch failed for all {n_restarts} restarts. "
                f"Best log-likelihood achieved: {best_log_likelihood}. "
                "Check warnings above for specific failure reasons."
            )
            return -np.inf

    def predict_rate_limit(
        self, observations: List[Tuple[bool, int]]
    ) -> Tuple[int, float]:
        """
        Predict the rate limit based on observations.

        Args:
            observations: List of (outcome, rate_limit) tuples

        Returns:
            Tuple of (max_requests, time_period)
        """
        if not observations:
            self.logger.debug("No observations provided, returning safe default")
            return 1, 1.0  # Safe default

        self.logger.debug(
            f"Predicting rate limit from {len(observations)} observations"
        )

        try:
            # Get the most likely state sequence using Viterbi
            state_sequence, log_prob = self.viterbi_algorithm(observations)
            self.logger.debug(f"Viterbi log probability: {log_prob:.4f}")

            # Use recent states to estimate current state distribution
            recent_states = state_sequence[-min(10, len(state_sequence)) :]
            state_counts = np.zeros(self.params.n_states)
            for state in recent_states:
                state_counts[state] += 1

            # Normalize to get state probabilities
            state_probs = state_counts / len(recent_states)
            self.logger.debug(
                f"Recent state distribution: {dict(enumerate(state_probs))}"
            )

            # Calculate the expected rate limit based on state probabilities
            # For max_requests, we use a weighted average of the rate_lambdas
            expected_lambda = np.sum(state_probs * self.params.rate_lambdas)

            # Convert to max_requests (add 1 for the shift in the Poisson)
            max_requests = max(1, int(np.ceil(expected_lambda + 1)))

            # For time period, we use a fixed value of 1.0 second
            # This simplifies the model and makes the rate limit interpretable as "requests per second"
            time_period = 1.0

            self.logger.debug(
                f"Predicted rate limit: {max_requests} requests per {time_period:.2f}s"
            )
            return max_requests, time_period

        except Exception as e:
            self.logger.error(f"Error in rate limit prediction: {e}")
            return 1, 1.0  # Safe default

    def _pack_parameters(self) -> np.ndarray:
        """Pack HMM parameters into a single vector for optimization."""
        # Pack parameters in order:
        # 1. Transition matrix (flattened)
        # 2. Initial probabilities
        # 3. Success probabilities
        # 4. Rate lambdas

        params_list = [
            self.params.transition_matrix.flatten(),
            self.params.initial_probs,
            self.params.success_probs,
            self.params.rate_lambdas,
        ]

        packed = np.concatenate(params_list)

        self.logger.debug(
            f"Packed parameters: {len(packed)} values, "
            f"transition={self.params.n_states ** 2}, "
            f"initial={self.params.n_states}, "
            f"success={self.params.n_states}, "
            f"lambdas={self.params.n_states}"
        )

        return packed

    def _unpack_parameters(self, params_vector: np.ndarray):
        """Unpack parameter vector and update HMM parameters."""
        n_states = self.params.n_states
        idx = 0

        self.logger.debug(f"Unpacking {len(params_vector)} parameter values")

        # Unpack transition matrix
        trans_size = n_states * n_states
        trans_flat = params_vector[idx : idx + trans_size]
        self.params.transition_matrix = trans_flat.reshape((n_states, n_states))
        idx += trans_size

        # Normalize rows of transition matrix
        for i in range(n_states):
            row_sum = np.sum(self.params.transition_matrix[i])
            if row_sum > 0:
                self.params.transition_matrix[i] /= row_sum
            else:
                self.logger.debug(
                    f"Zero row sum in transition matrix row {i}, using uniform distribution"
                )
                self.params.transition_matrix[i] = np.ones(n_states) / n_states

        # Unpack initial probabilities
        self.params.initial_probs = params_vector[idx : idx + n_states]
        idx += n_states

        # Normalize initial probabilities
        prob_sum = np.sum(self.params.initial_probs)
        if prob_sum > 0:
            self.params.initial_probs /= prob_sum
        else:
            self.logger.debug(
                "Zero sum in initial probabilities, using uniform distribution"
            )
            self.params.initial_probs = np.ones(n_states) / n_states

        # Unpack success probabilities
        self.params.success_probs = params_vector[idx : idx + n_states]
        # Ensure success probabilities are within valid range
        self.params.success_probs = np.clip(self.params.success_probs, 0.01, 0.99)
        idx += n_states

        # Unpack rate lambdas
        self.params.rate_lambdas = params_vector[idx : idx + n_states]
        # Ensure lambdas are positive and reasonable
        self.params.rate_lambdas = np.clip(self.params.rate_lambdas, 0.1, 1000.0)

        self.logger.debug(
            f"Unpacked parameters - transition matrix norm: {np.linalg.norm(self.params.transition_matrix):.4f}, "
            f"initial probs sum: {np.sum(self.params.initial_probs):.4f}, "
            f"success probs range: [{np.min(self.params.success_probs):.3f}, {np.max(self.params.success_probs):.3f}], "
            f"lambda range: [{np.min(self.params.rate_lambdas):.2f}, {np.max(self.params.rate_lambdas):.2f}]"
        )

    def _get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for parameter optimization."""
        n_states = self.params.n_states
        bounds = []

        # Transition matrix bounds (each element in [0, 1])
        for _ in range(n_states * n_states):
            bounds.append((1e-6, 1.0))

        # Initial probability bounds
        for _ in range(n_states):
            bounds.append((1e-6, 1.0))

        # Success probability bounds
        for _ in range(n_states):
            bounds.append((0.01, 0.99))

        # Rate lambda bounds
        for _ in range(n_states):
            bounds.append((0.1, 1000.0))

        self.logger.debug(
            f"Parameter bounds: {len(bounds)} total, "
            f"transition={n_states * n_states}, "
            f"initial={n_states}, "
            f"success={n_states}, "
            f"lambdas={n_states}"
        )

        return bounds
