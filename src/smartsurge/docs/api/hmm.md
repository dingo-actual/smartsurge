# HMM API Reference

The HMM module implements a Hidden Markov Model for intelligent rate limit detection and estimation. It statistically analyzes request patterns to identify rate limiting without requiring API-specific configuration.

## Overview

SmartSurge's HMM approach models API rate limiting as a stochastic process with:
- **Hidden states**: Traffic load levels (normal, approaching limit, rate limited)
- **Emissions**: Request outcomes (success/failure) and observed rate limits
- **Statistical learning**: MLE and EM algorithms for parameter estimation

## HMMParams

```python
from smartsurge.hmm import HMMParams
```

Configuration parameters for the Hidden Markov Model.

### Constructor

```python
HMMParams(
    n_states: int = 3,
    initial_probs: Optional[np.ndarray] = None,
    transition_matrix: Optional[np.ndarray] = None,
    success_probs: Optional[np.ndarray] = None,
    rate_lambdas: Optional[np.ndarray] = None
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_states` | `int` | `3` | Number of hidden states (2-10) |
| `initial_probs` | `Optional[np.ndarray]` | `None` | Initial state probabilities (auto-initialized) |
| `transition_matrix` | `Optional[np.ndarray]` | `None` | State transition probability matrix (n_states × n_states) |
| `success_probs` | `Optional[np.ndarray]` | `None` | Bernoulli parameters for request success per state |
| `rate_lambdas` | `Optional[np.ndarray]` | `None` | Poisson parameters for rate limits per state |

#### Attributes

All constructor parameters are available as instance attributes.

#### Validation

- Arrays must match `n_states` dimension
- 1D arrays must have length `n_states`
- 2D arrays must have shape `(n_states, n_states)`

#### Example

```python
# Default 3-state model
params = HMMParams()

# Custom 4-state model
params = HMMParams(
    n_states=4,
    initial_probs=np.array([0.7, 0.2, 0.08, 0.02]),
    success_probs=np.array([0.99, 0.85, 0.40, 0.10])
)
```

## HMM

```python
from smartsurge.hmm import HMM
```

Hidden Markov Model implementation for rate limit estimation.

### Constructor

```python
HMM(
    params: Optional[HMMParams] = None,
    logger: Optional[logging.Logger] = None,
    n_bootstrap: int = 10,
    **kwargs
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | `Optional[HMMParams]` | `None` | HMM parameters (auto-initialized if None) |
| `logger` | `Optional[logging.Logger]` | `None` | Custom logger instance |
| `**kwargs` | `Any` | - | Additional keyword arguments |

#### Attributes

- `params`: HMM parameters
- `logger`: Logger instance
- `parameter_history`: History of parameter estimates
- `n_bootstrap`: Number of bootstrap samples

### Core Methods

#### emission_probability()

Calculate the emission probability for an observation given a state.

```python
def emission_probability(
    outcome: bool,
    rate_limit: int,
    state: int
) -> float
```

##### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `outcome` | `bool` | Request success (True) or failure (False) |
| `rate_limit` | `int` | Observed rate limit (requests per interval) |
| `state` | `int` | Hidden state index (0 to n_states-1) |

##### Returns

`float`: Emission probability P(observation | state)

##### Model

- Request outcomes follow Bernoulli distribution with state-specific success probability
- Rate limits follow shifted Poisson distribution: rate_limit ~ 1 + Poisson(λ)

### Inference Methods

#### forward_backward()

Forward-backward algorithm for HMM inference.

```python
def forward_backward(
    observations: List[Tuple[bool, int]]
) -> Tuple[np.ndarray, np.ndarray, float]
```

##### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `observations` | `List[Tuple[bool, int]]` | List of (outcome, rate_limit) tuples |

##### Returns

`Tuple[np.ndarray, np.ndarray, float]`:
- `alpha`: Forward probabilities (T × n_states)
- `beta`: Backward probabilities (T × n_states)
- `log_likelihood`: Log-likelihood of observations

#### viterbi()

Find the most likely state sequence using the Viterbi algorithm.

```python
def viterbi(
    observations: List[Tuple[bool, int]]
) -> List[int]
```

##### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `observations` | `List[Tuple[bool, int]]` | List of (outcome, rate_limit) tuples |

##### Returns

`List[int]`: Most likely sequence of hidden states

### Learning Methods

#### fit_mle()

Fit HMM parameters using Maximum Likelihood Estimation with multiple random restarts.

```python
def fit_mle(
    observations: List[Tuple[bool, int]],
    max_iter: int = 1000,
    n_restarts: int = 5
) -> float
```

##### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `observations` | `List[Tuple[bool, int]]` | - | Training observations |
| `max_iter` | `int` | `1000` | Maximum iterations per optimization |
| `n_restarts` | `int` | `5` | Random restarts to avoid local optima |

##### Returns

`float`: Final log-likelihood

##### Algorithm

Uses L-BFGS-B optimization with log-transformed parameters for unconstrained optimization.

#### baum_welch()

Baum-Welch algorithm (EM for HMMs) for parameter learning with multiple random restarts.

```python
def baum_welch(
    observations: List[Tuple[bool, int]],
    max_iter: int = 100,
    tol: float = 1e-4,
    n_restarts: int = 5
) -> float
```

##### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `observations` | `List[Tuple[bool, int]]` | - | Training observations |
| `max_iter` | `int` | `100` | Maximum EM iterations per restart |
| `tol` | `float` | `1e-4` | Convergence tolerance |
| `n_restarts` | `int` | `5` | Random restarts to avoid local optima |

##### Returns

`float`: Best log-likelihood achieved across all restarts

##### Algorithm

Uses multiple random restarts to avoid local optima. Each restart:
1. Initializes with random parameters (except first restart)
2. Runs EM algorithm until convergence or max_iter
3. Keeps parameters with highest log-likelihood


### Rate Limit Prediction

#### predict_rate_limit()

Predict the rate limit based on observations and model parameters.

```python
def predict_rate_limit(
    observations: List[Tuple[bool, int]]
) -> Tuple[int, float]
```

##### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `observations` | `List[Tuple[bool, int]]` | Historical observations |

##### Returns

`Tuple[int, float]`:
- `max_requests`: Maximum requests allowed
- `time_period`: Time period in seconds (always 1.0)

### Helper Methods

#### flatten_params()

Flatten HMM parameters for optimization (log-transformed).

```python
def flatten_params() -> np.ndarray
```

#### unflatten_params()

Reconstruct HMM parameters from flattened array.

```python
def unflatten_params(
    params_flat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

#### negative_log_likelihood()

Compute negative log-likelihood for optimization.

```python
def negative_log_likelihood(
    params_flat: np.ndarray,
    observations: List[Tuple[bool, int]]
) -> float
```

## State Interpretation

The default 3-state model represents:

1. **State 0**: Normal operation
   - High success probability
   - High rate limits
   
2. **State 1**: Approaching limit
   - Medium success probability
   - Moderate rate limits
   
3. **State 2**: Rate limited
   - Low success probability
   - Restrictive rate limits

## Complete Example

```python
from smartsurge.hmm import HMM, HMMParams
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create HMM with custom parameters
params = HMMParams(
    n_states=3,
    initial_probs=np.array([0.8, 0.15, 0.05])
)

hmm = HMM(
    params=params,
)

# Collect observations (success/failure, observed_rate_limit)
observations = [
    (True, 10),   # Success, high rate
    (True, 10),   # Success, high rate
    (True, 8),    # Success, medium rate
    (False, 5),   # Failure, lower rate
    (False, 3),   # Failure, low rate
    (True, 5),    # Success, medium rate
]

# Train the model using MLE
log_likelihood = hmm.fit_mle(observations, n_restarts=10)
print(f"MLE log-likelihood: {log_likelihood:.4f}")

# Or train using Baum-Welch with random restarts
log_likelihood = hmm.baum_welch(observations, n_restarts=10)
print(f"Baum-Welch log-likelihood: {log_likelihood:.4f}")

# Find most likely state sequence
states = hmm.viterbi(observations)
print(f"Most likely states: {states}")

# Predict rate limit
max_requests, time_period = hmm.predict_rate_limit(observations)
print(f"Rate limit: {max_requests} requests per {time_period}s")
```

## Advanced Usage

### Custom State Models

```python
# 5-state model for finer granularity
params = HMMParams(
    n_states=5,
    success_probs=np.array([0.99, 0.90, 0.70, 0.40, 0.10])
)

hmm = HMM(params=params)
```

### Integration with RequestHistory

```python
from smartsurge.models import RequestHistory

# RequestHistory uses HMM internally
history = RequestHistory(
    endpoint="/api/users",
    method="GET",
    n_bootstrap=100  # Higher bootstrap for production
)

# HMM parameters are accessible
if history.hmm:
    print(f"HMM states: {history.hmm.params.n_states}")
    print(f"Success probs: {history.hmm.params.success_probs}")
```

## Performance Considerations

1. **MLE restarts**: More restarts (5-10) help avoid local optima
2. **Observation count**: Need at least 10-20 observations for reliable estimates

## Mathematical Details

### Emission Model

- **Request outcome**: P(success|state) = Bernoulli(success_probs[state])
- **Rate limit**: P(rate|state) = ShiftedPoisson(rate_lambdas[state])
  - Actual distribution: rate_limit ~ 1 + Poisson(λ)

### Parameter Optimization

- Uses log-transformation for unconstrained optimization
- L-BFGS-B algorithm for MLE
- Multiple random restarts to find global optimum

### Parameter Initialization

The HMM parameters are initialized using mathematically principled random distributions that encode sensible priors for rate limit detection:

#### Initial State Probabilities

Initial probabilities are sampled from a Dirichlet distribution with pseudo-counts that favor the first state:

```python
pseudo_counts = np.ones(n_states)
pseudo_counts[0] = 5  # Higher weight for state 0
initial_probs = np.sort(np.random.dirichlet(pseudo_counts))[::-1]
```

This ensures:
- State 0 (normal operation) has the highest initial probability
- States are ordered by decreasing probability
- The distribution is properly normalized (sums to 1)

#### Transition Matrix

The transition matrix is constructed to encourage state persistence with smooth transitions:

```python
# Sample rows from Dirichlet distribution
transition_probs = [np.sort(np.random.dirichlet(pseudo_counts))[::-1] 
                   for _ in range(n_states)]

# Circular shift to create diagonal dominance
transition_probs = [np.concatenate([vec[-ix:], vec[:-ix]]) 
                   for ix, vec in enumerate(transition_probs)]
```

This creates a matrix where:
- Diagonal elements have higher probabilities (states tend to persist)
- State 0 is most stable (highest self-transition probability)
- Transitions to adjacent states are more likely than distant jumps

#### Success Probabilities

Success probabilities follow a Beta distribution with parameters that decrease across states:

```python
success_probs = [np.random.beta(a, 1.0) 
                for a in np.linspace(5.0, 0.25, n_states)]
success_probs = np.sort(success_probs)[::-1]
```

The Beta(a, 1) distribution properties:
- When a > 1: Distribution skewed toward 1 (high success rate)
- When a < 1: Distribution skewed toward 0 (low success rate)
- State 0 gets Beta(5, 1) → ~0.83 expected success rate
- Final state gets Beta(0.25, 1) → ~0.20 expected success rate

#### Rate Lambdas

Rate parameters for the shifted Poisson distribution are sampled from an exponential distribution:

```python
rate_lambdas = np.sort(np.random.exponential(10.0, size=n_states))[::-1]
```

This ensures:
- Exponential(10) gives a reasonable spread of rate limits
- State 0 has the highest rate limit (least restrictive)
- States are ordered by decreasing rate limits
- The long tail of the exponential allows for discovering high rate limits

#### Mathematical Rationale

This initialization strategy encodes several key assumptions:

1. **Ordered States**: States represent increasing levels of rate limiting severity
2. **State Persistence**: APIs typically maintain consistent behavior over short periods
3. **Smooth Degradation**: Success rates and rate limits decrease monotonically across states
4. **Exploration-Exploitation**: Random initialization allows discovering diverse API behaviors while maintaining reasonable starting points

The use of conjugate priors (Dirichlet for multinomial, Beta for Bernoulli) ensures that:
- Parameters stay in valid ranges
- The model can adapt quickly when observations deviate from priors
- Numerical stability during optimization

