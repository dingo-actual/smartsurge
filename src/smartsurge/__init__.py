"""
SmartSurge: Enhanced Requests Library with Adaptive Rate Limit Estimation

This library extends the functionality of the requests library with:
- Automatic rate limit detection and enforcement
- Adaptive rate limit estimation
- Resumable streaming requests
- Robust error handling
"""

import logging
from importlib.metadata import version, PackageNotFoundError

# Always import core modules
from . import client
from . import exceptions
from . import logging_
from . import models
from . import streaming
from . import utilities

# Core imports that are always available
from .client import SmartSurgeClient, ClientConfig
Client = SmartSurgeClient

from .exceptions import (
    RateLimitExceeded,
    StreamingError,
    ResumeError,
    ValidationError,
    ConfigurationError,
)
from .streaming import (
    StreamingState,
    AbstractStreamingRequest,
    JSONStreamingRequest,
)
from .utilities import (
    SmartSurgeTimer,
    log_context,
    merge_histories,
    async_request_with_history,
)
from .logging_ import configure_logging

# Conditionally import benchmarks
_BENCHMARK_AVAILABLE = False
try:
    from . import benchmarks
    from .benchmarks import (
        BenchmarkMockServer,
        RateLimitConfig,
        RateLimitStrategy,
        create_benchmark_server,
        get_adaptive_config,
        get_dynamic_config,
        get_noisy_config,
        get_strict_rate_limit_config,
        get_token_bucket_config,
        get_load_dependent_config,
    )
    _BENCHMARK_AVAILABLE = True
except ImportError:
    # Benchmarks not installed
    pass

# Set up package version
try:
    __version__ = version("smartsurge")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

# Set up a default package-level logger
logger = logging.getLogger("smartsurge")

# Define what's available via the public API
__all__ = [
    # Core client
    "SmartSurgeClient",
    "Client",
    "ClientConfig",
    # Exceptions
    "RateLimitExceeded",
    "StreamingError",
    "ResumeError",
    "ValidationError",
    "ConfigurationError",
    # Streaming
    "StreamingState",
    "AbstractStreamingRequest",
    "JSONStreamingRequest",
    # Utilities
    "SmartSurgeTimer",
    "log_context",
    "merge_histories",
    "async_request_with_history",
    "configure_logging",
    # Version
    "__version__",
]

# Add benchmark exports only if available
if _BENCHMARK_AVAILABLE:
    __all__.extend([
        "benchmarks",
        "BenchmarkMockServer",
        "RateLimitConfig",
        "RateLimitStrategy",
        "create_benchmark_server",
        "get_adaptive_config",
        "get_dynamic_config",
        "get_noisy_config",
        "get_strict_rate_limit_config",
        "get_token_bucket_config",
        "get_load_dependent_config",
    ])


def has_benchmarks() -> bool:
    """Check if benchmark functionality is available.
    
    Returns:
        True if the package was installed with benchmark extras, False otherwise.
        
    Example:
        >>> import smartsurge
        >>> if smartsurge.has_benchmarks():
        ...     from smartsurge import create_benchmark_server
        ...     server = create_benchmark_server()
        ... else:
        ...     print("Install with: pip install smartsurge[benchmark]")
    """
    return _BENCHMARK_AVAILABLE


# Add the helper function to exports
__all__.append("has_benchmarks")