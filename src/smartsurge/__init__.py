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

from . import client
from . import exceptions
from . import logging_
from . import models
from . import streaming
from . import utilities
from .client import SmartSurgeClient, ClientConfig
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

# Set up package version
try:
    __version__ = version("smartsurge")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

# Set up a default package-level logger
logger = logging.getLogger("smartsurge")

# Define what's available via the public API
__all__ = [
    "SmartSurgeClient",
    "ClientConfig",
    "EnhancedRequestsException",
    "RateLimitExceeded",
    "StreamingError",
    "ResumeError",
    "ValidationError",
    "ConfigurationError",
    "RequestMethod",
    "SearchStatus",
    "RequestEntry",
    "RateLimit",
    "RequestHistory",
    "StreamingState",
    "AbstractStreamingRequest",
    "JSONStreamingRequest",
    "SmartSurgeTimer",
    "log_context",
    "merge_histories",
    "async_request_with_history",
    "configure_logging",
    "__version__",
]