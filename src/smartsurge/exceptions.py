"""
Exceptions used throughout the SmartSurge library.

This module defines a hierarchy of exceptions specific to SmartSurge,
providing detailed information about various error conditions.
"""

from typing import Optional, Union
import logging

from .models import RequestMethod

# Module-level logger
logger = logging.getLogger(__name__)

class EnhancedRequestsException(Exception):
    """Base exception for the SmartSurge library."""
    pass

class RateLimitExceeded(EnhancedRequestsException):
    """
    Exception raised when a rate limit is exceeded.
    
    Attributes:
        endpoint: The endpoint that was rate limited
        method: The HTTP method that was rate limited
        retry_after: Optional retry-after time in seconds
        message: The exception message
    """
    def __init__(self, message: str, endpoint: Optional[str] = None, 
                 method: Optional[Union[str, RequestMethod]] = None, 
                 retry_after: Optional[int] = None):
        self.endpoint = endpoint
        self.method = method
        self.retry_after = retry_after
        super().__init__(message)
        logger.warning(f"Rate limit exceeded for {endpoint} {method} - retry after {retry_after}s")

class StreamingError(EnhancedRequestsException):
    """
    Exception raised when a streaming request fails.
    
    Attributes:
        endpoint: The endpoint that was requested
        position: The position in the stream where the error occurred
        message: The exception message
    """
    def __init__(self, message: str, endpoint: Optional[str] = None, position: Optional[int] = None):
        self.endpoint = endpoint
        self.position = position
        super().__init__(message)
        logger.error(f"Streaming error for {endpoint} at position {position}: {message}")

class ResumeError(EnhancedRequestsException):
    """
    Exception raised when resuming a request fails.
    
    Attributes:
        state_file: The state file that failed to load
        message: The exception message
    """
    def __init__(self, message: str, state_file: Optional[str] = None):
        self.state_file = state_file
        super().__init__(message)
        logger.error(f"Resume error with state file {state_file}: {message}")

class ValidationError(EnhancedRequestsException):
    """
    Exception raised when validation fails.
    
    Attributes:
        field: The field that failed validation
        message: The exception message
    """
    def __init__(self, message: str, field: Optional[str] = None):
        self.field = field
        super().__init__(message)
        logger.error(f"Validation error for field {field}: {message}")

class ConfigurationError(EnhancedRequestsException):
    """
    Exception raised when configuration is invalid.
    
    Attributes:
        parameter: The parameter that is invalid
        message: The exception message
    """
    def __init__(self, message: str, parameter: Optional[str] = None):
        self.parameter = parameter
        super().__init__(message)
        logger.error(f"Configuration error for parameter {parameter}: {message}")
