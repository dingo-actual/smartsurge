"""
Exceptions used throughout the SmartSurge library.

This module defines a hierarchy of exceptions specific to SmartSurge,
providing detailed information about various error conditions.
"""

from typing import Optional, Union, Dict, Any, TYPE_CHECKING
import logging
import traceback

# Only import for type checking to avoid circular imports
if TYPE_CHECKING:
    from .models import RequestMethod


# Module-level logger
logger = logging.getLogger(__name__)

class SmartSurgeException(Exception):
    """Base exception for the SmartSurge library."""
    
    def __init__(self, message: str, **kwargs):
        """
        Initialize the exception.
        
        Args:
            message: The exception message
            **kwargs: Additional context information to include in the exception
        """
        self.message = message
        self.context = kwargs
        super().__init__(message)
        
        # Log the exception with context
        log_message = f"{message}"
        if kwargs:
            log_message += f" - Context: {kwargs}"
        logger.error(log_message)

class RateLimitExceeded(SmartSurgeException):
    """
    Exception raised when a rate limit is exceeded.
    
    Attributes:
        endpoint: The endpoint that was rate limited
        method: The HTTP method that was rate limited
        retry_after: Optional retry-after time in seconds
        message: The exception message
    """
    def __init__(self, message: str, endpoint: Optional[str] = None, 
                 method: Optional[Union[str, 'RequestMethod']] = None, 
                 retry_after: Optional[int] = None,
                 response_headers: Optional[Dict[str, str]] = None):
        """
        Initialize the exception.
        
        Args:
            message: The exception message
            endpoint: The endpoint that was rate limited
            method: The HTTP method that was rate limited
            retry_after: Optional retry-after time in seconds
            response_headers: Optional response headers that might contain rate limit information
        """
        self.endpoint = endpoint
        self.method = method
        self.retry_after = retry_after
        self.response_headers = response_headers
        super().__init__(
            message, 
            endpoint=endpoint,
            method=str(method) if method else None,
            retry_after=retry_after,
            response_headers=response_headers
        )

class StreamingError(SmartSurgeException):
    """
    Exception raised when a streaming request fails.
    
    Attributes:
        endpoint: The endpoint that was requested
        position: The position in the stream where the error occurred
        message: The exception message
        response: The response object if available
    """
    def __init__(self, message: str, endpoint: Optional[str] = None, 
                 position: Optional[int] = None, response: Optional[Any] = None):
        """
        Initialize the exception.
        
        Args:
            message: The exception message
            endpoint: The endpoint that was requested
            position: The position in the stream where the error occurred
            response: The response object if available
        """
        self.endpoint = endpoint
        self.position = position
        self.response = response
        super().__init__(
            message, 
            endpoint=endpoint,
            position=position,
            response_status=getattr(response, 'status_code', None) if response else None
        )

class ResumeError(SmartSurgeException):
    """
    Exception raised when resuming a request fails.
    
    Attributes:
        state_file: The state file that failed to load
        message: The exception message
    """
    def __init__(self, message: str, state_file: Optional[str] = None,
                original_error: Optional[Exception] = None):
        """
        Initialize the exception.
        
        Args:
            message: The exception message
            state_file: The state file that failed to load
            original_error: The original error that caused the resume failure
        """
        self.state_file = state_file
        self.original_error = original_error
        super().__init__(
            message, 
            state_file=state_file,
            original_error=str(original_error) if original_error else None,
            traceback=traceback.format_exc() if original_error else None
        )

class ValidationError(SmartSurgeException):
    """
    Exception raised when validation fails.
    
    Attributes:
        field: The field that failed validation
        message: The exception message
        value: The value that failed validation
    """
    def __init__(self, message: str, field: Optional[str] = None, 
                value: Optional[Any] = None):
        """
        Initialize the exception.
        
        Args:
            message: The exception message
            field: The field that failed validation
            value: The value that failed validation
        """
        self.field = field
        self.value = value
        super().__init__(
            message, 
            field=field,
            value=str(value) if value is not None else None
        )

class ConfigurationError(SmartSurgeException):
    """
    Exception raised when configuration is invalid.
    
    Attributes:
        parameter: The parameter that is invalid
        message: The exception message
        value: The value that is invalid
    """
    def __init__(self, message: str, parameter: Optional[str] = None,
                value: Optional[Any] = None):
        """
        Initialize the exception.
        
        Args:
            message: The exception message
            parameter: The parameter that is invalid
            value: The value that is invalid
        """
        self.parameter = parameter
        self.value = value
        super().__init__(
            message, 
            parameter=parameter,
            value=str(value) if value is not None else None
        )

class ConnectionError(SmartSurgeException):
    """
    Exception raised when a connection error occurs.
    
    Attributes:
        endpoint: The endpoint that was being connected to
        message: The exception message
        original_error: The original error that caused the connection failure
    """
    def __init__(self, message: str, endpoint: Optional[str] = None,
                original_error: Optional[Exception] = None):
        """
        Initialize the exception.
        
        Args:
            message: The exception message
            endpoint: The endpoint that was being connected to
            original_error: The original error that caused the connection failure
        """
        self.endpoint = endpoint
        self.original_error = original_error
        super().__init__(
            message, 
            endpoint=endpoint,
            original_error=str(original_error) if original_error else None,
            traceback=traceback.format_exc() if original_error else None
        )

class TimeoutError(SmartSurgeException):
    """
    Exception raised when a request times out.
    
    Attributes:
        endpoint: The endpoint that timed out
        timeout: The timeout value that was exceeded
        message: The exception message
    """
    def __init__(self, message: str, endpoint: Optional[str] = None,
                timeout: Optional[Union[float, int]] = None):
        """
        Initialize the exception.
        
        Args:
            message: The exception message
            endpoint: The endpoint that timed out
            timeout: The timeout value that was exceeded
        """
        self.endpoint = endpoint
        self.timeout = timeout
        super().__init__(
            message, 
            endpoint=endpoint,
            timeout=timeout
        )

class ServerError(SmartSurgeException):
    """
    Exception raised when a server error occurs.
    
    Attributes:
        endpoint: The endpoint that returned an error
        status_code: The HTTP status code returned by the server
        message: The exception message
        response: The response object if available
    """
    def __init__(self, message: str, endpoint: Optional[str] = None,
                status_code: Optional[int] = None, response: Optional[Any] = None):
        """
        Initialize the exception.
        
        Args:
            message: The exception message
            endpoint: The endpoint that returned an error
            status_code: The HTTP status code returned by the server
            response: The response object if available
        """
        self.endpoint = endpoint
        self.status_code = status_code
        self.response = response
        super().__init__(
            message, 
            endpoint=endpoint,
            status_code=status_code,
            response_text=getattr(response, 'text', '')[:200] if response else None
        )

class ClientError(SmartSurgeException):
    """
    Exception raised when a client error occurs.
    
    Attributes:
        endpoint: The endpoint that returned an error
        status_code: The HTTP status code returned by the server
        message: The exception message
        response: The response object if available
    """
    def __init__(self, message: str, endpoint: Optional[str] = None,
                status_code: Optional[int] = None, response: Optional[Any] = None):
        """
        Initialize the exception.
        
        Args:
            message: The exception message
            endpoint: The endpoint that returned an error
            status_code: The HTTP status code returned by the server
            response: The response object if available
        """
        self.endpoint = endpoint
        self.status_code = status_code
        self.response = response
        super().__init__(
            message, 
            endpoint=endpoint,
            status_code=status_code,
            response_text=getattr(response, 'text', '')[:200] if response else None
        )
