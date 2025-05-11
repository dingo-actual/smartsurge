"""
Client implementation for the SmartSurge library.

This module provides the main SmartSurgeClient class that handles HTTP requests
with configurable rate limiting strategies:
1. User-defined rate limits (primary)
2. Server-provided rate limits from 429 responses (fallback)
3. Adaptive rate limit estimation (last resort)

The client automatically selects the most appropriate rate limiting strategy
based on available information and configuration.
"""
from datetime import datetime, timezone
import os
import logging
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import asyncio
import aiohttp
from pydantic import BaseModel, Field, ValidationInfo, ValidationError, field_validator
from typing import TypeVar, Optional, Dict, Any, Union, Sequence, Tuple, Type, TYPE_CHECKING

from .models import RequestMethod, RequestEntry, RequestHistory
from .streaming import AbstractStreamingRequest
from .utilities import SmartSurgeTimer

if TYPE_CHECKING:
    from . import exceptions
    from .exceptions import RateLimitExceeded, StreamingError, ResumeError

# Module-level logger
logger = logging.getLogger(__name__)

# Type for SmartSurgeClient methods
T = TypeVar('T')


class ClientConfig(BaseModel):
    """
    Configuration for the SmartSurge client.
    
    This class centralizes all configuration options for the client, making it
    easier to manage and extend the client's behavior.
    
    Attributes:
        base_url: Base URL for all requests
        timeout: Default timeout for requests in seconds
        max_retries: Maximum number of retries for failed requests
        backoff_factor: Backoff factor for retries
        verify_ssl: Whether to verify SSL certificates
        min_time_period: Minimum time period to consider for rate limiting (seconds)
        max_time_period: Maximum time period to consider for rate limiting (seconds)
        confidence_threshold: Confidence threshold for rate limit estimation
        user_agent: User agent string for requests
        max_connections: Maximum number of connections to keep alive
        keep_alive: Whether to keep connections alive
        max_pool_size: Maximum size of the connection pool
        log_level: Log level for the client
    """
    base_url: Optional[str] = None
    timeout: Tuple[float, float] = Field(default=(10.0, 30.0))  # (connect, read)
    max_retries: int = Field(default=3, ge=0, le=10)
    backoff_factor: float = Field(default=0.3, ge=0.0, le=10.0)
    verify_ssl: bool = Field(default=True, strict=True)
    min_time_period: float = Field(default=1.0, gt=0.0)
    max_time_period: float = Field(default=3600.0, gt=0.0)
    confidence_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    user_agent: str = Field(default="SmartSurge/1.0")
    max_connections: int = Field(default=10, ge=1)
    keep_alive: bool = True
    max_pool_size: int = Field(default=10, ge=1)
    log_level: int = Field(default=logging.INFO)
    
    @field_validator('timeout', mode="before")
    def validate_timeout(cls, v):
        """Validate timeout is a tuple of two positive floats or a single positive float."""
        if isinstance(v, (int, float)):
            if v <= 0:
                raise ValueError("Timeout must be positive")
            return (v, v)
        elif isinstance(v, Sequence):
            if len(v) != 2:
                raise ValueError("Timeout must be a tuple of (connect_timeout, read_timeout)")
            if v[0] <= 0 or v[1] <= 0:
                raise ValueError("Both connect and read timeouts must be positive")
        return v
    
    @field_validator('min_time_period', 'max_time_period')
    def validate_time_periods(cls, v: float, info: ValidationInfo):
        """Validate min_time_period is less than max_time_period."""
        # When validating max_time_period
        if info.field_name == 'max_time_period' and 'min_time_period' in info.data:
            if v < info.data['min_time_period']:
                raise ValueError("max_time_period must be greater than min_time_period")
        # When validating min_time_period
        elif info.field_name == 'min_time_period' and 'max_time_period' in info.data:
            if v > info.data['max_time_period']:
                raise ValueError("min_time_period must be less than max_time_period")
        return v

class SmartSurgeClient:
    """
    A wrapper around requests library with adaptive rate limiting and resumable streaming.
    
    This client handles rate limits in the following priority:
    1. Uses caller-defined rate limits if provided
    2. Uses rate limits from HTTP 429 responses if available
    3. Falls back to adaptive rate limit estimation if neither are available/correct
    """
    def __init__(self, 
                 base_url: Optional[str] = None,
                 timeout: Union[float, Tuple[float, float]] = (10.0, 30.0),
                 rate_limit: Optional[Dict[str, Union[int, float]]] = None,
                 max_retries: int = 3,
                 backoff_factor: float = 0.3,
                 verify_ssl: bool = True,
                 min_time_period: float = 1.0,
                 max_time_period: float = 3600.0,
                 confidence_threshold: float = 0.9,
                 logger: Optional[logging.Logger] = None,
                 **kwargs):
        """
        Initialize a new SmartSurgeClient.
        
        Args:
            base_url: Base URL for all requests.
            timeout: Default timeout for requests in seconds, either a single value or (connect, read).
            max_retries: Maximum number of retries for failed requests.
            rate_limit: Optional dict specifying rate limits {'requests': int, 'period': float}.
            backoff_factor: Backoff factor for retries.
            verify_ssl: Whether to verify SSL certificates.
            min_time_period: Minimum time period to consider for rate limiting (seconds).
            max_time_period: Maximum time period for rate limiting (seconds).
            confidence_threshold: Confidence threshold for rate limit estimation (0.0-1.0).
            logger: Optional custom logger to use.
            **kwargs: Additional configuration options for ClientConfig.
        """
        # Create configuration from parameters and kwargs
        config_data = {
            'base_url': base_url,
            'timeout': timeout,
            'max_retries': max_retries,
            'backoff_factor': backoff_factor,
            'verify_ssl': verify_ssl,
            'min_time_period': min_time_period,
            'max_time_period': max_time_period,
            'confidence_threshold': confidence_threshold,
            **kwargs
        }
        self.config = ClientConfig(**config_data)
        
        # Rate limit hierarchy:
        # 1. Caller-defined rate limits
        # 2. Rate limits from 429 responses
        # 3. Adaptive rate limit estimation
        self.user_rate_limit = rate_limit
        self.response_rate_limit = None
        
        # Set up logger
        self.logger = logger or logging.getLogger("smartsurge.client")
        self.logger.setLevel(self.config.log_level)
        
        # Dictionary of RequestHistory objects, keyed by (endpoint, method)
        self.histories: Dict[Tuple[str, RequestMethod], RequestHistory] = {}
        
        # Create a session with retry capabilities
        self.session = self._create_session()
        
        self.logger.debug(f"Initialized SmartSurge client with config: {self.config}")
        
    def _create_session(self) -> requests.Session:
        """
        Create a new requests Session with appropriate retry capabilities.
        
        Returns:
            A configured requests Session
        """
        session = requests.Session()
        
        # Set default headers
        session.headers.update({
            'User-Agent': self.config.user_agent,
        })
        
        # Create retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=[500, 502, 503, 504],  # Exclude 429 to handle it ourselves
            respect_retry_after_header=True
        )
        
        # Mount adapters with retry strategy
        adapter = HTTPAdapter(
            max_retries=retry_strategy, 
            pool_connections=self.config.max_connections,
            pool_maxsize=self.config.max_pool_size
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
        
    def _get_full_url(self, endpoint: str) -> str:
        """
        Get the full URL for an endpoint.
        
        Args:
            endpoint: The endpoint path or URL
            
        Returns:
            The full URL
        """
        if self.config.base_url and not endpoint.startswith(('http://', 'https://')):
            return f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        return endpoint
        
    def _get_or_create_history(self, endpoint: str, method: Union[str, RequestMethod], 
                               confidence_threshold: float) -> RequestHistory:
        """
        Get or create a RequestHistory for an endpoint/method combination.
        
        Args:
            endpoint: The endpoint.
            method: The HTTP method.
            confidence_threshold: Confidence threshold for rate limit estimation.
            
        Returns:
            A RequestHistory object.
        """
        # Convert method to RequestMethod enum if it's a string
        if isinstance(method, str):
            try:
                method = RequestMethod(method.upper())
            except ValueError:
                raise ValueError(f"Invalid HTTP method: {method}")
            
        key = (endpoint, method)
        if key not in self.histories:
            # Use the client's confidence_threshold if None is provided
            conf_threshold = confidence_threshold if confidence_threshold is not None else self.config.confidence_threshold
            
            history_logger = self.logger.getChild(f"history.{endpoint}.{method}")
            self.histories[key] = RequestHistory(
                endpoint=endpoint,
                method=method,
                min_time_period=self.config.min_time_period,
                max_time_period=self.config.max_time_period,
                confidence_threshold=conf_threshold,
                logger=history_logger
            )
            
        return self.histories[key]
        
    def request(self, 
                method: Union[str, RequestMethod], 
                endpoint: str,
                params: Optional[Dict[str, Any]] = None,
                data: Optional[Union[Dict[str, Any], str, bytes]] = None,
                json: Optional[Dict[str, Any]] = None,
                headers: Optional[Dict[str, str]] = None,
                cookies: Optional[Dict[str, str]] = None,
                files: Optional[Dict[str, Any]] = None,
                auth: Optional[Any] = None,
                timeout: Optional[Union[float, Tuple[float, float]]] = None,
                allow_redirects: bool = True,
                verify: Optional[bool] = None,
                stream: bool = False,
                cert: Optional[Union[str, Tuple[str, str]]] = None,
                num_async: int = 1,
                confidence_threshold: Optional[float] = None,
                request_history: Optional[RequestHistory] = None) -> Tuple[requests.Response, RequestHistory]:
        """
        Make an HTTP request with adaptive rate limiting.
        
        Args:
            method: HTTP method to use.
            endpoint: Endpoint to request.
            params: Query parameters.
            data: Form data.
            json: JSON data.
            headers: HTTP headers.
            cookies: Cookies to send.
            files: Files to upload.
            auth: Authentication.
            timeout: Request timeout.
            allow_redirects: Whether to follow redirects.
            verify: Whether to verify SSL certificates.
            stream: Whether to stream the response.
            cert: SSL client certificate.
            num_async: Number of asynchronous requests.
            confidence_threshold: Confidence threshold for rate limit estimation (0.0-1.0).
            request_history: Explicit RequestHistory to use.
            
        Returns:
            Tuple containing:
            - The HTTP response
            - The RequestHistory used for this request
            
        Raises:
            RateLimitExceeded: If the rate limit has been exceeded.
            requests.RequestException: For other request failures.
        """
        # Convert method to RequestMethod enum if it's a string
        if isinstance(method, str):
            try:
                method = RequestMethod(method.upper())
            except ValueError:
                raise ValueError(f"Invalid HTTP method: {method}")
            
        # Get full URL
        full_url = self._get_full_url(endpoint)
        
        # Get or create RequestHistory
        if request_history:
            history = request_history
        else:
            conf_threshold = confidence_threshold if confidence_threshold is not None else self.config.confidence_threshold
            history = self._get_or_create_history(endpoint, method, conf_threshold)
        
        # Generate a request ID for logging correlation
        request_id = history.request_id
        
        # Intercept request for rate limiting
        history.intercept_request()
        
        # Prepare the timeout
        req_timeout = timeout or self.config.timeout
        
        # Make the request
        start_time = time.time()
        try:
            self.logger.debug(f"[{request_id}] Making {method.value} request to {endpoint}")
            with SmartSurgeTimer(f"request.{request_id}", self.logger):
                response = self.session.request(
                    method=method.value,
                    url=full_url,
                    params=params,
                    data=data,
                    json=json,
                    headers=headers,
                    cookies=cookies,
                    files=files,
                    auth=auth,
                    timeout=req_timeout,
                    allow_redirects=allow_redirects,
                    verify=verify if verify is not None else self.config.verify_ssl,
                    stream=stream,
                    cert=cert
                )
            
            # Record the request
            end_time = time.time()
            request_entry = RequestEntry(
                endpoint=endpoint,
                method=method,
                timestamp=datetime.now(timezone.utc),
                status_code=response.status_code,
                response_time=end_time - start_time,
                success=response.ok
            )
            
            # Log the response and update search status
            history.log_response_and_update(request_entry)
            
            # If we got a 429, raise exception after updating the history
            if response.status_code == 429:
                from .exceptions import RateLimitExceeded
                
                retry_after = None
                if 'Retry-After' in response.headers:
                    try:
                        retry_after = int(response.headers['Retry-After'])
                        self.logger.info(f"[{request_id}] Rate limit exceeded, server specified Retry-After: {retry_after} seconds")
                        time.sleep(retry_after)
                        # Recursive call after waiting
                        return self.request(
                            method=method,
                            endpoint=endpoint,
                            params=params,
                            data=data,
                            json=json,
                            headers=headers,
                            cookies=cookies,
                            files=files,
                            auth=auth,
                            timeout=timeout,
                            allow_redirects=allow_redirects,
                            verify=verify,
                            stream=stream,
                            cert=cert,
                            num_async=num_async,
                            confidence_threshold=confidence_threshold,
                            request_history=history
                        )
                    except ValueError:
                        # Not an integer, might be HTTP date format
                        self.logger.warning(f"[{request_id}] Could not parse Retry-After header: {response.headers['Retry-After']}")
                
                raise RateLimitExceeded(
                    message=f"Rate limit exceeded for {endpoint} {method}",
                    endpoint=endpoint,
                    method=method,
                    retry_after=retry_after
                )
            
            return response, history
        except requests.RequestException as e:
            self.logger.error(f"[{request_id}] Request failed: {e}")
            
            # Record the failed request
            end_time = time.time()
            request_entry = RequestEntry(
                endpoint=endpoint,
                method=method,
                timestamp=datetime.now(timezone.utc),
                status_code=0,  # No status code for failed request
                response_time=end_time - start_time,
                success=False
            )
            
            # Log the response and update search status
            history.log_response_and_update(request_entry)
            
            # Re-raise the exception
            raise
    
    def stream_request(self, 
                       streaming_class: Type[AbstractStreamingRequest],
                       endpoint: str,
                       params: Optional[Dict[str, Any]] = None,
                       headers: Optional[Dict[str, str]] = None,
                       state_file: Optional[str] = None,
                       chunk_size: int = 8192,
                       num_async: int = 1,
                       confidence_threshold: Optional[float] = None) -> Tuple[Any, RequestHistory]:
        """
        Make a streaming request with resumable functionality.
        
        Args:
            streaming_class: Class to handle streaming.
            endpoint: Endpoint to request.
            params: Query parameters.
            headers: HTTP headers.
            state_file: File to save state for resumption.
            chunk_size: Size of chunks to process.
            num_async: Number of asynchronous requests.
            confidence_threshold: Confidence threshold for rate limit estimation.
            
        Returns:
            Tuple containing:
            - The result of the streaming request
            - The RequestHistory used for this request
            
        Raises:
            StreamingError: If the streaming request fails.
        """
        # Get full URL
        full_url = self._get_full_url(endpoint)
        
        # Get or create RequestHistory - Assuming GET for streaming
        conf_threshold = confidence_threshold if confidence_threshold is not None else self.config.confidence_threshold
        history = self._get_or_create_history(endpoint, RequestMethod.GET, conf_threshold)
        
        # Generate a request ID for logging correlation
        request_id = history.request_id
        
        # Intercept request for rate limiting
        history.intercept_request()
        
        # Initialize streaming request
        headers = headers or {}
        # Add SmartSurge client headers
        headers.update({'User-Agent': self.config.user_agent})
        
        streaming_request = streaming_class(
            endpoint=full_url,
            headers=headers,
            params=params,
            chunk_size=chunk_size,
            state_file=state_file,
            logger=self.logger.getChild(f"streaming.{endpoint}"),
            request_id=request_id
        )
        
        start_time = time.time()
        try:
            self.logger.debug(f"[{request_id}] Starting streaming request to {endpoint}")
            
            with SmartSurgeTimer(f"streaming_request.{request_id}", self.logger):
                # Check if we can resume
                if state_file and os.path.exists(state_file):
                    self.logger.info(f"[{request_id}] Resuming streaming request from state file: {state_file}")
                    streaming_request.resume()
                else:
                    streaming_request.start()
                
            end_time = time.time()
            
            # Record the successful request
            request_entry = RequestEntry(
                endpoint=endpoint,
                method=RequestMethod.GET,
                timestamp=datetime.now(timezone.utc),
                status_code=200,  # Assuming success
                response_time=end_time - start_time,
                success=True
            )
            
            # Log the response and update search status
            history.log_response_and_update(request_entry)
            
            result = streaming_request.get_result()
            return result, history
        except Exception as e:
            from .exceptions import StreamingError, ResumeError
            
            self.logger.error(f"[{request_id}] Streaming request failed: {e}")
            
            # Save state for later resumption
            try:
                streaming_request.save_state()
            except Exception as save_error:
                self.logger.error(f"[{request_id}] Failed to save state: {save_error}")
            
            # Check if it's a rate limit error
            rate_limited = False
            status_code = 500
            
            if isinstance(e, StreamingError) and hasattr(e, 'response'):
                if hasattr(e.response, 'status_code'):
                    status_code = e.response.status_code
                    rate_limited = status_code == 429
            
            # Record the failed request
            end_time = time.time()
            request_entry = RequestEntry(
                endpoint=endpoint,
                method=RequestMethod.GET,
                timestamp=datetime.now(timezone.utc),
                status_code=status_code,
                response_time=end_time - start_time,
                success=False
            )
            
            # Log the response and update search status
            history.log_response_and_update(request_entry)
            
            # If rate limited, wait and retry if appropriate
            if rate_limited and hasattr(e, 'response') and hasattr(e.response, 'headers'):
                retry_after = e.response.headers.get('Retry-After')
                if retry_after:
                    try:
                        wait_time = int(retry_after)
                        self.logger.info(f"[{request_id}] Rate limit exceeded, server specified Retry-After: {wait_time} seconds")
                        time.sleep(wait_time)
                        # Recursive call after waiting
                        return self.stream_request(
                            streaming_class=streaming_class,
                            endpoint=endpoint,
                            params=params,
                            headers=headers,
                            state_file=state_file,
                            chunk_size=chunk_size,
                            num_async=num_async,
                            confidence_threshold=confidence_threshold
                        )
                    except ValueError:
                        # Not an integer, might be HTTP date format
                        self.logger.warning(f"[{request_id}] Could not parse Retry-After header: {retry_after}")
            
            # Re-raise as StreamingError if it's not already
            if isinstance(e, ResumeError):
                # Already the correct type, just re-raise
                raise
            elif state_file and os.path.exists(state_file):
                raise ResumeError(f"Failed to resume streaming request: {e}", state_file=state_file)
            elif not isinstance(e, StreamingError):
                # Convert other exceptions to StreamingError
                raise StreamingError(f"Streaming request failed: {e}", endpoint=endpoint)
            else:
                # Already a StreamingError, just re-raise
                raise
            
    def close(self) -> None:
        """
        Close the client and release resources.
        
        This method should be called when the client is no longer needed.
        """
        self.session.close()
        self.logger.debug("Closed SmartSurge client")
    
    def __enter__(self) -> 'SmartSurgeClient':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    # Convenience methods
    def get(self, 
            endpoint: str,
            params: Optional[Dict[str, Any]] = None,
            headers: Optional[Dict[str, str]] = None,
            num_async: int = 1,
            confidence_threshold: Optional[float] = None,
            **kwargs) -> Tuple[requests.Response, RequestHistory]:
        """
        Make a GET request.
        
        Args:
            endpoint: Endpoint to request.
            params: Query parameters.
            headers: HTTP headers.
            num_async: Number of asynchronous requests.
            confidence_threshold: Confidence threshold for rate limit estimation.
            **kwargs: Additional parameters to pass to the request method.
            
        Returns:
            Tuple containing:
            - The HTTP response
            - The RequestHistory used for this request
        """
        return self.request(
            method=RequestMethod.GET,
            endpoint=endpoint,
            params=params,
            headers=headers,
            num_async=num_async,
            confidence_threshold=confidence_threshold,
            **kwargs
        )
    
    def post(self, 
             endpoint: str,
             data: Optional[Union[Dict[str, Any], str, bytes]] = None,
             json: Optional[Dict[str, Any]] = None,
             headers: Optional[Dict[str, str]] = None,
             num_async: int = 1,
             confidence_threshold: Optional[float] = None,
             **kwargs) -> Tuple[requests.Response, RequestHistory]:
        """
        Make a POST request.
        
        Args:
            endpoint: Endpoint to request.
            data: Form data or body.
            json: JSON data.
            headers: HTTP headers.
            num_async: Number of asynchronous requests.
            confidence_threshold: Confidence threshold for rate limit estimation.
            **kwargs: Additional parameters to pass to the request method.
            
        Returns:
            Tuple containing:
            - The HTTP response
            - The RequestHistory used for this request
        """
        return self.request(
            method=RequestMethod.POST,
            endpoint=endpoint,
            data=data,
            json=json,
            headers=headers,
            num_async=num_async,
            confidence_threshold=confidence_threshold,
            **kwargs
        )
    
    def put(self, 
            endpoint: str,
            data: Optional[Union[Dict[str, Any], str, bytes]] = None,
            json: Optional[Dict[str, Any]] = None,
            headers: Optional[Dict[str, str]] = None,
            num_async: int = 1,
            confidence_threshold: Optional[float] = None,
            **kwargs) -> Tuple[requests.Response, RequestHistory]:
        """
        Make a PUT request.
        
        Args:
            endpoint: Endpoint to request.
            data: Form data or body.
            json: JSON data.
            headers: HTTP headers.
            num_async: Number of asynchronous requests.
            confidence_threshold: Confidence threshold for rate limit estimation.
            **kwargs: Additional parameters to pass to the request method.
            
        Returns:
            Tuple containing:
            - The HTTP response
            - The RequestHistory used for this request
        """
        return self.request(
            method=RequestMethod.PUT,
            endpoint=endpoint,
            data=data,
            json=json,
            headers=headers,
            num_async=num_async,
            confidence_threshold=confidence_threshold,
            **kwargs
        )
        
    def delete(self, 
               endpoint: str,
               params: Optional[Dict[str, Any]] = None,
               headers: Optional[Dict[str, str]] = None,
               num_async: int = 1,
               confidence_threshold: Optional[float] = None,
               **kwargs) -> Tuple[requests.Response, RequestHistory]:
        """
        Make a DELETE request.
        
        Args:
            endpoint: Endpoint to request.
            params: Query parameters.
            headers: HTTP headers.
            num_async: Number of asynchronous requests.
            confidence_threshold: Confidence threshold for rate limit estimation.
            **kwargs: Additional parameters to pass to the request method.
            
        Returns:
            Tuple containing:
            - The HTTP response
            - The RequestHistory used for this request
        """
        return self.request(
            method=RequestMethod.DELETE,
            endpoint=endpoint,
            params=params,
            headers=headers,
            num_async=num_async,
            confidence_threshold=confidence_threshold,
            **kwargs
        )

    def patch(self, 
              endpoint: str,
              data: Optional[Union[Dict[str, Any], str, bytes]] = None,
              json: Optional[Dict[str, Any]] = None,
              headers: Optional[Dict[str, str]] = None,
              num_async: int = 1,
              confidence_threshold: Optional[float] = None,
              **kwargs) -> Tuple[requests.Response, RequestHistory]:
        """
        Make a PATCH request.
        
        Args:
            endpoint: Endpoint to request.
            data: Form data or body.
            json: JSON data.
            headers: HTTP headers.
            num_async: Number of asynchronous requests.
            confidence_threshold: Confidence threshold for rate limit estimation.
            **kwargs: Additional parameters to pass to the request method.
            
        Returns:
            Tuple containing:
            - The HTTP response
            - The RequestHistory used for this request
        """
        return self.request(
            method=RequestMethod.PATCH,
            endpoint=endpoint,
            data=data,
            json=json,
            headers=headers,
            num_async=num_async,
            confidence_threshold=confidence_threshold,
            **kwargs
        )
        
    async def async_get(self, 
                       endpoint: str,
                       params: Optional[Dict[str, Any]] = None,
                       headers: Optional[Dict[str, str]] = None,
                       num_async: int = 1,
                       confidence_threshold: Optional[float] = None,
                       **kwargs) -> Tuple[aiohttp.ClientResponse, RequestHistory]:
        """
        Make an asynchronous GET request.
        
        Args:
            endpoint: Endpoint to request.
            params: Query parameters.
            headers: HTTP headers.
            num_async: Number of asynchronous requests.
            confidence_threshold: Confidence threshold for rate limit estimation.
            **kwargs: Additional parameters to pass to the async_request method.
            
        Returns:
            Tuple containing:
            - The HTTP response
            - The RequestHistory used for this request
        """
        return await self.async_request(
            method=RequestMethod.GET,
            endpoint=endpoint,
            params=params,
            headers=headers,
            num_async=num_async,
            confidence_threshold=confidence_threshold,
            **kwargs
        )
        
    async def async_post(self, 
                        endpoint: str,
                        data: Optional[Dict[str, Any]] = None,
                        json: Optional[Dict[str, Any]] = None,
                        headers: Optional[Dict[str, str]] = None,
                        num_async: int = 1,
                        confidence_threshold: Optional[float] = None,
                        **kwargs) -> Tuple[aiohttp.ClientResponse, RequestHistory]:
        """
        Make an asynchronous POST request.
        
        Args:
            endpoint: Endpoint to request.
            data: Form data or body.
            json: JSON data.
            headers: HTTP headers.
            num_async: Number of asynchronous requests.
            confidence_threshold: Confidence threshold for rate limit estimation.
            **kwargs: Additional parameters to pass to the async_request method.
            
        Returns:
            Tuple containing:
            - The HTTP response
            - The RequestHistory used for this request
        """
        return await self.async_request(
            method=RequestMethod.POST,
            endpoint=endpoint,
            data=data,
            json=json,
            headers=headers,
            num_async=num_async,
            confidence_threshold=confidence_threshold,
            **kwargs
        )
        
    async def async_request(self,
                           method: Union[str, RequestMethod],
                           endpoint: str,
                           params: Optional[Dict[str, Any]] = None,
                           data: Optional[Dict[str, Any]] = None,
                           json: Optional[Dict[str, Any]] = None,
                           headers: Optional[Dict[str, str]] = None,
                           cookies: Optional[Dict[str, str]] = None,
                           auth: Optional[aiohttp.BasicAuth] = None,
                           timeout: Optional[float] = None,
                           allow_redirects: bool = True,
                           verify_ssl: Optional[bool] = None,
                           num_async: int = 1,
                           confidence_threshold: Optional[float] = None,
                           request_history: Optional[RequestHistory] = None) -> Tuple[aiohttp.ClientResponse, RequestHistory]:
        """
        Make an asynchronous HTTP request with adaptive rate limiting.
        
        Args:
            method: HTTP method to use.
            endpoint: Endpoint to request.
            params: Query parameters.
            data: Form data.
            json: JSON data.
            headers: HTTP headers.
            cookies: Cookies to send.
            auth: Authentication.
            timeout: Request timeout.
            allow_redirects: Whether to follow redirects.
            verify_ssl: Whether to verify SSL certificates.
            num_async: Number of asynchronous requests.
            confidence_threshold: Confidence threshold for rate limit estimation.
            request_history: Explicit RequestHistory to use.
            
        Returns:
            Tuple containing:
            - The HTTP response
            - The RequestHistory used for this request
            
        Raises:
            RateLimitExceeded: If the rate limit has been exceeded.
            aiohttp.ClientError: For other request failures.
        """
        # Convert method to RequestMethod enum if it's a string
        if isinstance(method, str):
            try:
                method = RequestMethod(method.upper())
            except ValueError:
                raise ValueError(f"Invalid HTTP method: {method}")
            
        # Get full URL
        full_url = self._get_full_url(endpoint)
        
        # Get or create RequestHistory
        if request_history:
            history = request_history
        else:
            conf_threshold = confidence_threshold if confidence_threshold is not None else self.config.confidence_threshold
            history = self._get_or_create_history(endpoint, method, conf_threshold)
        
        # Generate a request ID for logging correlation
        request_id = history.request_id
        
        # Intercept request for rate limiting
        history.intercept_request()
        
        # Prepare the timeout
        req_timeout = timeout or self.config.timeout[1]  # Use read timeout
        verify = verify_ssl if verify_ssl is not None else self.config.verify_ssl
        
        # Add SmartSurge client headers
        headers = headers or {}
        headers.update({'User-Agent': self.config.user_agent})
        
        # Make the request
        start_time = time.time()
        try:
            self.logger.debug(f"[{request_id}] Making async {method.value} request to {endpoint}")
            
            async with aiohttp.ClientSession(headers=headers, cookies=cookies) as session:
                async with session.request(
                    method=method.value,
                    url=full_url,
                    params=params,
                    data=data,
                    json=json,
                    auth=auth,
                    timeout=req_timeout,
                    allow_redirects=allow_redirects,
                    ssl=verify
                ) as response:
                    # Record the request
                    end_time = time.time()
                    request_entry = RequestEntry(
                        endpoint=endpoint,
                        method=method,
                        timestamp=datetime.now(timezone.utc),
                        status_code=response.status,
                        response_time=end_time - start_time,
                        success=response.ok
                    )
                    
                    # Log the response and update search status
                    history.log_response_and_update(request_entry)
                    
                    # If we got a 429, raise exception after updating the history
                    if response.status == 429:
                        from .exceptions import RateLimitExceeded
                        
                        retry_after = None
                        if 'Retry-After' in response.headers:
                            try:
                                retry_after = int(response.headers['Retry-After'])
                                self.logger.info(f"[{request_id}] Rate limit exceeded, server specified Retry-After: {retry_after} seconds")
                                await asyncio.sleep(retry_after)
                                # Recursive call after waiting
                                return await self.async_request(
                                    method=method,
                                    endpoint=endpoint,
                                    params=params,
                                    data=data,
                                    json=json,
                                    headers=headers,
                                    cookies=cookies,
                                    auth=auth,
                                    timeout=timeout,
                                    allow_redirects=allow_redirects,
                                    verify_ssl=verify_ssl,
                                    num_async=num_async,
                                    confidence_threshold=confidence_threshold,
                                    request_history=history
                                )
                            except ValueError:
                                # Not an integer, might be HTTP date format
                                self.logger.warning(f"[{request_id}] Could not parse Retry-After header: {response.headers['Retry-After']}")
                        
                        raise RateLimitExceeded(
                            message=f"Rate limit exceeded for {endpoint} {method}",
                            endpoint=endpoint,
                            method=method,
                            retry_after=retry_after
                        )
                    
                    # When using aiohttp we need to read the body before returning
                    # to ensure the connection can be closed properly
                    await response.read()
                    return response, history
                    
        except aiohttp.ClientError as e:
            self.logger.error(f"[{request_id}] Async request failed: {e}")
            
            # Record the failed request
            end_time = time.time()
            request_entry = RequestEntry(
                endpoint=endpoint,
                method=method,
                timestamp=datetime.now(timezone.utc),
                status_code=0,  # No status code for failed request
                response_time=end_time - start_time,
                success=False
            )
            
            # Log the response and update search status
            history.log_response_and_update(request_entry)
            
            # Re-raise the exception
            raise
