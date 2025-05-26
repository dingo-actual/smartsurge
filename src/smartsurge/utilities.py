"""
Utility functions and classes for the SmartSurge library.

This module provides various utility functions and classes that are used
throughout the SmartSurge library, including timing, logging, and async utilities.
"""
from typing import List, Dict, Any, Optional, Callable, Coroutine, Generator, Tuple, TypeVar
import logging
import time
import uuid
import asyncio
from datetime import datetime, timezone
from contextlib import contextmanager

from .models import RequestHistory, RequestMethod, RateLimit


# Module-level logger
logger = logging.getLogger(__name__)

T = TypeVar('T')

class SmartSurgeTimer:
    """
    Context manager for timing code execution and logging the results.
    
    Example:
        with SmartSurgeTimer("operation_name", logger):
            # Do some operation
            result = expensive_operation()
    """
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            self.logger.debug(f"Operation '{self.operation_name}' completed in {elapsed:.4f} seconds")
        
        # Don't suppress any exceptions
        return False

@contextmanager
def log_context(context_name: str, logger: Optional[logging.Logger] = None, 
                level: int = logging.DEBUG, include_id: bool = True) -> Optional[Generator[Any, Any, Any]]:
    """
    Context manager for logging the start and end of an operation.
    
    Example:
        with log_context("api_request", logger):
            # Do some API request
            response = make_api_request()
    
    Args:
        context_name: Name of the context for logging
        logger: Logger to use
        level: Logging level
        include_id: Whether to include a correlation ID
    """
    log = logger if isinstance(logger, logging.Logger) else logging.getLogger(__name__)
    context_id = str(uuid.uuid4())[:6] if include_id else ""
    id_suffix = f" [{context_id}]" if include_id else ""
    
    try:
        log.log(level, f"Starting {context_name}{id_suffix}")
        start_time = time.time()
        yield
    except Exception as e:
        elapsed = time.time() - start_time
        log.log(level, f"Error in {context_name}{id_suffix} after {elapsed:.4f}s: {e}")
        raise
    else:
        elapsed = time.time() - start_time
        log.log(level, f"Completed {context_name}{id_suffix} in {elapsed:.4f}s")

async def async_request_with_history(
    request_func: Callable[..., Coroutine[Any, Any, Tuple[T, RequestHistory]]],
    endpoints: List[str],
    method: RequestMethod,
    max_concurrent: int = 5,
    min_time_period: float = 1.0,
    max_time_period: float = 3600.0,
    **kwargs
) -> Tuple[List[T], Dict[str, RequestHistory]]:
    """
    Make multiple async requests and consolidate histories for the same endpoint.
    
    Args:
        request_func: Async function to make the request.
        endpoints: List of endpoints to request.
        method: HTTP method to use.
        max_concurrent: Maximum number of concurrent requests.
        min_time_period: Minimum time period for rate limiting.
        max_time_period: Maximum time period for rate limiting.
        **kwargs: Additional arguments to pass to request_func.
        
    Returns:
        Tuple containing:
        - List of responses 
        - Dictionary of consolidated RequestHistory objects by endpoint
    """
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create histories keyed by endpoint
    histories = {
        endpoint: RequestHistory(
            endpoint=endpoint,
            method=method,
            min_time_period=min_time_period,
            max_time_period=max_time_period
        ) for endpoint in endpoints
    }
    
    async def request_with_semaphore(endpoint: str) -> Tuple[T, RequestHistory]:
        """Make a request with semaphore and history."""
        async with semaphore:
            logger.debug(f"Making async request to {endpoint}")
            response, history = await request_func(
                method=method,
                endpoint=endpoint,
                request_history=histories[endpoint],
                num_async=len(endpoints),
                **kwargs
            )
            return response, history
    
    # Create tasks
    tasks = [request_with_semaphore(endpoint) for endpoint in endpoints]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results and consolidate histories
    responses = []
    consolidated_histories = {}
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Request to {endpoints[i]} failed: {result}")
            responses.append(result)
        else:
            response, history = result
            responses.append(response)
            
            # Consolidate histories for the same endpoint
            endpoint = endpoints[i]
            if endpoint in consolidated_histories:
                consolidated_histories[endpoint].merge(history)
            else:
                consolidated_histories[endpoint] = history
    
    return responses, consolidated_histories

def merge_histories(histories: List[RequestHistory], rate_limit: Optional[float] = None) -> Dict[Tuple[str, RequestMethod], RequestHistory]:
    """
    Merge multiple RequestHistory objects by endpoint/method.
    
    This function:
    1. Groups RequestHistory objects by endpoint/method
    2. Applies rate limiting logic based on provided rate limits or 429 responses
    3. Merges the histories together, preserving all request data
    
    Args:
        histories: List of RequestHistory objects.
        rate_limit: Optional caller-defined rate limit in requests per second.
        
    Returns:
        Dictionary of merged RequestHistory objects keyed by (endpoint, method).
    """
    merged = {}
    log = logging.getLogger(f"{__name__}.merge_histories")
    
    for history in histories:
        key = (history.endpoint, history.method)
        
        # Apply rate limit logic
        if rate_limit is not None:
            # Set caller-defined rate limit
            history.rate_limit = RateLimit(
                endpoint=history.endpoint,
                method=history.method,
                max_requests=int(rate_limit),  # Assuming rate_limit is requests per second
                time_period=1.0,  # Assuming time_period is 1 second
                last_updated=datetime.now(timezone.utc)
            )
        else:
            # Check for 429 responses and set a conservative rate limit if needed
            has_429 = any(entry.status_code == 429 for entry in history.entries)
            if has_429:
                history.rate_limit = RateLimit(
                    endpoint=history.endpoint,
                    method=history.method,
                    max_requests=1,  # Very conservative rate limit
                    time_period=5.0,  # 5 seconds between requests
                    last_updated=datetime.now(timezone.utc)
                )
                log.warning(f"Found 429 responses for {history.endpoint}, setting conservative rate limit")
        
        # Merge the history into the appropriate bucket
        if key in merged:
            log.debug(f"Merging history for {key[0]} {key[1]}")
            merged[key].merge(history)
        else:
            merged[key] = history
            
    return merged
