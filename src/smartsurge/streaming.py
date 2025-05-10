"""
Streaming functionality for the SmartSurge library.

This module provides classes for handling streaming requests, with support
for resumable downloads and efficient processing of large responses.
"""

from base64 import b64decode
from typing import Any, Dict, Optional, Tuple, TypeVar, Union, Type
from datetime import datetime, timezone
from pydantic import BaseModel, Field
import logging
import os
import json
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import uuid
from abc import ABC, abstractmethod
from pydantic import field_serializer, field_validator, ValidationError

from .exceptions import StreamingError, ResumeError
from .utilities import SmartSurgeTimer

# Module-level logger
logger = logging.getLogger(__name__)

T = TypeVar('T')

class StreamingState(BaseModel):
    """
    State of a streaming request for resumption.
    
    Attributes:
        endpoint: The endpoint being requested
        method: The HTTP method being used
        headers: HTTP headers for the request
        params: Optional query parameters
        data: Optional request body data
        accumulated_data: Data accumulated so far
        last_position: Last position in the stream
        total_size: Total size of the stream if known
        etag: ETag of the resource if available
        last_updated: Timestamp of when this state was last updated
        request_id: Unique identifier for the request (for tracking)
    """
    endpoint: str = Field(..., min_length=1)
    method: str
    headers: Dict[str, str]
    params: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None
    accumulated_data: bytes
    last_position: int = Field(..., ge=0)
    total_size: Optional[int] = None
    etag: Optional[str] = None
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    @field_validator("accumulated_data", mode="before")
    def decode_accumulated_data(cls, v: Union[str, bytes]):
        """
        Decodes a string from base64 if it appears to be base64 encoded,
        otherwise returns the original input.
        
        Args:
            input_data: The input that might be base64 encoded. Can be a string or bytes.
            
        Returns:
            The decoded data if input was base64, otherwise the original input.
        """
        # Convert to string if bytes
        if isinstance(v, bytes):
            return v
        elif isinstance(v, str):
            # Check if the string matches base64 pattern
            # Base64 strings consist of letters, digits, '+', '/', and may end with '='
            if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', v):
                logger.warning(f"String does not match base64 pattern: {v}")
                try:
                    v = v.encode("utf-8")
                except UnicodeEncodeError:
                    logger.warning(f"Error encoding string as UTF-8: {v}")
                finally:
                    return v
            
            # The length of base64 encoded string should be a multiple of 4
            # (or it should be padded to be)
            if len(v) % 4 != 0:
                logger.warning(f"Base64 string length is not a multiple of 4: {v}")
                try:
                    v = v.encode("utf-8")
                except UnicodeEncodeError:
                    logger.error(f"Error encoding string as UTF-8: {v}")
                    v = b""
                finally:
                    return v
                
            try:
                # Try to decode
                decoded_data = b64decode(v)
                logger.debug(f"Decoded base64 string: {decoded_data}")
                
                # Try to convert to UTF-8 string if possible
                try:
                    decoded_data = decoded_data.decode("utf-8")
                    logger.debug(f"Decoded string: {decoded_data}")
                    return decoded_data.encode("utf-8")
                except UnicodeDecodeError:
                    # If it can't be decoded as UTF-8, return the bytes
                    logger.warning(f"Failed to decode as UTF-8, returning bytes: {decoded_data}")
                    return v
            except Exception:
                # If any other error occurs, return the original input
                logger.error(f"Unexpected error decoding {v}", exc_info=True)
                return b""
        else:
            return v
        
    @field_serializer("accumulated_data")
    def serialize_accumulated_data(self, v: bytes):
        """
        Serialize accumulated_data to a base64 string for JSON.
        """
        import base64
        if isinstance(v, bytes):
            return base64.b64encode(v).decode('ascii')
        return v

class AbstractStreamingRequest(ABC):
    """
    Abstract base class for resumable streaming requests.
    
    This class provides the common functionality for handling streaming requests
    that can be paused and resumed. It includes state management, data accumulation,
    and basic error handling.
    """
    def __init__(self, 
                 endpoint: str, 
                 headers: Dict[str, str],
                 chunk_size: int = 8192,
                 state_file: Optional[str] = None,
                 logger: Optional[logging.Logger] = None,
                 request_id: Optional[str] = None):
        """
        Initialize a streaming request.
        
        Args:
            endpoint: The endpoint to request
            headers: HTTP headers for the request
            chunk_size: Size of chunks to process
            state_file: File to save state for resumption
            logger: Optional custom logger to use
            request_id: Optional request ID for tracking and correlation
        """
        self.endpoint = endpoint
        self.headers = headers
        self.chunk_size = chunk_size
        self.state_file = state_file
        self.accumulated_data = bytearray()
        self.position = 0
        self.total_size = None
        self.etag = None
        self.completed = False
        self.request_id = request_id or str(uuid.uuid4())[:8]
        # Use provided logger or get a class-specific logger
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    def start(self) -> None:
        """
        Start the streaming request.
        
        This method should initiate the streaming request and begin
        processing chunks of data.
        
        Raises:
            StreamingError: If the request fails
        """
        pass
        
    @abstractmethod
    def resume(self) -> None:
        """
        Resume the streaming request from saved state.
        
        This method should resume a streaming request that was previously
        paused, using the saved state to continue from where it left off.
        
        Raises:
            ResumeError: If resuming the request fails
        """
        pass
        
    @abstractmethod
    def process_chunk(self, chunk: bytes) -> None:
        """
        Process a chunk of data.
        
        Args:
            chunk: A chunk of data to process
        """
        pass
        
    @abstractmethod
    def get_result(self) -> Any:
        """
        Get the final result after all chunks have been processed.
        
        Returns:
            The processed result
            
        Raises:
            StreamingError: If the streaming request is not complete
        """
        pass
        
    def save_state(self) -> None:
        """
        Save the current state for resumption.
        
        This method saves the current state of the streaming request to a file,
        allowing it to be resumed later.
        """
        if not self.state_file:
            self.logger.warning(f"[{self.request_id}] No state file specified, skipping state save")
            return
            
        state = StreamingState(
            endpoint=self.endpoint,
            method="GET",  # Assuming GET for streaming
            headers=self.headers,
            accumulated_data=bytes(self.accumulated_data),
            last_position=self.position,
            total_size=self.total_size,
            etag=self.etag,
            request_id=self.request_id
        )
        
        try:
            # Ensure directory exists
            state_dir = os.path.dirname(self.state_file)
            if state_dir and not os.path.exists(state_dir):
                os.makedirs(state_dir, exist_ok=True)
                
            with open(self.state_file, 'w') as f:
                f.write(state.model_dump_json())
            self.logger.debug(f"[{self.request_id}] Saved state to {self.state_file}")
        except Exception as e:
            self.logger.error(f"[{self.request_id}] Failed to save state: {e}")
            
    def load_state(self) -> Optional[StreamingState]:
        """
        Load saved state for resumption.
        
        Returns:
            The loaded state, or None if loading failed
        """
        if not self.state_file:
            self.logger.warning(f"[{self.request_id}] No state file specified, skipping state load")
            return None
            
        try:
            if not os.path.exists(self.state_file):
                self.logger.warning(f"[{self.request_id}] State file does not exist: {self.state_file}")
                return None
            
            with open(self.state_file, 'r') as f:
                state_data = f.read()
            state = StreamingState.model_validate_json(state_data)
            
            # Update instance variables
            self.endpoint = state.endpoint
            self.headers = state.headers
            self.accumulated_data = bytearray(state.accumulated_data)
            self.position = state.last_position
            self.total_size = state.total_size
            self.etag = state.etag
            if state.request_id:
                self.request_id = state.request_id
            
            self.logger.debug(f"[{self.request_id}] Loaded state from {self.state_file}")
            return state
        except Exception as e:
            self.logger.error(f"[{self.request_id}] Failed to load state: {e}")
            return None

class JSONStreamingRequest(AbstractStreamingRequest):
    """
    A streaming request implementation that accumulates JSON data.
    
    This class provides functionality for making HTTP requests that
    stream JSON data, with support for resuming downloads and
    handling large responses efficiently.
    """
    def __init__(self, 
                 endpoint: str, 
                 headers: Dict[str, str],
                 params: Optional[Dict[str, Any]] = None,
                 chunk_size: int = 8192,
                 state_file: Optional[str] = None,
                 logger: Optional[logging.Logger] = None,
                 request_id: Optional[str] = None):
        """
        Initialize a JSON streaming request.
        
        Args:
            endpoint: The endpoint to request
            headers: HTTP headers for the request
            params: Query parameters
            chunk_size: Size of chunks to process
            state_file: File to save state for resumption
            logger: Optional custom logger to use
            request_id: Optional request ID for tracking and correlation
        """
        super().__init__(endpoint, headers, chunk_size, state_file, logger, request_id)
        self.params = params or {}
        self.response = None
        
    def start(self) -> None:
        """
        Start the streaming request.
        
        Makes the HTTP request and processes the response in chunks.
        
        Raises:
            StreamingError: If the request fails
        """
        # Add Range header if resuming
        if self.position > 0:
            self.headers['Range'] = f'bytes={self.position}-'
            
        # Make the request
        self.logger.debug(f"[{self.request_id}] Starting streaming request to {self.endpoint}")
        try:
            # Use a session with appropriate retry settings
            session = requests.Session()
            retry_strategy = Retry(
                total=3,  # 3 retries
                backoff_factor=0.5,  # 0.5 * (2^(retry) - 1) seconds between retries
                status_forcelist=[500, 502, 503, 504],  # Retry on these status codes
                allowed_methods=["GET"]  # Only retry on GET
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            # Make the request with the configured session
            with session.get(
                self.endpoint,
                headers=self.headers,
                params=self.params,
                stream=True,
                timeout=(10, 30)  # (connect timeout, read timeout)
            ) as self.response:
                # Check response
                if not self.response.ok:
                    self.logger.error(f"[{self.request_id}] Streaming request failed with status {self.response.status_code}")
                    raise StreamingError(f"Request failed with status {self.response.status_code}: {self.response.text[:200]}")
                    
                # Get total size if available
                if 'Content-Length' in self.response.headers:
                    try:
                        self.total_size = int(self.response.headers['Content-Length'])
                        self.logger.debug(f"[{self.request_id}] Content length: {self.total_size} bytes")
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"[{self.request_id}] Invalid Content-Length header: {self.response.headers['Content-Length']}, {e}")
                    
                # Get ETag if available
                if 'ETag' in self.response.headers:
                    self.etag = self.response.headers['ETag']
                    self.logger.debug(f"[{self.request_id}] ETag: {self.etag}")
                    
                # Process chunks
                for chunk in self.response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        self.process_chunk(chunk)
                        
                self.completed = True
                self.logger.info(f"[{self.request_id}] Completed streaming request to {self.endpoint}, accumulated {len(self.accumulated_data)} bytes")
        
        except requests.RequestException as e:
            self.logger.error(f"[{self.request_id}] Streaming request failed: {e}")
            # Save state in case of error for later resumption
            self.save_state()
            raise StreamingError(f"Streaming request failed: {e}")
        
    def resume(self) -> None:
        """
        Resume the streaming request from saved state.
        
        Loads the saved state and continues the request from
        where it left off.
        
        Raises:
            ResumeError: If resuming the request fails
        """
        try:
            state = self.load_state()
        except Exception as e:
            self.logger.error(f"[{self.request_id}] Exception when loading state: {e}")
            raise ResumeError(f"Failed to load state for resumption: {e}")
        
        if not state:
            self.logger.error(f"[{self.request_id}] Failed to load state for resumption")
            raise ResumeError("Failed to load state for resumption")
            
        # Add Range header for resuming
        if self.position > 0:
            self.headers['Range'] = f'bytes={self.position}-'
            
        # Add ETag header if available
        if self.etag:
            self.headers['If-Match'] = self.etag
            
        self.logger.info(f"[{self.request_id}] Resuming streaming request to {self.endpoint} from position {self.position}")
        
        # Resume the request
        try:
            self.start()
        except Exception as e:
            self.logger.error(f"[{self.request_id}] Failed to resume streaming request: {e}")
            raise ResumeError(f"Failed to resume streaming request: {e}")
        
    def process_chunk(self, chunk: bytes) -> None:
        """
        Process a chunk of data.
        
        Extends the accumulated data with the new chunk and updates
        the position.
        
        Args:
            chunk: A chunk of data to process
        """
        
        self.accumulated_data.extend(chunk)
        chunk_size = len(chunk)
        self.position += chunk_size
        
        # Log progress for large responses
        if self.total_size and self.total_size > self.chunk_size * 10:
            progress = (self.position / self.total_size) * 100
            if self.position % (self.chunk_size * 10) < chunk_size:  # Log every ~10 chunks
                self.logger.debug(f"[{self.request_id}] Download progress: {progress:.1f}% ({self.position}/{self.total_size} bytes)")
        
        # Save state periodically
        if self.state_file and self.position % (self.chunk_size * 10) < chunk_size:
            try:
                self.save_state()
            except Exception as e:
                self.logger.error(f"[{self.request_id}] Failed to save state while processing chunk: {e}")
            
    def get_result(self) -> Any:
        """
        Get the final result after all chunks have been processed.
        
        Returns:
            The parsed JSON data
            
        Raises:
            StreamingError: If the streaming request is not completed or JSON parsing fails
        """
        if not self.completed:
            self.logger.error(f"[{self.request_id}] Streaming request not completed")
            raise StreamingError("Streaming request not completed")
        try:
            with SmartSurgeTimer(f"json_parse.{self.request_id}", self.logger):
                return json.loads(self.accumulated_data.decode('utf-8').strip())
        except json.JSONDecodeError as e:
            self.logger.error(f"[{self.request_id}] Failed to parse JSON: {e}")
            # Log a small sample of the data for debugging
            sample = self.accumulated_data[:200].decode('utf-8', errors='replace')
            self.logger.debug(f"[{self.request_id}] JSON sample: {sample}...")
            raise StreamingError(f"Failed to parse JSON: {e}") from e
