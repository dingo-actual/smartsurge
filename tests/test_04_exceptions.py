import pytest
import logging
from unittest.mock import patch, MagicMock

from smartsurge.exceptions import SmartSurgeException


class Test_SmartSurgeException_01_NominalBehaviors:
    """Tests for nominal behaviors of the SmartSurgeException class."""
    
    def test_initializes_with_message(self):
        """Test that exception initializes with a message that becomes accessible."""
        message = "Test exception message"
        exc = SmartSurgeException(message)
        assert exc.message == message
        assert str(exc) == message
    
    def test_accepts_and_stores_context(self):
        """Test that exception accepts and stores additional context via kwargs."""
        message = "Test with context"
        context = {"key1": "value1", "key2": "value2"}
        exc = SmartSurgeException(message, **context)
        assert exc.context == context
        for key, value in context.items():
            assert exc.context[key] == value
    
    def test_logs_message_and_context(self):
        """Test that message and context information are properly logged at error level."""
        with patch('logging.Logger.error') as mock_error:
            message = "Test log message"
            context = {"key": "value"}
            SmartSurgeException(message, **context)
            mock_error.assert_called_once()
            log_msg = mock_error.call_args[0][0]
            assert message in log_msg
            assert str(context) in log_msg
    
    def test_string_representation_includes_message(self):
        """Test that exception's string representation includes the provided message."""
        message = "Test string representation"
        exc = SmartSurgeException(message)
        assert str(exc) == message
    
    def test_parent_exception_initialized(self):
        """Test that parent Exception class is properly initialized with the message."""
        message = "Test parent initialization"
        exc = SmartSurgeException(message)
        assert isinstance(exc, Exception)
        # Check that the parent Exception's message is set correctly
        assert Exception.__str__(exc) == message


class Test_SmartSurgeException_02_NegativeBehaviors:
    """Tests for negative behaviors of the SmartSurgeException class."""
    
    def test_handles_empty_message(self):
        """Test that exception handles empty message strings without errors."""
        exc = SmartSurgeException("")
        assert exc.message == ""
        assert str(exc) == ""
    
    def test_handles_unusual_message_types(self):
        """Test that exception properly handles unusual or non-string message types."""
        # Test with integer
        exc = SmartSurgeException(123)
        assert exc.message == 123
        assert str(exc) == "123"
        
        # Test with None
        exc = SmartSurgeException(None)
        assert exc.message is None
        
        # Test with object
        class TestObj:
            def __str__(self):
                return "TestObj string representation"
        
        obj = TestObj()
        exc = SmartSurgeException(obj)
        assert exc.message == obj
        assert str(exc) == "TestObj string representation"
    
    def test_accepts_unusual_context_types(self):
        """Test that exception accepts and processes unusual keyword argument types."""
        context = {
            "none_value": None,
            "number": 42.5,
            "complex": complex(1, 2),
            "list": [1, 2, 3],
            "nested_dict": {"key": "value"}
        }
        exc = SmartSurgeException("Test", **context)
        for key, value in context.items():
            assert exc.context[key] == value


class Test_SmartSurgeException_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the SmartSurgeException class."""
    
    def test_very_long_message(self):
        """Test that exception functions correctly with extremely long message strings."""
        long_msg = "a" * 10000  # 10,000 character message
        exc = SmartSurgeException(long_msg)
        assert exc.message == long_msg
        assert len(str(exc)) == 10000
    
    def test_many_context_variables(self):
        """Test that exception handles a large number of context variables efficiently."""
        # Create a dictionary with 100 context variables
        context = {f"key{i}": f"value{i}" for i in range(100)}
        exc = SmartSurgeException("Test many context vars", **context)
        assert len(exc.context) == 100
        assert all(f"key{i}" in exc.context for i in range(100))
    
    def test_unicode_characters(self):
        """Test that exception processes Unicode characters in message strings properly."""
        unicode_msg = "Unicode test: ñáéíóúüÑ 你好 😊 🚀 ∑∫√"
        exc = SmartSurgeException(unicode_msg)
        assert exc.message == unicode_msg
        assert str(exc) == unicode_msg


class Test_SmartSurgeException_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the SmartSurgeException class."""
    
    def test_handles_logger_initialization_failure(self):
        """Test that exception gracefully handles logger initialization failures."""
        with patch('logging.getLogger') as mock_get_logger:
            # Simulate logger initialization failure
            mock_get_logger.side_effect = Exception("Logger init failed")
            
            # Exception should still be created without errors
            exc = SmartSurgeException("Test logger init failure")
            assert exc.message == "Test logger init failure"
    
    def test_continues_when_logging_fails(self):
        """Test that exception continues when logging fails rather than raising additional exceptions."""
        with patch('logging.Logger.error') as mock_error:
            # Simulate logging error
            mock_error.side_effect = Exception("Logging failed")
            
            # Exception should still be created without errors
            exc = SmartSurgeException("Test continues despite logging failure")
            assert exc.message == "Test continues despite logging failure"


## RateLimitExceeded Tests

class Test_RateLimitExceeded_01_NominalBehaviors:
    """Tests for nominal behaviors of the RateLimitExceeded exception."""
    
    def test_stores_attributes(self):
        """Test that exception properly stores endpoint, method, retry_after, and response_headers attributes."""
        from smartsurge.exceptions import RateLimitExceeded
        
        endpoint = "api/users"
        method = "GET"
        retry_after = 60
        headers = {"X-RateLimit-Limit": "100"}
        
        exc = RateLimitExceeded(
            "Rate limit exceeded", 
            endpoint=endpoint, 
            method=method, 
            retry_after=retry_after,
            response_headers=headers
        )
        
        assert exc.endpoint == endpoint
        assert exc.method == method
        assert exc.retry_after == retry_after
        assert exc.response_headers == headers
    
    def test_converts_method_to_string(self):
        """Test that exception converts method object to string representation when creating context dictionary."""
        from smartsurge.exceptions import RateLimitExceeded
        from smartsurge.models import RequestMethod
        
        method = RequestMethod.GET
        exc = RateLimitExceeded("Rate limit exceeded", method=method)
        
        # The original method should be preserved
        assert exc.method == method
        
        # The context should have the string representation
        assert exc.context["method"] == "GET"
    
    def test_includes_rate_limit_info_in_logs(self):
        """Test that exception includes all rate limit information in the logged error message."""
        from smartsurge.exceptions import RateLimitExceeded
        
        with patch('logging.Logger.error') as mock_error:
            endpoint = "api/users"
            method = "GET"
            retry_after = 60
            headers = {"X-RateLimit-Limit": "100"}
            
            RateLimitExceeded(
                "Rate limit exceeded", 
                endpoint=endpoint, 
                method=method, 
                retry_after=retry_after,
                response_headers=headers
            )
            
            mock_error.assert_called_once()
            log_msg = mock_error.call_args[0][0]
            assert "Rate limit exceeded" in log_msg
            assert f"'endpoint': '{endpoint}'" in log_msg
            assert f"'method': '{method}'" in log_msg
            assert f"'retry_after': {retry_after}" in log_msg
    
    def test_retry_after_accessible(self):
        """Test that retry_after value is accessible to client code for implementing waiting periods."""
        from smartsurge.exceptions import RateLimitExceeded
        
        retry_after = 60
        exc = RateLimitExceeded("Rate limit exceeded", retry_after=retry_after)
        assert exc.retry_after == retry_after
    
    def test_passes_context_to_parent(self):
        """Test that exception passes specialized context values to parent class properly."""
        from smartsurge.exceptions import RateLimitExceeded, SmartSurgeException
        
        endpoint = "api/users"
        method = "GET"
        retry_after = 60
        
        with patch.object(SmartSurgeException, '__init__') as mock_init:
            mock_init.return_value = None
            
            RateLimitExceeded(
                "Rate limit exceeded", 
                endpoint=endpoint, 
                method=method, 
                retry_after=retry_after
            )
            
            mock_init.assert_called_once()
            _, kwargs = mock_init.call_args
            assert kwargs["endpoint"] == endpoint
            assert kwargs["method"] == "GET"
            assert kwargs["retry_after"] == retry_after


class Test_RateLimitExceeded_02_NegativeBehaviors:
    """Tests for negative behaviors of the RateLimitExceeded exception."""
    
    def test_handles_none_or_empty_endpoint(self):
        """Test that exception handles None or empty endpoint gracefully."""
        from smartsurge.exceptions import RateLimitExceeded
        
        # Test with None endpoint
        exc = RateLimitExceeded("Rate limit exceeded", endpoint=None)
        assert exc.endpoint is None
        
        # Test with empty endpoint
        exc = RateLimitExceeded("Rate limit exceeded", endpoint="")
        assert exc.endpoint == ""
    
    def test_handles_none_method(self):
        """Test that exception functions correctly when method is None."""
        from smartsurge.exceptions import RateLimitExceeded
        
        exc = RateLimitExceeded("Rate limit exceeded", method=None)
        assert exc.method is None
        assert exc.context["method"] is None
    
    def test_handles_none_or_non_integer_retry_after(self):
        """Test that exception operates properly with None or non-integer retry_after value."""
        from smartsurge.exceptions import RateLimitExceeded
        
        # Test with None retry_after
        exc = RateLimitExceeded("Rate limit exceeded", retry_after=None)
        assert exc.retry_after is None
        
        # Test with string retry_after
        exc = RateLimitExceeded("Rate limit exceeded", retry_after="30")
        assert exc.retry_after == "30"
        
        # Test with float retry_after
        exc = RateLimitExceeded("Rate limit exceeded", retry_after=30.5)
        assert exc.retry_after == 30.5


class Test_RateLimitExceeded_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the RateLimitExceeded exception."""
    
    def test_very_long_endpoint(self):
        """Test that exception processes very long endpoint strings without errors."""
        from smartsurge.exceptions import RateLimitExceeded
        
        long_endpoint = "a/b/" * 1000  # Very long endpoint
        exc = RateLimitExceeded("Rate limit exceeded", endpoint=long_endpoint)
        assert exc.endpoint == long_endpoint
    
    def test_large_retry_after(self):
        """Test that exception handles extremely large retry_after values correctly."""
        from smartsurge.exceptions import RateLimitExceeded
        
        large_retry = 10**9  # 1 billion seconds
        exc = RateLimitExceeded("Rate limit exceeded", retry_after=large_retry)
        assert exc.retry_after == large_retry
    
    def test_various_method_types(self):
        """Test that exception functions with various method types (string or RequestMethod enum)."""
        from smartsurge.exceptions import RateLimitExceeded
        from smartsurge.models import RequestMethod
        
        # Test with string
        exc = RateLimitExceeded("Rate limit exceeded", method="GET")
        assert exc.method == "GET"
        
        # Test with RequestMethod enum
        exc = RateLimitExceeded("Rate limit exceeded", method=RequestMethod.POST)
        assert exc.method == RequestMethod.POST
        assert exc.context["method"] == "POST"


class Test_RateLimitExceeded_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the RateLimitExceeded exception."""
    
    def test_graceful_handling_of_string_conversion_errors(self):
        """Test that exception gracefully handles string conversion errors from unusual method objects."""
        from smartsurge.exceptions import RateLimitExceeded
        
        class BadStringObject:
            def __str__(self):
                raise ValueError("String conversion error")
        
        with patch('logging.Logger.error'):  # Suppress logging
            bad_method = BadStringObject()
            
            # Should not raise an exception during initialization
            exc = RateLimitExceeded("Rate limit exceeded", method=bad_method)
            assert exc.method == bad_method
    
    def test_handles_invalid_response_headers(self):
        """Test that exception continues initialization even if response_headers contains invalid data."""
        from smartsurge.exceptions import RateLimitExceeded
        
        class NonDictHeaders:
            pass
        
        invalid_headers = NonDictHeaders()  # Not a dictionary
        
        # Should not raise an exception during initialization
        exc = RateLimitExceeded("Rate limit exceeded", response_headers=invalid_headers)
        assert exc.response_headers == invalid_headers


## StreamingError Tests

class Test_StreamingError_01_NominalBehaviors:
    """Tests for nominal behaviors of the StreamingError exception."""
    
    def test_stores_attributes(self):
        """Test that exception properly stores endpoint, position, and response attributes."""
        from smartsurge.exceptions import StreamingError
        
        endpoint = "api/stream"
        position = 1024
        response = MagicMock()
        response.status_code = 500
        
        exc = StreamingError(
            "Streaming error", 
            endpoint=endpoint, 
            position=position,
            response=response
        )
        
        assert exc.endpoint == endpoint
        assert exc.position == position
        assert exc.response == response
    
    def test_extracts_response_status_code(self):
        """Test that exception extracts and stores response status code when available."""
        from smartsurge.exceptions import StreamingError
        
        response = MagicMock()
        response.status_code = 429
        
        exc = StreamingError("Streaming error", response=response)
        
        assert exc.context["response_status"] == 429
    
    def test_includes_streaming_context_in_logs(self):
        """Test that exception includes streaming-specific context in error logs."""
        from smartsurge.exceptions import StreamingError
        
        with patch('logging.Logger.error') as mock_error:
            endpoint = "api/stream"
            position = 1024
            
            StreamingError(
                "Streaming error", 
                endpoint=endpoint, 
                position=position
            )
            
            mock_error.assert_called_once()
            log_msg = mock_error.call_args[0][0]
            assert "Streaming error" in log_msg
            assert f"'endpoint': '{endpoint}'" in log_msg
            assert f"'position': {position}" in log_msg
    
    def test_passes_context_to_parent(self):
        """Test that exception passes appropriate context to parent class."""
        from smartsurge.exceptions import StreamingError, SmartSurgeException
        
        endpoint = "api/stream"
        position = 1024
        response = MagicMock()
        response.status_code = 404
        
        with patch.object(SmartSurgeException, '__init__') as mock_init:
            mock_init.return_value = None
            
            StreamingError(
                "Streaming error", 
                endpoint=endpoint, 
                position=position,
                response=response
            )
            
            mock_init.assert_called_once()
            _, kwargs = mock_init.call_args
            assert kwargs["endpoint"] == endpoint
            assert kwargs["position"] == position
            assert kwargs["response_status"] == 404


class Test_StreamingError_02_NegativeBehaviors:
    """Tests for negative behaviors of the StreamingError exception."""
    
    def test_handles_none_or_invalid_endpoint(self):
        """Test that exception handles None or invalid endpoint value gracefully."""
        from smartsurge.exceptions import StreamingError
        
        # Test with None endpoint
        exc = StreamingError("Streaming error", endpoint=None)
        assert exc.endpoint is None
        
        # Test with empty endpoint
        exc = StreamingError("Streaming error", endpoint="")
        assert exc.endpoint == ""
        
        # Test with non-string endpoint
        exc = StreamingError("Streaming error", endpoint=123)
        assert exc.endpoint == 123
    
    def test_handles_none_or_negative_position(self):
        """Test that exception functions correctly with None or negative position values."""
        from smartsurge.exceptions import StreamingError
        
        # Test with None position
        exc = StreamingError("Streaming error", position=None)
        assert exc.position is None
        
        # Test with negative position
        exc = StreamingError("Streaming error", position=-100)
        assert exc.position == -100
    
    def test_handles_none_response(self):
        """Test that exception operates properly when response object is None."""
        from smartsurge.exceptions import StreamingError
        
        exc = StreamingError("Streaming error", response=None)
        assert exc.response is None
        assert exc.context["response_status"] is None


class Test_StreamingError_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the StreamingError exception."""
    
    def test_large_position_values(self):
        """Test that exception processes very large position values correctly."""
        from smartsurge.exceptions import StreamingError
        
        large_position = 10**15  # Very large position
        exc = StreamingError("Streaming error", position=large_position)
        assert exc.position == large_position
    
    def test_zero_position(self):
        """Test that exception handles cases where position is zero appropriately."""
        from smartsurge.exceptions import StreamingError
        
        exc = StreamingError("Streaming error", position=0)
        assert exc.position == 0
    
    def test_different_response_types(self):
        """Test that exception functions with response objects of different types and structures."""
        from smartsurge.exceptions import StreamingError
        
        # Dictionary response
        dict_response = {"status_code": 500, "message": "Error"}
        exc = StreamingError("Streaming error", response=dict_response)
        assert exc.response == dict_response
        
        # Custom object response
        class CustomResponse:
            def __init__(self):
                self.status_code = 404
                self.body = "Not found"
        
        custom_response = CustomResponse()
        exc = StreamingError("Streaming error", response=custom_response)
        assert exc.response == custom_response
        assert exc.context["response_status"] == 404


class Test_StreamingError_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the StreamingError exception."""
    
    def test_handles_response_without_status_code(self):
        """Test that exception gracefully handles response objects without a status_code attribute."""
        from smartsurge.exceptions import StreamingError
        
        # Response without status_code
        class ResponseWithoutStatus:
            pass
        
        response = ResponseWithoutStatus()
        exc = StreamingError("Streaming error", response=response)
        assert exc.response == response
        assert exc.context["response_status"] is None
    
    def test_handles_unexpected_response_structure(self):
        """Test that exception continues initialization when response object has unexpected structure."""
        from smartsurge.exceptions import StreamingError
        
        class WeirdResponse:
            @property
            def status_code(self):
                raise AttributeError("Accessing status_code caused an error")
        
        response = WeirdResponse()
        
        # Should not raise during initialization
        exc = StreamingError("Streaming error", response=response)
        assert exc.response == response
        # Context might contain None or default for response_status


## ResumeError Tests

class Test_ResumeError_01_NominalBehaviors:
    """Tests for nominal behaviors of the ResumeError exception."""
    
    def test_stores_attributes(self):
        """Test that exception properly stores state_file and original_error attributes."""
        from smartsurge.exceptions import ResumeError
        
        state_file = "cache/resume_state.json"
        original_error = ValueError("Failed to parse JSON")
        
        exc = ResumeError(
            "Resume error", 
            state_file=state_file, 
            original_error=original_error
        )
        
        assert exc.state_file == state_file
        assert exc.original_error == original_error
    
    def test_captures_traceback(self):
        """Test that exception captures and includes traceback information when original_error is provided."""
        from smartsurge.exceptions import ResumeError
        
        original_error = ValueError("Failed to parse state file")
        
        exc = ResumeError("Resume error", original_error=original_error)
        
        assert exc.context["original_error"] == str(original_error)
        assert "traceback" in exc.context
        assert exc.context["traceback"] is not None
    
    def test_converts_original_error_to_string(self):
        """Test that exception converts original_error to string representation for context."""
        from smartsurge.exceptions import ResumeError
        
        class CustomError(Exception):
            def __str__(self):
                return "Custom error representation"
        
        original_error = CustomError()
        
        exc = ResumeError("Resume error", original_error=original_error)
        
        assert exc.context["original_error"] == "Custom error representation"
    
    def test_passes_context_to_parent(self):
        """Test that exception passes appropriate context to parent class."""
        from smartsurge.exceptions import ResumeError, SmartSurgeException
        
        state_file = "cache/resume_state.json"
        original_error = ValueError("Failed to parse state file")
        
        with patch.object(SmartSurgeException, '__init__') as mock_init:
            mock_init.return_value = None
            
            with patch('traceback.format_exc', return_value="Mocked traceback"):
                ResumeError(
                    "Resume error", 
                    state_file=state_file, 
                    original_error=original_error
                )
                
                mock_init.assert_called_once()
                _, kwargs = mock_init.call_args
                assert kwargs["state_file"] == state_file
                assert kwargs["original_error"] == str(original_error)
                assert kwargs["traceback"] == "Mocked traceback"


class Test_ResumeError_02_NegativeBehaviors:
    """Tests for negative behaviors of the ResumeError exception."""
    
    def test_handles_none_or_empty_state_file(self):
        """Test that exception handles None or empty state_file value gracefully."""
        from smartsurge.exceptions import ResumeError
        
        # Test with None state_file
        exc = ResumeError("Resume error", state_file=None)
        assert exc.state_file is None
        
        # Test with empty state_file
        exc = ResumeError("Resume error", state_file="")
        assert exc.state_file == ""
    
    def test_handles_none_original_error(self):
        """Test that exception functions correctly when original_error is None."""
        from smartsurge.exceptions import ResumeError
        
        exc = ResumeError("Resume error", original_error=None)
        assert exc.original_error is None
        assert exc.context["original_error"] is None
        assert exc.context["traceback"] is None
    
    def test_handles_various_error_types(self):
        """Test that exception operates properly with various error types as original_error."""
        from smartsurge.exceptions import ResumeError
        
        # Standard exception
        exc = ResumeError("Resume error", original_error=ValueError("Value error"))
        assert isinstance(exc.original_error, ValueError)
        
        # Custom exception
        class CustomError(Exception):
            pass
        
        custom_error = CustomError()
        exc = ResumeError("Resume error", original_error=custom_error)
        assert exc.original_error == custom_error
        
        # Non-exception object
        non_exception = object()
        exc = ResumeError("Resume error", original_error=non_exception)
        assert exc.original_error == non_exception


class Test_ResumeError_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ResumeError exception."""
    
    def test_long_state_file_path(self):
        """Test that exception processes very long state_file path strings correctly."""
        from smartsurge.exceptions import ResumeError
        
        long_path = "path/to/" * 100 + "state.json"  # Very long path
        exc = ResumeError("Resume error", state_file=long_path)
        assert exc.state_file == long_path
    
    def test_complex_original_error(self):
        """Test that exception handles complex or deeply nested original_error objects."""
        from smartsurge.exceptions import ResumeError
        
        # Create a chain of exceptions
        try:
            try:
                try:
                    raise ValueError("Inner error")
                except ValueError as e:
                    raise KeyError("Middle error") from e
            except KeyError as e:
                raise RuntimeError("Outer error") from e
        except RuntimeError as nested_error:
            exc = ResumeError("Resume error", original_error=nested_error)
            assert exc.original_error == nested_error
            assert "Outer error" in exc.context["original_error"]
    
    def test_original_error_without_standard_attributes(self):
        """Test that exception functions with original_error objects that don't provide standard attributes."""
        from smartsurge.exceptions import ResumeError
        
        class WeirdError:
            # Not a proper exception, doesn't inherit from Exception
            def __str__(self):
                return "Weird error object"
        
        weird_error = WeirdError()
        exc = ResumeError("Resume error", original_error=weird_error)
        assert exc.original_error == weird_error
        assert exc.context["original_error"] == "Weird error object"


class Test_ResumeError_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ResumeError exception."""
    
    def test_handles_traceback_generation_errors(self):
        """Test that exception gracefully handles errors during traceback generation."""
        from smartsurge.exceptions import ResumeError
        
        with patch('traceback.format_exc') as mock_format_exc:
            # Simulate traceback generation failure
            mock_format_exc.side_effect = Exception("Traceback generation failed")
            
            # Should not fail during initialization
            exc = ResumeError(
                "Resume error", 
                state_file="state.json", 
                original_error=ValueError("Original error")
            )
            assert exc.state_file == "state.json"
            assert isinstance(exc.original_error, ValueError)
    
    def test_handles_string_conversion_failure(self):
        """Test that exception continues initialization when string conversion of original_error fails."""
        from smartsurge.exceptions import ResumeError
        
        class BadStringError:
            def __str__(self):
                raise RuntimeError("String conversion failed")
        
        bad_error = BadStringError()
        
        # Should not fail during initialization
        exc = ResumeError("Resume error", original_error=bad_error)
        assert exc.original_error == bad_error
        # Context may contain None or some fallback for original_error


## ValidationError Tests

class Test_ValidationError_01_NominalBehaviors:
    """Tests for nominal behaviors of the ValidationError exception."""
    
    def test_stores_attributes(self):
        """Test that exception properly stores field and value attributes."""
        from smartsurge.exceptions import ValidationError
        
        field = "email"
        value = "invalid-email"
        
        exc = ValidationError(
            "Invalid email format", 
            field=field, 
            value=value
        )
        
        assert exc.field == field
        assert exc.value == value
    
    def test_converts_value_to_string(self):
        """Test that exception converts value to string representation for context."""
        from smartsurge.exceptions import ValidationError
        
        class CustomValue:
            def __str__(self):
                return "Custom value representation"
        
        custom_value = CustomValue()
        exc = ValidationError("Validation error", value=custom_value)
        
        assert exc.context["value"] == "Custom value representation"
    
    def test_includes_validation_context_in_logs(self):
        """Test that exception includes validation-specific context in error logs."""
        from smartsurge.exceptions import ValidationError
        
        with patch('logging.Logger.error') as mock_error:
            field = "username"
            value = "ab"  # Too short
            
            ValidationError(
                "Username too short", 
                field=field, 
                value=value
            )
            
            mock_error.assert_called_once()
            log_msg = mock_error.call_args[0][0]
            assert "Username too short" in log_msg
            assert f"'field': '{field}'" in log_msg
            assert f"'value': '{value}'" in log_msg
    
    def test_passes_context_to_parent(self):
        """Test that exception passes appropriate context to parent class."""
        from smartsurge.exceptions import ValidationError, SmartSurgeException
        
        field = "password"
        value = "weak"
        
        with patch.object(SmartSurgeException, '__init__') as mock_init:
            mock_init.return_value = None
            
            ValidationError(
                "Password too weak", 
                field=field, 
                value=value
            )
            
            mock_init.assert_called_once()
            _, kwargs = mock_init.call_args
            assert kwargs["field"] == field
            assert kwargs["value"] == "weak"


class Test_ValidationError_02_NegativeBehaviors:
    """Tests for negative behaviors of the ValidationError exception."""
    
    def test_handles_none_or_empty_field(self):
        """Test that exception handles None or empty field value gracefully."""
        from smartsurge.exceptions import ValidationError
        
        # Test with None field
        exc = ValidationError("Validation error", field=None)
        assert exc.field is None
        
        # Test with empty field
        exc = ValidationError("Validation error", field="")
        assert exc.field == ""
    
    def test_handles_none_value(self):
        """Test that exception functions correctly when value is None."""
        from smartsurge.exceptions import ValidationError
        
        exc = ValidationError("Validation error", value=None)
        assert exc.value is None
        assert exc.context["value"] is None
    
    def test_handles_various_value_types(self):
        """Test that exception operates properly with various non-string value types."""
        from smartsurge.exceptions import ValidationError
        
        # Integer value
        exc = ValidationError("Validation error", value=42)
        assert exc.value == 42
        assert exc.context["value"] == "42"
        
        # Dictionary value
        dict_value = {"key": "value"}
        exc = ValidationError("Validation error", value=dict_value)
        assert exc.value == dict_value
        assert exc.context["value"] == str(dict_value)
        
        # List value
        list_value = [1, 2, 3]
        exc = ValidationError("Validation error", value=list_value)
        assert exc.value == list_value
        assert exc.context["value"] == str(list_value)


class Test_ValidationError_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ValidationError exception."""
    
    def test_long_field_name(self):
        """Test that exception processes very long field name strings correctly."""
        from smartsurge.exceptions import ValidationError
        
        long_field = "user_profile_settings_notification_preferences_email_marketing_" * 5
        exc = ValidationError("Validation error", field=long_field)
        assert exc.field == long_field
    
    def test_complex_object_value(self):
        """Test that exception handles complex object values by converting to appropriate string representation."""
        from smartsurge.exceptions import ValidationError
        
        class ComplexObject:
            def __init__(self, id, name):
                self.id = id
                self.name = name
                
            def __str__(self):
                return f"ComplexObject(id={self.id}, name={self.name})"
        
        obj = ComplexObject(123, "Test Object")
        exc = ValidationError("Validation error", value=obj)
        assert exc.value == obj
        assert exc.context["value"] == str(obj)
    
    def test_unusual_string_representations(self):
        """Test that exception functions with values that have unusual string representations."""
        from smartsurge.exceptions import ValidationError
        
        class WeirdRepr:
            def __str__(self):
                return "🤔" * 50  # String with emojis
        
        weird = WeirdRepr()
        exc = ValidationError("Validation error", value=weird)
        assert exc.value == weird
        assert exc.context["value"] == str(weird)
        assert len(exc.context["value"]) == 50


class Test_ValidationError_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ValidationError exception."""
    
    def test_handles_string_conversion_errors(self):
        """Test that exception gracefully handles string conversion errors for unusual value types."""
        from smartsurge.exceptions import ValidationError
        
        class BadStringValue:
            def __str__(self):
                raise ValueError("String conversion error")
        
        bad_value = BadStringValue()
        
        # Should not fail during initialization
        exc = ValidationError("Validation error", value=bad_value)
        assert exc.value == bad_value
        # Context may have None or fallback for value
    
    def test_continues_when_value_conversion_fails(self):
        """Test that exception continues initialization when value cannot be properly converted to string."""
        from smartsurge.exceptions import ValidationError
        
        class RecursiveObject:
            def __init__(self):
                self.self_ref = self
                
            def __str__(self):
                # This would cause infinite recursion in str() conversion
                return f"RecursiveObject({self.self_ref})"
        
        recursive = RecursiveObject()
        
        # Should not fail during initialization
        with patch('logging.Logger.error'):  # Suppress logging
            exc = ValidationError("Validation error", value=recursive)
            assert exc.value == recursive


## ConfigurationError Tests

class Test_ConfigurationError_01_NominalBehaviors:
    """Tests for nominal behaviors of the ConfigurationError exception."""
    
    def test_stores_attributes(self):
        """Test that exception properly stores parameter and value attributes."""
        from smartsurge.exceptions import ConfigurationError
        
        parameter = "timeout"
        value = -5  # Invalid negative timeout
        
        exc = ConfigurationError(
            "Invalid timeout value", 
            parameter=parameter, 
            value=value
        )
        
        assert exc.parameter == parameter
        assert exc.value == value
    
    def test_converts_value_to_string(self):
        """Test that exception converts value to string representation for context."""
        from smartsurge.exceptions import ConfigurationError
        
        class ConfigValue:
            def __str__(self):
                return "Config value representation"
        
        config_value = ConfigValue()
        exc = ConfigurationError("Configuration error", value=config_value)
        
        assert exc.context["value"] == "Config value representation"
    
    def test_includes_configuration_context_in_logs(self):
        """Test that exception includes configuration-specific context in error logs."""
        from smartsurge.exceptions import ConfigurationError
        
        with patch('logging.Logger.error') as mock_error:
            parameter = "max_retries"
            value = 20  # Too high
            
            ConfigurationError(
                "max_retries exceeds maximum allowed", 
                parameter=parameter, 
                value=value
            )
            
            mock_error.assert_called_once()
            log_msg = mock_error.call_args[0][0]
            assert "max_retries exceeds maximum allowed" in log_msg
            assert f"'parameter': '{parameter}'" in log_msg
            assert f"'value': '{value}'" in log_msg
    
    def test_passes_context_to_parent(self):
        """Test that exception passes appropriate context to parent class."""
        from smartsurge.exceptions import ConfigurationError, SmartSurgeException
        
        parameter = "base_url"
        value = "invalid-url"
        
        with patch.object(SmartSurgeException, '__init__') as mock_init:
            mock_init.return_value = None
            
            ConfigurationError(
                "Invalid URL format", 
                parameter=parameter, 
                value=value
            )
            
            mock_init.assert_called_once()
            _, kwargs = mock_init.call_args
            assert kwargs["parameter"] == parameter
            assert kwargs["value"] == "invalid-url"


class Test_ConfigurationError_02_NegativeBehaviors:
    """Tests for negative behaviors of the ConfigurationError exception."""
    
    def test_handles_none_or_empty_parameter(self):
        """Test that exception handles None or empty parameter value gracefully."""
        from smartsurge.exceptions import ConfigurationError
        
        # Test with None parameter
        exc = ConfigurationError("Configuration error", parameter=None)
        assert exc.parameter is None
        
        # Test with empty parameter
        exc = ConfigurationError("Configuration error", parameter="")
        assert exc.parameter == ""
    
    def test_handles_none_value(self):
        """Test that exception functions correctly when value is None."""
        from smartsurge.exceptions import ConfigurationError
        
        exc = ConfigurationError("Configuration error", value=None)
        assert exc.value is None
        assert exc.context["value"] is None
    
    def test_handles_various_value_types(self):
        """Test that exception operates properly with various non-string value types."""
        from smartsurge.exceptions import ConfigurationError
        
        # Boolean value
        exc = ConfigurationError("Configuration error", value=False)
        assert exc.value is False
        assert exc.context["value"] == "False"
        
        # Complex object value
        complex_value = complex(3, 4)
        exc = ConfigurationError("Configuration error", value=complex_value)
        assert exc.value == complex_value
        assert exc.context["value"] == str(complex_value)


class Test_ConfigurationError_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ConfigurationError exception."""
    
    def test_long_parameter_name(self):
        """Test that exception processes very long parameter name strings correctly."""
        from smartsurge.exceptions import ConfigurationError
        
        long_parameter = "advanced_configuration_options_performance_tuning_thread_pool_" * 5
        exc = ConfigurationError("Configuration error", parameter=long_parameter)
        assert exc.parameter == long_parameter
    
    def test_complex_object_value(self):
        """Test that exception handles complex object values by converting to appropriate string representation."""
        from smartsurge.exceptions import ConfigurationError
        
        class ConfigObject:
            def __init__(self, settings):
                self.settings = settings
                
            def __str__(self):
                return f"ConfigObject(settings={self.settings})"
        
        obj = ConfigObject({"key1": "value1", "key2": "value2"})
        exc = ConfigurationError("Configuration error", value=obj)
        assert exc.value == obj
        assert exc.context["value"] == str(obj)
    
    def test_unusual_string_representations(self):
        """Test that exception functions with values that have unusual string representations."""
        from smartsurge.exceptions import ConfigurationError
        
        class CustomConfig:
            def __str__(self):
                return "\n".join([f"Setting{i}" for i in range(10)])  # Multi-line string
        
        custom = CustomConfig()
        exc = ConfigurationError("Configuration error", value=custom)
        assert exc.value == custom
        assert exc.context["value"] == str(custom)
        assert "\n" in exc.context["value"]


class Test_ConfigurationError_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ConfigurationError exception."""
    
    def test_handles_string_conversion_errors(self):
        """Test that exception gracefully handles string conversion errors for unusual value types."""
        from smartsurge.exceptions import ConfigurationError
        
        class BadConfigValue:
            def __str__(self):
                raise ValueError("String conversion error")
        
        bad_value = BadConfigValue()
        
        # Should not fail during initialization
        with patch('logging.Logger.error'):  # Suppress logging
            exc = ConfigurationError("Configuration error", value=bad_value)
            assert exc.value == bad_value
    
    def test_continues_when_value_conversion_fails(self):
        """Test that exception continues initialization when value cannot be properly converted to string."""
        from smartsurge.exceptions import ConfigurationError
        
        # Create a cyclic reference that would cause problems in string conversion
        a = {}
        b = {"a": a}
        a["b"] = b
        
        # Should not fail during initialization
        with patch('logging.Logger.error'):  # Suppress logging
            exc = ConfigurationError("Configuration error", value=a)
            assert exc.value == a


## ConnectionError Tests

class Test_ConnectionError_01_NominalBehaviors:
    """Tests for nominal behaviors of the ConnectionError exception."""
    
    def test_stores_attributes(self):
        """Test that exception properly stores endpoint and original_error attributes."""
        from smartsurge.exceptions import ConnectionError
        
        endpoint = "https://api.example.com/data"
        original_error = TimeoutError("Connection timed out")
        
        exc = ConnectionError(
            "Failed to connect", 
            endpoint=endpoint, 
            original_error=original_error
        )
        
        assert exc.endpoint == endpoint
        assert exc.original_error == original_error
    
    def test_captures_traceback(self):
        """Test that exception captures and includes traceback information when original_error is provided."""
        from smartsurge.exceptions import ConnectionError
        
        original_error = ConnectionRefusedError("Connection refused")
        
        exc = ConnectionError("Connection error", original_error=original_error)
        
        assert exc.context["original_error"] == str(original_error)
        assert "traceback" in exc.context
        assert exc.context["traceback"] is not None
    
    def test_converts_original_error_to_string(self):
        """Test that exception converts original_error to string representation for context."""
        from smartsurge.exceptions import ConnectionError
        
        class NetworkError(Exception):
            def __str__(self):
                return "Custom network error"
        
        original_error = NetworkError()
        
        exc = ConnectionError("Connection error", original_error=original_error)
        
        assert exc.context["original_error"] == "Custom network error"
    
    def test_passes_context_to_parent(self):
        """Test that exception passes appropriate context to parent class."""
        from smartsurge.exceptions import ConnectionError, SmartSurgeException
        
        endpoint = "https://api.example.com/data"
        original_error = TimeoutError("Connection timed out")
        
        with patch.object(SmartSurgeException, '__init__') as mock_init:
            mock_init.return_value = None
            
            with patch('traceback.format_exc', return_value="Mocked traceback"):
                ConnectionError(
                    "Connection error", 
                    endpoint=endpoint, 
                    original_error=original_error
                )
                
                mock_init.assert_called_once()
                _, kwargs = mock_init.call_args
                assert kwargs["endpoint"] == endpoint
                assert kwargs["original_error"] == str(original_error)
                assert kwargs["traceback"] == "Mocked traceback"


class Test_ConnectionError_02_NegativeBehaviors:
    """Tests for negative behaviors of the ConnectionError exception."""
    
    def test_handles_none_or_empty_endpoint(self):
        """Test that exception handles None or empty endpoint value gracefully."""
        from smartsurge.exceptions import ConnectionError
        
        # Test with None endpoint
        exc = ConnectionError("Connection error", endpoint=None)
        assert exc.endpoint is None
        
        # Test with empty endpoint
        exc = ConnectionError("Connection error", endpoint="")
        assert exc.endpoint == ""
    
    def test_handles_none_original_error(self):
        """Test that exception functions correctly when original_error is None."""
        from smartsurge.exceptions import ConnectionError
        
        exc = ConnectionError("Connection error", original_error=None)
        assert exc.original_error is None
        assert exc.context["original_error"] is None
        assert exc.context["traceback"] is None
    
    def test_handles_various_error_types(self):
        """Test that exception operates properly with various error types as original_error."""
        from smartsurge.exceptions import ConnectionError
        import socket
        
        # Standard exception
        exc = ConnectionError("Connection error", original_error=Exception("Generic error"))
        assert isinstance(exc.original_error, Exception)
        
        # Socket error
        socket_error = socket.error("Socket error")
        exc = ConnectionError("Connection error", original_error=socket_error)
        assert exc.original_error == socket_error
        
        # Non-exception object
        non_exception = "Connection refused"  # Just a string
        exc = ConnectionError("Connection error", original_error=non_exception)
        assert exc.original_error == non_exception


class Test_ConnectionError_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ConnectionError exception."""
    
    def test_long_endpoint(self):
        """Test that exception processes very long endpoint strings correctly."""
        from smartsurge.exceptions import ConnectionError
        
        long_endpoint = "https://api.example.com/" + "path/to/resource/" * 50
        exc = ConnectionError("Connection error", endpoint=long_endpoint)
        assert exc.endpoint == long_endpoint
    
    def test_complex_original_error(self):
        """Test that exception handles complex or deeply nested original_error objects."""
        from smartsurge.exceptions import ConnectionError
        
        # Create a chain of exceptions
        try:
            try:
                try:
                    raise ConnectionRefusedError("Connection refused")
                except ConnectionRefusedError as e:
                    raise TimeoutError("Connection timed out") from e
            except TimeoutError as e:
                raise OSError("Network error") from e
        except OSError as nested_error:
            exc = ConnectionError("Connection error", original_error=nested_error)
            assert exc.original_error == nested_error
            assert "Network error" in exc.context["original_error"]
    
    def test_original_error_without_standard_attributes(self):
        """Test that exception functions with original_error objects that don't provide standard attributes."""
        from smartsurge.exceptions import ConnectionError
        
        class CustomNetworkError:
            # Not a standard exception
            def __str__(self):
                return "Custom network error object"
        
        custom_error = CustomNetworkError()
        exc = ConnectionError("Connection error", original_error=custom_error)
        assert exc.original_error == custom_error
        assert exc.context["original_error"] == "Custom network error object"


class Test_ConnectionError_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ConnectionError exception."""
    
    def test_handles_traceback_generation_errors(self):
        """Test that exception gracefully handles errors during traceback generation."""
        from smartsurge.exceptions import ConnectionError
        
        with patch('traceback.format_exc') as mock_format_exc:
            # Simulate traceback generation failure
            mock_format_exc.side_effect = Exception("Traceback generation failed")
            
            # Should not fail during initialization
            exc = ConnectionError(
                "Connection error", 
                endpoint="https://api.example.com", 
                original_error=ValueError("Original error")
            )
            assert exc.endpoint == "https://api.example.com"
            assert isinstance(exc.original_error, ValueError)
    
    def test_handles_string_conversion_failure(self):
        """Test that exception continues initialization when string conversion of original_error fails."""
        from smartsurge.exceptions import ConnectionError
        
        class BadStringError:
            def __str__(self):
                raise RuntimeError("String conversion failed")
        
        bad_error = BadStringError()
        
        # Should not fail during initialization
        with patch('logging.Logger.error'):  # Suppress logging
            exc = ConnectionError("Connection error", original_error=bad_error)
            assert exc.original_error == bad_error


## TimeoutError Tests

class Test_TimeoutError_01_NominalBehaviors:
    """Tests for nominal behaviors of the TimeoutError exception."""
    
    def test_stores_attributes(self):
        """Test that exception properly stores endpoint and timeout attributes."""
        from smartsurge.exceptions import TimeoutError
        
        endpoint = "https://api.example.com/data"
        timeout = 30.5
        
        exc = TimeoutError(
            "Request timed out", 
            endpoint=endpoint, 
            timeout=timeout
        )
        
        assert exc.endpoint == endpoint
        assert exc.timeout == timeout
    
    def test_includes_timeout_context_in_logs(self):
        """Test that exception includes timeout-specific context in error logs."""
        from smartsurge.exceptions import TimeoutError
        
        with patch('logging.Logger.error') as mock_error:
            endpoint = "https://api.example.com"
            timeout = 15
            
            TimeoutError(
                "Request timed out", 
                endpoint=endpoint, 
                timeout=timeout
            )
            
            mock_error.assert_called_once()
            log_msg = mock_error.call_args[0][0]
            assert "Request timed out" in log_msg
            assert f"'endpoint': '{endpoint}'" in log_msg
            assert f"'timeout': {timeout}" in log_msg
    
    def test_passes_context_to_parent(self):
        """Test that exception passes appropriate context to parent class."""
        from smartsurge.exceptions import TimeoutError, SmartSurgeException
        
        endpoint = "https://api.example.com"
        timeout = 10
        
        with patch.object(SmartSurgeException, '__init__') as mock_init:
            mock_init.return_value = None
            
            TimeoutError(
                "Request timed out", 
                endpoint=endpoint, 
                timeout=timeout
            )
            
            mock_init.assert_called_once()
            _, kwargs = mock_init.call_args
            assert kwargs["endpoint"] == endpoint
            assert kwargs["timeout"] == timeout
    
    def test_timeout_value_accessible(self):
        """Test that timeout value is accessible to client code."""
        from smartsurge.exceptions import TimeoutError
        
        timeout = 20
        exc = TimeoutError("Request timed out", timeout=timeout)
        assert exc.timeout == timeout


class Test_TimeoutError_02_NegativeBehaviors:
    """Tests for negative behaviors of the TimeoutError exception."""
    
    def test_handles_none_or_empty_endpoint(self):
        """Test that exception handles None or empty endpoint value gracefully."""
        from smartsurge.exceptions import TimeoutError
        
        # Test with None endpoint
        exc = TimeoutError("Timeout error", endpoint=None)
        assert exc.endpoint is None
        
        # Test with empty endpoint
        exc = TimeoutError("Timeout error", endpoint="")
        assert exc.endpoint == ""
    
    def test_handles_none_timeout(self):
        """Test that exception functions correctly when timeout is None."""
        from smartsurge.exceptions import TimeoutError
        
        exc = TimeoutError("Timeout error", timeout=None)
        assert exc.timeout is None
    
    def test_handles_invalid_timeout_values(self):
        """Test that exception operates properly with invalid timeout values."""
        from smartsurge.exceptions import TimeoutError
        
        # Negative timeout
        exc = TimeoutError("Timeout error", timeout=-5)
        assert exc.timeout == -5
        
        # String timeout
        exc = TimeoutError("Timeout error", timeout="30s")
        assert exc.timeout == "30s"
        
        # Zero timeout
        exc = TimeoutError("Timeout error", timeout=0)
        assert exc.timeout == 0


class Test_TimeoutError_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the TimeoutError exception."""
    
    def test_long_endpoint(self):
        """Test that exception processes very long endpoint strings correctly."""
        from smartsurge.exceptions import TimeoutError
        
        long_endpoint = "https://api.example.com/" + "path/to/resource/" * 50
        exc = TimeoutError("Timeout error", endpoint=long_endpoint)
        assert exc.endpoint == long_endpoint
    
    def test_large_timeout_values(self):
        """Test that exception handles very large timeout values appropriately."""
        from smartsurge.exceptions import TimeoutError
        
        large_timeout = 10**6  # One million seconds
        exc = TimeoutError("Timeout error", timeout=large_timeout)
        assert exc.timeout == large_timeout
    
    def test_small_timeout_values(self):
        """Test that exception functions with very small (near-zero) timeout values correctly."""
        from smartsurge.exceptions import TimeoutError
        
        small_timeout = 0.0001  # Very small timeout
        exc = TimeoutError("Timeout error", timeout=small_timeout)
        assert exc.timeout == small_timeout


class Test_TimeoutError_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the TimeoutError exception."""
    
    def test_handles_non_numeric_timeout(self):
        """Test that exception gracefully handles non-numeric timeout values."""
        from smartsurge.exceptions import TimeoutError
        
        class CustomTimeout:
            def __str__(self):
                return "Custom timeout object"
        
        custom_timeout = CustomTimeout()
        
        # Should not fail during initialization
        exc = TimeoutError("Timeout error", timeout=custom_timeout)
        assert exc.timeout == custom_timeout
    
    def test_continues_with_invalid_timeout(self):
        """Test that exception continues initialization even with invalid timeout information."""
        from smartsurge.exceptions import TimeoutError
        
        class BadTimeout:
            def __float__(self):
                raise ValueError("Cannot convert to float")
                
            def __str__(self):
                return "Bad timeout object"
        
        bad_timeout = BadTimeout()
        
        # Should not fail during initialization
        with patch('logging.Logger.error'):  # Suppress logging
            exc = TimeoutError("Timeout error", timeout=bad_timeout)
            assert exc.timeout == bad_timeout


## ServerError Tests

class Test_ServerError_01_NominalBehaviors:
    """Tests for nominal behaviors of the ServerError exception."""
    
    def test_stores_attributes(self):
        """Test that exception properly stores endpoint, status_code, and response attributes."""
        from smartsurge.exceptions import ServerError
        
        endpoint = "https://api.example.com/data"
        status_code = 500
        response = MagicMock()
        response.text = "Internal Server Error"
        
        exc = ServerError(
            "Server error occurred", 
            endpoint=endpoint, 
            status_code=status_code,
            response=response
        )
        
        assert exc.endpoint == endpoint
        assert exc.status_code == status_code
        assert exc.response == response
    
    def test_extracts_limited_response_text(self):
        """Test that exception extracts and stores limited response text (first 200 characters) when available."""
        from smartsurge.exceptions import ServerError
        
        response = MagicMock()
        response.text = "Error details: " + "x" * 300  # Response text longer than 200 chars
        
        exc = ServerError("Server error", response=response)
        
        assert exc.context["response_text"].startswith("Error details:")
        assert len(exc.context["response_text"]) == 200  # Truncated to 200 chars
    
    def test_includes_server_error_context_in_logs(self):
        """Test that exception includes server error-specific context in error logs."""
        from smartsurge.exceptions import ServerError
        
        with patch('logging.Logger.error') as mock_error:
            endpoint = "https://api.example.com"
            status_code = 502
            
            ServerError(
                "Bad Gateway", 
                endpoint=endpoint, 
                status_code=status_code
            )
            
            mock_error.assert_called_once()
            log_msg = mock_error.call_args[0][0]
            assert "Bad Gateway" in log_msg
            assert f"'endpoint': '{endpoint}'" in log_msg
            assert f"'status_code': {status_code}" in log_msg
    
    def test_passes_context_to_parent(self):
        """Test that exception passes appropriate context to parent class."""
        from smartsurge.exceptions import ServerError, SmartSurgeException
        
        endpoint = "https://api.example.com"
        status_code = 503
        response = MagicMock()
        response.text = "Service Unavailable"
        
        with patch.object(SmartSurgeException, '__init__') as mock_init:
            mock_init.return_value = None
            
            ServerError(
                "Service Unavailable", 
                endpoint=endpoint, 
                status_code=status_code,
                response=response
            )
            
            mock_init.assert_called_once()
            _, kwargs = mock_init.call_args
            assert kwargs["endpoint"] == endpoint
            assert kwargs["status_code"] == status_code
            assert kwargs["response_text"] == "Service Unavailable"
    
    def test_status_code_accessible(self):
        """Test that status_code is accessible to client code for error handling."""
        from smartsurge.exceptions import ServerError
        
        status_code = 500
        exc = ServerError("Server error", status_code=status_code)
        assert exc.status_code == status_code


class Test_ServerError_02_NegativeBehaviors:
    """Tests for negative behaviors of the ServerError exception."""
    
    def test_handles_none_or_empty_endpoint(self):
        """Test that exception handles None or empty endpoint value gracefully."""
        from smartsurge.exceptions import ServerError
        
        # Test with None endpoint
        exc = ServerError("Server error", endpoint=None)
        assert exc.endpoint is None
        
        # Test with empty endpoint
        exc = ServerError("Server error", endpoint="")
        assert exc.endpoint == ""
    
    def test_handles_none_status_code(self):
        """Test that exception functions correctly when status_code is None."""
        from smartsurge.exceptions import ServerError
        
        exc = ServerError("Server error", status_code=None)
        assert exc.status_code is None
    
    def test_handles_none_response(self):
        """Test that exception operates properly when response object is None."""
        from smartsurge.exceptions import ServerError
        
        exc = ServerError("Server error", response=None)
        assert exc.response is None
        assert exc.context["response_text"] is None


class Test_ServerError_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ServerError exception."""
    
    def test_long_endpoint(self):
        """Test that exception processes very long endpoint strings correctly."""
        from smartsurge.exceptions import ServerError
        
        long_endpoint = "https://api.example.com/" + "path/to/resource/" * 50
        exc = ServerError("Server error", endpoint=long_endpoint)
        assert exc.endpoint == long_endpoint
    
    def test_unusual_status_codes(self):
        """Test that exception handles unusual or non-standard HTTP status codes appropriately."""
        from smartsurge.exceptions import ServerError
        
        # Unusual server error code
        exc = ServerError("Server error", status_code=599)
        assert exc.status_code == 599
        
        # Non-server error code
        exc = ServerError("Server error", status_code=418)  # I'm a teapot
        assert exc.status_code == 418
    
    def test_truncates_long_response_text(self):
        """Test that exception truncates very long response text to the first 200 characters."""
        from smartsurge.exceptions import ServerError
        
        response = MagicMock()
        response.text = "a" * 1000  # 1000 character response
        
        exc = ServerError("Server error", response=response)
        
        assert len(exc.context["response_text"]) == 200
        assert exc.context["response_text"] == "a" * 200


class Test_ServerError_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ServerError exception."""
    
    def test_handles_response_without_text(self):
        """Test that exception gracefully handles response objects without a text attribute."""
        from smartsurge.exceptions import ServerError
        
        class ResponseWithoutText:
            # Response-like object without text attribute
            status_code = 500
        
        response = ResponseWithoutText()
        
        # Should not fail during initialization
        exc = ServerError("Server error", response=response)
        assert exc.response == response
        assert exc.context["response_text"] is None or exc.context["response_text"] == ""
    
    def test_handles_unexpected_response_structure(self):
        """Test that exception continues initialization when response object has unexpected structure."""
        from smartsurge.exceptions import ServerError
        
        class WeirdResponse:
            @property
            def text(self):
                raise AttributeError("Accessing text caused an error")
        
        response = WeirdResponse()
        
        # Should not fail during initialization
        with patch('logging.Logger.error'):  # Suppress logging
            exc = ServerError("Server error", response=response)
            assert exc.response == response


## ClientError Tests

class Test_ClientError_01_NominalBehaviors:
    """Tests for nominal behaviors of the ClientError exception."""
    
    def test_stores_attributes(self):
        """Test that exception properly stores endpoint, status_code, and response attributes."""
        from smartsurge.exceptions import ClientError
        
        endpoint = "https://api.example.com/resource"
        status_code = 404
        response = MagicMock()
        response.text = "Resource not found"
        
        exc = ClientError(
            "Not Found", 
            endpoint=endpoint, 
            status_code=status_code,
            response=response
        )
        
        assert exc.endpoint == endpoint
        assert exc.status_code == status_code
        assert exc.response == response
    
    def test_extracts_limited_response_text(self):
        """Test that exception extracts and stores limited response text (first 200 characters) when available."""
        from smartsurge.exceptions import ClientError
        
        response = MagicMock()
        response.text = "Error details: " + "x" * 300  # Response text longer than 200 chars
        
        exc = ClientError("Client error", response=response)
        
        assert exc.context["response_text"].startswith("Error details:")
        assert len(exc.context["response_text"]) == 200  # Truncated to 200 chars
    
    def test_includes_client_error_context_in_logs(self):
        """Test that exception includes client error-specific context in error logs."""
        from smartsurge.exceptions import ClientError
        
        with patch('logging.Logger.error') as mock_error:
            endpoint = "https://api.example.com/resource"
            status_code = 401
            
            ClientError(
                "Unauthorized", 
                endpoint=endpoint, 
                status_code=status_code
            )
            
            mock_error.assert_called_once()
            log_msg = mock_error.call_args[0][0]
            assert "Unauthorized" in log_msg
            assert f"'endpoint': '{endpoint}'" in log_msg
            assert f"'status_code': {status_code}" in log_msg
    
    def test_passes_context_to_parent(self):
        """Test that exception passes appropriate context to parent class."""
        from smartsurge.exceptions import ClientError, SmartSurgeException
        
        endpoint = "https://api.example.com/resource"
        status_code = 403
        response = MagicMock()
        response.text = "Forbidden"
        
        with patch.object(SmartSurgeException, '__init__') as mock_init:
            mock_init.return_value = None
            
            ClientError(
                "Forbidden", 
                endpoint=endpoint, 
                status_code=status_code,
                response=response
            )
            
            mock_init.assert_called_once()
            _, kwargs = mock_init.call_args
            assert kwargs["endpoint"] == endpoint
            assert kwargs["status_code"] == status_code
            assert kwargs["response_text"] == "Forbidden"
    
    def test_status_code_accessible(self):
        """Test that status_code is accessible to client code for error handling."""
        from smartsurge.exceptions import ClientError
        
        status_code = 400
        exc = ClientError("Bad Request", status_code=status_code)
        assert exc.status_code == status_code


class Test_ClientError_02_NegativeBehaviors:
    """Tests for negative behaviors of the ClientError exception."""
    
    def test_handles_none_or_empty_endpoint(self):
        """Test that exception handles None or empty endpoint value gracefully."""
        from smartsurge.exceptions import ClientError
        
        # Test with None endpoint
        exc = ClientError("Client error", endpoint=None)
        assert exc.endpoint is None
        
        # Test with empty endpoint
        exc = ClientError("Client error", endpoint="")
        assert exc.endpoint == ""
    
    def test_handles_none_status_code(self):
        """Test that exception functions correctly when status_code is None."""
        from smartsurge.exceptions import ClientError
        
        exc = ClientError("Client error", status_code=None)
        assert exc.status_code is None
    
    def test_handles_none_response(self):
        """Test that exception operates properly when response object is None."""
        from smartsurge.exceptions import ClientError
        
        exc = ClientError("Client error", response=None)
        assert exc.response is None
        assert exc.context["response_text"] is None


class Test_ClientError_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ClientError exception."""
    
    def test_long_endpoint(self):
        """Test that exception processes very long endpoint strings correctly."""
        from smartsurge.exceptions import ClientError
        
        long_endpoint = "https://api.example.com/" + "path/to/resource/" * 50
        exc = ClientError("Client error", endpoint=long_endpoint)
        assert exc.endpoint == long_endpoint
    
    def test_unusual_status_codes(self):
        """Test that exception handles unusual or non-standard HTTP status codes appropriately."""
        from smartsurge.exceptions import ClientError
        
        # Unusual client error code
        exc = ClientError("Client error", status_code=499)
        assert exc.status_code == 499
        
        # Non-client error code
        exc = ClientError("Client error", status_code=302)  # Found/Redirect
        assert exc.status_code == 302
    
    def test_truncates_long_response_text(self):
        """Test that exception truncates very long response text to the first 200 characters."""
        from smartsurge.exceptions import ClientError
        
        response = MagicMock()
        response.text = "a" * 1000  # 1000 character response
        
        exc = ClientError("Client error", response=response)
        
        assert len(exc.context["response_text"]) == 200
        assert exc.context["response_text"] == "a" * 200


class Test_ClientError_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ClientError exception."""
    
    def test_handles_response_without_text(self):
        """Test that exception gracefully handles response objects without a text attribute."""
        from smartsurge.exceptions import ClientError
        
        class ResponseWithoutText:
            # Response-like object without text attribute
            status_code = 404
        
        response = ResponseWithoutText()
        
        # Should not fail during initialization
        exc = ClientError("Client error", response=response)
        assert exc.response == response
        assert exc.context["response_text"] is None or exc.context["response_text"] == ""
    
    def test_handles_unexpected_response_structure(self):
        """Test that exception continues initialization when response object has unexpected structure."""
        from smartsurge.exceptions import ClientError
        
        class WeirdResponse:
            @property
            def text(self):
                raise AttributeError("Accessing text caused an error")
        
        response = WeirdResponse()
        
        # Should not fail during initialization
        with patch('logging.Logger.error'):  # Suppress logging
            exc = ClientError("Client error", response=response)
            assert exc.response == response
