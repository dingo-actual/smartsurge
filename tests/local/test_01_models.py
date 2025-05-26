import pytest
from unittest.mock import patch

from smartsurge.models import (
    RequestMethod, SearchStatus, RequestEntry, RateLimit
)


class Test_RequestMethod_01_NominalBehaviors:
    def test_enum_values_can_be_referenced(self):
        """Test that enum values can be correctly referenced by name."""
        assert RequestMethod.GET is RequestMethod.GET
        assert RequestMethod.POST is RequestMethod.POST
        assert RequestMethod.PUT is RequestMethod.PUT
        assert RequestMethod.DELETE is RequestMethod.DELETE
        assert RequestMethod.HEAD is RequestMethod.HEAD
        assert RequestMethod.OPTIONS is RequestMethod.OPTIONS
        assert RequestMethod.PATCH is RequestMethod.PATCH
    
    def test_enum_values_convert_to_strings(self):
        """Test that enum values correctly convert to string representations."""
        assert str(RequestMethod.GET) == "GET"
        assert str(RequestMethod.POST) == "POST"
        assert str(RequestMethod.PUT) == "PUT"
        assert str(RequestMethod.DELETE) == "DELETE"
        assert str(RequestMethod.HEAD) == "HEAD"
        assert str(RequestMethod.OPTIONS) == "OPTIONS"
        assert str(RequestMethod.PATCH) == "PATCH"
    
    def test_enum_values_can_be_compared(self):
        """Test comparison and matching operations with enum values."""
        assert RequestMethod.GET == RequestMethod.GET
        assert RequestMethod.GET != RequestMethod.POST
        assert RequestMethod.GET in [RequestMethod.GET, RequestMethod.POST]
        assert RequestMethod.GET.value == "GET"
class Test_RequestMethod_02_NegativeBehaviors:
    def test_invalid_enum_creation(self):
        """Test that creating an invalid enum value raises ValueError."""
        with pytest.raises(ValueError):
            RequestMethod("INVALID")
    
    def test_compare_with_non_enum_string(self):
        """Test behavior when comparing enum instances with string values."""
        assert RequestMethod.GET == "GET"  # Direct comparison should be equal
        assert "GET" == RequestMethod.GET  # Reverse comparison should be equal
        assert RequestMethod.GET.value == "GET"  # Value comparison should work
class Test_RequestMethod_04_ErrorHandlingBehaviors:
    def test_access_nonexistent_enum_value(self):
        """Test accessing non-existent enum values raises appropriate errors."""
        with pytest.raises(KeyError):
            RequestMethod["NONEXISTENT"]
        
        with pytest.raises(AttributeError):
            # pylint: disable=no-member
            RequestMethod.NONEXISTENT
class Test_SearchStatus_01_NominalBehaviors:
    def test_enum_values_can_be_accessed(self):
        """Test that enum values can be correctly accessed by name."""
        assert SearchStatus.NOT_STARTED is SearchStatus.NOT_STARTED
        assert SearchStatus.WAITING_TO_ESTIMATE is SearchStatus.WAITING_TO_ESTIMATE
        assert SearchStatus.COMPLETED is SearchStatus.COMPLETED
    
    def test_enum_values_convert_to_strings(self):
        """Test that enum values convert to expected string representations."""
        assert str(SearchStatus.NOT_STARTED) == "NOT_STARTED"
        assert str(SearchStatus.WAITING_TO_ESTIMATE) == "WAITING_TO_ESTIMATE"
        assert str(SearchStatus.COMPLETED) == "COMPLETED"
    
    def test_enum_values_can_be_compared(self):
        """Test state comparison operations using enum values."""
        assert SearchStatus.NOT_STARTED == SearchStatus.NOT_STARTED
        assert SearchStatus.NOT_STARTED != SearchStatus.COMPLETED
        
        # Collection membership
        assert SearchStatus.WAITING_TO_ESTIMATE in [
            SearchStatus.NOT_STARTED, 
            SearchStatus.WAITING_TO_ESTIMATE, 
            SearchStatus.COMPLETED
        ]
        
        # Value comparison
        assert SearchStatus.NOT_STARTED.value == "NOT_STARTED"

class Test_SearchStatus_02_NegativeBehaviors:
    def test_invalid_enum_creation(self):
        """Test that creating an invalid enum value raises ValueError."""
        with pytest.raises(ValueError):
            SearchStatus("invalid_status")

class Test_RequestEntry_ValidateSuccessCodeConsistency_01_NominalBehaviors:
    def test_returns_self_when_success_matches_status(self):
        """Test that method returns self when success flag matches status code."""
        # Success with 200 status code
        entry = RequestEntry(
            endpoint="/test",
            method=RequestMethod.GET,
            status_code=200,
            response_time=0.1,
            success=True
        )
        
        result = entry.validate_success_code_consistency()
        assert result is entry
        
        # Failed with 404 status code
        entry = RequestEntry(
            endpoint="/test",
            method=RequestMethod.GET,
            status_code=404,
            response_time=0.1,
            success=False
        )
        
        result = entry.validate_success_code_consistency()
        assert result is entry
    
    def test_extracts_rate_limit_from_x_ratelimit_header(self):
        """Test extraction of rate limit info from X-RateLimit-Limit header."""
        entry = RequestEntry(
            endpoint="/test",
            method=RequestMethod.GET,
            status_code=200,
            response_time=0.1,
            success=True,
            response_headers={'X-RateLimit-Limit': '100'}
        )
        
        entry.validate_success_code_consistency()
        
        assert entry.max_requests == 100
        assert entry.max_request_period == 60.0  # Default period of 60s
    
    def test_extracts_rate_limit_from_ratelimit_header(self):
        """Test extraction of rate limit info from RateLimit-Limit header."""
        entry = RequestEntry(
            endpoint="/test",
            method=RequestMethod.GET,
            status_code=200,
            response_time=0.1,
            success=True,
            response_headers={'RateLimit-Limit': '150'}
        )
        
        entry.validate_success_code_consistency()
        
        assert entry.max_requests == 150
        assert entry.max_request_period == 60.0  # Default period of 60s
    
    def test_default_period_is_set_with_rate_limit_headers(self):
        """Verify default period of 60s is set when rate limit headers are found."""
        entry = RequestEntry(
            endpoint="/test",
            method=RequestMethod.GET,
            status_code=200,
            response_time=0.1,
            success=True,
            response_headers={'X-RateLimit-Limit': '200'}
        )
        
        entry.validate_success_code_consistency()
        
        assert entry.max_request_period == 60.0

class Test_RequestEntry_ValidateSuccessCodeConsistency_02_NegativeBehaviors:
    @patch('smartsurge.models.logger')
    def test_warning_logged_for_inconsistent_success_flag(self, mock_logger):
        """Test warning is logged when success=True with status code ≥ 400."""
        entry = RequestEntry(
            endpoint="/test",
            method=RequestMethod.GET,
            status_code=404,
            response_time=0.1,
            success=True  # Inconsistent with 404
        )
        
        # Verify warning was logged
        mock_logger.warning.assert_called_once()
        assert "Inconsistent success flag" in mock_logger.warning.call_args[0][0]
    
    @patch('smartsurge.models.logger')
    def test_max_requests_nullified_when_period_missing(self, mock_logger):
        """Verify max_requests is set to None when max_request_period is None."""
        entry = RequestEntry(
            endpoint="/test",
            method=RequestMethod.GET,
            status_code=200,
            response_time=0.1,
            success=True,
            max_requests=100,  # Set max_requests but not max_request_period
            max_request_period=None
        )
        
        # max_requests should be nullified
        assert entry.max_requests is None
        
        # Verify warning was logged
        mock_logger.warning.assert_called_once()
        assert "max_requests provided without max_request_period" in mock_logger.warning.call_args[0][0]
    
    def test_handles_invalid_values_in_rate_limit_headers(self):
        """Test handling of invalid (non-integer) values in rate limit headers."""
        entry = RequestEntry(
            endpoint="/test",
            method=RequestMethod.GET,
            status_code=200,
            response_time=0.1,
            success=True,
            response_headers={'X-RateLimit-Limit': 'not-a-number'}
        )
        
        # Should not raise an exception
        entry.validate_success_code_consistency()
        
        # max_requests should remain None
        assert entry.max_requests is None

class Test_RequestEntry_ValidateSuccessCodeConsistency_03_BoundaryBehaviors:
    @patch('smartsurge.models.logger')
    def test_boundary_status_codes(self, mock_logger):
        """Test behavior with status codes at boundary (399 vs 400)."""
        # Status code 399 (barely valid success)
        entry_valid = RequestEntry(
            endpoint="/test",
            method=RequestMethod.GET,
            status_code=399,
            response_time=0.1,
            success=True
        )
        
        # Should not log a warning
        mock_logger.warning.assert_not_called()
        
        # Reset mock
        mock_logger.reset_mock()
        
        # Status code 400 (barely invalid success)
        entry_invalid = RequestEntry(
            endpoint="/test",
            method=RequestMethod.GET,
            status_code=400,
            response_time=0.1,
            success=True
        )
        
        # Should log a warning
        mock_logger.warning.assert_called_once()
        assert "Inconsistent success flag" in mock_logger.warning.call_args[0][0]
    
    def test_case_insensitive_header_matching(self):
        """Verify case-insensitive matching of header names."""
        # Headers with mixed casing
        entry = RequestEntry(
            endpoint="/test",
            method=RequestMethod.GET,
            status_code=200,
            response_time=0.1,
            success=True,
            response_headers={'x-RATELIMIT-limit': '100'}  # Mixed case
        )
        
        entry.validate_success_code_consistency()
        
        # Should extract rate limit despite case differences
        assert entry.max_requests == 100
        assert entry.max_request_period == 60.0

class Test_RequestEntry_ValidateSuccessCodeConsistency_04_ErrorHandlingBehaviors:
    @patch('smartsurge.models.logger.warning')
    def test_handles_exceptions_when_parsing_headers(self, mock_logger):
        """Test method handles exceptions when parsing rate limit values."""
        # Create a header value that will cause an exception when parsed
        entry = RequestEntry(
            endpoint="/test",
            method=RequestMethod.GET,
            status_code=200,
            response_time=0.1,
            success=True,
            response_headers={
                'X-RateLimit-Limit': 'not-a-number',  # This will cause ValueError when parsed as int
            }
        )
        
        # Should not raise an exception
        result = entry.validate_success_code_consistency()
        
        # Verify the method continued execution despite parsing error
        assert result is entry
        assert entry.max_requests is None  # Should not have set max_requests due to parsing error
    
    def test_validation_continues_after_header_parsing_fails(self):
        """Test validation continues even when header parsing fails."""
        # Create a header with values that will cause errors
        entry = RequestEntry(
            endpoint="/test",
            method=RequestMethod.GET,
            status_code=200,
            response_time=0.1,
            success=True,
            max_requests=None,  # Should remain None after validation
            response_headers={
                'X-RateLimit-Limit': None,  # Will cause error in int conversion
                'RateLimit-Limit': {},  # Invalid type for string conversion
            }
        )
        
        # Should complete without exceptions
        result = entry.validate_success_code_consistency()
        
        # Validation should complete and return the entry
        assert result is entry
        assert entry.max_requests is None  # Should still be None

class Test_RequestEntry_ValidateSuccessCodeConsistency_05_StateTransitionBehaviors:
    def test_transition_from_no_rate_limit_to_header_extraction(self):
        """Test transition from having no rate limit info to extracting rate limit from headers."""
        # Create an entry with no initial rate limit info but with headers
        entry = RequestEntry(
            endpoint="/test",
            method=RequestMethod.GET,
            status_code=200,
            response_time=0.1,
            success=True,
            max_requests=None,  # No initial rate limit
            response_headers={'X-RateLimit-Limit': '100'}
        )
        
        # Should have extracted rate limit info
        assert entry.max_requests == 100
        assert entry.max_request_period == 60.0
    
    def test_max_requests_nullified_when_period_missing(self):
        """Verify that max_requests gets nullified when max_request_period is missing."""
        # Create an entry with values that should trigger the validation logic
        entry = RequestEntry(
            endpoint="/test",
            method=RequestMethod.GET,
            status_code=200,
            response_time=0.1,
            success=True,
            max_requests=50,  # This will be automatically nullified during initialization
            max_request_period=None
        )
        
        # After initialization, the validator should have nullified max_requests
        assert entry.max_requests is None

class Test_RateLimit_Str_01_NominalBehaviors:
    def test_string_includes_max_requests_and_time_period(self):
        """Verify string contains max_requests and time_period correctly formatted."""
        rate_limit = RateLimit(
            endpoint="/test",
            method=RequestMethod.GET,
            max_requests=100,
            time_period=60.0
        )
        
        result = str(rate_limit)
        
        assert "100 requests per 60.00s" in result
    
    def test_string_includes_source_information(self):
        """Confirm source information is included in string representation."""
        rate_limit = RateLimit(
            endpoint="/test",
            method=RequestMethod.GET,
            max_requests=100,
            time_period=60.0,
            source="headers"
        )
        
        result = str(rate_limit)
        
        assert "source: headers" in result
    
    def test_string_formatting_with_different_sources(self):
        """Test string formatting with different source values."""
        sources = ["estimated", "headers", "manual"]
        
        for source in sources:
            rate_limit = RateLimit(
                endpoint="/test",
                method=RequestMethod.GET,
                max_requests=100,
                time_period=60.0,
                source=source
            )
            
            result = str(rate_limit)
            
            assert f"source: {source}" in result

class Test_RateLimit_Str_02_NegativeBehaviors:
    def test_cooldown_info_omitted_when_none(self):
        """Verify cooldown info is omitted when cooldown is None."""
        rate_limit = RateLimit(
            endpoint="/test",
            method=RequestMethod.GET,
            max_requests=100,
            time_period=60.0,
            cooldown=None
        )
        
        result = str(rate_limit)
        
        assert "cooldown" not in result

class Test_RateLimit_Str_03_BoundaryBehaviors:
    def test_formatting_with_small_time_period(self):
        """Test formatting with very small time_period values."""
        rate_limit = RateLimit(
            endpoint="/test",
            method=RequestMethod.GET,
            max_requests=5,
            time_period=0.001  # Very small time period
        )
        
        result = str(rate_limit)
        
        assert "5 requests per 0.00s" in result or "5 requests per 0.001s" in result
    
    def test_formatting_with_large_max_requests(self):
        """Test formatting with very large max_requests values."""
        rate_limit = RateLimit(
            endpoint="/test",
            method=RequestMethod.GET,
            max_requests=1000000,  # Very large max_requests
            time_period=60.0
        )
        
        result = str(rate_limit)
        
        assert "1000000 requests per 60.00s" in result

class Test_RateLimit_GetRequestsPerSecond_01_NominalBehaviors:
    def test_correct_calculation(self):
        """Verify correct calculation of requests/second (max_requests / time_period)."""
        rate_limit = RateLimit(
            endpoint="/test",
            method=RequestMethod.GET,
            max_requests=120,
            time_period=60.0
        )
        
        result = rate_limit.get_requests_per_second()
        
        assert result == 2.0  # 120 requests / 60 seconds = 2 req/s
    
    def test_different_combinations(self):
        """Test with different combinations of max_requests and time_period values."""
        test_cases = [
            (100, 10.0, 10.0),  # 100 requests / 10 seconds = 10 req/s
            (5, 2.5, 2.0),      # 5 requests / 2.5 seconds = 2 req/s
            (1, 1.0, 1.0),      # 1 request / 1 second = 1 req/s
            (30, 15.0, 2.0)     # 30 requests / 15 seconds = 2 req/s
        ]
        
        for max_requests, time_period, expected in test_cases:
            rate_limit = RateLimit(
                endpoint="/test",
                method=RequestMethod.GET,
                max_requests=max_requests,
                time_period=time_period
            )
            
            result = rate_limit.get_requests_per_second()
            
            assert result == expected

class Test_RateLimit_GetRequestsPerSecond_02_NegativeBehaviors:
    def test_returns_zero_for_zero_time_period(self):
        """Confirm method returns 0 when time_period is 0 (preventing division by zero)."""
        # This is an edge case - Pydantic should prevent time_period=0 due to the gt=0.0 constraint,
        # but we'll test the method's behavior if it somehow happens
        rate_limit = RateLimit(
            endpoint="/test",
            method=RequestMethod.GET,
            max_requests=100,
            time_period=1.0  # Temporary value to bypass Pydantic validation
        )
        
        # Monkeypatch the time_period to bypass validation
        rate_limit.time_period = 0.0
        
        result = rate_limit.get_requests_per_second()
        
        assert result == 0

class Test_RateLimit_GetRequestsPerSecond_03_BoundaryBehaviors:
    def test_small_time_period_values(self):
        """Test with very small time_period values approaching zero."""
        rate_limit = RateLimit(
            endpoint="/test",
            method=RequestMethod.GET,
            max_requests=1,
            time_period=0.001  # Very small time period
        )
        
        result = rate_limit.get_requests_per_second()
        
        assert result == 1000.0  # 1 request / 0.001 seconds = 1000 req/s
    
    def test_large_max_requests_values(self):
        """Test with very large max_requests values."""
        rate_limit = RateLimit(
            endpoint="/test",
            method=RequestMethod.GET,
            max_requests=1000000,  # Very large max_requests
            time_period=1.0
        )
        
        result = rate_limit.get_requests_per_second()
        
        assert result == 1000000.0  # 1,000,000 requests / 1 second = 1,000,000 req/s
