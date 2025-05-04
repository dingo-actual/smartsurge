import logging
import pytest
from datetime import datetime, timezone
from enum import Enum

from smartsurge.models import (SearchStatus, RequestEntry, RateLimit, 
                               RequestMethod, RequestHistory)
from smartsurge.exceptions import ValidationError

        
class TestSearchStatus:

    # Enum members can be accessed as attributes (SearchStatus.NOT_STARTED)
    def test_enum_members_accessible_as_attributes(self):
        # Arrange
        # No arrangement needed
    
        # Act
        not_started = SearchStatus.NOT_STARTED
        waiting = SearchStatus.WAITING_FOR_FIRST_REFUSAL
        estimating = SearchStatus.ESTIMATING
        completed = SearchStatus.COMPLETED
    
        # Assert
        assert not_started == "not_started"
        assert waiting == "waiting_for_first_refusal"
        assert estimating == "estimating"
        assert completed == "completed"

    # Enum values can be compared with string literals
    def test_enum_values_compare_with_string_literals(self):
        # Arrange
        # No arrangement needed
    
        # Act & Assert
        assert "not_started" == SearchStatus.NOT_STARTED
        assert "waiting_for_first_refusal" == SearchStatus.WAITING_FOR_FIRST_REFUSAL
        assert "estimating" == SearchStatus.ESTIMATING
        assert "completed" == SearchStatus.COMPLETED

    # Enum can be converted to string representation
    def test_enum_string_conversion(self):
        # Arrange
        # No arrangement needed
    
        # Act
        not_started_str = str(SearchStatus.NOT_STARTED)
        waiting_str = str(SearchStatus.WAITING_FOR_FIRST_REFUSAL)
        estimating_str = str(SearchStatus.ESTIMATING)
        completed_str = str(SearchStatus.COMPLETED)
    
        # Assert
        assert not_started_str == "not_started"
        assert waiting_str == "waiting_for_first_refusal"
        assert estimating_str == "estimating"
        assert completed_str == "completed"

    # Enum members can be iterated over
    def test_enum_members_iteration(self):
        # Arrange
        expected_values = ["not_started", "waiting_for_first_refusal", "estimating", "completed"]
    
        # Act
        actual_values = [status for status in SearchStatus]
    
        # Assert
        assert len(actual_values) == 4
        for status, expected in zip(actual_values, expected_values):
            assert status == expected

    # Enum members maintain insertion order
    def test_enum_members_maintain_insertion_order(self):
        # Arrange
        expected_order = [
            SearchStatus.NOT_STARTED,
            SearchStatus.WAITING_FOR_FIRST_REFUSAL,
            SearchStatus.ESTIMATING,
            SearchStatus.COMPLETED
        ]
    
        # Act
        actual_order = list(SearchStatus)
    
        # Assert
        assert actual_order == expected_order

    # Case sensitivity in enum values
    def test_enum_case_sensitivity(self):
        # Arrange
        # No arrangement needed
    
        # Act & Assert
        assert SearchStatus.NOT_STARTED != "NOT_STARTED"
        assert SearchStatus.WAITING_FOR_FIRST_REFUSAL != "WAITING_FOR_FIRST_REFUSAL"
        assert SearchStatus.ESTIMATING != "ESTIMATING"
        assert SearchStatus.COMPLETED != "COMPLETED"

    # Attempting to create new enum members at runtime
    def test_cannot_create_new_enum_members(self):
        # Arrange
        # No arrangement needed
    
        # Act & Assert
        with pytest.raises(AttributeError):
            SearchStatus.NEW_STATUS = "new_status"

    # Comparing with non-string values
    def test_comparing_with_non_string_values(self):
        # Arrange
        # No arrangement needed
    
        # Act & Assert
        assert SearchStatus.NOT_STARTED != 123
        assert SearchStatus.NOT_STARTED != None
        assert SearchStatus.NOT_STARTED != True
        assert SearchStatus.NOT_STARTED != ["not_started"]

    # Serialization and deserialization of enum values
    def test_serialization_deserialization(self):
        # Arrange
        import json
    
        # Act - Serialization
        serialized = json.dumps(SearchStatus.ESTIMATING)
    
        # Assert - Serialization
        assert serialized == '"estimating"'
    
        # Act - Deserialization
        deserialized_value = json.loads(serialized)
    
        # Assert - Deserialization
        assert deserialized_value == "estimating"
        assert SearchStatus(deserialized_value) == SearchStatus.ESTIMATING

    # Using enum values in switch/match statements
    def test_enum_in_match_statement(self):
        # Arrange
        status = SearchStatus.ESTIMATING
        result = ""
    
        # Act
        match status:
            case SearchStatus.NOT_STARTED:
                result = "not started"
            case SearchStatus.WAITING_FOR_FIRST_REFUSAL:
                result = "waiting"
            case SearchStatus.ESTIMATING:
                result = "estimating now"
            case SearchStatus.COMPLETED:
                result = "done"
            case _:
                result = "unknown"
    
        # Assert
        assert result == "estimating now"

    # Enum members are hashable and can be used as dictionary keys
    def test_enum_members_as_dictionary_keys(self):
        # Arrange
        status_descriptions = {
            SearchStatus.NOT_STARTED: "Search has not started yet",
            SearchStatus.WAITING_FOR_FIRST_REFUSAL: "Waiting for first refusal",
            SearchStatus.ESTIMATING: "Currently estimating",
            SearchStatus.COMPLETED: "Search completed"
        }
    
        # Act
        not_started_desc = status_descriptions[SearchStatus.NOT_STARTED]
        completed_desc = status_descriptions[SearchStatus.COMPLETED]
    
        # Assert
        assert not_started_desc == "Search has not started yet"
        assert completed_desc == "Search completed"
        assert len(status_descriptions) == 4

    # Enum members are singleton instances
    def test_enum_members_are_singletons(self):
        # Arrange
        status1 = SearchStatus.NOT_STARTED
        status2 = SearchStatus.NOT_STARTED
    
        # Act & Assert
        assert status1 is status2
        assert id(status1) == id(status2)
    
        # Also check with constructor
        status3 = SearchStatus("not_started")
        assert status1 is status3
        assert id(status1) == id(status3)

class TestRequestEntry:

    def test_endpoint_minimum_length(self):
        with pytest.raises(ValidationError):
            RequestEntry(endpoint="", method="GET", timestamp=datetime.now(timezone.utc), 
                        status_code=200, response_time=0.5, success=True)

    def test_endpoint_valid_string(self):
        endpoint = "/api/v1/test"
        entry = RequestEntry(endpoint=endpoint, method="GET", timestamp=datetime.now(timezone.utc),
                           status_code=200, response_time=0.5, success=True)
        assert entry.endpoint == endpoint

    def test_endpoint_with_query_parameters(self):
        endpoint = "/api/v1/test?param1=value1&param2=value2"
        entry = RequestEntry(endpoint=endpoint, method="GET", timestamp=datetime.now(timezone.utc),
                           status_code=200, response_time=0.5, success=True)
        assert entry.endpoint == endpoint

    def test_endpoint_with_special_characters(self):
        endpoint = "/api/v1/test-endpoint/with_underscore/and-dash"
        entry = RequestEntry(endpoint=endpoint, method="GET", timestamp=datetime.now(timezone.utc),
                           status_code=200, response_time=0.5, success=True)
        assert entry.endpoint == endpoint

    def test_endpoint_with_unicode_characters(self):
        endpoint = "/api/v1/test/ünicode/páth"
        entry = RequestEntry(endpoint=endpoint, method="GET", timestamp=datetime.now(timezone.utc),
                           status_code=200, response_time=0.5, success=True)
        assert entry.endpoint == endpoint

    def test_endpoint_maximum_length(self):
        long_endpoint = "/api/" + "x" * 2048
        entry = RequestEntry(endpoint=long_endpoint, method="GET", timestamp=datetime.now(timezone.utc),
                           status_code=200, response_time=0.5, success=True)
        assert entry.endpoint == long_endpoint

    def test_endpoint_with_encoded_characters(self):
        endpoint = "/api/v1/test%20with%20spaces"
        entry = RequestEntry(endpoint=endpoint, method="GET", timestamp=datetime.now(timezone.utc),
                           status_code=200, response_time=0.5, success=True)
        assert entry.endpoint == endpoint

    def test_endpoint_with_multiple_slashes(self):
        endpoint = "/api//v1///test"
        entry = RequestEntry(endpoint=endpoint, method="GET", timestamp=datetime.now(timezone.utc),
                           status_code=200, response_time=0.5, success=True)
        assert entry.endpoint == endpoint


        assert entry.endpoint == endpoint

class TestRateLimit:

    # Creating a RateLimit instance with valid required parameters
    def test_create_rate_limit_with_valid_required_parameters(self):
        # Arrange
        endpoint = "/api/v1/users"
        method = RequestMethod.GET
        max_requests = 100
        time_period = 60.0
    
        # Act
        rate_limit = RateLimit(
            endpoint=endpoint,
            method=method,
            max_requests=max_requests,
            time_period=time_period
        )
    
        # Assert
        assert rate_limit.endpoint == endpoint
        assert rate_limit.method == method
        assert rate_limit.max_requests == max_requests
        assert rate_limit.time_period == time_period
        assert rate_limit.confidence == 0.0
        assert isinstance(rate_limit.last_updated, datetime)

    # Verifying the string representation contains max_requests, time_period and confidence
    def test_string_representation_contains_required_info(self):
        # Arrange
        rate_limit = RateLimit(
            endpoint="/api/v1/users",
            method=RequestMethod.GET,
            max_requests=100,
            time_period=60.0,
            confidence=0.95
        )
    
        # Act
        string_repr = str(rate_limit)
    
        # Assert
        assert "100 requests" in string_repr
        assert "60.00s" in string_repr
        assert "confidence: 0.95" in string_repr

    # Creating a RateLimit with default values for optional parameters
    def test_create_rate_limit_with_default_optional_parameters(self):
        # Arrange & Act
        rate_limit = RateLimit(
            endpoint="/api/v1/users",
            method=RequestMethod.GET,
            max_requests=100,
            time_period=60.0
        )
    
        # Assert
        assert rate_limit.confidence == 0.0
        assert rate_limit.time_period_lower is None
        assert rate_limit.time_period_upper is None
        assert rate_limit.max_requests_lower is None
        assert rate_limit.max_requests_upper is None
        assert isinstance(rate_limit.last_updated, datetime)
        assert rate_limit.last_updated.tzinfo == timezone.utc

    # Creating a RateLimit with all parameters specified
    def test_create_rate_limit_with_all_parameters(self):
        # Arrange
        now = datetime.now(timezone.utc)
    
        # Act
        rate_limit = RateLimit(
            endpoint="/api/v1/users",
            method=RequestMethod.POST,
            max_requests=100,
            time_period=60.0,
            confidence=0.95,
            last_updated=now,
            time_period_lower=55.0,
            time_period_upper=65.0,
            max_requests_lower=90,
            max_requests_upper=110
        )
    
        # Assert
        assert rate_limit.endpoint == "/api/v1/users"
        assert rate_limit.method == RequestMethod.POST
        assert rate_limit.max_requests == 100
        assert rate_limit.time_period == 60.0
        assert rate_limit.confidence == 0.95
        assert rate_limit.last_updated == now
        assert rate_limit.time_period_lower == 55.0
        assert rate_limit.time_period_upper == 65.0
        assert rate_limit.max_requests_lower == 90
        assert rate_limit.max_requests_upper == 110

    # Setting the confidence level to different values between 0.0 and 1.0
    def test_set_different_confidence_levels(self):
        # Arrange
        confidence_levels = [0.0, 0.25, 0.5, 0.75, 0.99]
    
        # Act & Assert
        for confidence in confidence_levels:
            rate_limit = RateLimit(
                endpoint="/api/v1/users",
                method=RequestMethod.GET,
                max_requests=100,
                time_period=60.0,
                confidence=confidence
            )
            assert rate_limit.confidence == confidence

    # Creating a RateLimit with minimum valid values (max_requests=1, time_period slightly above 0)
    def test_create_rate_limit_with_minimum_valid_values(self):
        # Arrange
        min_max_requests = 1
        min_time_period = 0.000001
    
        # Act
        rate_limit = RateLimit(
            endpoint="/api/v1/users",
            method=RequestMethod.GET,
            max_requests=min_max_requests,
            time_period=min_time_period
        )
    
        # Assert
        assert rate_limit.max_requests == min_max_requests
        assert rate_limit.time_period == min_time_period

    # Creating a RateLimit with empty endpoint string
    def test_create_rate_limit_with_empty_endpoint(self):
        # Arrange & Act & Assert
        from pydantic import ValidationError
        import pytest
    
        with pytest.raises(ValidationError) as exc_info:
            RateLimit(
                endpoint="",
                method=RequestMethod.GET,
                max_requests=100,
                time_period=60.0
            )
    
        # Check that the error message mentions the endpoint field
        assert "endpoint" in str(exc_info.value)
        assert "min_length" in str(exc_info.value)

    # Creating a RateLimit with invalid method not in RequestMethod enum
    def test_create_rate_limit_with_invalid_method(self):
        # Arrange & Act & Assert
        from pydantic import ValidationError
        import pytest
    
        with pytest.raises(ValidationError) as exc_info:
            RateLimit(
                endpoint="/api/v1/users",
                method="INVALID_METHOD",
                max_requests=100,
                time_period=60.0
            )
    
        # Check that the error message mentions the method field
        assert "method" in str(exc_info.value)

    # Creating a RateLimit with negative max_requests
    def test_create_rate_limit_with_negative_max_requests(self):
        # Arrange & Act & Assert
        from pydantic import ValidationError
        import pytest
    
        with pytest.raises(ValidationError) as exc_info:
            RateLimit(
                endpoint="/api/v1/users",
                method=RequestMethod.GET,
                max_requests=-10,
                time_period=60.0
            )
    
        # Check that the error message mentions the max_requests field
        assert "max_requests" in str(exc_info.value)
        assert "ge=" in str(exc_info.value)

    # Creating a RateLimit with zero or negative time_period
    def test_create_rate_limit_with_invalid_time_period(self):
        # Arrange & Act & Assert
        from pydantic import ValidationError
        import pytest
    
        # Test with zero time_period
        with pytest.raises(ValidationError) as exc_info:
            RateLimit(
                endpoint="/api/v1/users",
                method=RequestMethod.GET,
                max_requests=100,
                time_period=0.0
            )
        assert "time_period" in str(exc_info.value)
        assert "gt=" in str(exc_info.value)
    
        # Test with negative time_period
        with pytest.raises(ValidationError) as exc_info:
            RateLimit(
                endpoint="/api/v1/users",
                method=RequestMethod.GET,
                max_requests=100,
                time_period=-10.0
            )
        assert "time_period" in str(exc_info.value)

    # Creating a RateLimit with confidence level outside 0.0-1.0 range
    def test_create_rate_limit_with_invalid_confidence(self):
        # Arrange & Act & Assert
        from pydantic import ValidationError
        import pytest
    
        # Test with confidence below 0.0
        with pytest.raises(ValidationError) as exc_info:
            RateLimit(
                endpoint="/api/v1/users",
                method=RequestMethod.GET,
                max_requests=100,
                time_period=60.0,
                confidence=-0.1
            )
        assert "confidence" in str(exc_info.value)
    
        # Test with confidence above 1.0
        with pytest.raises(ValidationError) as exc_info:
            RateLimit(
                endpoint="/api/v1/users",
                method=RequestMethod.GET,
                max_requests=100,
                time_period=60.0,
                confidence=1.1
            )
        assert "confidence" in str(exc_info.value)

    # Setting credible interval bounds that are inconsistent with main values
    def test_create_rate_limit_with_inconsistent_credible_intervals(self):
        # Arrange & Act
        rate_limit = RateLimit(
            endpoint="/api/v1/users",
            method=RequestMethod.GET,
            max_requests=100,
            time_period=60.0,
            time_period_lower=70.0,  # Higher than time_period
            time_period_upper=50.0,  # Lower than time_period
            max_requests_lower=110,  # Higher than max_requests
            max_requests_upper=90    # Lower than max_requests
        )
    
        # Assert - Pydantic doesn't validate these relationships by default
        # so we're just checking the values are stored as provided
        assert rate_limit.time_period == 60.0
        assert rate_limit.time_period_lower == 70.0
        assert rate_limit.time_period_upper == 50.0
        assert rate_limit.max_requests == 100
        assert rate_limit.max_requests_lower == 110
        assert rate_limit.max_requests_upper == 90

class TestRequestHistory:

    # Adding a request entry to the history and validating endpoint/method match
    def test_add_request_with_matching_endpoint_and_method(self):
        # Arrange
        history = RequestHistory(endpoint="/api/users", method=RequestMethod.GET)
        entry = RequestEntry(
            endpoint="/api/users",
            method=RequestMethod.GET,
            status_code=200,
            response_time=0.5,
            success=True
        )
    
        # Act
        history.add_request(entry)
    
        # Assert
        assert len(history.entries) == 1
        assert history.entries[0] == entry

    # Validating endpoint and method consistency between RequestHistory and RequestEntry
    def test_add_request_with_mismatched_endpoint_raises_error(self):
        # Arrange
        history = RequestHistory(endpoint="/api/users", method=RequestMethod.GET)
        entry = RequestEntry(
            endpoint="/api/products",  # Different endpoint
            method=RequestMethod.GET,
            status_code=200,
            response_time=0.5,
            success=True
        )
    
        # Act & Assert
        with pytest.raises(ValueError) as excinfo:
            history.add_request(entry)
    
        assert "Entry endpoint and method must match RequestHistory" in str(excinfo.value)

    # Validating endpoint and method consistency between RequestHistory and RequestEntry
    def test_add_request_with_mismatched_method_raises_error(self):
        # Arrange
        history = RequestHistory(endpoint="/api/users", method=RequestMethod.GET)
        entry = RequestEntry(
            endpoint="/api/users",
            method=RequestMethod.POST,  # Different method
            status_code=200,
            response_time=0.5,
            success=True
        )
    
        # Act & Assert
        with pytest.raises(ValueError) as excinfo:
            history.add_request(entry)
    
        assert "Entry endpoint and method must match RequestHistory" in str(excinfo.value)

    # Merging two RequestHistory objects with the same endpoint and method
    def test_merge_with_incompatible_history_raises_error(self):
        # Arrange
        history1 = RequestHistory(endpoint="/api/users", method=RequestMethod.GET)
        history2 = RequestHistory(endpoint="/api/products", method=RequestMethod.GET)  # Different endpoint
    
        # Act & Assert
        with pytest.raises(ValueError) as excinfo:
            history1.merge(history2)
    
        assert "Can only merge RequestHistory objects with the same endpoint and method" in str(excinfo.value)

    # Handling merges with different search statuses and Bayesian parameters
    def test_merge_preserves_more_advanced_search_status(self):
        # Arrange
        history1 = RequestHistory(endpoint="/api/users", method=RequestMethod.GET)
        history2 = RequestHistory(endpoint="/api/users", method=RequestMethod.GET)
    
        history1.search_status = SearchStatus.WAITING_FOR_FIRST_REFUSAL
        history2.search_status = SearchStatus.ESTIMATING
    
        # Act
        history1.merge(history2)
    
        # Assert
        assert history1.search_status == SearchStatus.ESTIMATING

    # Intercepting a request and enforcing rate limits based on search status
    def test_intercept_request_with_not_started_status(self):
        # Arrange
        history = RequestHistory(endpoint="/api/users", method=RequestMethod.GET)
        assert history.search_status == SearchStatus.NOT_STARTED
    
        # Act
        history.intercept_request()
    
        # Assert
        assert history.search_status == SearchStatus.WAITING_FOR_FIRST_REFUSAL

    # Logging responses and updating search status based on response codes
    def test_log_response_updates_search_status_on_first_refusal(self):
        # Arrange
        history = RequestHistory(endpoint="/api/users", method=RequestMethod.GET)
        history.search_status = SearchStatus.WAITING_FOR_FIRST_REFUSAL
    
        entry = RequestEntry(
            endpoint="/api/users",
            method=RequestMethod.GET,
            status_code=429,  # Rate limit exceeded
            response_time=0.5,
            success=False
        )
    
        # Act
        history.log_response_and_update(entry)
    
        # Assert
        assert history.search_status == SearchStatus.ESTIMATING
        assert len(history.entries) == 1

    # Handling rate limit errors (429) after search is completed by restarting estimation
    def test_log_response_restarts_estimation_on_429_after_completion(self):
        # Arrange
        history = RequestHistory(endpoint="/api/users", method=RequestMethod.GET)
        history.search_status = SearchStatus.COMPLETED
        history.restart_count = 0
    
        entry = RequestEntry(
            endpoint="/api/users",
            method=RequestMethod.GET,
            status_code=429,  # Rate limit exceeded
            response_time=0.5,
            success=False
        )
    
        # Act
        history.log_response_and_update(entry)
    
        # Assert
        assert history.search_status == SearchStatus.ESTIMATING
        assert history.restart_count == 1

    # Estimating rate limits using Bayesian statistical models
    def test_update_bayesian_estimates_updates_rate_limit(self):
        # Arrange
        history = RequestHistory(endpoint="/api/users", method=RequestMethod.GET)
    
        # Add two entries to have enough data for Bayesian update
        entry1 = RequestEntry(
            endpoint="/api/users",
            method=RequestMethod.GET,
            status_code=200,
            response_time=0.5,
            success=True,
            timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc)
        )
    
        entry2 = RequestEntry(
            endpoint="/api/users",
            method=RequestMethod.GET,
            status_code=429,
            response_time=0.3,
            success=False,
            timestamp=datetime(2023, 1, 1, 1, tzinfo=timezone.utc)
        )
    
        history.add_request(entry1)
        history.add_request(entry2)
    
        # Initialize Bayesian priors
        history._initialize_bayesian_priors()
        old_rate_limit = history.rate_limit
    
        # Act
        history._update_bayesian_estimates(True)
    
        # Assert
        assert history.rate_limit is not None
        assert history.rate_limit != old_rate_limit
        assert history.rate_limit.max_requests > 0
        assert history.rate_limit.time_period > 0
        assert 0 <= history.rate_limit.confidence <= 0.99
    # Handling insufficient data for Bayesian updates with appropriate fallbacks
    def test_update_bayesian_estimates_with_insufficient_data(self):
        # Arrange
        history = RequestHistory(endpoint="/api/users", method=RequestMethod.GET)
        # No entries added, insufficient data
    
        # Act
        history._update_bayesian_estimates(False)
    
        # Assert
        # Should not raise an exception and should handle the insufficient data gracefully
        assert history.rate_limit is None

    # Adapting confidence thresholds based on observation count and restart history
    def test_check_estimation_confidence_adapts_to_observation_count(self):
        # Arrange
        history = RequestHistory(endpoint="/api/users", method=RequestMethod.GET)
    
        # Simulate having collected observations
        history.poisson_alpha = 25  # High observation count
    
        # Create a rate limit with high confidence
        history.rate_limit = RateLimit(
            endpoint="/api/users",
            method=RequestMethod.GET,
            max_requests=100,
            time_period=60.0,
            confidence=0.95  # High confidence
        )
    
        # Act
        result = history._check_estimation_confidence()
    
        # Assert
        assert result is True  # Should consider estimation complete with high confidence and many observations

    # Completing rate limit estimation when confidence threshold is reached
    def test_complete_rate_limit_estimation(self):
        # Arrange
        from datetime import datetime, timedelta, timezone
        from smartsurge.models import RequestHistory, RequestEntry, RequestMethod, SearchStatus
    
        history = RequestHistory(endpoint="api/test", method=RequestMethod.GET)
        history.poisson_alpha = 10.0
        history.poisson_beta = 1.0
        history.beta_a = 10.0
        history.beta_b = 1.0
        history.search_status = SearchStatus.ESTIMATING
    
        # Simulate entries to reach confidence threshold
        for i in range(10):
            entry = RequestEntry(
                endpoint="api/test",
                method=RequestMethod.GET,
                timestamp=datetime.now(timezone.utc) - timedelta(seconds=i),
                status_code=200,
                response_time=0.1,
                success=True
            )
            history.add_request(entry)
    
        # Act
        history._update_bayesian_estimates(limit_exceeded=False)
    
        # Assert
        assert history.search_status == SearchStatus.COMPLETED

    # Handling numerical instability in statistical calculations with regularization
    def test_handle_numerical_instability_with_regularization(self):
        # Arrange
        from smartsurge.models import RequestHistory, RequestEntry, RequestMethod
    
        history = RequestHistory(endpoint="api/test", method=RequestMethod.GET)
        history.poisson_alpha = 1.0
        history.poisson_beta = 0.001  # Very small beta to simulate instability
    
        # Act
        history._update_rate_limit_from_priors()
    
        # Assert
        assert history.rate_limit is not None
        assert history.rate_limit.time_period > 0  # Ensure time period is valid

    # Using credible intervals for conservative rate limiting during estimation
    def test_conservative_rate_limiting_with_credible_intervals(self):
        # Arrange
        from smartsurge.models import RequestHistory, RequestEntry, RequestMethod, SearchStatus
    
        history = RequestHistory(endpoint="api/test", method=RequestMethod.GET)
        history.poisson_alpha = 5.0
        history.poisson_beta = 1.0
        history.search_status = SearchStatus.ESTIMATING
    
        entry = RequestEntry(
            endpoint="api/test",
            method=RequestMethod.GET,
            timestamp=datetime.now(timezone.utc),
            status_code=200,
            response_time=0.1,
            success=True
        )
        history.add_request(entry)
    
        # Act
        history._enforce_bayesian_estimate()
    
        # Assert
        assert len(history.entries) == 1  # Ensure no additional entries were added during enforcement

    # Weighting failed requests differently based on variance ratio for time period estimation
    def test_weighting_failed_requests_based_on_variance_ratio(self):
        # Arrange
        from datetime import datetime, timedelta, timezone
        from smartsurge.models import RequestHistory, RequestEntry, RequestMethod
    
        history = RequestHistory(endpoint="api/test", method=RequestMethod.GET)
        now = datetime.now(timezone.utc)
    
        # Add successful requests
        for i in range(5):
            entry = RequestEntry(
                endpoint="api/test",
                method=RequestMethod.GET,
                timestamp=now - timedelta(seconds=i * 10),
                status_code=200,
                response_time=0.1,
                success=True
            )
            history.add_request(entry)
    
        # Add a failed request (rate limit exceeded)
        failed_entry = RequestEntry(
            endpoint="api/test",
            method=RequestMethod.GET,
            timestamp=now,
            status_code=429,
            response_time=0.1,
            success=False
        )
    
        # Act
        history.log_response_and_update(failed_entry)
    
        # Assert
        assert history.poisson_alpha > 5.0  # Check if alpha increased due to weighting
        assert history.poisson_beta > 5.0 / (60 / 60.0)  # Check if beta increased appropriately

    # Initializing Bayesian priors using domain knowledge of common API rate limits
    def test_initializing_bayesian_priors_with_domain_knowledge(self):
        # Arrange
        from smartsurge.models import RequestHistory, RequestMethod
    
        history = RequestHistory(endpoint="api/test", method=RequestMethod.GET)
    
        # Act
        history._initialize_bayesian_priors()
    
        # Assert
        assert history.poisson_alpha == 5.0
        assert history.poisson_beta == 5.0 / (60 / 60.0)  # Implied rate from "default" common rate limit
        assert history.beta_a == 60  # Prior successes from "default" common rate limit
        assert history.beta_b == 2.0  # Prior failures for slight conservatism

    # Validating that the entry's endpoint matches the history's endpoint
    def test_add_request_endpoint_validation(self):
        # Arrange
        history = RequestHistory(endpoint="api/test", method=RequestMethod.GET)
        entry = RequestEntry(endpoint="api/other", method=RequestMethod.GET, timestamp=datetime.now(timezone.utc), status_code=200, response_time=0.1, success=True)
    
        # Act & Assert
        with pytest.raises(ValueError, match="Entry endpoint and method must match RequestHistory"):
            history.add_request(entry)

    # Combining and sorting entries from both histories by timestamp
    def test_merge_combines_and_sorts_entries(self):
        # Arrange
        history1 = RequestHistory(endpoint="api/test", method=RequestMethod.GET)
        history2 = RequestHistory(endpoint="api/test", method=RequestMethod.GET)
        entry1 = RequestEntry(endpoint="api/test", method=RequestMethod.GET, timestamp=datetime(2023, 10, 1, tzinfo=timezone.utc), status_code=200, response_time=0.1, success=True)
        entry2 = RequestEntry(endpoint="api/test", method=RequestMethod.GET, timestamp=datetime(2023, 10, 2, tzinfo=timezone.utc), status_code=200, response_time=0.1, success=True)
        history1.add_request(entry2)
        history2.add_request(entry1)
    
        # Act
        history1.merge(history2)
    
        # Assert
        assert len(history1.entries) == 2
        assert history1.entries[0].timestamp < history1.entries[1].timestamp

    # Transitioning from WAITING_FOR_FIRST_REFUSAL to ESTIMATING when first 429 error is received
    def test_transition_to_estimating_on_first_429(self):
        # Arrange
        history = RequestHistory(endpoint="api/test", method=RequestMethod.GET)
        entry = RequestEntry(endpoint="api/test", method=RequestMethod.GET, timestamp=datetime.now(timezone.utc), status_code=429, response_time=0.1, success=False)
    
        # Act
        history.log_response_and_update(entry)
    
        # Assert
        assert history.search_status == SearchStatus.ESTIMATING

class Test_UpdateRateLimitFromPriors:

    # Updates rate limit based on Bayesian priors when all parameters are properly initialized
    def test_update_rate_limit_with_valid_priors(self, mocker):
        # Arrange
        from smartsurge.models import RequestHistory, RequestMethod
    
        history = RequestHistory(
            endpoint="/api/test",
            method=RequestMethod.GET,
            poisson_alpha=10.0,
            poisson_beta=5.0,
            beta_a=8.0,
            beta_b=2.0
        )
    
        # Act
        history._update_rate_limit_from_priors()
    
        # Assert
        assert history.rate_limit is not None
        assert history.rate_limit.endpoint == "/api/test"
        assert history.rate_limit.method == RequestMethod.GET
        assert history.rate_limit.max_requests >= 1
        assert history.rate_limit.time_period > 0
        assert 0 < history.rate_limit.confidence < 1

    # Handles case when Bayesian priors are not initialized by returning early
    def test_early_return_when_priors_not_initialized(self, mocker):
        # Arrange
        from smartsurge.models import RequestHistory, RequestMethod
    
        # Create history with uninitialized priors
        history = RequestHistory(
            endpoint="/api/test",
            method=RequestMethod.GET,
            poisson_alpha=None,
            poisson_beta=10.0,
            beta_a=8.0,
            beta_b=2.0
        )
    
        # Act
        history._update_rate_limit_from_priors()
    
        # Assert
        assert history.rate_limit is None
    
        # Test with different uninitialized parameters
        history = RequestHistory(
            endpoint="/api/test",
            method=RequestMethod.GET,
            poisson_alpha=10.0,
            poisson_beta=None,
            beta_a=None,
            beta_b=2.0
        )
    
        history._update_rate_limit_from_priors()
        assert history.rate_limit is None

    # Calculates time period using gamma distribution statistics from poisson_alpha and poisson_beta
    def test_time_period_calculation_from_gamma_distribution(self, mocker):
        # Arrange
        from smartsurge.models import RequestHistory, RequestMethod
        import scipy.stats as stats
    
        # Mock the stats.gamma.ppf function to return predictable values
        mock_ppf = mocker.patch.object(stats.gamma, 'ppf')
        mock_ppf.side_effect = [0.5, 2.0]  # lower and upper bounds
    
        history = RequestHistory(
            endpoint="/api/test",
            method=RequestMethod.GET,
            poisson_alpha=10.0,
            poisson_beta=5.0,
            beta_a=8.0,
            beta_b=2.0
        )
    
        # Act
        history._update_rate_limit_from_priors()
    
        # Assert
        assert history.rate_limit is not None
        # Expected time_period = 1.0 / (10.0 / 5.0) = 0.5
        assert history.rate_limit.time_period == 0.5
        # Expected time_period_lower = 1.0 / 2.0 = 0.5
        assert history.rate_limit.time_period_lower == 0.5
        # Expected time_period_upper = 1.0 / 0.5 = 2.0
        assert history.rate_limit.time_period_upper == 2.0
    
        # Verify gamma.ppf was called with correct parameters
        mock_ppf.assert_any_call(0.025, 10.0, scale=1/5.0)
        mock_ppf.assert_any_call(0.975, 10.0, scale=1/5.0)

    # Calculates max_requests using beta distribution statistics and inter-arrival times
    def test_max_requests_calculation_from_beta_distribution(self, mocker):
        # Arrange
        from smartsurge.models import RequestHistory, RequestMethod, RequestEntry
        import scipy.stats as stats
        from datetime import datetime, timezone, timedelta
    
        # Mock the stats.beta.ppf function to return predictable values
        mock_ppf = mocker.patch.object(stats.beta, 'ppf')
        mock_ppf.side_effect = [0.7, 0.9]  # lower and upper bounds
    
        history = RequestHistory(
            endpoint="/api/test",
            method=RequestMethod.GET,
            poisson_alpha=10.0,
            poisson_beta=5.0,
            beta_a=8.0,
            beta_b=2.0
        )
    
        # Add some successful requests with known timestamps
        now = datetime.now(timezone.utc)
        history.entries = [
            RequestEntry(timestamp=now - timedelta(seconds=10), status_code=200, response_time=0.1, success=True),
            RequestEntry(timestamp=now - timedelta(seconds=8), status_code=200, response_time=0.1, success=True),
            RequestEntry(timestamp=now - timedelta(seconds=6), status_code=200, response_time=0.1, success=True),
            RequestEntry(timestamp=now - timedelta(seconds=4), status_code=200, response_time=0.1, success=True),
            RequestEntry(timestamp=now - timedelta(seconds=2), status_code=200, response_time=0.1, success=True)
        ]
    
        # Act
        history._update_rate_limit_from_priors()
    
        # Assert
        assert history.rate_limit is not None
        # Expected p_mean = 8.0 / (8.0 + 2.0) = 0.8
        # Expected avg_delta_t = 2.0 seconds
        # Expected max_requests = floor(0.8 * (0.5 / 2.0)) = floor(0.2) = 0, but min is 1
        assert history.rate_limit.max_requests >= 1
    
        # Verify beta.ppf was called with correct parameters
        mock_ppf.assert_any_call(0.025, 8.0, 2.0)
        mock_ppf.assert_any_call(0.975, 8.0, 2.0)

    # Computes confidence level based on credible interval widths using harmonic mean
    def test_confidence_calculation_using_harmonic_mean(self, mocker):
        # Arrange
        from smartsurge.models import RequestHistory, RequestMethod
        import scipy.stats as stats
    
        # Mock the stats functions to return predictable values
        mocker.patch.object(stats.gamma, 'ppf', side_effect=[0.5, 2.0])
        mocker.patch.object(stats.beta, 'ppf', side_effect=[0.7, 0.9])
    
        history = RequestHistory(
            endpoint="/api/test",
            method=RequestMethod.GET,
            poisson_alpha=10.0,
            poisson_beta=5.0,
            beta_a=8.0,
            beta_b=2.0
        )
    
        # Act
        history._update_rate_limit_from_priors()
    
        # Assert
        assert history.rate_limit is not None
        assert 0.01 <= history.rate_limit.confidence <= 0.99
    
        # Manually calculate expected confidence
        time_rel_width = (2.0 - 0.5) / 0.5  # (time_period_upper - time_period_lower) / time_period
        req_rel_width = (history.rate_limit.max_requests_upper - history.rate_limit.max_requests_lower) / history.rate_limit.max_requests
        expected_confidence = 2.0 / (1.0 / (time_rel_width + 1e-10) + 1.0 / (req_rel_width + 1e-10))
        expected_confidence = min(0.99, max(0.01, expected_confidence))
    
        assert abs(history.rate_limit.confidence - expected_confidence) < 1e-6

    # Creates and returns a new RateLimit object with calculated parameters
    def test_creates_rate_limit_object_with_correct_parameters(self, mocker):
        # Arrange
        from smartsurge.models import RequestHistory, RequestMethod, RateLimit
        from datetime import datetime, timezone
    
        # Mock datetime.now to return a fixed time
        fixed_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        mocker.patch('datetime.datetime', mocker.MagicMock(now=mocker.MagicMock(return_value=fixed_time)))
    
        history = RequestHistory(
            endpoint="/api/test",
            method=RequestMethod.GET,
            poisson_alpha=10.0,
            poisson_beta=5.0,
            beta_a=8.0,
            beta_b=2.0
        )
    
        # Act
        history._update_rate_limit_from_priors()
    
        # Assert
        assert isinstance(history.rate_limit, RateLimit)
        assert history.rate_limit.endpoint == "/api/test"
        assert history.rate_limit.method == RequestMethod.GET
        assert history.rate_limit.last_updated == fixed_time
        assert history.rate_limit.time_period_lower is not None
        assert history.rate_limit.time_period_upper is not None
        assert history.rate_limit.max_requests_lower is not None
        assert history.rate_limit.max_requests_upper is not None

    # Handles ValueError and ZeroDivisionError in gamma distribution calculations with fallback logic
    def test_handles_errors_in_gamma_distribution_calculations(self, mocker):
        # Arrange
        from smartsurge.models import RequestHistory, RequestMethod
        import scipy.stats as stats
    
        # Mock stats.gamma.ppf to raise an exception
        mock_ppf = mocker.patch.object(stats.gamma, 'ppf')
        mock_ppf.side_effect = ValueError("Test error")
    
        # Mock logger to verify warning
        mock_logger = mocker.MagicMock()
    
        history = RequestHistory(
            endpoint="/api/test",
            method=RequestMethod.GET,
            poisson_alpha=10.0,
            poisson_beta=5.0,
            beta_a=8.0,
            beta_b=2.0,
            logger=mock_logger
        )
    
        # Act
        history._update_rate_limit_from_priors()
    
        # Assert
        assert history.rate_limit is not None
        mock_logger.warning.assert_called_once()
        assert "Gamma credible interval calculation failed" in mock_logger.warning.call_args[0][0]
    
        # Test with ZeroDivisionError
        mock_ppf.side_effect = ZeroDivisionError("Division by zero")
        history._update_rate_limit_from_priors()
        assert history.rate_limit is not None
        assert mock_logger.warning.call_count == 2

    # Handles ValueError and ZeroDivisionError in beta distribution calculations with fallback logic
    def test_handles_errors_in_beta_distribution_calculations(self, mocker):
        # Arrange
        from smartsurge.models import RequestHistory, RequestMethod
        import scipy.stats as stats
    
        # Mock gamma.ppf to return normal values but beta.ppf to raise an exception
        mocker.patch.object(stats.gamma, 'ppf', side_effect=[0.5, 2.0])
        mock_beta_ppf = mocker.patch.object(stats.beta, 'ppf')
        mock_beta_ppf.side_effect = ValueError("Test error")
    
        # Mock logger to verify warning
        mock_logger = mocker.MagicMock()
    
        history = RequestHistory(
            endpoint="/api/test",
            method=RequestMethod.GET,
            poisson_alpha=10.0,
            poisson_beta=5.0,
            beta_a=8.0,
            beta_b=2.0,
            logger=mock_logger
        )
    
        # Act
        history._update_rate_limit_from_priors()
    
        # Assert
        assert history.rate_limit is not None
        mock_logger.warning.assert_called_once()
        assert "Beta credible interval calculation failed" in mock_logger.warning.call_args[0][0]
    
        # Test with ZeroDivisionError
        mock_beta_ppf.side_effect = ZeroDivisionError("Division by zero")
        history._update_rate_limit_from_priors()
        assert history.rate_limit is not None
        assert mock_logger.warning.call_count == 2

    # Ensures time_period values are bounded between min_time_period and max_time_period
    def test_time_period_values_are_bounded(self, mocker):
        # Arrange
        from smartsurge.models import RequestHistory, RequestMethod
        import scipy.stats as stats
    
        # Mock stats.gamma.ppf to return extreme values
        mock_ppf = mocker.patch.object(stats.gamma, 'ppf')
        mock_ppf.side_effect = [0.0001, 1000.0]  # very small lower bound, very large upper bound
    
        # Set custom min and max time periods
        history = RequestHistory(
            endpoint="/api/test",
            method=RequestMethod.GET,
            poisson_alpha=10.0,
            poisson_beta=5.0,
            beta_a=8.0,
            beta_b=2.0,
            min_time_period=2.0,
            max_time_period=100.0
        )
    
        # Act
        history._update_rate_limit_from_priors()
    
        # Assert
        assert history.rate_limit is not None
        # time_period_lower should be bounded by min_time_period
        assert history.rate_limit.time_period_lower >= 2.0
        # time_period_upper should be bounded by max_time_period
        assert history.rate_limit.time_period_upper <= 100.0
        # time_period should be between min and max
        assert 2.0 <= history.rate_limit.time_period <= 100.0

    # Ensures max_requests values are reasonable (between 1 and 10000)
    def test_max_requests_values_are_reasonable(self, mocker):
        # Arrange
        from smartsurge.models import RequestHistory, RequestMethod
        import scipy.stats as stats
    
        # Mock stats functions to return normal gamma values but extreme beta values
        mocker.patch.object(stats.gamma, 'ppf', side_effect=[0.5, 2.0])
        mocker.patch.object(stats.beta, 'ppf', side_effect=[0.999, 0.9999])  # very high probabilities
    
        # Create a history with parameters that would lead to very high max_requests
        history = RequestHistory(
            endpoint="/api/test",
            method=RequestMethod.GET,
            poisson_alpha=10.0,
            poisson_beta=5.0,
            beta_a=1000.0,  # Very high success probability
            beta_b=1.0
        )
    
        # Add entries with very small time differences to make avg_delta_t very small
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        history.entries = [
            RequestEntry(timestamp=now - timedelta(microseconds=100), status_code=200, response_time=0.001, success=True),
            RequestEntry(timestamp=now - timedelta(microseconds=50), status_code=200, response_time=0.001, success=True)
        ]
    
        # Act
        history._update_rate_limit_from_priors()
    
        # Assert
        assert history.rate_limit is not None
        # max_requests should be capped at 10000
        assert history.rate_limit.max_requests <= 10000
        assert history.rate_limit.max_requests >= 1

    # Ensures max_requests_lower is at least 1 and max_requests_upper is greater than max_requests_lower
    def test_max_requests_bounds_are_valid(self, mocker):
        # Arrange
        from smartsurge.models import RequestHistory, RequestMethod
        import scipy.stats as stats
    
        # Mock stats functions to return values that would lead to invalid bounds
        mocker.patch.object(stats.gamma, 'ppf', side_effect=[0.5, 2.0])
        mocker.patch.object(stats.beta, 'ppf', side_effect=[-0.1, 0.1])  # negative lower bound
    
        history = RequestHistory(
            endpoint="/api/test",
            method=RequestMethod.GET,
            poisson_alpha=10.0,
            poisson_beta=5.0,
            beta_a=0.1,  # Very low success probability
            beta_b=10.0
        )
    
        # Act
        history._update_rate_limit_from_priors()
    
        # Assert
        assert history.rate_limit is not None
        # max_requests_lower should be at least 1
        assert history.rate_limit.max_requests_lower >= 1
        # max_requests_upper should be greater than max_requests_lower
        assert history.rate_limit.max_requests_upper > history.rate_limit.max_requests_lower

    # Uses recent successful requests to calculate average inter-arrival time
    def test_uses_recent_successful_requests_for_inter_arrival_time(self, mocker):
        # Arrange
        from smartsurge.models import RequestHistory, RequestMethod, RequestEntry
        import scipy.stats as stats
        from datetime import datetime, timezone, timedelta
    
        # Mock stats functions to return predictable values
        mocker.patch.object(stats.gamma, 'ppf', side_effect=[0.5, 2.0])
        mocker.patch.object(stats.beta, 'ppf', side_effect=[0.7, 0.9])
    
        history = RequestHistory(
            endpoint="/api/test",
            method=RequestMethod.GET,
            poisson_alpha=10.0,
            poisson_beta=5.0,
            beta_a=8.0,
            beta_b=2.0
        )
    
        # Add a mix of successful and failed requests with known timestamps
        now = datetime.now(timezone.utc)
        history.entries = [
            # Successful requests with 2-second intervals
            RequestEntry(timestamp=now - timedelta(seconds=10), status_code=200, response_time=0.1, success=True),
            RequestEntry(timestamp=now - timedelta(seconds=8), status_code=200, response_time=0.1, success=True),
            # Failed request (should be ignored)
            RequestEntry(timestamp=now - timedelta(seconds=7), status_code=500, response_time=0.1, success=False),
            # More successful requests
            RequestEntry(timestamp=now - timedelta(seconds=6), status_code=200, response_time=0.1, success=True),
            RequestEntry(timestamp=now - timedelta(seconds=4), status_code=200, response_time=0.1, success=True),
            RequestEntry(timestamp=now - timedelta(seconds=2), status_code=200, response_time=0.1, success=True)
        ]
    
        # Act
        history._update_rate_limit_from_priors()
    
        # Assert
        assert history.rate_limit is not None
    
        # Create a history with no successful requests for comparison
        history_no_success = RequestHistory(
            endpoint="/api/test",
            method=RequestMethod.GET,
            poisson_alpha=10.0,
            poisson_beta=5.0,
            beta_a=8.0,
            beta_b=2.0
        )
    
        history_no_success.entries = [
            RequestEntry(timestamp=now - timedelta(seconds=10), status_code=500, response_time=0.1, success=False),
            RequestEntry(timestamp=now - timedelta(seconds=8), status_code=500, response_time=0.1, success=False)
        ]
    
        history_no_success._update_rate_limit_from_priors()
    
        # The rate limits should be different due to different inter-arrival times
        assert history.rate_limit.max_requests != history_no_success.rate_limit.max_requests

    # Falls back to standard deviation approximation when statistical functions fail
    def test_fallback_to_standard_deviation_approximation(self, mocker):
        # Arrange
        mocker.patch('src.smartsurge.models.stats.gamma.ppf', side_effect=ValueError("Mocked error"))
        instance = RequestHistory(
            poisson_alpha=2.0,
            poisson_beta=0.0,  # This will cause division by zero
            beta_a=2.0,
            beta_b=2.0,
            entries=[]
        )
        instance.logger = mocker.Mock()

        # Act
        instance._update_rate_limit_from_priors()

        # Assert
        assert instance.rate_limit is not None
        assert instance.rate_limit.max_requests > 0
        assert instance.rate_limit.time_period > 0
        instance.logger.warning.assert_called_once_with(
            f"[{instance.request_id}] Gamma credible interval calculation failed: Mocked error, using fallback"
        )

    # Bounds confidence value between 0.01 and 0.99 to prevent numerical instabilities
    def test_confidence_value_bounding(self):
        # Arrange
        instance = RequestHistory(
            poisson_alpha=100.0,
            poisson_beta=1.0,
            beta_a=100.0,
            beta_b=1.0,
            entries=[]
        )

        # Act
        instance._update_rate_limit_from_priors()

        # Assert
        assert instance.rate_limit is not None
        assert 0.01 <= instance.rate_limit.confidence <= 0.99

    # Logs detailed debug information about updated rate limit parameters
    def test_logs_detailed_debug_information(self, mocker):
        # Arrange
        instance = RequestHistory(
            poisson_alpha=2.0,
            poisson_beta=2.0,
            beta_a=2.0,
            beta_b=2.0,
            entries=[]
        )
        instance.logger = mocker.Mock()

        # Act
        instance._update_rate_limit_from_priors()

        # Assert
        assert instance.rate_limit is not None
        instance.logger.debug.assert_called_once()

    # Returns early without updating rate limit if any of poisson_alpha, poisson_beta, beta_a, or beta_b is None
    def test_early_return_when_priors_none(self, mocker):
        # Arrange
        mock_logger = mocker.Mock()
        request_history = RequestHistory(
            endpoint="test_endpoint",
            method=RequestMethod.GET,
            logger=mock_logger
        )
        request_history.poisson_alpha = None  # Priors are not initialized

        # Act
        request_history._update_rate_limit_from_priors()

        # Assert
        assert request_history.rate_limit is None
        mock_logger.debug.assert_not_called()

    # Creates a new RateLimit object with calculated parameters and assigns it to self.rate_limit
    def test_creates_rate_limit_object(self, mocker):
        # Arrange
        mock_logger = mocker.Mock()
        request_history = RequestHistory(
            endpoint="test_endpoint",
            method=RequestMethod.GET,
            logger=mock_logger,
            poisson_alpha=2.0,
            poisson_beta=1.0,
            beta_a=2.0,
            beta_b=2.0
        )
        request_history.entries = [mocker.Mock(success=True, timestamp=datetime.now(timezone.utc)) for _ in range(5)]

        # Act
        request_history._update_rate_limit_from_priors()

        # Assert
        assert request_history.rate_limit is not None
        assert isinstance(request_history.rate_limit, RateLimit)
        assert request_history.rate_limit.endpoint == "test_endpoint"
        assert request_history.rate_limit.method == RequestMethod.GET

    # Uses 95% credible intervals (0.025 to 0.975) for statistical calculations
    def test_credible_intervals_usage(self, mocker):
        from scipy import stats
        
        # Arrange
        mocker.patch('src.smartsurge.models.stats.gamma.ppf', return_value=1.0)
        mocker.patch('src.smartsurge.models.stats.beta.ppf', return_value=0.5)
        instance = RequestHistory(poisson_alpha=2.0, poisson_beta=2.0, beta_a=2.0, beta_b=2.0)
    
        # Act
        instance._update_rate_limit_from_priors()
    
        # Assert
        stats.gamma.ppf.assert_any_call(0.025, 2.0, scale=1/2.0)
        stats.gamma.ppf.assert_any_call(0.975, 2.0, scale=1/2.0)
        stats.beta.ppf.assert_any_call(0.025, 2.0, 2.0)
        stats.beta.ppf.assert_any_call(0.975, 2.0, 2.0)

    # Updates rate limit with current timestamp in the last_updated field
    def test_rate_limit_update_with_timestamp(self, mocker):
        # Arrange
        mocker.patch('src.smartsurge.models.datetime', wraps=datetime)
        instance = RequestHistory(poisson_alpha=2.0, poisson_beta=2.0, beta_a=2.0, beta_b=2.0)
    
        # Act
        instance._update_rate_limit_from_priors()
    
        # Assert
        assert instance.rate_limit is not None
        assert instance.rate_limit.last_updated == datetime.now(timezone.utc)

class Test_UpdateBayesianEstimates:

    # Updates Bayesian estimates based on request outcome (limit_exceeded=True/False)
    def test_updates_bayesian_estimates_based_on_request_outcome(self, mocker):
        # Arrange
        from datetime import datetime, timezone, timedelta
        from smartsurge.models import RequestEntry
    
        rate_limiter = mocker.MagicMock()
        rate_limiter.poisson_alpha = 1.0
        rate_limiter.poisson_beta = 1.0
        rate_limiter.beta_a = 1.0
        rate_limiter.beta_b = 1.0
        rate_limiter.entries = [
            RequestEntry(timestamp=datetime.now(timezone.utc) - timedelta(seconds=2), status_code=200, response_time=0.1, success=True),
            RequestEntry(timestamp=datetime.now(timezone.utc), status_code=200, response_time=0.1, success=True)
        ]
        rate_limiter.max_time_period = 10.0
        rate_limiter.regularization_strength = 0.01
        rate_limiter.request_id = "test-id"
        rate_limiter.logger = mocker.MagicMock()
        rate_limiter._update_rate_limit_from_priors = mocker.MagicMock()
    
        # Act
        rate_limiter._update_bayesian_estimates(limit_exceeded=False)
    
        # Assert
        assert rate_limiter.poisson_alpha == 2.0  # 1.0 + 1.0
        assert rate_limiter.beta_a == 2.0  # 1.0 + 1.0
        assert rate_limiter.beta_b == 1.0  # Unchanged
        assert rate_limiter._update_rate_limit_from_priors.called

    # Calculates inter-arrival time between latest and previous requests
    def test_calculates_inter_arrival_time_correctly(self, mocker):
        # Arrange
        from datetime import datetime, timezone, timedelta
        from smartsurge.models import RequestEntry
    
        rate_limiter = mocker.MagicMock()
        rate_limiter.poisson_alpha = 1.0
        rate_limiter.poisson_beta = 1.0
        rate_limiter.beta_a = 1.0
        rate_limiter.beta_b = 1.0
    
        # Create entries with exactly 3 seconds difference
        previous_time = datetime.now(timezone.utc) - timedelta(seconds=3)
        latest_time = datetime.now(timezone.utc)
    
        rate_limiter.entries = [
            RequestEntry(timestamp=previous_time, status_code=200, response_time=0.1, success=True),
            RequestEntry(timestamp=latest_time, status_code=200, response_time=0.1, success=True)
        ]
        rate_limiter.max_time_period = 10.0
        rate_limiter.regularization_strength = 0.01
        rate_limiter.request_id = "test-id"
        rate_limiter.logger = mocker.MagicMock()
        rate_limiter._update_rate_limit_from_priors = mocker.MagicMock()
    
        # Act
        rate_limiter._update_bayesian_estimates(limit_exceeded=False)
    
        # Assert
        time_diff = (latest_time - previous_time).total_seconds()
        assert rate_limiter.poisson_beta == 1.0 + time_diff
        rate_limiter._update_rate_limit_from_priors.assert_called_once()

    # Updates Gamma distribution parameters for time period estimation
    def test_updates_gamma_distribution_parameters(self, mocker):
        # Arrange
        from datetime import datetime, timezone, timedelta
        from smartsurge.models import RequestEntry
    
        rate_limiter = mocker.MagicMock()
        rate_limiter.poisson_alpha = 2.5
        rate_limiter.poisson_beta = 3.7
        rate_limiter.beta_a = 1.0
        rate_limiter.beta_b = 1.0
    
        rate_limiter.entries = [
            RequestEntry(timestamp=datetime.now(timezone.utc) - timedelta(seconds=2), status_code=200, response_time=0.1, success=True),
            RequestEntry(timestamp=datetime.now(timezone.utc), status_code=200, response_time=0.1, success=True)
        ]
        rate_limiter.max_time_period = 10.0
        rate_limiter.regularization_strength = 0.01
        rate_limiter.request_id = "test-id"
        rate_limiter.logger = mocker.MagicMock()
        rate_limiter._update_rate_limit_from_priors = mocker.MagicMock()
    
        # Act
        rate_limiter._update_bayesian_estimates(limit_exceeded=False)
    
        # Assert
        assert rate_limiter.poisson_alpha == 3.5  # 2.5 + 1.0
        assert rate_limiter.poisson_beta > 3.7  # 3.7 + time_diff
        rate_limiter._update_rate_limit_from_priors.assert_called_once()

    # Updates Beta distribution parameters for success probability model
    def test_updates_beta_distribution_parameters(self, mocker):
        # Arrange
        from datetime import datetime, timezone, timedelta
        from smartsurge.models import RequestEntry
    
        rate_limiter = mocker.MagicMock()
        rate_limiter.poisson_alpha = 1.0
        rate_limiter.poisson_beta = 1.0
        rate_limiter.beta_a = 4.2
        rate_limiter.beta_b = 2.3
    
        rate_limiter.entries = [
            RequestEntry(timestamp=datetime.now(timezone.utc) - timedelta(seconds=2), status_code=200, response_time=0.1, success=True),
            RequestEntry(timestamp=datetime.now(timezone.utc), status_code=200, response_time=0.1, success=True)
        ]
        rate_limiter.max_time_period = 10.0
        rate_limiter.regularization_strength = 0.01
        rate_limiter.request_id = "test-id"
        rate_limiter.logger = mocker.MagicMock()
        rate_limiter._update_rate_limit_from_priors = mocker.MagicMock()
    
        # Act - Test success case
        rate_limiter._update_bayesian_estimates(limit_exceeded=False)
    
        # Assert
        assert rate_limiter.beta_a == 5.2  # 4.2 + 1.0
        assert rate_limiter.beta_b == 2.3  # Unchanged
    
        # Act - Test failure case
        rate_limiter._update_bayesian_estimates(limit_exceeded=True)
    
        # Assert
        assert rate_limiter.beta_a == 5.2  # Unchanged
        assert rate_limiter.beta_b == 3.3  # 2.3 + 1.0

    # Calculates weight for time period update based on variance ratio for failed requests
    def test_calculates_weight_for_failed_requests(self, mocker):
        # Arrange
        from datetime import datetime, timezone, timedelta
        from smartsurge.models import RequestEntry
    
        rate_limiter = mocker.MagicMock()
        rate_limiter.poisson_alpha = 1.0
        rate_limiter.poisson_beta = 1.0
        rate_limiter.beta_a = 1.0
        rate_limiter.beta_b = 1.0
    
        # Create 10 entries with varying time differences and success status
        base_time = datetime.now(timezone.utc) - timedelta(seconds=50)
        entries = []
        for i in range(10):
            # Create entries with different time gaps and alternating success status
            timestamp = base_time + timedelta(seconds=i*5 + (i % 3))
            success = i % 2 == 0  # Alternating success status
            entries.append(RequestEntry(timestamp=timestamp, status_code=200 if success else 429, 
                                      response_time=0.1, success=success))
    
        # Add the latest entry
        entries.append(RequestEntry(timestamp=datetime.now(timezone.utc), status_code=429, 
                                  response_time=0.1, success=False))
    
        rate_limiter.entries = entries
        rate_limiter.max_time_period = 10.0
        rate_limiter.regularization_strength = 0.01
        rate_limiter.request_id = "test-id"
        rate_limiter.logger = mocker.MagicMock()
        rate_limiter._update_rate_limit_from_priors = mocker.MagicMock()
    
        # Act
        rate_limiter._update_bayesian_estimates(limit_exceeded=True)
    
        # Assert
        # Weight should be between 0.3 and 0.7 for failed requests
        assert 1.3 <= rate_limiter.poisson_alpha <= 1.7
        assert rate_limiter.beta_b == 2.0  # 1.0 + 1.0 (one more failure)
        rate_limiter._update_rate_limit_from_priors.assert_called_once()

    # Calls _update_rate_limit_from_priors() after updating estimates
    def test_calls_update_rate_limit_from_priors(self, mocker):
        # Arrange
        from datetime import datetime, timezone, timedelta
        from smartsurge.models import RequestEntry
    
        rate_limiter = mocker.MagicMock()
        rate_limiter.poisson_alpha = 1.0
        rate_limiter.poisson_beta = 1.0
        rate_limiter.beta_a = 1.0
        rate_limiter.beta_b = 1.0
        rate_limiter.entries = [
            RequestEntry(timestamp=datetime.now(timezone.utc) - timedelta(seconds=2), status_code=200, response_time=0.1, success=True),
            RequestEntry(timestamp=datetime.now(timezone.utc), status_code=200, response_time=0.1, success=True)
        ]
        rate_limiter.max_time_period = 10.0
        rate_limiter.regularization_strength = 0.01
        rate_limiter.request_id = "test-id"
        rate_limiter.logger = mocker.MagicMock()
        rate_limiter._update_rate_limit_from_priors = mocker.MagicMock()
    
        # Act
        rate_limiter._update_bayesian_estimates(limit_exceeded=False)
    
        # Assert
        rate_limiter._update_rate_limit_from_priors.assert_called_once()

    # Handles case when Bayesian priors are not initialized
    def test_handles_uninitialized_bayesian_priors(self, mocker):
        # Arrange
        from datetime import datetime, timezone, timedelta
        from smartsurge.models import RequestEntry
    
        rate_limiter = mocker.MagicMock()
        rate_limiter.poisson_alpha = None
        rate_limiter.poisson_beta = 1.0
        rate_limiter.beta_a = 1.0
        rate_limiter.beta_b = 1.0
        rate_limiter.entries = [
            RequestEntry(timestamp=datetime.now(timezone.utc) - timedelta(seconds=2), status_code=200, response_time=0.1, success=True),
            RequestEntry(timestamp=datetime.now(timezone.utc), status_code=200, response_time=0.1, success=True)
        ]
        rate_limiter.request_id = "test-id"
        rate_limiter.logger = mocker.MagicMock()
        rate_limiter._update_rate_limit_from_priors = mocker.MagicMock()
    
        # Act
        rate_limiter._update_bayesian_estimates(limit_exceeded=False)
    
        # Assert
        rate_limiter.logger.warning.assert_called_once()
        assert not rate_limiter._update_rate_limit_from_priors.called
    
        # Test with another uninitialized parameter
        rate_limiter.logger.reset_mock()
        rate_limiter.poisson_alpha = 1.0
        rate_limiter.beta_a = None
    
        # Act
        rate_limiter._update_bayesian_estimates(limit_exceeded=False)
    
        # Assert
        rate_limiter.logger.warning.assert_called_once()
        assert not rate_limiter._update_rate_limit_from_priors.called

    # Handles case when there are fewer than 2 entries in history
    def test_handles_insufficient_history_entries(self, mocker):
        # Arrange
        from datetime import datetime, timezone
        from smartsurge.models import RequestEntry
    
        rate_limiter = mocker.MagicMock()
        rate_limiter.poisson_alpha = 1.0
        rate_limiter.poisson_beta = 1.0
        rate_limiter.beta_a = 1.0
        rate_limiter.beta_b = 1.0
    
        # Test with empty entries
        rate_limiter.entries = []
        rate_limiter.request_id = "test-id"
        rate_limiter.logger = mocker.MagicMock()
        rate_limiter._update_rate_limit_from_priors = mocker.MagicMock()
    
        # Act
        rate_limiter._update_bayesian_estimates(limit_exceeded=False)
    
        # Assert
        rate_limiter.logger.warning.assert_called_once()
        assert not rate_limiter._update_rate_limit_from_priors.called
    
        # Test with only one entry
        rate_limiter.logger.reset_mock()
        rate_limiter.entries = [
            RequestEntry(timestamp=datetime.now(timezone.utc), status_code=200, response_time=0.1, success=True)
        ]
    
        # Act
        rate_limiter._update_bayesian_estimates(limit_exceeded=False)
    
        # Assert
        rate_limiter.logger.warning.assert_called_once()
        assert not rate_limiter._update_rate_limit_from_priors.called

    # Handles insufficient data for variance estimation (less than 5 recent entries)
    def test_handles_insufficient_data_for_variance_estimation(self, mocker):
        # Arrange
        from datetime import datetime, timezone, timedelta
        from smartsurge.models import RequestEntry
    
        rate_limiter = mocker.MagicMock()
        rate_limiter.poisson_alpha = 1.0
        rate_limiter.poisson_beta = 1.0
        rate_limiter.beta_a = 1.0
        rate_limiter.beta_b = 1.0
    
        # Create only 4 entries (less than 5 needed for variance estimation)
        base_time = datetime.now(timezone.utc) - timedelta(seconds=10)
        entries = []
        for i in range(4):
            timestamp = base_time + timedelta(seconds=i*2)
            entries.append(RequestEntry(timestamp=timestamp, status_code=200, response_time=0.1, success=True))
    
        rate_limiter.entries = entries
        rate_limiter.max_time_period = 10.0
        rate_limiter.regularization_strength = 0.01
        rate_limiter.request_id = "test-id"
        rate_limiter.logger = mocker.MagicMock()
        rate_limiter._update_rate_limit_from_priors = mocker.MagicMock()
    
        # Act
        rate_limiter._update_bayesian_estimates(limit_exceeded=True)
    
        # Assert
        # Should use fallback weight of 0.5 for insufficient data
        assert rate_limiter.poisson_alpha == 1.5  # 1.0 + 0.5
        assert rate_limiter.beta_b == 2.0  # 1.0 + 1.0 (one more failure)
        rate_limiter._update_rate_limit_from_priors.assert_called_once()

    # Prevents division by zero when calculating variance ratios
    def test_prevents_division_by_zero_in_variance_calculation(self, mocker):
        # Arrange
        from datetime import datetime, timezone, timedelta
        from smartsurge.models import RequestEntry
        import math
    
        rate_limiter = mocker.MagicMock()
        rate_limiter.poisson_alpha = 1.0
        rate_limiter.poisson_beta = 1.0
        rate_limiter.beta_a = 1.0
        rate_limiter.beta_b = 1.0
    
        # Create entries with identical timestamps to force zero variance
        base_time = datetime.now(timezone.utc) - timedelta(seconds=10)
        entries = []
    
        # Create 10 entries with identical time differences (will lead to zero variance)
        for i in range(10):
            timestamp = base_time + timedelta(seconds=i*2)  # Exactly 2 seconds apart
            entries.append(RequestEntry(timestamp=timestamp, status_code=200, response_time=0.1, success=True))
    
        rate_limiter.entries = entries
        rate_limiter.max_time_period = 10.0
        rate_limiter.regularization_strength = 0.0  # Set to zero to test division by zero protection
        rate_limiter.request_id = "test-id"
        rate_limiter.logger = mocker.MagicMock()
        rate_limiter._update_rate_limit_from_priors = mocker.MagicMock()
    
        # Act - this should not raise an exception
        rate_limiter._update_bayesian_estimates(limit_exceeded=True)
    
        # Assert
        # Should use fallback weight of 0.5 when variance calculation fails
        assert rate_limiter.poisson_alpha == 1.5  # 1.0 + 0.5
        assert rate_limiter.beta_b == 2.0  # 1.0 + 1.0 (one more failure)
        rate_limiter._update_rate_limit_from_priors.assert_called_once()

    # Ensures numerical stability by enforcing minimum values for parameters
    def test_ensures_numerical_stability_of_parameters(self, mocker):
        # Arrange
        from datetime import datetime, timezone, timedelta
        from smartsurge.models import RequestEntry
    
        rate_limiter = mocker.MagicMock()
        rate_limiter.poisson_alpha = 0.1  # Below minimum of 1.0
        rate_limiter.poisson_beta = 0.01  # Potentially below minimum
        rate_limiter.beta_a = 0.2  # Below minimum of 0.5
        rate_limiter.beta_b = 0.3  # Below minimum of 0.5
    
        rate_limiter.entries = [
            RequestEntry(timestamp=datetime.now(timezone.utc) - timedelta(seconds=2), status_code=200, response_time=0.1, success=True),
            RequestEntry(timestamp=datetime.now(timezone.utc), status_code=200, response_time=0.1, success=True)
        ]
        rate_limiter.max_time_period = 10.0
        rate_limiter.regularization_strength = 0.01
        rate_limiter.request_id = "test-id"
        rate_limiter.logger = mocker.MagicMock()
        rate_limiter._update_rate_limit_from_priors = mocker.MagicMock()
    
        # Act
        rate_limiter._update_bayesian_estimates(limit_exceeded=False)
    
        # Assert
        assert rate_limiter.poisson_alpha == 1.0  # Enforced minimum
        assert rate_limiter.poisson_beta >= 1.0 / rate_limiter.max_time_period  # Enforced minimum
        assert rate_limiter.beta_a == 1.2  # 0.2 + 1.0, then enforced minimum of 0.5
        assert rate_limiter.beta_b == 0.5  # Enforced minimum
        rate_limiter._update_rate_limit_from_priors.assert_called_once()

    # Bounds weight to reasonable range (0.3 to 0.7) for failed requests
    def test_bounds_weight_to_reasonable_range(self, mocker):
        # Arrange
        from datetime import datetime, timezone, timedelta
        from smartsurge.models import RequestEntry
    
        rate_limiter = mocker.MagicMock()
        rate_limiter.poisson_alpha = 1.0
        rate_limiter.poisson_beta = 1.0
        rate_limiter.beta_a = 1.0
        rate_limiter.beta_b = 1.0
    
        # Create mock for variance calculation that would result in extreme weights
        mocker.patch('scipy.stats.variation', side_effect=[0.1, 1.0])  # This would give a very small ratio
    
        # Create entries with varying time differences
        base_time = datetime.now(timezone.utc) - timedelta(seconds=50)
        entries = []
    
        # Create entries with different patterns to force extreme variance ratios
        for i in range(20):
            # Create entries with different time gaps
            if i % 2 == 0:
                timestamp = base_time + timedelta(seconds=i*1)  # Small gaps for even indices
                success = True
            else:
                timestamp = base_time + timedelta(seconds=i*10)  # Large gaps for odd indices
                success = False
        
            entries.append(RequestEntry(timestamp=timestamp, status_code=200 if success else 429, 
                                      response_time=0.1, success=success))
    
        rate_limiter.entries = entries
        rate_limiter.max_time_period = 10.0
        rate_limiter.regularization_strength = 0.01
        rate_limiter.request_id = "test-id"
        rate_limiter.logger = mocker.MagicMock()
        rate_limiter._update_rate_limit_from_priors = mocker.MagicMock()
    
        # Act - Test with extreme variance ratio that would give weight < 0.3
        rate_limiter._update_bayesian_estimates(limit_exceeded=True)
    
        # Assert - Weight should be bounded to 0.3
        assert rate_limiter.poisson_alpha == 1.3  # 1.0 + 0.3 (bounded minimum)
    
        # Reset for next test
        rate_limiter.poisson_alpha = 1.0
        rate_limiter.beta_b = 1.0
    
        # Patch to force weight > 0.7
        mocker.patch('scipy.stats.variation', side_effect=[1.0, 0.1])  # This would give a very large ratio
    
        # Act - Test with extreme variance ratio that would give weight > 0.7
        rate_limiter._update_bayesian_estimates(limit_exceeded=True)
    
        # Assert - Weight should be bounded to 0.7
        assert rate_limiter.poisson_alpha == 1.7  # 1.0 + 0.7 (bounded maximum)

class Test_InitializeBayesianPriors:

    # Balances prior strength between confidence and adaptability
    def test_prior_strength_balance(self, mocker):
        # Arrange
        mocker.patch('src.smartsurge.models.logger')
        history = RequestHistory()
        history.entries = [{'request_id': '1'}, {'request_id': '2'}]
        history.common_rate_limits = {"default": (100, 60.0)}
        history.request_id = 'test_request'

        # Act
        history._initialize_bayesian_priors()

        # Assert
        assert history.poisson_alpha == 5.0
        assert history.poisson_beta == 5.0 / (100 / 60)
        assert history.beta_a == 100
        assert history.beta_b == 2.0

    # Handles the transition from uninformed to informed priors
    def test_transition_from_uninformed_to_informed_priors(self, mocker):
        # Arrange
        mocker.patch('src.smartsurge.models.logger')
        history = RequestHistory()
        history.entries = [{'request_id': '1'}]  # Less than 2 entries
        history.common_rate_limits = {"default": (100, 60.0)}
        history.request_id = 'test_request'

        # Act
        history._initialize_bayesian_priors()

        # Assert
        history.logger.warning.assert_called_once_with(
            "[test_request] Not enough entries to initialize priors"
        )

    # Initializes Bayesian priors with default rate limit values when entries exist
    def test_initialize_priors_with_entries(self, mocker):
        # Arrange
        mock_logger = mocker.patch('src.smartsurge.models.logger')
        mock_update_rate_limit = mocker.patch.object(RequestHistory, '_update_rate_limit_from_priors')
        history = RequestHistory()
        history.entries = [1, 2, 3]
        history.common_rate_limits = {"default": (100, 60.0)}
        history.request_id = "test_request_id"
    
        # Act
        history._initialize_bayesian_priors()
    
        # Assert
        assert history.poisson_alpha == 5.0
        assert history.poisson_beta == 3.0  # 5.0 / (100/60)
        assert history.beta_a == 100
        assert history.beta_b == 2.0
        mock_logger.debug.assert_called_once()
        mock_update_rate_limit.assert_called_once()

    # Logs a warning and does not initialize priors when there are fewer than two entries
    def test_initialize_priors_with_insufficient_entries(self, mocker):
        # Arrange
        mock_logger = mocker.patch('src.smartsurge.models.logger')
        history = RequestHistory()
        history.entries = [1]
        history.request_id = "test_request_id"
    
        # Act
        history._initialize_bayesian_priors()
    
        # Assert
        mock_logger.warning.assert_called_once_with("[test_request_id] Not enough entries to initialize priors")

    # Initializes Bayesian priors with default rate limit values when entries exist
    def test_initialize_priors_with_entries(self, mocker):
        # Arrange
        mock_logger = mocker.patch('src.smartsurge.models.logger')
        mock_update_rate_limit = mocker.patch.object(RequestHistory, '_update_rate_limit_from_priors')
        history = RequestHistory()
        history.entries = [1, 2, 3]
        history.common_rate_limits = {"default": (100, 60)}
        history.request_id = "test_request_id"
    
        # Act
        history._initialize_bayesian_priors()
    
        # Assert
        assert history.poisson_alpha == 5.0
        assert history.poisson_beta == 3.0  # 5.0 / (100/60)
        assert history.beta_a == 100
        assert history.beta_b == 2.0
        mock_logger.debug.assert_called_once_with(
            "[test_request_id] Initialized Bayesian priors with domain knowledge: "
            "Gamma(5.0000, 3.0000), Beta(100.0000, 2.0000)"
        )
        mock_update_rate_limit.assert_called_once()

    # Logs a warning and does not initialize priors when there are insufficient entries
    def test_insufficient_entries_warning(self, mocker):
        # Arrange
        mock_logger = mocker.patch('src.smartsurge.models.logger')
        history = RequestHistory()
        history.entries = [1]
        history.request_id = "test_request_id"
    
        # Act
        history._initialize_bayesian_priors()
    
        # Assert
        mock_logger.warning.assert_called_once_with(
            "[test_request_id] Not enough entries to initialize priors"
        )

    # Initializes Bayesian priors when there are enough entries
    def test_initialize_bayesian_priors_with_sufficient_entries(self, mocker):
        # Arrange
        mock_logger = mocker.Mock()
        history = RequestHistory(entries=[1, 2, 3], logger=mock_logger)
        history.common_rate_limits = {"default": (60, 60.0)}
    
        # Act
        history._initialize_bayesian_priors()
    
        # Assert
        assert history.poisson_alpha == 5.0
        assert history.poisson_beta == 5.0 / (60 / 60.0)
        assert history.beta_a == 60
        assert history.beta_b == 2.0
        mock_logger.debug.assert_called_once()

    # Does not initialize Bayesian priors when there are insufficient entries
    def test_initialize_bayesian_priors_with_insufficient_entries(self, mocker):
        # Arrange
        mock_logger = mocker.Mock()
        history = RequestHistory(entries=[1], logger=mock_logger)
    
        # Act
        history._initialize_bayesian_priors()
    
        # Assert
        assert history.poisson_alpha is None
        assert history.poisson_beta is None
        assert history.beta_a is None
        assert history.beta_b is None
        mock_logger.warning.assert_called_once()

class Test_ResetPriorsForRestart:

    # Adjusts Bayesian priors by reducing effective sample size when rate limit exists
    def test_reduces_effective_sample_size_when_rate_limit_exists(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="test", method="GET")
        history.rate_limit = RateLimit(requests=100, period=60.0)
        history.poisson_alpha = 10.0
        history.poisson_beta = 2.0
        history._update_rate_limit_from_priors = mocker.MagicMock()
    
        # Act
        history._reset_priors_for_restart()
    
        # Assert
        assert history.poisson_alpha == 10.0 * 0.7
        assert history.poisson_beta < 2.0

    # Preserves mean but increases variance for Poisson parameters
    def test_preserves_poisson_mean_increases_variance(self, mocker):
        import math
        
        # Arrange
        history = RequestHistory(endpoint="test", method="GET")
        history.rate_limit = RateLimit(requests=100, period=60.0)
        history.poisson_alpha = 10.0
        history.poisson_beta = 2.0
        old_mean = history.poisson_alpha / history.poisson_beta
        history._update_rate_limit_from_priors = mocker.MagicMock()
    
        # Act
        history._reset_priors_for_restart()
    
        # Assert
        new_mean = history.poisson_alpha / history.poisson_beta
        assert math.isclose(new_mean, old_mean, rel_tol=1e-9)
        assert history.poisson_alpha < 10.0

    # Preserves mean but increases variance for Beta parameters
    def test_preserves_beta_mean_increases_variance(self, mocker):
        import math
        
        # Arrange
        history = RequestHistory(endpoint="test", method="GET")
        history.rate_limit = RateLimit(requests=100, period=60.0)
        history.beta_a = 8.0
        history.beta_b = 2.0
        old_mean = history.beta_a / (history.beta_a + history.beta_b)
        history._update_rate_limit_from_priors = mocker.MagicMock()
    
        # Act
        history._reset_priors_for_restart()
    
        # Assert
        new_mean = history.beta_a / (history.beta_a + history.beta_b)
        assert math.isclose(new_mean, old_mean, rel_tol=1e-9)
        assert history.beta_a < 8.0
        assert history.beta_b < 2.0

    # Updates rate limit with adjusted priors after restart
    def test_updates_rate_limit_after_restart(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="test", method="GET")
        history.rate_limit = RateLimit(requests=100, period=60.0)
        update_mock = mocker.patch.object(history, '_update_rate_limit_from_priors')
    
        # Act
        history._reset_priors_for_restart()
    
        # Assert
        update_mock.assert_called_once()

    # Initializes Bayesian priors normally when no rate limit exists
    def test_initializes_priors_when_no_rate_limit(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="test", method="GET")
        history.rate_limit = None
        initialize_mock = mocker.patch.object(history, '_initialize_bayesian_priors')
    
        # Act
        history._reset_priors_for_restart()
    
        # Assert
        initialize_mock.assert_called_once()

    # Handles division by zero when calculating old_mean from poisson parameters
    def test_handles_division_by_zero_in_poisson_parameters(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="test", method="GET")
        history.rate_limit = RateLimit(requests=100, period=60.0)
        history.poisson_alpha = 10.0
        history.poisson_beta = 0.0
        history.max_time_period = 3600.0
        history._update_rate_limit_from_priors = mocker.MagicMock()
    
        # Act
        history._reset_priors_for_restart()
    
        # Assert
        assert history.poisson_alpha == 10.0 * 0.7
        # Should use max_time_period when poisson_beta is 0
        assert history.poisson_beta == history.poisson_alpha / history.max_time_period

    # Handles zero total when calculating p_mean from beta parameters
    def test_handles_zero_total_in_beta_parameters(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="test", method="GET")
        history.rate_limit = RateLimit(requests=100, period=60.0)
        history.beta_a = 0.0
        history.beta_b = 0.0
        history._update_rate_limit_from_priors = mocker.MagicMock()
    
        # Act
        history._reset_priors_for_restart()
    
        # Assert
        assert history.beta_a == 0.5
        assert history.beta_b == 0.5

    # Enforces minimum values for poisson_alpha (1.0) and poisson_beta (0.001)
    def test_enforces_minimum_values_for_poisson_parameters(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="test", method="GET")
        history.rate_limit = RateLimit(requests=100, period=60.0)
        history.poisson_alpha = 1.2  # Will be below 1.0 after reduction
        history.poisson_beta = 0.001  # Will be below 0.001 after reduction
        history._update_rate_limit_from_priors = mocker.MagicMock()
    
        # Act
        history._reset_priors_for_restart()
    
        # Assert
        assert history.poisson_alpha == 1.0  # Minimum enforced
        assert history.poisson_beta == 0.001  # Minimum enforced

    # Enforces minimum values for beta_a and beta_b (0.5)
    def test_enforces_minimum_values_for_beta_parameters(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="test", method="GET")
        history.rate_limit = RateLimit(requests=100, period=60.0)
        history.beta_a = 0.6  # Will be below 0.5 after reduction
        history.beta_b = 0.6  # Will be below 0.5 after reduction
        history._update_rate_limit_from_priors = mocker.MagicMock()
    
        # Act
        history._reset_priors_for_restart()
    
        # Assert
        assert history.beta_a == 0.5  # Minimum enforced
        assert history.beta_b == 0.5  # Minimum enforced

    # Handles case when rate_limit is None
    def test_handles_none_rate_limit(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="test", method="GET")
        history.rate_limit = None
        initialize_mock = mocker.patch.object(history, '_initialize_bayesian_priors')
        update_mock = mocker.patch.object(history, '_update_rate_limit_from_priors')
    
        # Act
        history._reset_priors_for_restart()
    
        # Assert
        initialize_mock.assert_called_once()
        update_mock.assert_not_called()

    # Uses uncertainty_factor (0.7) to reduce effective sample size by 30%
    def test_uses_correct_uncertainty_factor(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="test", method="GET")
        history.rate_limit = RateLimit(requests=100, period=60.0)
        history.poisson_alpha = 10.0
        history.poisson_beta = 2.0
        history.beta_a = 8.0
        history.beta_b = 2.0
        history._update_rate_limit_from_priors = mocker.MagicMock()
    
        # Act
        history._reset_priors_for_restart()
    
        # Assert
        assert history.poisson_alpha == 7.0  # 10.0 * 0.7
        assert history.beta_a + history.beta_b == 7.0  # (8.0 + 2.0) * 0.7

    # Logs restart information with uncertainty factor
    def test_logs_restart_information(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="test", method="GET")
        history.rate_limit = RateLimit(requests=100, period=60.0)
        history._update_rate_limit_from_priors = mocker.MagicMock()
        log_mock = mocker.patch.object(history.logger, 'info')
    
        # Act
        history._reset_priors_for_restart()
    
        # Assert
        log_mock.assert_called_once()
        log_call_args = log_mock.call_args[0][0]
        assert "Restarted priors with increased uncertainty" in log_call_args
        assert "factor=0.7" in log_call_args

class TestLogResponseAndUpdate:

    # Adding a successful request entry updates search status from NOT_STARTED to WAITING_FOR_FIRST_REFUSAL
    def test_successful_request_updates_status_from_not_started_to_waiting(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        assert history.search_status == SearchStatus.NOT_STARTED
        entry = RequestEntry(
            endpoint="/api/test",
            method=RequestMethod.GET,
            status_code=200,
            response_time=0.1,
            success=True
        )
    
        # Act
        history.log_response_and_update(entry)
    
        # Assert
        assert history.search_status == SearchStatus.WAITING_FOR_FIRST_REFUSAL

    # Adding a request with status code 429 updates search status from WAITING_FOR_FIRST_REFUSAL to ESTIMATING
    def test_rate_limit_request_updates_status_from_waiting_to_estimating(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history.search_status = SearchStatus.WAITING_FOR_FIRST_REFUSAL
        mocker.patch.object(history, '_update_bayesian_estimates')
        entry = RequestEntry(
            endpoint="/api/test",
            method=RequestMethod.GET,
            status_code=429,
            response_time=0.1,
            success=False
        )
    
        # Act
        history.log_response_and_update(entry)
    
        # Assert
        assert history.search_status == SearchStatus.ESTIMATING
        history._update_bayesian_estimates.assert_called_once_with(True)

    # Adding a successful request during ESTIMATING phase updates Bayesian estimates with _update_bayesian_estimates(False)
    def test_successful_request_during_estimating_updates_bayesian_estimates(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history.search_status = SearchStatus.ESTIMATING
        mocker.patch.object(history, '_update_bayesian_estimates')
        mocker.patch.object(history, '_check_estimation_confidence', return_value=False)
        entry = RequestEntry(
            endpoint="/api/test",
            method=RequestMethod.GET,
            status_code=200,
            response_time=0.1,
            success=True
        )
    
        # Act
        history.log_response_and_update(entry)
    
        # Assert
        history._update_bayesian_estimates.assert_called_once_with(False)
        assert history.search_status == SearchStatus.ESTIMATING

    # Adding a request with status code 429 during ESTIMATING phase updates Bayesian estimates with _update_bayesian_estimates(True)
    def test_rate_limit_request_during_estimating_updates_bayesian_estimates(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history.search_status = SearchStatus.ESTIMATING
        mocker.patch.object(history, '_update_bayesian_estimates')
        mocker.patch.object(history, '_check_estimation_confidence', return_value=False)
        entry = RequestEntry(
            endpoint="/api/test",
            method=RequestMethod.GET,
            status_code=429,
            response_time=0.1,
            success=False
        )
    
        # Act
        history.log_response_and_update(entry)
    
        # Assert
        history._update_bayesian_estimates.assert_called_once_with(True)
        assert history.search_status == SearchStatus.ESTIMATING

    # When confidence threshold is met during ESTIMATING, search status is updated to COMPLETED
    def test_confidence_threshold_met_updates_status_to_completed(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history.search_status = SearchStatus.ESTIMATING
        history.rate_limit = RateLimit(requests=100, period=60.0)
        mocker.patch.object(history, '_update_bayesian_estimates')
        mocker.patch.object(history, '_check_estimation_confidence', return_value=True)
        entry = RequestEntry(
            endpoint="/api/test",
            method=RequestMethod.GET,
            status_code=200,
            response_time=0.1,
            success=True
        )
    
        # Act
        history.log_response_and_update(entry)
    
        # Assert
        assert history.search_status == SearchStatus.COMPLETED
        history._update_bayesian_estimates.assert_called_once_with(False)

    # Receiving status code 429 after search is COMPLETED triggers restart of estimation process
    def test_rate_limit_after_completed_triggers_restart(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history.search_status = SearchStatus.COMPLETED
        mocker.patch.object(history, '_reset_priors_for_restart')
        entry = RequestEntry(
            endpoint="/api/test",
            method=RequestMethod.GET,
            status_code=429,
            response_time=0.1,
            success=False
        )
        initial_restart_count = history.restart_count
    
        # Act
        history.log_response_and_update(entry)
    
        # Assert
        assert history.search_status == SearchStatus.ESTIMATING
        assert history.restart_count == initial_restart_count + 1
        history._reset_priors_for_restart.assert_called_once()

    # First request immediately receives status code 429, skipping WAITING_FOR_FIRST_REFUSAL and going directly to ESTIMATING
    def test_first_request_with_429_skips_waiting_phase(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        assert history.search_status == SearchStatus.NOT_STARTED
        mocker.patch.object(history, '_update_bayesian_estimates')
        entry = RequestEntry(
            endpoint="/api/test",
            method=RequestMethod.GET,
            status_code=429,
            response_time=0.1,
            success=False
        )
    
        # Act
        history.log_response_and_update(entry)
    
        # Assert
        assert history.search_status == SearchStatus.ESTIMATING
        history._update_bayesian_estimates.assert_called_once_with(True)

    # Adding request with status code 429 during COMPLETED state increments restart_count
    def test_rate_limit_during_completed_increments_restart_count(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history.search_status = SearchStatus.COMPLETED
        history.restart_count = 3
        mocker.patch.object(history, '_reset_priors_for_restart')
        entry = RequestEntry(
            endpoint="/api/test",
            method=RequestMethod.GET,
            status_code=429,
            response_time=0.1,
            success=False
        )
    
        # Act
        history.log_response_and_update(entry)
    
        # Assert
        assert history.restart_count == 4

    # Adding request with status code 429 during COMPLETED state resets search status to ESTIMATING
    def test_rate_limit_during_completed_resets_search_status(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history.search_status = SearchStatus.COMPLETED
        mocker.patch.object(history, '_reset_priors_for_restart')
        entry = RequestEntry(
            endpoint="/api/test",
            method=RequestMethod.GET,
            status_code=429,
            response_time=0.1,
            success=False
        )
    
        # Act
        history.log_response_and_update(entry)
    
        # Assert
        assert history.search_status == SearchStatus.ESTIMATING

    # Adding request with status code 429 during COMPLETED state calls _reset_priors_for_restart()
    def test_rate_limit_during_completed_calls_reset_priors(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history.search_status = SearchStatus.COMPLETED
        reset_priors_mock = mocker.patch.object(history, '_reset_priors_for_restart')
        entry = RequestEntry(
            endpoint="/api/test",
            method=RequestMethod.GET,
            status_code=429,
            response_time=0.1,
            success=False
        )
    
        # Act
        history.log_response_and_update(entry)
    
        # Assert
        reset_priors_mock.assert_called_once()

    # Method logs detailed information about state transitions and rate limit estimation progress
    def test_method_logs_detailed_information(self, mocker):
        # Arrange
        mock_logger = mocker.MagicMock()
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET, logger=mock_logger)
        history.search_status = SearchStatus.COMPLETED
        entry = RequestEntry(
            endpoint="/api/test",
            method=RequestMethod.GET,
            status_code=429,
            response_time=0.1,
            success=False
        )
        mocker.patch.object(history, '_reset_priors_for_restart')
    
        # Act
        history.log_response_and_update(entry)
    
        # Assert
        assert mock_logger.debug.call_count >= 1
        assert mock_logger.warning.call_count >= 1
        assert mock_logger.info.call_count >= 1

    # Method maintains consistency between HTTP status codes and search status transitions
    def test_maintains_consistency_between_status_codes_and_transitions(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        mocker.patch.object(history, '_update_bayesian_estimates')
        mocker.patch.object(history, '_check_estimation_confidence', return_value=False)
    
        # Act & Assert - Test consistency across different states
    
        # NOT_STARTED state with 200 status
        history.search_status = SearchStatus.NOT_STARTED
        entry1 = RequestEntry(endpoint="/api/test", method=RequestMethod.GET, status_code=200, response_time=0.1, success=True)
        history.log_response_and_update(entry1)
        assert history.search_status == SearchStatus.WAITING_FOR_FIRST_REFUSAL
    
        # WAITING_FOR_FIRST_REFUSAL state with 429 status
        history.search_status = SearchStatus.WAITING_FOR_FIRST_REFUSAL
        entry2 = RequestEntry(endpoint="/api/test", method=RequestMethod.GET, status_code=429, response_time=0.1, success=False)
        history.log_response_and_update(entry2)
        assert history.search_status == SearchStatus.ESTIMATING
    
        # ESTIMATING state with 200 status
        history.search_status = SearchStatus.ESTIMATING
        entry3 = RequestEntry(endpoint="/api/test", method=RequestMethod.GET, status_code=200, response_time=0.1, success=True)
        history.log_response_and_update(entry3)
        assert history.search_status == SearchStatus.ESTIMATING  # Stays in ESTIMATING until confidence threshold met

    # Method handles all possible SearchStatus enum states (NOT_STARTED, WAITING_FOR_FIRST_REFUSAL, ESTIMATING, COMPLETED)
    def test_search_status_transitions(self, mocker):
        # Arrange
        mock_logger = mocker.Mock()
        history = RequestHistory(endpoint="test_endpoint", method=RequestMethod.GET, logger=mock_logger)
        entry_429 = RequestEntry(endpoint="test_endpoint", method=RequestMethod.GET, status_code=429, response_time=0.1, success=False)
        entry_200 = RequestEntry(endpoint="test_endpoint", method=RequestMethod.GET, status_code=200, response_time=0.1, success=True)
    
        # Act & Assert
        # Initial state: NOT_STARTED
        history.log_response_and_update(entry_200)
        assert history.search_status == SearchStatus.WAITING_FOR_FIRST_REFUSAL
    
        # Transition to WAITING_FOR_FIRST_REFUSAL
        history.log_response_and_update(entry_429)
        assert history.search_status == SearchStatus.ESTIMATING
    
        # Transition to ESTIMATING
        history.log_response_and_update(entry_200)
        assert history.search_status == SearchStatus.ESTIMATING  # Still estimating
    
        # Transition to COMPLETED
        mocker.patch.object(history, '_check_estimation_confidence', return_value=True)
        history.log_response_and_update(entry_200)
        assert history.search_status == SearchStatus.COMPLETED
    
        # Handle COMPLETED state with 429
        history.log_response_and_update(entry_429)
        assert history.search_status == SearchStatus.ESTIMATING  # Restarted estimation

    # Method uses add_request() to store the entry before processing it
    def test_add_request_called(self, mocker):
        # Arrange
        mock_logger = mocker.Mock()
        history = RequestHistory(endpoint="test_endpoint", method=RequestMethod.GET, logger=mock_logger)
        entry = RequestEntry(endpoint="test_endpoint", method=RequestMethod.GET, status_code=200, response_time=0.1, success=True)
    
        # Mock add_request method
        add_request_spy = mocker.spy(history, 'add_request')
    
        # Act
        history.log_response_and_update(entry)
    
        # Assert
        add_request_spy.assert_called_once_with(entry)

    # Method checks estimation confidence after updating Bayesian estimates during ESTIMATING phase
    def test_estimation_confidence_check_during_estimating_phase(self, mocker):
        # Arrange
        mock_logger = mocker.Mock()
        mock_request_entry = mocker.Mock()
        mock_request_entry.status_code = 200
        mock_request_entry.success = True
        mock_request_entry.timestamp = datetime.now(timezone.utc)
    
        history = RequestHistory(endpoint="test_endpoint", method=RequestMethod.GET, logger=mock_logger)
        history.search_status = SearchStatus.ESTIMATING
        history._update_bayesian_estimates = mocker.Mock()
        history._check_estimation_confidence = mocker.Mock(return_value=True)
    
        # Act
        history.log_response_and_update(mock_request_entry)
    
        # Assert
        history._update_bayesian_estimates.assert_called_once_with(False)
        history._check_estimation_confidence.assert_called_once()
        assert history.search_status == SearchStatus.COMPLETED

    # Method updates Bayesian estimates with a failed request during ESTIMATING phase
    def test_update_bayesian_estimates_on_failed_request_during_estimating(self, mocker):
        # Arrange
        mock_logger = mocker.Mock()
        mock_request_entry = mocker.Mock()
        mock_request_entry.status_code = 429
        mock_request_entry.success = False
        mock_request_entry.timestamp = datetime.now(timezone.utc)
    
        history = RequestHistory(endpoint="test_endpoint", method=RequestMethod.GET, logger=mock_logger)
        history.search_status = SearchStatus.ESTIMATING
        history._update_bayesian_estimates = mocker.Mock()
    
        # Act
        history.log_response_and_update(mock_request_entry)
    
        # Assert
        history._update_bayesian_estimates.assert_called_once_with(True)
        
class TestInterceptRequest:

    def test_intercept_request_completed_status_with_rate_limit(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history.search_status = SearchStatus.COMPLETED
        history.rate_limit = RateLimit(requests=100, period=60.0)
        mock_enforce = mocker.patch.object(history, '_enforce_rate_limit')
        
        # Act
        history.intercept_request()
        
        # Assert
        mock_enforce.assert_called_once()

    def test_intercept_request_completed_status_without_rate_limit(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history.search_status = SearchStatus.COMPLETED
        history.rate_limit = None
        mock_enforce = mocker.patch.object(history, '_enforce_rate_limit')
        
        # Act
        history.intercept_request()
        
        # Assert
        mock_enforce.assert_not_called()

    def test_intercept_request_not_started_status_updates_to_waiting(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history.search_status = SearchStatus.NOT_STARTED
        mock_logger = mocker.Mock()
        history.logger = mock_logger
        
        # Act
        history.intercept_request()
        
        # Assert
        assert history.search_status == SearchStatus.WAITING_FOR_FIRST_REFUSAL
        mock_logger.debug.assert_called_with(f"[{history.request_id}] Search status set to WAITING_FOR_FIRST_REFUSAL")

    def test_intercept_request_waiting_status_no_action(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history.search_status = SearchStatus.WAITING_FOR_FIRST_REFUSAL
        mock_enforce = mocker.patch.object(history, '_enforce_rate_limit')
        mock_enforce_bayesian = mocker.patch.object(history, '_enforce_bayesian_estimate')
        
        # Act
        history.intercept_request()
        
        # Assert
        mock_enforce.assert_not_called()
        mock_enforce_bayesian.assert_not_called()

    def test_intercept_request_estimating_status_with_priors(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history.search_status = SearchStatus.ESTIMATING
        history.poisson_alpha = 10.0
        history.poisson_beta = 2.0
        mock_enforce_bayesian = mocker.patch.object(history, '_enforce_bayesian_estimate')
        
        # Act
        history.intercept_request()
        
        # Assert
        mock_enforce_bayesian.assert_called_once()

    def test_intercept_request_estimating_status_without_priors(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history.search_status = SearchStatus.ESTIMATING
        history.poisson_alpha = None
        history.poisson_beta = None
        mock_enforce_bayesian = mocker.patch.object(history, '_enforce_bayesian_estimate')
        
        # Act
        history.intercept_request()
        
        # Assert
        mock_enforce_bayesian.assert_not_called()

    def test_intercept_request_logs_debug_message(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        mock_logger = mocker.Mock()
        history.logger = mock_logger
        
        # Act
        history.intercept_request()
        
        # Assert
        mock_logger.debug.assert_any_call(f"[{history.request_id}] Intercepting request with search status: {history.search_status}")

    def test_intercept_request_estimating_with_partial_priors(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history.search_status = SearchStatus.ESTIMATING
        history.poisson_alpha = 10.0
        history.poisson_beta = None
        mock_enforce_bayesian = mocker.patch.object(history, '_enforce_bayesian_estimate')
        
        # Act
        history.intercept_request()
        
        # Assert
        mock_enforce_bayesian.assert_not_called()

    def test_intercept_request_multiple_status_transitions(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        mock_enforce = mocker.patch.object(history, '_enforce_rate_limit')
        mock_enforce_bayesian = mocker.patch.object(history, '_enforce_bayesian_estimate')
        
        # Act & Assert - Test multiple transitions
        history.search_status = SearchStatus.NOT_STARTED
        history.intercept_request()
        assert history.search_status == SearchStatus.WAITING_FOR_FIRST_REFUSAL
        
        history.intercept_request()
        assert history.search_status == SearchStatus.WAITING_FOR_FIRST_REFUSAL
        
        history.search_status = SearchStatus.ESTIMATING
        history.poisson_alpha = 10.0
        history.poisson_beta = 2.0
        history.intercept_request()
        mock_enforce_bayesian.assert_called_once()
        
        history.search_status = SearchStatus.COMPLETED
        history.rate_limit = RateLimit(requests=100, period=60.0)
        history.intercept_request()
        mock_enforce.assert_called_once()

    def test_intercept_request_preserves_rate_limit_during_transitions(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        rate_limit = RateLimit(requests=100, period=60.0)
        history.rate_limit = rate_limit
        
        # Act
        history.search_status = SearchStatus.NOT_STARTED
        history.intercept_request()
        
        # Assert
        assert history.rate_limit == rate_limit

class TestRequestHistoryMerge:

    def test_merge_combines_entries_in_correct_order(self, mocker):
        # Arrange
        from datetime import datetime, timezone, timedelta
        history1 = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history2 = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        
        now = datetime.now(timezone.utc)
        entry1 = RequestEntry(endpoint="/api/test", method=RequestMethod.GET, timestamp=now - timedelta(seconds=10), status_code=200, response_time=0.1, success=True)
        entry2 = RequestEntry(endpoint="/api/test", method=RequestMethod.GET, timestamp=now - timedelta(seconds=5), status_code=200, response_time=0.1, success=True)
        entry3 = RequestEntry(endpoint="/api/test", method=RequestMethod.GET, timestamp=now, status_code=200, response_time=0.1, success=True)
        
        history1.entries = [entry1, entry3]
        history2.entries = [entry2]
        
        # Act
        history1.merge(history2)
        
        # Assert
        assert len(history1.entries) == 3
        assert history1.entries[0] == entry1
        assert history1.entries[1] == entry2
        assert history1.entries[2] == entry3

    def test_merge_prefers_more_recent_rate_limit(self, mocker):
        # Arrange
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        
        history1 = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history2 = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        
        old_rate_limit = RateLimit(endpoint="/api/test", method=RequestMethod.GET, max_requests=100, time_period=60.0, last_updated=now - timedelta(minutes=5))
        new_rate_limit = RateLimit(endpoint="/api/test", method=RequestMethod.GET, max_requests=50, time_period=30.0, last_updated=now)
        
        history1.rate_limit = old_rate_limit
        history2.rate_limit = new_rate_limit
        
        # Act
        history1.merge(history2)
        
        # Assert
        assert history1.rate_limit == new_rate_limit

    def test_merge_preserves_old_rate_limit_when_newer(self, mocker):
        # Arrange
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        
        history1 = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history2 = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        
        newer_rate_limit = RateLimit(endpoint="/api/test", method=RequestMethod.GET, max_requests=100, time_period=60.0, last_updated=now)
        older_rate_limit = RateLimit(endpoint="/api/test", method=RequestMethod.GET, max_requests=50, time_period=30.0, last_updated=now - timedelta(minutes=5))
        
        history1.rate_limit = newer_rate_limit
        history2.rate_limit = older_rate_limit
        
        # Act
        history1.merge(history2)
        
        # Assert
        assert history1.rate_limit == newer_rate_limit

    def test_merge_prefers_more_informed_bayesian_parameters(self, mocker):
        # Arrange
        history1 = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history2 = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        
        history1.poisson_alpha = 5.0
        history1.poisson_beta = 2.0
        history1.beta_a = 3.0
        history1.beta_b = 1.0
        
        history2.poisson_alpha = 10.0
        history2.poisson_beta = 4.0
        history2.beta_a = 6.0
        history2.beta_b = 2.0
        
        # Act
        history1.merge(history2)
        
        # Assert
        assert history1.poisson_alpha == 10.0
        assert history1.poisson_beta == 4.0
        assert history1.beta_a == 6.0
        assert history1.beta_b == 2.0

    def test_merge_preserves_existing_bayesian_parameters_when_more_informed(self, mocker):
        # Arrange
        history1 = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history2 = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        
        history1.poisson_alpha = 10.0
        history1.poisson_beta = 4.0
        history1.beta_a = 6.0
        history1.beta_b = 2.0
        
        history2.poisson_alpha = 5.0
        history2.poisson_beta = 2.0
        history2.beta_a = 3.0
        history2.beta_b = 1.0
        
        # Act
        history1.merge(history2)
        
        # Assert
        assert history1.poisson_alpha == 10.0
        assert history1.poisson_beta == 4.0
        assert history1.beta_a == 6.0
        assert history1.beta_b == 2.0

    def test_merge_updates_search_status_to_more_advanced(self, mocker):
        # Arrange
        history1 = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history2 = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        
        history1.search_status = SearchStatus.WAITING_FOR_FIRST_REFUSAL
        history2.search_status = SearchStatus.ESTIMATING
        
        # Act
        history1.merge(history2)
        
        # Assert
        assert history1.search_status == SearchStatus.ESTIMATING

    def test_merge_preserves_more_advanced_search_status(self, mocker):
        # Arrange
        history1 = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history2 = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        
        history1.search_status = SearchStatus.COMPLETED
        history2.search_status = SearchStatus.ESTIMATING
        
        # Act
        history1.merge(history2)
        
        # Assert
        assert history1.search_status == SearchStatus.COMPLETED

    def test_merge_takes_higher_restart_count(self, mocker):
        # Arrange
        history1 = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history2 = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        
        history1.restart_count = 3
        history2.restart_count = 5
        
        # Act
        history1.merge(history2)
        
        # Assert
        assert history1.restart_count == 5

    def test_merge_preserves_higher_restart_count(self, mocker):
        # Arrange
        history1 = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history2 = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        
        history1.restart_count = 5
        history2.restart_count = 3
        
        # Act
        history1.merge(history2)
        
        # Assert
        assert history1.restart_count == 5

    def test_merge_uses_stricter_confidence_threshold(self, mocker):
        # Arrange
        history1 = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history2 = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        
        history1.confidence_threshold = 0.9
        history2.confidence_threshold = 0.95
        
        # Act
        history1.merge(history2)
        
        # Assert
        assert history1.confidence_threshold == 0.95

    def test_merge_preserves_stricter_confidence_threshold(self, mocker):
        # Arrange
        history1 = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history2 = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        
        history1.confidence_threshold = 0.95
        history2.confidence_threshold = 0.9
        
        # Act
        history1.merge(history2)
        
        # Assert
        assert history1.confidence_threshold == 0.95

    def test_merge_handles_empty_other_history(self, mocker):
        # Arrange
        history1 = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history2 = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        
        entry = RequestEntry(endpoint="/api/test", method=RequestMethod.GET, timestamp=datetime.now(timezone.utc), status_code=200, response_time=0.1, success=True)
        history1.entries = [entry]
        history1.rate_limit = RateLimit(endpoint="/api/test", method=RequestMethod.GET, max_requests=100, time_period=60.0)
        history1.search_status = SearchStatus.ESTIMATING
        
        # Act
        history1.merge(history2)
        
        # Assert
        assert len(history1.entries) == 1
        assert history1.entries[0] == entry
        assert history1.rate_limit is not None
        assert history1.search_status == SearchStatus.ESTIMATING

    def test_merge_handles_none_rate_limit_in_both(self, mocker):
        # Arrange
        history1 = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        history2 = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        
        # Act
        history1.merge(history2)
        
        # Assert
        assert history1.rate_limit is None

class TestAddRequest:

    def test_add_request_maintains_timestamp_order(self, mocker):
        # Arrange
        from datetime import datetime, timezone, timedelta
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        
        now = datetime.now(timezone.utc)
        entry1 = RequestEntry(endpoint="/api/test", method=RequestMethod.GET, timestamp=now - timedelta(seconds=5), status_code=200, response_time=0.1, success=True)
        entry2 = RequestEntry(endpoint="/api/test", method=RequestMethod.GET, timestamp=now - timedelta(seconds=10), status_code=200, response_time=0.1, success=True)
        entry3 = RequestEntry(endpoint="/api/test", method=RequestMethod.GET, timestamp=now, status_code=200, response_time=0.1, success=True)
        
        # Act
        history.add_request(entry1)
        history.add_request(entry2)
        history.add_request(entry3)
        
        # Assert
        assert len(history.entries) == 3
        assert history.entries[0] == entry2  # Oldest timestamp first
        assert history.entries[1] == entry1
        assert history.entries[2] == entry3  # Most recent timestamp last

    def test_add_request_with_identical_timestamps(self, mocker):
        # Arrange
        from datetime import datetime, timezone
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        
        timestamp = datetime.now(timezone.utc)
        entry1 = RequestEntry(endpoint="/api/test", method=RequestMethod.GET, timestamp=timestamp, status_code=200, response_time=0.1, success=True)
        entry2 = RequestEntry(endpoint="/api/test", method=RequestMethod.GET, timestamp=timestamp, status_code=429, response_time=0.2, success=False)
        
        # Act
        history.add_request(entry1)
        history.add_request(entry2)
        
        # Assert
        assert len(history.entries) == 2
        assert history.entries[0].timestamp == history.entries[1].timestamp

    def test_add_request_with_microsecond_precision(self, mocker):
        # Arrange
        from datetime import datetime, timezone, timedelta
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        
        base_time = datetime.now(timezone.utc)
        entry1 = RequestEntry(endpoint="/api/test", method=RequestMethod.GET, timestamp=base_time + timedelta(microseconds=100), status_code=200, response_time=0.1, success=True)
        entry2 = RequestEntry(endpoint="/api/test", method=RequestMethod.GET, timestamp=base_time + timedelta(microseconds=50), status_code=200, response_time=0.1, success=True)
        
        # Act
        history.add_request(entry1)
        history.add_request(entry2)
        
        # Assert
        assert len(history.entries) == 2
        assert history.entries[0] == entry2
        assert history.entries[1] == entry1

    def test_add_request_with_method_case_sensitivity(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        entry = RequestEntry(endpoint="/api/test", method="get", status_code=200, response_time=0.1, success=True)
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            history.add_request(entry)
        assert "Entry endpoint and method must match RequestHistory" in str(exc_info.value)

    def test_add_request_with_trailing_slashes(self, mocker):
        # Arrange
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        entry = RequestEntry(endpoint="/api/test/", method=RequestMethod.GET, status_code=200, response_time=0.1, success=True)
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            history.add_request(entry)
        assert "Entry endpoint and method must match RequestHistory" in str(exc_info.value)

    def test_add_request_logs_debug_message(self, mocker):
        # Arrange
        mock_logger = mocker.Mock()
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET, logger=mock_logger)
        entry = RequestEntry(endpoint="/api/test", method=RequestMethod.GET, status_code=200, response_time=0.1, success=True)
        
        # Act
        history.add_request(entry)
        
        # Assert
        mock_logger.debug.assert_called_once()
        debug_message = mock_logger.debug.call_args[0][0]
        assert "/api/test" in debug_message
        assert "GET" in debug_message

    def test_add_request_logs_error_on_mismatch(self, mocker):
        # Arrange
        mock_logger = mocker.Mock()
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET, logger=mock_logger)
        entry = RequestEntry(endpoint="/api/wrong", method=RequestMethod.GET, status_code=200, response_time=0.1, success=True)
        
        # Act & Assert
        with pytest.raises(ValueError):
            history.add_request(entry)
        
        mock_logger.error.assert_called_once()
        error_message = mock_logger.error.call_args[0][0]
        assert "mismatch" in error_message
        assert "/api/wrong" in error_message
        assert "/api/test" in error_message

class TestRequestHistoryInit:

    def test_logger_initialization_with_custom_logger(self, mocker):
        # Arrange
        mock_logger = mocker.Mock()
        
        # Act
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET, logger=mock_logger)
        
        # Assert
        assert history.logger == mock_logger

    def test_logger_initialization_with_default_logger(self, mocker):
        # Arrange
        mock_child_logger = mocker.Mock()
        mock_logger = mocker.Mock()
        mock_logger.getChild.return_value = mock_child_logger
        mocker.patch('src.smartsurge.models.logger', mock_logger)
        
        # Act
        history = RequestHistory(endpoint="/api/test", method=RequestMethod.GET)
        
        # Assert
        mock_logger.getChild.assert_called_once_with("RequestHistory./api/test.GET")
        assert history.logger == mock_child_logger

    def test_logger_initialization_with_unicode_endpoint(self, mocker):
        # Arrange
        mock_child_logger = mocker.Mock()
        mock_logger = mocker.Mock()
        mock_logger.getChild.return_value = mock_child_logger
        mocker.patch('src.smartsurge.models.logger', mock_logger)
        
        # Act
        history = RequestHistory(endpoint="/api/test/ünicode", method=RequestMethod.GET)
        
        # Assert
        mock_logger.getChild.assert_called_once_with("RequestHistory./api/test/ünicode.GET")
        assert history.logger == mock_child_logger

    def test_bayesian_priors_initialization_when_partially_set(self, mocker):
        # Arrange
        mock_initialize = mocker.patch.object(RequestHistory, '_initialize_bayesian_priors')
        
        # Act
        history = RequestHistory(
            endpoint="/api/test",
            method=RequestMethod.GET,
            poisson_alpha=10.0,
            poisson_beta=None,
            beta_a=5.0,
            beta_b=2.0
        )
        
        # Assert
        mock_initialize.assert_called_once()

    def test_bayesian_priors_not_initialized_when_all_set(self, mocker):
        # Arrange
        mock_initialize = mocker.patch.object(RequestHistory, '_initialize_bayesian_priors')
        
        # Act
        history = RequestHistory(
            endpoint="/api/test",
            method=RequestMethod.GET,
            poisson_alpha=10.0,
            poisson_beta=2.0,
            beta_a=5.0,
            beta_b=2.0
        )
        
        # Assert
        mock_initialize.assert_not_called()

    def test_initialization_with_special_characters_in_endpoint(self, mocker):
        # Arrange
        mock_child_logger = mocker.Mock()
        mock_logger = mocker.Mock()
        mock_logger.getChild.return_value = mock_child_logger
        mocker.patch('src.smartsurge.models.logger', mock_logger)
        
        # Act
        history = RequestHistory(endpoint="/api/test!@#$%^&*()", method=RequestMethod.GET)
        
        # Assert
        mock_logger.getChild.assert_called_once_with("RequestHistory./api/test!@#$%^&*().GET")
        assert history.logger == mock_child_logger

    def test_initialization_with_very_long_endpoint(self, mocker):
        # Arrange
        mock_child_logger = mocker.Mock()
        mock_logger = mocker.Mock()
        mock_logger.getChild.return_value = mock_child_logger
        mocker.patch('src.smartsurge.models.logger', mock_logger)
        long_endpoint = "/api/" + "x" * 1000
        
        # Act
        history = RequestHistory(endpoint=long_endpoint, method=RequestMethod.GET)
        
        # Assert
        mock_logger.getChild.assert_called_once_with(f"RequestHistory.{long_endpoint}.GET")
        assert history.logger == mock_child_logger

    def test_initialization_with_all_request_methods(self, mocker):
        # Arrange
        mock_child_logger = mocker.Mock()
        mock_logger = mocker.Mock()
        mock_logger.getChild.return_value = mock_child_logger
        mocker.patch('src.smartsurge.models.logger', mock_logger)
        
        # Act & Assert
        for method in RequestMethod:
            history = RequestHistory(endpoint="/api/test", method=method)
            mock_logger.getChild.assert_called_with(f"RequestHistory./api/test.{method}")
            assert history.logger == mock_child_logger
            mock_logger.getChild.reset_mock()

    def test_initialization_with_empty_endpoint(self, mocker):
        # Arrange
        mock_child_logger = mocker.Mock()
        mock_logger = mocker.Mock()
        mock_logger.getChild.return_value = mock_child_logger
        mocker.patch('src.smartsurge.models.logger', mock_logger)
        
        # Act & Assert
        with pytest.raises(ValidationError):
            RequestHistory(endpoint="", method=RequestMethod.GET)
