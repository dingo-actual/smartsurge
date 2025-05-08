"""
Tests for the SmartSurge utility functions.

This module contains tests for:
- SmartSurgeTimer
- log_context
- async_request_with_history
- merge_histories
"""

import pytest
import logging
import time
import asyncio
from datetime import datetime, timezone, timedelta

from smartsurge.utilities import (
    SmartSurgeTimer, 
    log_context, 
    async_request_with_history, 
    merge_histories
)
from smartsurge.models import RequestHistory, RequestMethod, RateLimit, RequestEntry


class Test_SmartSurgeTimer_Init_01_NominalBehaviors:
    """Tests for nominal behaviors of SmartSurgeTimer.__init__"""

    def test_init_with_valid_operation_name_and_logger(self):
        """Test initialization with a valid operation name and logger."""
        # Arrange
        operation_name = "test_operation"
        test_logger = logging.getLogger("test_logger")
        
        # Act
        timer = SmartSurgeTimer(operation_name, test_logger)
        
        # Assert
        assert timer.operation_name == operation_name
        assert timer.logger == test_logger
        assert timer.start_time is None

    def test_init_with_default_logger(self):
        """Test initialization with a default logger when none is provided."""
        # Arrange
        operation_name = "test_operation"
        
        # Act
        timer = SmartSurgeTimer(operation_name)
        
        # Assert
        assert timer.operation_name == operation_name
        assert timer.logger is not None
        assert timer.start_time is None

class Test_SmartSurgeTimer_Init_02_NegativeBehaviors:
    """Tests for negative behaviors of SmartSurgeTimer.__init__"""

    def test_init_with_non_string_operation_name(self):
        """Test initialization with a non-string operation name."""
        # Arrange & Act
        timer = SmartSurgeTimer(123)  # Not a string
        
        # Assert
        assert timer.operation_name == 123
        assert timer.start_time is None

class Test_SmartSurgeTimer_Init_03_BoundaryBehaviors:
    """Tests for boundary behaviors of SmartSurgeTimer.__init__"""

    def test_init_with_empty_string_operation_name(self):
        """Test initialization with an empty string as operation name."""
        # Arrange & Act
        timer = SmartSurgeTimer("")
        
        # Assert
        assert timer.operation_name == ""
        assert timer.start_time is None

    def test_init_with_very_long_operation_name(self):
        """Test initialization with a very long operation name."""
        # Arrange & Act
        operation_name = "a" * 10000  # Very long name
        timer = SmartSurgeTimer(operation_name)
        
        # Assert
        assert timer.operation_name == operation_name
        assert timer.start_time is None

class Test_SmartSurgeTimer_Init_05_StateTransitionBehaviors:
    """Tests for state transition behaviors of SmartSurgeTimer.__init__"""

    def test_initial_state_setup(self):
        """Test that the initial state has start_time set to None."""
        # Arrange & Act
        timer = SmartSurgeTimer("test_operation")
        
        # Assert
        assert timer.start_time is None

class Test_SmartSurgeTimer_Enter_01_NominalBehaviors:
    """Tests for nominal behaviors of SmartSurgeTimer.__enter__"""

    def test_records_start_time(self):
        """Test that __enter__ records the current time as start_time."""
        # Arrange
        timer = SmartSurgeTimer("test_operation")
        
        # Act
        result = timer.__enter__()
        
        # Assert
        assert timer.start_time is not None
        assert isinstance(timer.start_time, float)
        assert timer.start_time <= time.time()  # Should be now or in the past

    def test_returns_self(self):
        """Test that __enter__ returns self for use in the context manager."""
        # Arrange
        timer = SmartSurgeTimer("test_operation")
        
        # Act
        result = timer.__enter__()
        
        # Assert
        assert result is timer

class Test_SmartSurgeTimer_Enter_05_StateTransitionBehaviors:
    """Tests for state transition behaviors of SmartSurgeTimer.__enter__"""

    def test_transitions_to_active_timing_state(self):
        """Test transition from uninitialized timer to active timing state."""
        # Arrange
        timer = SmartSurgeTimer("test_operation")
        assert timer.start_time is None  # Verify initial state
        
        # Act
        timer.__enter__()
        
        # Assert
        assert timer.start_time is not None  # State has changed

class Test_SmartSurgeTimer_Exit_01_NominalBehaviors:
    """Tests for nominal behaviors of SmartSurgeTimer.__exit__"""

    def test_calculates_elapsed_time_correctly(self):
        """Test that __exit__ calculates elapsed time correctly."""
        # Arrange
        timer = SmartSurgeTimer("test_operation")
        timer.__enter__()
        time.sleep(0.01)  # Small delay to ensure elapsed time > 0
        
        # Act
        result = timer.__exit__(None, None, None)
        
        # Assert
        assert result is False  # Should not suppress exceptions

    def test_logs_operation_with_elapsed_time(self, caplog):
        """Test that __exit__ logs the operation name with elapsed time."""
        # Arrange
        caplog.set_level(logging.DEBUG)
        operation_name = "test_operation"
        timer = SmartSurgeTimer(operation_name)
        timer.__enter__()
        
        # Act
        timer.__exit__(None, None, None)
        
        # Assert
        assert any(f"Operation '{operation_name}' completed in" in record.message for record in caplog.records)
        assert any("seconds" in record.message for record in caplog.records)

    def test_returns_false_to_not_suppress_exceptions(self):
        """Test that __exit__ returns False to not suppress exceptions."""
        # Arrange
        timer = SmartSurgeTimer("test_operation")
        timer.__enter__()
        
        # Act
        result = timer.__exit__(Exception, Exception("Test"), None)
        
        # Assert
        assert result is False  # Should not suppress exceptions

class Test_SmartSurgeTimer_Exit_02_NegativeBehaviors:
    """Tests for negative behaviors of SmartSurgeTimer.__exit__"""

    def test_handles_none_start_time(self, caplog):
        """Test that __exit__ handles the case when start_time is None."""
        # Arrange
        caplog.set_level(logging.DEBUG)
        timer = SmartSurgeTimer("test_operation")
        # Deliberately not calling __enter__()
        
        # Act
        result = timer.__exit__(None, None, None)
        
        # Assert
        assert result is False

class Test_SmartSurgeTimer_Exit_03_BoundaryBehaviors:
    """Tests for boundary behaviors of SmartSurgeTimer.__exit__"""

    def test_handles_very_short_execution_time(self, caplog):
        """Test that __exit__ properly handles very short execution times."""
        # Arrange
        caplog.set_level(logging.DEBUG)
        timer = SmartSurgeTimer("test_operation")
        timer.__enter__()
        # No sleep to test near-zero execution time
        
        # Act
        timer.__exit__(None, None, None)
        
        # Assert
        assert any("completed in" in record.message for record in caplog.records)

    def test_handles_long_execution_time(self, caplog):
        """Test that __exit__ properly handles long execution times."""
        # Arrange
        caplog.set_level(logging.DEBUG)
        timer = SmartSurgeTimer("test_operation")
        # Manually set start_time to simulate long execution
        timer.__enter__()
        timer.start_time = time.time() - 3600  # Simulate 1 hour execution
        
        # Act
        timer.__exit__(None, None, None)
        
        # Assert
        assert any("completed in" in record.message for record in caplog.records)

class Test_SmartSurgeTimer_Exit_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of SmartSurgeTimer.__exit__"""

    def test_functions_correctly_with_exceptions(self, caplog):
        """Test that __exit__ functions correctly when exceptions occur within the context."""
        # Arrange
        caplog.set_level(logging.DEBUG)
        timer = SmartSurgeTimer("test_operation")
        timer.__enter__()
        exc_type = ValueError
        exc_val = ValueError("Test exception")
        exc_tb = None
        
        # Act
        result = timer.__exit__(exc_type, exc_val, exc_tb)
        
        # Assert
        assert result is False  # Should not suppress exceptions
        assert any("completed in" in record.message for record in caplog.records)

class Test_LogContext_01_NominalBehaviors:
    """Tests for nominal behaviors of log_context function"""

    def test_creates_log_entries(self, caplog):
        """Test that log_context creates log entries with start and completion messages."""
        # Arrange
        caplog.set_level(logging.DEBUG)
        context_name = "test_context"
        
        # Act
        with log_context(context_name):
            pass  # Do nothing in the context
        
        # Assert
        assert any(f"Starting {context_name}" in record.message for record in caplog.records)
        assert any(f"Completed {context_name}" in record.message for record in caplog.records)

    def test_calculates_elapsed_time(self, caplog):
        """Test that log_context correctly calculates and includes elapsed time in completion log."""
        # Arrange
        caplog.set_level(logging.DEBUG)
        context_name = "test_context"
        
        # Act
        with log_context(context_name):
            time.sleep(0.01)  # Small delay to ensure elapsed time > 0
        
        # Assert
        assert any(f"Completed {context_name}" in record.message and "s" in record.message for record in caplog.records)

    def test_generates_unique_context_id(self, caplog):
        """Test that log_context generates a unique context ID when include_id is True."""
        # Arrange
        caplog.set_level(logging.DEBUG)
        context_name = "test_context"
        
        # Act
        with log_context(context_name, include_id=True):
            pass
        
        # Assert
        for record in caplog.records:
            if f"Starting {context_name}" in record.message:
                assert "[" in record.message and "]" in record.message

class Test_LogContext_02_NegativeBehaviors:
    """Tests for negative behaviors of log_context function"""

    def test_handles_invalid_logger(self, caplog):
        """Test that log_context handles invalid logger parameter by creating a default logger."""
        # Arrange
        caplog.set_level(logging.DEBUG)
        context_name = "test_context"
        invalid_logger = "not_a_logger"  # Not a valid logger
        
        # Act & Assert (should not raise exception)
        with log_context(context_name, logger=invalid_logger):
            pass

class Test_LogContext_03_BoundaryBehaviors:
    """Tests for boundary behaviors of log_context function"""

    def test_works_with_different_logging_levels(self, caplog):
        """Test that log_context works with different logging levels."""
        # Arrange
        caplog.set_level(logging.INFO)  # Set level to INFO
        context_name = "test_context"
        
        # Act - use DEBUG level (lower than current level)
        with log_context(context_name, level=logging.DEBUG):
            pass
        
        # Assert - no messages should be captured at DEBUG level when level is INFO
        assert not any(f"Starting {context_name}" in record.message for record in caplog.records)
        
        # Act - use INFO level (same as current level)
        with log_context(context_name, level=logging.INFO):
            pass
        
        # Assert - messages should be captured at INFO level
        assert any(f"Starting {context_name}" in record.message for record in caplog.records)

class Test_LogContext_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of log_context function"""

    def test_catches_and_logs_exceptions(self, caplog):
        """Test that log_context catches and logs exceptions that occur within the context."""
        # Arrange
        caplog.set_level(logging.DEBUG)
        context_name = "test_context"
        
        # Act & Assert
        with pytest.raises(ValueError):
            with log_context(context_name):
                raise ValueError("Test exception")
        
        # Assert that error was logged
        assert any(f"Error in {context_name}" in record.message for record in caplog.records)
        assert any("Test exception" in record.message for record in caplog.records)

    def test_reraises_exceptions(self):
        """Test that log_context re-raises exceptions after logging them."""
        # Arrange
        context_name = "test_context"
        
        # Act & Assert
        with pytest.raises(ValueError) as excinfo:
            with log_context(context_name):
                raise ValueError("Test exception")
        
        # Assert exception is re-raised properly
        assert "Test exception" in str(excinfo.value)

    def test_includes_elapsed_time_in_error_log(self, caplog):
        """Test that log_context includes elapsed time even when exceptions occur."""
        # Arrange
        caplog.set_level(logging.DEBUG)
        context_name = "test_context"
        
        # Act
        try:
            with log_context(context_name):
                time.sleep(0.01)  # Small delay
                raise ValueError("Test exception")
        except ValueError:
            pass  # Catch the exception to continue the test
        
        # Assert
        for record in caplog.records:
            if "Error in" in record.message:
                assert "after" in record.message
                assert "s:" in record.message  # Check for elapsed time format

class Test_LogContext_05_StateTransitionBehaviors:
    """Tests for state transition behaviors of log_context function"""

    def test_transitions_from_starting_to_execution(self, caplog):
        """Test transition from starting to execution state when yielding."""
        # Arrange
        caplog.set_level(logging.DEBUG)
        context_name = "test_context"
        execution_flag = False
        
        # Act
        with log_context(context_name):
            # Check that "Starting" message is already logged
            assert any(f"Starting {context_name}" in record.message for record in caplog.records)
            # Mark that we're in execution state
            execution_flag = True
        
        # Assert
        assert execution_flag  # Verify we reached execution state

    def test_transitions_to_completed_state(self, caplog):
        """Test transition from execution to completed state after context block finishes."""
        # Arrange
        caplog.set_level(logging.DEBUG)
        context_name = "test_context"
        
        # Act
        with log_context(context_name):
            pass
        
        # Assert
        # Check transition to completed state
        assert any(f"Completed {context_name}" in record.message for record in caplog.records)
        
    def test_transitions_to_error_state(self, caplog):
        """Test transition from execution to error state on exception."""
        # Arrange
        caplog.set_level(logging.DEBUG)
        context_name = "test_context"
        
        # Act
        try:
            with log_context(context_name):
                raise ValueError("Test exception")
        except ValueError:
            pass  # Catch the exception to continue the test
        
        # Assert
        # Check transition to error state
        assert any(f"Error in {context_name}" in record.message for record in caplog.records)

class Test_AsyncRequestWithHistory_01_NominalBehaviors:
    """Tests for nominal behaviors of async_request_with_history function"""

    @pytest.mark.asyncio
    async def test_limits_concurrent_requests(self):
        """Test that async_request_with_history limits concurrent requests using semaphore."""
        # Arrange
        endpoints = [f"endpoint_{i}" for i in range(10)]
        max_concurrent = 2
        active_requests = 0
        max_active_requests = 0
        
        async def mock_request_func(**kwargs):
            nonlocal active_requests, max_active_requests
            active_requests += 1
            max_active_requests = max(max_active_requests, active_requests)
            await asyncio.sleep(0.05)  # Small delay
            active_requests -= 1
            return "response", RequestHistory(
                endpoint=kwargs['endpoint'],
                method=kwargs['method'],
                min_time_period=0.1,
                max_time_period=1.0,
                confidence_threshold=0.9
            )
        
        # Act
        await async_request_with_history(
            request_func=mock_request_func,
            endpoints=endpoints,
            method=RequestMethod.GET,
            max_concurrent=max_concurrent
        )
        
        # Assert
        assert max_active_requests <= max_concurrent

    @pytest.mark.asyncio
    async def test_makes_requests_to_all_endpoints(self):
        """Test that async_request_with_history makes requests to all endpoints in the list."""
        # Arrange
        endpoints = [f"endpoint_{i}" for i in range(5)]
        requested_endpoints = []
        
        async def mock_request_func(**kwargs):
            requested_endpoints.append(kwargs['endpoint'])
            return "response", RequestHistory(
                endpoint=kwargs['endpoint'],
                method=kwargs['method'],
                min_time_period=0.1,
                max_time_period=1.0,
                confidence_threshold=0.9
            )
        
        # Act
        await async_request_with_history(
            request_func=mock_request_func,
            endpoints=endpoints,
            method=RequestMethod.GET
        )
        
        # Assert
        assert sorted(requested_endpoints) == sorted(endpoints)

    @pytest.mark.asyncio
    async def test_returns_responses_and_histories(self):
        """Test that async_request_with_history returns both responses and consolidated histories."""
        # Arrange
        endpoints = ["endpoint_1", "endpoint_2"]
        
        async def mock_request_func(**kwargs):
            return f"response_{kwargs['endpoint']}", RequestHistory(
                endpoint=kwargs['endpoint'],
                method=kwargs['method'],
                min_time_period=0.1,
                max_time_period=1.0,
                confidence_threshold=0.9
            )
        
        # Act
        responses, histories = await async_request_with_history(
            request_func=mock_request_func,
            endpoints=endpoints,
            method=RequestMethod.GET
        )
        
        # Assert
        assert len(responses) == len(endpoints)
        assert len(histories) == len(endpoints)
        assert all(isinstance(h, RequestHistory) for h in histories.values())
        assert all(f"response_{endpoint}" in responses for endpoint in endpoints)

    @pytest.mark.asyncio
    async def test_preserves_request_order(self):
        """Test that async_request_with_history preserves request order in returned responses."""
        # Arrange
        endpoints = [f"endpoint_{i}" for i in range(5)]
        
        async def mock_request_func(**kwargs):
            # Simulate non-deterministic completion times
            await asyncio.sleep(0.01 * (5 - int(kwargs['endpoint'].split('_')[1])))
            return f"response_{kwargs['endpoint']}", RequestHistory(
                endpoint=kwargs['endpoint'],
                method=kwargs['method'],
                min_time_period=0.1,
                max_time_period=1.0,
                confidence_threshold=0.9
            )
        
        # Act
        responses, _ = await async_request_with_history(
            request_func=mock_request_func,
            endpoints=endpoints,
            method=RequestMethod.GET
        )
        
        # Assert
        for i, endpoint in enumerate(endpoints):
            assert responses[i] == f"response_{endpoint}"

    @pytest.mark.asyncio
    async def test_passes_additional_kwargs(self):
        """Test that async_request_with_history correctly passes additional kwargs to request function."""
        # Arrange
        endpoints = ["endpoint_1"]
        received_kwargs = {}
        
        async def mock_request_func(**kwargs):
            # Store all kwargs for later inspection
            nonlocal received_kwargs
            received_kwargs = kwargs
            return "response", RequestHistory(
                endpoint=kwargs['endpoint'],
                method=kwargs['method'],
                min_time_period=0.1,
                max_time_period=1.0,
                confidence_threshold=0.9
            )
        
        # Act
        await async_request_with_history(
            request_func=mock_request_func,
            endpoints=endpoints,
            method=RequestMethod.GET,
            additional_param="value",
            another_param=123
        )
        
        # Assert
        assert received_kwargs["additional_param"] == "value"
        assert received_kwargs["another_param"] == 123

class Test_AsyncRequestWithHistory_02_NegativeBehaviors:
    """Tests for negative behaviors of async_request_with_history function"""

    @pytest.mark.asyncio
    async def test_handles_request_func_failures(self):
        """Test that async_request_with_history handles failures in the request_func."""
        # Arrange
        endpoints = ["success_endpoint", "error_endpoint"]
        
        async def mock_request_func(**kwargs):
            if kwargs['endpoint'] == "error_endpoint":
                raise ValueError("Test error")
            return "success", RequestHistory(
                endpoint=kwargs['endpoint'],
                method=kwargs['method'],
                min_time_period=0.1,
                max_time_period=1.0,
                confidence_threshold=0.9
            )
        
        # Act
        responses, histories = await async_request_with_history(
            request_func=mock_request_func,
            endpoints=endpoints,
            method=RequestMethod.GET
        )
        
        # Assert
        assert len(responses) == 2
        assert isinstance(responses[0], str)  # Success case
        assert isinstance(responses[1], ValueError)  # Error case
        assert "success_endpoint" in histories
        assert "error_endpoint" not in histories

    @pytest.mark.asyncio
    async def test_includes_exceptions_in_response_list(self):
        """Test that async_request_with_history includes exceptions in the response list when requests fail."""
        # Arrange
        endpoints = ["endpoint_1", "endpoint_2", "endpoint_3"]
        
        async def mock_request_func(**kwargs):
            if kwargs['endpoint'] == "endpoint_2":
                raise ValueError("Test error")
            return "success", RequestHistory(
                endpoint=kwargs['endpoint'],
                method=kwargs['method'],
                min_time_period=0.1,
                max_time_period=1.0,
                confidence_threshold=0.9
            )
        
        # Act
        responses, _ = await async_request_with_history(
            request_func=mock_request_func,
            endpoints=endpoints,
            method=RequestMethod.GET
        )
        
        # Assert
        assert len(responses) == 3
        assert responses[0] == "success"
        assert isinstance(responses[1], ValueError)
        assert responses[2] == "success"

class Test_AsyncRequestWithHistory_03_BoundaryBehaviors:
    """Tests for boundary behaviors of async_request_with_history function"""

    @pytest.mark.asyncio
    async def test_works_with_max_concurrent_set_to_one(self):
        """Test that async_request_with_history works with max_concurrent set to 1 (sequential processing)."""
        # Arrange
        endpoints = [f"endpoint_{i}" for i in range(3)]
        active_requests = 0
        max_active_requests = 0
        
        async def mock_request_func(**kwargs):
            nonlocal active_requests, max_active_requests
            active_requests += 1
            max_active_requests = max(max_active_requests, active_requests)
            await asyncio.sleep(0.01)
            active_requests -= 1
            return "response", RequestHistory(
                endpoint=kwargs['endpoint'],
                method=kwargs['method'],
                min_time_period=0.1,
                max_time_period=1.0,
                confidence_threshold=0.9
            )
        
        # Act
        await async_request_with_history(
            request_func=mock_request_func,
            endpoints=endpoints,
            method=RequestMethod.GET,
            max_concurrent=1  # Only one concurrent request
        )
        
        # Assert
        assert max_active_requests == 1

    @pytest.mark.asyncio
    async def test_works_with_max_concurrent_matching_endpoints(self):
        """Test that async_request_with_history works with max_concurrent set to match number of endpoints (fully parallel)."""
        # Arrange
        endpoints = [f"endpoint_{i}" for i in range(3)]
        active_requests = 0
        max_active_requests = 0
        
        async def mock_request_func(**kwargs):
            nonlocal active_requests, max_active_requests
            active_requests += 1
            max_active_requests = max(max_active_requests, active_requests)
            await asyncio.sleep(0.01)
            active_requests -= 1
            return "response", RequestHistory(
                endpoint=kwargs['endpoint'],
                method=kwargs['method'],
                min_time_period=0.1,
                max_time_period=1.0,
                confidence_threshold=0.9
            )
        
        # Act
        await async_request_with_history(
            request_func=mock_request_func,
            endpoints=endpoints,
            method=RequestMethod.GET,
            max_concurrent=len(endpoints)  # Match number of endpoints
        )
        
        # Assert
        assert max_active_requests == len(endpoints)

class Test_AsyncRequestWithHistory_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of async_request_with_history function"""

    @pytest.mark.asyncio
    async def test_logs_errors_when_requests_fail(self, caplog):
        """Test that async_request_with_history logs errors when requests fail."""
        # Arrange
        caplog.set_level(logging.ERROR)
        endpoints = ["success_endpoint", "error_endpoint"]
        
        async def mock_request_func(**kwargs):
            if kwargs['endpoint'] == "error_endpoint":
                raise ValueError("Test error")
            return "success", RequestHistory(
                endpoint=kwargs['endpoint'],
                method=kwargs['method'],
                min_time_period=0.1,
                max_time_period=1.0,
                confidence_threshold=0.9
            )
        
        # Act
        await async_request_with_history(
            request_func=mock_request_func,
            endpoints=endpoints,
            method=RequestMethod.GET
        )
        
        # Assert
        assert any("error_endpoint" in record.message for record in caplog.records)
        assert any("failed" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_continues_processing_when_some_requests_fail(self):
        """Test that async_request_with_history continues processing other requests when some fail."""
        # Arrange
        endpoints = ["endpoint_1", "error_endpoint", "endpoint_3"]
        processed_endpoints = []
        
        async def mock_request_func(**kwargs):
            if kwargs['endpoint'] == "error_endpoint":
                raise ValueError("Test error")
            processed_endpoints.append(kwargs['endpoint'])
            return "success", RequestHistory(
                endpoint=kwargs['endpoint'],
                method=kwargs['method'],
                min_time_period=0.1,
                max_time_period=1.0,
                confidence_threshold=0.9
            )
        
        # Act
        await async_request_with_history(
            request_func=mock_request_func,
            endpoints=endpoints,
            method=RequestMethod.GET
        )
        
        # Assert
        assert "endpoint_1" in processed_endpoints
        assert "endpoint_3" in processed_endpoints
        assert len(processed_endpoints) == 2

class Test_AsyncRequestWithHistory_05_StateTransitionBehaviors:
    """Tests for state transition behaviors of async_request_with_history function"""

    @pytest.mark.asyncio
    async def test_consolidates_histories_for_same_endpoint(self):
        """Test that async_request_with_history consolidates histories for the same endpoint."""
        # Arrange
        endpoints = ["endpoint_1", "endpoint_1"]  # Same endpoint twice
        
        async def mock_request_func(**kwargs):
            history = RequestHistory(
                endpoint=kwargs['endpoint'],
                method=kwargs['method'],
                min_time_period=0.1,
                max_time_period=1.0,
                confidence_threshold=0.9
            )
            # Add an entry to the history
            history.entries.append(RequestEntry(
                timestamp=datetime.now(timezone.utc),
                endpoint=kwargs['endpoint'],
                method=kwargs['method'],
                status_code=200,
                response_time=0.1,
                success=True
            ))
            return "response", history
        
        # Act
        _, histories = await async_request_with_history(
            request_func=mock_request_func,
            endpoints=endpoints,
            method=RequestMethod.GET
        )
        
        # Assert
        assert len(histories) == 1  # Only one endpoint key
        assert len(histories["endpoint_1"].entries) == 2  # Two entries

    @pytest.mark.asyncio
    async def test_updates_rate_limit_based_on_responses(self):
        """Test that async_request_with_history updates rate limit understanding based on responses."""
        # Arrange
        endpoints = ["endpoint_1"]
        rate_limit = RateLimit(
            endpoint="endpoint_1",
            method=RequestMethod.GET,
            max_requests=10,
            time_period=1.0,
            last_updated=datetime.now(timezone.utc)
        )
        
        async def mock_request_func(**kwargs):
            history = RequestHistory(
                endpoint=kwargs['endpoint'],
                method=kwargs['method'],
                min_time_period=0.1,
                max_time_period=1.0,
                confidence_threshold=0.9
            )
            # Set rate limit in the history
            history.rate_limit = rate_limit
            return "response", history
        
        # Act
        _, histories = await async_request_with_history(
            request_func=mock_request_func,
            endpoints=endpoints,
            method=RequestMethod.GET
        )
        
        # Assert
        assert histories["endpoint_1"].rate_limit is not None
        assert histories["endpoint_1"].rate_limit.max_requests == 10
        assert histories["endpoint_1"].rate_limit.time_period == 1.0

class Test_MergeHistories_01_NominalBehaviors:
    """Tests for nominal behaviors of merge_histories function"""

    def test_groups_histories_by_endpoint_method(self):
        """Test that merge_histories groups request histories by endpoint/method combinations."""
        # Arrange
        histories = [
            RequestHistory(endpoint="endpoint_1", method=RequestMethod.GET),
            RequestHistory(endpoint="endpoint_1", method=RequestMethod.POST),
            RequestHistory(endpoint="endpoint_2", method=RequestMethod.GET)
        ]
        
        # Act
        result = merge_histories(histories)
        
        # Assert
        assert len(result) == 3
        assert ("endpoint_1", RequestMethod.GET) in result
        assert ("endpoint_1", RequestMethod.POST) in result
        assert ("endpoint_2", RequestMethod.GET) in result

    def test_merges_multiple_histories_for_same_endpoint_method(self):
        """Test that merge_histories merges multiple histories for the same endpoint/method."""
        # Arrange
        histories = [
            RequestHistory(endpoint="endpoint_1", method=RequestMethod.GET),
            RequestHistory(endpoint="endpoint_1", method=RequestMethod.GET)
        ]
        
        # Add entries to both histories
        entry1 = RequestEntry(
            timestamp=datetime.now(timezone.utc),
            endpoint="endpoint_1",
            method=RequestMethod.GET,
            status_code=200,
            response_time=0.1,
            success=True
        )
        entry2 = RequestEntry(
            timestamp=datetime.now(timezone.utc),
            endpoint="endpoint_1",
            method=RequestMethod.GET,
            status_code=200,
            response_time=0.2,
            success=True
        )
        
        histories[0].entries.append(entry1)
        histories[1].entries.append(entry2)
        
        # Act
        result = merge_histories(histories)
        
        # Assert
        assert len(result) == 1
        assert len(result[("endpoint_1", RequestMethod.GET)].entries) == 2

    def test_applies_user_provided_rate_limits(self):
        """Test that merge_histories applies user-provided rate limits when specified."""
        # Arrange
        histories = [
            RequestHistory(endpoint="endpoint_1", method=RequestMethod.GET)
        ]
        rate_limit = 5.0  # 5 requests per second
        
        # Act
        result = merge_histories(histories, rate_limit=rate_limit)
        
        # Assert
        key = ("endpoint_1", RequestMethod.GET)
        assert result[key].rate_limit is not None
        assert result[key].rate_limit.max_requests == 5
        assert result[key].rate_limit.time_period == 1.0

    def test_preserves_all_request_data(self):
        """Test that merge_histories preserves all request data during merging."""
        # Arrange
        history1 = RequestHistory(endpoint="endpoint_1", method=RequestMethod.GET)
        history2 = RequestHistory(endpoint="endpoint_1", method=RequestMethod.GET)
        
        # Add entries with distinct data
        entry1 = RequestEntry(
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=5),
            endpoint="endpoint_1",
            method=RequestMethod.GET,
            status_code=200,
            response_time=0.1,
            success=True
        )
        entry2 = RequestEntry(
            timestamp=datetime.now(timezone.utc),
            endpoint="endpoint_1",
            method=RequestMethod.GET,
            status_code=404,
            response_time=0.2,
            success=False
        )
        
        history1.entries.append(entry1)
        history2.entries.append(entry2)
        
        # Act
        result = merge_histories([history1, history2])
        
        # Assert
        merged_history = result[("endpoint_1", RequestMethod.GET)]
        assert len(merged_history.entries) == 2
        
        # Check that both entries exist with their original data
        status_codes = [e.status_code for e in merged_history.entries]
        assert 200 in status_codes
        assert 404 in status_codes
        
        response_times = [e.response_time for e in merged_history.entries]
        assert 0.1 in response_times
        assert 0.2 in response_times

class Test_MergeHistories_02_NegativeBehaviors:
    """Tests for negative behaviors of merge_histories function"""

    def test_handles_empty_histories_list(self):
        """Test that merge_histories handles empty histories list."""
        # Arrange
        histories = []
        
        # Act
        result = merge_histories(histories)
        
        # Assert
        assert result == {}

class Test_MergeHistories_03_BoundaryBehaviors:
    """Tests for boundary behaviors of merge_histories function"""

    def test_handles_single_history_in_list(self):
        """Test that merge_histories handles single history in the list."""
        # Arrange
        history = RequestHistory(endpoint="endpoint_1", method=RequestMethod.GET)
        entry = RequestEntry(
            timestamp=datetime.now(timezone.utc),
            endpoint="endpoint_1",
            method=RequestMethod.GET,
            status_code=200,
            response_time=0.1,
            success=True
        )
        history.entries.append(entry)
        
        # Act
        result = merge_histories([history])
        
        # Assert
        assert len(result) == 1
        assert len(result[("endpoint_1", RequestMethod.GET)].entries) == 1

    def test_processes_rate_limits_at_extremes(self):
        """Test that merge_histories processes rate limits at extremes (very low values)."""
        # Arrange
        histories = [
            RequestHistory(endpoint="endpoint_1", method=RequestMethod.GET)
        ]
        rate_limit = 1  # Very low rate limit
        
        # Act
        result = merge_histories(histories, rate_limit=rate_limit)
        
        # Assert
        key = ("endpoint_1", RequestMethod.GET)
        assert result[key].rate_limit is not None
        assert result[key].rate_limit.max_requests == 1  # Should be at least 1
        assert result[key].rate_limit.time_period == 1.0

class Test_MergeHistories_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of merge_histories function"""

    def test_sets_conservative_rate_limits_for_429(self):
        """Test that merge_histories sets conservative rate limits when 429 responses are detected."""
        # Arrange
        history = RequestHistory(endpoint="endpoint_1", method=RequestMethod.GET)
        # Add a 429 (Too Many Requests) entry
        entry = RequestEntry(
            timestamp=datetime.now(timezone.utc),
            endpoint="endpoint_1",
            method=RequestMethod.GET,
            status_code=429,  # Too Many Requests
            response_time=0.1,
            success=True
        )
        history.entries.append(entry)
        
        # Act
        result = merge_histories([history])
        
        # Assert
        key = ("endpoint_1", RequestMethod.GET)
        assert result[key].rate_limit is not None
        assert result[key].rate_limit.max_requests == 1  # Conservative
        assert result[key].rate_limit.time_period == 5.0  # 5 seconds between requests

class Test_MergeHistories_05_StateTransitionBehaviors:
    """Tests for state transition behaviors of merge_histories function"""

    def test_updates_rate_limit_for_429_responses(self):
        """Test that merge_histories updates rate limit understanding when 429 responses are detected."""
        # Arrange
        history = RequestHistory(endpoint="endpoint_1", method=RequestMethod.GET)
        # Set an initial rate limit
        history.rate_limit = RateLimit(
            endpoint="endpoint_1",
            method=RequestMethod.GET,
            max_requests=10,
            time_period=1.0,
            last_updated=datetime.now(timezone.utc) - timedelta(minutes=10)
        )
        # Add a 429 (Too Many Requests) entry
        entry = RequestEntry(
            timestamp=datetime.now(timezone.utc),
            endpoint="endpoint_1",
            method=RequestMethod.GET,
            status_code=429,  # Too Many Requests
            response_time=0.1,
            success=True
        )
        history.entries.append(entry)
        
        # Act
        result = merge_histories([history])
        
        # Assert
        key = ("endpoint_1", RequestMethod.GET)
        assert result[key].rate_limit is not None
        assert result[key].rate_limit.max_requests == 1  # Conservative
        assert result[key].rate_limit.time_period == 5.0  # 5 seconds between requests

    def test_transitions_to_most_appropriate_rate_limit(self):
        """Test that merge_histories transitions merged histories to use the most appropriate rate limit settings."""
        # Arrange
        history1 = RequestHistory(endpoint="endpoint_1", method=RequestMethod.GET)
        history2 = RequestHistory(endpoint="endpoint_1", method=RequestMethod.GET)
        
        # Set different rate limits
        history1.rate_limit = RateLimit(
            endpoint="endpoint_1",
            method=RequestMethod.GET,
            max_requests=5,
            time_period=1.0,
            last_updated=datetime.now(timezone.utc) - timedelta(minutes=10)
        )
        
        history2.rate_limit = RateLimit(
            endpoint="endpoint_1",
            method=RequestMethod.GET,
            max_requests=2,
            time_period=1.0,
            last_updated=datetime.now(timezone.utc)  # More recent
        )
        
        # Act
        result = merge_histories([history1, history2])
        
        # Assert
        key = ("endpoint_1", RequestMethod.GET)
        # Should use the more recent rate limit from history2
        assert result[key].rate_limit.max_requests == 2
        assert result[key].rate_limit.last_updated == history2.rate_limit.last_updated
