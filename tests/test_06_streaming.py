import pytest
import json
import os
import uuid
import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, mock_open, call
from pydantic import ValidationError
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException

# Imports from SmartSurge
from smartsurge.streaming import StreamingState, AbstractStreamingRequest, JSONStreamingRequest
from smartsurge.exceptions import StreamingError, ResumeError


# Test class for StreamingState behaviors

class Test_StreamingState_01_NominalBehaviors:
    """Test nominal behaviors of the StreamingState model."""

    def test_initialize_with_required_fields(self):
        """Test that the model initializes correctly with all required fields."""
        data = {
            "endpoint": "https://api.example.com/data",
            "method": "GET",
            "headers": {"Authorization": "Bearer token123"},
            "accumulated_data": b"test data",
            "last_position": 9
        }
        state = StreamingState(**data)
        assert state.endpoint == data["endpoint"]
        assert state.method == data["method"]
        assert state.headers == data["headers"]
        assert state.accumulated_data == data["accumulated_data"]
        assert state.last_position == data["last_position"]

    def test_generates_timestamp(self):
        """Test that the model generates a current timestamp for last_updated if not provided."""
        before = datetime.now(timezone.utc)
        state = StreamingState(
            endpoint="https://api.example.com/data",
            method="GET",
            headers={},
            accumulated_data=b"",
            last_position=0
        )
        after = datetime.now(timezone.utc)
        assert before <= state.last_updated <= after
        assert state.last_updated.tzinfo == timezone.utc

    def test_generates_request_id(self):
        """Test that the model generates a unique request_id if not provided."""
        state1 = StreamingState(
            endpoint="https://api.example.com/data",
            method="GET",
            headers={},
            accumulated_data=b"",
            last_position=0
        )
        state2 = StreamingState(
            endpoint="https://api.example.com/data",
            method="GET",
            headers={},
            accumulated_data=b"",
            last_position=0
        )
        assert isinstance(state1.request_id, str)
        assert len(state1.request_id) == 8  # UUID[:8]
        assert state1.request_id != state2.request_id  # Should be unique

    def test_validates_field_types(self):
        """Test that the model validates all fields with proper types."""
        # Create with proper types
        state = StreamingState(
            endpoint="https://api.example.com/data",
            method="GET",
            headers={"Content-Type": "application/json"},
            params={"key": "value"},
            data={"field": "value"},
            accumulated_data=b"test data",
            last_position=9,
            total_size=100,
            etag="abc123"
        )
        # Verify all fields have correct types
        assert isinstance(state.endpoint, str)
        assert isinstance(state.method, str)
        assert isinstance(state.headers, dict)
        assert isinstance(state.params, dict)
        assert isinstance(state.data, dict)
        assert isinstance(state.accumulated_data, bytes)
        assert isinstance(state.last_position, int)
        assert isinstance(state.total_size, int)
        assert isinstance(state.etag, str)

    def test_accepts_optional_fields(self):
        """Test that the model accepts optional fields properly."""
        state = StreamingState(
            endpoint="https://api.example.com/data",
            method="GET",
            headers={},
            accumulated_data=b"",
            last_position=0,
            # Optional fields:
            params={"query": "test"},
            data={"body": "content"},
            total_size=1024,
            etag="W/\"123-456\""
        )
        assert state.params == {"query": "test"}
        assert state.data == {"body": "content"}
        assert state.total_size == 1024
        assert state.etag == "W/\"123-456\""


class Test_StreamingState_02_NegativeBehaviors:
    """Test negative behaviors of the StreamingState model."""

    def test_raises_error_for_missing_required_fields(self):
        """Test that the model raises validation error when required fields are missing."""
        # Test missing endpoint
        with pytest.raises(ValidationError):
            StreamingState(
                method="GET", 
                headers={}, 
                accumulated_data=b"", 
                last_position=0
            )
        
        # Test missing method
        with pytest.raises(ValidationError):
            StreamingState(
                endpoint="https://api.example.com/data", 
                headers={}, 
                accumulated_data=b"", 
                last_position=0
            )
        
        # Test missing headers
        with pytest.raises(ValidationError):
            StreamingState(
                endpoint="https://api.example.com/data", 
                method="GET", 
                accumulated_data=b"", 
                last_position=0
            )
        
        # Test missing accumulated_data
        with pytest.raises(ValidationError):
            StreamingState(
                endpoint="https://api.example.com/data", 
                method="GET", 
                headers={}, 
                last_position=0
            )
        
        # Test missing last_position
        with pytest.raises(ValidationError):
            StreamingState(
                endpoint="https://api.example.com/data", 
                method="GET", 
                headers={}, 
                accumulated_data=b""
            )

    def test_raises_error_for_incorrect_field_types(self):
        """Test that the model raises validation error when field types are incorrect."""
        # Test incorrect endpoint type
        with pytest.raises(ValidationError):
            StreamingState(
                endpoint=123,  # Should be string
                method="GET", 
                headers={}, 
                accumulated_data=b"", 
                last_position=0
            )
        
        # Test incorrect headers type
        with pytest.raises(ValidationError):
            StreamingState(
                endpoint="https://api.example.com/data", 
                method="GET", 
                headers="not-a-dict",  # Should be dict
                accumulated_data=b"", 
                last_position=0
            )
        
        # Test incorrect accumulated_data type
        with pytest.raises(ValidationError):
            StreamingState(
                endpoint="https://api.example.com/data", 
                method="GET", 
                headers={}, 
                accumulated_data=123,  # Should be bytes
                last_position=0
            )
        
        # Test incorrect last_position type
        with pytest.raises(ValidationError):
            StreamingState(
                endpoint="https://api.example.com/data", 
                method="GET", 
                headers={}, 
                accumulated_data=b"", 
                last_position="zero"  # Should be int
            )

    def test_raises_error_for_zero_length_endpoint(self):
        """Test that the model raises validation error when endpoint has zero length."""
        with pytest.raises(ValidationError):
            StreamingState(
                endpoint="",  # Empty string
                method="GET",
                headers={},
                accumulated_data=b"",
                last_position=0
            )

    def test_raises_error_for_negative_position(self):
        """Test that the model raises validation error when last_position is negative."""
        with pytest.raises(ValidationError):
            StreamingState(
                endpoint="https://api.example.com/data",
                method="GET",
                headers={},
                accumulated_data=b"",
                last_position=-1  # Negative value
            )


class Test_StreamingState_03_BoundaryBehaviors:
    """Test boundary behaviors of the StreamingState model."""

    def test_accepts_minimum_length_endpoint(self):
        """Test that the model accepts endpoint with minimum length (1 character)."""
        state = StreamingState(
            endpoint="x",  # Minimum length
            method="GET",
            headers={},
            accumulated_data=b"",
            last_position=0
        )
        assert state.endpoint == "x"

    def test_handles_zero_position(self):
        """Test that the model handles last_position of exactly 0."""
        state = StreamingState(
            endpoint="https://api.example.com/data",
            method="GET",
            headers={},
            accumulated_data=b"",
            last_position=0  # Exactly zero
        )
        assert state.last_position == 0

    def test_handles_empty_dicts(self):
        """Test that the model handles empty dictionaries for headers, params, and data."""
        state = StreamingState(
            endpoint="https://api.example.com/data",
            method="GET",
            headers={},  # Empty dict
            params={},   # Empty dict
            data={},     # Empty dict
            accumulated_data=b"",
            last_position=0
        )
        assert state.headers == {}
        assert state.params == {}
        assert state.data == {}

    def test_handles_large_data(self):
        """Test that the model handles large accumulated_data byte objects."""
        # Create a 1MB byte object
        large_data = b"x" * (1024 * 1024)
        state = StreamingState(
            endpoint="https://api.example.com/data",
            method="GET",
            headers={},
            accumulated_data=large_data,
            last_position=len(large_data)
        )
        assert len(state.accumulated_data) == 1024 * 1024
        assert state.last_position == 1024 * 1024


# Create a test implementation of AbstractStreamingRequest for testing
class TestStreamingRequest(AbstractStreamingRequest):
    """Concrete implementation of AbstractStreamingRequest for testing."""
    
    def start(self):
        pass
        
    def resume(self):
        pass
        
    def process_chunk(self, chunk):
        pass
        
    def get_result(self):
        return self.accumulated_data


class Test_AbstractStreamingRequest_SaveState_01_NominalBehaviors:
    """Test nominal behaviors of AbstractStreamingRequest.save_state method."""
    
    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_creates_state_with_current_values(self, mock_file, mock_makedirs, mock_exists):
        """Test it creates a StreamingState object with current values."""
        mock_exists.return_value = True
        
        # Initialize test object
        req = TestStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={"Authorization": "Bearer token123"},
            state_file="/tmp/test-state.json"
        )
        req.position = 100
        req.accumulated_data = bytearray(b"test data")
        req.total_size = 500
        req.etag = "etag123"
        req.request_id = "test-id"
        
        # Call save_state
        req.save_state()
        
        # Verify the state was written to the file with correct values
        mock_file.assert_called_once_with("/tmp/test-state.json", "w")
        written_data = mock_file().write.call_args[0][0]
        state_dict = json.loads(written_data)
        
        assert state_dict["endpoint"] == "https://api.example.com/data"
        assert state_dict["method"] == "GET"
        assert state_dict["headers"] == {"Authorization": "Bearer token123"}
        assert state_dict["accumulated_data"] == "dGVzdCBkYXRh"  # Base64 encoded "test data"
        assert state_dict["last_position"] == 100
        assert state_dict["total_size"] == 500
        assert state_dict["etag"] == "etag123"
        assert state_dict["request_id"] == "test-id"

    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_writes_state_to_file(self, mock_file, mock_makedirs, mock_exists):
        """Test it writes state to specified file in JSON format."""
        mock_exists.return_value = True
        
        req = TestStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            state_file="/tmp/test-state.json"
        )
        
        req.save_state()
        
        # Verify file was opened and written to
        mock_file.assert_called_once_with("/tmp/test-state.json", "w")
        mock_file().write.assert_called_once()

    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_creates_parent_directories(self, mock_file, mock_makedirs, mock_exists):
        """Test it creates parent directories if they don't exist."""
        mock_exists.return_value = False
        
        req = TestStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            state_file="/tmp/directory/test-state.json"
        )
        
        req.save_state()
        
        # Verify directory was created
        mock_makedirs.assert_called_once_with("/tmp/directory", exist_ok=True)

    def test_logs_debug_message_on_success(self):
        """Test it logs debug message when state is successfully saved."""
        # Setup a mock logger
        mock_logger = MagicMock(spec=logging.Logger)
        
        with patch("os.path.exists", return_value=True), \
             patch("os.makedirs"), \
             patch("builtins.open", mock_open()):
            
            req = TestStreamingRequest(
                endpoint="https://api.example.com/data",
                headers={},
                state_file="/tmp/test-state.json",
                logger=mock_logger,
                request_id="test-id"
            )
            
            req.save_state()
            
            # Verify debug message was logged
            mock_logger.debug.assert_called_once_with("[test-id] Saved state to /tmp/test-state.json")


class Test_AbstractStreamingRequest_SaveState_02_NegativeBehaviors:
    """Test negative behaviors of AbstractStreamingRequest.save_state method."""
    
    def test_logs_warning_without_state_file(self):
        """Test it logs warning and returns early if state_file is not specified."""
        # Setup a mock logger
        mock_logger = MagicMock(spec=logging.Logger)
        
        req = TestStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            state_file=None,  # No state file
            logger=mock_logger,
            request_id="test-id"
        )
        
        req.save_state()
        
        # Verify warning was logged
        mock_logger.warning.assert_called_once_with("[test-id] No state file specified, skipping state save")

    def test_handles_file_writing_exceptions(self):
        """Test it handles and logs exceptions during file writing operations."""
        # Setup a mock logger
        mock_logger = MagicMock(spec=logging.Logger)
        
        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", side_effect=IOError("Test IO error")):
            
            req = TestStreamingRequest(
                endpoint="https://api.example.com/data",
                headers={},
                state_file="/tmp/test-state.json",
                logger=mock_logger,
                request_id="test-id"
            )
            
            req.save_state()
            
            # Verify error was logged
            mock_logger.error.assert_called_once()
            error_msg = mock_logger.error.call_args[0][0]
            assert "[test-id] Failed to save state:" in error_msg
            assert "Test IO error" in error_msg


class Test_AbstractStreamingRequest_SaveState_03_BoundaryBehaviors:
    """Test boundary behaviors of AbstractStreamingRequest.save_state method."""
    
    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_handles_empty_accumulated_data(self, mock_file, mock_exists, mock_makedirs):
        """Test it handles empty accumulated_data."""
        mock_exists.return_value = True
        
        req = TestStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            state_file="/tmp/test-state.json"
        )
        req.accumulated_data = bytearray()  # Empty data
        
        req.save_state()
        
        # Verify state includes empty accumulated_data
        written_data = mock_file().write.call_args[0][0]
        state_dict = json.loads(written_data)
        assert state_dict["accumulated_data"] == ""  # Empty base64 string

    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_handles_position_zero(self, mock_file, mock_exists, mock_makedirs):
        """Test it handles position of 0."""
        mock_exists.return_value = True
        
        req = TestStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            state_file="/tmp/test-state.json"
        )
        req.position = 0  # Position zero
        
        req.save_state()
        
        # Verify state includes position 0
        written_data = mock_file().write.call_args[0][0]
        state_dict = json.loads(written_data)
        assert state_dict["last_position"] == 0

    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_creates_minimal_valid_state(self, mock_file, mock_exists, mock_makedirs):
        """Test it creates minimal valid state with required fields."""
        mock_exists.return_value = True
        
        # Create object with minimal required data
        req = TestStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            state_file="/tmp/test-state.json"
        )
        
        req.save_state()
        
        # Verify state has all required fields
        written_data = mock_file().write.call_args[0][0]
        state_dict = json.loads(written_data)
        
        required_fields = ["endpoint", "method", "headers", "accumulated_data", "last_position", "last_updated", "request_id"]
        for field in required_fields:
            assert field in state_dict


class Test_AbstractStreamingRequest_SaveState_04_ErrorHandlingBehaviors:
    """Test error handling behaviors of AbstractStreamingRequest.save_state method."""
    
    def test_catches_directory_creation_exceptions(self):
        """Test it catches and logs exceptions during directory creation."""
        # Setup a mock logger
        mock_logger = MagicMock(spec=logging.Logger)
        
        with patch("os.path.exists", return_value=False), \
             patch("os.makedirs", side_effect=OSError("Permission denied")):
            
            req = TestStreamingRequest(
                endpoint="https://api.example.com/data",
                headers={},
                state_file="/tmp/directory/test-state.json",
                logger=mock_logger,
                request_id="test-id"
            )
            
            # Method should not raise exception
            req.save_state()
            
            # Verify error was logged
            mock_logger.error.assert_called_once()
            error_msg = mock_logger.error.call_args[0][0]
            assert "[test-id] Failed to save state:" in error_msg
            assert "Permission denied" in error_msg

    def test_catches_file_operations_exceptions(self):
        """Test it catches and logs exceptions during file operations."""
        # Setup a mock logger
        mock_logger = MagicMock(spec=logging.Logger)
        
        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", side_effect=PermissionError("Access denied")):
            
            req = TestStreamingRequest(
                endpoint="https://api.example.com/data",
                headers={},
                state_file="/tmp/test-state.json",
                logger=mock_logger,
                request_id="test-id"
            )
            
            # Method should not raise exception
            req.save_state()
            
            # Verify error was logged
            mock_logger.error.assert_called_once()
            error_msg = mock_logger.error.call_args[0][0]
            assert "[test-id] Failed to save state:" in error_msg
            assert "Access denied" in error_msg

    def test_does_not_propagate_exceptions(self):
        """Test it doesn't propagate exceptions to the caller."""
        with patch("os.path.exists", side_effect=Exception("Unexpected error")):
            
            req = TestStreamingRequest(
                endpoint="https://api.example.com/data",
                headers={},
                state_file="/tmp/test-state.json"
            )
            
            # Method should not raise exception
            try:
                req.save_state()
                exception_raised = False
            except:
                exception_raised = True
            
            assert not exception_raised


class Test_AbstractStreamingRequest_LoadState_01_NominalBehaviors:
    """Test nominal behaviors of AbstractStreamingRequest.load_state method."""
    
    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_reads_and_parses_state_file(self, mock_file, mock_exists):
        """Test it reads and parses state file correctly."""
        mock_exists.return_value = True
        
        # Setup mock file with test state data
        state_data = json.dumps({
            "endpoint": "https://api.example.com/data",
            "method": "GET",
            "headers": {"Authorization": "Bearer token123"},
            "accumulated_data": "dGVzdCBkYXRh",  # "test data" in base64
            "last_position": 9,
            "total_size": 100,
            "etag": "etag123",
            "request_id": "saved-id"
        })
        mock_file.return_value.read.return_value = state_data
        
        req = TestStreamingRequest(
            endpoint="https://api.example.com/original",  # Different from saved state
            headers={"Original": "header"},
            state_file="/tmp/test-state.json"
        )
        
        # Call load_state
        state = req.load_state()
        
        # Verify file was read
        mock_file.assert_called_once_with("/tmp/test-state.json", "r")
        mock_file().read.assert_called_once()
        
        # Verify returned state object
        assert state is not None
        assert state.endpoint == "https://api.example.com/data"
        assert state.method == "GET"
        assert state.headers == {"Authorization": "Bearer token123"}
        assert state.accumulated_data == b"test data"
        assert state.last_position == 9
        assert state.total_size == 100
        assert state.etag == "etag123"
        assert state.request_id == "saved-id"

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_updates_instance_variables(self, mock_file, mock_exists):
        """Test it updates instance variables from loaded state."""
        mock_exists.return_value = True
        
        # Setup mock file with test state data
        state_data = json.dumps({
            "endpoint": "https://api.example.com/data",
            "method": "GET",
            "headers": {"Authorization": "Bearer token123"},
            "accumulated_data": "dGVzdCBkYXRh",  # "test data" in base64
            "last_position": 9,
            "total_size": 100,
            "etag": "etag123",
            "request_id": "saved-id"
        })
        mock_file.return_value.read.return_value = state_data
        
        req = TestStreamingRequest(
            endpoint="https://api.example.com/original",  # Different from saved state
            headers={"Original": "header"},
            state_file="/tmp/test-state.json"
        )
        
        # Call load_state
        req.load_state()
        
        # Verify instance variables were updated
        assert req.endpoint == "https://api.example.com/data"
        assert req.headers == {"Authorization": "Bearer token123"}
        assert bytes(req.accumulated_data) == b"test data"
        assert req.position == 9
        assert req.total_size == 100
        assert req.etag == "etag123"
        assert req.request_id == "saved-id"  # Should be updated from state

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_returns_loaded_state(self, mock_file, mock_exists):
        """Test it returns the loaded state."""
        mock_exists.return_value = True
        
        # Setup mock file with test state data
        state_data = json.dumps({
            "endpoint": "https://api.example.com/data",
            "method": "GET",
            "headers": {"Authorization": "Bearer token123"},
            "accumulated_data": "dGVzdCBkYXRh",  # "test data" in base64
            "last_position": 9,
            "total_size": 100,
            "etag": "etag123"
        })
        mock_file.return_value.read.return_value = state_data
        
        req = TestStreamingRequest(
            endpoint="https://api.example.com/original",
            headers={"Original": "header"},
            state_file="/tmp/test-state.json"
        )
        
        # Call load_state
        state = req.load_state()
        
        # Verify returned state is not None
        assert state is not None
        assert isinstance(state, StreamingState)

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_preserves_request_id_from_state(self, mock_file, mock_exists):
        """Test it preserves original request_id if available in state."""
        mock_exists.return_value = True
        
        # Setup mock file with test state data including request_id
        state_data = json.dumps({
            "endpoint": "https://api.example.com/data",
            "method": "GET",
            "headers": {},
            "accumulated_data": "",
            "last_position": 0,
            "request_id": "saved-request-id"
        })
        mock_file.return_value.read.return_value = state_data
        
        req = TestStreamingRequest(
            endpoint="https://api.example.com/original",
            headers={},
            state_file="/tmp/test-state.json",
            request_id="original-request-id"
        )
        
        # Call load_state
        req.load_state()
        
        # Verify request_id was updated from state
        assert req.request_id == "saved-request-id"


class Test_AbstractStreamingRequest_LoadState_02_NegativeBehaviors:
    """Test negative behaviors of AbstractStreamingRequest.load_state method."""
    
    @patch("os.path.exists")
    def test_returns_none_if_state_file_does_not_exist(self, mock_exists):
        """Test it returns None if state file doesn't exist."""
        mock_exists.return_value = False
        
        # Setup mock logger
        mock_logger = MagicMock(spec=logging.Logger)
        
        req = TestStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            state_file="/tmp/nonexistent-state.json",
            logger=mock_logger,
            request_id="test-id"
        )
        
        # Call load_state
        state = req.load_state()
        
        # Verify returned None
        assert state is None

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_returns_none_on_file_error(self, mock_file, mock_exists):
        """Test it returns None if there's an error reading the state file."""
        mock_exists.return_value = True
        mock_file.side_effect = IOError("Test IO error")
        
        # Setup mock logger
        mock_logger = MagicMock(spec=logging.Logger)
        
        req = TestStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            state_file="/tmp/test-state.json",
            logger=mock_logger,
            request_id="test-id"
        )
        
        # Call load_state
        state = req.load_state()
        
        # Verify returned None
        assert state is None
        
        # Verify error was logged
        mock_logger.error.assert_called_once()
        error_msg = mock_logger.error.call_args[0][0]
        assert "[test-id] Failed to load state:" in error_msg
        assert "Test IO error" in error_msg

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_returns_none_on_json_parse_error(self, mock_file, mock_exists):
        """Test it returns None if there's an error parsing the state file."""
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = "invalid json {{"
        
        # Setup mock logger
        mock_logger = MagicMock(spec=logging.Logger)
        
        req = TestStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            state_file="/tmp/test-state.json",
            logger=mock_logger,
            request_id="test-id"
        )
        
        # Call load_state
        state = req.load_state()
        
        # Verify returned None
        assert state is None
        
        # Verify error was logged
        mock_logger.error.assert_called_once()
        error_msg = mock_logger.error.call_args[0][0]
        assert "[test-id] Failed to load state:" in error_msg

    def test_logs_warning_if_state_file_does_not_exist(self):
        """Test it logs appropriate warning if state file doesn't exist."""
        # Setup mock logger
        mock_logger = MagicMock(spec=logging.Logger)
        
        with patch("os.path.exists", return_value=False):
            req = TestStreamingRequest(
                endpoint="https://api.example.com/data",
                headers={},
                state_file="/tmp/nonexistent-state.json",
                logger=mock_logger,
                request_id="test-id"
            )
            
            req.load_state()
            
            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "[test-id] State file does not exist:" in warning_msg
            assert "/tmp/nonexistent-state.json" in warning_msg


class Test_AbstractStreamingRequest_LoadState_03_BoundaryBehaviors:
    """Test boundary behaviors of AbstractStreamingRequest.load_state method."""
    
    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_handles_state_with_position_zero(self, mock_file, mock_exists):
        """Test it handles state with position of 0."""
        mock_exists.return_value = True
        
        # Setup mock file with test state data with position 0
        state_data = json.dumps({
            "endpoint": "https://api.example.com/data",
            "method": "GET",
            "headers": {},
            "accumulated_data": "",
            "last_position": 0,  # Position zero
            "request_id": "test-id"
        })
        mock_file.return_value.read.return_value = state_data
        
        req = TestStreamingRequest(
            endpoint="https://api.example.com/original",
            headers={},
            state_file="/tmp/test-state.json"
        )
        
        # Call load_state
        state = req.load_state()
        
        # Verify state was loaded with position 0
        assert state is not None
        assert state.last_position == 0
        assert req.position == 0

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_handles_state_with_empty_accumulated_data(self, mock_file, mock_exists):
        """Test it handles state with empty accumulated_data."""
        mock_exists.return_value = True
        
        # Setup mock file with test state data with empty accumulated_data
        state_data = json.dumps({
            "endpoint": "https://api.example.com/data",
            "method": "GET",
            "headers": {},
            "accumulated_data": "",  # Empty data
            "last_position": 0,
            "request_id": "test-id"
        })
        mock_file.return_value.read.return_value = state_data
        
        req = TestStreamingRequest(
            endpoint="https://api.example.com/original",
            headers={},
            state_file="/tmp/test-state.json"
        )
        req.accumulated_data = bytearray(b"original data")  # Set some original data
        
        # Call load_state
        state = req.load_state()
        
        # Verify state was loaded with empty accumulated_data
        assert state is not None
        assert state.accumulated_data == b""
        assert bytes(req.accumulated_data) == b""  # Should be empty now


class Test_AbstractStreamingRequest_LoadState_04_ErrorHandlingBehaviors:
    """Test error handling behaviors of AbstractStreamingRequest.load_state method."""
    
    def test_catches_file_operation_exceptions(self):
        """Test it catches and logs exceptions during file operations."""
        # Setup mock logger
        mock_logger = MagicMock(spec=logging.Logger)
        
        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", side_effect=IOError("Test IO error")):
            
            req = TestStreamingRequest(
                endpoint="https://api.example.com/data",
                headers={},
                state_file="/tmp/test-state.json",
                logger=mock_logger,
                request_id="test-id"
            )
            
            # Call load_state - should not raise exception
            state = req.load_state()
            
            # Verify returned None
            assert state is None
            
            # Verify error was logged
            mock_logger.error.assert_called_once()
            error_msg = mock_logger.error.call_args[0][0]
            assert "[test-id] Failed to load state:" in error_msg
            assert "Test IO error" in error_msg

    def test_catches_json_parsing_exceptions(self):
        """Test it catches and logs exceptions during JSON parsing."""
        # Setup mock logger
        mock_logger = MagicMock(spec=logging.Logger)
        
        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", mock_open(read_data="invalid json {")):
            
            req = TestStreamingRequest(
                endpoint="https://api.example.com/data",
                headers={},
                state_file="/tmp/test-state.json",
                logger=mock_logger,
                request_id="test-id"
            )
            
            # Call load_state - should not raise exception
            state = req.load_state()
            
            # Verify returned None
            assert state is None
            
            # Verify error was logged
            mock_logger.error.assert_called_once()
            error_msg = mock_logger.error.call_args[0][0]
            assert "[test-id] Failed to load state:" in error_msg

    def test_returns_none_instead_of_raising_exceptions(self):
        """Test it returns None instead of raising exceptions."""
        # Test with file operation exception
        with patch("os.path.exists", side_effect=Exception("Unexpected error")):
            req = TestStreamingRequest(
                endpoint="https://api.example.com/data",
                headers={},
                state_file="/tmp/test-state.json"
            )
            
            # Call load_state - should not raise exception
            state = req.load_state()
            
            # Verify returned None
            assert state is None


class Test_AbstractStreamingRequest_LoadState_05_StateTransitionBehaviors:
    """Test state transition behaviors of AbstractStreamingRequest.load_state method."""
    
    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_transitions_object_to_saved_state(self, mock_file, mock_exists):
        """Test it transitions object to the saved state by updating all properties."""
        mock_exists.return_value = True
        
        # Setup mock file with test state data
        state_data = json.dumps({
            "endpoint": "https://api.example.com/data",
            "method": "GET",
            "headers": {"Authorization": "Bearer token123"},
            "accumulated_data": "dGVzdCBkYXRh",  # "test data" in base64
            "last_position": 9,
            "total_size": 100,
            "etag": "etag123",
            "request_id": "saved-id"
        })
        mock_file.return_value.read.return_value = state_data
        
        # Create object with different initial state
        req = TestStreamingRequest(
            endpoint="https://api.example.com/original",
            headers={"Original": "header"},
            state_file="/tmp/test-state.json"
        )
        req.accumulated_data = bytearray(b"original data")
        req.position = 0
        req.total_size = None
        req.etag = None
        req.request_id = "original-id"
        
        # Call load_state
        req.load_state()
        
        # Verify all properties were transitioned
        assert req.endpoint == "https://api.example.com/data"
        assert req.headers == {"Authorization": "Bearer token123"}
        assert bytes(req.accumulated_data) == b"test data"
        assert req.position == 9
        assert req.total_size == 100
        assert req.etag == "etag123"
        assert req.request_id == "saved-id"

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_maintains_loaded_state_consistency(self, mock_file, mock_exists):
        """Test it maintains consistency between loaded state and object properties."""
        mock_exists.return_value = True
        
        # Setup mock file with test state data
        state_data = json.dumps({
            "endpoint": "https://api.example.com/data",
            "method": "GET",
            "headers": {"Authorization": "Bearer token123"},
            "accumulated_data": "dGVzdCBkYXRh",  # "test data" in base64
            "last_position": 9,
            "total_size": 100,
            "etag": "etag123"
        })
        mock_file.return_value.read.return_value = state_data
        
        req = TestStreamingRequest(
            endpoint="https://api.example.com/original",
            headers={"Original": "header"},
            state_file="/tmp/test-state.json"
        )
        
        # Call load_state
        state = req.load_state()
        
        # Verify consistency between returned state and object properties
        assert state.endpoint == req.endpoint
        assert state.headers == req.headers
        assert state.accumulated_data == bytes(req.accumulated_data)
        assert state.last_position == req.position
        assert state.total_size == req.total_size
        assert state.etag == req.etag


class Test_JSONStreamingRequest_Start_01_NominalBehaviors:
    """Test nominal behaviors of JSONStreamingRequest.start method."""
    
    @patch("requests.Session")
    def test_sets_up_session_with_retry(self, mock_session):
        """Test it sets up session with retry capabilities."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.iter_content.return_value = [b"test"]
        mock_response.headers = {}
        
        # Setup mock session
        mock_session_instance = MagicMock()
        mock_session_instance.__enter__.return_value = mock_response
        mock_session.return_value.get.return_value = mock_session_instance
        
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={"Authorization": "Bearer token123"}
        )
        
        # Call start
        request.start()
        
        # Verify session was created with retry
        mock_session.assert_called_once()
        
        # Verify adapter was mounted with retry strategy
        mount_calls = mock_session.return_value.mount.call_args_list
        assert len(mount_calls) == 2  # Should mount http and https
        assert mount_calls[0][0][0] == "http://"
        assert mount_calls[1][0][0] == "https://"
        
        # Verify adapter was created with retry strategy
        for call in mount_calls:
            adapter = call[0][1]
            assert isinstance(adapter, HTTPAdapter)
    
    @patch("requests.Session")
    def test_makes_streaming_get_request(self, mock_session):
        """Test it makes streaming GET request with appropriate headers."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.iter_content.return_value = [b"test"]
        mock_response.headers = {}
        
        # Setup mock session
        mock_session_instance = MagicMock()
        mock_session_instance.__enter__.return_value = mock_response
        mock_session.return_value.get.return_value = mock_session_instance
        
        # Create JSONStreamingRequest with custom headers
        custom_headers = {"Authorization": "Bearer token123", "Custom-Header": "Value"}
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers=custom_headers,
            params={"param1": "value1"}
        )
        
        # Call start
        request.start()
        
        # Verify GET request was made with correct parameters
        mock_session.return_value.get.assert_called_once_with(
            "https://api.example.com/data",
            headers=custom_headers,
            params={"param1": "value1"},
            stream=True,
            timeout=(10, 30)  # Default timeout
        )
    
    @patch("requests.Session")
    def test_processes_response_chunks(self, mock_session):
        """Test it processes response in chunks of specified size."""
        # Setup mock chunks
        chunks = [b"chunk1", b"chunk2", b"chunk3"]
        
        # Setup mock response
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.iter_content.return_value = chunks
        mock_response.headers = {}
        
        # Setup mock session
        mock_session_instance = MagicMock()
        mock_session_instance.__enter__.return_value = mock_response
        mock_session.return_value.get.return_value = mock_session_instance
        
        # Create JSONStreamingRequest with process_chunk spy
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            chunk_size=1024  # Custom chunk size
        )
        process_chunk_spy = MagicMock(wraps=request.process_chunk)
        request.process_chunk = process_chunk_spy
        
        # Call start
        request.start()
        
        # Verify iter_content was called with specified chunk size
        mock_response.iter_content.assert_called_once_with(chunk_size=1024)
        
        # Verify process_chunk was called for each chunk
        assert process_chunk_spy.call_count == 3
        process_chunk_spy.assert_has_calls([
            call(b"chunk1"),
            call(b"chunk2"),
            call(b"chunk3")
        ])
    
    @patch("requests.Session")
    def test_marks_request_completed(self, mock_session):
        """Test it marks request as completed when finished."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.iter_content.return_value = [b"test"]
        mock_response.headers = {}
        
        # Setup mock session
        mock_session_instance = MagicMock()
        mock_session_instance.__enter__.return_value = mock_response
        mock_session.return_value.get.return_value = mock_session_instance
        
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={}
        )
        
        # Verify not completed before starting
        assert not request.completed
        
        # Call start
        request.start()
        
        # Verify completed after finishing
        assert request.completed
    
    @patch("requests.Session")
    def test_extracts_headers(self, mock_session):
        """Test it extracts Content-Length and ETag headers when present."""
        # Setup mock response with headers
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.iter_content.return_value = [b"test"]
        mock_response.headers = {
            "Content-Length": "1024",
            "ETag": "\"abc123\""
        }
        
        # Setup mock session
        mock_session_instance = MagicMock()
        mock_session_instance.__enter__.return_value = mock_response
        mock_session.return_value.get.return_value = mock_session_instance
        
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={}
        )
        
        # Call start
        request.start()
        
        # Verify headers were extracted
        assert request.total_size == 1024
        assert request.etag == "\"abc123\""


class Test_JSONStreamingRequest_Start_02_NegativeBehaviors:
    """Test negative behaviors of JSONStreamingRequest.start method."""
    
    @patch("requests.Session")
    def test_raises_error_for_bad_response(self, mock_session):
        """Test it raises StreamingError if response status is not OK."""
        # Setup mock response with non-OK status
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        
        # Setup mock session
        mock_session_instance = MagicMock()
        mock_session_instance.__enter__.return_value = mock_response
        mock_session.return_value.get.return_value = mock_session_instance
        
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={}
        )
        
        # Call start - should raise StreamingError
        with pytest.raises(StreamingError) as excinfo:
            request.start()
        
        # Verify error message contains status code
        assert "404" in str(excinfo.value)

    @patch("requests.Session")
    def test_handles_invalid_content_length(self, mock_session):
        """Test it handles invalid Content-Length headers gracefully."""
        # Setup mock response with invalid Content-Length
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.iter_content.return_value = [b"test"]
        mock_response.headers = {"Content-Length": "not-a-number"}
        
        # Setup mock session
        mock_session_instance = MagicMock()
        mock_session_instance.__enter__.return_value = mock_response
        mock_session.return_value.get.return_value = mock_session_instance
        
        # Create JSONStreamingRequest with a mock logger to capture warnings
        mock_logger = MagicMock(spec=logging.Logger)
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            logger=mock_logger
        )
        
        # Call start - should not raise exception despite invalid header
        request.start()
        
        # Verify no total_size was set
        assert request.total_size is None
        
        # Verify warning was logged
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "Invalid Content-Length header" in warning_msg

    @patch("requests.Session")
    def test_logs_error_on_failure(self, mock_session):
        """Test it logs error before raising exception."""
        # Setup mock response with error
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_response.text = "Server Error"
        
        # Setup mock session
        mock_session_instance = MagicMock()
        mock_session_instance.__enter__.return_value = mock_response
        mock_session.return_value.get.return_value = mock_session_instance
        
        # Create JSONStreamingRequest with a mock logger
        mock_logger = MagicMock(spec=logging.Logger)
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            logger=mock_logger
        )
        
        # Call start - should raise StreamingError
        with pytest.raises(StreamingError):
            request.start()
        
        # Verify error was logged
        mock_logger.error.assert_called_once()
        error_msg = mock_logger.error.call_args[0][0]
        assert "500" in error_msg


class Test_JSONStreamingRequest_Start_03_BoundaryBehaviors:
    """Test boundary behaviors of JSONStreamingRequest.start method."""
    
    @patch("requests.Session")
    def test_adds_range_header_when_resuming(self, mock_session):
        """Test it adds Range header when resuming from non-zero position."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.iter_content.return_value = [b"test"]
        mock_response.headers = {}
        
        # Setup mock session
        mock_session_instance = MagicMock()
        mock_session_instance.__enter__.return_value = mock_response
        mock_session.return_value.get.return_value = mock_session_instance
        
        # Create JSONStreamingRequest with non-zero position
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={"Authorization": "Bearer token123"}
        )
        request.position = 100  # Set non-zero position
        
        # Call start
        request.start()
        
        # Verify Range header was added
        headers_sent = mock_session.return_value.get.call_args[1]["headers"]
        assert "Range" in headers_sent
        assert headers_sent["Range"] == "bytes=100-"
    
    @patch("requests.Session")
    def test_handles_empty_response_chunks(self, mock_session):
        """Test it handles empty response chunks."""
        # Setup mock response with empty chunks
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.iter_content.return_value = [b"", b"data", b""]  # Empty chunks
        mock_response.headers = {}
        
        # Setup mock session
        mock_session_instance = MagicMock()
        mock_session_instance.__enter__.return_value = mock_response
        mock_session.return_value.get.return_value = mock_session_instance
        
        # Create JSONStreamingRequest with a spy on process_chunk
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={}
        )
        process_chunk_spy = MagicMock(wraps=request.process_chunk)
        request.process_chunk = process_chunk_spy
        
        # Call start
        request.start()
        
        # Verify process_chunk was called only for non-empty chunks
        assert process_chunk_spy.call_count == 1
        process_chunk_spy.assert_called_once_with(b"data")
    
    @patch("requests.Session")
    def test_handles_large_responses(self, mock_session):
        """Test it processes large responses in manageable chunks."""
        # Setup mock response with large content
        chunk_size = 1024
        large_chunk1 = b"x" * chunk_size
        large_chunk2 = b"y" * chunk_size
        
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.iter_content.return_value = [large_chunk1, large_chunk2]
        mock_response.headers = {"Content-Length": str(chunk_size * 2)}
        
        # Setup mock session
        mock_session_instance = MagicMock()
        mock_session_instance.__enter__.return_value = mock_response
        mock_session.return_value.get.return_value = mock_session_instance
        
        # Create JSONStreamingRequest with the same chunk size
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            chunk_size=chunk_size
        )
        
        # Call start
        request.start()
        
        # Verify response was processed in chunks
        mock_response.iter_content.assert_called_once_with(chunk_size=chunk_size)
        assert bytes(request.accumulated_data) == large_chunk1 + large_chunk2
        assert request.position == len(large_chunk1) + len(large_chunk2)
    
    @patch("requests.Session")
    def test_functions_without_content_length(self, mock_session):
        """Test it functions correctly without Content-Length header."""
        # Setup mock response without Content-Length
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_response.headers = {}  # No Content-Length
        
        # Setup mock session
        mock_session_instance = MagicMock()
        mock_session_instance.__enter__.return_value = mock_response
        mock_session.return_value.get.return_value = mock_session_instance
        
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={}
        )
        
        # Call start
        request.start()
        
        # Verify request was processed successfully
        assert bytes(request.accumulated_data) == b"chunk1chunk2"
        assert request.position == 12
        assert request.total_size is None  # Should be None without Content-Length


class Test_JSONStreamingRequest_Start_04_ErrorHandlingBehaviors:
    """Test error handling behaviors of JSONStreamingRequest.start method."""
    
    @patch("requests.Session")
    def test_catches_request_exception(self, mock_session):
        """Test it catches RequestException and wraps in StreamingError."""
        # Setup mock session to raise RequestException
        mock_session.return_value.get.side_effect = requests.RequestException("Connection failed")
        
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={}
        )
        
        # Call start - should raise StreamingError
        with pytest.raises(StreamingError) as excinfo:
            request.start()
        
        # Verify exception was properly wrapped
        assert "Connection failed" in str(excinfo.value)
    
    @patch("requests.Session")
    def test_saves_state_before_raising_exception(self, mock_session):
        """Test it saves state before raising exception for possible resumption."""
        # Setup mock session to raise RequestException
        mock_session.return_value.get.side_effect = requests.RequestException("Test error")
        
        # Create JSONStreamingRequest with a mock save_state method
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            state_file="/tmp/test-state.json"
        )
        mock_save_state = MagicMock()
        request.save_state = mock_save_state
        
        # Call start - should raise StreamingError
        with pytest.raises(StreamingError):
            request.start()
        
        # Verify save_state was called before raising exception
        mock_save_state.assert_called_once()
    
    @patch("requests.Session")
    def test_logs_detailed_error_information(self, mock_session):
        """Test it logs detailed error information."""
        # Setup mock session to raise RequestException
        error_message = "Detailed connection error"
        mock_session.return_value.get.side_effect = requests.RequestException(error_message)
        
        # Create JSONStreamingRequest with a mock logger
        mock_logger = MagicMock(spec=logging.Logger)
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            logger=mock_logger
        )
        
        # Call start - should raise StreamingError
        with pytest.raises(StreamingError):
            request.start()
        
        # Verify detailed error was logged
        mock_logger.error.assert_called_once()
        error_log = mock_logger.error.call_args[0][0]
        assert error_message in error_log


class Test_JSONStreamingRequest_Start_05_StateTransitionBehaviors:
    """Test state transition behaviors of JSONStreamingRequest.start method."""
    
    @patch("requests.Session")
    def test_transitions_from_not_started_to_completed(self, mock_session):
        """Test it transitions from 'not started' to 'in progress' to 'completed'."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.iter_content.return_value = [b"test"]
        mock_response.headers = {}
        
        # Setup mock session
        mock_session_instance = MagicMock()
        mock_session_instance.__enter__.return_value = mock_response
        mock_session.return_value.get.return_value = mock_session_instance
        
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={}
        )
        
        # Initial state should be not completed
        assert request.completed is False
        
        # Call start
        request.start()
        
        # Final state should be completed
        assert request.completed is True
    
    @patch("requests.Session")
    def test_updates_position_and_data_during_streaming(self, mock_session):
        """Test it updates position and accumulated_data throughout streaming."""
        # Setup mock response with multiple chunks
        chunks = [b"chunk1", b"chunk2", b"chunk3"]
        
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.iter_content.return_value = chunks
        mock_response.headers = {}
        
        # Setup mock session
        mock_session_instance = MagicMock()
        mock_session_instance.__enter__.return_value = mock_response
        mock_session.return_value.get.return_value = mock_session_instance
        
        # Create JSONStreamingRequest with a modified process_chunk to track states
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={}
        )
        
        # Track position after each chunk
        original_process_chunk = request.process_chunk
        positions = []
        accumulated_sizes = []
        
        def tracking_process_chunk(chunk):
            original_process_chunk(chunk)
            positions.append(request.position)
            accumulated_sizes.append(len(request.accumulated_data))
        
        request.process_chunk = tracking_process_chunk
        
        # Call start
        request.start()
        
        # Verify position was updated incrementally
        assert positions == [6, 12, 18]  # Length of each chunk
        assert accumulated_sizes == [6, 12, 18]  # Accumulated size after each chunk
    
    @patch("requests.Session")
    def test_sets_header_metadata_when_available(self, mock_session):
        """Test it sets total_size and etag when available."""
        # Setup mock response with headers
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.iter_content.return_value = [b"test"]
        mock_response.headers = {
            "Content-Length": "1024",
            "ETag": "\"abc123\""
        }
        
        # Setup mock session
        mock_session_instance = MagicMock()
        mock_session_instance.__enter__.return_value = mock_response
        mock_session.return_value.get.return_value = mock_session_instance
        
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={}
        )
        
        # Initial state
        assert request.total_size is None
        assert request.etag is None
        
        # Call start
        request.start()
        
        # Verify metadata was updated
        assert request.total_size == 1024
        assert request.etag == "\"abc123\""


class Test_JSONStreamingRequest_Resume_01_NominalBehaviors:
    """Test nominal behaviors of JSONStreamingRequest.resume method."""
    
    @patch.object(JSONStreamingRequest, "load_state")
    @patch.object(JSONStreamingRequest, "start")
    def test_loads_saved_state_correctly(self, mock_start, mock_load_state):
        """Test it loads saved state correctly."""
        # Setup mock state
        mock_state = MagicMock(spec=StreamingState)
        mock_load_state.return_value = mock_state
        
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            state_file="/tmp/test-state.json"
        )
        
        # Call resume
        request.resume()
        
        # Verify load_state was called
        mock_load_state.assert_called_once()
        # Verify start was called
        mock_start.assert_called_once()
    
    @patch.object(JSONStreamingRequest, "load_state")
    @patch.object(JSONStreamingRequest, "start")
    def test_adds_range_header_if_position_greater_than_zero(self, mock_start, mock_load_state):
        """Test it adds Range header if position > 0."""
        # Setup mock state
        mock_state = MagicMock(spec=StreamingState)
        mock_load_state.return_value = mock_state
        
        # Create JSONStreamingRequest with position > 0
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={"Original": "header"},
            state_file="/tmp/test-state.json"
        )
        request.position = 100
        
        # Call resume
        request.resume()
        
        # Verify Range header was added
        assert "Range" in request.headers
        assert request.headers["Range"] == "bytes=100-"
    
    @patch.object(JSONStreamingRequest, "load_state")
    @patch.object(JSONStreamingRequest, "start")
    def test_adds_if_match_header_if_etag_available(self, mock_start, mock_load_state):
        """Test it adds If-Match header if etag is available."""
        # Setup mock state
        mock_state = MagicMock(spec=StreamingState)
        mock_load_state.return_value = mock_state
        
        # Create JSONStreamingRequest with etag
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={"Original": "header"},
            state_file="/tmp/test-state.json"
        )
        request.etag = "\"abc123\""
        
        # Call resume
        request.resume()
        
        # Verify If-Match header was added
        assert "If-Match" in request.headers
        assert request.headers["If-Match"] == "\"abc123\""
    
    @patch.object(JSONStreamingRequest, "load_state")
    @patch.object(JSONStreamingRequest, "start")
    def test_calls_start_to_continue_request(self, mock_start, mock_load_state):
        """Test it calls start() to continue the request."""
        # Setup mock state
        mock_state = MagicMock(spec=StreamingState)
        mock_load_state.return_value = mock_state
        
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            state_file="/tmp/test-state.json"
        )
        
        # Call resume
        request.resume()
        
        # Verify start was called to continue the request
        mock_start.assert_called_once()
    
    @patch.object(JSONStreamingRequest, "load_state")
    @patch.object(JSONStreamingRequest, "start")
    def test_logs_information_about_resumption(self, mock_start, mock_load_state):
        """Test it logs information about resumption."""
        # Setup mock state
        mock_state = MagicMock(spec=StreamingState)
        mock_load_state.return_value = mock_state
        
        # Create JSONStreamingRequest with a mock logger
        mock_logger = MagicMock(spec=logging.Logger)
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            state_file="/tmp/test-state.json",
            logger=mock_logger,
            request_id="test-id"
        )
        request.position = 100
        
        # Call resume
        request.resume()
        
        # Verify resumption was logged
        mock_logger.info.assert_called_once()
        log_msg = mock_logger.info.call_args[0][0]
        assert "[test-id] Resuming" in log_msg
        assert "100" in log_msg  # Position should be in log message


class Test_JSONStreamingRequest_Resume_02_NegativeBehaviors:
    """Test negative behaviors of JSONStreamingRequest.resume method."""
    
    @patch.object(JSONStreamingRequest, "load_state", return_value=None)
    def test_raises_resume_error_if_loading_state_fails(self, mock_load_state):
        """Test it raises ResumeError if loading state fails."""
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            state_file="/tmp/test-state.json"
        )
        
        # Call resume - should raise ResumeError
        with pytest.raises(ResumeError) as excinfo:
            request.resume()
        
        # Verify error message
        assert "Failed to load state" in str(excinfo.value)
        
        # Verify load_state was called
        mock_load_state.assert_called_once()
    
    @patch.object(JSONStreamingRequest, "load_state")
    @patch.object(JSONStreamingRequest, "start", side_effect=StreamingError("Streaming failed"))
    def test_raises_resume_error_if_resumed_request_fails(self, mock_start, mock_load_state):
        """Test it raises ResumeError if resumed request fails."""
        # Setup mock state
        mock_state = MagicMock(spec=StreamingState)
        mock_load_state.return_value = mock_state
        
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            state_file="/tmp/test-state.json"
        )
        
        # Call resume - should raise ResumeError
        with pytest.raises(ResumeError) as excinfo:
            request.resume()
        
        # Verify error message
        assert "Failed to resume streaming request" in str(excinfo.value)
        assert "Streaming failed" in str(excinfo.value)
        
        # Verify start was called
        mock_start.assert_called_once()
    
    @patch.object(JSONStreamingRequest, "load_state")
    @patch.object(JSONStreamingRequest, "start")
    def test_logs_error_before_raising_exception(self, mock_start, mock_load_state):
        """Test it logs error before raising exception."""
        # Setup mock state
        mock_state = MagicMock(spec=StreamingState)
        mock_load_state.return_value = mock_state
        
        # Setup mock start to raise exception
        mock_start.side_effect = StreamingError("Streaming failed")
        
        # Create JSONStreamingRequest with a mock logger
        mock_logger = MagicMock(spec=logging.Logger)
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            state_file="/tmp/test-state.json",
            logger=mock_logger,
            request_id="test-id"
        )
        
        # Call resume - should raise ResumeError
        with pytest.raises(ResumeError):
            request.resume()
        
        # Verify error was logged
        mock_logger.error.assert_called_once()
        error_msg = mock_logger.error.call_args[0][0]
        assert "[test-id] Failed to resume" in error_msg
        assert "Streaming failed" in error_msg


class Test_JSONStreamingRequest_Resume_03_BoundaryBehaviors:
    """Test boundary behaviors of JSONStreamingRequest.resume method."""
    
    @patch.object(JSONStreamingRequest, "load_state")
    @patch.object(JSONStreamingRequest, "start")
    def test_handles_resuming_from_position_zero(self, mock_start, mock_load_state):
        """Test it handles resuming from position 0."""
        # Setup mock state
        mock_state = MagicMock(spec=StreamingState)
        mock_load_state.return_value = mock_state
        
        # Create JSONStreamingRequest with position = 0
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            state_file="/tmp/test-state.json"
        )
        request.position = 0  # Explicitly set to zero
        
        # Call resume
        request.resume()
        
        # Verify Range header was NOT added
        assert "Range" not in request.headers
        
        # Verify start was called
        mock_start.assert_called_once()
    
    @patch.object(JSONStreamingRequest, "load_state")
    @patch.object(JSONStreamingRequest, "start")
    def test_functions_without_etag(self, mock_start, mock_load_state):
        """Test it functions with or without etag."""
        # Setup mock state
        mock_state = MagicMock(spec=StreamingState)
        mock_load_state.return_value = mock_state
        
        # Create JSONStreamingRequest without etag
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            state_file="/tmp/test-state.json"
        )
        request.etag = None  # Explicitly set to None
        
        # Call resume
        request.resume()
        
        # Verify If-Match header was NOT added
        assert "If-Match" not in request.headers
        
        # Verify start was called
        mock_start.assert_called_once()
    
    @patch.object(JSONStreamingRequest, "load_state")
    @patch.object(JSONStreamingRequest, "start")
    def test_functions_with_partial_data(self, mock_start, mock_load_state):
        """Test it functions with partial or complete data."""
        # Setup mock state
        mock_state = MagicMock(spec=StreamingState)
        mock_load_state.return_value = mock_state
        
        # Create JSONStreamingRequest with some accumulated data
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            state_file="/tmp/test-state.json"
        )
        request.accumulated_data = bytearray(b"partial data")
        request.position = len(request.accumulated_data)
        
        # Call resume
        request.resume()
        
        # Verify Range header was added with correct position
        assert "Range" in request.headers
        assert request.headers["Range"] == f"bytes={len(b'partial data')}-"
        
        # Verify start was called
        mock_start.assert_called_once()


class Test_JSONStreamingRequest_Resume_04_ErrorHandlingBehaviors:
    """Test error handling behaviors of JSONStreamingRequest.resume method."""
    
    @patch.object(JSONStreamingRequest, "load_state", side_effect=Exception("Unexpected error"))
    def test_catches_exceptions_from_load_state(self, mock_load_state):
        """Test it catches exceptions from load_state() and raises ResumeError."""
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            state_file="/tmp/test-state.json"
        )
        
        # Call resume - should raise ResumeError
        with pytest.raises(ResumeError) as excinfo:
            request.resume()
        
        # Verify error message
        assert "Failed to load state" in str(excinfo.value)
        
        # Verify load_state was called
        mock_load_state.assert_called_once()
    
    @patch.object(JSONStreamingRequest, "load_state")
    @patch.object(JSONStreamingRequest, "start", side_effect=Exception("Unexpected error"))
    def test_catches_exceptions_from_start(self, mock_start, mock_load_state):
        """Test it catches exceptions from start() and wraps in ResumeError."""
        # Setup mock state
        mock_state = MagicMock(spec=StreamingState)
        mock_load_state.return_value = mock_state
        
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            state_file="/tmp/test-state.json"
        )
        
        # Call resume - should raise ResumeError
        with pytest.raises(ResumeError) as excinfo:
            request.resume()
        
        # Verify error message
        assert "Failed to resume streaming request" in str(excinfo.value)
        assert "Unexpected error" in str(excinfo.value)
        
        # Verify start was called
        mock_start.assert_called_once()
    
    @patch.object(JSONStreamingRequest, "load_state")
    @patch.object(JSONStreamingRequest, "start")
    def test_provides_detailed_error_context(self, mock_start, mock_load_state):
        """Test it provides detailed error context in exception message."""
        # Setup mock state
        mock_state = MagicMock(spec=StreamingState)
        mock_load_state.return_value = mock_state
        
        # Setup mock start to raise exception with detailed message
        error_msg = "Connection refused: detailed diagnostic information"
        mock_start.side_effect = StreamingError(error_msg)
        
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            state_file="/tmp/test-state.json"
        )
        
        # Call resume - should raise ResumeError
        with pytest.raises(ResumeError) as excinfo:
            request.resume()
        
        # Verify detailed error context is included
        assert "Failed to resume streaming request" in str(excinfo.value)
        assert error_msg in str(excinfo.value)


class Test_JSONStreamingRequest_Resume_05_StateTransitionBehaviors:
    """Test state transition behaviors of JSONStreamingRequest.resume method."""
    
    @patch.object(JSONStreamingRequest, "load_state")
    @patch.object(JSONStreamingRequest, "start")
    def test_transitions_from_saved_state_to_active(self, mock_start, mock_load_state):
        """Test it transitions from saved state to active streaming state."""
        # Setup mock state
        mock_state = MagicMock(spec=StreamingState)
        mock_load_state.return_value = mock_state
        
        # Create JSONStreamingRequest with initial state
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/original",  # Different from saved state
            headers={"Original": "header"},
            state_file="/tmp/test-state.json"
        )
        
        # Initial state
        assert not request.completed
        
        # Call resume
        request.resume()
        
        # Verify start was called to continue request
        mock_start.assert_called_once()
    
    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open)
    @patch.object(JSONStreamingRequest, "start")
    def test_maintains_data_integrity(self, mock_start, mock_file, mock_exists):
        """Test it maintains data integrity when resuming from interrupted download."""
        # Setup mock file with test state data
        state_data = json.dumps({
            "endpoint": "https://api.example.com/data",
            "method": "GET",
            "headers": {"Authorization": "Bearer token123"},
            "accumulated_data": "cGFydGlhbCBkYXRh",  # "partial data" in base64
            "last_position": 12,
            "total_size": 100,
            "etag": "etag123",
            "request_id": "saved-id"
        })
        mock_file.return_value.read.return_value = state_data
        
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/original",  # Different from saved state
            headers={"Original": "header"},
            state_file="/tmp/test-state.json"
        )
        
        # Ensure load_state actually works by not mocking it
        # We want to test real data loading and state transition
        
        # Call resume
        request.resume()
        
        # Verify state was properly loaded and integrity maintained
        assert request.endpoint == "https://api.example.com/data"  # From saved state
        assert request.headers["Authorization"] == "Bearer token123"  # From saved state
        assert request.headers["Range"] == "bytes=12-"  # Position from saved state
        assert bytes(request.accumulated_data) == b"partial data"
        assert request.position == 12
        assert request.total_size == 100
        assert request.etag == "etag123"
        assert request.request_id == "saved-id"
        
        # Verify start was called to continue request
        mock_start.assert_called_once()


class Test_JSONStreamingRequest_ProcessChunk_01_NominalBehaviors:
    """Test nominal behaviors of JSONStreamingRequest.process_chunk method."""
    
    def test_extends_accumulated_data(self):
        """Test it extends accumulated_data with new chunk."""
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={}
        )
        request.accumulated_data = bytearray(b"initial")
        
        # Process a chunk
        chunk = b"data"
        request.process_chunk(chunk)
        
        # Verify accumulated_data was extended
        assert bytes(request.accumulated_data) == b"initialdata"
    
    def test_updates_position_based_on_chunk_size(self):
        """Test it updates position based on chunk size."""
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={}
        )
        request.position = 10
        
        # Process a chunk
        chunk = b"test data"
        request.process_chunk(chunk)
        
        # Verify position was updated
        assert request.position == 10 + len(chunk)
    
    def test_logs_progress_for_large_responses(self):
        """Test it logs progress for large responses."""
        # Create JSONStreamingRequest with a mock logger
        mock_logger = MagicMock(spec=logging.Logger)
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            logger=mock_logger,
            request_id="test-id",
            chunk_size=10
        )
        request.total_size = 200  # Large enough to trigger progress logging
        
        # Process a chunk that will trigger logging (more than chunk_size * 10)
        chunk = b"x" * 100
        request.process_chunk(chunk)
        
        # Verify progress was logged
        mock_logger.debug.assert_called_once()
        log_msg = mock_logger.debug.call_args[0][0]
        assert "[test-id] Download progress" in log_msg
        assert "50.0%" in log_msg  # 100 / 200 = 50%
    
    def test_saves_state_periodically(self):
        """Test it saves state periodically for resumability."""
        # Create JSONStreamingRequest with a mock save_state method
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            state_file="/tmp/test-state.json",
            chunk_size=10
        )
        mock_save_state = MagicMock()
        request.save_state = mock_save_state
        
        # Process a chunk that should trigger state saving (more than chunk_size * 10)
        chunk = b"x" * 100
        request.process_chunk(chunk)
        
        # Verify save_state was called
        mock_save_state.assert_called_once()
    
    def test_processes_chunks_efficiently(self):
        """Test it processes chunks efficiently."""
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={}
        )
        
        # Process multiple chunks and measure performance
        chunk_size = 1024 * 1024  # 1MB
        chunk = b"x" * chunk_size
        
        # Process the chunk
        import time
        start_time = time.time()
        request.process_chunk(chunk)
        end_time = time.time()
        
        # Verify chunk was processed efficiently (should be very fast)
        processing_time = end_time - start_time
        assert processing_time < 0.1  # Should take less than 100ms for 1MB
        assert len(request.accumulated_data) == chunk_size


class Test_JSONStreamingRequest_ProcessChunk_03_BoundaryBehaviors:
    """Test boundary behaviors of JSONStreamingRequest.process_chunk method."""
    
    def test_handles_empty_chunks(self):
        """Test it handles empty chunks."""
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={}
        )
        request.position = 10
        
        # Process an empty chunk
        request.process_chunk(b"")
        
        # Verify position wasn't changed
        assert request.position == 10
        assert len(request.accumulated_data) == 0
    
    def test_handles_different_chunk_sizes(self):
        """Test it handles different chunk sizes."""
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={}
        )
        
        # Test with various chunk sizes
        chunk_sizes = [1, 10, 1024, 1024*1024]
        
        for size in chunk_sizes:
            request.accumulated_data = bytearray()
            request.position = 0
            
            chunk = b"x" * size
            request.process_chunk(chunk)
            
            # Verify chunk was processed correctly
            assert len(request.accumulated_data) == size
            assert request.position == size
    
    def test_functions_correctly_when_total_size_unknown(self):
        """Test it functions correctly when total_size is unknown."""
        # Create JSONStreamingRequest with a mock logger
        mock_logger = MagicMock(spec=logging.Logger)
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            logger=mock_logger,
            chunk_size=10
        )
        request.total_size = None  # Total size unknown
        
        # Process a large chunk
        chunk = b"x" * 100
        request.process_chunk(chunk)
        
        # Verify progress was not logged (no total_size to compare against)
        for call in mock_logger.debug.call_args_list:
            assert "Download progress" not in call[0][0]
        
        # Verify chunk was processed correctly
        assert len(request.accumulated_data) == 100
        assert request.position == 100
    
    def test_handles_very_large_data_incrementally(self):
        """Test it handles very large response data incrementally."""
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            chunk_size=1024
        )
        
        # Process multiple large chunks to simulate large response
        chunk = b"x" * 1024
        
        # Process 10 chunks for a total of 10KB
        for _ in range(10):
            request.process_chunk(chunk)
        
        # Verify all data was accumulated correctly
        assert len(request.accumulated_data) == 10 * 1024
        assert request.position == 10 * 1024


class Test_JSONStreamingRequest_ProcessChunk_04_ErrorHandlingBehaviors:
    """Test error handling behaviors of JSONStreamingRequest.process_chunk method."""
    
    def test_implicit_error_handling_via_save_state(self):
        """Test implicit error handling through save_state() method."""
        # Create JSONStreamingRequest with a mock save_state that raises exception
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            state_file="/tmp/test-state.json",
            chunk_size=10
        )
        mock_save_state = MagicMock(side_effect=Exception("Test error"))
        request.save_state = mock_save_state
        
        # Process a chunk large enough to trigger save_state
        chunk = b"x" * 100
        
        # Should not raise exception despite save_state failing
        try:
            request.process_chunk(chunk)
            exception_raised = False
        except:
            exception_raised = True
        
        # Verify no exception was propagated
        assert not exception_raised
        
        # Verify save_state was called
        mock_save_state.assert_called_once()
        
        # Verify chunk was still processed
        assert len(request.accumulated_data) == 100
        assert request.position == 100


class Test_JSONStreamingRequest_ProcessChunk_05_StateTransitionBehaviors:
    """Test state transition behaviors of JSONStreamingRequest.process_chunk method."""
    
    def test_incrementally_updates_position(self):
        """Test it incrementally updates position as chunks are processed."""
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={}
        )
        
        # Process multiple chunks
        chunks = [b"chunk1", b"chunk2", b"chunk3"]
        expected_positions = []
        position = 0
        
        for chunk in chunks:
            position += len(chunk)
            expected_positions.append(position)
            request.process_chunk(chunk)
            # Verify position after each chunk
            assert request.position == expected_positions[-1]
        
        # Verify final position
        assert request.position == sum(len(chunk) for chunk in chunks)
    
    def test_extends_accumulated_data_progressively(self):
        """Test it extends accumulated_data progressively."""
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={}
        )
        
        # Process multiple chunks
        chunks = [b"chunk1", b"chunk2", b"chunk3"]
        expected_data = b""
        
        for chunk in chunks:
            expected_data += chunk
            request.process_chunk(chunk)
            # Verify accumulated_data after each chunk
            assert bytes(request.accumulated_data) == expected_data
        
        # Verify final accumulated_data
        assert bytes(request.accumulated_data) == b"chunk1chunk2chunk3"
    
    def test_maintains_consistent_state(self):
        """Test it maintains consistent state during streaming."""
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={}
        )
        
        # Process multiple chunks
        chunks = [b"chunk1", b"chunk2", b"chunk3"]
        position = 0
        
        for chunk in chunks:
            position += len(chunk)
            request.process_chunk(chunk)
            # Verify consistency between position and accumulated_data length
            assert request.position == len(request.accumulated_data)
        
        # Verify final state consistency
        assert request.position == len(request.accumulated_data)
        assert request.position == len(b"chunk1chunk2chunk3")


class Test_JSONStreamingRequest_GetResult_01_NominalBehaviors:
    """Test nominal behaviors of JSONStreamingRequest.get_result method."""
    
    def test_verifies_request_completion(self):
        """Test it verifies request is completed before returning result."""
        # Create JSONStreamingRequest in completed state
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={}
        )
        request.completed = True
        request.accumulated_data = bytearray(b'{"key": "value"}') # Valid JSON
        
        # Call get_result - should not raise error
        try:
            request.get_result()
            error_raised = False
        except StreamingError:
            error_raised = True
        
        assert not error_raised, "StreamingError raised even though request was completed"

    def test_parses_accumulated_data_as_json(self):
        """Test it parses accumulated_data as UTF-8 encoded JSON."""
        # Create JSONStreamingRequest with JSON data
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={}
        )
        request.completed = True
        json_data = {"key": "value", "list": [1, 2, 3]}
        request.accumulated_data = bytearray(json.dumps(json_data).encode('utf-8'))
        
        # Call get_result
        result = request.get_result()
        
        # Verify result matches original JSON data
        assert result == json_data

    def test_returns_parsed_json_data(self):
        """Test it returns parsed JSON data."""
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={}
        )
        request.completed = True
        expected_result = {"message": "success"}
        request.accumulated_data = bytearray(json.dumps(expected_result).encode('utf-8'))
        
        # Call get_result
        actual_result = request.get_result()
        
        # Verify returned data
        assert actual_result == expected_result

    @patch("smartsurge.streaming.SmartSurgeTimer")
    def test_times_json_parsing_operation(self, mock_timer_constructor):
        """Test it times JSON parsing operation for performance tracking."""
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            request_id="test-timer-id"
        )
        request.completed = True
        request.accumulated_data = bytearray(b'{"data": "test"}')
        
        # Call get_result
        request.get_result()
        
        # Verify SmartSurgeTimer was used with correct context
        mock_timer_constructor.assert_called_once()
        args, kwargs = mock_timer_constructor.call_args
        assert args[0] == "json_parse.test-timer-id"
        assert len(args) >= 2  # Ensure there's at least 2 arguments
        assert isinstance(args[1], logging.Logger)  # Check second positional argument


class Test_JSONStreamingRequest_GetResult_02_NegativeBehaviors:
    """Test negative behaviors of JSONStreamingRequest.get_result method."""

    def test_raises_error_if_request_not_completed(self):
        """Test it raises StreamingError if request is not completed."""
        # Create JSONStreamingRequest in non-completed state
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={}
        )
        request.completed = False # Not completed
        
        # Call get_result - should raise StreamingError
        with pytest.raises(StreamingError) as excinfo:
            request.get_result()
        
        # Verify error message
        assert "Streaming request not completed" in str(excinfo.value)

    def test_raises_error_if_json_parsing_fails(self):
        """Test it raises StreamingError if JSON parsing fails."""
        # Create JSONStreamingRequest with invalid JSON data
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={}
        )
        request.completed = True
        request.accumulated_data = bytearray(b"this is not valid json")
        
        # Call get_result - should raise StreamingError
        with pytest.raises(StreamingError) as excinfo:
            request.get_result()
        
        # Verify error message contains JSON parsing failure
        assert "Failed to parse JSON" in str(excinfo.value)

    def test_logs_detailed_error_information_on_parse_failure(self):
        """Test it logs detailed error information when JSON parsing fails."""
        # Create JSONStreamingRequest with a mock logger
        mock_logger = MagicMock(spec=logging.Logger)
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            logger=mock_logger,
            request_id="test-parse-fail-id"
        )
        request.completed = True
        invalid_json_data = b"invalid { json"
        request.accumulated_data = bytearray(invalid_json_data)
        
        # Call get_result - should raise StreamingError
        with pytest.raises(StreamingError):
            request.get_result()
        
        # Verify error was logged
        error_log_call = None
        debug_log_call = None
        for call_args in mock_logger.error.call_args_list:
            if "[test-parse-fail-id] Failed to parse JSON" in call_args[0][0]:
                error_log_call = call_args
                break
        
        for call_args in mock_logger.debug.call_args_list:
             if "[test-parse-fail-id] JSON sample" in call_args[0][0]:
                debug_log_call = call_args
                break

        assert error_log_call is not None, "Error message not logged"
        assert debug_log_call is not None, "Debug sample not logged"
        
        # Verify sample of data was logged for debugging
        logged_sample = debug_log_call[0][0]
        assert invalid_json_data.decode('utf-8', errors='replace')[:200] in logged_sample


class Test_JSONStreamingRequest_GetResult_03_BoundaryBehaviors:
    """Test boundary behaviors of JSONStreamingRequest.get_result method."""

    def test_handles_empty_json_data(self):
        """Test it handles empty JSON data (e.g., {} or [])."""
        # Test with empty object
        request_obj = JSONStreamingRequest(endpoint="https://api.example.com/data", headers={})
        request_obj.completed = True
        request_obj.accumulated_data = bytearray(b'{}')
        assert request_obj.get_result() == {}
        
        # Test with empty array
        request_arr = JSONStreamingRequest(endpoint="https://api.example.com/data", headers={})
        request_arr.completed = True
        request_arr.accumulated_data = bytearray(b'[]')
        assert request_arr.get_result() == []

    def test_handles_large_json_structures(self):
        """Test it handles large JSON structures."""
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={}
        )
        request.completed = True
        
        # Create a large JSON object (e.g., list of 10000 items)
        large_json_data = {"data": list(range(10000))}
        request.accumulated_data = bytearray(json.dumps(large_json_data).encode('utf-8'))
        
        # Call get_result
        result = request.get_result()
        
        # Verify result matches original large JSON data
        assert result == large_json_data

    def test_strips_whitespace_from_json_data(self):
        """Test it strips whitespace from JSON data before parsing."""
        # Create JSONStreamingRequest with whitespace around JSON
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={}
        )
        request.completed = True
        json_data = {"key": "value"}
        # Add leading/trailing whitespace
        request.accumulated_data = bytearray(f"  \n {json.dumps(json_data)} \t ".encode('utf-8'))
        
        # Call get_result
        result = request.get_result()
        
        # Verify result matches original JSON data
        assert result == json_data


class Test_JSONStreamingRequest_GetResult_04_ErrorHandlingBehaviors:
    """Test error handling behaviors of JSONStreamingRequest.get_result method."""

    def test_catches_json_decode_error_and_wraps(self):
        """Test it catches JSONDecodeError and wraps in StreamingError."""
        # Create JSONStreamingRequest with invalid JSON
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={}
        )
        request.completed = True
        request.accumulated_data = bytearray(b"{'malformed': json,}") # Malformed JSON
        
        # Call get_result - should raise StreamingError
        with pytest.raises(StreamingError) as excinfo:
            request.get_result()
        
        # Verify the original exception was a JSONDecodeError
        assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)
        assert "Failed to parse JSON" in str(excinfo.value)

    def test_logs_sample_of_failed_json_data(self):
        """Test it logs a sample of the failed JSON data for debugging."""
        # Create JSONStreamingRequest with a mock logger
        mock_logger = MagicMock(spec=logging.Logger)
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={},
            logger=mock_logger,
            request_id="test-debug-sample-id"
        )
        request.completed = True
        # Create data that's longer than the sample to test truncation
        failed_json_data = bytearray(b"This is some very long invalid json data that will be truncated in the log sample " * 10)
        request.accumulated_data = failed_json_data
        
        # Call get_result
        with pytest.raises(StreamingError):
            request.get_result()
        
        # Verify debug log call contains a sample
        debug_log_call = None
        for call_args in mock_logger.debug.call_args_list:
            if "[test-debug-sample-id] JSON sample:" in call_args[0][0]:
                debug_log_call = call_args
                break
        
        assert debug_log_call is not None, "Debug sample not logged"
        
        logged_sample_text = debug_log_call[0][0]
        expected_sample = failed_json_data[:200].decode('utf-8', errors='replace')
        assert expected_sample in logged_sample_text
        assert "..." in logged_sample_text # Indicates truncation

    def test_provides_meaningful_error_messages(self):
        """Test it provides meaningful error messages with context."""
        # Test for "not completed" error
        request_not_completed = JSONStreamingRequest(endpoint="e1", headers={}, request_id="nc-id")
        with pytest.raises(StreamingError) as excinfo_nc:
            request_not_completed.get_result()
        assert "Streaming request not completed" in str(excinfo_nc.value)
        
        # Test for JSON parsing error
        request_parse_error = JSONStreamingRequest(endpoint="e2", headers={}, request_id="pe-id")
        request_parse_error.completed = True
        request_parse_error.accumulated_data = bytearray(b"bad-json")
        with pytest.raises(StreamingError) as excinfo_pe:
            request_parse_error.get_result()
        assert "Failed to parse JSON" in str(excinfo_pe.value)
        # Check if original JSONDecodeError message is included
        assert "bad-json" not in str(excinfo_pe.value.__cause__) # The actual error message will be more specific


class Test_JSONStreamingRequest_GetResult_05_StateTransitionBehaviors:
    """Test state transition behaviors of JSONStreamingRequest.get_result method."""

    def test_maintains_read_only_behavior(self):
        """Test it maintains read-only behavior without changing object state."""
        # Create JSONStreamingRequest
        request = JSONStreamingRequest(
            endpoint="https://api.example.com/data",
            headers={"X-Test": "Header"},
            params={"q": "test"},
            request_id="test-state-id"
        )
        request.completed = True
        request.accumulated_data = bytearray(b'{"result": "data"}')
        request.position = len(request.accumulated_data)
        request.total_size = request.position
        request.etag = "test-etag"
        
        # Store original state
        original_vars = vars(request).copy()
        
        # Call get_result
        request.get_result()
        
        # Verify object state hasn't changed
        current_vars = vars(request).copy()
        
        # Remove logger from comparison as it might be a mock or child logger
        original_vars.pop('logger', None)
        current_vars.pop('logger', None)
        # response is not part of the core state to preserve
        original_vars.pop('response', None)
        current_vars.pop('response', None)

        assert original_vars == current_vars, "Object state changed after calling get_result"
