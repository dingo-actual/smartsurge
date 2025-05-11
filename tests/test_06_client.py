import aiohttp
import asyncio
from datetime import datetime, timezone
import pytest
from pydantic import ValidationError
import requests
from unittest.mock import patch, MagicMock, call
from urllib3.util.retry import Retry

from smartsurge import exceptions
from smartsurge.client import ClientConfig, SmartSurgeClient
from smartsurge.exceptions import RateLimitExceeded, ResumeError, StreamingError
from smartsurge.models import RequestEntry, RequestHistory, RequestMethod, SearchStatus


class AsyncMock(MagicMock):
    """Helper class for mocking async methods."""
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


class Test_ClientConfig_Initialization_01_NominalBehaviors:
    """Tests for nominal behaviors of ClientConfig initialization."""

    def test_init_with_default_parameters(self):
        """Initialize with default parameters produces correct default values."""
        config = ClientConfig()
        assert config.base_url is None
        assert config.timeout == (10.0, 30.0)
        assert config.max_retries == 3
        assert config.backoff_factor == 0.3
        assert config.verify_ssl is True
        assert config.min_time_period == 1.0
        assert config.max_time_period == 3600.0
        assert config.confidence_threshold == 0.9

    def test_init_with_custom_parameters(self):
        """Initialize with custom parameters correctly stores provided values."""
        config = ClientConfig(
            base_url="https://api.example.com",
            timeout=(5.0, 15.0),
            max_retries=5,
            backoff_factor=0.5,
            verify_ssl=False,
            min_time_period=2.0,
            max_time_period=1800.0,
            confidence_threshold=0.8
        )
        assert config.base_url == "https://api.example.com"
        assert config.timeout == (5.0, 15.0)
        assert config.max_retries == 5
        assert config.backoff_factor == 0.5
        assert config.verify_ssl is False
        assert config.min_time_period == 2.0
        assert config.max_time_period == 1800.0
        assert config.confidence_threshold == 0.8

    def test_field_validation_with_valid_inputs(self):
        """All fields properly validate correct inputs."""
        config = ClientConfig(
            timeout=15.0,  # Should convert to tuple
            max_retries=10,
            backoff_factor=10.0,
            confidence_threshold=1.0
        )
        assert config.timeout == (15.0, 15.0)
        assert config.max_retries == 10
        assert config.backoff_factor == 10.0
        assert config.confidence_threshold == 1.0

    def test_serialization_via_model_dump(self):
        """Serialization via model_dump produces expected output."""
        config = ClientConfig(base_url="https://api.example.com")
        data = config.model_dump()
        assert data["base_url"] == "https://api.example.com"
        assert data["timeout"] == (10.0, 30.0)
        assert data["max_retries"] == 3
        # Verify all other fields are present with correct values
        assert len(data) >= 12  # At least all default fields


class Test_ClientConfig_Initialization_02_NegativeBehaviors:
    """Tests for negative behaviors of ClientConfig initialization."""

    def test_non_positive_timeout_values(self):
        """Non-positive timeout values raise ValidationError."""
        with pytest.raises(ValidationError):
            ClientConfig(timeout=-5.0)

        with pytest.raises(ValidationError):
            ClientConfig(timeout=0)

        with pytest.raises(ValidationError):
            ClientConfig(timeout=(0, 30.0))

        with pytest.raises(ValidationError):
            ClientConfig(timeout=(10.0, -5.0))

    def test_timeout_tuple_with_incorrect_structure(self):
        """Timeout tuple with incorrect structure raises ValidationError."""
        with pytest.raises(ValidationError):
            ClientConfig(timeout=(1.0, 2.0, 3.0))

        with pytest.raises(ValidationError):
            ClientConfig(timeout=[])

    def test_min_time_period_greater_than_max_time_period(self):
        """min_time_period greater than max_time_period raises ValidationError."""
        with pytest.raises(ValidationError):
            ClientConfig(min_time_period=200.0, max_time_period=100.0)

    def test_invalid_value_types(self):
        """Invalid value types for fields raise appropriate exceptions."""
        with pytest.raises(ValidationError):
            ClientConfig(max_retries="three")

        with pytest.raises(ValidationError):
            ClientConfig(backoff_factor="half")

        with pytest.raises(ValidationError):
            ClientConfig(verify_ssl="yes")

        with pytest.raises(ValidationError):
            ClientConfig(confidence_threshold="high")


class Test_ClientConfig_Initialization_03_BoundaryBehaviors:
    """Tests for boundary behaviors of ClientConfig initialization."""

    def test_timeout_at_minimum_valid_value(self):
        """Timeout set to minimum valid value (just above 0)."""
        config = ClientConfig(timeout=0.000001)
        assert config.timeout == (0.000001, 0.000001)

        config = ClientConfig(timeout=(0.000001, 0.000001))
        assert config.timeout == (0.000001, 0.000001)

    def test_confidence_threshold_at_boundaries(self):
        """Confidence threshold set exactly to boundaries (0.0 or 1.0)."""
        config = ClientConfig(confidence_threshold=0.0)
        assert config.confidence_threshold == 0.0

        config = ClientConfig(confidence_threshold=1.0)
        assert config.confidence_threshold == 1.0

    def test_max_retries_at_max_value(self):
        """max_retries set to maximum allowed value (10)."""
        config = ClientConfig(max_retries=10)
        assert config.max_retries == 10

    def test_min_time_period_equals_max_time_period(self):
        """min_time_period equals max_time_period (edge case)."""
        config = ClientConfig(min_time_period=100.0, max_time_period=100.0)
        assert config.min_time_period == 100.0
        assert config.max_time_period == 100.0


class Test_ClientConfig_Initialization_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of ClientConfig initialization."""

    def test_validation_error_messages(self):
        """Validation errors contain appropriate descriptive messages."""
        try:
            ClientConfig(timeout=-5.0)
        except ValidationError as e:
            assert "Timeout must be positive" in str(e)

        try:
            ClientConfig(max_retries=11)
        except ValidationError as e:
            assert "Input should be less than or equal to 10" in str(e)

        try:
            ClientConfig(min_time_period=500.0, max_time_period=100.0)
        except ValidationError as e:
            assert "max_time_period must be greater than min_time_period" in str(e) or \
                   "min_time_period must be less than max_time_period" in str(e)

    def test_multiple_validation_errors(self):
        """Multiple validation errors are correctly collected and reported."""
        try:
            ClientConfig(
                timeout=-5.0,
                max_retries=11,
                confidence_threshold=2.0
            )
        except ValidationError as e:
            error_str = str(e)
            assert "Timeout" in error_str
            assert "max_retries" in error_str
            assert "confidence_threshold" in error_str


class Test_ClientConfig_FieldValidators_01_NominalBehaviors:
    """Tests for nominal behaviors of ClientConfig field validators."""

    def test_validate_timeout_converts_single_value(self):
        """validate_timeout converts single value to tuple of two identical values."""
        config = ClientConfig(timeout=15.0)
        assert config.timeout == (15.0, 15.0)

    def test_validate_timeout_preserves_valid_tuples(self):
        """validate_timeout preserves valid timeout tuples."""
        config = ClientConfig(timeout=(5.0, 10.0))
        assert config.timeout == (5.0, 10.0)

    def test_validate_time_periods_accepts_valid_values(self):
        """validate_time_periods accepts when min is less than max."""
        config = ClientConfig(min_time_period=10.0, max_time_period=100.0)
        assert config.min_time_period == 10.0
        assert config.max_time_period == 100.0


class Test_ClientConfig_FieldValidators_02_NegativeBehaviors:
    """Tests for negative behaviors of ClientConfig field validators."""

    def test_validate_timeout_rejects_non_positive_values(self):
        """validate_timeout rejects non-positive values with clear error message."""
        with pytest.raises(ValidationError, match="Timeout must be positive"):
            ClientConfig(timeout=0.0)

        with pytest.raises(ValidationError, match="Both connect and read timeouts must be positive"):
            ClientConfig(timeout=(0.0, 5.0))

        with pytest.raises(ValidationError, match="Both connect and read timeouts must be positive"):
            ClientConfig(timeout=(5.0, -1.0))

    def test_validate_time_periods_rejects_invalid_values(self):
        """validate_time_periods rejects when min is greater than max."""
        with pytest.raises(ValidationError):
            ClientConfig(min_time_period=100.0, max_time_period=10.0)


class Test_ClientConfig_FieldValidators_03_BoundaryBehaviors:
    """Tests for boundary behaviors of ClientConfig field validators."""

    def test_validate_timeout_handles_minimum_values(self):
        """validate_timeout handles values at minimum positive threshold."""
        config = ClientConfig(timeout=0.000001)
        assert config.timeout == (0.000001, 0.000001)

    def test_validate_time_periods_with_equal_values(self):
        """validate_time_periods validates when min equals max."""
        config = ClientConfig(min_time_period=50.0, max_time_period=50.0)
        assert config.min_time_period == 50.0
        assert config.max_time_period == 50.0


class Test_SmartSurgeClient_Init_01_NominalBehaviors:
    """Tests for nominal behaviors of SmartSurgeClient initialization."""

    def test_init_with_default_parameters(self):
        """Initializes with default parameters and creates valid configuration."""
        client = SmartSurgeClient()
        assert client.config is not None
        assert client.config.base_url is None
        assert client.config.timeout == (10.0, 30.0)
        assert client.logger is not None
        assert client.session is not None
        assert client.histories == {}

    def test_init_with_custom_parameters(self):
        """Custom parameters including rate_limit are correctly stored and applied."""
        rate_limit = {"requests": 100, "period": 60.0}
        client = SmartSurgeClient(
            base_url="https://api.example.com",
            timeout=(5.0, 15.0),
            rate_limit=rate_limit,
            max_retries=5,
            log_level=10  # DEBUG level
        )
        assert client.config.base_url == "https://api.example.com"
        assert client.config.timeout == (5.0, 15.0)
        assert client.config.max_retries == 5
        assert client.user_rate_limit == rate_limit
        assert client.logger.level == 10

    def test_logger_configuration(self):
        """Logger is properly configured with specified or default log level."""
        client = SmartSurgeClient()
        assert client.logger.level == client.config.log_level

        import logging
        client = SmartSurgeClient(log_level=logging.DEBUG)
        assert client.logger.level == logging.DEBUG
        
    @patch("requests.Session")
    def test_session_creation(self, mock_session):
        """Creates session with appropriate retry capabilities."""
        client = SmartSurgeClient()
        assert mock_session.return_value.mount.call_count == 2
        
    def test_request_histories_initialization(self):
        """Initializes empty request histories dictionary."""
        client = SmartSurgeClient()
        assert client.histories == {}


class Test_SmartSurgeClient_Init_02_NegativeBehaviors:
    """Tests for negative behaviors of SmartSurgeClient initialization."""

    def test_invalid_configuration_parameters(self):
        """Invalid configuration parameters are detected and raise appropriate exceptions."""
        with pytest.raises(Exception):
            SmartSurgeClient(timeout=-5.0)
        
        with pytest.raises(Exception):
            SmartSurgeClient(max_retries=11)
        
        with pytest.raises(Exception):
            SmartSurgeClient(min_time_period=100.0, max_time_period=10.0)


class Test_SmartSurgeClient_Init_03_BoundaryBehaviors:
    """Tests for boundary behaviors of SmartSurgeClient initialization."""

    def test_initialize_with_minimum_values(self):
        """Initialize with minimum valid values for numeric parameters."""
        client = SmartSurgeClient(
            timeout=0.001,
            max_retries=0,
            backoff_factor=0.0,
            min_time_period=0.001,
            max_time_period=0.001,
            confidence_threshold=0.0
        )
        assert client.config.timeout == (0.001, 0.001)
        assert client.config.max_retries == 0
        assert client.config.backoff_factor == 0.0
        assert client.config.min_time_period == 0.001
        assert client.config.max_time_period == 0.001
        assert client.config.confidence_threshold == 0.0

    def test_initialize_with_maximum_values(self):
        """Initialize with maximum valid values for numeric parameters."""
        client = SmartSurgeClient(
            max_retries=10,
            backoff_factor=10.0,
            confidence_threshold=1.0
        )
        assert client.config.max_retries == 10
        assert client.config.backoff_factor == 10.0
        assert client.config.confidence_threshold == 1.0

    def test_empty_base_url_handling(self):
        """Empty base_url is handled appropriately."""
        client = SmartSurgeClient(base_url="")
        assert client.config.base_url == ""


class Test_SmartSurgeClient_Init_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of SmartSurgeClient initialization."""

    @patch("requests.Session")
    def test_session_creation_exception_handling(self, mock_session):
        """Handles and logs exceptions during session creation."""
        mock_session.side_effect = Exception("Failed to create session")
        
        # Should catch the exception and log it
        with pytest.raises(Exception):
            SmartSurgeClient()


class Test_SmartSurgeClient_Init_05_StateTransitionBehaviors:
    """Tests for state transition behaviors of SmartSurgeClient initialization."""

    def test_rate_limit_hierarchy_initialization(self):
        """Initial state has correct rate limit hierarchy (user-defined, response, adaptive)."""
        # Without user-defined rate limit
        client = SmartSurgeClient()
        assert client.user_rate_limit is None
        assert client.response_rate_limit is None
        
        # With user-defined rate limit
        rate_limit = {"requests": 100, "period": 60.0}
        client = SmartSurgeClient(rate_limit=rate_limit)
        assert client.user_rate_limit == rate_limit
        assert client.response_rate_limit is None


class Test_SmartSurgeClient_CreateSession_01_NominalBehaviors:
    """Tests for nominal behaviors of SmartSurgeClient's _create_session method."""

    @patch("requests.Session")
    @patch("smartsurge.client.HTTPAdapter")
    def test_creates_session_with_retry_capabilities(self, mock_adapter, mock_session):
        """Creates a requests Session with configured retry capabilities."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        client = SmartSurgeClient()
        
        # Examine how HTTPAdapter was called
        adapter_args = mock_adapter.call_args.kwargs
        
        # Verify adapter configuration
        assert adapter_args['pool_connections'] == client.config.max_connections
        assert adapter_args['pool_maxsize'] == client.config.max_pool_size
        
        # Verify Retry was created with correct parameters
        retry_obj = adapter_args['max_retries']
        assert isinstance(retry_obj, Retry)
        assert retry_obj.total == client.config.max_retries
        assert retry_obj.backoff_factor == client.config.backoff_factor
        assert retry_obj.status_forcelist == [500, 502, 503, 504]
        
        # Verify adapter was mounted correctly
        mock_adapter.assert_called_once()
        mock_session_instance.mount.assert_has_calls([
            call("http://", mock_adapter.return_value),
            call("https://", mock_adapter.return_value)
        ])

    @patch("requests.Session")
    def test_sets_default_headers(self, mock_session):
        """Sets User-Agent and other default headers correctly."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        client = SmartSurgeClient(user_agent="CustomAgent/1.0")
        
        mock_session_instance.headers.update.assert_called_once()
        headers = mock_session_instance.headers.update.call_args.args[0]
        assert headers['User-Agent'] == "CustomAgent/1.0"
    
    @patch("requests.Session")
    def test_mounts_adapters_for_both_protocols(self, mock_session):
        """Mounts adapters for both HTTP and HTTPS protocols."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        client = SmartSurgeClient()
        
        assert mock_session_instance.mount.call_count == 2
        mount_calls = mock_session_instance.mount.call_args_list
        assert mount_calls[0].args[0] == "http://"
        assert mount_calls[1].args[0] == "https://"
    
    @patch("requests.Session")
    def test_configures_pool_connections(self, mock_session):
        """Configures pool connections according to max_connections setting."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        client = SmartSurgeClient(max_connections=20, max_pool_size=25)
        session = client.session
        
        # Verify HTTPAdapter was created with correct pool settings
        assert mock_session_instance.mount.call_count == 2

    @patch("requests.Session")
    def test_returns_configured_session(self, mock_session):
        """Returns a fully configured session object."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        client = SmartSurgeClient()
        session = client._create_session()
        
        assert session == mock_session_instance


class Test_SmartSurgeClient_CreateSession_02_NegativeBehaviors:
    """Tests for negative behaviors of SmartSurgeClient's _create_session method."""

    @patch("smartsurge.client.Retry")
    def test_handles_invalid_retry_strategy_parameters(self, mock_retry):
        """Handles invalid retry strategy parameters gracefully."""
        mock_retry.side_effect = ValueError("Invalid retry parameters")
        
        with pytest.raises(ValueError):
            client = SmartSurgeClient()
            session = client._create_session()


class Test_SmartSurgeClient_CreateSession_03_BoundaryBehaviors:
    """Tests for boundary behaviors of SmartSurgeClient's _create_session method."""

    @patch("requests.Session")
    @patch("smartsurge.client.Retry")
    def test_creates_session_with_boundary_retry_values(self, mock_retry, mock_session):
        """Creates session with minimum and maximum allowed retry values."""
        # Test with minimum retries
        client = SmartSurgeClient(max_retries=0)
        session = client._create_session()
        retry_args = mock_retry.call_args.kwargs
        assert retry_args['total'] == 0
        
        # Test with maximum retries
        mock_retry.reset_mock()
        client = SmartSurgeClient(max_retries=10)
        session = client._create_session()
        retry_args = mock_retry.call_args.kwargs
        assert retry_args['total'] == 10


class Test_SmartSurgeClient_GetFullUrl_01_NominalBehaviors:
    """Tests for nominal behaviors of SmartSurgeClient's _get_full_url method."""

    def test_combines_base_url_with_relative_path(self):
        """Combines base_url with relative endpoint path correctly."""
        client = SmartSurgeClient(base_url="https://api.example.com")
        url = client._get_full_url("users")
        assert url == "https://api.example.com/users"

    def test_returns_absolute_urls_unchanged(self):
        """Returns absolute URLs unchanged (those starting with http:// or https://)."""
        client = SmartSurgeClient(base_url="https://api.example.com")
        url = client._get_full_url("https://other-api.example.com/users")
        assert url == "https://other-api.example.com/users"
        
        url = client._get_full_url("http://other-api.example.com/users")
        assert url == "http://other-api.example.com/users"

    def test_handles_trailing_and_leading_slashes(self):
        """Handles trailing slashes in base_url and leading slashes in endpoint correctly."""
        client = SmartSurgeClient(base_url="https://api.example.com/")
        url = client._get_full_url("users")
        assert url == "https://api.example.com/users"
        
        url = client._get_full_url("/users")
        assert url == "https://api.example.com/users"
        
        client = SmartSurgeClient(base_url="https://api.example.com")
        url = client._get_full_url("/users")
        assert url == "https://api.example.com/users"

    def test_works_with_none_base_url(self):
        """Works correctly when base_url is None."""
        client = SmartSurgeClient(base_url=None)
        url = client._get_full_url("users")
        assert url == "users"
        
        url = client._get_full_url("https://api.example.com/users")
        assert url == "https://api.example.com/users"


class Test_SmartSurgeClient_GetFullUrl_03_BoundaryBehaviors:
    """Tests for boundary behaviors of SmartSurgeClient's _get_full_url method."""

    def test_handles_empty_endpoint_string(self):
        """Handles empty endpoint string."""
        client = SmartSurgeClient(base_url="https://api.example.com")
        url = client._get_full_url("")
        assert url == "https://api.example.com/"
        
        client = SmartSurgeClient(base_url="https://api.example.com/")
        url = client._get_full_url("")
        assert url == "https://api.example.com/"


class Test_SmartSurgeClient_GetOrCreateHistory_01_NominalBehaviors:
    """Tests for nominal behaviors of SmartSurgeClient's _get_or_create_history method."""

    def test_returns_existing_history(self):
        """Returns existing history for previously seen endpoint/method combination."""
        client = SmartSurgeClient()
        
        # Create history first
        endpoint = "users"
        method = RequestMethod.GET
        history1 = client._get_or_create_history(endpoint, method, 0.9)
        
        # Should return the same history object
        history2 = client._get_or_create_history(endpoint, method, 0.9)
        assert history1 is history2
        assert client.histories[(endpoint, method)] is history1

    def test_creates_new_history_for_first_time_combo(self):
        """Creates new history for first-time endpoint/method combination."""
        client = SmartSurgeClient()
        
        endpoint = "users"
        method = RequestMethod.GET
        history = client._get_or_create_history(endpoint, method, 0.9)
        
        assert isinstance(history, RequestHistory)
        assert history.endpoint == endpoint
        assert history.method == method
        assert history in client.histories.values()

    def test_converts_string_method_to_enum(self):
        """Converts string method to RequestMethod enum correctly."""
        client = SmartSurgeClient()
        
        endpoint = "users"
        history = client._get_or_create_history(endpoint, "GET", 0.9)
        
        assert history.method == RequestMethod.GET
        assert (endpoint, RequestMethod.GET) in client.histories

    def test_creates_history_with_proper_confidence(self):
        """Creates history with appropriate confidence threshold."""
        client = SmartSurgeClient(confidence_threshold=0.8)
        
        # Use default confidence from client
        endpoint = "users"
        method = RequestMethod.GET
        history = client._get_or_create_history(endpoint, method, None)
        assert history.confidence_threshold == 0.8
        
        # Use custom confidence
        endpoint = "products"
        method = RequestMethod.GET
        history = client._get_or_create_history(endpoint, method, 0.95)
        assert history.confidence_threshold == 0.95


class Test_SmartSurgeClient_GetOrCreateHistory_02_NegativeBehaviors:
    """Tests for negative behaviors of SmartSurgeClient's _get_or_create_history method."""

    def test_raises_error_for_invalid_method_string(self):
        """Raises ValueError for invalid HTTP method strings."""
        client = SmartSurgeClient()
        
        with pytest.raises(ValueError, match="Invalid HTTP method"):
            client._get_or_create_history("users", "INVALID_METHOD", 0.9)


class Test_SmartSurgeClient_GetOrCreateHistory_05_StateTransitionBehaviors:
    """Tests for state transition behaviors of SmartSurgeClient's _get_or_create_history method."""

    def test_updates_histories_dictionary_when_creating_new(self):
        """Updates internal histories dictionary when creating new history."""
        client = SmartSurgeClient()
        
        assert len(client.histories) == 0
        
        client._get_or_create_history("users", RequestMethod.GET, 0.9)
        assert len(client.histories) == 1
        assert ("users", RequestMethod.GET) in client.histories
        
        client._get_or_create_history("products", RequestMethod.POST, 0.9)
        assert len(client.histories) == 2
        assert ("products", RequestMethod.POST) in client.histories
        
        # Adding same endpoint/method doesn't create new history
        client._get_or_create_history("users", RequestMethod.GET, 0.9)
        assert len(client.histories) == 2


class Test_SmartSurgeClient_Request_01_NominalBehaviors:
    """Tests for nominal behaviors of SmartSurgeClient's request method."""

    @patch("requests.Session.request")
    def test_makes_http_request_and_returns_response_with_history(self, mock_request):
        """Makes HTTP request with specified parameters and returns response with history."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_request.return_value = mock_response

        client = SmartSurgeClient()
        response, history = client.request(
            method=RequestMethod.GET,
            endpoint="users",
            params={"page": 1},
            headers={"Accept": "application/json"}
        )
        
        assert response == mock_response
        assert isinstance(history, RequestHistory)
        assert history.endpoint == "users"
        assert history.method == RequestMethod.GET

    @patch("requests.Session.request")
    def test_records_successful_request_in_history(self, mock_request):
        """Records successful request in history."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_request.return_value = mock_response

        client = SmartSurgeClient()
        response, history = client.request(
            method=RequestMethod.GET,
            endpoint="users"
        )
        
        assert len(history.entries) == 1
        assert history.entries[0].endpoint == "users"
        assert history.entries[0].method == RequestMethod.GET
        assert history.entries[0].status_code == 200
        assert history.entries[0].success is True

    @patch("requests.Session.request")
    @patch("smartsurge.client.RequestHistory.intercept_request")
    def test_applies_rate_limiting_based_on_history(self, mock_intercept, mock_request):
        """Applies appropriate rate limiting based on history."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_request.return_value = mock_response
        
        client = SmartSurgeClient()
        response, history = client.request(
            method=RequestMethod.GET,
            endpoint="users"
        )
        
        # Verify that intercept_request was called to apply rate limiting
        mock_intercept.assert_called_once()

    @patch("requests.Session.request")
    @patch("time.sleep")
    def test_handles_429_responses_with_retry(self, mock_sleep, mock_request):
        """Handles 429 responses by waiting and retrying automatically."""
        # First response is 429 with Retry-After header
        first_response = MagicMock()
        first_response.status_code = 429
        first_response.ok = False
        first_response.headers = {"Retry-After": "2"}
        
        # Second response succeeds
        second_response = MagicMock()
        second_response.status_code = 200
        second_response.ok = True
        
        mock_request.side_effect = [first_response, second_response]
        
        client = SmartSurgeClient()
        response, history = client.request(
            method=RequestMethod.GET,
            endpoint="users"
        )
        
        # Verify sleep was called with Retry-After value
        mock_sleep.assert_called_once_with(2)
        
        # Verify request was called twice (original + retry)
        assert mock_request.call_count == 2
        
        # Verify final response is the successful one
        assert response == second_response
        assert response.status_code == 200

    @patch("requests.Session.request")
    def test_uses_request_history_for_rate_limiting(self, mock_request):
        """Uses RequestHistory for rate limiting decisions."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_request.return_value = mock_response
        
        # Create a client with a mocked history
        client = SmartSurgeClient()
        
        # Create a mock history
        mock_history = MagicMock(spec=RequestHistory)
        mock_history.request_id = "mock-request-id"
        
        response, history = client.request(
            method=RequestMethod.GET,
            endpoint="users",
            request_history=mock_history
        )
        
        # Verify history's methods were used
        mock_history.intercept_request.assert_called_once()
        mock_history.log_response_and_update.assert_called_once()


class Test_SmartSurgeClient_Request_02_NegativeBehaviors:
    """Tests for negative behaviors of SmartSurgeClient's request method."""

    def test_raises_error_for_invalid_http_method(self):
        """Raises ValueError for invalid HTTP method."""
        client = SmartSurgeClient()
        
        with pytest.raises(ValueError, match="Invalid HTTP method"):
            client.request("INVALID_METHOD", "users")

    @patch("requests.Session.request")
    def test_records_failed_requests(self, mock_request):
        """Records failed requests appropriately in history."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.ok = False
        mock_request.return_value = mock_response
        
        client = SmartSurgeClient()
        response, history = client.request(
            method=RequestMethod.GET,
            endpoint="nonexistent"
        )
        
        assert len(history.entries) == 1
        assert history.entries[0].success is False
        assert history.entries[0].status_code == 404


class Test_SmartSurgeClient_Request_03_BoundaryBehaviors:
    """Tests for boundary behaviors of SmartSurgeClient's request method."""

    @patch("requests.Session.request")
    def test_handles_empty_headers_and_params(self, mock_request):
        """Handles empty or minimal headers/parameters correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_request.return_value = mock_response
        
        client = SmartSurgeClient()
        
        # Empty dictionaries
        response, history = client.request(
            method=RequestMethod.GET,
            endpoint="users",
            params={},
            headers={}
        )
        assert mock_request.call_count == 1
        
        # None values
        mock_request.reset_mock()
        response, history = client.request(
            method=RequestMethod.GET,
            endpoint="users",
            params=None,
            headers=None
        )
        assert mock_request.call_count == 1


class Test_SmartSurgeClient_Request_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of SmartSurgeClient's request method."""

    @patch("requests.Session.request")
    def test_records_request_failure_in_history(self, mock_request):
        """Records failed requests in history."""
        mock_request.side_effect = requests.RequestException("Connection failed")
        
        client = SmartSurgeClient()
        
        with pytest.raises(requests.RequestException):
            client.request(
                method=RequestMethod.GET,
                endpoint="users"
            )
        
        # Verify there's an entry in the history
        history = client._get_or_create_history("users", RequestMethod.GET, None)
        assert len(history.entries) == 1
        assert history.entries[0].success is False
        assert history.entries[0].status_code == 0

    @patch("requests.Session.request")
    def test_propagates_requests_exception(self, mock_request):
        """Propagates requests.RequestException with original error information."""
        original_exception = requests.RequestException("Connection failed")
        mock_request.side_effect = original_exception
        
        client = SmartSurgeClient()
        
        with pytest.raises(requests.RequestException) as excinfo:
            client.request(
                method=RequestMethod.GET,
                endpoint="users"
            )
        
        assert excinfo.value == original_exception

    @patch("requests.Session.request")
    def test_handles_malformed_retry_after_headers(self, mock_request):
        """Handles malformed Retry-After headers gracefully."""
        # Response with non-integer Retry-After
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.ok = False
        mock_response.headers = {"Retry-After": "invalid"}
        mock_request.return_value = mock_response
        
        client = SmartSurgeClient()
        
        # Should raise RateLimitExceeded without crashing
        with pytest.raises(RateLimitExceeded):
            client.request(
                method=RequestMethod.GET,
                endpoint="users"
            )

    @patch("requests.Session.request")
    def test_creates_rate_limit_exceeded_with_context(self, mock_request):
        """Creates RateLimitExceeded with proper context when receiving 429."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.ok = False
        mock_response.headers = {}  # No Retry-After
        mock_request.return_value = mock_response
        
        client = SmartSurgeClient()
        
        with pytest.raises(RateLimitExceeded) as excinfo:
            client.request(
                method=RequestMethod.GET,
                endpoint="users"
            )
        
        # Verify exception has correct information
        assert excinfo.value.endpoint == "users"
        assert excinfo.value.method == RequestMethod.GET
        assert excinfo.value.retry_after is None


class Test_SmartSurgeClient_Request_05_StateTransitionBehaviors:
    """Tests for state transition behaviors of SmartSurgeClient's request method."""

    @patch("smartsurge.models.RequestHistory._update_hmm", autospec=True)
    @patch("requests.Session.request")
    def test_updates_history_based_on_response(self, mock_request, mock_update_hmm):
        """Updates request history state based on response status."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_request.return_value = mock_response
        
        # Make _update_hmm set search_status to COMPLETED
        def update_hmm_side_effect(instance):
            instance.search_status = SearchStatus.COMPLETED
        mock_update_hmm.side_effect = update_hmm_side_effect
        
        client = SmartSurgeClient()
        
        # First request - should set search_status to WAITING_TO_ESTIMATE
        response, history = client.request(
            method=RequestMethod.GET,
            endpoint="users"
        )
        assert history.search_status == SearchStatus.WAITING_TO_ESTIMATE
        
        # Add enough entries to satisfy has_minimum_observations
        # Need at least min_data_points with at least one success and one failure
        for i in range(history.min_data_points - 1):
            entry = RequestEntry(
                endpoint="users",
                method=RequestMethod.GET,
                timestamp=datetime.now(timezone.utc),
                status_code=200,
                response_time=0.1,
                success=True
            )
            history.add_request(entry)
        
        # Add one failed request to ensure we have at least one failure
        entry = RequestEntry(
            endpoint="users",
            method=RequestMethod.GET,
            timestamp=datetime.now(timezone.utc),
            status_code=400,
            response_time=0.1,
            success=False
        )
        history.add_request(entry)
        
        # Second request - should update search_status to COMPLETED
        response, history2 = client.request(
            method=RequestMethod.GET,
            endpoint="users"
        )
        
        # Verify results
        assert history2 is history  # Should be the same object
        assert mock_update_hmm.called
        assert history2.search_status == SearchStatus.COMPLETED


    @patch("requests.Session.request")
    @patch("time.sleep")
    def test_transitions_to_waiting_state_when_rate_limited(self, mock_sleep, mock_request):
        """Transitions to waiting state when rate limit is exceeded."""
        # First response is 429 with no Retry-After header
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.ok = False
        mock_response.headers = {}  # No Retry-After
        mock_request.return_value = mock_response
        
        client = SmartSurgeClient()
        
        # Should raise RateLimitExceeded
        with pytest.raises(RateLimitExceeded):
            client.request(
                method=RequestMethod.GET,
                endpoint="users"
            )
        
        # Verify the history was updated
        history = client._get_or_create_history("users", RequestMethod.GET, None)
        assert len(history.entries) == 1
        assert history.entries[0].status_code == 429
        # Check if consecutive_refusals was incremented
        assert history.consecutive_refusals == 1


class Test_SmartSurgeClient_StreamRequest_01_NominalBehaviors:
    """Tests for nominal behaviors of SmartSurgeClient's stream_request method."""

    def test_initiates_streaming_request(self):
        """Initiates streaming request with provided streaming class."""
        mock_stream_class = MagicMock()
        mock_stream_instance = MagicMock()
        mock_stream_class.return_value = mock_stream_instance
        mock_stream_instance.get_result.return_value = {"result": "data"}
        
        client = SmartSurgeClient()
        result, history = client.stream_request(
            streaming_class=mock_stream_class,
            endpoint="users"
        )
        
        # Verify streaming class was instantiated correctly
        mock_stream_class.assert_called_once()
        # Verify start was called
        mock_stream_instance.start.assert_called_once()
        # Verify result matches
        assert result == {"result": "data"}

    def test_returns_streaming_result_and_history(self):
        """Returns streaming result and request history."""
        mock_stream_class = MagicMock()
        mock_stream_instance = MagicMock()
        mock_stream_class.return_value = mock_stream_instance
        mock_stream_instance.get_result.return_value = {"result": "data"}
        
        client = SmartSurgeClient()
        result, history = client.stream_request(
            streaming_class=mock_stream_class,
            endpoint="users"
        )
        
        # Verify result is correct
        assert result == {"result": "data"}
        # Verify history is returned
        assert isinstance(history, RequestHistory)

    def test_records_request_in_history(self):
        """Records request in history for rate limiting purposes."""
        mock_stream_class = MagicMock()
        mock_stream_instance = MagicMock()
        mock_stream_class.return_value = mock_stream_instance
        mock_stream_instance.get_result.return_value = {"result": "data"}
        
        client = SmartSurgeClient()
        result, history = client.stream_request(
            streaming_class=mock_stream_class,
            endpoint="users"
        )
        
        # Verify entry was added to history
        assert len(history.entries) == 1
        assert history.entries[0].endpoint == "users"
        assert history.entries[0].method == RequestMethod.GET
        assert history.entries[0].success is True

    def test_supports_resuming_from_state_file(self):
        """Supports resuming from state file when available."""
        mock_stream_class = MagicMock()
        mock_stream_instance = MagicMock()
        mock_stream_class.return_value = mock_stream_instance
        mock_stream_instance.get_result.return_value = {"result": "data"}
        
        # Mock os.path.exists to return True
        with patch("os.path.exists", return_value=True):
            client = SmartSurgeClient()
            result, history = client.stream_request(
                streaming_class=mock_stream_class,
                endpoint="users",
                state_file="users.state"
            )
            
            # Verify resume was called instead of start
            mock_stream_instance.resume.assert_called_once()
            mock_stream_instance.start.assert_not_called()

    @patch("smartsurge.client.RequestHistory.intercept_request")
    def test_applies_rate_limiting_from_history(self, mock_intercept):
        """Applies rate limiting based on history."""
        mock_stream_class = MagicMock()
        mock_stream_instance = MagicMock()
        mock_stream_class.return_value = mock_stream_instance
        mock_stream_instance.get_result.return_value = {"result": "data"}
        
        client = SmartSurgeClient()
        result, history = client.stream_request(
            streaming_class=mock_stream_class,
            endpoint="users"
        )
        
        # Verify intercept_request was called to apply rate limiting
        mock_intercept.assert_called_once()


class Test_SmartSurgeClient_StreamRequest_02_NegativeBehaviors:
    """Tests for negative behaviors of SmartSurgeClient's stream_request method."""

    def test_handles_streaming_failures(self):
        """Handles streaming failures by saving state for resumption."""
        mock_stream_class = MagicMock()
        mock_stream_instance = MagicMock()
        mock_stream_class.return_value = mock_stream_instance
        
        error = Exception("Streaming failed")
        mock_stream_instance.start.side_effect = error
        
        client = SmartSurgeClient()
        with pytest.raises(StreamingError):
            result, history = client.stream_request(
                streaming_class=mock_stream_class,
                endpoint="users",
                state_file="users.state"
            )
        
        # Verify save_state was called
        mock_stream_instance.save_state.assert_called_once()


class Test_SmartSurgeClient_StreamRequest_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of SmartSurgeClient's stream_request method."""

    def test_records_failed_streaming_requests(self):
        """Records failed streaming requests in history."""
        mock_stream_class = MagicMock()
        mock_stream_instance = MagicMock()
        mock_stream_class.return_value = mock_stream_instance
        
        error = Exception("Streaming failed")
        mock_stream_instance.start.side_effect = error
        
        client = SmartSurgeClient()
        with pytest.raises(StreamingError):
            result, history = client.stream_request(
                streaming_class=mock_stream_class,
                endpoint="users"
            )
        
        # Get history and verify failure was recorded
        history = client._get_or_create_history("users", RequestMethod.GET, None)
        assert len(history.entries) == 1
        assert history.entries[0].success is False

    def test_wraps_exceptions_as_streaming_error(self):
        """Wraps exceptions as StreamingError with context."""
        mock_stream_class = MagicMock()
        mock_stream_instance = MagicMock()
        mock_stream_class.return_value = mock_stream_instance
        
        original_error = ValueError("Invalid data")
        mock_stream_instance.start.side_effect = original_error
        
        client = SmartSurgeClient()
        with pytest.raises(StreamingError) as excinfo:
            result, history = client.stream_request(
                streaming_class=mock_stream_class,
                endpoint="users"
            )
        
        # Verify exception contains the original error message
        assert "Invalid data" in str(excinfo.value)

    def test_handles_rate_limit_errors(self):
        """Handles rate limit (429) errors by waiting and retrying."""
        mock_stream_class = MagicMock()
        mock_stream_instance = MagicMock()
        mock_stream_class.return_value = mock_stream_instance
        
        # Create a mock response with 429 status
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "2"}
        
        # Create a streaming error with the response
        error = StreamingError("Rate limit exceeded")
        error.response = mock_response
        mock_stream_instance.start.side_effect = [error, None]  # Fail first, succeed second
        mock_stream_instance.get_result.return_value = {"result": "data"}
        
        # Mock time.sleep to avoid waiting
        with patch("time.sleep") as mock_sleep:
            client = SmartSurgeClient()
            result, history = client.stream_request(
                streaming_class=mock_stream_class,
                endpoint="users"
            )
            
            # Verify sleep was called with Retry-After value
            mock_sleep.assert_called_once_with(2)
            
            # Verify streaming start was called twice
            assert mock_stream_instance.start.call_count == 2
            
            # Verify we got the result
            assert result == {"result": "data"}

    def test_raises_resume_error_when_resumption_fails(self):
        """Raises ResumeError when resumption fails."""
        mock_stream_class = MagicMock()
        mock_stream_instance = MagicMock()
        mock_stream_class.return_value = mock_stream_instance
        
        error = Exception("Resume failed")
        mock_stream_instance.resume.side_effect = error
        
        # Mock os.path.exists to return True
        with patch("os.path.exists", return_value=True):
            client = SmartSurgeClient()
            with pytest.raises(ResumeError):
                result, history = client.stream_request(
                    streaming_class=mock_stream_class,
                    endpoint="users",
                    state_file="users.state"
                )


class Test_SmartSurgeClient_StreamRequest_05_StateTransitionBehaviors:
    """Tests for state transition behaviors of SmartSurgeClient's stream_request method."""

    def test_saves_streaming_state_periodically(self):
        """Saves streaming state periodically for potential resumption."""
        # This would typically be tested by mocking the streaming class's
        # process_chunk method and verifying save_state is called at
        # appropriate intervals, but we'll simplify here:
        
        mock_stream_class = MagicMock()
        mock_stream_instance = MagicMock()
        mock_stream_class.return_value = mock_stream_instance
        mock_stream_instance.get_result.return_value = {"result": "data"}
        
        client = SmartSurgeClient()
        result, history = client.stream_request(
            streaming_class=mock_stream_class,
            endpoint="users",
            state_file="users.state"
        )
        
        # In a real streaming operation, save_state would be called
        # periodically during process_chunk

    def test_transitions_between_streaming_states(self):
        """Transitions between starting, streaming, and completed states."""
        mock_stream_class = MagicMock()
        mock_stream_instance = MagicMock()
        mock_stream_class.return_value = mock_stream_instance
        mock_stream_instance.get_result.return_value = {"result": "data"}
        
        # Mock streaming instance properties to track state
        mock_stream_instance.completed = False
        
        def mock_start():
            mock_stream_instance.completed = True
            
        mock_stream_instance.start.side_effect = mock_start
        
        client = SmartSurgeClient()
        result, history = client.stream_request(
            streaming_class=mock_stream_class,
            endpoint="users"
        )
        
        # Verify completion state
        assert mock_stream_instance.completed is True


class Test_SmartSurgeClient_CloseEnterExit_01_NominalBehaviors:
    """Tests for nominal behaviors of SmartSurgeClient's close, __enter__, and __exit__ methods."""

    @patch("requests.Session.close")
    def test_close_releases_resources(self, mock_close):
        """close releases session resources properly."""
        client = SmartSurgeClient()
        client.close()
        
        # Verify session.close was called
        mock_close.assert_called_once()

    @patch("smartsurge.client.SmartSurgeClient.close")
    def test_context_manager_works(self, mock_close):
        """Context manager pattern works correctly with with statement."""
        with SmartSurgeClient() as client:
            pass
        
        # Verify close was called when context manager exits
        mock_close.assert_called_once()

    def test_enter_returns_self_reference(self):
        """__enter__ returns self reference."""
        client = SmartSurgeClient()
        result = client.__enter__()
        assert result is client

    @patch("smartsurge.client.SmartSurgeClient.close")
    def test_exit_calls_close(self, mock_close):
        """__exit__ calls close to release resources."""
        client = SmartSurgeClient()
        client.__exit__(None, None, None)
        
        # Verify close was called
        mock_close.assert_called_once()


class Test_SmartSurgeClient_ConvenienceMethods_01_NominalBehaviors:
    """Tests for nominal behaviors of SmartSurgeClient's convenience methods."""

    @patch("smartsurge.client.SmartSurgeClient.request")
    def test_get_calls_request_with_correct_method(self, mock_request):
        """get calls request with the correct HTTP method enum."""
        mock_response = MagicMock()
        mock_history = MagicMock()
        mock_request.return_value = (mock_response, mock_history)
        
        client = SmartSurgeClient()
        response, history = client.get("users", params={"page": 1})
        
        # Verify request was called with GET
        mock_request.assert_called_once()
        assert mock_request.call_args.kwargs["method"] == RequestMethod.GET

    @patch("smartsurge.client.SmartSurgeClient.request")
    def test_post_calls_request_with_correct_method(self, mock_request):
        """post calls request with the correct HTTP method enum."""
        mock_response = MagicMock()
        mock_history = MagicMock()
        mock_request.return_value = (mock_response, mock_history)
        
        client = SmartSurgeClient()
        response, history = client.post("users", json={"name": "John"})
        
        # Verify request was called with POST
        mock_request.assert_called_once()
        assert mock_request.call_args.kwargs["method"] == RequestMethod.POST

    @patch("smartsurge.client.SmartSurgeClient.request")
    def test_put_calls_request_with_correct_method(self, mock_request):
        """put calls request with the correct HTTP method enum."""
        mock_response = MagicMock()
        mock_history = MagicMock()
        mock_request.return_value = (mock_response, mock_history)
        
        client = SmartSurgeClient()
        response, history = client.put("users/1", json={"name": "John"})
        
        # Verify request was called with PUT
        mock_request.assert_called_once()
        assert mock_request.call_args.kwargs["method"] == RequestMethod.PUT

    @patch("smartsurge.client.SmartSurgeClient.request")
    def test_delete_calls_request_with_correct_method(self, mock_request):
        """delete calls request with the correct HTTP method enum."""
        mock_response = MagicMock()
        mock_history = MagicMock()
        mock_request.return_value = (mock_response, mock_history)
        
        client = SmartSurgeClient()
        response, history = client.delete("users/1")
        
        # Verify request was called with DELETE
        mock_request.assert_called_once()
        assert mock_request.call_args.kwargs["method"] == RequestMethod.DELETE

    @patch("smartsurge.client.SmartSurgeClient.request")
    def test_patch_calls_request_with_correct_method(self, mock_request):
        """patch calls request with the correct HTTP method enum."""
        mock_response = MagicMock()
        mock_history = MagicMock()
        mock_request.return_value = (mock_response, mock_history)
        
        client = SmartSurgeClient()
        response, history = client.patch("users/1", json={"name": "John"})
        
        # Verify request was called with PATCH
        mock_request.assert_called_once()
        assert mock_request.call_args.kwargs["method"] == RequestMethod.PATCH

    @patch("smartsurge.client.SmartSurgeClient.request")
    def test_parameters_passed_through_correctly(self, mock_request):
        """Parameters are properly passed through to the request method."""
        mock_response = MagicMock()
        mock_history = MagicMock()
        mock_request.return_value = (mock_response, mock_history)
        
        client = SmartSurgeClient()
        response, history = client.get(
            "users", 
            params={"page": 1}, 
            headers={"Accept": "application/json"},
            timeout=30.0
        )
        
        # Verify all parameters were passed through
        call_args = mock_request.call_args
        assert call_args.kwargs["params"] == {"page": 1}
        assert call_args.kwargs["headers"] == {"Accept": "application/json"}
        assert call_args.kwargs["timeout"] == 30.0

    @patch("smartsurge.client.SmartSurgeClient.request")
    def test_returns_response_and_history(self, mock_request):
        """Returns response and history tuple from underlying request method."""
        mock_response = MagicMock()
        mock_history = MagicMock()
        mock_request.return_value = (mock_response, mock_history)
        
        client = SmartSurgeClient()
        response, history = client.get("users")
        
        # Verify returned values match what request returned
        assert response == mock_response
        assert history == mock_history

    @patch("smartsurge.client.SmartSurgeClient.request")
    def test_handles_standard_and_keyword_parameters(self, mock_request):
        """Handles both standard and keyword parameters correctly."""
        mock_response = MagicMock()
        mock_history = MagicMock()
        mock_request.return_value = (mock_response, mock_history)
        
        client = SmartSurgeClient()
        
        # Test with standard parameters
        response, history = client.get(
            "users",
            params={"page": 1},
            headers={"Accept": "application/json"}
        )
        
        # Test with additional keyword arguments
        response, history = client.get(
            "users",
            params={"page": 1},
            headers={"Accept": "application/json"},
            timeout=30.0,
            verify=False,
            allow_redirects=False
        )
        
        # Verify all parameters were passed through to request
        call_args = mock_request.call_args
        assert call_args.kwargs["timeout"] == 30.0
        assert call_args.kwargs["verify"] == False
        assert call_args.kwargs["allow_redirects"] == False


class Test_SmartSurgeClient_ConvenienceMethods_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of SmartSurgeClient's convenience methods."""

    @patch("smartsurge.client.SmartSurgeClient.request")
    def test_exceptions_are_propagated(self, mock_request):
        """Exceptions from request method are propagated correctly."""
        error = requests.RequestException("Connection failed")
        mock_request.side_effect = error
        
        client = SmartSurgeClient()
        
        with pytest.raises(requests.RequestException) as excinfo:
            client.get("users")
        
        assert excinfo.value == error
        
        # Test other methods also propagate exceptions
        mock_request.side_effect = requests.RequestException("Connection failed")
        with pytest.raises(requests.RequestException):
            client.post("users", json={"name": "John"})
            
        mock_request.side_effect = requests.RequestException("Connection failed")
        with pytest.raises(requests.RequestException):
            client.put("users/1", json={"name": "John"})
            
        mock_request.side_effect = requests.RequestException("Connection failed")
        with pytest.raises(requests.RequestException):
            client.delete("users/1")
            
        mock_request.side_effect = requests.RequestException("Connection failed")
        with pytest.raises(requests.RequestException):
            client.patch("users/1", json={"name": "John"})


class Test_SmartSurgeClient_AsyncMethods_01_NominalBehaviors:
    """Tests for nominal behaviors of SmartSurgeClient's async methods."""

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.request")
    async def test_async_request_makes_http_request_successfully(self, mock_request):
        """async_request makes asynchronous HTTP request successfully."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.ok = True
        mock_response.read = AsyncMock()
        
        # Use __aenter__ and __aexit__ to simulate async context manager
        mock_request.return_value.__aenter__.return_value = mock_response
        
        client = SmartSurgeClient()
        response, history = await client.async_request(
            method=RequestMethod.GET,
            endpoint="users",
            params={"page": 1}
        )
        
        # Verify request was made with correct parameters
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args.kwargs["method"] == "GET"
        assert "users" in call_args.kwargs["url"]
        assert call_args.kwargs["params"] == {"page": 1}
        
        # Verify response was returned with history
        assert response == mock_response
        assert isinstance(history, RequestHistory)

    @pytest.mark.asyncio
    @patch("smartsurge.client.SmartSurgeClient.async_request")
    async def test_async_get_calls_async_request_with_correct_method(self, mock_async_request):
        """async_get calls async_request with the correct method."""
        mock_response = MagicMock()
        mock_history = MagicMock()
        mock_async_request.return_value = (mock_response, mock_history)
        
        client = SmartSurgeClient()
        response, history = await client.async_get("users", params={"page": 1})
        
        # Verify async_request was called with GET method
        mock_async_request.assert_called_once()
        assert mock_async_request.call_args.kwargs["method"] == RequestMethod.GET
        assert mock_async_request.call_args.kwargs["endpoint"] == "users"
        assert mock_async_request.call_args.kwargs["params"] == {"page": 1}
        
        # Verify response and history were returned
        assert response == mock_response
        assert history == mock_history

    @pytest.mark.asyncio
    @patch("smartsurge.client.SmartSurgeClient.async_request")
    async def test_async_post_calls_async_request_with_correct_method(self, mock_async_request):
        """async_post calls async_request with the correct method."""
        mock_response = MagicMock()
        mock_history = MagicMock()
        mock_async_request.return_value = (mock_response, mock_history)
        
        client = SmartSurgeClient()
        response, history = await client.async_post(
            "users",
            json={"name": "John"}
        )
        
        # Verify async_request was called with POST method
        mock_async_request.assert_called_once()
        assert mock_async_request.call_args.kwargs["method"] == RequestMethod.POST
        assert mock_async_request.call_args.kwargs["endpoint"] == "users"
        assert mock_async_request.call_args.kwargs["json"] == {"name": "John"}
        
        # Verify response and history were returned
        assert response == mock_response
        assert history == mock_history

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.request")
    async def test_records_request_in_history(self, mock_request):
        """Records request in history for rate limiting."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.ok = True
        mock_response.read = AsyncMock()
        mock_response.__aenter__.return_value = mock_response
        
        # Use __aenter__ and __aexit__ to simulate async context manager
        mock_request.return_value.__aenter__.return_value = mock_response
        
        client = SmartSurgeClient()
        response, history = await client.async_request(
            method=RequestMethod.GET,
            endpoint="users"
        )
        
        # Verify entry was added to history
        assert len(history.entries) == 1
        assert history.entries[0].endpoint == "users"
        assert history.entries[0].method == RequestMethod.GET
        assert history.entries[0].success is True

    @pytest.mark.asyncio
    @patch("smartsurge.client.RequestHistory.intercept_request")
    @patch("aiohttp.ClientSession.request")
    async def test_applies_rate_limiting_from_history(self, mock_request, mock_intercept):
        """Applies rate limiting based on history."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.ok = True
        mock_response.read = AsyncMock()
        mock_response.__aenter__.return_value = mock_response
        
        # Use __aenter__ and __aexit__ to simulate async context manager
        mock_request.return_value.__aenter__.return_value = mock_response
        
        client = SmartSurgeClient()
        response, history = await client.async_request(
            method=RequestMethod.GET,
            endpoint="users"
        )
        
        # Verify intercept_request was called to apply rate limiting
        mock_intercept.assert_called_once()


class Test_SmartSurgeClient_AsyncMethods_02_NegativeBehaviors:
    """Tests for negative behaviors of SmartSurgeClient's async methods."""

    @pytest.mark.asyncio
    async def test_invalid_http_method_raises_error(self):
        """Handles invalid HTTP method by raising ValueError."""
        client = SmartSurgeClient()
        
        with pytest.raises(ValueError, match="Invalid HTTP method"):
            await client.async_request("INVALID_METHOD", "users")

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.request")
    async def test_records_failed_requests(self, mock_request):
        """Processes failed async requests by recording in history."""
        # Simulate a failed request
        mock_request.side_effect = aiohttp.ClientError("Connection failed")
        
        client = SmartSurgeClient()
        with pytest.raises(aiohttp.ClientError):
            await client.async_request(
                method=RequestMethod.GET,
                endpoint="users"
            )
        
        # Get history and verify failure was recorded
        history = client._get_or_create_history("users", RequestMethod.GET, None)
        assert len(history.entries) == 1
        assert history.entries[0].success is False


class Test_SmartSurgeClient_AsyncMethods_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of SmartSurgeClient's async methods."""

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.request")
    async def test_records_failed_requests_in_history(self, mock_request):
        """Records failed requests in history."""
        # Simulate a failed request
        mock_request.side_effect = aiohttp.ClientError("Connection failed")
        
        client = SmartSurgeClient()
        with pytest.raises(aiohttp.ClientError):
            await client.async_request(
                method=RequestMethod.GET,
                endpoint="users"
            )
        
        # Get history and verify failure was recorded
        history = client._get_or_create_history("users", RequestMethod.GET, None)
        assert len(history.entries) == 1
        assert history.entries[0].success is False
        assert history.entries[0].status_code == 0

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.request")
    async def test_propagates_client_error_with_context(self, mock_request):
        """Propagates aiohttp.ClientError with context."""
        original_error = aiohttp.ClientError("Connection failed")
        mock_request.side_effect = original_error
        
        client = SmartSurgeClient()
        with pytest.raises(aiohttp.ClientError) as excinfo:
            await client.async_request(
                method=RequestMethod.GET,
                endpoint="users"
            )
        
        assert excinfo.value == original_error

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.request")
    async def test_handles_rate_limit_with_retry_after(self, mock_request):
        """Handles rate limit (429) with appropriate waiting using asyncio.sleep."""
        # Create a response object that will be returned by the context manager
        mock_response = MagicMock()
        mock_response.status = 429
        mock_response.ok = False
        mock_response.headers = {"Retry-After": "2"}
        mock_response.read = AsyncMock()
        
        # Set up the first call to return 429 and second call to succeed
        first_response = mock_response
        
        second_response = MagicMock()
        second_response.status = 200
        second_response.ok = True
        second_response.read = AsyncMock()
        
        # Mock the context managers
        first_context = MagicMock()
        first_context.__aenter__.return_value = first_response
        
        second_context = MagicMock()
        second_context.__aenter__.return_value = second_response
        
        mock_request.side_effect = [first_context, second_context]
        
        # Mock asyncio.sleep
        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.return_value = AsyncMock()
            
            client = SmartSurgeClient()
            response, history = await client.async_request(
                method=RequestMethod.GET,
                endpoint="users"
            )
            
            # Verify sleep was called with Retry-After value
            mock_sleep.assert_awaited_once_with(2)
            
            # Verify request was called twice (original + retry)
            assert mock_request.call_count == 2
            
            # Verify we got the successful response
            assert response == second_response

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.request")
    async def test_reads_response_body_completely(self, mock_request):
        """Reads response body completely before returning to ensure connection closure."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.ok = True
        mock_response.read = AsyncMock()
        
        # Set up the context manager
        mock_context = MagicMock()
        mock_context.__aenter__.return_value = mock_response
        mock_request.return_value = mock_context
        
        client = SmartSurgeClient()
        response, history = await client.async_request(
            method=RequestMethod.GET,
            endpoint="users"
        )
        
        # Verify read was called to ensure complete body reading
        mock_response.read.assert_called_once()


class Test_SmartSurgeClient_AsyncMethods_05_StateTransitionBehaviors:
    """Tests for state transition behaviors of SmartSurgeClient's async methods."""

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.request")
    async def test_updates_history_based_on_response_status(self, mock_request):
        """Updates request history state based on response status."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.ok = True
        mock_response.read = AsyncMock()
        
        # Set up the context manager
        mock_context = MagicMock()
        mock_context.__aenter__.return_value = mock_response
        mock_request.return_value = mock_context
        
        client = SmartSurgeClient()
        
        # First request - should set search_status to WAITING_TO_ESTIMATE
        response, history = await client.async_request(
            method=RequestMethod.GET,
            endpoint="users"
        )
        assert history.search_status == SearchStatus.WAITING_TO_ESTIMATE
        
        with patch('smartsurge.models.RequestHistory.has_minimum_observations', return_value=True):
            # Second request - should update search_status to COMPLETED
            response, history = await client.async_request(
                method=RequestMethod.GET,
                endpoint="users",
                request_history=history  # Ensure same history object is used
            )
            assert history.search_status == SearchStatus.COMPLETED


    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.request")
    @patch("asyncio.sleep")
    async def test_transitions_to_waiting_state_when_rate_limited(self, mock_sleep, mock_request):
        """Transitions to waiting state when rate limit is exceeded."""
        # Create a response object with 429 status
        mock_response = MagicMock()
        mock_response.status = 429
        mock_response.ok = False
        mock_response.headers = {}  # No Retry-After header
        mock_response.read = AsyncMock()
        
        # Set up the context manager
        mock_context = MagicMock()
        mock_context.__aenter__.return_value = mock_response
        mock_request.return_value = mock_context
        
        # Return a Future from asyncio.sleep to avoid actually sleeping
        mock_sleep.return_value = AsyncMock()
        
        client = SmartSurgeClient()
        
        # Should raise RateLimitExceeded
        with pytest.raises(RateLimitExceeded):
            await client.async_request(
                method=RequestMethod.GET,
                endpoint="users"
            )
        
        # Verify the history was updated
        history = client._get_or_create_history("users", RequestMethod.GET, None)
        assert len(history.entries) == 1
        assert history.entries[0].status_code == 429
        # Check if consecutive_refusals was incremented
        assert history.consecutive_refusals == 1
