import pytest
import logging
import re
import os
import sys
from unittest.mock import patch, MagicMock, call

from smartsurge.logging_ import SensitiveInfoFilter, PerformanceFilter, configure_logging, get_logger

# Constants for testing
DEFAULT_SENSITIVE_PATTERNS = [
    r'(password=)([^&\s]+)',
    r'(token=)([^&\s]+)',
    r'(key=)([^&\s]+)',
    r'(secret=)([^&\s]+)',
    r'(auth=)([^&\s]+)',
    r'(Authorization:)\s*([^\s]+)',
    r'(apikey=)([^&\s]+)',
]


def _create_log_record(level: int, msg_content: str) -> logging.LogRecord:
    """Creates a mock LogRecord."""
    return logging.LogRecord(
        name='test_logger', 
        level=level,
        pathname='test_pathname', 
        lineno=123, 
        msg=msg_content,
        args=(), 
        exc_info=None, 
        func='test_func_name'
    )


class Test_SensitiveInfoFilter_Init_01_NominalBehaviors:
    """Tests for nominal behaviors of SensitiveInfoFilter.__init__ method."""

    def test_initialize_with_default_patterns(self):
        """Test that filter initializes with default patterns when none are provided."""
        from smartsurge.logging_ import SensitiveInfoFilter
        
        filter_obj = SensitiveInfoFilter()
        
        # Verify default patterns are used
        assert filter_obj.patterns == DEFAULT_SENSITIVE_PATTERNS
        assert filter_obj.replacement == '******'
        assert len(filter_obj.compiled_patterns) == len(DEFAULT_SENSITIVE_PATTERNS)
        
        # Verify patterns are compiled correctly
        for i, pattern in enumerate(DEFAULT_SENSITIVE_PATTERNS):
            assert isinstance(filter_obj.compiled_patterns[i], re.Pattern)
            assert filter_obj.compiled_patterns[i].pattern == pattern

    def test_initialize_with_custom_patterns(self):
        """Test initializing with custom patterns list."""
        from smartsurge.logging_ import SensitiveInfoFilter
        
        custom_patterns = [r'(custom=)([^&\s]+)', r'(secret=)([^&\s]+)']
        filter_obj = SensitiveInfoFilter(patterns=custom_patterns)
        
        # Verify custom patterns are used
        assert filter_obj.patterns == custom_patterns
        assert filter_obj.replacement == '******'
        assert len(filter_obj.compiled_patterns) == len(custom_patterns)
        
        # Verify patterns are compiled correctly
        for i, pattern in enumerate(custom_patterns):
            assert isinstance(filter_obj.compiled_patterns[i], re.Pattern)
            assert filter_obj.compiled_patterns[i].pattern == pattern

    def test_initialize_with_custom_replacement(self):
        """Test initializing with custom replacement string."""
        from smartsurge.logging_ import SensitiveInfoFilter
        
        custom_replacement = '[REDACTED]'
        filter_obj = SensitiveInfoFilter(replacement=custom_replacement)
        
        # Verify custom replacement is used
        assert filter_obj.patterns == DEFAULT_SENSITIVE_PATTERNS
        assert filter_obj.replacement == custom_replacement

    def test_successful_pattern_compilation(self):
        """Test that all regex patterns are successfully compiled."""
        from smartsurge.logging_ import SensitiveInfoFilter
        
        filter_obj = SensitiveInfoFilter()
        
        # Verify all patterns are compiled
        assert len(filter_obj.compiled_patterns) == len(filter_obj.patterns)
        
        # Verify each pattern is a compiled regex pattern
        for pattern in filter_obj.compiled_patterns:
            assert isinstance(pattern, re.Pattern)


class Test_SensitiveInfoFilter_Init_02_NegativeBehaviors:
    """Tests for negative behaviors of SensitiveInfoFilter.__init__ method."""

    def test_handle_invalid_regex_patterns(self):
        """Test that invalid regex patterns raise appropriate exceptions."""
        from smartsurge.logging_ import SensitiveInfoFilter
        
        # Create a pattern that will cause a re.error when compiled
        invalid_patterns = [r'[invalid(regex', r'(valid)(.+)']
        
        # The error should come from within re.compile during initialization
        with pytest.raises(re.error):
            SensitiveInfoFilter(patterns=invalid_patterns)


class Test_SensitiveInfoFilter_Init_03_BoundaryBehaviors:
    """Tests for boundary behaviors of SensitiveInfoFilter.__init__ method."""

    def test_initialize_with_empty_patterns_list(self):
        """Test initializing with an empty patterns list."""
        from smartsurge.logging_ import SensitiveInfoFilter, DEFAULT_SENSITIVE_PATTERNS
        
        filter_obj = SensitiveInfoFilter(patterns=[])
        
        # Verify empty patterns list results in empty compiled patterns
        assert filter_obj.patterns == DEFAULT_SENSITIVE_PATTERNS
        assert filter_obj.compiled_patterns == [re.compile(pattern) for pattern in DEFAULT_SENSITIVE_PATTERNS]

    def test_initialize_with_empty_replacement_string(self):
        """Test initializing with an empty replacement string."""
        from smartsurge.logging_ import SensitiveInfoFilter
        
        filter_obj = SensitiveInfoFilter(replacement='')
        
        # Verify empty replacement string is used
        assert filter_obj.replacement == ''
        
        # Test that pattern substitution still works with empty replacement
        record = logging.LogRecord(
            name='test', level=logging.INFO, 
            pathname='', lineno=0, msg='password=secret123', 
            args=(), exc_info=None
        )
        filter_obj.filter(record)
        assert record.msg == 'password='  # Password should be replaced with empty string


class Test_SensitiveInfoFilter_Init_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of SensitiveInfoFilter.__init__ method."""

    def test_handle_regex_compilation_errors(self):
        """Test handling of regex compilation errors for invalid patterns."""
        from smartsurge.logging_ import SensitiveInfoFilter
        
        # Mock re.compile to simulate an error for specific patterns
        original_compile = re.compile
        
        def mock_compile(pattern, *args, **kwargs):
            if pattern == r'(bad=)([^&\s]+)':
                raise re.error("Simulated regex compilation error")
            return original_compile(pattern, *args, **kwargs)
        
        with patch('re.compile', side_effect=mock_compile):
            # This should raise the re.error from our mock
            with pytest.raises(re.error, match="Simulated regex compilation error"):
                SensitiveInfoFilter(patterns=[
                    r'(good=)([^&\s]+)',
                    r'(bad=)([^&\s]+)',
                ])


class Test_SensitiveInfoFilter_Filter_01_NominalBehaviors:
    """Tests for nominal behaviors of SensitiveInfoFilter.filter method."""

    def test_redact_sensitive_information_in_messages(self):
        """Test that sensitive information is successfully redacted in string messages."""
        from smartsurge.logging_ import SensitiveInfoFilter
        
        filter_obj = SensitiveInfoFilter()
        
        # Create a log record with sensitive information
        record = logging.LogRecord(
            name='test', level=logging.INFO, 
            pathname='', lineno=0, msg='Request with password=secret123 and token=abc123', 
            args=(), exc_info=None
        )
        
        # Apply the filter
        result = filter_obj.filter(record)
        
        # Verify the result
        assert result is True  # Filter should always return True
        assert 'password=******' in record.msg
        assert 'token=******' in record.msg
        assert 'secret123' not in record.msg
        assert 'abc123' not in record.msg

    def test_redact_sensitive_information_in_args(self):
        """Test that sensitive information is redacted in string arguments."""
        from smartsurge.logging_ import SensitiveInfoFilter
        
        filter_obj = SensitiveInfoFilter()
        
        # Create a log record with sensitive information in args
        record = logging.LogRecord(
            name='test', level=logging.INFO, 
            pathname='', lineno=0, msg='Request info: %s, %s', 
            args=('password=secret123', 'token=abc123'), 
            exc_info=None
        )
        
        # Apply the filter
        result = filter_obj.filter(record)
        
        # Verify the result
        assert result is True  # Filter should always return True
        assert record.args[0] == 'password=******'
        assert record.args[1] == 'token=******'

    def test_return_true_to_allow_record(self):
        """Test that the filter always returns True to allow the record to pass through."""
        from smartsurge.logging_ import SensitiveInfoFilter
        
        filter_obj = SensitiveInfoFilter()
        
        # Create a log record with no sensitive information
        record = logging.LogRecord(
            name='test', level=logging.INFO, 
            pathname='', lineno=0, msg='Regular message with no sensitive data', 
            args=(), exc_info=None
        )
        
        # Apply the filter
        result = filter_obj.filter(record)
        
        # Verify the result
        assert result is True  # Filter should always return True
        assert record.msg == 'Regular message with no sensitive data'  # Message remains unchanged

    def test_process_multiple_patterns(self):
        """Test that multiple patterns matching in the same string are all processed."""
        from smartsurge.logging_ import SensitiveInfoFilter
        
        filter_obj = SensitiveInfoFilter()
        
        # Create a log record with multiple sensitive information types
        record = logging.LogRecord(
            name='test', level=logging.INFO, 
            pathname='', lineno=0, 
            msg='Request with password=secret123, token=abc123, key=key123, secret=s3cr3t', 
            args=(), exc_info=None
        )
        
        # Apply the filter
        filter_obj.filter(record)
        
        # Verify all patterns are redacted
        assert 'password=******' in record.msg
        assert 'token=******' in record.msg
        assert 'key=******' in record.msg
        assert 'secret=******' in record.msg
        assert 'secret123' not in record.msg
        assert 'abc123' not in record.msg
        assert 'key123' not in record.msg
        assert 's3cr3t' not in record.msg

    def test_use_custom_replacement(self):
        """Test that sensitive content is replaced with the configured replacement string."""
        from smartsurge.logging_ import SensitiveInfoFilter
        
        custom_replacement = '[REDACTED]'
        filter_obj = SensitiveInfoFilter(replacement=custom_replacement)
        
        # Create a log record with sensitive information
        record = logging.LogRecord(
            name='test', level=logging.INFO, 
            pathname='', lineno=0, msg='Request with password=secret123', 
            args=(), exc_info=None
        )
        
        # Apply the filter
        filter_obj.filter(record)
        
        # Verify the custom replacement is used
        assert f'password={custom_replacement}' in record.msg
        assert 'secret123' not in record.msg


class Test_SensitiveInfoFilter_Filter_02_NegativeBehaviors:
    """Tests for negative behaviors of SensitiveInfoFilter.filter method."""

    def test_handle_non_string_messages(self):
        """Test that non-string messages are handled gracefully."""
        from smartsurge.logging_ import SensitiveInfoFilter
        
        filter_obj = SensitiveInfoFilter()
        
        # Create log records with non-string messages
        record_int = logging.LogRecord(
            name='test', level=logging.INFO, 
            pathname='', lineno=0, msg=123, 
            args=(), exc_info=None
        )
        
        record_dict = logging.LogRecord(
            name='test', level=logging.INFO, 
            pathname='', lineno=0, msg={'key': 'value'}, 
            args=(), exc_info=None
        )
        
        # Apply the filter
        result_int = filter_obj.filter(record_int)
        result_dict = filter_obj.filter(record_dict)
        
        # Verify the results
        assert result_int is True  # Filter should always return True
        assert result_dict is True  # Filter should always return True
        assert record_int.msg == 123  # Message should remain unchanged
        assert record_dict.msg == {'key': 'value'}  # Message should remain unchanged

    def test_handle_records_with_no_args(self):
        """Test that records with no args attribute are handled gracefully."""
        from smartsurge.logging_ import SensitiveInfoFilter
        
        filter_obj = SensitiveInfoFilter()
        
        # Create a log record but remove the args attribute
        record = logging.LogRecord(
            name='test', level=logging.INFO, 
            pathname='', lineno=0, msg='Regular message', 
            args=(), exc_info=None
        )
        delattr(record, 'args')
        
        # Apply the filter
        result = filter_obj.filter(record)
        
        # Verify the result
        assert result is True  # Filter should always return True
        assert record.msg == 'Regular message'  # Message should remain unchanged

    def test_skip_non_string_arguments(self):
        """Test that non-string arguments are skipped."""
        from smartsurge.logging_ import SensitiveInfoFilter
        
        filter_obj = SensitiveInfoFilter()
        
        # Create a log record with mixed type arguments
        record = logging.LogRecord(
            name='test', level=logging.INFO, 
            pathname='', lineno=0, msg='Request info: %s, %s, %s, %s', 
            args=('password=secret123', 123, {'key': 'value'}, None), 
            exc_info=None
        )
        
        # Apply the filter
        filter_obj.filter(record)
        
        # Verify the result
        assert record.args[0] == 'password=******'  # String arg is redacted
        assert record.args[1] == 123  # Non-string args remain unchanged
        assert record.args[2] == {'key': 'value'}
        assert record.args[3] is None


class Test_SensitiveInfoFilter_Filter_03_BoundaryBehaviors:
    """Tests for boundary behaviors of SensitiveInfoFilter.filter method."""

    def test_handle_empty_message_strings(self):
        """Test handling of empty message strings."""
        from smartsurge.logging_ import SensitiveInfoFilter
        
        filter_obj = SensitiveInfoFilter()
        
        # Create a log record with an empty message string
        record = logging.LogRecord(
            name='test', level=logging.INFO, 
            pathname='', lineno=0, msg='', 
            args=(), exc_info=None
        )
        
        # Apply the filter
        result = filter_obj.filter(record)
        
        # Verify the result
        assert result is True  # Filter should always return True
        assert record.msg == ''  # Empty message should remain empty

    def test_multiple_instances_of_sensitive_patterns(self):
        """Test processing a message with multiple instances of the same sensitive pattern."""
        from smartsurge.logging_ import SensitiveInfoFilter
        
        filter_obj = SensitiveInfoFilter()
        
        # Create a log record with multiple instances of the same sensitive pattern
        record = logging.LogRecord(
            name='test', level=logging.INFO, 
            pathname='', lineno=0, 
            msg='First password=secret123, then another password=abc123', 
            args=(), exc_info=None
        )
        
        # Apply the filter
        filter_obj.filter(record)
        
        # Verify all instances are redacted
        assert 'password=******' in record.msg
        assert record.msg.count('password=******') == 2
        assert 'secret123' not in record.msg
        assert 'abc123' not in record.msg


class Test_SensitiveInfoFilter_Filter_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of SensitiveInfoFilter.filter method."""

    def test_handle_regex_substitution_errors(self):
        """Test handling regex substitution errors without breaking the filtering process."""
        from unittest.mock import Mock
        from smartsurge.logging_ import SensitiveInfoFilter
        
        filter_obj = SensitiveInfoFilter()
        
        mock_pattern = Mock()
        mock_pattern.sub = r'(password=)([^&\s]+)'
        
        def mock_sub(repl, string):
            if 'trigger_error' in string:
                raise re.error("Simulated regex substitution error")
            return string
        
        mock_pattern.sub = mock_sub
        
        filter_obj.compiled_patterns[0] = mock_pattern
        
        # Create a log record that will trigger the substitution error
        record = logging.LogRecord(
            name='test', level=logging.INFO, 
            pathname='', lineno=0, 
            msg='trigger_error password=secret123', 
            args=(), exc_info=None
        )
        
        # Apply the filter - it should not raise an exception
        result = filter_obj.filter(record)
        assert result is True  # Filter should still return True
        assert 'secret123' in record.msg  # Message unchanged due to error

    def test_continue_processing_after_pattern_failure(self):
        """Test that processing continues with remaining patterns when one pattern fails."""
        from smartsurge.logging_ import SensitiveInfoFilter
        import unittest.mock as mock
        
        # Use a custom list with password as the first pattern to ensure it fails first
        filter_obj = SensitiveInfoFilter(patterns=[
            r'(password=)([^&\s]+)',
            r'(token=)([^&\s]+)',
        ])
        
         # Create a mock pattern that raises an exception when sub is called
        failing_pattern = mock.Mock()
        failing_pattern.sub.side_effect = Exception("Simulated error for password pattern")
        
        # Replace the first compiled pattern with our mock
        original_patterns = filter_obj.compiled_patterns
        filter_obj.compiled_patterns = [failing_pattern, original_patterns[1]]
        
        # Create a log record with multiple sensitive information types
        record = logging.LogRecord(
            name='test', level=logging.INFO, 
            pathname='', lineno=0, 
            msg='Request with password=secret123 and token=abc123', 
            args=(), exc_info=None
        )
        
        # Apply the filter
        filter_obj.filter(record)
        
        # Password should remain unchanged but token should be redacted
        assert 'password=secret123' in record.msg
        assert 'token=******' in record.msg
        assert 'abc123' not in record.msg


class Test_PerformanceFilter_Init_01_NominalBehaviors:
    """Tests for nominal behaviors of PerformanceFilter.__init__ method."""

    def test_initialize_with_specified_rates(self):
        """Test initializing with specified debug_sample_rate and info_sample_rate."""
        from smartsurge.logging_ import PerformanceFilter
        
        debug_rate = 0.25
        info_rate = 0.75
        filter_obj = PerformanceFilter(debug_sample_rate=debug_rate, info_sample_rate=info_rate)
        
        # Verify rates are set correctly
        assert filter_obj.debug_sample_rate == debug_rate
        assert filter_obj.info_sample_rate == info_rate

    def test_initialize_counters_to_zero(self):
        """Test that counters are initialized to zero."""
        from smartsurge.logging_ import PerformanceFilter
        
        filter_obj = PerformanceFilter()
        
        # Verify counters start at zero
        assert filter_obj.debug_count == 0
        assert filter_obj.info_count == 0

    def test_normalize_sampling_rates_within_range(self):
        """Test that sampling rates are normalized to the 0.0-1.0 range."""
        from smartsurge.logging_ import PerformanceFilter
        
        # Initialize with rates inside valid range
        filter_obj = PerformanceFilter(debug_sample_rate=0.3, info_sample_rate=0.7)
        
        # Verify rates are unchanged
        assert filter_obj.debug_sample_rate == 0.3
        assert filter_obj.info_sample_rate == 0.7


class Test_PerformanceFilter_Init_02_NegativeBehaviors:
    """Tests for negative behaviors of PerformanceFilter.__init__ method."""

    def test_clip_negative_sampling_rates(self):
        """Test that negative sampling rates are clipped to 0.0."""
        from smartsurge.logging_ import PerformanceFilter
        
        filter_obj = PerformanceFilter(debug_sample_rate=-0.5, info_sample_rate=-0.3)
        
        # Verify rates are clipped to 0.0
        assert filter_obj.debug_sample_rate == 0.0
        assert filter_obj.info_sample_rate == 0.0

    def test_clip_sampling_rates_above_one(self):
        """Test that sampling rates > 1.0 are clipped to 1.0."""
        from smartsurge.logging_ import PerformanceFilter
        
        filter_obj = PerformanceFilter(debug_sample_rate=1.5, info_sample_rate=2.0)
        
        # Verify rates are clipped to 1.0
        assert filter_obj.debug_sample_rate == 1.0
        assert filter_obj.info_sample_rate == 1.0


class Test_PerformanceFilter_Init_03_BoundaryBehaviors:
    """Tests for boundary behaviors of PerformanceFilter.__init__ method."""

    def test_initialize_with_zero_sampling_rates(self):
        """Test initializing with 0.0 sampling rates."""
        from smartsurge.logging_ import PerformanceFilter
        
        filter_obj = PerformanceFilter(debug_sample_rate=0.0, info_sample_rate=0.0)
        
        # Verify rates are set correctly
        assert filter_obj.debug_sample_rate == 0.0
        assert filter_obj.info_sample_rate == 0.0

    def test_initialize_with_one_sampling_rates(self):
        """Test initializing with 1.0 sampling rates."""
        from smartsurge.logging_ import PerformanceFilter
        
        filter_obj = PerformanceFilter(debug_sample_rate=1.0, info_sample_rate=1.0)
        
        # Verify rates are set correctly
        assert filter_obj.debug_sample_rate == 1.0
        assert filter_obj.info_sample_rate == 1.0

    def test_initialize_with_very_small_sampling_rates(self):
        """Test initializing with very small sampling rates."""
        from smartsurge.logging_ import PerformanceFilter
        
        filter_obj = PerformanceFilter(debug_sample_rate=0.001, info_sample_rate=0.001)
        
        # Verify rates are set correctly
        assert filter_obj.debug_sample_rate == 0.001
        assert filter_obj.info_sample_rate == 0.001


class Test_PerformanceFilter_Filter_01_NominalBehaviors:
    """Tests for nominal behaviors of PerformanceFilter.filter method."""

    def test_allow_debug_messages_based_on_rate(self):
        """Test that DEBUG messages are allowed based on the debug_sample_rate."""
        from smartsurge.logging_ import PerformanceFilter
        
        # Set 50% sampling rate - allow every 2nd message
        filter_obj = PerformanceFilter(debug_sample_rate=0.5)
        
        # Create DEBUG level records
        records = [
            logging.LogRecord(
                name='test', level=logging.DEBUG, 
                pathname='', lineno=0, msg=f'Debug message {i}', 
                args=(), exc_info=None
            ) for i in range(10)
        ]
        
        # Apply the filter to all records
        results = [filter_obj.filter(record) for record in records]
        
        # With 50% sampling rate (1/2), we expect True on indices 0, 2, 4, 6, 8
        assert results[0] is True
        assert results[1] is False
        assert results[2] is True
        assert results[3] is False
        assert results[4] is True

    def test_allow_info_messages_based_on_rate(self):
        """Test that INFO messages are allowed based on the info_sample_rate."""
        from smartsurge.logging_ import PerformanceFilter
        
        # Set 33% sampling rate - allow every 3rd message
        filter_obj = PerformanceFilter(info_sample_rate=1/3)
        
        # Create INFO level records
        records = [
            logging.LogRecord(
                name='test', level=logging.INFO, 
                pathname='', lineno=0, msg=f'Info message {i}', 
                args=(), exc_info=None
            ) for i in range(9)
        ]
        
        # Apply the filter to all records
        results = [filter_obj.filter(record) for record in records]
        
        # With 33% sampling rate (1/3), we expect True on indices 0, 3, 6
        assert results[0] is True
        assert results[1] is False
        assert results[2] is False
        assert results[3] is True
        assert results[4] is False
        assert results[5] is False
        assert results[6] is True

    def test_always_allow_warning_error_critical(self):
        """Test that WARNING, ERROR, and CRITICAL messages are always allowed."""
        from smartsurge.logging_ import PerformanceFilter
        
        # Set very low sampling rates
        filter_obj = PerformanceFilter(debug_sample_rate=0.01, info_sample_rate=0.01)
        
        # Create records for different levels
        warning_record = logging.LogRecord(
            name='test', level=logging.WARNING, 
            pathname='', lineno=0, msg='Warning message', 
            args=(), exc_info=None
        )
        
        error_record = logging.LogRecord(
            name='test', level=logging.ERROR, 
            pathname='', lineno=0, msg='Error message', 
            args=(), exc_info=None
        )
        
        critical_record = logging.LogRecord(
            name='test', level=logging.CRITICAL, 
            pathname='', lineno=0, msg='Critical message', 
            args=(), exc_info=None
        )
        
        # Apply the filter
        warning_result = filter_obj.filter(warning_record)
        error_result = filter_obj.filter(error_record)
        critical_result = filter_obj.filter(critical_record)
        
        # Verify all higher level messages are allowed
        assert warning_result is True
        assert error_result is True
        assert critical_result is True

    def test_increment_counters(self):
        """Test that counters are only incremented when sample rates are less than 1.0."""
        from smartsurge.logging_ import PerformanceFilter
        
        # Test with sample rates of 1.0 - counters should NOT increment
        filter_obj_no_sampling = PerformanceFilter(debug_sample_rate=1.0, info_sample_rate=1.0)
        
        # Create records for different levels
        debug_record = logging.LogRecord(
            name='test', level=logging.DEBUG, 
            pathname='', lineno=0, msg='Debug message', 
            args=(), exc_info=None
        )
        
        info_record = logging.LogRecord(
            name='test', level=logging.INFO, 
            pathname='', lineno=0, msg='Info message', 
            args=(), exc_info=None
        )
        
        # Apply the filter multiple times
        for _ in range(5):
            filter_obj_no_sampling.filter(debug_record)
        
        for _ in range(3):
            filter_obj_no_sampling.filter(info_record)
        
        # Verify counters are NOT incremented when sample rates are 1.0
        assert filter_obj_no_sampling.debug_count == 0
        assert filter_obj_no_sampling.info_count == 0
        
        # Test with sample rates less than 1.0 - counters should increment
        filter_obj_with_sampling = PerformanceFilter(debug_sample_rate=0.5, info_sample_rate=0.5)
        
        # Apply the filter multiple times
        for _ in range(5):
            filter_obj_with_sampling.filter(debug_record)
        
        for _ in range(3):
            filter_obj_with_sampling.filter(info_record)
        
        # Verify counters ARE incremented when sample rates are less than 1.0
        assert filter_obj_with_sampling.debug_count == 5
        assert filter_obj_with_sampling.info_count == 3

    def test_filter_based_on_modulo_operation(self):
        """Test that filtering is based on modulo operation with counter."""
        from smartsurge.logging_ import PerformanceFilter
        
        # Set 25% rate (allow every 4th message)
        filter_obj = PerformanceFilter(debug_sample_rate=0.25)
        
        # Create a DEBUG record
        debug_record = logging.LogRecord(
            name='test', level=logging.DEBUG, 
            pathname='', lineno=0, msg='Debug message', 
            args=(), exc_info=None
        )
        
        # Apply the filter multiple times and collect results
        results = [filter_obj.filter(debug_record) for _ in range(10)]
        
        # For 25% rate, with modulo 4, we expect True on indices 0, 4, 8
        assert results[0] is True
        assert results[1] is False
        assert results[2] is False
        assert results[3] is False
        assert results[4] is True
        assert results[5] is False
        assert results[6] is False
        assert results[7] is False
        assert results[8] is True
        assert results[9] is False


class Test_PerformanceFilter_Filter_02_NegativeBehaviors:
    """Tests for negative behaviors of PerformanceFilter.filter method."""

    def test_handle_unexpected_level_numbers(self):
        """Test handling of records with unexpected level numbers."""
        from smartsurge.logging_ import PerformanceFilter
        
        filter_obj = PerformanceFilter()
        
        # Create a record with a custom level number that isn't DEBUG or INFO
        custom_level = 15  # Between DEBUG (10) and INFO (20)
        custom_record = logging.LogRecord(
            name='test', level=custom_level, 
            pathname='', lineno=0, msg='Custom level message', 
            args=(), exc_info=None
        )
        
        # Apply the filter
        result = filter_obj.filter(custom_record)
        
        # Verify that non-standard levels are treated as if they were WARNING or higher
        assert result is True


class Test_PerformanceFilter_Filter_03_BoundaryBehaviors:
    """Tests for boundary behaviors of PerformanceFilter.filter method."""

    def test_debug_filter_one_shy_then_logs(self):
        """
        Tests PerformanceFilter for DEBUG messages:
        1. When the debug_count is one shy of the logging threshold, 
           the current message is NOT logged.
        2. The debug_count increments.
        3. The immediately following DEBUG message IS logged.
        """
        from smartsurge.logging_ import PerformanceFilter # As per user's test style

        threshold = 4
        sample_rate = 1 / threshold  # Log 1 every 4 messages. Threshold N = 4.
                            # Logs when debug_count is 0, 4, 8, ...
        
        filter_obj = PerformanceFilter(debug_sample_rate=sample_rate, info_sample_rate=1.0)
        
        # Set debug_count to N-1 (i.e., 3). This is "one shy" of 4.
        filter_obj.debug_count = threshold - 1 

        # First record (when debug_count is 3)
        record1 = _create_log_record(logging.DEBUG, "Debug message processed when count is N-1")
        
        result1 = filter_obj.filter(record1)
        
        # Assertions for the first record
        assert result1 is False, \
            "DEBUG Record when count is N-1 (one shy) should NOT be logged."
        assert filter_obj.debug_count == threshold, \
            f"DEBUG count should have incremented to N (expected {threshold}, got {filter_obj.debug_count})."

        # Second record (debug_count is now N, i.e., 4)
        record2 = _create_log_record(logging.DEBUG, "Debug message processed when count is N")
        
        result2 = filter_obj.filter(record2)
        
        # Assertions for the second record
        assert result2 is True, \
            "DEBUG Record when count is N (at threshold) SHOULD be logged."
        assert filter_obj.debug_count == threshold + 1, \
            f"DEBUG count should have incremented to N+1 (expected {threshold + 1}, got {filter_obj.debug_count})."

    def test_info_filter_one_shy_then_logs(self):
        """
        Tests PerformanceFilter for INFO messages:
        1. When the info_count is one shy of the logging threshold,
           the current message is NOT logged.
        2. The info_count increments.
        3. The immediately following INFO message IS logged.
        """
        from smartsurge.logging_ import PerformanceFilter # As per user's test style

        threshold = 2  # Log 1 every 2 messages. Threshold N = 2.
        sample_rate = 1 / threshold
                           # Logs when info_count is 0, 2, 4, ...
        
        filter_obj = PerformanceFilter(debug_sample_rate=1.0, info_sample_rate=sample_rate)
        
        # Set info_count to N-1 (i.e., 1). This is "one shy" of 2.
        filter_obj.info_count = threshold - 1

        # First record (when info_count is 1)
        record1 = _create_log_record(logging.INFO, "Info message processed when count is N-1")
        
        result1 = filter_obj.filter(record1)
        
        # Assertions for the first record
        assert result1 is False, \
            "INFO Record when count is N-1 (one shy) should NOT be logged."
        assert filter_obj.info_count == threshold, \
            f"INFO count should have incremented to N (expected {threshold}, got {filter_obj.info_count})."

        # Second record (info_count is now N, i.e., 2)
        record2 = _create_log_record(logging.INFO, "Info message processed when count is N")
        
        result2 = filter_obj.filter(record2)
        
        # Assertions for the second record
        assert result2 is True, \
            "INFO Record when count is N (at threshold) SHOULD be logged."
        assert filter_obj.info_count == threshold + 1, \
            f"INFO count should have incremented to N+1 (expected {threshold + 1}, got {filter_obj.info_count})."

    def test_filter_with_zero_sampling_rate(self):
        """Test filtering with 0% sampling rate (allow no messages)."""
        from smartsurge.logging_ import PerformanceFilter
        
        filter_obj = PerformanceFilter(debug_sample_rate=0.0, info_sample_rate=0.0)
        
        # Create DEBUG and INFO records
        debug_record = logging.LogRecord(
            name='test', level=logging.DEBUG, 
            pathname='', lineno=0, msg='Debug message', 
            args=(), exc_info=None
        )
        
        info_record = logging.LogRecord(
            name='test', level=logging.INFO, 
            pathname='', lineno=0, msg='Info message', 
            args=(), exc_info=None
        )
        
        # Apply the filter
        debug_result = filter_obj.filter(debug_record)
        info_result = filter_obj.filter(info_record)
        
        # Verify the results
        assert debug_result is False  # No DEBUG messages should be allowed
        assert info_result is False   # No INFO messages should be allowed

    def test_filter_with_full_sampling_rate(self):
        """Test filtering with 100% sampling rate (allow all messages)."""
        from smartsurge.logging_ import PerformanceFilter
        
        filter_obj = PerformanceFilter(debug_sample_rate=1.0, info_sample_rate=1.0)
        
        # Create DEBUG and INFO records
        debug_record = logging.LogRecord(
            name='test', level=logging.DEBUG, 
            pathname='', lineno=0, msg='Debug message', 
            args=(), exc_info=None
        )
        
        info_record = logging.LogRecord(
            name='test', level=logging.INFO, 
            pathname='', lineno=0, msg='Info message', 
            args=(), exc_info=None
        )
        
        # Apply the filter
        debug_result = filter_obj.filter(debug_record)
        info_result = filter_obj.filter(info_record)
        
        # Verify the results
        assert debug_result is True  # All DEBUG messages should be allowed
        assert info_result is True   # All INFO messages should be allowed

    def test_handle_very_small_sampling_rates(self):
        """Test handling of very small sampling rates."""
        from smartsurge.logging_ import PerformanceFilter
        
        # Use a very small sampling rate (0.001)
        filter_obj = PerformanceFilter(debug_sample_rate=0.001)
        
        # Create a DEBUG record
        debug_record = logging.LogRecord(
            name='test', level=logging.DEBUG, 
            pathname='', lineno=0, msg='Debug message', 
            args=(), exc_info=None
        )
        
        # Apply the filter many times
        results = [filter_obj.filter(debug_record) for _ in range(2000)]
        
        # With a rate of 0.001, we expect about 2 messages to pass in 2000 attempts
        # but at minimum, one message should pass (the first one)
        assert sum(results) >= 1
        assert sum(results) <= 5  # Allow for some variation


class Test_PerformanceFilter_Filter_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of PerformanceFilter.filter method."""

    def test_prevent_division_by_zero(self):
        """Test that division by zero is prevented for small sampling rates."""
        from smartsurge.logging_ import PerformanceFilter
        
        # Test with an extremely small rate that would normally cause division by zero
        filter_obj = PerformanceFilter(debug_sample_rate=0.0000001)
        
        # Create a DEBUG record
        debug_record = logging.LogRecord(
            name='test', level=logging.DEBUG, 
            pathname='', lineno=0, msg='Debug message', 
            args=(), exc_info=None
        )
        
        # Apply the filter - it should not raise ZeroDivisionError
        try:
            result = filter_obj.filter(debug_record)
            assert isinstance(result, bool)  # Should return a boolean without error
        except ZeroDivisionError:
            pytest.fail("filter() raised ZeroDivisionError unexpectedly")


class Test_PerformanceFilter_Filter_05_StateTransitionBehaviors:
    """Tests for state transition behaviors of PerformanceFilter.filter method."""

    def test_counter_increment_affects_filtering(self):
        """Test counter increment's effect on filtering decisions (state transition from 'allow' to 'block' and back)."""
        from smartsurge.logging_ import PerformanceFilter
        
        # Set 33% rate (allow every 3rd message)
        filter_obj = PerformanceFilter(debug_sample_rate=1/3)
        
        # Create a DEBUG record
        debug_record = logging.LogRecord(
            name='test', level=logging.DEBUG, 
            pathname='', lineno=0, msg='Debug message', 
            args=(), exc_info=None
        )
        
        # Apply the filter multiple times and observe transitions between
        # "allow" (True) and "block" (False) states based on counter value
        results = []
        for _ in range(9):  # Test through 3 cycles
            results.append(filter_obj.filter(debug_record))
            
        # With 33% rate (allowing every 3rd message), we expect:
        # - Allow index 0 (count=1) - divisible by 3 when using modulo
        # - Block index 1 (count=2) - not divisible by 3
        # - Block index 2 (count=3) - not divisible by 3
        # - Allow index 3 (count=4) - divisible by 3 when using modulo
        # - etc.
        assert results[0] is True   # First message (count=1)
        assert results[1] is False  # Second message (count=2)
        assert results[2] is False  # Third message (count=3)
        assert results[3] is True   # Fourth message (count=4)
        assert results[4] is False  # Fifth message (count=5)
        assert results[5] is False  # Sixth message (count=6)
        assert results[6] is True   # Seventh message (count=7)
        assert results[7] is False  # Eighth message (count=8)
        assert results[8] is False  # Ninth message (count=9)
        
        # Verify the counter has reached the expected value
        assert filter_obj.debug_count == 9


class Test_ConfigureLogging_01_NominalBehaviors:
    """Tests for nominal behaviors of configure_logging function."""

    def test_configure_with_specified_level(self):
        """Test configuring logger with specified level."""
        from smartsurge.logging_ import configure_logging
        
        # Test with numeric level
        logger = configure_logging(level=logging.DEBUG, console_output=False)
        assert logger.level == logging.DEBUG
        
        # Test with string level
        logger = configure_logging(level="WARNING", console_output=False)
        assert logger.level == logging.WARNING
        
        # Remove filters and handlers to preserve test isolation
        logger.handlers.clear()
        logger.filters.clear()

    def test_set_up_console_output(self):
        """Test setting up console output handler when requested."""
        from smartsurge.logging_ import configure_logging
        
        # Create logger with console output
        logger = configure_logging(console_output=True)
        
        # Verify console handler is added
        has_console_handler = any(
            isinstance(handler, logging.StreamHandler) and 
            handler.stream == sys.stdout 
            for handler in logger.handlers
        )
        assert has_console_handler
        
        # Create logger without console output
        logger = configure_logging(console_output=False)
        
        # Verify no console handler is added
        has_console_handler = any(
            isinstance(handler, logging.StreamHandler) and 
            handler.stream == sys.stdout 
            for handler in logger.handlers
        )
        assert not has_console_handler
        
        # Remove filters and handlers to preserve test isolation
        logger.handlers.clear()
        logger.filters.clear()

    def test_set_up_file_output(self):
        """Test setting up file output handler when requested."""
        from smartsurge.logging_ import configure_logging
        import tempfile
        
        # Create a temporary log file
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as temp_file:
            log_file = temp_file.name
        
        try:
            # Create logger with file output
            logger = configure_logging(output_file=log_file, console_output=False)
            
            # Verify file handler is added
            has_file_handler = any(
                isinstance(handler, logging.FileHandler) and 
                handler.baseFilename == log_file 
                for handler in logger.handlers
            )
            assert has_file_handler
            
            # Test that we can write to the log file
            test_message = "Test file logging"
            logger.info(test_message)
            
            # Verify message was written to file
            with open(log_file, 'r') as f:
                log_content = f.read()
                assert test_message in log_content
        finally:
            # Clean up temporary file
            for handler in logger.handlers:
                handler.close()
                logger.removeHandler(handler)
            
            if os.path.exists(log_file):
                os.unlink(log_file)
            
            # Remove filters and handlers to preserve test isolation
            logger.handlers.clear()
            logger.filters.clear()

    def test_apply_sensitive_info_filter(self):
        """Test applying sensitive information filter when filter_sensitive=True."""
        from smartsurge.logging_ import configure_logging, SensitiveInfoFilter
        
        # Create logger with sensitive info filtering and ensure we have at least one handler
        logger = configure_logging(filter_sensitive=True, console_output=True)
        
        # Verify SensitiveInfoFilter is added to at least one handler
        has_sensitive_filter = any(
            any(isinstance(filter_, SensitiveInfoFilter) for filter_ in handler.filters)
            for handler in logger.handlers
        )
        assert has_sensitive_filter
        
        # Remove filters and handlers to preserve test isolation
        logger.handlers.clear()
        logger.filters.clear()

    def test_apply_performance_filter(self):
        """Test applying performance filter when sample rates < 1.0."""
        from smartsurge.logging_ import configure_logging, PerformanceFilter
        
        # Create logger with performance filtering
        logger = configure_logging(
            debug_sample_rate=0.5, 
            info_sample_rate=0.5, 
            console_output=False
        )
        
        # Verify PerformanceFilter is added
        has_performance_filter = any(
            isinstance(filter_, PerformanceFilter)
            for filter_ in logger.filters
        )
        assert has_performance_filter
        
        # Remove filters and handlers to preserve test isolation
        logger.handlers.clear()
        logger.filters.clear()


class Test_ConfigureLogging_02_NegativeBehaviors:
    """Tests for negative behaviors of configure_logging function."""

    def test_handle_invalid_log_level_strings(self):
        """Test handling invalid log level strings by defaulting to INFO."""
        from smartsurge.logging_ import configure_logging
        
        # Test with invalid string level
        logger = configure_logging(level="INVALID_LEVEL", console_output=False)
        
        # Should default to INFO
        assert logger.level == logging.INFO
        
        # Remove filters and handlers to preserve test isolation
        logger.handlers.clear()
        logger.filters.clear()

    def test_skip_invalid_handlers(self):
        """Test skipping invalid handlers in additional_handlers."""
        from smartsurge.logging_ import configure_logging
        
        # Create a mock for warning method to check it's called
        with patch.object(logging.Logger, 'warning') as mock_warning:
            # Try to add an invalid handler
            logger = configure_logging(
                additional_handlers={'invalid_handler': 'not a handler'},
                console_output=False
            )
            
            # Check that warning was called
            mock_warning.assert_called_with("Invalid handler provided: invalid_handler")
            
            # Remove filters and handlers to preserve test isolation
            logger.handlers.clear()
            logger.filters.clear()

    def test_handle_missing_directory_for_log_files(self):
        """Test handling missing directory for log files by creating it."""
        import tempfile
        import shutil
        
        logger_instance = logging.getLogger("smartsurge")
        original_filters = list(logger_instance.filters)  # Make a copy of current filters

        # Clear filters before the test runs
        logger_instance.filters.clear()
        
        # Create a temporary directory structure
        temp_dir = tempfile.mkdtemp()
        log_dir = os.path.join(temp_dir, 'logs')
        log_file = 'test.log'
        
        try:
            # Verify directory doesn't exist yet
            assert not os.path.exists(log_dir)
            
            # Configure logging with nonexistent directory
            logger = configure_logging(
                output_file=log_file,
                log_directory=log_dir,
                console_output=False
            )
            
            # Directory should be created
            assert os.path.exists(log_dir)
            
            # Full path to log file should exist
            full_log_path = os.path.join(log_dir, log_file)
            
            # Test writing to the log file
            test_message = "Test log directory creation"
            logger.info(test_message)
            
            # Flush and close all handlers
            for handler in logger.handlers:
                handler.flush()
                handler.close()
            
            # Verify log file was created in the directory
            assert os.path.exists(full_log_path)
            
            # Verify message was written to file
            with open(full_log_path, 'r') as f:
                log_content = f.read()
                assert test_message in log_content
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            # Clear any filters that might have been added during the test
            logger_instance.filters.clear()

            # Restore the original filters
            for f_orig in original_filters:
                logger_instance.addFilter(f_orig)
            
            # Remove filters and handlers to preserve test isolation
            logger.handlers.clear()
            logger.filters.clear()


class Test_ConfigureLogging_03_BoundaryBehaviors:
    """Tests for boundary behaviors of configure_logging function."""

    def test_function_with_minimum_log_level(self, caplog):
        """Test function behavior with minimum log level (DEBUG)."""
        from smartsurge.logging_ import configure_logging
        
        # Configure with DEBUG level
        logger = configure_logging(level=logging.DEBUG, console_output=False)
        
        # DEBUG messages should be logged
        with caplog.at_level(logging.DEBUG):
            logger.debug("Debug message")
            assert "Debug message" in caplog.text
            
        # Remove filters and handlers to preserve test isolation
        logger.handlers.clear()
        logger.filters.clear()

    def test_function_with_maximum_log_level(self, caplog):
        """Test function behavior with maximum log level (CRITICAL)."""
        from smartsurge.logging_ import configure_logging
        
        # Configure with CRITICAL level
        logger = configure_logging(level=logging.CRITICAL, console_output=False)
        
        # Only CRITICAL messages should be logged
        with caplog.at_level(logging.DEBUG):
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")
            
            assert "Debug message" not in caplog.text
            assert "Info message" not in caplog.text
            assert "Warning message" not in caplog.text
            assert "Error message" not in caplog.text
            assert "Critical message" in caplog.text
            
        # Remove filters and handlers to preserve test isolation
        logger.handlers.clear()
        logger.filters.clear()

    def test_configure_with_zero_percent_sampling_rates(self):
        """Test configure with 0% sampling rates."""
        
        # Get the logger instance
        logger_instance = logging.getLogger("smartsurge")
        
        # Clear any handlers or filters that might have been added during tests
        logger_instance.handlers.clear()
        logger_instance.filters.clear()
        
        # Configure with 0.0 sampling rates
        logger = configure_logging(
            debug_sample_rate=0.0,
            info_sample_rate=0.0,
            console_output=False
        )
        
        # Verify PerformanceFilter is added with zero rates
        performance_filter = next(
            (f for f in logger.filters if isinstance(f, PerformanceFilter)),
            None
        )
        assert performance_filter is not None
        assert performance_filter.debug_sample_rate == 0.0
        assert performance_filter.info_sample_rate == 0.0
        
        # Clear any handlers or filters that might have been added during tests
        logger_instance.handlers.clear()
        logger_instance.filters.clear()


class Test_ConfigureLogging_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of configure_logging function."""

    def test_handle_directory_creation_errors(self):
        """Test handling directory creation errors when creating log_directory."""
        from smartsurge.logging_ import configure_logging
        
        # Mock os.makedirs to raise an error
        with patch('os.makedirs', side_effect=PermissionError("Permission denied")):
            # This should not raise an exception but log an error and continue without file handler
            logger = configure_logging(
                output_file="test.log",
                log_directory="/invalid/directory",
                console_output=True
            )
            
            # Verify there's no file handler
            has_file_handler = any(
                isinstance(handler, logging.FileHandler)
                for handler in logger.handlers
            )
            assert not has_file_handler
            
            # Remove filters and handlers to preserve test isolation
            logger.handlers.clear()
            logger.filters.clear()

    def test_handle_file_creation_errors(self):
        """Test handling file creation errors for output_file."""
        from smartsurge.logging_ import configure_logging
        
        with patch('logging.FileHandler', side_effect=PermissionError("Permission denied")):
            logger = configure_logging(
                output_file="/invalid/path/test.log",
                console_output=True
            )
            
            # Check by class name instead of using isinstance
            has_file_handler = any(
                handler.__class__.__name__ == 'FileHandler'
                for handler in logger.handlers
            )
            assert not has_file_handler
            
            # Remove filters and handlers to preserve test isolation
            logger.handlers.clear()
            logger.filters.clear()

    def test_log_warnings_for_invalid_handlers(self):
        """Test logging warnings for invalid handlers instead of failing."""
        from smartsurge.logging_ import configure_logging
        
        # Create a real handler and an invalid one
        valid_handler = logging.NullHandler()
        
        # Create a mock for warning method
        with patch.object(logging.Logger, 'warning') as mock_warning:
            # Configure with a mix of valid and invalid handlers
            logger = configure_logging(
                additional_handlers={
                    'valid': valid_handler,
                    'invalid': 'not a handler'
                },
                console_output=False
            )
            
            # Check that valid handler was added
            assert valid_handler in logger.handlers
            
            # Check that warning was called for invalid handler
            mock_warning.assert_called_with("Invalid handler provided: invalid")
            
            # Remove filters and handlers to preserve test isolation
            logger.handlers.clear()
            logger.filters.clear()


class Test_ConfigureLogging_05_StateTransitionBehaviors:
    """Tests for state transition behaviors of configure_logging function."""

    def test_clear_existing_handlers(self):
        """Test clearing existing handlers before adding new ones to prevent duplicate logging."""
        # Clear any existing handlers first
        logger = logging.getLogger("smartsurge")
        logger.handlers.clear()
        
        # Set up logger with initial handlers
        initial_handler = logging.NullHandler()
        logger.addHandler(initial_handler)
        
        # Verify initial handler is present
        assert initial_handler in logger.handlers
        assert len(logger.handlers) == 1
        
        # Configure logging - should clear existing handlers
        configure_logging(console_output=True)
        
        # Verify initial handler is removed
        assert initial_handler not in logger.handlers
        
        # Verify new handler is added (should be console handler)
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)
        
        # Remove filters and handlers to preserve test isolation
        logger.handlers.clear()
        logger.filters.clear()


class Test_GetLogger_01_NominalBehaviors:
    """Tests for nominal behaviors of get_logger function."""

    def test_return_logger_with_specified_name(self):
        """Test returning logger with the specified name prefixed with 'smartsurge.'"""
        from smartsurge.logging_ import get_logger
        
        logger_name = "test_module"
        logger = get_logger(logger_name)
        
        # Verify logger name has correct prefix
        assert logger.name == f"smartsurge.{logger_name}"
        
        # Remove filters and handlers to preserve test isolation
        logger.handlers.clear()
        logger.filters.clear()

    def test_set_logger_level_numeric(self):
        """Test setting logger level when numeric level is provided."""
        from smartsurge.logging_ import get_logger
        
        logger = get_logger("test_numeric_level", level=logging.DEBUG)
        
        # Verify level is set correctly
        assert logger.level == logging.DEBUG
        
        # Remove filters and handlers to preserve test isolation
        logger.handlers.clear()
        logger.filters.clear()

    def test_set_logger_level_string(self):
        """Test setting logger level when string level is provided."""
        from smartsurge.logging_ import get_logger
        
        logger = get_logger("test_string_level", level="WARNING")
        
        # Verify level is set correctly
        assert logger.level == logging.WARNING
        
        # Remove filters and handlers to preserve test isolation
        logger.handlers.clear()
        logger.filters.clear()


class Test_GetLogger_02_NegativeBehaviors:
    """Tests for negative behaviors of get_logger function."""

    def test_handle_invalid_level_strings(self):
        """Test handling invalid level strings by defaulting to INFO."""
        from smartsurge.logging_ import get_logger
        
        logger = get_logger("test_invalid_level", level="INVALID_LEVEL")
        
        # Should default to INFO
        assert logger.level == logging.INFO
        
        # Remove filters and handlers to preserve test isolation
        logger.handlers.clear()
        logger.filters.clear()


class Test_GetLogger_03_BoundaryBehaviors:
    """Tests for boundary behaviors of get_logger function."""

    def test_function_with_minimum_log_level(self):
        """Test function behavior with minimum log level (DEBUG)."""
        from smartsurge.logging_ import get_logger
        
        logger = get_logger("test_min_level", level=logging.DEBUG)
        
        # Verify level is set correctly
        assert logger.level == logging.DEBUG
        
        # Remove filters and handlers to preserve test isolation
        logger.handlers.clear()
        logger.filters.clear()

    def test_function_with_maximum_log_level(self):
        """Test function behavior with maximum log level (CRITICAL)."""
        from smartsurge.logging_ import get_logger
        
        logger = get_logger("test_max_level", level=logging.CRITICAL)
        
        # Verify level is set correctly
        assert logger.level == logging.CRITICAL
        
        # Remove filters and handlers to preserve test isolation
        logger.handlers.clear()
        logger.filters.clear()
