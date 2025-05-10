"""
Logging configuration for SmartSurge.

This module provides functions to configure logging for the SmartSurge library.
It includes features for filtering sensitive information and controlling log levels.
"""

import logging
import os
import sys
import re
from typing import Optional, Dict, Any, Union, List


# Default patterns for sensitive information
DEFAULT_SENSITIVE_PATTERNS = [
    r'(password=)([^&\s]+)',
    r'(token=)([^&\s]+)',
    r'(key=)([^&\s]+)',
    r'(secret=)([^&\s]+)',
    r'(auth=)([^&\s]+)',
    r'(Authorization:)\s*([^\s]+)',
    r'(apikey=)([^&\s]+)',
]

class SensitiveInfoFilter(logging.Filter):
    """
    Filter that redacts sensitive information from log messages.
    
    This filter searches for patterns that might contain sensitive information
    (like passwords, tokens, etc.) and replaces them with a redacted version.
    """
    def __init__(self, patterns: Optional[List[str]] = None, replacement: str = '******'):
        """
        Initialize the filter with patterns to redact.
        
        Args:
            patterns: Regex patterns for sensitive information.
                     Defaults to `DEFAULT_SENSITIVE_PATTERNS`.
            replacement: String to use for redaction.
        """
        super().__init__()
        self.patterns = patterns or DEFAULT_SENSITIVE_PATTERNS
        self.replacement = replacement
        
        # Compile patterns for better performance
        self.compiled_patterns = [re.compile(pattern) for pattern in self.patterns]
        
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log records by redacting sensitive information.
        
        Args:
            record: LogRecord to filter
            
        Returns:
            True (always allows the record, but redacts sensitive info)
        """
        def replacement_func(match):
            return match.group(1) + self.replacement

        if isinstance(record.msg, str):
            for pattern in self.compiled_patterns:
                try:
                    record.msg = pattern.sub(replacement_func, record.msg)
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Error applying sensitive info filter: {e}")
        
        # Also check args if they are strings
        if hasattr(record, 'args') and record.args:
            args = list(record.args)
            for i, arg in enumerate(args):
                if isinstance(arg, str):
                    arg_accum = arg
                    for pattern in self.compiled_patterns:
                        try:
                            arg_accum = pattern.sub(replacement_func, arg_accum)
                        except Exception as e:
                            logger = logging.getLogger(__name__)
                            logger.warning(f"Error applying sensitive info filter: {e}")
                    args[i] = arg_accum
            record.args = tuple(args)
        
        return True

class PerformanceFilter(logging.Filter):
    """
    Filter that limits the volume of log messages for better performance.
    
    This filter only allows a certain percentage of messages through at
    DEBUG and INFO levels, to reduce log volume in high-traffic scenarios.
    """
    def __init__(self, debug_sample_rate: float = 0.1, info_sample_rate: float = 0.5):
        """
        Initialize the filter with sampling rates.
        
        Args:
            debug_sample_rate: Fraction of DEBUG messages to allow (0.0-1.0)
            info_sample_rate: Fraction of INFO messages to allow (0.0-1.0)
        """
        super().__init__()
        self.debug_sample_rate = max(0.0, min(1.0, debug_sample_rate))
        self.info_sample_rate = max(0.0, min(1.0, info_sample_rate))
        self.debug_count = 0
        self.info_count = 0
        
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log records based on sampling rates.
        
        Args:
            record: LogRecord to filter
            
        Returns:
            True if the record should be logged, False otherwise
        """
        if record.levelno == logging.DEBUG:
            if self.debug_sample_rate >= 1.0:  # Explicitly pass all messages when rate is 1.0
                return True
            elif self.debug_sample_rate <= 0.0:
                return False
            else:
                result = (self.debug_count % int(1/max(0.001, self.debug_sample_rate)) == 0)
                self.debug_count += 1
                return result
        elif record.levelno == logging.INFO:
            if self.info_sample_rate >= 1.0:  # Explicitly pass all messages when rate is 1.0
                return True
            elif self.info_sample_rate <= 0.0:
                return False
            else:
                result = (self.info_count % int(1/max(0.001, self.info_sample_rate)) == 0)
                self.info_count += 1
                return result
        else:
            # Always log WARNING, ERROR, CRITICAL
            return True

def configure_logging(
    level: Union[int, str] = logging.INFO,
    format_string: Optional[str] = None,
    output_file: Optional[str] = None,
    capture_warnings: bool = True,
    console_output: bool = True,
    log_directory: Optional[str] = None,
    additional_handlers: Optional[Dict[str, Any]] = None,
    filter_sensitive: bool = True,
    sensitive_patterns: Optional[List[str]] = None,
    debug_sample_rate: float = 1.0,
    info_sample_rate: float = 1.0,
) -> logging.Logger:
    """
    Configure logging for the SmartSurge library.
    
    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string for log messages
        output_file: File to write logs to
        capture_warnings: Whether to capture warnings via logging
        console_output: Whether to output logs to console
        log_directory: Directory to store log files
        additional_handlers: Additional logging handlers to add
        filter_sensitive: Whether to filter sensitive information
        sensitive_patterns: Custom patterns for sensitive information
        debug_sample_rate: Fraction of DEBUG messages to log (0.0-1.0)
        info_sample_rate: Fraction of INFO messages to log (0.0-1.0)
        
    Returns:
        The configured logger.
        
    Example:
        >>> from smartsurge import configure_logging
        >>> logger = configure_logging(level="DEBUG", output_file="smartsurge.log")
    """
    # Convert string level to integer if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Set up the logger
    logger = logging.getLogger("smartsurge")
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicate logs
    if logger.handlers:
        logger.handlers.clear()
    
    # Default format if not specified
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Add filters
    if filter_sensitive:
        sensitive_filter = SensitiveInfoFilter(sensitive_patterns)
    
    if debug_sample_rate < 1.0 or info_sample_rate < 1.0:
        performance_filter = PerformanceFilter(
            debug_sample_rate=debug_sample_rate, 
            info_sample_rate=info_sample_rate
        )
        logger.addFilter(performance_filter)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        if filter_sensitive:
            console_handler.addFilter(sensitive_filter)
        
        logger.addHandler(console_handler)
    
    # Add file handler if requested
    if output_file:
        try:    
            if log_directory:
                # Ensure log directory exists with absolute path handling
                log_directory = os.path.abspath(log_directory)
                os.makedirs(log_directory, exist_ok=True)
                output_file = os.path.join(log_directory, os.path.basename(output_file))
                
            file_handler = logging.FileHandler(output_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            if filter_sensitive:
                file_handler.addFilter(sensitive_filter)
            
            try:
                # Test if file is actually writable
                with open(output_file, 'a') as f:
                    pass  # Just checking if we can open for appending
            except (OSError, PermissionError) as e:
                raise ValueError(f"Log file '{output_file}' is not writable: {e}")
            
            logger.addHandler(file_handler)
            if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
                logger.warning(f"File handler for '{output_file}' was not properly added")
            
            # Verify handler was added successfully by writing a test message
            logger.debug(f"Logging configured with file handler for {output_file}")
        except (OSError, PermissionError) as e:
            if not console_output:
                # If file handler failed and no console output is configured, add emergency console handler
                console_handler = logging.StreamHandler(sys.stderr)
                console_handler.setFormatter(formatter)
                console_handler.setLevel(logging.WARNING)  # Only warnings and above
                logger.addHandler(console_handler)
            
            logger.warning(f"Could not set up log file '{output_file}': {e}")
            
        except Exception as e:
            # Log unexpected errors
            if not console_output:
                # If file handler failed and no console output is configured, add emergency console handler
                console_handler = logging.StreamHandler(sys.stderr)
                console_handler.setFormatter(formatter)
                console_handler.setLevel(logging.WARNING)  # Only warnings and above
                logger.addHandler(console_handler)
            logger.error(f"Unexpected error creating file handler: {e}", file=sys.stderr)
    
    # Add any additional handlers
    if additional_handlers:
        for handler_name, handler in additional_handlers.items():
            if isinstance(handler, logging.Handler):
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            else:
                logger.warning(f"Invalid handler provided: {handler_name}")
    
    # Configure capturing warnings
    if capture_warnings:
        logging.captureWarnings(True)
    
    logger.debug(f"Logging configured at level {logging.getLevelName(level)}")
    return logger

def get_logger(name: str, level: Optional[Union[int, str]] = None) -> logging.Logger:
    """
    Get a logger with the given name, inheriting from the SmartSurge root logger.
    
    Args:
        name: Logger name
        level: Optional specific level for this logger
        
    Returns:
        A configured logger
    """
    logger = logging.getLogger(f"smartsurge.{name}")
    
    if level is not None:
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(level)
    
    return logger
