"""
Logging configuration for SmartSurge.

This module provides functions to configure logging for the SmartSurge library.
"""

import logging
import os
import sys
from typing import Optional, Dict, Any, Union

def configure_logging(
    level: Union[int, str] = logging.INFO,
    format_string: Optional[str] = None,
    output_file: Optional[str] = None,
    capture_warnings: bool = True,
    console_output: bool = True,
    log_directory: Optional[str] = None,
    additional_handlers: Optional[Dict[str, Any]] = None,
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
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    
    # Add file handler if requested
    if output_file:
        if log_directory:
            # Ensure log directory exists
            os.makedirs(log_directory, exist_ok=True)
            output_file = os.path.join(log_directory, output_file)
        
        file_handler = logging.FileHandler(output_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    
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
