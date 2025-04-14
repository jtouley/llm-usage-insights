"""Logging configuration module using structlog and standard logging."""

import structlog
import logging
import sys


def setup_logging():
    """Configure and return a structlog logger instance.

    Sets up structured logging with ISO timestamps, log levels, and JSON rendering.
    Outputs to stdout at INFO level.
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger()
