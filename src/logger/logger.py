import sys
from loguru import logger
import os

def setup_logger():
    """Sets up the Loguru logger to output to console and a file."""
    # Ensure the logs directory exists
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Remove default handler
    logger.remove()

    # Configure console logger
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
        backtrace=True,
        diagnose=True
    )

    # Configure file logger
    log_file_path = os.path.join(log_dir, "trading_system.log")
    logger.add(
        log_file_path,
        level="DEBUG", # Log DEBUG level and above to file
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB", # Rotate the log file when it reaches 10 MB
        retention="7 days", # Keep logs for 7 days
        enqueue=True, # Make logging thread-safe
        backtrace=True,
        diagnose=True,
        serialize=False # Write logs in plain text
    )
    logger.info("Logger setup complete.")

# Initialize the logger immediately
setup_logger()

# Expose the logger instance for importing
log = logger
