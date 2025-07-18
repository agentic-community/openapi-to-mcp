"""
Simplified configuration for OpenAPI to MCP converter.
Provides path helpers and basic utilities.
All configuration parameters are in config/config.yml via config_loader.
"""

import logging
from pathlib import Path

from .services.config_loader import config

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    # Define log message format
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


def get_temp_dir() -> Path:
    """Get the temporary directory path from config."""
    return config.get_path("temp_dir", "./temp")


def get_templates_dir() -> Path:
    """Get the templates directory path from config."""
    return config.get_path("templates_dir", "./templates")


def get_output_dir() -> Path:
    """Get the output directory path from config."""
    return config.get_path("output_dir", "./output")


def ensure_directories() -> None:
    """Ensure all required directories exist."""
    try:
        directories = [get_temp_dir(), get_templates_dir(), get_output_dir()]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        raise


# Ensure directories exist on import
ensure_directories()


if __name__ == "__main__":
    """Standalone testing of simplified configuration."""
    logger.info("Testing simplified configuration module...")
    
    # Test directory helpers
    logger.info(f"Temp dir: {get_temp_dir()}")
    logger.info(f"Templates dir: {get_templates_dir()}")
    logger.info(f"Output dir: {get_output_dir()}")
    
    # Test config loader access
    logger.info(f"Model from config.yml: {config.get_str('model')}")
    logger.info(f"Max tokens from config.yml: {config.get_int('max_tokens')}")
    
    logger.info("Simplified configuration module test completed successfully")