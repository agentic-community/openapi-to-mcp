"""
Configuration loader for YAML-based configuration.
Loads application configuration from config/config.yml.

Sample input: config/config.yml with model, max_tokens parameters
Expected output: ConfigLoader instance with loaded configuration
"""

import yaml
import logging
from pathlib import Path
from typing import Any, Dict

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    # Define log message format
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


def _load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Loaded configuration as dictionary
    """
    try:
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return {}
            
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        if config is None:
            logger.warning(f"Empty configuration file: {config_path}")
            return {}
            
        logger.info(f"Loaded configuration from {config_path}")
        
        # Print configuration parameters as key-value pairs
        if config:
            logger.info("📋 Configuration parameters loaded:")
            for key, value in config.items():
                logger.info(f"   {key}: {value}")
        
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return {}


class ConfigLoader:
    """Configuration loader for YAML-based configuration."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize the configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_path = self.config_dir / "config.yml"
        self._config = {}
        self._load_config()
        
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        self._config = _load_yaml_config(self.config_path)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self._config.get(key, default)
    
    def get_str(self, key: str, default: str = "") -> str:
        """Get string configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            String configuration value
        """
        value = self.get(key, default)
        return str(value) if value is not None else default
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Integer configuration value
        """
        value = self.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Float configuration value
        """
        value = self.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Boolean configuration value
        """
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "yes", "y", "1", "on")
        return bool(value)
    
    def get_path(self, key: str, default: str = "") -> Path:
        """Get path configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Path configuration value
        """
        value = self.get_str(key, default)
        return Path(value)
    
    def reload(self) -> None:
        """Reload configuration from disk."""
        self._load_config()
        logger.info("Configuration reloaded")
    
    @property
    def all(self) -> Dict[str, Any]:
        """Get all configuration values.
        
        Returns:
            All configuration values
        """
        return dict(self._config)


# Global configuration instance
config = ConfigLoader()


if __name__ == "__main__":
    """Standalone testing of configuration loader."""
    import sys
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    # Write test configuration
    test_config_dir = Path("./test_config")
    test_config_dir.mkdir(exist_ok=True)
    test_config_path = test_config_dir / "config.yml"
    
    with open(test_config_path, "w") as f:
        f.write("""
model: test-model
max_tokens: 1000
temperature: 0.5
debug: true
temp_dir: ./test_temp
        """)
    
    # Test 1: Load test configuration
    total_tests += 1
    test_loader = ConfigLoader(config_dir=str(test_config_dir))
    if not test_loader._config:
        all_validation_failures.append("Failed to load test configuration")
    
    # Test 2: Get string value
    total_tests += 1
    model = test_loader.get_str("model")
    expected = "test-model"
    if model != expected:
        all_validation_failures.append(f"get_str: Expected {expected}, got {model}")
    
    # Test 3: Get integer value
    total_tests += 1
    max_tokens = test_loader.get_int("max_tokens")
    expected = 1000
    if max_tokens != expected:
        all_validation_failures.append(f"get_int: Expected {expected}, got {max_tokens}")
    
    # Test 4: Get float value
    total_tests += 1
    temperature = test_loader.get_float("temperature")
    expected = 0.5
    if temperature != expected:
        all_validation_failures.append(f"get_float: Expected {expected}, got {temperature}")
    
    # Test 5: Get boolean value
    total_tests += 1
    debug = test_loader.get_bool("debug")
    expected = True
    if debug != expected:
        all_validation_failures.append(f"get_bool: Expected {expected}, got {debug}")
    
    # Test 6: Get path value
    total_tests += 1
    temp_dir = test_loader.get_path("temp_dir")
    expected = Path("./test_temp")
    if temp_dir != expected:
        all_validation_failures.append(f"get_path: Expected {expected}, got {temp_dir}")
    
    # Test 7: Get non-existent value with default
    total_tests += 1
    default_value = "default"
    non_existent = test_loader.get_str("non_existent", default=default_value)
    if non_existent != default_value:
        all_validation_failures.append(f"Default value: Expected {default_value}, got {non_existent}")
    
    # Clean up test files
    test_config_path.unlink()
    test_config_dir.rmdir()
    
    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)  # Exit with error code
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Configuration loader is validated and ready for use")
        sys.exit(0)  # Exit with success code