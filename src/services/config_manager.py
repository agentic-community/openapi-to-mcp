"""
Configuration and environment management utilities.
Provides functions for provider detection, logging setup, and environment information display.

Sample input: model="bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
Expected output: provider="bedrock"
"""

import logging
from pathlib import Path
from .config_loader import config

logger = logging.getLogger(__name__)


def _detect_provider_from_model(model: str) -> str:
    """Detect provider from model string using LiteLLM conventions."""
    if model.startswith("bedrock/"):
        return "bedrock"
    elif "claude" in model.lower() and not model.startswith("bedrock/"):
        return "anthropic"
    else:
        # Default to anthropic for unknown models
        logger.warning(f"Unknown model format '{model}', defaulting to anthropic provider")
        return "anthropic"


def _setup_logging(verbose: bool) -> None:
    """Setup logging configuration based on verbose flag."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")


def _show_environment_info() -> None:
    """Display current environment configuration."""
    try:
        model = config.get_str("model")
        max_tokens = config.get_int("max_tokens", 4096)
        temperature = config.get_float("temperature", 0.1)
        timeout_seconds = config.get_int("timeout_seconds", 300)
        debug = config.get_bool("debug", False)
        
        templates_dir = config.get_path("templates_dir", "./templates")
        temp_dir = config.get_path("temp_dir", "./temp")
        output_dir = config.get_path("output_dir", "./output")
        
        print(f"\n{'='*60}")
        print(f"🔧 ENVIRONMENT CONFIGURATION")
        print(f"{'='*60}")
        print(f"📍 Current Directory: {Path.cwd()}")
        print(f"🤖 Model: {model}")
        print(f"🏢 Provider: {_detect_provider_from_model(model)}")
        print(f"📊 Max Tokens: {max_tokens}")
        print(f"🌡️  Temperature: {temperature}")
        print(f"⏱️  Timeout: {timeout_seconds}s")
        print(f"🐛 Debug Mode: {debug}")
        
        print(f"\n🔑 PROVIDER:")
        print(f"   Using Provider: {_detect_provider_from_model(model)}")
        
        print(f"\n📂 DIRECTORIES:")
        print(f"   Templates: {templates_dir}")
        print(f"   Temp: {temp_dir}")
        print(f"   Output: {output_dir}")
        
        print(f"\n🔧 CONFIGURATION FILES:")
        config_file = Path("config/config.yml")
        print(f"   config.yml: {'✅ Found' if config_file.exists() else '❌ Not found'}")
        
        if config_file.exists():
            print(f"   config.yml path: {config_file.absolute()}")
        
        print(f"{'='*60}")
    except Exception as e:
        logger.error(f"Failed to show environment info: {e}")
        print(f"❌ Error reading environment configuration: {e}")


if __name__ == "__main__":
    import sys
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    # Test 1: Provider detection - bedrock
    total_tests += 1
    test_model = "bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    result = _detect_provider_from_model(test_model)
    expected = "bedrock"
    if result != expected:
        all_validation_failures.append(f"Bedrock detection: Expected {expected}, got {result}")
    
    # Test 2: Provider detection - anthropic
    total_tests += 1
    test_model = "claude-3-5-sonnet-20241022"
    result = _detect_provider_from_model(test_model)
    expected = "anthropic"
    if result != expected:
        all_validation_failures.append(f"Anthropic detection: Expected {expected}, got {result}")
    
    # Test 3: Provider detection - unknown (defaults to anthropic)
    total_tests += 1
    test_model = "unknown-model"
    result = _detect_provider_from_model(test_model)
    expected = "anthropic"
    if result != expected:
        all_validation_failures.append(f"Unknown model detection: Expected {expected}, got {result}")
    
    # Test 4: Logging setup (basic functionality test)
    total_tests += 1
    try:
        original_level = logging.getLogger().level
        _setup_logging(True)
        debug_level = logging.getLogger().level
        _setup_logging(False)
        if debug_level != logging.DEBUG:
            all_validation_failures.append(f"Verbose logging: Expected DEBUG level, got {debug_level}")
    except Exception as e:
        all_validation_failures.append(f"Logging setup error: {e}")
    
    # Test 5: Environment info display (basic functionality test)
    total_tests += 1
    try:
        _show_environment_info()
        # If no exception raised, test passes
    except Exception as e:
        all_validation_failures.append(f"Environment info display error: {e}")
    
    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Configuration manager functions are validated and ready for use")
        sys.exit(0)