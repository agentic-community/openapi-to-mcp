"""
OpenAPI specification loading utilities.
Handles loading OpenAPI specs from files and URLs with proper error handling.

Sample input: filename="api-spec.yaml" or url="https://example.com/openapi.json"
Expected output: OpenAPI specification content as string
"""

import httpx
import logging
import argparse
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _load_openapi_spec(filename: str) -> str:
    """Load OpenAPI specification from file."""
    try:
        file_path = Path(filename)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filename}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        logger.info(f"Loaded OpenAPI spec from {filename}: {len(content)} characters")
        return content
    except Exception as e:
        logger.error(f"Failed to load OpenAPI spec from {filename}: {e}")
        raise


async def _fetch_openapi_spec_from_url(url: str) -> str:
    """Fetch OpenAPI specification from URL."""
    try:
        logger.info(f"Fetching OpenAPI spec from URL: {url}")

        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()

            content = response.text
            logger.info(f"Fetched OpenAPI spec from URL: {len(content)} characters")
            return content
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP {e.response.status_code} error fetching URL {url}: {e.response.reason_phrase}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    except httpx.ConnectError as e:
        error_msg = f"Connection error fetching URL {url}: {e}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    except Exception as e:
        error_msg = f"Failed to fetch OpenAPI spec from URL {url}: {e}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)


async def _load_specification(args: argparse.Namespace, filename: Optional[str]) -> str:
    """Load OpenAPI specification from file or URL."""
    if hasattr(args, "url") and args.url:
        return await _fetch_openapi_spec_from_url(args.url)
    elif filename and (
        filename.startswith("http://") or filename.startswith("https://")
    ):
        # Handle case where URL was passed as filename argument
        logger.info(f"Detected URL in filename argument: {filename}")
        return await _fetch_openapi_spec_from_url(filename)
    else:
        return _load_openapi_spec(filename)


if __name__ == "__main__":
    import sys
    import tempfile
    import asyncio

    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0

    # Test 1: Load OpenAPI spec from file
    total_tests += 1
    try:
        # Create a temporary test file
        test_content = """
openapi: 3.0.3
info:
  title: Test API
  version: 1.0.0
paths:
  /test:
    get:
      summary: Test endpoint
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(test_content)
            temp_file = f.name

        result = _load_openapi_spec(temp_file)
        if "Test API" not in result:
            all_validation_failures.append(
                "File loading: Test API not found in loaded content"
            )

        # Cleanup
        Path(temp_file).unlink()

    except Exception as e:
        all_validation_failures.append(f"File loading error: {e}")

    # Test 2: Load non-existent file (should raise exception)
    total_tests += 1
    try:
        _load_openapi_spec("non_existent_file.yaml")
        all_validation_failures.append(
            "Non-existent file: Expected FileNotFoundError but no exception was raised"
        )
    except FileNotFoundError:
        # This is expected - test passes
        pass
    except Exception as e:
        all_validation_failures.append(
            f"Non-existent file: Expected FileNotFoundError but got {type(e).__name__}"
        )

    # Test 3: Mock args for load_specification
    total_tests += 1
    try:
        # Create mock args
        class MockArgs:
            def __init__(self, url=None):
                self.url = url

        # Test file loading path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(test_content)
            temp_file = f.name

        args = MockArgs()
        result = asyncio.run(_load_specification(args, temp_file))
        if "Test API" not in result:
            all_validation_failures.append(
                "Load specification: Test API not found in loaded content"
            )

        # Cleanup
        Path(temp_file).unlink()

    except Exception as e:
        all_validation_failures.append(f"Load specification error: {e}")

    # Final validation result
    if all_validation_failures:
        print(
            f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:"
        )
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(
            f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results"
        )
        print("Spec loader functions are validated and ready for use")
        sys.exit(0)
