"""
Simple LLM client using LiteLLM for unified provider access.
Supports all LiteLLM-compatible providers through a single interface.
"""

import litellm
import logging
from functools import lru_cache
from .config_loader import config
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    # Define log message format
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Configure LiteLLM with default debug value
# Will be properly set when LLMClient is initialized
litellm.set_verbose = False


class LLMRequest(BaseModel):
    """Request model for LLM API calls."""

    prompt: str = Field(description="The prompt to send to the model")
    max_tokens: int = Field(default=4000, description="Maximum tokens in response")
    temperature: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Response temperature"
    )
    top_p: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter"
    )
    stop_sequences: List[str] = Field(
        default_factory=list, description="Stop sequences for generation"
    )

    model_config = {"str_strip_whitespace": True}


class LLMResponse(BaseModel):
    """Response model from LLM APIs."""

    text: str = Field(description="Generated text response")
    stop_reason: Optional[str] = Field(
        default=None, description="Reason generation stopped"
    )
    usage: Optional[Dict[str, Any]] = Field(
        default=None, description="Token usage information"
    )
    model: str = Field(description="Model that generated the response")

    model_config = {"str_strip_whitespace": True}


class LLMClient:
    """Simple LLM client using LiteLLM for unified provider access."""

    def __init__(
        self,
        model: str = None,
        max_tokens: int = None,
        temperature: float = None,
        timeout_seconds: int = None,
        debug: bool = None,
    ):
        """Initialize the LLM client.

        Args:
            model: Model string (format: provider/model_name)
            max_tokens: Maximum tokens for responses
            temperature: Temperature for responses
            timeout_seconds: Timeout for requests
            debug: Whether to enable debug mode
        """
        try:
            import os

            # Use passed parameters or fall back to config.yml
            self.model = model if model is not None else config.get_str("model")
            self.max_tokens = (
                max_tokens
                if max_tokens is not None
                else config.get_int("max_tokens", 4096)
            )
            self.temperature = (
                temperature
                if temperature is not None
                else config.get_float("temperature", 0.1)
            )
            self.timeout_seconds = (
                timeout_seconds
                if timeout_seconds is not None
                else config.get_int("timeout_seconds", 300)
            )
            self.debug = debug if debug is not None else config.get_bool("debug", False)

            # Let LiteLLM handle all credential validation
            logger.info(f"Initialized LLM client with model: {self.model}")

            # Print all parameters for debugging
            logger.info("=== LLM Client Configuration ===")
            logger.info(f"  Model: {self.model}")
            logger.info(f"  Max Tokens: {self.max_tokens}")
            logger.info(f"  Temperature: {self.temperature}")
            logger.info(f"  Timeout: {self.timeout_seconds} seconds")
            logger.info(f"  Debug Mode: {self.debug}")

            # Print environment variables related to credentials
            logger.info("=== Environment Variables ===")
            logger.info(f"  AWS_REGION: {os.environ.get('AWS_REGION', 'Not set')}")
            logger.info(f"  AWS_PROFILE: {os.environ.get('AWS_PROFILE', 'Not set')}")
            logger.info(
                f"  ANTHROPIC_API_KEY: {'Set' if os.environ.get('ANTHROPIC_API_KEY') else 'Not set'}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup any open sessions."""
        # LiteLLM should handle session cleanup internally
        # but we can add explicit cleanup if needed
        pass

    # Removed _setup_credentials method - letting LiteLLM handle credential validation natively

    async def generate_text(self, request: LLMRequest) -> LLMResponse:
        """Generate text using the configured model via LiteLLM."""
        try:
            logger.info(
                f"Generating text with model: {self.model}, prompt length: {len(request.prompt)}"
            )

            # Print the prompt for debugging
            print("=" * 80)
            print(f"🔵 PROMPT SENT TO {self.model.upper()}:")
            print("=" * 80)
            print(request.prompt)
            print("=" * 80)

            # Prepare parameters for LiteLLM
            params = {
                "model": self.model,
                "messages": [{"role": "user", "content": request.prompt}],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "timeout": self.timeout_seconds,
            }

            if request.stop_sequences:
                params["stop"] = request.stop_sequences

            # Make the API call using LiteLLM
            logger.info(f"Calling LiteLLM completion with parameters: {params}")
            response = await litellm.acompletion(**params)

            # Extract response data
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            # Extract usage and cost information
            usage = None
            if hasattr(response, "usage") and response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens

                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }

                # Calculate cost using LiteLLM's completion_cost function
                try:
                    total_cost = litellm.completion_cost(completion_response=response)
                    if total_cost and total_cost > 0:
                        usage["total_cost_usd"] = total_cost

                        # Calculate cost per token to estimate prompt vs completion costs
                        if total_tokens > 0:
                            cost_per_token = total_cost / total_tokens
                            usage["prompt_cost_usd"] = prompt_tokens * cost_per_token
                            usage["completion_cost_usd"] = (
                                completion_tokens * cost_per_token
                            )
                except Exception as cost_error:
                    logger.debug(f"Could not calculate cost: {cost_error}")
                    # Fallback to direct attributes if available
                    if hasattr(response.usage, "prompt_tokens_cost_usd"):
                        usage["prompt_cost_usd"] = response.usage.prompt_tokens_cost_usd
                    if hasattr(response.usage, "completion_tokens_cost_usd"):
                        usage["completion_cost_usd"] = (
                            response.usage.completion_tokens_cost_usd
                        )
                    if hasattr(response.usage, "total_cost_usd"):
                        usage["total_cost_usd"] = response.usage.total_cost_usd

            # Create response
            llm_response = LLMResponse(
                text=content, stop_reason=finish_reason, usage=usage, model=self.model
            )

            # Print the completion for debugging
            print(f"🟢 COMPLETION FROM {self.model.upper()}:")
            print("=" * 80)
            print(content)
            print("=" * 80)

            # Log usage and cost information
            if usage:
                logger.info("📊 Token Usage:")
                logger.info(f"   Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
                logger.info(
                    f"   Completion tokens: {usage.get('completion_tokens', 'N/A')}"
                )
                logger.info(f"   Total tokens: {usage.get('total_tokens', 'N/A')}")

                if "total_cost_usd" in usage:
                    logger.info("💰 Cost Information:")
                    if "prompt_cost_usd" in usage:
                        logger.info(f"   Prompt cost: ${usage['prompt_cost_usd']:.6f}")
                    if "completion_cost_usd" in usage:
                        logger.info(
                            f"   Completion cost: ${usage['completion_cost_usd']:.6f}"
                        )
                    logger.info(f"   Total cost: ${usage['total_cost_usd']:.6f}")

            logger.info("Successfully generated text response")
            return llm_response

        except Exception as e:
            logger.error(f"Failed to generate text: {e}")
            raise

    def test_connection(self) -> bool:
        """Test the connection with a simple request."""
        try:
            logger.info(f"Testing connection for model: {self.model}")

            # Use synchronous version for testing
            response = litellm.completion(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": "Hello, please respond with 'Connection successful' to test the API.",
                    }
                ],
                max_tokens=50,
                temperature=0.1,
                timeout=30,
            )

            content = response.choices[0].message.content
            logger.info(f"Connection test successful: {content[:100]}")
            return True

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        try:
            # Extract provider from model string using LiteLLM convention
            # LiteLLM uses format: "provider/model_name" or just "model_name"
            provider = "unknown"
            if "/" in self.model:
                provider = self.model.split("/")[0]

            model_info = {
                "model": self.model,
                "provider": provider,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "timeout_seconds": self.timeout_seconds,
            }

            logger.info(f"Retrieved model info: {model_info}")
            return model_info
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            raise


@lru_cache(maxsize=8)
def _create_llm_client_cached(
    model: str, max_tokens: int, temperature: float, timeout_seconds: int, debug: bool
) -> LLMClient:
    """Create a cached LLM client instance with specific parameters.

    This internal function uses lru_cache to cache client instances
    based on their configuration parameters.
    """
    client = LLMClient(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        debug=debug,
    )
    logger.info(f"Created cached LLM client with model: {model}")
    return client


def get_llm_client(
    model: str = None,
    max_tokens: int = None,
    temperature: float = None,
    timeout_seconds: int = None,
    debug: bool = None,
) -> LLMClient:
    """Get an LLM client instance with the specified configuration.

    This function returns a cached client instance if one exists with
    the same parameters, otherwise creates a new one. Parameters default
    to values from config.yml if not provided.

    Args:
        model: Model string (format: provider/model_name)
        max_tokens: Maximum tokens for responses
        temperature: Temperature for responses
        timeout_seconds: Timeout for requests
        debug: Whether to enable debug mode

    Returns:
        Configured LLM client instance
    """
    # Use provided parameters or fall back to config defaults
    actual_model = model if model is not None else config.get_str("model")
    actual_max_tokens = (
        max_tokens if max_tokens is not None else config.get_int("max_tokens", 4096)
    )
    actual_temperature = (
        temperature if temperature is not None else config.get_float("temperature", 0.1)
    )
    actual_timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else config.get_int("timeout_seconds", 300)
    )
    actual_debug = debug if debug is not None else config.get_bool("debug", False)

    # Get cached client or create new one
    return _create_llm_client_cached(
        model=actual_model,
        max_tokens=actual_max_tokens,
        temperature=actual_temperature,
        timeout_seconds=actual_timeout,
        debug=actual_debug,
    )


async def cleanup_llm_client():
    """Clear the LLM client cache.

    This clears all cached client instances. Individual clients
    should handle their own cleanup through context managers.
    """
    logger.info("Clearing LLM client cache...")
    _create_llm_client_cached.cache_clear()
    logger.info("LLM client cache cleared")


if __name__ == "__main__":
    """Standalone testing of LLM client."""
    import asyncio

    async def test_llm_client():
        """Test the LLM client functionality."""
        logger.info("Testing LLM client...")

        try:
            # Test client creation
            client = LLMClient()
            logger.info(f"Created client for model: {client.model}")

            # Test connection
            connection_ok = client.test_connection()
            logger.info(f"Connection test result: {connection_ok}")

            # Get model info
            model_info = client.get_model_info()
            logger.info(f"Model info: {model_info}")

            # Test text generation (only if connection works)
            if connection_ok:
                test_request = LLMRequest(
                    prompt="Evaluate this simple OpenAPI operation: GET /users/{id}. Provide a brief assessment.",
                    max_tokens=200,
                    temperature=0.1,
                )

                response = await client.generate_text(test_request)
                logger.info(
                    f"Generated response from {response.model}: {response.text[:200]}..."
                )

            logger.info("LLM client test completed successfully")

        except Exception as e:
            logger.error(f"LLM client test failed: {e}")
            raise

    # Run the test
    asyncio.run(test_llm_client())
