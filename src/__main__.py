"""
Main entry point for the openapi_to_mcp package.
Allows running the CLI with: python -m openapi_to_mcp
"""

import asyncio
from .cli import main_cli

if __name__ == "__main__":
    success = asyncio.run(main_cli())
    exit(0 if success else 1) 