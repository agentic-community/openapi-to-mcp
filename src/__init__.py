"""
OpenAPI to MCP Converter Package.
Converts OpenAPI specifications to MCP servers with LLM enhancement.
"""

from .cli import main_cli

__version__ = "0.1.0"
__author__ = "OpenAPI to MCP Team"
__description__ = "Convert OpenAPI specifications to MCP servers with LLM enhancement"

# Make CLI main function available at package level
__all__ = ["main_cli"]
