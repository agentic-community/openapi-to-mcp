"""
MCP Server Code Generator
Generates FastMCP server code from OpenAPI specifications using LLM prompts.
"""

import re
import ast
import logging
from pathlib import Path
from datetime import datetime
from .llm_client import get_llm_client, LLMRequest
from typing import Dict, Any, List, Optional, Tuple


# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    # Define log message format
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


def _extract_function_signature(func_node: ast.FunctionDef) -> str:
    """Extract the function signature from an AST function node."""
    args = []

    # Handle regular arguments
    for arg in func_node.args.args:
        if arg.annotation:
            args.append(f"{arg.arg}: {ast.unparse(arg.annotation)}")
        else:
            args.append(arg.arg)

    # Handle keyword-only arguments
    for arg in func_node.args.kwonlyargs:
        if arg.annotation:
            args.append(f"{arg.arg}: {ast.unparse(arg.annotation)}")
        else:
            args.append(arg.arg)

    # Handle return annotation
    return_annotation = ""
    if func_node.returns:
        return_annotation = f" -> {ast.unparse(func_node.returns)}"

    signature = f"def {func_node.name}({', '.join(args)}){return_annotation}"
    return signature


def _extract_docstring(func_node: ast.FunctionDef) -> Optional[str]:
    """Extract the docstring from an AST function node."""
    if (
        func_node.body
        and isinstance(func_node.body[0], ast.Expr)
        and isinstance(func_node.body[0].value, ast.Constant)
        and isinstance(func_node.body[0].value.value, str)
    ):
        return func_node.body[0].value.value
    return None


def _has_mcp_tool_decorator(func_node: ast.FunctionDef) -> bool:
    """Check if a function has the @mcp.tool() decorator."""
    for decorator in func_node.decorator_list:
        # Handle @mcp.tool() case
        if (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and isinstance(decorator.func.value, ast.Name)
            and decorator.func.value.id == "mcp"
            and decorator.func.attr == "tool"
        ):
            return True

        # Handle direct @mcp.tool case (without parentheses)
        if (
            isinstance(decorator, ast.Attribute)
            and isinstance(decorator.value, ast.Name)
            and decorator.value.id == "mcp"
            and decorator.attr == "tool"
        ):
            return True

    return False


def _parse_mcp_tools_from_code(
    server_code: str,
) -> List[Tuple[str, str, Optional[str]]]:
    """
    Parse MCP tools from server code and extract their signatures and docstrings.

    Args:
        server_code: The generated server code

    Returns:
        List of tuples containing (function_name, signature, docstring)
    """
    try:
        # Parse the code into an AST
        tree = ast.parse(server_code)

        mcp_tools = []

        # Walk through all function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if this function has @mcp.tool() decorator
                if _has_mcp_tool_decorator(node):
                    signature = _extract_function_signature(node)
                    docstring = _extract_docstring(node)
                    mcp_tools.append((node.name, signature, docstring))

        logger.info(f"Extracted {len(mcp_tools)} MCP tools from generated code")
        return mcp_tools

    except Exception as e:
        logger.error(f"Failed to parse MCP tools from code: {e}")
        return []


def _generate_tool_spec_content(
    mcp_tools: List[Tuple[str, str, Optional[str]]], api_title: str
) -> str:
    """
    Generate the content for tool_spec.txt file.

    Args:
        mcp_tools: List of (function_name, signature, docstring) tuples
        api_title: The API title

    Returns:
        String content for the tool_spec.txt file
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    content = f"""# MCP Tool Specifications for {api_title}
# Generated on: {timestamp}
# 
# This file contains the function signatures and docstrings for all MCP tools
# that are decorated with @mcp.tool() in the generated server code.
#
# Format:
# ========================================
# TOOL: <function_name>
# SIGNATURE: <function_signature>
# DOCSTRING:
# <docstring_content>
# ========================================

"""

    if not mcp_tools:
        content += "No MCP tools found in the generated server code.\n"
        return content

    for function_name, signature, docstring in mcp_tools:
        content += "=" * 80 + "\n"
        content += f"TOOL: {function_name}\n"
        content += f"SIGNATURE: {signature}\n"
        content += "DOCSTRING:\n"
        if docstring:
            # Indent the docstring for better readability
            indented_docstring = "\n".join(
                f"  {line}" if line.strip() else "" for line in docstring.split("\n")
            )
            content += indented_docstring + "\n"
        else:
            content += "  (No docstring provided)\n"
        content += "=" * 80 + "\n\n"

    content += f"\nTotal MCP Tools: {len(mcp_tools)}\n"
    return content


class MCPServerGenerator:
    """Generates MCP server code from OpenAPI specifications."""

    def __init__(
        self,
        model: str = None,
        max_tokens: int = None,
        temperature: float = None,
        timeout_seconds: int = None,
        debug: bool = None,
    ):
        """Initialize the MCP server generator.

        Args:
            model: Model string (format: provider/model_name)
            max_tokens: Maximum tokens for responses
            temperature: Temperature for responses
            timeout_seconds: Timeout for requests
            debug: Whether to enable debug mode
        """
        self.llm_client = get_llm_client(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            debug=debug,
        )

        # Store parameters as class member variables (use llm_client values as source of truth)
        self.model = self.llm_client.model
        self.max_tokens = self.llm_client.max_tokens
        self.temperature = self.llm_client.temperature
        self.timeout_seconds = self.llm_client.timeout_seconds
        self.debug = self.llm_client.debug

        logger.info("Initialized MCP server generator with unified LLM client")
        logger.info(
            f"MCP Generator parameters - max_tokens: {self.max_tokens}, temperature: {self.temperature}"
        )

    async def generate_mcp_server(
        self, openapi_spec: Dict[str, Any], output_dir: Path
    ) -> Dict[str, Any]:
        """
        Generate MCP server code from OpenAPI specification.

        Args:
            openapi_spec: The parsed OpenAPI specification
            output_dir: Directory to save the generated server code
        """
        try:
            logger.info("🔧 MCPServerGenerator.generate_mcp_server called")
            logger.info(f"   Target output directory: {output_dir}")
            logger.info(
                f"   OpenAPI spec keys: {list(openapi_spec.keys()) if openapi_spec else 'None'}"
            )

            # Create a minimal evaluation result for compatibility
            logger.info("📝 Creating minimal evaluation result for compatibility...")
            evaluation_result = {
                "overall": {
                    "completeness_score": 4,
                    "ai_readiness_score": 4,
                    "overall_quality": "good",
                }
            }
            logger.info("✅ Evaluation result created")

            # Call the existing generate_server_code method
            logger.info("🚀 Calling generate_server_code method...")
            try:
                generated_files, mcp_usage = await self.generate_server_code(
                    evaluation_result, openapi_spec, output_dir
                )
                logger.info("✅ generate_server_code completed successfully")
                logger.info(
                    f"   Generated files: {list(generated_files.keys()) if generated_files else 'None'}"
                )
                logger.info(
                    f"   MCP Usage Summary - Total tokens: {mcp_usage.get('total_tokens', 0)}, Cost: ${mcp_usage.get('total_cost_usd', 0.0):.6f}"
                )
            except Exception as e:
                logger.error(f"❌ generate_server_code failed: {e}")
                raise

            logger.info(
                f"🎉 Successfully generated MCP server with {len(generated_files)} files"
            )
            return mcp_usage

        except Exception as e:
            logger.error(f"💥 MCPServerGenerator.generate_mcp_server failed: {e}")
            logger.error(f"   Exception type: {type(e).__name__}")
            import traceback

            logger.error(f"   Full traceback: {traceback.format_exc()}")
            raise

    async def generate_server_code(
        self,
        evaluation_result: Dict[str, Any],
        openapi_spec: Dict[str, Any],
        output_dir: Path,
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Generate MCP server code from evaluation results and OpenAPI specification using LLM.

        Args:
            evaluation_result: The evaluation results from the OpenAPI enhancer
            openapi_spec: The parsed OpenAPI specification
            output_dir: Directory to save the generated server code

        Returns:
            Dict containing paths to generated files
        """
        try:
            logger.info("🔄 Starting LLM-based MCP server code generation")

            # Initialize usage tracking variables
            server_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "calls_count": 0,
            }
            client_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "calls_count": 0,
            }

            # Extract API information for metadata
            logger.info("📋 Extracting API metadata...")
            api_info = openapi_spec.get("info", {})
            api_title = api_info.get("title", "Unknown API")
            api_description = api_info.get("description", "API Server")
            api_version = api_info.get("version", "1.0.0")
            module_name = self._clean_name(api_title)

            logger.info(f"   API Title: {api_title}")
            logger.info(f"   API Description: {api_description[:100]}...")
            logger.info(f"   API Version: {api_version}")
            logger.info(f"   Module Name: {module_name}")

            # Generate main server file using LLM
            logger.info("🤖 Generating server.py using LLM prompt...")
            try:
                server_code, server_usage = await self._generate_server_file_with_llm(
                    openapi_spec
                )
                logger.info(f"✅ Server code generated: {len(server_code)} characters")
            except Exception as e:
                logger.error(f"❌ Failed to generate server code: {e}")
                raise

            # Parse tools from generated server code
            logger.info("🔍 Parsing tools from generated server code...")
            try:
                mcp_tools = _parse_mcp_tools_from_code(server_code)
                logger.info(f"✅ Parsed {len(mcp_tools)} MCP tools from server code")
                for tool_name, _, _ in mcp_tools:
                    logger.info(f"   - {tool_name}")
            except Exception as e:
                logger.error(f"❌ Failed to parse tools from server code: {e}")
                raise

            # Create output directory
            logger.info(f"📁 Creating output directory: {output_dir}")
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.info("✅ Output directory created/verified")
            except Exception as e:
                logger.error(f"❌ Failed to create output directory: {e}")
                raise

            # Write server.py
            logger.info("💾 Writing server.py file...")
            try:
                server_path = output_dir / "server.py"
                with open(server_path, "w", encoding="utf-8") as f:
                    f.write(server_code)
                logger.info(f"✅ Server code written to: {server_path}")
            except Exception as e:
                logger.error(f"❌ Failed to write server.py: {e}")
                raise

            # Generate and write client.py
            logger.info("🤖 Generating client.py using LLM prompt...")
            try:
                client_code, client_usage = await self._generate_client_file_with_llm(
                    mcp_tools, api_title
                )
                logger.info(f"✅ Client code generated: {len(client_code)} characters")

                client_path = output_dir / "client.py"
                with open(client_path, "w", encoding="utf-8") as f:
                    f.write(client_code)
                logger.info(f"✅ Client code written to: {client_path}")
            except Exception as e:
                logger.error(f"❌ Failed to generate/write client.py: {e}")
                raise

            # Generate tool specifications file
            logger.info("📄 Generating tool specifications file...")
            try:
                tool_spec_content = _generate_tool_spec_content(mcp_tools, api_title)
                tool_spec_path = output_dir / "tool_spec.txt"
                with open(tool_spec_path, "w", encoding="utf-8") as f:
                    f.write(tool_spec_content)
                logger.info(f"✅ Tool specifications written to: {tool_spec_path}")
            except Exception as e:
                logger.error(f"❌ Failed to generate/write tool_spec.txt: {e}")
                raise

            # Generate requirements.txt
            logger.info("📦 Generating requirements.txt file...")
            try:
                requirements_content = self._generate_requirements_file()
                requirements_path = output_dir / "requirements.txt"
                with open(requirements_path, "w", encoding="utf-8") as f:
                    f.write(requirements_content)
                logger.info(f"✅ Requirements file written to: {requirements_path}")
            except Exception as e:
                logger.error(f"❌ Failed to generate/write requirements.txt: {e}")
                raise

            # Generate README.md
            logger.info("📖 Generating README.md file...")
            try:
                readme_content = self._generate_readme_file(
                    api_title, api_description, module_name
                )
                readme_path = output_dir / "README.md"
                with open(readme_path, "w", encoding="utf-8") as f:
                    f.write(readme_content)
                logger.info(f"✅ README file written to: {readme_path}")
            except Exception as e:
                logger.error(f"❌ Failed to generate/write README.md: {e}")
                raise

            # Create final file mapping
            generated_files = {
                "server.py": str(server_path),
                "client.py": str(client_path),
                "tool_spec.txt": str(tool_spec_path),
                "requirements.txt": str(requirements_path),
                "README.md": str(readme_path),
            }

            # Calculate total usage across server and client generation
            total_mcp_usage = {
                "server_usage": server_usage,
                "client_usage": client_usage,
                "total_tokens": server_usage.get("total_tokens", 0)
                + client_usage.get("total_tokens", 0),
                "total_cost_usd": server_usage.get("total_cost_usd", 0.0)
                + client_usage.get("total_cost_usd", 0.0),
                "calls_count": server_usage.get("calls_count", 0)
                + client_usage.get("calls_count", 0),
            }

            logger.info(
                "🎉 MCP server and client code generation completed successfully"
            )
            logger.info(f"   Generated {len(generated_files)} files in {output_dir}")
            for file_type, file_path in generated_files.items():
                logger.info(f"   - {file_type}: {file_path}")
            logger.info(
                f"   Total MCP Generation - Tokens: {total_mcp_usage['total_tokens']}, Cost: ${total_mcp_usage['total_cost_usd']:.6f}"
            )
            logger.info(f"MCP generator total_mcp_usage: {total_mcp_usage}")

            return generated_files, total_mcp_usage

        except Exception as e:
            logger.error(f"💥 Failed to generate MCP server code: {e}")
            logger.error(f"   Exception type: {type(e).__name__}")
            import traceback

            logger.error(f"   Full traceback: {traceback.format_exc()}")
            raise

    def _clean_name(self, name: str) -> str:
        """Clean a name to be a valid Python identifier."""
        # Remove special characters and replace with underscores
        cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        # Remove multiple underscores
        cleaned = re.sub(r"_+", "_", cleaned)
        # Remove leading/trailing underscores
        cleaned = cleaned.strip("_")
        # Ensure it doesn't start with a number
        if cleaned and cleaned[0].isdigit():
            cleaned = f"api_{cleaned}"
        return cleaned.lower() or "api_server"

    async def _generate_server_file_with_llm(
        self, openapi_spec: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate the main server.py file using LLM with the prompt template."""
        server_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "calls_count": 0,
        }

        try:
            # Create the prompt using the template
            prompt = self._create_mcp_generation_prompt(openapi_spec)

            logger.info("Sending request to LLM for MCP server generation")
            logger.info(f"Prompt length: {len(prompt)} characters")

            # Create LLM request
            llm_request = LLMRequest(
                prompt=prompt, max_tokens=self.max_tokens, temperature=self.temperature
            )

            # Call LLM API using unified client
            response = await self.llm_client.generate_text(llm_request)

            raw_response = response.text
            logger.info(f"Received response from LLM: {len(raw_response)} characters")

            # Debug: Log first 500 characters of LLM response
            logger.info(f"LLM response preview: {raw_response[:500]}...")

            # Debug: Check for key patterns in the response
            key_patterns = [
                "from mcp.server.fastmcp import FastMCP",
                "from fastmcp import FastMCP",
                "FastMCP(",
                "@mcp.tool()",
                "def main():",
            ]

            for pattern in key_patterns:
                if pattern in raw_response:
                    logger.info(f"✓ Found pattern: {pattern}")
                else:
                    logger.info(f"✗ Missing pattern: {pattern}")

            # Track and log usage information for MCP server generation
            if response.usage:
                server_usage.update(
                    {
                        "prompt_tokens": response.usage.get("prompt_tokens", 0),
                        "completion_tokens": response.usage.get("completion_tokens", 0),
                        "total_tokens": response.usage.get("total_tokens", 0),
                        "total_cost_usd": response.usage.get("total_cost_usd", 0.0),
                        "calls_count": 1,
                    }
                )
                logger.info(
                    f"🔧 MCP Server Generation - Prompt: {server_usage['prompt_tokens']}, Completion: {server_usage['completion_tokens']}, Total: {server_usage['total_tokens']} tokens"
                )
                if server_usage["total_cost_usd"] > 0:
                    logger.info(
                        f"🔧 MCP Server Generation - Cost: ${server_usage['total_cost_usd']:.6f}"
                    )
            else:
                logger.info(
                    "🔧 MCP Server Generation - Usage information not available"
                )

            # Extract Python code from response
            generated_code = self._extract_python_code(raw_response)
            logger.info(f"Extracted code length: {len(generated_code)} characters")

            # Debug: Log first 500 characters of extracted code
            logger.info(f"Extracted code preview: {generated_code[:500]}...")

            # Validate the generated code has basic structure
            if not self._validate_generated_code(generated_code):
                logger.error(
                    f"Validation failed. Full extracted code: {generated_code}"
                )
                raise ValueError("Generated code failed basic validation")

            logger.info(
                "Successfully generated and validated MCP server code using LLM"
            )
            return generated_code, server_usage

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fallback to template-based generation
            logger.warning("Falling back to template-based generation")
            return self._generate_fallback_server(openapi_spec), server_usage

    def _extract_python_code(self, response_text: str) -> str:
        """Extract Python code from LLM response, handling various markdown formats."""
        # Try to find Python code blocks
        if "```python" in response_text:
            start = response_text.find("```python") + 9
            end = response_text.find("```", start)
            if end != -1:
                return response_text[start:end].strip()

        # Try to find generic code blocks
        if "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            if end != -1:
                code = response_text[start:end].strip()
                # Skip if it looks like a language identifier line
                if "\n" in code and not code.startswith(
                    ("python", "py", "bash", "shell")
                ):
                    return code

        # If no code blocks found, return the whole response (assuming it's all code)
        return response_text.strip()

    def _validate_generated_code(self, code: str) -> bool:
        """Basic validation of generated MCP server code."""
        required_patterns = [
            "from mcp.server.fastmcp import FastMCP",
            "FastMCP(",
            "@mcp.tool()",
            "def main():",
            'if __name__ == "__main__":',
        ]

        for pattern in required_patterns:
            if pattern not in code:
                logger.warning(f"Generated code missing required pattern: {pattern}")
                return False

        return True

    def _generate_fallback_server(self, openapi_spec: Dict[str, Any]) -> str:
        """Generate a basic fallback server when LLM generation fails."""
        api_info = openapi_spec.get("info", {})
        api_title = api_info.get("title", "Unknown API")

        return f'''"""
Generated MCP Server for {api_title}
Fallback implementation when LLM generation fails.
"""

import os
import json
import logging
import argparse
import requests
from typing import Annotated, Optional, Dict, Any
from fastmcp import FastMCP
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{{%(filename)s:%(lineno)d}},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="{api_title} MCP Server")
    parser.add_argument("--port", type=str, default=os.environ.get("MCP_SERVER_LISTEN_PORT", "9000"))
    parser.add_argument("--transport", type=str, default=os.environ.get("MCP_TRANSPORT", "sse"))
    parser.add_argument("--base-url", type=str, default=os.environ.get("API_BASE_URL", ""))
    return parser.parse_args()

args = parse_arguments()
mcp = FastMCP("{api_title}", host="0.0.0.0", port=int(args.port))

@mcp.tool()
def api_info() -> Dict[str, Any]:
    """Get API information."""
    return {{
        "title": "{api_title}",
        "message": "This is a fallback MCP server. LLM generation failed.",
        "endpoints": {len(openapi_spec.get('paths', {}))}
    }}

@mcp.prompt()
def system_prompt_for_agent() -> str:
    """System prompt for AI agents."""
    return "You are using a fallback MCP server for {api_title}."

@mcp.resource("config://app")
def get_config() -> str:
    """Configuration data"""
    return json.dumps({{"api_title": "{api_title}", "status": "fallback"}})

def main():
    logger.info("Starting {api_title} MCP Server (fallback mode)")
    mcp.run(transport=args.transport)

if __name__ == "__main__":
    main()
'''

    def _create_mcp_generation_prompt(self, openapi_spec: Dict[str, Any]) -> str:
        """Create the prompt for LLM to generate MCP server code using the template."""
        try:
            # Convert OpenAPI spec to YAML format for better readability
            import yaml

            openapi_yaml = yaml.dump(openapi_spec, default_flow_style=False, indent=2)

            # Load the prompt template
            template_path = (
                Path(__file__).parent.parent.parent
                / "templates"
                / "mcp_server_create.txt"
            )
            logger.info(f"Loading prompt template from: {template_path}")

            with open(template_path, "r", encoding="utf-8") as f:
                template = f.read()

            # Replace the placeholder with the actual OpenAPI spec
            prompt = template.replace("{openapi_spec}", openapi_yaml)

            logger.info(
                f"Created prompt with OpenAPI spec ({len(openapi_yaml)} chars YAML)"
            )
            return prompt

        except Exception as e:
            logger.error(f"Failed to load prompt template: {e}")
            # Return a basic fallback prompt
            import yaml

            openapi_yaml = yaml.dump(openapi_spec, default_flow_style=False)
            return f"""You are an expert software developer who writes MCP servers using FastMCP.

Generate a complete MCP server for the following OpenAPI specification.
Each API endpoint should become a separate @mcp.tool() function.

OpenAPI Specification:
{openapi_yaml}

Generate clean Python code with proper imports, logging, and error handling."""

    def _generate_requirements_file(self) -> str:
        """Generate requirements.txt file for the MCP server and client."""
        return """# MCP Server and Client Requirements
fastmcp>=2.9.0
requests>=2.31.0
pydantic>=2.0.0
PyYAML>=6.0.0
httpx>=0.24.0
"""

    def _generate_readme_file(
        self, api_title: str, api_description: str, module_name: str
    ) -> str:
        """Generate README.md file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return f"""# {api_title} MCP Server & Client

{api_description}

This MCP server and client were auto-generated from an OpenAPI specification on {timestamp}.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running the Server

```bash
python server.py --port 9000 --transport sse
```

### Running the Client

```bash
python client.py --server-url http://localhost:9000/sse
```

The client will automatically connect to the server, list all available tools, and test each one with sample parameters.

### Environment Variables

**Server:**
- `MCP_SERVER_LISTEN_PORT`: Port for the server to listen on (default: 9000)
- `MCP_TRANSPORT`: Transport type (default: sse)
- `API_BASE_URL`: Base URL for the API

**Client:**
- `MCP_SERVER_URL`: URL of the MCP server SSE endpoint (default: http://localhost:9000/sse)

### Command Line Arguments

**Server:**
- `--port`: Port for the MCP server to listen on (default: 9000)
- `--transport`: Transport type for the MCP server (default: sse)
- `--base-url`: Base URL for the API

**Client:**
- `--server-url`: URL of the MCP server SSE endpoint
- `--transport`: Transport method (sse or stdio, default: sse)
- `--output-file`: File to save test results (default: client_results.json)

## Features

- Auto-generated tools for each API endpoint
- Proper error handling and logging
- FastMCP integration for easy deployment
- Type-safe parameter handling with Pydantic
- Comprehensive client for testing all server functionality
- SSE transport support by default

## Generated Tools

This server provides MCP tools for each API endpoint defined in the OpenAPI specification.
Each tool corresponds to an API operation and includes proper parameter validation and documentation.

See `tool_spec.txt` for detailed specifications of all available MCP tools, including their
function signatures and docstrings.

## Generated Files

- `server.py`: MCP server implementation
- `client.py`: MCP client for testing the server
- `tool_spec.txt`: Detailed tool specifications
- `requirements.txt`: Python dependencies
- `README.md`: This documentation

## Testing the Implementation

1. Start the server in one terminal:
   ```bash
   python server.py
   ```

2. Run the client in another terminal:
   ```bash
   python client.py
   ```

The client will automatically test all available tools and save results to `client_results.json`.

## Configuration

The server and client can be configured through environment variables or command line arguments.
See the usage section above for available options.

---

*Auto-generated by OpenAPI to MCP Converter*
"""

    async def _generate_client_file_with_llm(
        self, mcp_tools: List[Tuple[str, str, Optional[str]]], api_title: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate the client.py file using LLM with the client template."""
        client_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "calls_count": 0,
        }

        try:
            # Create tool specifications content for the prompt
            tool_specifications = self._format_tool_specifications_for_prompt(
                mcp_tools, api_title
            )

            # Create the prompt using the client template
            prompt = self._create_mcp_client_generation_prompt(tool_specifications)

            logger.info("Sending request to LLM for MCP client generation")
            logger.info(f"Prompt length: {len(prompt)} characters")

            # Create LLM request
            llm_request = LLMRequest(
                prompt=prompt, max_tokens=self.max_tokens, temperature=self.temperature
            )

            # Call LLM API using unified client
            response = await self.llm_client.generate_text(llm_request)

            raw_response = response.text
            logger.info(f"Received response from LLM: {len(raw_response)} characters")

            # Track and log usage information for MCP client generation
            if response.usage:
                client_usage.update(
                    {
                        "prompt_tokens": response.usage.get("prompt_tokens", 0),
                        "completion_tokens": response.usage.get("completion_tokens", 0),
                        "total_tokens": response.usage.get("total_tokens", 0),
                        "total_cost_usd": response.usage.get("total_cost_usd", 0.0),
                        "calls_count": 1,
                    }
                )
                logger.info(
                    f"👥 MCP Client Generation - Prompt: {client_usage['prompt_tokens']}, Completion: {client_usage['completion_tokens']}, Total: {client_usage['total_tokens']} tokens"
                )
                if client_usage["total_cost_usd"] > 0:
                    logger.info(
                        f"👥 MCP Client Generation - Cost: ${client_usage['total_cost_usd']:.6f}"
                    )
            else:
                logger.info(
                    "👥 MCP Client Generation - Usage information not available"
                )

            # Extract Python code from response
            generated_code = self._extract_python_code(raw_response)

            # Validate the generated client code has basic structure
            if not self._validate_generated_client_code(generated_code):
                raise ValueError("Generated client code failed basic validation")

            logger.info(
                "Successfully generated and validated MCP client code using LLM"
            )
            return generated_code, client_usage

        except Exception as e:
            logger.error(f"LLM client generation failed: {e}")
            # Fallback to basic client generation
            logger.warning("Falling back to basic client generation")
            return self._generate_fallback_client(api_title), client_usage

    def _format_tool_specifications_for_prompt(
        self, mcp_tools: List[Tuple[str, str, Optional[str]]], api_title: str
    ) -> str:
        """Format tool specifications for use in the client generation prompt."""
        if not mcp_tools:
            return f"No MCP tools found for {api_title}."

        tool_specs = []
        for function_name, signature, docstring in mcp_tools:
            spec = f"Tool: {function_name}\n"
            spec += f"Signature: {signature}\n"
            if docstring:
                spec += f"Description: {docstring}\n"
            else:
                spec += "Description: No description available\n"
            tool_specs.append(spec)

        return "\n".join(tool_specs)

    def _create_mcp_client_generation_prompt(self, tool_specifications: str) -> str:
        """Create the prompt for LLM to generate MCP client code using the template."""
        try:
            # Load the client prompt template
            template_path = (
                Path(__file__).parent.parent.parent
                / "templates"
                / "mcp_client_create.txt"
            )
            logger.info(f"Loading client prompt template from: {template_path}")

            with open(template_path, "r", encoding="utf-8") as f:
                template = f.read()

            # Replace the placeholder with the actual tool specifications
            prompt = template.replace("{tool_specifications}", tool_specifications)

            logger.info(
                f"Created client prompt with tool specifications ({len(tool_specifications)} chars)"
            )
            return prompt

        except Exception as e:
            logger.error(f"Failed to load client prompt template: {e}")
            # Return a basic fallback prompt
            return f"""You are an expert software developer who writes MCP clients.

Generate a complete MCP client that connects to an MCP server and tests all available tools.

Tool Specifications:
{tool_specifications}

Generate clean Python code with proper imports, logging, and error handling."""

    def _validate_generated_client_code(self, code: str) -> bool:
        """Basic validation of generated MCP client code."""
        required_patterns = [
            "import asyncio",
            "import logging",
            "from mcp import ClientSession",
            "from mcp.client.streamable_http import streamablehttp_client",
            "async def main():",
            'if __name__ == "__main__":',
        ]

        for pattern in required_patterns:
            if pattern not in code:
                logger.warning(
                    f"Generated client code missing required pattern: {pattern}"
                )
                return False

        return True

    def _generate_fallback_client(self, api_title: str) -> str:
        """Generate a basic fallback client when LLM generation fails."""
        return f'''"""
Generated MCP Client for {api_title}
Fallback implementation when LLM generation fails.
"""

import os
import json
import asyncio
import logging
import argparse
from typing import Any, Dict, List, Optional
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{{%(filename)s:%(lineno)d}},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


def _parse_arguments():
    """Parse command line arguments for MCP server connection."""
    parser = argparse.ArgumentParser(description="MCP Client for {api_title}")
    
    parser.add_argument(
        "--server-url",
        type=str,
        default=os.environ.get("MCP_SERVER_URL", "http://localhost:8000/api"),
        help="URL of the MCP server endpoint (default: http://localhost:8000/api)",
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default="client_results.json",
        help="File to save client interaction results (default: client_results.json)",
    )
    
    return parser.parse_args()


async def main():
    """Main function to run the MCP client."""
    args = _parse_arguments()
    
    try:
        logger.info("Starting MCP client test with streamable-http transport")
        print(f"Connecting to: {{args.server_url}}")
        
        async with streamablehttp_client(url=args.server_url) as (read, write, get_session_id):
            print("StreamableHTTP connection established")
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("Session initialized successfully")
                
                # List available tools
                tools_response = await session.list_tools()
                tools = tools_response.tools
                
                logger.info(f"Found {{len(tools)}} available tools")
                print(f"Available tools: {{len(tools)}}")
                
                for tool in tools:
                    name = tool.name
                    description = getattr(tool, 'description', 'No description')
                    logger.info(f"Tool: {{name}} - {{description}}")
                    print(f"  - {{name}}: {{description}}")
                
                print("Fallback client - basic tool listing completed")
        
    except Exception as e:
        logger.error(f"Client execution failed: {{e}}")
        print(f"Error: {{e}}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
'''


def should_generate_mcp_server(evaluation_result: Dict[str, Any]) -> bool:
    """
    Determine if MCP server code should be generated based on evaluation scores.

    Args:
        evaluation_result: The evaluation results

    Returns:
        bool: True if scores are sufficient for code generation
    """
    try:
        evaluation = evaluation_result.get("evaluation", {})
        overall = evaluation.get("overall", {})

        completeness_score = overall.get("completeness_score", 0)
        ai_readiness_score = overall.get("ai_readiness_score", 0)

        return completeness_score >= 3 and ai_readiness_score >= 3

    except Exception as e:
        logger.error(f"Error checking evaluation scores: {e}")
        return False
