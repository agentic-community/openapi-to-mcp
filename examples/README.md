# Testing with Sample API

This README describes how to test the OpenAPI to MCP converter using the provided `sample_api.yaml` and a stub server that provides a backend for the API implementation. You can test with the generated MCP server and client code.

## Prerequisites

We assume you have followed the steps in the main project's README.md quickstart section and have:

- Installed the project dependencies
- Configured your API credentials
- Successfully generated MCP server code for the sample API

## Testing Commands

Run these commands in separate terminals:

### Terminal 1: Start the Stub Server
```bash
uv run python examples/stub_server.py
```

### Terminal 2: Set Auth Token and Run MCP Server
```bash
export MISSION_AUTH_TOKEN=secret-token
uv run python results/anthropic/sample_api/mcpserver/server.py
```

### Terminal 3: Run MCP Client
```bash
uv run python results/anthropic/sample_api/mcpserver/client.py
```

## What This Tests

- The stub server provides mock backend responses for the sample API
- The MCP server connects to the stub server and exposes API endpoints as MCP tools
- The MCP client tests all available tools and validates the integration

## Expected Results

You should see:
1. Stub server starts and serves mock API responses
2. MCP server starts and registers tools based on the sample API
3. MCP client connects, lists available tools, and tests each one

This validates the complete flow from OpenAPI specification to working MCP server integration.