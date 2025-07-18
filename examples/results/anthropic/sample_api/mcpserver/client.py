"""
MCP Client for testing quantum gravity research vessel tools.
This client connects to an MCP server via streamable-http transport and demonstrates 
how to interact with all available quantum gravity experiment tools.
"""

import os
import json
import asyncio
import logging
import argparse
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime, timedelta
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


def _parse_arguments():
    """Parse command line arguments for MCP server connection."""
    parser = argparse.ArgumentParser(description="MCP Client for Testing Quantum Gravity Research Tools")
    
    parser.add_argument(
        "--server-url",
        type=str,
        default=os.environ.get("MCP_SERVER_URL", "http://localhost:8000/api"),
        help="URL of the MCP server endpoint (default: http://localhost:8000/api)",
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default="quantum_gravity_results.json",
        help="File to save client interaction results (default: quantum_gravity_results.json)",
    )
    
    return parser.parse_args()


async def _list_available_tools(session: ClientSession) -> List[Dict[str, Any]]:
    """
    List all available tools from the MCP server.
    
    Args:
        session: Connected MCP client session
        
    Returns:
        List of tool definitions
        
    Raises:
        Exception: If listing tools fails
    """
    try:
        logger.info("Requesting list of available tools")
        tools_response = await session.list_tools()
        
        tools = tools_response.tools
        logger.info(f"Found {len(tools)} available tools")
        
        # Convert tools to dictionaries for easier processing
        result_tools = []
        for tool in tools:
            tool_dict = {
                'name': tool.name,
                'description': tool.description,
                'inputSchema': tool.inputSchema
            }
            result_tools.append(tool_dict)
            logger.info(f"Tool: {tool.name} - {tool.description}")
        
        return result_tools
        
    except Exception as e:
        logger.error(f"Failed to list tools: {e}")
        raise


def _generate_sample_parameters(tool_name: str, tool_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate sample parameters for a specific tool based on its input schema.
    
    Args:
        tool_name: Name of the tool
        tool_schema: Tool's input schema definition
        
    Returns:
        Dictionary of sample parameters
    """
    if not tool_schema or "properties" not in tool_schema:
        return {}
    
    sample_params = {}
    properties = tool_schema["properties"]
    required_fields = tool_schema.get("required", [])
    
    # Generate tool-specific parameters based on the quantum gravity research context
    if tool_name == "get_spaceship_telemetry":
        sample_params = {
            "timeframe": "realtime",
            "reference_frame": "schwarzschild"
        }
    elif tool_name == "get_black_hole_proximity":
        # This tool has no parameters
        sample_params = {}
    elif tool_name == "get_quantum_gravity_data":
        sample_params = {
            "experiment_type": "vacuum_fluctuations",
            "start_time": (datetime.now() - timedelta(hours=1)).isoformat(),
            "end_time": datetime.now().isoformat()
        }
    elif tool_name == "control_experiment":
        sample_params = {
            "experiment_type": "hawking_radiation",
            "duration_seconds": 3600,
            "measurement_frequency": 1.0,
            "coupling_constant": 0.5,
            "energy_scale": 1.602e-19
        }
    elif tool_name == "get_experiment_results":
        sample_params = {
            "experiment_id": "550e8400-e29b-41d4-a716-446655440000",
            "format": "json",
            "analysis_level": "processed"
        }
    else:
        # Fallback: generate parameters based on schema
        for param_name, param_def in properties.items():
            param_type = param_def.get("type", "string")
            default_value = param_def.get("default")
            
            if default_value is not None:
                sample_params[param_name] = default_value
            elif param_type == "string":
                sample_params[param_name] = param_def.get("example", "sample_string")
            elif param_type == "integer":
                sample_params[param_name] = param_def.get("example", 42)
            elif param_type == "number":
                sample_params[param_name] = param_def.get("example", 3.14)
            elif param_type == "boolean":
                sample_params[param_name] = param_def.get("example", True)
            elif param_type == "array":
                sample_params[param_name] = param_def.get("example", ["sample_item"])
            elif param_type == "object":
                sample_params[param_name] = param_def.get("example", {})
            else:
                sample_params[param_name] = "sample_value"
    
    # Filter to only include parameters that exist in the schema
    if properties:
        filtered_params = {k: v for k, v in sample_params.items() if k in properties}
        
        # Ensure all required parameters are included
        for required_param in required_fields:
            if required_param not in filtered_params:
                param_def = properties[required_param]
                param_type = param_def.get("type", "string")
                if param_type == "string":
                    filtered_params[required_param] = "required_value"
                elif param_type == "integer":
                    filtered_params[required_param] = 1
                elif param_type == "number":
                    filtered_params[required_param] = 1.0
                elif param_type == "boolean":
                    filtered_params[required_param] = True
                else:
                    filtered_params[required_param] = "required_value"
        
        return filtered_params
    
    return sample_params


async def _call_tool(session: ClientSession, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call a specific tool with the provided parameters.
    
    Args:
        session: Connected MCP client session
        tool_name: Name of the tool to call
        parameters: Parameters to pass to the tool
        
    Returns:
        Tool execution result
        
    Raises:
        Exception: If tool call fails
    """
    try:
        logger.info(f"Calling tool '{tool_name}' with parameters: {json.dumps(parameters, indent=2)}")
        
        result = await session.call_tool(tool_name, arguments=parameters)
        
        # Extract text content from result
        result_content = []
        if result.content:
            for content_item in result.content:
                if hasattr(content_item, 'text'):
                    result_content.append(content_item.text)
                else:
                    result_content.append(str(content_item))
        
        response = {
            "tool_name": tool_name,
            "parameters": parameters,
            "success": True,
            "result": result_content,
            "is_error": getattr(result, 'isError', False)
        }
        
        # Print the response for user visibility
        print(f"\n🔧 Tool: {tool_name}")
        print(f"📝 Parameters: {json.dumps(parameters, indent=2)}")
        print(f"✅ Status: SUCCESS")
        print(f"📋 Response:")
        for content in result_content:
            print(f"   {content}")
        print("-" * 60)
        
        logger.info(f"Tool '{tool_name}' executed successfully")
        logger.info(f"Result: {result_content}")
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to call tool '{tool_name}': {e}")
        
        response = {
            "tool_name": tool_name,
            "parameters": parameters,
            "success": False,
            "error": str(e),
            "result": None
        }
        
        # Print the error response for user visibility
        print(f"\n🔧 Tool: {tool_name}")
        print(f"📝 Parameters: {json.dumps(parameters, indent=2)}")
        print(f"❌ Status: FAILED")
        print(f"🚨 Error: {str(e)}")
        print("-" * 60)
        
        return response


async def _test_all_tools(session: ClientSession, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Test all available tools by calling them with sample parameters.
    
    Args:
        session: Connected MCP client session
        tools: List of tool definitions
        
    Returns:
        List of tool execution results
    """
    results = []
    
    print(f"\n🚀 Starting quantum gravity research vessel tool testing...")
    print(f"📡 Testing {len(tools)} available tools\n")
    
    for tool in tools:
        tool_name = tool["name"]
        tool_description = tool.get("description", "No description")
        input_schema = tool.get("inputSchema", {})
        
        logger.info(f"Testing tool: {tool_name}")
        logger.info(f"Description: {tool_description}")
        
        # Generate sample parameters based on the tool's schema
        sample_params = _generate_sample_parameters(tool_name, input_schema)
        
        # Call the tool
        result = await _call_tool(session, tool_name, sample_params)
        results.append(result)
        
        # Add a small delay between tool calls
        await asyncio.sleep(0.5)
    
    return results


def _save_results(results: List[Dict[str, Any]], output_file: str):
    """
    Save the tool execution results to a JSON file.
    
    Args:
        results: List of tool execution results
        output_file: Path to the output file
    """
    try:
        # Create output file path
        if not os.path.isabs(output_file):
            script_dir = Path(__file__).parent
            output_path = script_dir / output_file
        else:
            output_path = Path(output_file)
        
        # Add metadata to results
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "total_tools": len(results),
            "successful_calls": sum(1 for r in results if r["success"]),
            "failed_calls": sum(1 for r in results if not r["success"]),
            "results": results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
        print(f"\n💾 Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        print(f"❌ Failed to save results: {e}")


async def _test_quantum_gravity_server(server_url: str, output_file: str):
    """
    Test the quantum gravity research MCP server using streamable-http transport.
    
    Args:
        server_url: URL of the MCP server
        output_file: Path to save results
    """
    print(f"\n🌌 Quantum Gravity Research Vessel - MCP Client")
    print(f"📡 Connecting to server: {server_url}")
    
    try:
        async with streamablehttp_client(url=server_url) as (read, write, get_session_id):
            print(f"  ✓ StreamableHTTP connection established")
            
            async with ClientSession(read, write) as session:
                await session.initialize()
                print(f"  ✓ Session initialized successfully")
                
                # List available tools
                print(f"\n🔍 Discovering available tools...")
                tools = await _list_available_tools(session)
                
                print(f"\n📋 Available Tools ({len(tools)} found):")
                for tool in tools:
                    print(f"  • {tool['name']}: {tool['description']}")
                
                # Test each tool
                results = await _test_all_tools(session, tools)
                
                # Save results
                _save_results(results, output_file)
                
                # Print summary
                successful_calls = sum(1 for r in results if r["success"])
                failed_calls = len(results) - successful_calls
                
                print(f"\n📊 Quantum Gravity Research Test Summary:")
                print(f"   🔬 Total experiments: {len(results)}")
                print(f"   ✅ Successful: {successful_calls}")
                print(f"   ❌ Failed: {failed_calls}")
                print(f"   📈 Success rate: {(successful_calls/len(results)*100):.1f}%" if results else "0%")
                
                if failed_calls > 0:
                    print(f"\n⚠️  Failed tools:")
                    for result in results:
                        if not result["success"]:
                            print(f"     • {result['tool_name']}: {result.get('error', 'Unknown error')}")
                
                print(f"\n🎯 Quantum gravity research vessel testing completed!")
                    
    except Exception as e:
        print(f"  ❌ Connection failed: {str(e)}")
        logger.error(f"Server connection failed: {e}")
        raise


async def main():
    """Main function to run the quantum gravity research MCP client."""
    args = _parse_arguments()
    
    try:
        logger.info("Starting quantum gravity research MCP client")
        await _test_quantum_gravity_server(args.server_url, args.output_file)
        
    except Exception as e:
        logger.error(f"Client execution failed: {e}")
        print(f"❌ Client execution failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())