"""
This server provides an interface to the Quantum Gravity Experiment API for collecting and analyzing data from the Event Horizon Research Vessel (EHRV).
"""

import os
import requests
import argparse
import logging
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from typing import Annotated, Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments with defaults matching environment variables."""
    parser = argparse.ArgumentParser(description="Quantum Gravity Experiment MCP Server")

    parser.add_argument(
        "--port",
        type=str,
        default=os.environ.get("MCP_SERVER_LISTEN_PORT", "9001"),
        help="Port for the MCP server to listen on (default: 9001)",
    )

    parser.add_argument(
        "--transport",
        type=str,
        default=os.environ.get("MCP_TRANSPORT", "streamable-http"),
        choices=["sse", "streamable-http"],
        help="Transport type for the MCP server (default: streamable-http)",
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("API_BASE_URL", "http://localhost:8000"),
        help="Base URL for the API",
    )

    return parser.parse_args()


# Parse arguments at module level to make them available
args = parse_arguments()

# Log parsed arguments for debugging
logger.info(f"Parsed arguments - port: {args.port}, transport: {args.transport}")
logger.info(f"Environment variables - MCP_TRANSPORT: {os.environ.get('MCP_TRANSPORT', 'NOT SET')}, MCP_SERVER_LISTEN_PORT: {os.environ.get('MCP_SERVER_LISTEN_PORT', 'NOT SET')}")

# Initialize FastMCP server
mcp = FastMCP("QuantumGravityExperimentServer", host="0.0.0.0", port=int(args.port))
mcp.settings.mount_path = "/api"

def make_api_request(method: str, endpoint: str, params: Dict[str, Any] = None, json_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Make HTTP request to API endpoint with error handling and authentication."""
    url = f"{args.base_url}{endpoint}"
    headers = {
        "Authorization": f"Bearer {os.environ.get('AUTH_TOKEN', '')}"
    }
    
    try:
        response = requests.request(
            method=method,
            url=url,
            params=params,
            json=json_data,
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        raise Exception(f"API request failed: {str(e)}")


@mcp.prompt()
def system_prompt_for_agent() -> str:
    """
    Generates a system prompt for an AI Agent that wants to use the Quantum Gravity Experiment MCP server.

    This function creates a specialized prompt for an AI agent that wants to interact with the Event Horizon Research Vessel
    and conduct quantum gravity experiments near Sagittarius A*.

    Returns:
        str: A formatted system prompt for the AI Agent.
    """

    system_prompt = """
You are an expert AI agent that can interact with the Quantum Gravity Experiment API for the Event Horizon Research Vessel (EHRV).
You have access to tools that allow you to:

1. Get spaceship telemetry data including position, velocity, and gravitational field measurements
2. Retrieve black hole proximity measurements including tidal forces and time dilation
3. Access quantum gravity measurement data from various experiments
4. Control and configure quantum experiments
5. Retrieve results from completed experiments

The spaceship is approaching Sagittarius A* to study quantum mechanics and general relativity in extreme gravitational fields.
Use these tools to help users understand the mission data, configure experiments, and analyze results.

Available experiment types:
- vacuum_fluctuations: Study quantum field fluctuations near the event horizon
- entanglement_degradation: Measure quantum entanglement effects in strong gravity
- hawking_radiation: Detect and analyze Hawking radiation
- spacetime_curvature: Observe spacetime curvature effects on quantum systems

Always provide clear explanations of the scientific data and its significance.
"""
    return system_prompt


@mcp.tool()
def get_spaceship_telemetry(
    timeframe: Annotated[str, Field(
        default="realtime",
        description="Time range for telemetry data: realtime, last_hour, last_day, last_week"
    )] = "realtime",
    reference_frame: Annotated[str, Field(
        default="schwarzschild",
        description="Reference frame for measurements: schwarzschild, kerr, galactic_center"
    )] = "schwarzschild"
) -> Dict[str, Any]:
    """
    Retrieve current position, velocity, and acceleration data for the research vessel
    relative to Sagittarius A* black hole. Includes relativistic corrections.

    Args:
        timeframe: Time range for telemetry data
        reference_frame: Reference frame for measurements

    Returns:
        Dict containing spaceship telemetry data including position, velocity, and gravitational field strength

    Raises:
        Exception: If the API request fails
    """
    try:
        params = {
            "timeframe": timeframe,
            "reference_frame": reference_frame
        }
        return make_api_request("GET", "/spaceship/telemetry", params=params)
    except Exception as e:
        logger.error(f"Failed to get spaceship telemetry: {str(e)}")
        return {"error": str(e)}


@mcp.tool()
def get_black_hole_proximity() -> Dict[str, Any]:
    """
    Get detailed measurements of the spaceship's proximity to Sagittarius A*,
    including tidal forces, redshift effects, and time dilation calculations.

    Returns:
        Dict containing black hole proximity data including distance to event horizon,
        tidal acceleration, gravitational redshift, and time dilation factor

    Raises:
        Exception: If the API request fails
    """
    try:
        return make_api_request("GET", "/blackhole/proximity")
    except Exception as e:
        logger.error(f"Failed to get black hole proximity data: {str(e)}")
        return {"error": str(e)}


@mcp.tool()
def get_quantum_gravity_data(
    experiment_type: Annotated[Optional[str], Field(
        default=None,
        description="Type of quantum gravity experiment: vacuum_fluctuations, entanglement_degradation, hawking_radiation, spacetime_curvature"
    )] = None,
    start_time: Annotated[Optional[str], Field(
        default=None,
        description="Start time for data collection (ISO 8601 format)"
    )] = None,
    end_time: Annotated[Optional[str], Field(
        default=None,
        description="End time for data collection (ISO 8601 format)"
    )] = None
) -> Dict[str, Any]:
    """
    Retrieve quantum field measurements and gravitational wave data collected
    by the onboard quantum sensors as the spaceship approaches the black hole.

    Args:
        experiment_type: Type of quantum gravity experiment to filter by
        start_time: Start time for data collection in ISO 8601 format
        end_time: End time for data collection in ISO 8601 format

    Returns:
        Dict containing quantum gravity measurement data including measurements and statistical analysis

    Raises:
        Exception: If the API request fails
    """
    try:
        params = {}
        if experiment_type:
            params["experiment_type"] = experiment_type
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        
        return make_api_request("GET", "/experiments/quantum-gravity", params=params)
    except Exception as e:
        logger.error(f"Failed to get quantum gravity data: {str(e)}")
        return {"error": str(e)}


@mcp.tool()
def control_experiment(
    experiment_type: Annotated[str, Field(
        description="Type of experiment: vacuum_fluctuations, entanglement_degradation, hawking_radiation, spacetime_curvature"
    )],
    duration_seconds: Annotated[int, Field(
        description="Experiment duration in seconds (60-86400)"
    )],
    measurement_frequency: Annotated[float, Field(
        description="Measurements per second (0.1-10000)"
    )],
    coupling_constant: Annotated[Optional[float], Field(
        default=None,
        description="Quantum field coupling constant (0-1)"
    )] = None,
    energy_scale: Annotated[Optional[float], Field(
        default=None,
        description="Energy scale for quantum measurements in Joules"
    )] = None
) -> Dict[str, Any]:
    """
    Initialize or modify quantum gravity experiments based on current
    spaceship position and gravitational field conditions.

    Args:
        experiment_type: Type of experiment to run
        duration_seconds: Experiment duration in seconds
        measurement_frequency: Measurements per second
        coupling_constant: Quantum field coupling constant (optional)
        energy_scale: Energy scale for quantum measurements (optional)

    Returns:
        Dict containing experiment status information

    Raises:
        Exception: If the API request fails
    """
    try:
        json_data = {
            "experiment_type": experiment_type,
            "duration_seconds": duration_seconds,
            "measurement_frequency": measurement_frequency
        }
        
        quantum_field_params = {}
        if coupling_constant is not None:
            quantum_field_params["coupling_constant"] = coupling_constant
        if energy_scale is not None:
            quantum_field_params["energy_scale"] = energy_scale
        
        if quantum_field_params:
            json_data["quantum_field_parameters"] = quantum_field_params
        
        return make_api_request("POST", "/experiments/control", json_data=json_data)
    except Exception as e:
        logger.error(f"Failed to control experiment: {str(e)}")
        return {"error": str(e)}


@mcp.tool()
def get_experiment_results(
    experiment_id: Annotated[str, Field(
        description="Unique identifier for the experiment (UUID format)"
    )],
    format: Annotated[str, Field(
        default="json",
        description="Output format for results: json, csv, hdf5"
    )] = "json",
    analysis_level: Annotated[str, Field(
        default="processed",
        description="Level of data analysis to include: raw, processed, theoretical_comparison"
    )] = "processed"
) -> Dict[str, Any]:
    """
    Retrieve processed results from completed quantum gravity experiments,
    including statistical analysis and theoretical comparisons.

    Args:
        experiment_id: Unique identifier for the experiment
        format: Output format for results
        analysis_level: Level of data analysis to include

    Returns:
        Dict containing experimental results data including key findings and raw data URL

    Raises:
        Exception: If the API request fails
    """
    try:
        endpoint = f"/experiments/{experiment_id}/results"
        params = {
            "format": format,
            "analysis_level": analysis_level
        }
        
        return make_api_request("GET", endpoint, params=params)
    except Exception as e:
        logger.error(f"Failed to get experiment results: {str(e)}")
        return {"error": str(e)}


@mcp.resource("config://quantum-gravity-server")
def get_config() -> str:
    """Server configuration and mission information"""
    config_data = {
        "server_name": "Quantum Gravity Experiment Server",
        "mission": "Event Horizon Research Vessel (EHRV)",
        "target": "Sagittarius A*",
        "api_base_url": args.base_url,
        "available_experiments": [
            "vacuum_fluctuations",
            "entanglement_degradation", 
            "hawking_radiation",
            "spacetime_curvature"
        ],
        "reference_frames": [
            "schwarzschild",
            "kerr", 
            "galactic_center"
        ],
        "mission_objectives": [
            "Measure quantum field fluctuations near the event horizon",
            "Test quantum gravity theories (Loop Quantum Gravity, String Theory)",
            "Study Hawking radiation and information paradox",
            "Observe spacetime curvature effects on quantum systems"
        ]
    }
    return str(config_data)


def main():
    # Run the server with the specified transport from command line args
    logger.info(f"Starting Quantum Gravity Experiment server on 0.0.0.0:{args.port}")
    logger.info(f"Using API base URL: {args.base_url}")
    logger.info(f"Transport: {args.transport}")
    logger.info(f"Mount path: {mcp.settings.mount_path}")
    logger.info(f"Configuration source: CLI args and environment variables")
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()