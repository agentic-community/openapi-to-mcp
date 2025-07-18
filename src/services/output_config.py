"""
Output Configuration Service

This module provides the OutputConfig class for managing file output paths and configurations
for the OpenAPI to MCP converter. It handles the creation and management of output files
including evaluation results, enhanced specifications, and summary reports.

Third-party packages:
- pathlib (built-in): For path manipulation and file operations
- typing (built-in): For type hints and annotations

Sample input:
- output_path: Path("temp")
- provider_name: "anthropic"
- spec_name: "petstore"
- timestamp: "20240106_143022"

Expected output:
- OutputConfig instance with configured file paths for all output types
"""

import logging
from pathlib import Path
from typing import Optional, Tuple


def _ensure_directory_exists(directory: Path) -> None:
    """Ensure a directory exists, creating it if necessary."""
    directory.mkdir(parents=True, exist_ok=True)
    logging.info(f"Ensured directory exists: {directory}")


class OutputConfig:
    """Configuration for output file generation."""
    
    def __init__(self, output_path: Optional[Path] = None, provider_name: str = "", spec_name: str = "", timestamp: str = ""):
        self.output_path = output_path
        self.provider_name = provider_name
        self.spec_name = spec_name
        self.timestamp = timestamp
    
    def get_results_dir(self) -> Path:
        """Get the results directory path."""
        # Always use the proper results/<provider>/<spec_name> structure
        return Path("results") / self.provider_name / self.spec_name
    
    def get_output_files(self) -> Tuple[Path, Path, Path, Path, Path]:
        """Get paths for all output files."""
        results_dir = self.get_results_dir()
        results_dir.mkdir(parents=True, exist_ok=True)
        
        eval_file = results_dir / f"evaluation_{self.timestamp}.json"
        enhanced_file = results_dir / f"enhanced_spec_{self.timestamp}.yaml"
        original_file = results_dir / f"original_spec_{self.timestamp}.yaml"
        summary_file = results_dir / f"summary_{self.timestamp}.md"
        usage_file = results_dir / f"usage_{self.timestamp}.json"
        
        return eval_file, enhanced_file, original_file, summary_file, usage_file


if __name__ == "__main__":
    import sys
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    # Test 1: Basic OutputConfig creation
    total_tests += 1
    try:
        config = OutputConfig(None, "anthropic", "petstore", "20240106_143022")
        if config.provider_name != "anthropic":
            all_validation_failures.append(f"Basic creation: Expected provider_name 'anthropic', got '{config.provider_name}'")
        if config.spec_name != "petstore":
            all_validation_failures.append(f"Basic creation: Expected spec_name 'petstore', got '{config.spec_name}'")
        if config.timestamp != "20240106_143022":
            all_validation_failures.append(f"Basic creation: Expected timestamp '20240106_143022', got '{config.timestamp}'")
    except Exception as e:
        all_validation_failures.append(f"Basic creation: Unexpected exception {type(e).__name__}: {e}")
    
    # Test 2: Results directory path generation
    total_tests += 1
    try:
        config = OutputConfig(None, "openai", "example", "20240101_120000")
        results_dir = config.get_results_dir()
        expected_dir = Path("results/openai/example")
        if results_dir != expected_dir:
            all_validation_failures.append(f"Results directory: Expected '{expected_dir}', got '{results_dir}'")
    except Exception as e:
        all_validation_failures.append(f"Results directory: Unexpected exception {type(e).__name__}: {e}")
    
    # Test 3: Output files generation
    total_tests += 1
    try:
        config = OutputConfig(None, "claude", "api", "20240202_180000")
        eval_file, enhanced_file, original_file, summary_file, usage_file = config.get_output_files()
        
        expected_eval = Path("results/claude/api/evaluation_20240202_180000.json")
        if eval_file != expected_eval:
            all_validation_failures.append(f"Output files: Expected eval_file '{expected_eval}', got '{eval_file}'")
            
        expected_enhanced = Path("results/claude/api/enhanced_spec_20240202_180000.yaml")
        if enhanced_file != expected_enhanced:
            all_validation_failures.append(f"Output files: Expected enhanced_file '{expected_enhanced}', got '{enhanced_file}'")
            
        expected_summary = Path("results/claude/api/summary_20240202_180000.md")
        if summary_file != expected_summary:
            all_validation_failures.append(f"Output files: Expected summary_file '{expected_summary}', got '{summary_file}'")
    except Exception as e:
        all_validation_failures.append(f"Output files: Unexpected exception {type(e).__name__}: {e}")
    
    # Test 4: Directory creation verification
    total_tests += 1
    try:
        config = OutputConfig(None, "test", "spec", "20240303_090000")
        eval_file, _, _, _, _ = config.get_output_files()
        expected_dir = Path("results/test/spec")
        if not expected_dir.exists():
            all_validation_failures.append("Directory creation: Results directory was not created by get_output_files()")
    except Exception as e:
        all_validation_failures.append(f"Directory creation: Unexpected exception {type(e).__name__}: {e}")
    
    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("OutputConfig class is validated and ready for use")
        sys.exit(0)