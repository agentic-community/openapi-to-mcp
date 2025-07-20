"""
Display and output formatting utilities.
Handles console output formatting for evaluation results and summaries.

Sample input: evaluation result with scores and metadata
Expected output: Formatted console display with status indicators
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _check_mcp_generation_criteria(evaluation, generate_threshold: float = 3.0) -> bool:
    """Check if the evaluation meets criteria for MCP server generation.

    Args:
        evaluation: The evaluation result object
        generate_threshold: Minimum score threshold for MCP generation (default: 3.0)

    Returns:
        bool: True if both completeness and AI readiness scores meet the threshold
    """
    completeness_ok = evaluation.overall.completeness_score >= generate_threshold
    ai_readiness_ok = evaluation.overall.ai_readiness_score >= generate_threshold

    logger.info("🔍 MCP Generation Criteria Check:")
    logger.info(
        f"   Completeness Score: {evaluation.overall.completeness_score} (required: >= {generate_threshold})"
    )
    logger.info(
        f"   AI Readiness Score: {evaluation.overall.ai_readiness_score} (required: >= {generate_threshold})"
    )
    logger.info(f"   Completeness OK: {completeness_ok}")
    logger.info(f"   AI Readiness OK: {ai_readiness_ok}")
    logger.info(
        f"   Overall MCP Generation Criteria Met: {completeness_ok and ai_readiness_ok}"
    )

    return completeness_ok and ai_readiness_ok


def _display_evaluation_results(evaluation) -> None:
    """Display evaluation results to console."""
    print("\n📊 EVALUATION RESULTS")
    print(f"API Title: {evaluation.api_title}")
    print(f"⭐ Overall Quality: {evaluation.overall.overall_quality.value.upper()}")
    print(f"📊 Completeness Score: {evaluation.overall.completeness_score}/5")
    print(f"🤖 AI Readiness Score: {evaluation.overall.ai_readiness_score}/5")


def _print_results_summary(
    evaluation,
    output_config,
    mcpserver_path: Optional[Path] = None,
    generate_threshold: float = 3.0,
) -> None:
    """Print a summary of the evaluation results.

    Args:
        evaluation: The evaluation result object
        output_config: Output configuration object
        mcpserver_path: Path to generated MCP server directory (optional)
        generate_threshold: Minimum score threshold for MCP generation (default: 3.0)
    """
    try:
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"📁 Results Directory: {output_config.get_results_dir()}")
        print(f"🤖 Model: {evaluation.model}")
        print(f"📊 Enhancement Level: {evaluation.enhancement_level}")
        print(f"⏰ Timestamp: {evaluation.timestamp}")
        print("")
        print("📊 SCORES:")
        print(f"   Overall Quality: {evaluation.overall.overall_quality.value.upper()}")
        print(f"   Completeness: {evaluation.overall.completeness_score}/5")
        print(f"   AI Readiness: {evaluation.overall.ai_readiness_score}/5")

        # Linting summary
        if hasattr(evaluation, "linting") and evaluation.linting:
            print("\n🔍 LINTING RESULTS:")
            print(f"   Total Issues: {evaluation.linting.total_issues}")
            if evaluation.linting.critical_issues > 0:
                print(f"   🔴 Critical: {evaluation.linting.critical_issues}")
            if evaluation.linting.major_issues > 0:
                print(f"   🟠 Major: {evaluation.linting.major_issues}")
            if evaluation.linting.minor_issues > 0:
                print(f"   🟡 Minor: {evaluation.linting.minor_issues}")
            if evaluation.linting.info_issues > 0:
                print(f"   ℹ️  Info: {evaluation.linting.info_issues}")

        # MCP Generation Status
        print("\n🚀 MCP GENERATION:")
        if mcpserver_path:
            print(f"   [SUCCESS] MCP server generated: {mcpserver_path}")
            print(f"   📝 Test with: cd {mcpserver_path} && python client.py")
        else:
            meets_criteria = _check_mcp_generation_criteria(
                evaluation, generate_threshold
            )
            if not meets_criteria:
                print(
                    f"   [FAILED] Criteria not met (need >={generate_threshold} for both completeness and AI readiness)"
                )
            else:
                print("   ⚠️  Generation failed (check logs)")

        print("\n📁 OUTPUT FILES:")
        eval_file, enhanced_file, original_file, summary_file, usage_file = (
            output_config.get_output_files()
        )
        print(f"   📊 Evaluation: {eval_file}")
        print(f"   📄 Summary: {summary_file}")
        print(f"   📝 Enhanced Spec: {enhanced_file}")
        print(f"   Original Spec: {original_file}")

        print("\n💡 QUICK ACTIONS:")
        print(f"   View summary: cat {summary_file}")
        print(f"   View evaluation: cat {eval_file}")
        if mcpserver_path:
            print(f"   Test MCP server: cd {mcpserver_path} && python client.py")

        print(f"{'='*80}")
    except Exception as e:
        logger.error(f"Failed to print results summary: {e}")


if __name__ == "__main__":
    import sys
    from datetime import datetime

    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0

    # Create mock evaluation object for testing
    class MockOverall:
        def __init__(self):
            self.overall_quality = type("Quality", (), {"value": "good"})()
            self.completeness_score = 4
            self.ai_readiness_score = 3

    class MockEvaluation:
        def __init__(self):
            self.api_title = "Test API"
            self.model = "claude-3-5-sonnet-20241022"
            self.enhancement_level = "comprehensive"
            self.timestamp = datetime.now()
            self.overall = MockOverall()

    class MockOutputConfig:
        def get_results_dir(self):
            return Path("test/results")

        def get_output_files(self):
            base = Path("test/results")
            return (
                base / "eval.json",
                base / "enhanced.yaml",
                base / "original.yaml",
                base / "summary.md",
                base / "usage.json",
            )

    # Test 1: Check MCP generation criteria - meets criteria
    total_tests += 1
    test_eval = MockEvaluation()
    result = _check_mcp_generation_criteria(test_eval, generate_threshold=3.0)
    expected = True  # completeness=4, ai_readiness=3, both >= 3
    if result != expected:
        all_validation_failures.append(
            f"MCP criteria check (meets): Expected {expected}, got {result}"
        )

    # Test 2: Check MCP generation criteria - doesn't meet criteria
    total_tests += 1
    test_eval_fail = MockEvaluation()
    test_eval_fail.overall.completeness_score = 2  # Below threshold
    result = _check_mcp_generation_criteria(test_eval_fail, generate_threshold=3.0)
    expected = False
    if result != expected:
        all_validation_failures.append(
            f"MCP criteria check (fails): Expected {expected}, got {result}"
        )

    # Test 3: Display evaluation results (basic functionality test)
    total_tests += 1
    try:
        _display_evaluation_results(test_eval)
        # If no exception raised, test passes
    except Exception as e:
        all_validation_failures.append(f"Display evaluation results error: {e}")

    # Test 4: Print results summary (basic functionality test)
    total_tests += 1
    try:
        mock_config = MockOutputConfig()
        _print_results_summary(test_eval, mock_config, generate_threshold=3.0)
        # If no exception raised, test passes
    except Exception as e:
        all_validation_failures.append(f"Print results summary error: {e}")

    # Final validation result
    if all_validation_failures:
        print(
            f"[FAILED] VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:"
        )
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(
            f"[PASSED] VALIDATION PASSED - All {total_tests} tests produced expected results"
        )
        print("Output formatter functions are validated and ready for use")
        sys.exit(0)
