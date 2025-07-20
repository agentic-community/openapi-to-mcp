"""
Pydantic models for OpenAPI specification evaluation and enhancement results.
Based on the evaluation template and example files from the GitHub issue.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    # Define log message format
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


class QualityScore(str, Enum):
    """Quality scoring enum for various aspects of OpenAPI specs."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    MISSING = "missing"


class LintingSeverity(str, Enum):
    """Severity levels for linting issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUGGESTION = "suggestion"


class LintingIssue(BaseModel):
    """A single linting issue found in the OpenAPI spec."""

    severity: LintingSeverity = Field(description="Severity level of the issue")
    rule: str = Field(description="Name of the linting rule that triggered")
    message: str = Field(description="Human-readable description of the issue")
    path: Optional[str] = Field(
        default=None, description="JSON path where the issue was found"
    )
    location: Optional[str] = Field(
        default=None, description="Specific location description"
    )
    suggestion: Optional[str] = Field(
        default=None, description="Suggested fix for the issue"
    )

    model_config = {"str_strip_whitespace": True}


class LintingResult(BaseModel):
    """Results of OpenAPI spec linting."""

    total_issues: int = Field(description="Total number of issues found")
    error_count: int = Field(default=0, description="Number of error-level issues")
    warning_count: int = Field(default=0, description="Number of warning-level issues")
    info_count: int = Field(default=0, description="Number of info-level issues")
    suggestion_count: int = Field(
        default=0, description="Number of suggestion-level issues"
    )

    issues: List[LintingIssue] = Field(
        default_factory=list, description="List of all issues found"
    )
    linting_score: int = Field(
        ge=1, le=5, description="Overall linting score (1=many issues, 5=clean spec)"
    )
    linting_summary: str = Field(description="Summary of linting results")

    model_config = {"str_strip_whitespace": True}

    def get_issues_by_severity(self, severity: LintingSeverity) -> List[LintingIssue]:
        """Get all issues of a specific severity level."""
        return [issue for issue in self.issues if issue.severity == severity]

    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return self.error_count > 0

    def is_clean(self) -> bool:
        """Check if the spec has no issues."""
        return self.total_issues == 0

    @property
    def critical_issues(self) -> int:
        """Alias for error_count for backward compatibility."""
        return self.error_count

    @property
    def major_issues(self) -> int:
        """Alias for warning_count for backward compatibility."""
        return self.warning_count

    @property
    def minor_issues(self) -> int:
        """Alias for suggestion_count for backward compatibility."""
        return self.suggestion_count

    @property
    def info_issues(self) -> int:
        """Alias for info_count for backward compatibility."""
        return self.info_count


class SecurityRequirement(BaseModel):
    """Security requirement evaluation model."""

    type: str = Field(description="Type of security (e.g., apiKey, oauth2, http)")
    name: str = Field(description="Name of the security scheme")
    description: Optional[str] = Field(
        default=None, description="Description of security requirement"
    )
    required: bool = Field(
        default=True, description="Whether this security is required"
    )

    model_config = {"str_strip_whitespace": True}


class ParameterEvaluation(BaseModel):
    """Evaluation of a parameter's completeness."""

    name: str = Field(description="Parameter name")
    location: str = Field(description="Parameter location (query, path, header, etc.)")
    description_quality: QualityScore = Field(
        description="Quality of parameter description"
    )
    example_provided: bool = Field(
        default=False, description="Whether example is provided"
    )
    constraints_defined: bool = Field(
        default=False, description="Whether constraints are defined"
    )
    suggestions: List[str] = Field(
        default_factory=list, description="Improvement suggestions"
    )

    model_config = {"str_strip_whitespace": True}


class ResponseEvaluation(BaseModel):
    """Evaluation of a response definition."""

    status_code: str = Field(description="HTTP status code")
    description_quality: QualityScore = Field(
        description="Quality of response description"
    )
    schema_provided: bool = Field(
        default=False, description="Whether response schema is provided"
    )
    examples_provided: bool = Field(
        default=False, description="Whether examples are provided"
    )
    headers_documented: bool = Field(
        default=False, description="Whether headers are documented"
    )
    suggestions: List[str] = Field(
        default_factory=list, description="Improvement suggestions"
    )

    model_config = {"str_strip_whitespace": True}


class OperationEvaluation(BaseModel):
    """Evaluation of an API operation (endpoint)."""

    operation_id: Optional[str] = Field(default=None, description="Operation ID")
    method: str = Field(description="HTTP method")
    path: str = Field(description="API path")
    summary: Optional[str] = Field(default=None, description="Operation summary")
    description: Optional[str] = Field(
        default=None, description="Operation description"
    )

    # Quality assessments
    description_quality: QualityScore = Field(
        description="Quality of operation description"
    )
    parameter_completeness: QualityScore = Field(
        description="Completeness of parameters"
    )
    response_completeness: QualityScore = Field(description="Completeness of responses")

    # Detailed evaluations
    parameters: List[ParameterEvaluation] = Field(
        default_factory=list, description="Parameter evaluations"
    )
    responses: List[ResponseEvaluation] = Field(
        default_factory=list, description="Response evaluations"
    )

    # Suggestions
    missing_parameters: List[str] = Field(
        default_factory=list, description="Suggested missing parameters"
    )
    missing_responses: List[str] = Field(
        default_factory=list, description="Suggested missing responses"
    )
    enhancement_suggestions: List[str] = Field(
        default_factory=list, description="General enhancement suggestions"
    )

    model_config = {"str_strip_whitespace": True}


class SchemaEvaluation(BaseModel):
    """Evaluation of data schemas/models."""

    schema_name: str = Field(description="Name of the schema")
    description_quality: QualityScore = Field(
        description="Quality of schema description"
    )
    properties_documented: bool = Field(
        default=False, description="Whether properties are well documented"
    )
    examples_provided: bool = Field(
        default=False, description="Whether examples are provided"
    )
    required_fields_specified: bool = Field(
        default=False, description="Whether required fields are specified"
    )
    suggestions: List[str] = Field(
        default_factory=list, description="Improvement suggestions"
    )

    model_config = {"str_strip_whitespace": True}


class OverallEvaluation(BaseModel):
    """Overall evaluation of the OpenAPI specification."""

    # Overall scores
    overall_quality: QualityScore = Field(description="Overall quality assessment")
    completeness_score: int = Field(
        ge=1, le=5, description="Completeness score (1-5 Likert scale, 1=worst, 5=best)"
    )
    ai_readiness_score: int = Field(
        ge=1,
        le=5,
        description="AI/LLM readiness score (1-5 Likert scale, 1=worst, 5=best)",
    )

    # Specific assessments
    has_comprehensive_descriptions: bool = Field(
        default=False, description="Has comprehensive descriptions"
    )
    has_good_examples: bool = Field(
        default=False, description="Has good examples throughout"
    )
    has_proper_error_handling: bool = Field(
        default=False, description="Has proper error response documentation"
    )
    security_well_defined: bool = Field(
        default=False, description="Security is well defined"
    )

    # Recommendations
    major_improvements_needed: List[str] = Field(
        default_factory=list, description="Major improvements needed"
    )
    minor_improvements_suggested: List[str] = Field(
        default_factory=list, description="Minor improvements suggested"
    )
    key_strengths: List[str] = Field(
        default_factory=list, description="Key strengths of the API specification"
    )
    areas_for_improvement: List[str] = Field(
        default_factory=list, description="Areas that need improvement"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="General recommendations for improvement"
    )

    model_config = {"str_strip_whitespace": True}


class OpenAPIEvaluationResult(BaseModel):
    """Complete evaluation result for an OpenAPI specification."""

    # Metadata
    evaluation_id: str = Field(description="Unique evaluation identifier")
    original_spec_url: Optional[str] = Field(
        default=None, description="URL of original spec"
    )
    original_spec_filename: Optional[str] = Field(
        default=None, description="Filename of original spec"
    )
    evaluation_timestamp: datetime = Field(
        default_factory=datetime.now, description="When evaluation was performed"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When evaluation was performed (alias)",
    )
    model: str = Field(default="unknown", description="LLM model used for evaluation")
    enhancement_level: str = Field(
        default="comprehensive", description="Level of enhancement requested"
    )

    # Specification info
    openapi_version: str = Field(description="OpenAPI specification version")
    api_title: str = Field(description="API title from spec")
    api_version: str = Field(description="API version from spec")

    # Security evaluation
    security_schemes: List[SecurityRequirement] = Field(
        default_factory=list, description="Security schemes found"
    )

    # Detailed evaluations
    operations: List[OperationEvaluation] = Field(
        default_factory=list, description="Operation evaluations"
    )
    schemas: List[SchemaEvaluation] = Field(
        default_factory=list, description="Schema evaluations"
    )
    overall: OverallEvaluation = Field(description="Overall evaluation")

    # Enhancement details
    needs_enhancement: bool = Field(
        default=True, description="Whether enhancement is recommended"
    )
    enhancement_priority: str = Field(
        default="medium", description="Priority level for enhancement"
    )
    enhanced_openapi_spec: Optional[str] = Field(
        default=None, description="Rewritten OpenAPI spec with improvements"
    )

    # Linting results
    linting_results: Optional[LintingResult] = Field(
        default=None, description="OpenAPI linting results"
    )

    # Usage and cost tracking
    total_tokens: Optional[int] = Field(
        default=None, description="Total tokens used across all LLM calls"
    )
    total_cost_usd: Optional[float] = Field(
        default=None, description="Total cost in USD for all LLM calls"
    )
    llm_calls_count: Optional[int] = Field(
        default=None, description="Number of LLM API calls made"
    )

    # Detailed usage breakdown
    evaluation_usage: Optional[Dict[str, Any]] = Field(
        default=None, description="Usage for evaluation phase"
    )
    mcp_server_usage: Optional[Dict[str, Any]] = Field(
        default=None, description="Usage for MCP server generation"
    )
    mcp_client_usage: Optional[Dict[str, Any]] = Field(
        default=None, description="Usage for MCP client generation"
    )

    @property
    def linting(self) -> Optional[LintingResult]:
        """Alias for linting_results for backward compatibility."""
        return self.linting_results

    model_config = {
        "str_strip_whitespace": True,
        "json_encoders": {datetime: lambda v: v.isoformat()},
    }

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the evaluation results."""
        try:
            summary = {
                "evaluation_id": self.evaluation_id,
                "api_title": self.api_title,
                "overall_quality": self.overall.overall_quality.value,
                "completeness_score": self.overall.completeness_score,
                "ai_readiness_score": self.overall.ai_readiness_score,
                "total_operations": len(self.operations),
                "total_schemas": len(self.schemas),
                "needs_enhancement": self.needs_enhancement,
                "enhancement_priority": self.enhancement_priority,
                "major_improvements_count": len(self.overall.major_improvements_needed),
                "timestamp": self.evaluation_timestamp.isoformat(),
            }

            # Add linting information if available
            if self.linting_results:
                summary.update(
                    {
                        "linting_score": self.linting_results.linting_score,
                        "linting_total_issues": self.linting_results.total_issues,
                        "linting_errors": self.linting_results.error_count,
                        "linting_warnings": self.linting_results.warning_count,
                        "linting_summary": self.linting_results.linting_summary,
                    }
                )

            logger.info(f"Generated evaluation summary for {self.api_title}")
            return summary
        except Exception as e:
            logger.error(f"Failed to generate evaluation summary: {e}")
            raise


if __name__ == "__main__":
    """Standalone testing of evaluation models."""
    logger.info("Testing evaluation models...")

    # Test basic models
    security = SecurityRequirement(
        type="apiKey",
        name="X-API-Key",
        description="API key for authentication",
        required=True,
    )
    logger.info(f"Created security requirement: {security.name}")

    # Test parameter evaluation
    param_eval = ParameterEvaluation(
        name="user_id",
        location="path",
        description_quality=QualityScore.GOOD,
        example_provided=True,
        constraints_defined=False,
        suggestions=["Add format validation", "Provide more examples"],
    )
    logger.info(f"Created parameter evaluation: {param_eval.name}")

    # Test complete evaluation result
    evaluation = OpenAPIEvaluationResult(
        evaluation_id="test-eval-001",
        openapi_version="3.0.3",
        api_title="Test API",
        api_version="1.0.0",
        security_schemes=[security],
        operations=[],
        schemas=[],
        overall=OverallEvaluation(
            overall_quality=QualityScore.FAIR,
            completeness_score=3,
            ai_readiness_score=3,
            major_improvements_needed=[
                "Add detailed descriptions",
                "Include more examples",
            ],
        ),
    )

    # Test summary generation
    summary = evaluation.get_summary()
    logger.info(f"Generated summary: {summary}")

    logger.info("Evaluation models test completed successfully")
