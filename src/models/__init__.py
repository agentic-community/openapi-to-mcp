"""
Data models for OpenAPI to MCP converter.
"""

from .evaluation import (
    OpenAPIEvaluationResult,
    OverallEvaluation,
    OperationEvaluation,
    ParameterEvaluation,
    QualityScore,
    ResponseEvaluation,
    SchemaEvaluation,
    SecurityRequirement,
)

__all__ = [
    "OpenAPIEvaluationResult",
    "OverallEvaluation",
    "OperationEvaluation",
    "ParameterEvaluation",
    "QualityScore",
    "ResponseEvaluation",
    "SchemaEvaluation",
    "SecurityRequirement",
]
