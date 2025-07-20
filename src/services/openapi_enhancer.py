"""
OpenAPI Enhancement Service for evaluating and improving OpenAPI specifications.
Uses Amazon Bedrock with Anthropic Claude to analyze specs and provide enhancement suggestions.
"""

import json
import uuid
import yaml
import logging
import tempfile
from jinja2 import Template
from datetime import datetime
from .config_loader import config
from typing import Any, Dict, List, Optional
from .llm_client import get_llm_client, LLMRequest
from pydantic import BaseModel, Field, ValidationError
from ..models.evaluation import (
    OpenAPIEvaluationResult,
    LintingResult,
    LintingIssue,
    LintingSeverity,
)

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    # Define log message format
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


class OpenAPILinter:
    """OpenAPI specification linter for validating best practices and common issues."""

    def __init__(self):
        """Initialize the OpenAPI linter."""
        self.issues = []

    def lint_specification(self, spec_dict: Dict[str, Any]) -> LintingResult:
        """Perform comprehensive linting of an OpenAPI specification."""
        try:
            self.issues = []
            logger.info("Starting OpenAPI specification linting")

            # Run all linting checks
            self._check_basic_structure(spec_dict)
            self._check_info_section(spec_dict)
            self._check_servers(spec_dict)
            self._check_paths(spec_dict)
            self._check_components(spec_dict)
            self._check_security(spec_dict)
            self._check_documentation_quality(spec_dict)
            self._check_naming_conventions(spec_dict)

            # Calculate counts by severity
            error_count = len(
                [i for i in self.issues if i.severity == LintingSeverity.ERROR]
            )
            warning_count = len(
                [i for i in self.issues if i.severity == LintingSeverity.WARNING]
            )
            info_count = len(
                [i for i in self.issues if i.severity == LintingSeverity.INFO]
            )
            suggestion_count = len(
                [i for i in self.issues if i.severity == LintingSeverity.SUGGESTION]
            )

            # Calculate linting score (1-5 scale)
            linting_score = self._calculate_linting_score(
                error_count, warning_count, info_count
            )

            # Generate summary
            summary = self._generate_linting_summary(
                error_count, warning_count, info_count, suggestion_count
            )

            result = LintingResult(
                total_issues=len(self.issues),
                error_count=error_count,
                warning_count=warning_count,
                info_count=info_count,
                suggestion_count=suggestion_count,
                issues=self.issues,
                linting_score=linting_score,
                linting_summary=summary,
            )

            logger.info(
                f"Linting completed: {len(self.issues)} issues found (Score: {linting_score}/5)"
            )
            return result

        except Exception as e:
            logger.error(f"Linting failed: {e}")
            raise

    def _add_issue(
        self,
        severity: LintingSeverity,
        rule: str,
        message: str,
        path: Optional[str] = None,
        location: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        """Add a linting issue to the results."""
        issue = LintingIssue(
            severity=severity,
            rule=rule,
            message=message,
            path=path,
            location=location,
            suggestion=suggestion,
        )
        self.issues.append(issue)

    def _check_basic_structure(self, spec: Dict[str, Any]):
        """Check basic OpenAPI structure requirements."""
        # Check required root fields
        required_fields = {
            "openapi": "OpenAPI version is required",
            "info": "Info section is required",
            "paths": "Paths section is required",
        }

        for field, message in required_fields.items():
            if field not in spec:
                self._add_issue(
                    LintingSeverity.ERROR,
                    "required-field",
                    message,
                    path=f"$.{field}",
                    suggestion=f"Add the required '{field}' field to the root of your specification",
                )

        # Check OpenAPI version
        if "openapi" in spec:
            version = spec["openapi"]
            if not isinstance(version, str):
                self._add_issue(
                    LintingSeverity.ERROR,
                    "openapi-version",
                    "OpenAPI version must be a string",
                )
            elif not version.startswith("3."):
                self._add_issue(
                    LintingSeverity.WARNING,
                    "openapi-version",
                    f"Consider upgrading to OpenAPI 3.x (current: {version})",
                    suggestion="Upgrade to OpenAPI 3.0.x or 3.1.x for better tooling support",
                )

    def _check_info_section(self, spec: Dict[str, Any]):
        """Check info section completeness."""
        if "info" not in spec:
            return

        info = spec["info"]

        # Required fields in info
        required_info_fields = {
            "title": "API title is required",
            "version": "API version is required",
        }

        for field, message in required_info_fields.items():
            if field not in info:
                self._add_issue(
                    LintingSeverity.ERROR,
                    "info-required",
                    message,
                    path=f"$.info.{field}",
                    suggestion=f"Add '{field}' to the info section",
                )

        # Recommended fields
        recommended_fields = {
            "description": "API description helps users understand the purpose",
            "contact": "Contact information helps users get support",
            "license": "License information clarifies usage rights",
        }

        for field, message in recommended_fields.items():
            if field not in info:
                self._add_issue(
                    LintingSeverity.SUGGESTION,
                    "info-recommended",
                    message,
                    path=f"$.info.{field}",
                    suggestion=f"Consider adding '{field}' to the info section",
                )

        # Check description quality
        if "description" in info:
            description = info["description"]
            if isinstance(description, str) and len(description.strip()) < 20:
                self._add_issue(
                    LintingSeverity.WARNING,
                    "description-quality",
                    "API description is too brief",
                    path="$.info.description",
                    suggestion="Provide a more detailed description of your API's purpose and functionality",
                )

    def _check_servers(self, spec: Dict[str, Any]):
        """Check servers configuration."""
        if "servers" not in spec:
            self._add_issue(
                LintingSeverity.INFO,
                "servers-missing",
                "No servers defined - consider adding server URLs",
                path="$.servers",
                suggestion="Add a 'servers' array with your API base URLs",
            )
            return

        servers = spec["servers"]
        if not isinstance(servers, list) or len(servers) == 0:
            self._add_issue(
                LintingSeverity.WARNING,
                "servers-empty",
                "Servers array is empty",
                path="$.servers",
            )
            return

        for i, server in enumerate(servers):
            if not isinstance(server, dict):
                continue

            if "url" not in server:
                self._add_issue(
                    LintingSeverity.ERROR,
                    "server-url-missing",
                    f"Server {i} is missing URL",
                    path=f"$.servers[{i}].url",
                )

            if "description" not in server:
                self._add_issue(
                    LintingSeverity.SUGGESTION,
                    "server-description-missing",
                    f"Server {i} has no description",
                    path=f"$.servers[{i}].description",
                    suggestion="Add description to help users understand the server purpose",
                )

    def _check_paths(self, spec: Dict[str, Any]):
        """Check paths and operations."""
        if "paths" not in spec:
            return

        paths = spec["paths"]
        if not isinstance(paths, dict):
            self._add_issue(
                LintingSeverity.ERROR, "paths-invalid", "Paths must be an object"
            )
            return

        if len(paths) == 0:
            self._add_issue(
                LintingSeverity.WARNING,
                "paths-empty",
                "No paths defined in the API",
                path="$.paths",
            )
            return

        for path_name, path_obj in paths.items():
            if not isinstance(path_obj, dict):
                continue

            self._check_path_parameters(path_name, path_obj)

            # Check operations
            http_methods = [
                "get",
                "post",
                "put",
                "delete",
                "patch",
                "head",
                "options",
                "trace",
            ]

            for method in http_methods:
                if method in path_obj:
                    operation = path_obj[method]
                    self._check_operation(path_name, method, operation)

    def _check_path_parameters(self, path_name: str, path_obj: Dict[str, Any]):
        """Check path-level parameters."""
        # Check if path has parameters but no path-level parameter definitions
        if "{" in path_name and "}" in path_name:
            if "parameters" not in path_obj:
                # Check if any operations define these parameters
                has_param_definitions = False
                for method in [
                    "get",
                    "post",
                    "put",
                    "delete",
                    "patch",
                    "head",
                    "options",
                ]:
                    if method in path_obj and "parameters" in path_obj[method]:
                        has_param_definitions = True
                        break

                if not has_param_definitions:
                    self._add_issue(
                        LintingSeverity.WARNING,
                        "path-parameters-undefined",
                        f"Path '{path_name}' has parameters but no parameter definitions",
                        path=f"$.paths.{path_name}.parameters",
                        suggestion="Define path parameters at the path level or in operations",
                    )

    def _check_operation(self, path_name: str, method: str, operation: Dict[str, Any]):
        """Check individual operation."""
        operation_path = f"$.paths.{path_name}.{method}"

        # Check for operationId
        if "operationId" not in operation:
            self._add_issue(
                LintingSeverity.SUGGESTION,
                "operation-id-missing",
                f"Operation {method.upper()} {path_name} has no operationId",
                path=f"{operation_path}.operationId",
                suggestion="Add operationId for better code generation and tooling support",
            )

        # Check for summary
        if "summary" not in operation:
            self._add_issue(
                LintingSeverity.WARNING,
                "operation-summary-missing",
                f"Operation {method.upper()} {path_name} has no summary",
                path=f"{operation_path}.summary",
                suggestion="Add a brief summary describing what this operation does",
            )

        # Check for description
        if "description" not in operation:
            self._add_issue(
                LintingSeverity.INFO,
                "operation-description-missing",
                f"Operation {method.upper()} {path_name} has no description",
                path=f"{operation_path}.description",
                suggestion="Add a detailed description explaining the operation",
            )

        # Check responses
        if "responses" not in operation:
            self._add_issue(
                LintingSeverity.ERROR,
                "operation-responses-missing",
                f"Operation {method.upper()} {path_name} has no responses",
                path=f"{operation_path}.responses",
                suggestion="Define at least one response for this operation",
            )
        else:
            self._check_responses(operation_path, operation["responses"])

        # Check parameters
        if "parameters" in operation:
            self._check_parameters(operation_path, operation["parameters"])

        # Check request body for operations that typically have them
        if method.lower() in ["post", "put", "patch"]:
            if "requestBody" not in operation:
                self._add_issue(
                    LintingSeverity.INFO,
                    "request-body-missing",
                    f"Operation {method.upper()} {path_name} might need a requestBody",
                    path=f"{operation_path}.requestBody",
                    suggestion="Consider if this operation should have a request body",
                )

    def _check_responses(self, operation_path: str, responses: Dict[str, Any]):
        """Check operation responses."""
        if not isinstance(responses, dict):
            return

        # Check for success responses
        success_codes = [code for code in responses.keys() if code.startswith("2")]
        if not success_codes:
            self._add_issue(
                LintingSeverity.WARNING,
                "no-success-response",
                "No success (2xx) response defined",
                path=f"{operation_path}.responses",
                suggestion="Define at least one 2xx response",
            )

        # Check for error responses
        error_codes = [
            code
            for code in responses.keys()
            if code.startswith("4") or code.startswith("5")
        ]
        if not error_codes:
            self._add_issue(
                LintingSeverity.SUGGESTION,
                "no-error-response",
                "No error (4xx/5xx) responses defined",
                path=f"{operation_path}.responses",
                suggestion="Consider defining common error responses (400, 401, 404, 500)",
            )

        # Check individual responses
        for status_code, response in responses.items():
            if not isinstance(response, dict):
                continue

            response_path = f"{operation_path}.responses.{status_code}"

            if "description" not in response:
                self._add_issue(
                    LintingSeverity.ERROR,
                    "response-description-missing",
                    f"Response {status_code} has no description",
                    path=f"{response_path}.description",
                    suggestion="Add description explaining what this response represents",
                )

    def _check_parameters(self, operation_path: str, parameters: List[Dict[str, Any]]):
        """Check operation parameters."""
        if not isinstance(parameters, list):
            return

        for i, param in enumerate(parameters):
            if not isinstance(param, dict):
                continue

            param_path = f"{operation_path}.parameters[{i}]"

            # Required fields
            required_param_fields = ["name", "in"]
            for field in required_param_fields:
                if field not in param:
                    self._add_issue(
                        LintingSeverity.ERROR,
                        "parameter-required-field",
                        f"Parameter {i} is missing required field '{field}'",
                        path=f"{param_path}.{field}",
                    )

            # Check description
            if "description" not in param:
                param_name = param.get("name", f"parameter {i}")
                self._add_issue(
                    LintingSeverity.WARNING,
                    "parameter-description-missing",
                    f"Parameter '{param_name}' has no description",
                    path=f"{param_path}.description",
                    suggestion="Add description explaining the parameter's purpose",
                )

            # Check schema
            if "schema" not in param and "content" not in param:
                self._add_issue(
                    LintingSeverity.ERROR,
                    "parameter-schema-missing",
                    f"Parameter '{param.get('name', i)}' has no schema or content",
                    path=f"{param_path}.schema",
                )

    def _check_components(self, spec: Dict[str, Any]):
        """Check components section."""
        if "components" not in spec:
            self._add_issue(
                LintingSeverity.INFO,
                "components-missing",
                "No components section defined",
                path="$.components",
                suggestion="Consider using components to reuse schemas, parameters, and responses",
            )
            return

        components = spec["components"]
        if not isinstance(components, dict):
            return

        # Check schemas
        if "schemas" in components:
            schemas = components["schemas"]
            if isinstance(schemas, dict):
                for schema_name, schema in schemas.items():
                    self._check_schema(
                        f"$.components.schemas.{schema_name}", schema_name, schema
                    )

    def _check_schema(self, path: str, name: str, schema: Dict[str, Any]):
        """Check individual schema."""
        if not isinstance(schema, dict):
            return

        # Check for description
        if "description" not in schema:
            self._add_issue(
                LintingSeverity.SUGGESTION,
                "schema-description-missing",
                f"Schema '{name}' has no description",
                path=f"{path}.description",
                suggestion="Add description explaining what this schema represents",
            )

        # Check for type
        if (
            "type" not in schema
            and "$ref" not in schema
            and "allOf" not in schema
            and "oneOf" not in schema
            and "anyOf" not in schema
        ):
            self._add_issue(
                LintingSeverity.WARNING,
                "schema-type-missing",
                f"Schema '{name}' has no type definition",
                path=f"{path}.type",
                suggestion="Define the type of this schema",
            )

        # Check object properties
        if schema.get("type") == "object":
            if "properties" not in schema:
                self._add_issue(
                    LintingSeverity.INFO,
                    "object-properties-missing",
                    f"Object schema '{name}' has no properties",
                    path=f"{path}.properties",
                )

    def _check_security(self, spec: Dict[str, Any]):
        """Check security definitions."""
        has_security_schemes = (
            "components" in spec and "securitySchemes" in spec["components"]
        )
        has_global_security = "security" in spec

        if not has_security_schemes and not has_global_security:
            self._add_issue(
                LintingSeverity.INFO,
                "security-not-defined",
                "No security schemes or requirements defined",
                suggestion="Consider adding security if your API requires authentication",
            )

    def _check_documentation_quality(self, spec: Dict[str, Any]):
        """Check overall documentation quality."""
        # Count missing descriptions
        missing_descriptions = 0

        # Check path descriptions
        if "paths" in spec:
            for path_name, path_obj in spec["paths"].items():
                if isinstance(path_obj, dict):
                    for method in [
                        "get",
                        "post",
                        "put",
                        "delete",
                        "patch",
                        "head",
                        "options",
                    ]:
                        if method in path_obj:
                            operation = path_obj[method]
                            if (
                                isinstance(operation, dict)
                                and "description" not in operation
                            ):
                                missing_descriptions += 1

        if missing_descriptions > 0:
            self._add_issue(
                LintingSeverity.INFO,
                "documentation-quality",
                f"{missing_descriptions} operations are missing descriptions",
                suggestion="Add descriptions to improve API documentation",
            )

    def _check_naming_conventions(self, spec: Dict[str, Any]):
        """Check naming conventions."""
        if "paths" not in spec:
            return

        paths = spec["paths"]
        for path_name in paths.keys():
            # Check for consistent path naming (should start with /)
            if not path_name.startswith("/"):
                self._add_issue(
                    LintingSeverity.ERROR,
                    "path-format",
                    f"Path '{path_name}' should start with '/'",
                    path=f"$.paths.{path_name}",
                )

            # Check for camelCase vs snake_case inconsistencies (basic check)
            if "_" in path_name and any(c.isupper() for c in path_name):
                self._add_issue(
                    LintingSeverity.SUGGESTION,
                    "naming-consistency",
                    f"Path '{path_name}' mixes camelCase and snake_case",
                    path=f"$.paths.{path_name}",
                    suggestion="Use consistent naming convention throughout your API",
                )

    def _calculate_linting_score(
        self, error_count: int, warning_count: int, info_count: int
    ) -> int:
        """Calculate a linting score from 1-5 based on issue counts."""
        # Weighted scoring: errors are worse than warnings, warnings worse than info
        penalty_score = (error_count * 3) + (warning_count * 2) + (info_count * 1)

        if penalty_score == 0:
            return 5  # Perfect score
        elif penalty_score <= 2:
            return 4  # Good
        elif penalty_score <= 5:
            return 3  # Fair
        elif penalty_score <= 10:
            return 2  # Poor
        else:
            return 1  # Very poor

    def _generate_linting_summary(
        self,
        error_count: int,
        warning_count: int,
        info_count: int,
        suggestion_count: int,
    ) -> str:
        """Generate a human-readable summary of linting results."""
        total = error_count + warning_count + info_count + suggestion_count

        if total == 0:
            return "OpenAPI specification is clean with no issues found."

        parts = []
        if error_count > 0:
            parts.append(f"{error_count} error{'s' if error_count != 1 else ''}")
        if warning_count > 0:
            parts.append(f"{warning_count} warning{'s' if warning_count != 1 else ''}")
        if info_count > 0:
            parts.append(f"{info_count} info item{'s' if info_count != 1 else ''}")
        if suggestion_count > 0:
            parts.append(
                f"{suggestion_count} suggestion{'s' if suggestion_count != 1 else ''}"
            )

        summary = (
            f"Found {total} issue{'s' if total != 1 else ''}: " + ", ".join(parts) + "."
        )

        if error_count > 0:
            summary += (
                " Address errors first as they may prevent proper API functionality."
            )
        elif warning_count > 0:
            summary += " Consider addressing warnings to improve API quality."

        return summary


class EnhancementRequest(BaseModel):
    """Request model for OpenAPI enhancement."""

    spec_content: str = Field(
        description="OpenAPI specification content (JSON or YAML)"
    )
    spec_format: str = Field(
        default="yaml", description="Format of the spec (json or yaml)"
    )
    original_filename: Optional[str] = Field(
        default=None, description="Original filename"
    )
    original_url: Optional[str] = Field(
        default=None, description="Original URL if from web"
    )
    enhancement_level: str = Field(
        default="comprehensive", description="Level of enhancement"
    )

    model_config = {"str_strip_whitespace": True}


class EnhancementResult(BaseModel):
    """Result of OpenAPI enhancement process."""

    request_id: str = Field(description="Unique request identifier")
    evaluation: OpenAPIEvaluationResult = Field(description="Evaluation results")
    enhanced_spec: Optional[str] = Field(
        default=None, description="Enhanced specification"
    )
    enhancement_summary: Dict[str, Any] = Field(
        description="Summary of enhancements made"
    )
    processing_time_seconds: float = Field(description="Time taken for processing")

    model_config = {"str_strip_whitespace": True}


class OpenAPIEnhancer:
    """OpenAPI specification enhancer using LLM evaluation and enhancement."""

    def __init__(
        self,
        model: str = None,
        max_tokens: int = None,
        temperature: float = None,
        timeout_seconds: int = None,
        debug: bool = None,
    ):
        """Initialize the OpenAPI enhancer.

        Args:
            model: Model string (format: provider/model_name)
            max_tokens: Maximum tokens for responses
            temperature: Temperature for responses
            timeout_seconds: Timeout for requests
            debug: Whether to enable debug mode
        """
        try:
            self.llm_client = get_llm_client(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                debug=debug,
            )
            self.evaluation_template = self._load_evaluation_template()
            self.linter = OpenAPILinter()
            logger.info("Initialized OpenAPI enhancer with linting support")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAPI enhancer: {e}")
            raise

    def _load_evaluation_template(self) -> Template:
        """Load the evaluation prompt template."""
        try:
            templates_dir = config.get_path("templates_dir", "./templates")
            template_path = templates_dir / "evaluation_prompt.txt"
            if not template_path.exists():
                raise FileNotFoundError(
                    f"Evaluation template not found: {template_path}"
                )

            with open(template_path, "r", encoding="utf-8") as f:
                template_content = f.read()

            template = Template(template_content)
            logger.info(f"Loaded evaluation template from {template_path}")
            return template
        except Exception as e:
            logger.error(f"Failed to load evaluation template: {e}")
            raise

    def _parse_openapi_spec(
        self, content: str, format_hint: str = "yaml"
    ) -> Dict[str, Any]:
        """Parse OpenAPI specification from string content."""
        try:
            # Try to determine format from content
            content = content.strip()

            if format_hint.lower() == "json" or content.startswith("{"):
                # Parse as JSON
                spec_dict = json.loads(content)
                logger.info("Parsed OpenAPI spec as JSON")
            else:
                # Parse as YAML
                spec_dict = yaml.safe_load(content)
                logger.info("Parsed OpenAPI spec as YAML")

            # Validate it's an OpenAPI spec
            if "openapi" not in spec_dict and "swagger" not in spec_dict:
                raise ValueError(
                    "Content does not appear to be an OpenAPI specification"
                )

            return spec_dict
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse spec as JSON: {e}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse spec as YAML: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to parse OpenAPI spec: {e}")
            raise

    def _extract_spec_metadata(self, spec_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive metadata from OpenAPI specification."""
        try:
            info = spec_dict.get("info", {})
            metadata = {
                "openapi_version": spec_dict.get(
                    "openapi", spec_dict.get("swagger", "unknown")
                ),
                "api_title": info.get("title", "Unknown API"),
                "api_version": info.get("version", "unknown"),
                "description": info.get("description", ""),
                "contact": info.get("contact", {}),
                "license": info.get("license", {}),
                "servers": spec_dict.get("servers", []),
                "security": spec_dict.get("security", []),
                "has_contact": bool(info.get("contact")),
                "has_license": bool(info.get("license")),
                "has_description": bool(info.get("description")),
                "path_count": len(spec_dict.get("paths", {})),
                "schema_count": len(spec_dict.get("components", {}).get("schemas", {})),
            }
            logger.info(f"Extracted metadata for API: {metadata['api_title']}")
            logger.info(f"  - Paths: {metadata['path_count']}, Schemas: {metadata['schema_count']}")
            logger.info(f"  - Has contact: {metadata['has_contact']}, Has license: {metadata['has_license']}")
            return metadata
        except Exception as e:
            logger.error(f"Failed to extract spec metadata: {e}")
            raise

    async def evaluate_specification(
        self, request: EnhancementRequest
    ) -> OpenAPIEvaluationResult:
        """Evaluate an OpenAPI specification for completeness and AI readiness."""
        try:
            logger.info(
                f"Starting evaluation for {request.original_filename or 'spec'}"
            )

            # Parse the specification
            spec_dict = self._parse_openapi_spec(
                request.spec_content, request.spec_format
            )
            metadata = self._extract_spec_metadata(spec_dict)

            # Run linting analysis
            logger.info("Running OpenAPI linting analysis")
            linting_results = self.linter.lint_specification(spec_dict)
            logger.info(f"Linting completed: {linting_results.linting_summary}")

            # Add metadata checks to linting results if missing
            if not spec_dict.get("info", {}).get("contact"):
                logger.warning("API specification missing contact information")
            if not spec_dict.get("info", {}).get("license"):
                logger.warning("API specification missing license information")
            if not spec_dict.get("info", {}).get("description"):
                logger.warning("API specification missing description")

            # Prepare the evaluation prompt with metadata context
            evaluation_prompt = self.evaluation_template.render(
                openapi_spec=request.spec_content,
                metadata=metadata  # Pass metadata to template for context
            )

            # Send to LLM for evaluation
            max_tokens = config.get_int("max_tokens", 4096)
            temperature = config.get_float("temperature", 0.1)

            llm_request = LLMRequest(
                prompt=evaluation_prompt, max_tokens=max_tokens, temperature=temperature
            )

            logger.info(
                f"LLM inference parameters - max_tokens: {max_tokens}, temperature: {temperature}"
            )
            logger.info(f"Prompt length: {len(evaluation_prompt)} characters")

            response = await self.llm_client.generate_text(llm_request)

            # Save LLM response to temporary file for debugging
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", prefix="llm_response_", delete=False
            ) as temp_file:
                temp_file.write(response.text)
                temp_file_path = temp_file.name

            logger.info(f"LLM response saved to temporary file: {temp_file_path}")
            logger.info(f"Response length: {len(response.text)} characters")
            logger.info(f"To view the full response, run: cat {temp_file_path}")

            # Log token usage and cost information if available
            usage_info = {"tokens": 0, "cost": 0.0, "calls": 1}
            if hasattr(response, "usage") and response.usage:
                prompt_tokens = response.usage.get("prompt_tokens", 0)
                completion_tokens = response.usage.get("completion_tokens", 0)
                total_tokens = response.usage.get("total_tokens", 0)

                # Get cost information
                prompt_cost = response.usage.get("prompt_cost_usd", 0.0)
                completion_cost = response.usage.get("completion_cost_usd", 0.0)
                total_cost = response.usage.get("total_cost_usd", 0.0)

                usage_info.update({"tokens": total_tokens, "cost": total_cost})

                logger.info(
                    f"📊 Token usage - prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}"
                )

                if total_cost > 0:
                    logger.info(
                        f"💰 Cost breakdown - prompt: ${prompt_cost:.6f}, completion: ${completion_cost:.6f}, total: ${total_cost:.6f}"
                    )
                else:
                    logger.info("💰 Cost information not available from provider")

                # Check if response might be truncated
                if completion_tokens >= max_tokens * 0.95:
                    logger.warning(
                        f"⚠️ Response may be truncated! Completion tokens ({completion_tokens}) near max_tokens limit ({max_tokens})"
                    )
            else:
                logger.info("📊 Token usage information not available in response")

            # Parse the JSON response and extract enhanced spec
            try:
                response_text = response.text.strip()

                # Extract JSON content
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1

                if json_start == -1 or json_end == 0:
                    raise ValueError("No JSON found in response")

                json_content = response_text[json_start:json_end]
                logger.info(
                    f"Extracted JSON content from position {json_start} to {json_end}"
                )
                logger.info(f"JSON content length: {len(json_content)} characters")

                # Try to parse JSON with better error reporting
                try:
                    evaluation_data = json.loads(json_content)
                except json.JSONDecodeError as json_err:
                    logger.error(
                        f"JSON parsing failed at line {json_err.lineno}, column {json_err.colno}"
                    )
                    logger.error(f"JSON error message: {json_err.msg}")
                    logger.error(
                        f"JSON content around error (chars {max(0, json_err.pos-50)}:{json_err.pos+50}): {json_content[max(0, json_err.pos-50):json_err.pos+50]}"
                    )
                    raise

                # No longer extracting enhanced spec since we removed it from the prompt
                logger.info(
                    "Enhanced spec generation removed from prompt - focusing on evaluation only"
                )

                # Add missing fields to evaluation data
                model = config.get_str("model")
                evaluation_data.update(
                    {
                        "model": model,
                        "enhancement_level": request.enhancement_level,
                        "timestamp": datetime.now(),
                        "evaluation_timestamp": datetime.now(),
                    }
                )

                # Create evaluation result
                evaluation = OpenAPIEvaluationResult(**evaluation_data)

                # Add linting results to the evaluation
                evaluation.linting_results = linting_results

                # Add usage tracking
                evaluation.total_tokens = usage_info["tokens"]
                evaluation.total_cost_usd = usage_info["cost"]
                evaluation.llm_calls_count = usage_info["calls"]

                # Store detailed evaluation usage
                evaluation.evaluation_usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": usage_info["tokens"],
                    "total_cost_usd": usage_info["cost"],
                    "calls_count": usage_info["calls"],
                }
                logger.info(
                    f"Evaluation evaluation_usage: {evaluation.evaluation_usage}"
                )
                logger.info(f"Successfully evaluated spec: {evaluation.api_title}")
                logger.info(
                    f"Linting score: {linting_results.linting_score}/5 ({linting_results.total_issues} issues)"
                )
                return evaluation

            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Failed to parse evaluation response: {e}")
                logger.error(f"Full LLM response saved to: {temp_file_path}")
                logger.error(f"Response text preview: {response.text[:500]}...")
                raise ValueError(f"Invalid evaluation response format: {e}")

        except Exception as e:
            logger.error(f"Failed to evaluate specification: {e}")
            raise

    async def enhance_specification(
        self, request: EnhancementRequest
    ) -> EnhancementResult:
        """Enhance an OpenAPI specification based on evaluation results."""
        import time

        try:
            start_time = time.time()
            request_id = str(uuid.uuid4())

            logger.info(f"Starting enhancement process for request {request_id}")

            # First, evaluate the specification
            evaluation = await self.evaluate_specification(request)

            # Extract the enhanced specification from evaluation if provided
            enhanced_spec = getattr(evaluation, "enhanced_openapi_spec", None)
            processing_time = time.time() - start_time

            enhancement_summary = {
                "evaluation_completed": True,
                "enhancement_applied": bool(
                    enhanced_spec
                ),  # True if enhanced spec was provided
                "overall_quality": evaluation.overall.overall_quality.value,
                "completeness_score": evaluation.overall.completeness_score,
                "ai_readiness_score": evaluation.overall.ai_readiness_score,
                "major_improvements_identified": len(
                    evaluation.overall.major_improvements_needed
                ),
                "minor_improvements_identified": len(
                    evaluation.overall.minor_improvements_suggested
                ),
                "enhanced_spec_length": len(enhanced_spec) if enhanced_spec else 0,
                "linting_score": (
                    evaluation.linting_results.linting_score
                    if evaluation.linting_results
                    else None
                ),
                "linting_issues_total": (
                    evaluation.linting_results.total_issues
                    if evaluation.linting_results
                    else None
                ),
                "linting_errors": (
                    evaluation.linting_results.error_count
                    if evaluation.linting_results
                    else None
                ),
                "linting_warnings": (
                    evaluation.linting_results.warning_count
                    if evaluation.linting_results
                    else None
                ),
            }

            result = EnhancementResult(
                request_id=request_id,
                evaluation=evaluation,
                enhanced_spec=enhanced_spec,  # Include the enhanced spec from LLM
                enhancement_summary=enhancement_summary,
                processing_time_seconds=processing_time,
            )

            logger.info(
                f"Completed enhancement process for request {request_id} in {processing_time:.2f}s"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to enhance specification: {e}")
            raise

    def validate_openapi_spec(self, content: str, format_hint: str = "yaml") -> bool:
        """Validate if content is a valid OpenAPI specification."""
        try:
            spec_dict = self._parse_openapi_spec(content, format_hint)

            # Basic validation checks
            required_fields = ["info"]
            if "openapi" in spec_dict:
                # OpenAPI 3.x
                required_fields.extend(["paths"])
            elif "swagger" in spec_dict:
                # Swagger 2.x
                required_fields.extend(["paths"])
            else:
                return False

            for field in required_fields:
                if field not in spec_dict:
                    logger.warning(f"Missing required field: {field}")
                    return False

            logger.info("OpenAPI specification validation passed")
            return True

        except Exception as e:
            logger.error(f"OpenAPI specification validation failed: {e}")
            return False

    def get_enhancement_stats(self) -> Dict[str, Any]:
        """Get statistics about the enhancement service."""
        try:
            model_info = self.llm_client.get_model_info()
            model = config.get_str("model")
            max_tokens = config.get_int("max_tokens", 4096)
            temperature = config.get_float("temperature", 0.1)
            templates_dir = config.get_path("templates_dir", "./templates")

            stats = {
                "service_name": "OpenAPI Enhancer",
                "model": model,
                "provider": model_info.get("provider", "unknown"),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "templates_directory": str(templates_dir),
                "supported_formats": ["json", "yaml", "yml"],
                "enhancement_levels": ["basic", "comprehensive", "ai-optimized"],
            }

            logger.info("Retrieved enhancement service stats")
            return stats
        except Exception as e:
            logger.error(f"Failed to get enhancement stats: {e}")
            raise


if __name__ == "__main__":
    """Standalone testing of OpenAPI enhancer."""
    import asyncio

    async def test_openapi_enhancer():
        """Test the OpenAPI enhancer functionality."""
        logger.info("Testing OpenAPI enhancer...")

        try:
            # Initialize enhancer
            enhancer = OpenAPIEnhancer()

            # Get service stats
            stats = enhancer.get_enhancement_stats()
            logger.info(f"Service stats: {stats}")

            # Test with a simple OpenAPI spec
            simple_spec = """
openapi: 3.0.3
info:
  title: Test API
  version: 1.0.0
  description: A simple test API
paths:
  /users:
    get:
      summary: Get users
      responses:
        '200':
          description: List of users
components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: integer
        name:
          type: string
"""

            # Validate the spec
            is_valid = enhancer.validate_openapi_spec(simple_spec, "yaml")
            logger.info(f"Spec validation result: {is_valid}")

            if is_valid:
                # Create enhancement request
                request = EnhancementRequest(
                    spec_content=simple_spec,
                    spec_format="yaml",
                    original_filename="test_spec.yaml",
                    enhancement_level="comprehensive",
                )

                # Test evaluation (this will fail without AWS credentials)
                try:
                    result = await enhancer.enhance_specification(request)
                    logger.info(f"Enhancement completed: {result.enhancement_summary}")
                except Exception as e:
                    logger.warning(
                        f"Enhancement failed (expected without AWS credentials): {e}"
                    )

            logger.info("OpenAPI enhancer test completed")

        except Exception as e:
            logger.error(f"OpenAPI enhancer test failed: {e}")
            raise

    # Run the test
    asyncio.run(test_openapi_enhancer())
