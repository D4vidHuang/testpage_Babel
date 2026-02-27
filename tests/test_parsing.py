"""Tests for the unified ResponseParser."""

import json
from pathlib import Path

from core.parsing import ResponseParser, ModelEvaluation, ParseResult


class TestDirectParsing:
    """Test direct JSON parsing."""

    def test_standard_format(self):
        """Parse standard evaluations format."""
        response = json.dumps({
            "evaluations": [
                {"model_name": "model_a", "errors": ["SE-MD", "LG-GR1"]},
                {"model_name": "model_b", "errors": []},
            ]
        })
        result = ResponseParser.parse(response)
        
        assert result.success
        assert result.parse_method == "direct"
        assert len(result.evaluations) == 2
        assert result.evaluations[0].errors == ["SE-MD", "LG-GR1"]

    def test_results_key(self):
        """Parse with 'results' key instead of 'evaluations'."""
        response = json.dumps({
            "results": [{"model_name": "x", "errors": ["SE-HA"]}]
        })
        result = ResponseParser.parse(response)
        
        assert result.success
        assert result.evaluations[0].errors == ["SE-HA"]

    def test_direct_list(self):
        """Parse direct list of evaluations."""
        response = json.dumps([
            {"model_name": "m1", "errors": ["SE-MD"]},
            {"model_name": "m2", "errors": ["LG-GR2"]}
        ])
        result = ResponseParser.parse(response)
        
        assert result.success
        assert len(result.evaluations) == 2


class TestCodeFenceParsing:
    """Test JSON extraction from markdown code fences."""

    def test_json_in_code_fence(self):
        """Extract JSON from ```json ``` blocks."""
        response = '''Here is my analysis:

```json
{"evaluations": [{"model_name": "test", "errors": ["SE-MD"]}]}
```

The above shows errors.
'''
        result = ResponseParser.parse(response)
        
        assert result.success
        assert result.parse_method == "code_fence"
        assert result.evaluations[0].errors == ["SE-MD"]

    def test_plain_code_fence(self):
        """Extract JSON from ``` ``` blocks without json tag."""
        response = '''Analysis:

```
{"evaluations": [{"model_name": "x", "errors": ["LG-GR1"]}]}
```
'''
        result = ResponseParser.parse(response)
        
        assert result.success
        assert result.evaluations[0].errors == ["LG-GR1"]


class TestRubricFormat:
    """Test PRESENT/ABSENT rubric format parsing."""

    def test_rubric_present_absent(self):
        """Convert PRESENT/ABSENT dict to error list."""
        response = json.dumps({
            "model_predictions": [{
                "model_name": "qwen",
                "errors": {
                    "SE-MD": "PRESENT",
                    "LG-GR1": "ABSENT",
                    "SE-HA": "PRESENT"
                }
            }]
        })
        result = ResponseParser.parse(response)
        
        assert result.success
        assert set(result.evaluations[0].errors) == {"SE-MD", "SE-HA"}

    def test_case_insensitive(self):
        """PRESENT matching should be case-insensitive."""
        response = json.dumps({
            "model_predictions": [{
                "model_name": "test",
                "errors": {"SE-MD": "present", "LG-GR1": "Present"}
            }]
        })
        result = ResponseParser.parse(response)
        
        assert set(result.evaluations[0].errors) == {"SE-MD", "LG-GR1"}


class TestCoTFormat:
    """Test Chain-of-Thought format with error objects."""

    def test_cot_error_objects(self):
        """Parse errors as objects with error_id."""
        response = json.dumps({
            "evaluations": [{
                "model_name": "cot_model",
                "reasoning": "Analyzed the code...",
                "errors": [
                    {"error_id": "SE-MD", "confidence": 0.9},
                    {"error_id": "LG-GR1", "confidence": 0.7}
                ]
            }]
        })
        result = ResponseParser.parse(response)
        
        assert result.success
        assert set(result.evaluations[0].errors) == {"SE-MD", "LG-GR1"}
        assert result.evaluations[0].reasoning == "Analyzed the code..."

    def test_mixed_error_formats(self):
        """Handle mix of string and object errors."""
        response = json.dumps({
            "evaluations": [{
                "model_name": "mixed",
                "errors": ["SE-MD", {"error_id": "LG-GR1"}, "SE-HA"]
            }]
        })
        result = ResponseParser.parse(response)
        
        assert set(result.evaluations[0].errors) == {"SE-MD", "LG-GR1", "SE-HA"}


class TestBalancedBraces:
    """Test JSON extraction from embedded text."""

    def test_json_in_text(self):
        """Find JSON embedded in explanatory text."""
        response = 'Results: {"evaluations": [{"model_name": "x", "errors": ["SE-MD"]}]} end.'
        result = ResponseParser.parse(response)
        
        assert result.success
        assert result.parse_method == "balanced_braces"


class TestEdgeCases:
    """Test error handling and edge cases."""

    def test_empty_response(self):
        """Handle empty response."""
        result = ResponseParser.parse("")
        
        assert not result.success
        assert len(result.warnings) > 0

    def test_none_response(self):
        """Handle None response."""
        result = ResponseParser.parse(None)
        
        assert not result.success

    def test_no_json(self):
        """Handle response with no JSON."""
        result = ResponseParser.parse("Just plain text.")
        
        assert not result.success
        assert result.parse_method == "failed"

    def test_empty_errors(self):
        """Handle empty errors list."""
        response = json.dumps({
            "evaluations": [{"model_name": "x", "errors": []}]
        })
        result = ResponseParser.parse(response)
        
        assert result.success
        assert result.evaluations[0].errors == []

    def test_to_dict_serializable(self):
        """ParseResult.to_dict() should be JSON serializable."""
        response = json.dumps({
            "evaluations": [{"model_name": "x", "errors": ["SE-MD"]}]
        })
        result = ResponseParser.parse(response)
        
        # Should not raise
        json.dumps(result.to_dict())

    def test_overall_quality_field(self):
        """Parse overall_quality field from judge response."""
        response = json.dumps({
            "evaluations": [
                {
                    "model_name": "model_a",
                    "errors": ["SE-MD"],
                    "explanation": "Minor issue found.",
                    "overall_quality": "partially_correct"
                },
                {
                    "model_name": "model_b",
                    "errors": [],
                    "explanation": "No issues found.",
                    "overall_quality": "correct"
                }
            ]
        })
        result = ResponseParser.parse(response)
        
        assert result.success
        assert len(result.evaluations) == 2
        assert result.evaluations[0].overall_quality == "partially_correct"
        assert result.evaluations[1].overall_quality == "correct"
        
        # Check it's included in serialization
        result_dict = result.to_dict()
        assert result_dict["evaluations"][0]["overall_quality"] == "partially_correct"
        assert result_dict["evaluations"][1]["overall_quality"] == "correct"
