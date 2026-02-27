"""Unified response parsing for LLM judge outputs.

This module is the SINGLE SOURCE OF TRUTH for parsing LLM responses into
normalized evaluation structures. It handles all known output formats:

1. Standard format: {"evaluations": [{"model_name": "...", "errors": ["SE-MD", ...]}]}
2. CoT format: {"evaluations": [{"model_name": "...", "errors": [{"error_id": "SE-MD", ...}]}]}
3. Rubric format: {"model_predictions": [{"model_name": "...", "errors": {"SE-MD": "PRESENT", ...}}]}
4. Hierarchical format: Multiple cluster results to be merged
5. Multilingual format: Translated field names mapped back to English

All other modules should import and use this parser instead of implementing their own.
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple


# =============================================================================
# Multilingual Field Mappings
# =============================================================================
# Maps translated field names back to English canonical keys.
# Each language maps its translated key -> English key.

FIELD_MAPPINGS: Dict[str, Dict[str, str]] = {
    "polish": {
        "ewaluacje": "evaluations",
        "nazwa_modelu": "model_name",
        "bledy": "errors",
        "id_bledu": "error_id",
        "pewnosc": "confidence",
        "uzasadnienie": "justification",
        "wyjasnienie": "explanation",
        "rozumowanie": "reasoning",
        "ogolna_jakosc": "overall_quality",
        "poprawne": "correct",
        "czesciowo_poprawne": "partially_correct",
        "niepoprawne": "incorrect",
    },
    "dutch": {
        "evaluaties": "evaluations",
        "modelnaam": "model_name",
        "fouten": "errors",
        "fout_id": "error_id",
        "zekerheid": "confidence",
        "rechtvaardiging": "justification",
        "uitleg": "explanation",
        "redenering": "reasoning",
        "algehele_kwaliteit": "overall_quality",
        "correct": "correct",
        "gedeeltelijk_correct": "partially_correct",
        "incorrect": "incorrect",
    },
    "chinese": {
        "评估结果": "evaluations",
        "模型名称": "model_name",
        "错误": "errors",
        "错误标识": "error_id",
        "置信度": "confidence",
        "理由": "justification",
        "解释": "explanation",
        "推理": "reasoning",
        "整体质量": "overall_quality",
        "正确": "correct",
        "部分正确": "partially_correct",
        "不正确": "incorrect",
    },
    "greek": {
        "αξιολογησεις": "evaluations",
        "ονομα_μοντελου": "model_name",
        "σφαλματα": "errors",
        "αναγνωριστικο_σφαλματος": "error_id",
        "εμπιστοσυνη": "confidence",
        "αιτιολογηση": "justification",
        "εξηγηση": "explanation",
        "συλλογισμος": "reasoning",
        "συνολικη_ποιοτητα": "overall_quality",
        "σωστο": "correct",
        "μερικως_σωστο": "partially_correct",
        "λαθος": "incorrect",
    },
}

# Quality value mappings - derived from FIELD_MAPPINGS for efficiency
# Used to translate string values (not keys) like "poprawne" -> "correct"
QUALITY_MAPPINGS: Dict[str, str] = {}
for lang_map in FIELD_MAPPINGS.values():
    for k, v in lang_map.items():
        if v in ("correct", "partially_correct", "incorrect"):
            QUALITY_MAPPINGS[k] = v


@dataclass
class ModelEvaluation:
    """Normalized evaluation result for a single model."""
    model_name: str
    errors: List[str]  # Always a list of error IDs (e.g., ["SE-MD", "LG-GR1"])
    explanation: str = ""
    reasoning: str = ""  # For CoT workflows
    confidence: Optional[float] = None
    overall_quality: str = ""  # Expert accuracy: correct, partially_correct, incorrect


@dataclass
class ParseResult:
    """Result of parsing an LLM response."""
    evaluations: List[ModelEvaluation]
    raw_response: str
    parse_method: str  # Which extraction strategy succeeded
    warnings: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True if parsing produced at least one evaluation."""
        return len(self.evaluations) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "evaluations": [
                {
                    "model_name": e.model_name,
                    "errors": e.errors,
                    "explanation": e.explanation,
                    "reasoning": e.reasoning,
                    "overall_quality": e.overall_quality,
                }
                for e in self.evaluations
            ],
            "parse_method": self.parse_method,
            "warnings": self.warnings,
        }


class ResponseParser:
    """Single source of truth for parsing LLM judge responses.
    
    Usage:
        result = ResponseParser.parse(raw_llm_response)
        if result.success:
            for eval in result.evaluations:
                print(f"{eval.model_name}: {eval.errors}")
        else:
            print(f"Parse failed: {result.warnings}")
    """

    @staticmethod
    def parse(raw_response: str) -> ParseResult:
        """Parse any LLM response format into normalized evaluations.
        
        Args:
            raw_response: Raw text from LLM response (may include markdown, explanations, etc.)
            
        Returns:
            ParseResult with normalized evaluations and metadata
        """
        if not raw_response or not isinstance(raw_response, str):
            return ParseResult(
                evaluations=[],
                raw_response=raw_response or "",
                parse_method="empty_input",
                warnings=["Empty or non-string response"],
            )

        warnings: List[str] = []

        # Step 1: Extract JSON from response
        json_obj, method = ResponseParser._extract_json(raw_response)

        if json_obj is None:
            return ParseResult(
                evaluations=[],
                raw_response=raw_response,
                parse_method="failed",
                warnings=["Could not extract valid JSON from response"],
            )

        # Step 2: Normalize to standard format
        evaluations = ResponseParser._normalize_evaluations(json_obj, warnings)

        return ParseResult(
            evaluations=evaluations,
            raw_response=raw_response,
            parse_method=method,
            warnings=warnings,
        )

    @staticmethod
    def _extract_json(text: str) -> Tuple[Optional[Any], str]:
        """Try multiple strategies to extract JSON from text.
        
        Returns:
            Tuple of (parsed_json_or_none, method_name)
        """
        # Strategy 1: Direct parse (response is pure JSON)
        try:
            return json.loads(text.strip()), "direct"
        except json.JSONDecodeError:
            pass

        # Strategy 2: Code fence extraction (```json ... ```)
        fence_pattern = re.compile(
            r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```",
            re.IGNORECASE
        )
        for match in fence_pattern.finditer(text):
            try:
                return json.loads(match.group(1)), "code_fence"
            except json.JSONDecodeError:
                continue

        # Strategy 3: Find balanced braces (handles JSON embedded in text)
        result = ResponseParser._find_balanced_json(text)
        if result is not None:
            return result, "balanced_braces"

        return None, "none"

    @staticmethod
    def _find_balanced_json(text: str) -> Optional[Any]:
        """Find and parse the first balanced JSON object or array in text.
        
        Uses stack-based parsing that handles quoted strings correctly.
        """
        n = len(text)
        i = 0
        
        while i < n:
            if text[i] not in "{[":
                i += 1
                continue

            start = i
            open_char = text[i]
            close_char = "}" if open_char == "{" else "]"
            stack = [open_char]
            i += 1
            in_string = False
            escape = False

            while i < n and stack:
                c = text[i]
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_string = not in_string
                elif not in_string:
                    if c == "{":
                        stack.append("{")
                    elif c == "[":
                        stack.append("[")
                    elif c == "}":
                        if stack and stack[-1] == "{":
                            stack.pop()
                        else:
                            break
                    elif c == "]":
                        if stack and stack[-1] == "[":
                            stack.pop()
                        else:
                            break
                i += 1

            if not stack:
                candidate = text[start:i]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # Continue searching
                    pass
            
            # Move past start position if this attempt failed
            i = start + 1

        return None

    @staticmethod
    def _translate_keys(data: Any) -> Any:
        """Recursively translate non-English field names to English.
        
        Handles nested dicts and lists, translating keys using FIELD_MAPPINGS.
        """
        if isinstance(data, dict):
            translated = {}
            for key, value in data.items():
                # Try to find English key from any language mapping
                english_key = key
                for lang_mapping in FIELD_MAPPINGS.values():
                    if key in lang_mapping:
                        english_key = lang_mapping[key]
                        break
                translated[english_key] = ResponseParser._translate_keys(value)
            return translated
        elif isinstance(data, list):
            return [ResponseParser._translate_keys(item) for item in data]
        elif isinstance(data, str):
            # Translate quality values
            return QUALITY_MAPPINGS.get(data, data)
        return data

    @staticmethod
    def _normalize_evaluations(data: Any, warnings: List[str]) -> List[ModelEvaluation]:
        """Convert any known format to list of ModelEvaluation.
        
        Handles:
        - {"evaluations": [...]}
        - {"results": [...]}
        - {"model_predictions": [...]}
        - [...]  (direct list)
        - Multilingual field names (translated to English)
        """
        # First, translate any non-English keys to English
        data = ResponseParser._translate_keys(data)
        
        evaluations: List[ModelEvaluation] = []

        # Find the evaluations list from various possible keys
        raw_evals = None
        if isinstance(data, dict):
            # Try common keys in order of preference
            for key in ("evaluations", "results", "model_predictions", "evaluation_results"):
                if key in data:
                    raw_evals = data[key]
                    break
        elif isinstance(data, list):
            raw_evals = data

        if not raw_evals or not isinstance(raw_evals, list):
            warnings.append(f"No evaluations list found in structure (type: {type(data).__name__})")
            return evaluations

        for item in raw_evals:
            if not isinstance(item, dict):
                continue

            model_name = item.get("model_name")
            if not model_name:
                warnings.append("Evaluation item missing model_name")
                continue

            errors = ResponseParser._normalize_errors(item.get("errors", []))
            
            evaluations.append(ModelEvaluation(
                model_name=model_name,
                errors=errors,
                explanation=item.get("explanation", ""),
                reasoning=item.get("reasoning", ""),
                confidence=item.get("confidence"),
                overall_quality=item.get("overall_quality", ""),
            ))

        return evaluations

    @staticmethod
    def _normalize_errors(errors: Any) -> List[str]:
        """Convert any error format to list of error ID strings.
        
        Handles:
        - List of strings: ["SE-MD", "LG-GR1"]
        - List of CoT objects: [{"error_id": "SE-MD", "confidence": 0.9}, ...]
        - Rubric dict: {"SE-MD": "PRESENT", "LG-GR1": "ABSENT", ...}
        - Multilingual field names (already translated by _translate_keys)
        """
        if errors is None:
            return []

        # Already a list
        if isinstance(errors, list):
            result = []
            for e in errors:
                if isinstance(e, str):
                    # Standard format: string error ID
                    result.append(e)
                elif isinstance(e, dict):
                    # CoT format: {"error_id": "SE-MD", ...}
                    # Keys should already be translated to English
                    error_id = e.get("error_id") or e.get("id")
                    if error_id:
                        result.append(error_id)
            return result

        # Rubric format: {"SE-MD": "PRESENT", "LG-GR1": "ABSENT"}
        if isinstance(errors, dict):
            return [
                error_id for error_id, status in errors.items()
                if isinstance(status, str) and status.upper() == "PRESENT"
            ]

        return []


def extract_evaluations_from_judge_output(
    judge_eval: Dict[str, Any]
) -> Tuple[Optional[str], ParseResult]:
    """Extract and parse evaluations from JudgeManager output.
    
    The JudgeManager returns: {config_name: {model_name: api_response}}
    This function extracts the content and parses it.
    
    Args:
        judge_eval: Output from manager.run()
        
    Returns:
        Tuple of (raw_response_string, ParseResult)
    """
    raw_response = None
    
    # Extract raw content from nested API response structure
    for config_name, model_results in judge_eval.items():
        for model_name, output in model_results.items():
            if "choices" in output and len(output["choices"]) > 0:
                raw_response = output["choices"][0]["message"]["content"]
                break
        if raw_response:
            break

    if raw_response is None:
        return None, ParseResult(
            evaluations=[],
            raw_response="",
            parse_method="no_content",
            warnings=["No content found in API response"],
        )

    return raw_response, ResponseParser.parse(raw_response)


def extract_tokens_from_judge_output(judge_eval: Dict[str, Any]) -> Dict[str, int]:
    """Extract token usage from JudgeManager output.
    
    The JudgeManager returns: {config_name: {model_name: api_response}}
    
    Returns:
        Dict with 'prompt_tokens', 'thinking_tokens', 'completion_tokens', 'total_tokens'
    """
    totals = {"prompt_tokens": 0, "thinking_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for config_name, model_results in judge_eval.items():
        for model_name, output in model_results.items():
            if not isinstance(output, dict):
                continue
            if "usage" in output:
                usage = output["usage"]
                totals["prompt_tokens"] += usage.get("prompt_tokens", 0)
                totals["thinking_tokens"] += usage.get("thinking_tokens", 0)  # For models with thinking/reasoning tokens
                totals["completion_tokens"] += usage.get("completion_tokens", 0)
                totals["total_tokens"] += usage.get("total_tokens", 0)
            elif "prompt_eval_count" in output:
                # Ollama support
                totals["prompt_tokens"] += output.get("prompt_eval_count", 0)
                totals["completion_tokens"] += output.get("eval_count", 0)
                totals["total_tokens"] += (output.get("prompt_eval_count", 0) + output.get("eval_count", 0))
    return totals
