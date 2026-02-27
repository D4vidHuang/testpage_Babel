"""Service for loading and exporting grouped judge data from JSON files."""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple, Optional

from .parsing import (
    ResponseParser, 
    extract_evaluations_from_judge_output,
    extract_tokens_from_judge_output
)


def extract_tokens(data: Any) -> Dict[str, int]:
    """Recursively search for 'usage' fields and sum up tokens.
    
    Supports both standard API usage formats and Ollama specific formats.
    """
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    if isinstance(data, dict):
        if "usage" in data and isinstance(data["usage"], dict):
            usage = data["usage"]
            totals["prompt_tokens"] += usage.get("prompt_tokens", 0)
            totals["completion_tokens"] += usage.get("completion_tokens", 0)
            totals["total_tokens"] += usage.get("total_tokens", 0)
        elif "prompt_eval_count" in data:
            # Ollama support
            totals["prompt_tokens"] += data.get("prompt_eval_count", 0)
            totals["completion_tokens"] += data.get("eval_count", 0)
            totals["total_tokens"] += (data.get("prompt_eval_count", 0) + data.get("eval_count", 0))
        
        for key, value in data.items():
            if key != "usage": # Avoid double counting if usage is nested
                sub_totals = extract_tokens(value)
                for k in totals:
                    totals[k] += sub_totals[k]
                    
    elif isinstance(data, list):
        for item in data:
            sub_totals = extract_tokens(item)
            for k in totals:
                totals[k] += sub_totals[k]
                
    return totals


class DataLoader:
    """Loads and exports grouped instances from/to JSON files."""

    @staticmethod
    def _extract_raw_and_parsed_response(
        judge_eval: Dict[str, Any]
    ) -> Tuple[Optional[str], Any]:
        """Extract raw response content and parsed JSON from judge evaluation.

        Args:
            judge_eval: Dictionary from manager.run() with {config_name: {model_name: api_response}}

        Returns:
            Tuple of (raw_content_string, parsed_json_or_string)
        """
        raw_response, parse_result = extract_evaluations_from_judge_output(judge_eval)
        
        if parse_result.success:
            # Return normalized structure
            return raw_response, parse_result.to_dict()
        elif raw_response:
            # Fallback: return raw response for debugging
            return raw_response, raw_response
        else:
            return None, None

    @staticmethod
    def load_from_json(file_path: str) -> List[Dict[str, Any]]:
        """Load grouped instances from a JSON file.

        Args:
            file_path: Path to the JSON file containing grouped instances.

        Returns:
            List of instance dictionaries with metadata, code_context, and model_predictions.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
            data = data.replace("_x000D_", "")  # Clean up any unwanted characters
            return json.loads(data)

    @staticmethod
    def get_next_output_folder(base_output_dir: Path = None) -> Path:
        """Get the next incremented output folder path.

        Args:
            base_output_dir: Base output directory. If None, uses './output'

        Returns:
            Path to the next numbered folder (e.g., output_1, output_2, etc.)
        """
        if base_output_dir is None:
            base_output_dir = Path.cwd() / "output"

        base_output_dir.mkdir(parents=True, exist_ok=True)

        # Find the highest existing output_N folder
        existing_folders = [d for d in base_output_dir.glob("output_*") if d.is_dir()]

        if not existing_folders:
            next_num = 1
        else:
            # Extract numbers and find the max
            numbers = []
            for folder in existing_folders:
                try:
                    num = int(folder.name.split("_")[1])
                    numbers.append(num)
                except (IndexError, ValueError):
                    continue
            next_num = max(numbers) + 1 if numbers else 1

        output_folder = base_output_dir / f"output_{next_num}"
        output_folder.mkdir(parents=True, exist_ok=True)

        return output_folder

    @staticmethod
    def extract_json_from_response(response: str) -> Union[Dict[str, Any], str]:
        """Extract valid JSON object/array from model response text.

        DEPRECATED: Use parsing.ResponseParser.parse() directly for full functionality.
        This method is kept for backward compatibility.

        Args:
            response: Model response text that may contain JSON

        Returns:
            Parsed JSON object/array if found, otherwise returns original string
        """
        if not isinstance(response, str):
            return response

        result = ResponseParser.parse(response)
        
        # If parsing succeeded, return the normalized dict
        if result.success:
            return result.to_dict()
        
        # Try to extract just the raw JSON structure (for non-evaluation responses)
        json_obj, _ = ResponseParser._extract_json(response)
        if json_obj is not None:
            return json_obj
            
        return response

    @staticmethod
    def export_judge_results_v1(
        results: Dict[str, Dict[str, Any]],
        output_folder: Path,
        include_raw_response: bool = True
    ) -> Path:
        """Export judge evaluation results to JSON format (Workflow 4 format).

        Stores results with preserved judge response format:
        - instance_id, metadata, code_context, model_predictions from input
        - judge_evaluations with the exact JSON format returned by the model (unescaped)
        - raw_response (optional) with the unprocessed API response content

        Format: Standard grouped evaluation with errors array per model.

        Args:
            results: Dictionary with instance_id -> {instance, judge_evaluation}
            output_folder: Path to export results to
            include_raw_response: Whether to include raw API response content (default True)

        Returns:
            Path to the exported JSON file
        """
        export_data = []

        for instance_id, result in results.items():
            instance = result["instance"]
            judge_eval = result["judge_evaluation"]

            # Extract raw and parsed response
            raw_response, parsed_response = DataLoader._extract_raw_and_parsed_response(
                judge_eval
            )
            
            # Extract token usage
            tokens = extract_tokens_from_judge_output(judge_eval)

            # Combine instance data with judge evaluations
            export_item = {
                "instance_id": instance_id,
                "metadata": instance.get("metadata", {}),
                "code_context": instance.get("code_context", {}),
                "model_predictions": instance.get("model_predictions", []),
                "judge_evaluations": parsed_response,
                "usage": tokens
            }

            if include_raw_response and raw_response is not None:
                export_item["raw_response"] = raw_response

            export_data.append(export_item)

        # Save to JSON file
        output_file = output_folder / "judge_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(export_data, indent=2, ensure_ascii=False))

        return output_file

    @staticmethod
    def export_judge_results_hierarchical(
        results: Dict[str, Dict[str, Any]],
        output_folder: Path,
        include_cluster_details: bool = True
    ) -> Path:
        """Export hierarchical evaluation results with cluster details preserved.

        Consolidates cluster evaluations for scorer compatibility while preserving
        detailed cluster breakdown for analysis. Handles both standard format
        (explanation field) and CoT format (reasoning field).

        Format: Hierarchical evaluation with cluster-level details (Workflow 6 & 8).

        Args:
            results: Dictionary with instance_id -> {instance, hierarchical_evaluation, cluster_details, ...}
            output_folder: Path to export results to
            include_cluster_details: Whether to include cluster details field (default True)

        Returns:
            Path to the exported JSON file
        """
        export_data = []

        for instance_id, result in results.items():
            instance = result["instance"]
            hierarchical_eval = result.get("hierarchical_evaluation", [])
            cluster_details = result.get("cluster_details", [])
            cluster_errors = result.get("cluster_errors", [])

            # Consolidate evaluations by model (for scorer compatibility)
            consolidated = {}
            for eval_item in hierarchical_eval:
                if isinstance(eval_item, dict):
                    model_name = eval_item.get("model_name")
                    if not model_name:
                        continue

                    if model_name not in consolidated:
                        consolidated[model_name] = {
                            "errors": set(),
                            "explanations": [],
                            "reasonings": []
                        }

                    # Merge errors (use set to avoid duplicates)
                    errors = eval_item.get("errors", [])
                    if isinstance(errors, list):
                        consolidated[model_name]["errors"].update(errors)

                    # Collect explanations (standard format)
                    explanation = eval_item.get("explanation", "")
                    if explanation:
                        consolidated[model_name]["explanations"].append(explanation)

                    # Collect reasoning (CoT format)
                    reasoning = eval_item.get("reasoning", "")
                    if reasoning:
                        consolidated[model_name]["reasonings"].append(reasoning)

            # Format as standard judge_evaluations structure
            evaluations = []
            for model_name, data in consolidated.items():
                eval_entry = {
                    "model_name": model_name,
                    "errors": sorted(list(data["errors"])),
                }
                # Include explanation if present (join all cluster explanations)
                if data["explanations"]:
                    eval_entry["explanation"] = "\n\n---\n\n".join(data["explanations"])
                else:
                    eval_entry["explanation"] = ""

                # Include reasoning if present (CoT workflows)
                if data["reasonings"]:
                    eval_entry["reasoning"] = "\n\n---\n\n".join(data["reasonings"])

                evaluations.append(eval_entry)

            export_item = {
                "instance_id": instance_id,
                "metadata": instance.get("metadata", {}),
                "code_context": instance.get("code_context", {}),
                "model_predictions": instance.get("model_predictions", []),
                "judge_evaluations": {
                    "evaluations": evaluations
                }
            }

            # Preserve cluster details for analysis (optional)
            if include_cluster_details and cluster_details:
                export_item["cluster_details"] = cluster_details

            # Include cluster errors if any (for debugging)
            if cluster_errors:
                export_item["cluster_errors"] = cluster_errors

            export_data.append(export_item)

        output_file = output_folder / "judge_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return output_file

    # Backward compatibility aliases
    export_judge_results = export_judge_results_v1
    export_judge_results_v2_cot = export_judge_results_v1
    export_judge_results_v3_rubric = export_judge_results_v1


