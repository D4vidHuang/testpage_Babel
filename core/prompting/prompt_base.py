class PromptBase:
    """Language-agnostic prompt utilities.
    
    These methods work identically across all languages:
    - JSON formatting
    - Comment reconstruction
    - Taxonomy filtering
    - Error clustering
    - Translated taxonomy loading
    """
    
    # Language code for this prompt library (override in subclasses)
    LANGUAGE_CODE = "en"
    
    @classmethod
    def load_taxonomy(cls) -> dict:
        """Load taxonomy for this language.
        
        Loads the translated taxonomy file if available, otherwise falls back to English.
        """
        import json
        from pathlib import Path
        
        base_path = Path(__file__).parent.parent.parent / "taxonomy"
        
        # Try language-specific file first
        lang_file = base_path / f"error_taxonomy_{cls.LANGUAGE_CODE}.json"
        if lang_file.exists():
            with open(lang_file, encoding="utf-8") as f:
                return json.load(f)
        
        # Fallback to English
        with open(base_path / "error_taxonomy.json", encoding="utf-8") as f:
            return json.load(f)
    
    @classmethod
    def get_taxonomy_json(cls) -> str:
        """Get taxonomy as formatted JSON string."""
        import json
        return json.dumps(cls.load_taxonomy(), indent=2, ensure_ascii=False)
    
    @staticmethod
    def get_output_field_names() -> dict:
        """Return translated JSON field names for output format.

        Override in language-specific subclasses.
        Returns mapping of semantic key -> translated field name.
        """
        return {
            "evaluations": "evaluations",
            "model_name": "model_name",
            "errors": "errors",
            "error_id": "error_id",
            "confidence": "confidence",
            "justification": "justification",
            "explanation": "explanation",
            "reasoning": "reasoning",
            "overall_quality": "overall_quality",
            # Quality values
            "correct": "correct",
            "partially_correct": "partially_correct",
            "incorrect": "incorrect",
        }

    @staticmethod
    def get_rubric_labels() -> dict:
        """Return translated labels for rubric formatting.

        Override in language-specific subclasses.
        Returns mapping of semantic key -> translated label.
        """
        return {
            "rubric_title": "Error Classification Rubric",
            "definition": "Definition",
            "mark_present": "Mark as PRESENT if",
            "mark_absent": "Mark as ABSENT if",
        }
    
    @staticmethod
    def format_grouped_instance(instance: dict, model_names: list = None) -> str:
        """Format instance as JSON for judge evaluation.

        Args:
            instance: A grouped instance dict with code_context and model_predictions
            model_names: Optional list to filter which models to include

        Returns:
            JSON string with code context and filtered model predictions
        """
        import json
        import re

        code_ctx = instance.get("code_context", {})
        model_predictions = instance.get("model_predictions", [])
        original_comment = code_ctx.get("original_comment", "")

        # Filter models if specified
        if model_names:
            model_predictions = [m for m in model_predictions if m.get("model_name") in model_names]

        # Build structured JSON for judge

        # This ensures we can use the same cpntext for multiple tokenizers 
        context = code_ctx.get("source_code", "")
        context = context.replace(original_comment, "<ORIGINAL_COMMENT_LOCATION>")

        judge_input = {
            "source_code": context,
            "original_comment": original_comment,
            "model_predictions": [
                {
                    "model_name": pred.get("model_name", "Unknown"),
                    "predicted_comment": PromptBase._reconstruct_full_comment(
                        pred.get("masked_code", ""),
                        pred.get("predicted_comment", ""),
                        original_comment
                    ),
                }
                for pred in model_predictions
            ]
        }

        return json.dumps(judge_input, ensure_ascii=False)

        
    @staticmethod
    def _reconstruct_full_comment(masked_code: str, predicted: str, original: str) -> str:
        """Reconstruct the full predicted comment from masked_code prefix + prediction.
        
        The masked_code contains the code with the comment partially masked.
        The predicted_comment is only the generated portion after the mask.
        We need to find where the comment starts before the FIM suffix token
        and concatenate that prefix with the predicted_comment.
        
        FIM suffix tokens by model:
        - <fim_suffix> : Qwen, Starcoder, Granite
        - <|fim_suffix|> : CodeGemma  
        - <SUF> : CodeLlama
        """
        import re
        
        # Find the FIM suffix token position
        suffix_pattern = r'<\|?fim_suffix\|?>| <SUF>'
        match = re.search(suffix_pattern, masked_code)
        
        if not match:
            # No FIM token found, return predicted as-is
            return predicted
        
        # Get everything before the suffix token
        before_suffix = masked_code[:match.start()]
        
        # Find the last comment start marker (// or /* or /**) in the prefix
        # This locates where the current comment begins
        comment_start_match = None
        for m in re.finditer(r'//|/\*\*?', before_suffix):
            comment_start_match = m  # Keep the last match
        
        if comment_start_match:
            # Extract from comment start to end of prefix
            comment_prefix = before_suffix[comment_start_match.start():]
            return comment_prefix + predicted
        
        # Fallback: return predicted as-is
        return predicted

        
    @classmethod
    def format_taxonomy_as_rubric(cls, taxonomy_data: dict) -> str:
        """Format taxonomy as rubric with PRESENT/ABSENT criteria.

        Converts raw taxonomy JSON into a structured rubric format that makes
        inclusion/exclusion criteria more prominent and easier to apply.

        Args:
            taxonomy_data: Dictionary loaded from taxonomy/error_taxonomy.json

        Returns:
            Formatted rubric string with clear decision criteria

        Example output format:
            # Error Classification Rubric

            ## Grammar Errors

            ### [LG-GR1] Subject-Verb Agreement
            **Mark as PRESENT if:**
              - Subject and verb do not agree in number
              - Example: "The function return..." (should be "returns")

            **Mark as ABSENT if:**
              - Code identifiers are involved (e.g., variable names)
              - Technical terminology follows domain conventions
        """
        labels = cls.get_rubric_labels()
        lines = [f"# {labels['rubric_title']}\\n"]

        for category, errors in taxonomy_data.items():
            if not category or not errors:
                continue
            lines.append(f"\\n## {category}\\n")

            for error in errors:
                error_id = error.get("id", "")
                error_name = error.get("name", "")
                description = error.get("description", "").strip()
                inclusion = error.get("inclusion", "").strip()
                exclusion = error.get("exclusion", "").strip()

                lines.append(f"### [{error_id}] {error_name}\\n")

                if description:
                    lines.append(f"**{labels['definition']}:** {description}\\n")

                lines.append(f"**{labels['mark_present']}:**")
                for criterion in inclusion.split('\\n'):
                    if criterion.strip():
                        lines.append(f"  - {criterion.strip()}")

                if exclusion:
                    lines.append(f"\\n**{labels['mark_absent']}:**")
                    for criterion in exclusion.split('\\n'):
                        if criterion.strip():
                            lines.append(f"  - {criterion.strip()}")

                lines.append("")  # Empty line between errors

        return "\\n".join(lines)

        
    @staticmethod
    def get_category_clusters() -> dict:
        """Group taxonomy categories for hierarchical evaluation.
    
        Returns a dictionary mapping cluster names to lists of error IDs.
        Clusters are designed to group semantically related errors together
        to reduce cognitive load during evaluation.
    
        Returns:
            Dictionary with cluster names and error ID lists
        """
        return {
            "linguistic_grammar": [
                "LG-GR1", "LG-GR2", "LG-GR3", "LG-GR4",
                "LG-GR5", "LG-GR6", "LG-IS"
            ],
            "linguistic_language": [
                "LG-WL1", "LG-WL2"
            ],
            "semantic_accuracy": [
                "SE-MD", "SE-TS", "SE-HA1", "SE-HA2", "SE-HA3"
            ],
            "semantic_code": [
                "SE-CS1", "SE-CS2"
            ],
            "model_behavior": [
                "MS-IG", "MS-CC", "MS-ME1", "MS-ME2",
                "MS-ME3", "MS-ET", "MS-LT", "MS-RE1", "MS-RE2"
            ],
            "syntax_format": [
                "ST-IF1", "ST-IF2"
            ],
            "meta": ["E", "M"]
        }
        
    @staticmethod
    def filter_taxonomy(taxonomy_data: dict, error_ids: list) -> str:
        """Filter taxonomy to specific error IDs.
    
        Args:
            taxonomy_data: Full taxonomy dictionary
            error_ids: List of error IDs to include 
        Returns:
            Formatted taxonomy string for specified error IDs
        """
        import json
        filtered = {}   
        for category, errors in taxonomy_data.items():
            filtered_errors = [e for e in errors if e.get("id") in error_ids]
            if filtered_errors:
                filtered[category] = filtered_errors

        return json.dumps(filtered, ensure_ascii=False, indent=2)

