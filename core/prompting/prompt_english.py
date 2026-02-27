from .prompt_base import PromptBase


class PromptLibrary(PromptBase):
    """English prompt templates for LLM-as-Judge evaluation.

    Inherits language-agnostic utilities from PromptBase.
    Only English-specific prompt content is defined here.

    Research-Based Enhancement Roadmap:
    ====================================
    This class requires updates based on recent LLM-as-a-Judge research.
    See references.bib for full citations of papers mentioned below.

    Key improvements needed:
    1. Chain-of-thought reasoning prompts [@liu-etal-2023-g]
    2. Bias mitigation in system prompts [@shi2025judgingjudgessystematicstudy]
    3. Rubric-based taxonomy formatting [@kim2023prometheus, @Pathak_2025]
    4. Hierarchical category evaluation [@lee2024improvingllmclassificationlogical]
    5. Multilingual evaluation strategies [@xuan2025mmluproxmultilingualbenchmarkadvanced]
    """
    
    # English uses default taxonomy file (error_taxonomy.json)
    LANGUAGE_CODE = "en"

    @staticmethod
    def system_basic() -> str:
        """Generate the base system prompt for the evaluator role.

        Returns:
            System prompt describing the evaluator's role

        CURRENT ISSUE: Minimal prompt ("You are an expert evaluator...") lacks explicit
        bias mitigation instructions. This can lead to:
        - Inconsistent evaluation across model predictions
        - Preference for verbose but incorrect comments
        - Systematic bias in multi-model comparisons

        """
        return (
            "You are an expert evaluator. You analyze Java code, comments, "
            "and documentation for correctness based on a given taxonomy of errors.\\n\\n"
            "Bias Mitigation Instructions:\\n"
            "- Avoid Verbosity Bias: Do not favor longer comments. Evaluate based on correctness.\\n"
            "- Ensure Consistency: Apply the same standards to all predictions.\\n"
            "- Prevent Systematic Bias: Evaluate each instance independently based on the taxonomy."
        )

    @staticmethod
    def system_enhanced() -> str:
        """Generate the bias mitigation system prompt for the evaluator role.

        Returns:
            Detailed system prompt describing the evaluator's role

        TODO [PRIORITY 1]: ENHANCE SYSTEM PROMPT WITH BIAS MITIGATION
        ==============================================================
        Research: [@shi2025judgingjudgessystematicstudy] identifies three critical biases
        in LLM-as-a-Judge evaluations:
        - Position bias: Judges favor first/last items in lists
        - Verbosity bias: Judges rate longer responses higher regardless of quality
        - Self-enhancement bias: Judges favor their own model's outputs (GPT-4 shows ~10%
          bias toward GPT-4 outputs; Claude shows ~25% bias toward Claude outputs)

        Also see: [@liu-etal-2023-g] MT-Bench foundational study on bias mitigation

        """
        return (
            "You are an expert code documentation evaluator with experience in "
            "software engineering and multilingual text analysis. Your task is to "
            "identify errors in LLM-generated code comments using a detailed taxonomy.\\n\\n"

            "Evaluation Principles:\n"
            "- Be consistent: Apply the same standards regardless of comment length or verbosity\n"
            "- Be precise: Only flag errors when inclusion criteria are clearly met\n"
            "- Check inclusion/exclusion: Always verify inclusion/exclusion criteria before confirming an error\n"
            "- Language-aware: Consider that comments may be in non-English languages; evaluate grammar and semantics appropriately for the target language\n"
            "- Context-dependent: Use the surrounding code context to assess semantic accuracy\n"
            "- Position-independent: Evaluate each prediction independently; do not let order influence your judgment\n\n"

            "IMPORTANT: Do not let comment length influence your judgment of quality. "
            "Short comments can be correct; long comments can contain errors. "
            "Evaluate each comment solely on whether it accurately describes the code "
            "according to the taxonomy criteria."
        )

    @staticmethod
    def expert_accuracy_guidance() -> str:
        return (
            "Expert accuracy labels:\n"
            "- correct: The predicted comment captures all information from the original comment; extra relevant details are acceptable.\n"
            "- partially_correct: The comment is mostly right but has a small mistake (e.g., minor typo, wrong variable name, or slight omission).\n"
            "- incorrect: The comment misses important information or contains substantial errors beyond small fixes."
        )

    @staticmethod
    def assignment_evaluate_models(taxonomy: str) -> str:
        """Generate assignment prompt for evaluating grouped model predictions.

        Args:
            taxonomy: Formatted string representation of the error taxonomy

        Returns:
            Task assignment prompt for grouped evaluation
        """
        return (
            "You are given Java code with an original comment and multiple model-generated predictions in JSON format. "
            "Your task is to evaluate each model's predicted comment and identify errors from the taxonomy.\n\n"
            f"Taxonomy of possible comment errors:\n{taxonomy}\n\n"
            "For each model prediction in the provided JSON:\n"
            "- Analyze the predicted_comment field.\n"
            "- Decide which error types from the taxonomy are present.\n"
            "- Refer to inclusion and exclusion criteria when making decisions.\n"
            "- If no errors are present, indicate that explicitly.\n\n"
            "Evaluation Principles:\n"
            "- Be consistent: Apply the same standards regardless of comment length or verbosity\n"
            "- Be precise: Only flag errors when inclusion criteria are clearly met\n"
            "- Check inclusion/exclusion: Always verify inclusion/exclusion criteria before confirming an error\n"
            "- Language-aware: Consider that comments may be in non-English languages; evaluate grammar and semantics appropriately for the target language\n"
            "- Context-dependent: Use the surrounding code context to assess semantic accuracy\n"
            "- Position-independent: Evaluate each prediction independently; do not let order influence your judgment\n\n"
            "The input is provided as JSON with source_code, original_comment, and model_predictions array."
        )

    @staticmethod
    def _structured_output_instruction_v2_cot() -> str:
        """DEPRECATED: Use output_with_reasoning() instead.
        
        This method is kept for reference only.
        
        Generate instructions for structured JSON output.
        Returns:
            Instruction string for JSON format specification

        [PRIORITY 1]: ADD CHAIN-OF-THOUGHT REASONING (G-EVAL STYLE)
        =================================================================
        Research: [@liu-etal-2023-g] G-Eval achieves 0.514 Spearman correlation with
        human judgments—approximately 2x improvement over baseline LLM evaluators—by
        requiring step-by-step reasoning BEFORE scoring.

        CURRENT ISSUE: Current prompt asks for classification directly without explicit
        reasoning steps. This "System 1" thinking leads to:
        - Lower agreement with human annotators
        - Inconsistent error detection across similar cases
        - Difficulty debugging why certain errors were/weren't flagged

        RESEARCH FINDING: Requiring models to articulate their reasoning process
        (Chain-of-Thought) before making judgments significantly improves:
        - Consistency across similar examples (+15-20%)
        - Alignment with human judgments (+2x correlation)
        - Debuggability and explainability of decisions

        ALTERNATIVE: Few-shot prompting with examples
        ----------------------------------------------
        Instead of explicit CoT instructions, provide 2-3 examples of correct
        evaluations showing the reasoning process. See [@liu-etal-2023-g] Section 3.2
        for few-shot prompt engineering strategies.
        """
        return (
            "Follow these steps for each error category:\\n"
            "1. Read the predicted comment carefully\\n"
            "2. Understand what part of the code the generated comment refers to, based on Fill-in-the-Middle (FIM) masking technique\\n"
            "3. Compare it against the original comment and code context\\n"
            "4. For each potential error, reason about whether inclusion criteria are met\\n"
            "5. Check exclusion criteria before confirming an error\\n"
            "6. Provide your reasoning, then your classification\\n"
            "7. Based on the above reasoning, determine the overall quality of the generated comment as Correct, Partial, Incorrect\\n\\n"
            "Evaluation Principles:\\n"
            "- Be consistent: Apply the same standards regardless of comment length or verbosity\\n"
            "- Be precise: Only flag errors when inclusion criteria are clearly met\\n"
            "- Check inclusion/exclusion: Always verify inclusion/exclusion criteria before confirming an error\\n"
            "- Language-aware: Consider that comments may be in non-English languages; evaluate grammar and semantics appropriately for the target language\\n"
            "- Context-dependent: Use the surrounding code context to assess semantic accuracy\\n"
            "- Position-independent: Evaluate each prediction independently; do not let order influence your judgment\\n\\n"
            f"{PromptLibrary.expert_accuracy_guidance()}\n\n"
            "Return structured JSON with the following format:\\n"
            "{\\n"
            '  "reasoning": "Step-by-step analysis: First, I observe that...",\\n'
            '  "errors": [\\n'
            "    {\\n"
            '      "error_id": "<error_id>",\\n'
            '      "confidence": <0.0-1.0>,\\n'
            '      "justification": "This error applies because..."\\n'
            "    }\\n"
            "  ],\\n"
            '  "overall_quality": "<correct|partially_correct|incorrect>"\n'
            "}\\n"
            "If no errors are present, return empty errors array with reasoning "
            "explaining why the comment is correct."
        )

    @staticmethod
    def output_basic() -> str:
        """Generate instructions for grouped data evaluation with multiple models.
        Returns:
            Instruction string for JSON format with model-specific results
        """
        return (
            f"{PromptLibrary.expert_accuracy_guidance()}\n\n"
            "Return structured JSON with results for each model:\n"
            "{\n"
            "  \"evaluations\": [\n"
            "    {\n"
            "      \"model_name\": \"<model_identifier>\",\n"
            "      \"errors\": [\"<error_id>\", ...],\n"
            "      \"explanation\": \"Short explanation of errors found.\",\n"
            "      \"overall_quality\": \"<correct|partially_correct|incorrect>\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "If a model has no errors, use an empty errors array."
        )

    @staticmethod
    def output_with_reasoning() -> str:
        """Generate instructions for grouped data evaluation with chain-of-thought reasoning.

        Combines grouped multi-model evaluation (from v1) with chain-of-thought reasoning
        structure (from v2_cot) for improved accuracy and explainability.

        Research basis: [@liu-etal-2023-g] G-Eval demonstrates that requiring explicit
        reasoning steps before classification improves human agreement by 15-20% and
        achieves ~2x better correlation with human judgments.

        Returns:
            Instruction string for JSON format with model-specific CoT results
        """
        return (
            "For each model's predicted comment, follow these steps:\\n"
            "1. Read the predicted comment carefully\\n"
            "2. Understand what part of the code the generated comment refers to, based on Fill-in-the-Middle (FIM) masking technique\\n"
            "3. Compare it against the original comment and code context\\n"
            "4. For each potential error, reason about whether inclusion criteria are met\\n"
            "5. Check exclusion criteria before confirming an error\\n"
            "6. Provide your reasoning, then your classification\\n"
            "7. Based on the above reasoning, determine the overall quality of the generated comment as Correct, Partial, Incorrect\\n\\n"
            "Evaluation Principles:\\n"
            "- Be consistent: Apply the same standards regardless of comment length or verbosity\\n"
            "- Be precise: Only flag errors when inclusion criteria are clearly met\\n"
            "- Check inclusion/exclusion: Always verify inclusion/exclusion criteria before confirming an error\\n"
            "- Language-aware: Consider that comments may be in non-English languages; evaluate grammar and semantics appropriately for the target language\\n"
            "- Context-dependent: Use the surrounding code context to assess semantic accuracy\\n"
            "- Position-independent: Evaluate each prediction independently; do not let order influence your judgment\\n\\n"
            f"{PromptLibrary.expert_accuracy_guidance()}\n\n"
            "Return structured JSON with results for each model:\\n"
            "{\\n"
            '  "evaluations": [\\n'
            "    {\\n"
            '      "model_name": "<model_identifier>",\\n'
            '      "reasoning": "Step-by-step analysis: First, I observe that...",\\n'
            '      "errors": [\\n'
            "        {\\n"
            '          "error_id": "<error_id>",\\n'
            '          "confidence": <0.0-1.0>,\\n'
            '          "justification": "This error applies because..."\\n'
            "        }\\n"
            "      ],\\n"
            '      "overall_quality": "<correct|partially_correct|incorrect>"\n'
            "    }\\n"
            "  ]\\n"
            "}\\n"
            "If a model has no errors, return empty errors array with reasoning "
            "explaining why the comment is correct."
        )

    @staticmethod
    def assignment_evaluate_cluster(cluster_name: str, cluster_taxonomy: str) -> str:
        """Generate assignment prompt for a specific category cluster.
    
        Creates a focused prompt that asks the judge to evaluate only a subset
        of error categories, reducing cognitive load and improving accuracy.
    
        Args:
            cluster_name: Name of the error cluster (e.g., "linguistic_grammar")
            cluster_taxonomy: Formatted taxonomy string for this cluster only
    
        Returns:
            Assignment prompt focused on specific error cluster
        """
        cluster_descriptions = {
            "linguistic_grammar": "linguistic and grammatical correctness",
            "linguistic_language": "language consistency and appropriateness",
            "semantic_accuracy": "semantic accuracy and completeness",
            "semantic_code": "code snippet accuracy",
            "model_behavior": "model-specific errors and artifacts",
            "syntax_format": "comment format and structure",
            "meta": "exclusion criteria and miscellaneous issues"
        }
    
        description = cluster_descriptions.get(cluster_name, cluster_name)
    
        return (
            f"You are evaluating code comments for {description.upper()} only.\\n\\n"
            f"Relevant error categories for this evaluation:\\n{cluster_taxonomy}\\n\\n"
            "IMPORTANT: Focus ONLY on the error types listed above. "
            "Ignore other potential issues—they will be evaluated separately.\\n\\n"
            "For each error category above, determine if the error is present based on "
            "the inclusion criteria. Always check exclusion criteria before confirming."
        )

