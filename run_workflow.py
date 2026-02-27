#!/usr/bin/env python3
"""Unified workflow runner for EMSE-Babel evaluation pipeline.

This is the main entry point for running LLM-as-Judge evaluations.
Supports all workflow types through CLI flags.

Usage:
    # Standard evaluation (workflow 4)
    python run_workflow.py --type standard --num 30

    # Chain-of-thought evaluation (workflow 5)
    python run_workflow.py --type cot --num 30

    # Hierarchical evaluation (workflow 6)
    python run_workflow.py --type hierarchical --num 30

    # Rubric-based evaluation (workflow 7)
    python run_workflow.py --type rubric --num 30

    # Skip evaluation, only run scoring on existing results
    python run_workflow.py --score-only --pred output/experiments/rubric_evaluation/rubric_results.json
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional

from core import (
    DataLoader,
    JudgeConfig,
    JudgeManager,
    PromptLibrary,
    ResponseParser,
    extract_evaluations_from_judge_output,
    extract_tokens_from_judge_output,
)
from scoring.scorer import (
    load_ground_truth,
    load_predictions_from_output,
    compute_model_label_metrics,
    summarize_metrics,
)


# =============================================================================
# Helper Functions
# =============================================================================

def _extract_comment_prefix(masked_code: str) -> str:
    """Extract the comment prefix from masked code (text between last // or /* and FIM suffix).
    
    Used for verbose display to show what prefix is being combined with the prediction.
    Includes the FIM suffix token in the output for clarity.
    """
    suffix_pattern = r'<\|?fim_suffix\|?>| <SUF>'
    match = re.search(suffix_pattern, masked_code)
    
    if not match:
        return "(no FIM token found)"
    
    before_suffix = masked_code[:match.start()]
    suffix_token = match.group()  # The actual token found (e.g., <fim_suffix>)
    
    # Find the last comment start marker
    comment_start_match = None
    for m in re.finditer(r'//|/\*\*?', before_suffix):
        comment_start_match = m
    
    if comment_start_match:
        return before_suffix[comment_start_match.start():] + suffix_token
    
    return "(no comment prefix found)"


# =============================================================================
# Workflow Types Configuration
# =============================================================================

WORKFLOW_TYPES = {
    "standard": {
        "name": "Standard Evaluation",
        "description": "Basic evaluation with structured JSON output (Workflow 4)",
        "system_prompt": "basic",        # Uses system_basic()
        "output_instruction": "basic",    # Uses output_basic()
    },
    "cot": {
        "name": "Chain-of-Thought Evaluation", 
        "description": "G-Eval style with explicit reasoning steps (Workflow 5)",
        "system_prompt": "enhanced",     # Uses system_enhanced()
        "output_instruction": "reasoning", # Uses output_with_reasoning()
    },
    "hierarchical": {
        "name": "Hierarchical Evaluation",
        "description": "Category-clustered evaluation (Workflow 6)",
        "system_prompt": "basic",        # Uses system_basic()
        "output_instruction": "basic",    # Uses output_basic()
        "use_clusters": True,
    },
    "rubric": {
        "name": "Rubric-Based Evaluation",
        "description": "Taxonomy as rubric with PRESENT/ABSENT (Workflow 7)",
        "system_prompt": "enhanced",     # Uses system_enhanced()
        "output_instruction": "basic",    # Uses output_basic()
    },
    "combined": {
        "name": "Combined Evaluation (Workflow 8)",
        "description": "Hierarchical + Rubric + CoT + Bias Mitigation",
        "system_prompt": "enhanced",     # Uses system_enhanced()
        "output_instruction": "reasoning", # Uses output_with_reasoning()
        "use_clusters": True,
        "use_rubric": True,
    },
}


# =============================================================================
# Run Statistics Tracking
# =============================================================================

@dataclass
class RunStats:
    """Track statistics for a workflow run."""
    total_instances: int = 0
    successful_parses: int = 0
    failed_parses: int = 0
    total_api_calls: int = 0
    failed_api_calls: int = 0
    total_input_tokens: int = 0
    total_thinking_tokens: int = 0
    total_output_tokens: int = 0
    parse_warnings: List[str] = field(default_factory=list)
    
    def record_parse(self, success: bool, warnings: List[str] = None):
        """Record a parse attempt."""
        if success:
            self.successful_parses += 1
        else:
            self.failed_parses += 1
        if warnings:
            self.parse_warnings.extend(warnings)
    
    def record_api_call(self, success: bool, input_tokens: int = 0, thinking_tokens: int = 0, output_tokens: int = 0):
        """Record an API call attempt."""
        self.total_api_calls += 1
        if not success:
            self.failed_api_calls += 1
        self.total_input_tokens += input_tokens
        self.total_thinking_tokens += thinking_tokens
        self.total_output_tokens += output_tokens
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "total_instances": self.total_instances,
            "successful_parses": self.successful_parses,
            "failed_parses": self.failed_parses,
            "parse_success_rate": self.successful_parses / max(1, self.total_instances),
            "total_api_calls": self.total_api_calls,
            "failed_api_calls": self.failed_api_calls,
            "api_success_rate": (self.total_api_calls - self.failed_api_calls) / max(1, self.total_api_calls),
            "total_input_tokens": self.total_input_tokens,
            "total_thinking_tokens": self.total_thinking_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_thinking_tokens + self.total_output_tokens,
            "parse_warnings_count": len(self.parse_warnings),
        }
    
    def print_summary(self):
        """Print summary to console."""
        print(f"\n{'='*60}")
        print("RUN STATISTICS")
        print(f"{'='*60}")
        print(f"Instances processed: {self.total_instances}")
        print(f"Parse success rate:  {self.successful_parses}/{self.total_instances} "
              f"({100*self.successful_parses/max(1,self.total_instances):.1f}%)")
        print(f"Parse failures:      {self.failed_parses}")
        print(f"API calls:           {self.total_api_calls}")
        print(f"API failures:        {self.failed_api_calls}")
        print(f"Tokens:              In: {self.total_input_tokens}, Thinking: {self.total_thinking_tokens}, Out: {self.total_output_tokens}, "
              f"Total: {self.total_input_tokens + self.total_thinking_tokens + self.total_output_tokens}")
        if self.parse_warnings:
            print(f"Parse warnings:      {len(self.parse_warnings)}")


# =============================================================================
# Core Workflow Functions
# =============================================================================

def load_taxonomy() -> dict:
    """Load error taxonomy for the current language.
    
    Uses PromptLibrary.load_taxonomy() which automatically loads
    the translated taxonomy for the selected language.
    """
    return PromptLibrary.load_taxonomy()


def get_taxonomy_json() -> str:
    """Get taxonomy as formatted JSON string for the current language."""
    return PromptLibrary.get_taxonomy_json()


def build_config(
    workflow_type: str,
    model: str,
    cluster_name: str = None,
    cluster_taxonomy: dict = None
) -> JudgeConfig:
    """Build JudgeConfig for the specified workflow type."""
    wf = WORKFLOW_TYPES[workflow_type]
    
    # Select system prompt
    if wf["system_prompt"] == "enhanced":
        system_prompt = PromptLibrary.system_enhanced()
    else:
        system_prompt = PromptLibrary.system_basic()

    
    # Build assignment message
    if cluster_name and cluster_taxonomy:
        # Hierarchical or Combined: use cluster-specific taxonomy
        taxonomy_str = json.dumps(cluster_taxonomy, indent=2, ensure_ascii=False)
        assignment = PromptLibrary.assignment_evaluate_cluster(cluster_name, taxonomy_str)
        # For combined workflow: add rubric formatting to clusters
        if wf.get("use_rubric"):
            assignment += "\n\n" + PromptLibrary.format_taxonomy_as_rubric(cluster_taxonomy)
    elif wf["output_instruction"] == "rubric":
        # Rubric: use formatted rubric
        taxonomy = load_taxonomy()
        assignment = PromptLibrary.assignment_evaluate_models(taxonomy)
        assignment += "\n\n" + PromptLibrary.format_taxonomy_as_rubric(taxonomy)
    else:
        # Standard/CoT: use JSON taxonomy
        taxonomy = get_taxonomy_json()
        assignment = PromptLibrary.assignment_evaluate_models(taxonomy)
    
    # Add output instruction
    if wf["output_instruction"] == "reasoning":
        assignment += "\n\n" + PromptLibrary.output_with_reasoning()
    else:
        assignment += "\n\n" + PromptLibrary.output_basic()
    
    config_name = f"{workflow_type}_evaluator"
    if cluster_name:
        config_name = f"cluster_{cluster_name}"
    
    return JudgeConfig(
        name=config_name,
        models=[model],
        temperature=0,
        system_message=system_prompt,
        assignment_message=assignment,
        structured_output=True,
    )


def run_standard_workflow(
    data: List[Dict],
    workflow_type: str,
    model: str,
    stats: RunStats,
    verbose: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Run standard, CoT, or rubric workflow (single API call per instance)."""
    config = build_config(workflow_type, model)
    manager = JudgeManager([config])
    results = {}
    
    for idx, instance in enumerate(data, 1):
        instance_id = instance.get("metadata", {}).get("file_id", f"instance_{idx}")
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"[{idx}/{len(data)}] INSTANCE: {instance_id}")
            print(f"{'='*70}")
            
            # Show instance details
            meta = instance.get("metadata", {})
            code_ctx = instance.get("code_context", {})
            print(f"\n[METADATA]")
            print(f"   Language: {meta.get('language', 'unknown')}")
            print(f"   File ID:  {instance_id}")
            
            print(f"\n[ORIGINAL COMMENT]")
            orig_comment = code_ctx.get("original_comment", "N/A")
            print(f"   {orig_comment[:200]}{'...' if len(orig_comment) > 200 else ''}")
            
            print(f"\n[RECONSTRUCTION STEPS FOR EACH MODEL]")
            for pred in instance.get("model_predictions", []):
                model_name = pred.get("model_name", "unknown")
                masked_code = pred.get("masked_code", "")
                predicted_comment = pred.get("predicted_comment", "")
                
                # Extract the comment prefix from masked code
                comment_prefix = _extract_comment_prefix(masked_code)
                
                # Reconstruct full comment
                full_comment = PromptLibrary._reconstruct_full_comment(
                    masked_code, predicted_comment, orig_comment
                )
                
                # Format for display (escape control chars)
                def escape(s, max_len=80):
                    s = s[:max_len].replace('\r', '\\r').replace('\n', '\\n').replace('\t', '\\t')
                    return s + ('...' if len(s) >= max_len else '')
                
                print(f"\n   [{model_name.split('/')[-1]}]")
                print(f"      Prefix:      {escape(comment_prefix)}")
                print(f"      Prediction:  {escape(predicted_comment)}")
                print(f"      => Combined: {escape(full_comment, 100)}")
        else:
            print(f"[{idx}/{len(data)}] Processing {instance_id}...", end=" ", flush=True)
        
        formatted = PromptLibrary.format_grouped_instance(instance)
        
        if verbose:
            print(f"\n[FULL PROMPT TO JUDGE: {model}]")
            print("=" * 70)
            print("[SYSTEM MESSAGE]")
            print("-" * 70)
            print(config.system_message)
            print("-" * 70)
            print("\n[USER MESSAGE]")
            print("-" * 70)
            print(config.assignment_message)
            print("\n--- TEXT TO EVALUATE ---")
            # Show formatted instance (truncate if too long)
            if len(formatted) > 1500:
                print(formatted[:1500])
                print(f"\n... [truncated, {len(formatted)} chars total]")
            else:
                print(formatted)
            print("=" * 70)
        
        try:
            judge_result = manager.run(formatted)
            tokens = extract_tokens_from_judge_output(judge_result)
            stats.record_api_call(
                success=True, 
                input_tokens=tokens["prompt_tokens"],
                thinking_tokens=tokens["thinking_tokens"],
                output_tokens=tokens["completion_tokens"]
            )
            
            # Parse and validate response
            raw_text, parse_result = extract_evaluations_from_judge_output(judge_result)
            
            if verbose:
                print(f"\n[RAW MODEL RESPONSE]")
                print("-" * 50)
                if raw_text:
                    # Truncate very long responses
                    display_text = raw_text[:2000] if len(raw_text) > 2000 else raw_text
                    print(display_text)
                    if len(raw_text) > 2000:
                        print(f"\n... [truncated, {len(raw_text)} chars total]")
                else:
                    print("(no response content)")
                print("-" * 50)
                
                print(f"\n[PARSING RESULT]")
                print(f"   Method: {parse_result.parse_method}")
                print(f"   Success: {parse_result.success}")
                if parse_result.warnings:
                    print(f"   Warnings: {parse_result.warnings}")
                
                if parse_result.success:
                    print(f"\n[PARSED EVALUATIONS: {len(parse_result.evaluations)} models]")
                    for ev in parse_result.evaluations:
                        errors_str = ", ".join(ev.errors) if ev.errors else "(no errors)"
                        print(f"   * {ev.model_name}:")
                        print(f"     Errors: [{errors_str}]")
                        if ev.overall_quality:
                            print(f"     Quality: {ev.overall_quality}")
                        if ev.explanation:
                            expl = ev.explanation[:150] + "..." if len(ev.explanation) > 150 else ev.explanation
                            print(f"     Explanation: {expl}")
            
            if parse_result.success:
                stats.record_parse(success=True)
                if not verbose:
                    print(f"OK ({len(parse_result.evaluations)} evaluations)")
            else:
                stats.record_parse(success=False, warnings=parse_result.warnings)
                if not verbose:
                    print(f"FAIL: {parse_result.warnings[:1]}")
            
            results[instance_id] = {
                "instance": instance,
                "judge_evaluation": judge_result,
            }
            
        except Exception as e:
            stats.record_api_call(success=False)
            stats.record_parse(success=False, warnings=[str(e)])
            if verbose:
                print(f"\n[API ERROR]: {e}")
            else:
                print(f"ERROR: {e}")
            results[instance_id] = {
                "instance": instance,
                "judge_evaluation": {},
                "error": str(e),
            }
        
        stats.total_instances += 1
    
    return results


def run_hierarchical_workflow(
    data: List[Dict],
    model: str,
    stats: RunStats,
    verbose: bool = False,
    workflow_type: str = "hierarchical"
) -> Dict[str, Dict[str, Any]]:
    """Run hierarchical workflow (multiple API calls per instance, one per cluster).

    Args:
        workflow_type: 'hierarchical' or 'combined' - determines prompt configuration
    """
    taxonomy_data = load_taxonomy()
    clusters = PromptLibrary.get_category_clusters()

    results = {}

    for idx, instance in enumerate(data, 1):
        instance_id = instance.get("metadata", {}).get("file_id", f"instance_{idx}")
        print(f"[{idx}/{len(data)}] Processing {instance_id}...")

        formatted = PromptLibrary.format_grouped_instance(instance)
        all_evaluations = []
        cluster_details = []  # Store raw responses for each cluster
        cluster_errors = []

        # Process each cluster
        for cluster_name, error_ids in clusters.items():
            print(f"  - Cluster: {cluster_name}...", end=" ", flush=True)

            # Filter taxonomy to this cluster
            cluster_taxonomy = {}
            for cat, errors in taxonomy_data.items():
                filtered = [e for e in errors if e.get("id") in error_ids]
                if filtered:
                    cluster_taxonomy[cat] = filtered

            if not cluster_taxonomy:
                print("(no errors in cluster)")
                continue

            config = build_config(workflow_type, model, cluster_name, cluster_taxonomy)
            manager = JudgeManager([config])

            try:
                cluster_result = manager.run(formatted)
                tokens = extract_tokens_from_judge_output(cluster_result)
                stats.record_api_call(
                    success=True, 
                    input_tokens=tokens["prompt_tokens"],
                    thinking_tokens=tokens["thinking_tokens"],
                    output_tokens=tokens["completion_tokens"]
                )
                
                # Extract evaluations from cluster
                raw_text, parse_result = extract_evaluations_from_judge_output(cluster_result)

                # Store cluster details for debugging and full CoT preservation
                cluster_detail = {
                    "cluster_name": cluster_name,
                    "error_ids_in_cluster": error_ids,
                    "raw_response": raw_text,
                    "parse_success": parse_result.success,
                    "parse_method": parse_result.parse_method,
                }

                if parse_result.success:
                    stats.record_parse(success=True)
                    # Store full evaluation dicts (includes reasoning, explanation)
                    cluster_evals = [e.__dict__ for e in parse_result.evaluations]
                    all_evaluations.extend(cluster_evals)
                    cluster_detail["evaluations"] = cluster_evals
                    print(f"✓ {len(parse_result.evaluations)} evals")
                else:
                    stats.record_parse(success=False, warnings=parse_result.warnings)
                    cluster_detail["warnings"] = parse_result.warnings
                    print(f"✗ Parse failed")
                    cluster_errors.append({
                        "cluster": cluster_name,
                        "error": parse_result.warnings,
                    })

                cluster_details.append(cluster_detail)

            except Exception as e:
                stats.record_api_call(success=False)
                print(f"✗ API error: {e}")
                cluster_errors.append({
                    "cluster": cluster_name,
                    "error": str(e),
                })
                cluster_details.append({
                    "cluster_name": cluster_name,
                    "error_ids_in_cluster": error_ids,
                    "api_error": str(e),
                })

        results[instance_id] = {
            "instance": instance,
            "hierarchical_evaluation": all_evaluations,
            "cluster_details": cluster_details,  # Now populated!
            "cluster_errors": cluster_errors,
        }
        stats.total_instances += 1

    return results


def run_evaluation(
    workflow_type: str,
    data_path: str,
    num_instances: int,
    model: str,
    output_dir: str = None,
    verbose: bool = False,
    randomize: bool = False,
) -> tuple[Path, RunStats]:
    """Run evaluation workflow and return output path and stats."""
    import random
    
    print(f"\n{'='*60}")
    print(f"EMSE-Babel: {WORKFLOW_TYPES[workflow_type]['name']}")
    print(f"{'='*60}")
    print(f"Data: {data_path}")
    print(f"Instances: {num_instances}")
    print(f"Model: {model}")
    if verbose:
        print(f"Mode: VERBOSE (detailed output)")
    if randomize:
        print(f"Selection: RANDOMIZED")
    print(f"{'='*60}\n")
    
    # Load data
    all_data = DataLoader.load_from_json(data_path)
    
    if randomize:
        data = random.sample(all_data, min(num_instances, len(all_data)))
        print(f"Randomly selected {len(data)} instances from {len(all_data)} total\n")
    else:
        data = all_data[:num_instances]
        print(f"Loaded {len(data)} instances\n")
    
    stats = RunStats()
    
    # Run workflow
    if workflow_type in ["hierarchical", "combined"]:
        results = run_hierarchical_workflow(data, model, stats, verbose=verbose, workflow_type=workflow_type)
    else:
        results = run_standard_workflow(data, workflow_type, model, stats, verbose=verbose)
    
    # Determine output folder
    if output_dir:
        output_folder = Path(output_dir)
    else:
        base_dir = Path("output") / f"{workflow_type}_evaluation"
        output_folder = DataLoader.get_next_output_folder(base_dir)
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Export results
    if workflow_type in ["hierarchical", "combined"]:
        output_file = DataLoader.export_judge_results_hierarchical(results, output_folder)
    else:
        output_file = DataLoader.export_judge_results(results, output_folder)
    
    # Save stats
    stats_file = output_folder / "run_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, indent=2)
    
    stats.print_summary()
    print(f"\n=> Results saved to: {output_file}")
    print(f"=> Stats saved to: {stats_file}")
    
    return output_file, stats


def run_scoring(pred_path: str, ground_path: str, output_path: str = None) -> Dict:
    """Run scoring on predictions vs ground truth."""
    print(f"\n{'='*60}")
    print("SCORING")
    print(f"{'='*60}")
    
    gt = load_ground_truth(ground_path)
    preds = load_predictions_from_output(pred_path)
    
    matched = set(gt.keys()) & set(preds.keys())
    print(f"Ground truth: {len(gt)} instances")
    print(f"Predictions:  {len(preds)} instances")
    print(f"Matched:      {len(matched)} instances")
    
    if not matched:
        print("\n⚠ No matching instances found!")
        return {}
    
    metrics = compute_model_label_metrics(gt, preds)
    summary = summarize_metrics(metrics)
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS (per model, macro-averaged)")
    print(f"{'='*60}")
    
    for model_name, stats in summary.items():
        print(f"\n{model_name}:")
        print(f"  Precision: {stats.get('avg_precision', 0):.4f}")
        print(f"  Recall:    {stats.get('avg_recall', 0):.4f}")
        print(f"  F1:        {stats.get('avg_f1', 0):.4f}")
        if 'avg_kappa' in stats:
            print(f"  Kappa:     {stats['avg_kappa']:.4f}")
    
    # Save if output path specified
    if output_path:
        output = {"metrics": metrics, "summary": summary}
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n=> Metrics saved to: {output_path}")
    
    return summary


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EMSE-Babel: Unified workflow runner for LLM-as-Judge evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run standard evaluation on 30 instances
  python run_workflow.py --type standard --num 30

  # Run all workflow types for comparison
  python run_workflow.py --type standard --num 50
  python run_workflow.py --type cot --num 50
  python run_workflow.py --type rubric --num 50

  # Score existing results
  python run_workflow.py --score-only --pred output/standard_evaluation/output_1/judge_results.json

  # Full workflow: evaluate + score
  python run_workflow.py --type standard --num 30 --score
        """
    )
    
    # Workflow selection
    parser.add_argument(
        "--type", "-t",
        choices=list(WORKFLOW_TYPES.keys()),
        default="standard",
        help="Workflow type (default: standard)"
    )
    
    # Data options
    parser.add_argument(
        "--data", "-d",
        default="prepared_data_English/final_batch_judge.json",
        help="Path to input data file"
    )
    parser.add_argument(
        "--ground", "-g",
        default="prepared_data_English/final_batch_pipeline.json",
        help="Path to ground truth file"
    )
    parser.add_argument("--num", "-n", type=int, default=50, help="Number of instances")
    
    # Model options
    parser.add_argument(
        "--model", "-m",
        default="anthropic/claude-3.5-sonnet",
        help="Judge model to use"
    )

        # Model options
    parser.add_argument(
        "--language", "-l",
        default="English",
        help="Language of the code comments (default: English; Options: English, Chinese, Greek, Polish, Dutch)"
    )
    
    # Output options
    parser.add_argument("--output", "-o", help="Custom output directory")
    parser.add_argument("--score-output", help="Path to save scoring results")
    
    # Debug/inspection options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed step-by-step output (model responses, parsing, etc.)"
    )
    parser.add_argument(
        "--random", "-r",
        action="store_true",
        help="Randomize instance selection"
    )
    
    # Mode flags
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Skip evaluation, only run scoring"
    )
    parser.add_argument(
        "--pred", "-p",
        help="Path to predictions file (required with --score-only)"
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Run scoring after evaluation"
    )
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="List available workflow types"
    )
    
    args = parser.parse_args()
    
    # Auto-select data path based on language if not explicitly provided
    _lang_name = (args.language or "English").strip().capitalize()
    if _lang_name == "Zh":
        _lang_name = "Chinese"
    elif _lang_name == "Nl":
        _lang_name = "Dutch"
    elif _lang_name == "El":
        _lang_name = "Greek"
    elif _lang_name == "Pl":
        _lang_name = "Polish"
    elif _lang_name == "En":
        _lang_name = "English"
    
    # Override default data paths based on language
    if args.data == "prepared_data_English/final_batch_judge_fixed.json":
        args.data = f"prepared_data_{_lang_name}/final_batch_judge_fixed.json"
    if args.ground == "prepared_data_English/final_batch_pipeline_fixed.json":
        args.ground = f"prepared_data_{_lang_name}/final_batch_pipeline_fixed.json"
    
    # Select language-specific PromptLibrary implementation
    _lang = (args.language or "English").strip().lower()
    _lang_map = {
        "english": "prompt_english",
        "en": "prompt_english",
        "chinese": "prompt_chinese",
        "zh": "prompt_chinese",
        "dutch": "prompt_dutch",
        "nl": "prompt_dutch",
        "greek": "prompt_greek",
        "el": "prompt_greek",
        "polish": "prompt_polish",
        "pl": "prompt_polish",
    }
    _module_name = _lang_map.get(_lang, "prompt_english")
    try:
        _mod = __import__(f"core.prompting.{_module_name}", fromlist=["PromptLibrary"])
        # Override the module-level PromptLibrary used by the workflows
        globals()["PromptLibrary"] = getattr(_mod, "PromptLibrary")
    except Exception as _e:
        print(f"WARNING: failed to load prompt library for '{args.language}': {_e}. Using default.")
    
    # List workflow types
    if args.list_types:
        print("\nAvailable workflow types:\n")
        for key, wf in WORKFLOW_TYPES.items():
            print(f"  {key:15} - {wf['description']}")
        print()
        return
    
    # Score-only mode
    if args.score_only:
        if not args.pred:
            print("ERROR: --pred is required with --score-only")
            sys.exit(1)
        run_scoring(args.pred, args.ground, args.score_output)
        return
    
    # Run evaluation
    pred_path, stats = run_evaluation(
        workflow_type=args.type,
        data_path=args.data,
        num_instances=args.num,
        model=args.model,
        output_dir=args.output,
        verbose=args.verbose,
        randomize=args.random,
    )
    
    # Optionally run scoring
    if args.score:
        run_scoring(str(pred_path), args.ground, args.score_output)


if __name__ == "__main__":
    main()
