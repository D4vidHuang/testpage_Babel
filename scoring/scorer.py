"""Scoring helpers.

Functions:
- load_ground_truth(path): read prepared pipeline JSON with expert error_codes
- load_predictions_from_output(path): read judge output JSON and extract model-level predicted errors
- compute_model_label_metrics(ground_truth, predictions): compute TP/FP/FN/TN, precision, recall, f1, support, kappa
- summarize_metrics(metrics): produce a compact summary dict

The code treats each (instance_id, model_name) as one sample. Ground truth and
predictions are matched by instance_id and model_name.
"""
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

import argparse
import json
import sys

# Ensure repo root is importable when running as a script so top-level modules resolve.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.parsing import ResponseParser


def load_ground_truth(pipeline_file: str) -> Dict[str, Dict[str, Set[str]]]:
    """Load ground-truth error codes from a prepared pipeline JSON file.

    The pipeline file contains instances with:
    - metadata.file_id: instance identifier
    - model_predictions: array of {model_name, error_codes: [...]}

    Args:
        pipeline_file: Path to the pipeline JSON file (can be relative to project root)

    Returns: mapping instance_id -> model_name -> set(error_codes)
    """
    path = Path(pipeline_file)
    
    # If relative path, resolve from project root (parent of scoring/ directory)
    if not path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        path = project_root / path
    
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path} (looked in: {path.resolve()})")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ground: Dict[str, Dict[str, Set[str]]] = {}

    for inst in data:
        # Extract instance_id from metadata
        metadata = inst.get("metadata", {})
        instance_id = metadata.get("file_id")
        if not instance_id:
            continue

        model_map: Dict[str, Set[str]] = {}

        # Extract error_codes from each model prediction
        model_predictions = inst.get("model_predictions", [])
        for pred in model_predictions:
            model_name = pred.get("model_name")
            if not model_name:
                continue

            # error_codes is the ground truth list of error codes
            error_codes = pred.get("error_codes", [])
            model_map[model_name] = set(error_codes) if error_codes else set()

        ground[instance_id] = model_map

    return ground


def load_predictions_from_output(output_file: str) -> Dict[str, Dict[str, Set[str]]]:
    """Load judge predictions from an output JSON file produced by the pipeline.

    Uses the unified ResponseParser to handle all known formats:
    - Workflow 4 (standard): {"evaluations": [{"model_name": "...", "errors": ["SE-MD", ...]}]}
    - Workflow 5 (CoT): {"evaluations": [{"model_name": "...", "errors": [{"error_id": "SE-MD", ...}]}]}
    - Workflow 7 (rubric): {"model_predictions": [{"model_name": "...", "errors": {"SE-MD": "PRESENT", ...}}]}
    - Workflow 6 (hierarchical): Multiple cluster results

    Handles both file formats:
    - List format: [{instance_id: ..., judge_evaluations: ...}, ...]
    - Dict format: {instance_id: {instance: ..., judge_evaluation: ...}, ...}

    Returns: mapping instance_id -> model_name -> set(predicted_error_codes)
    """
    path = Path(output_file)

    # If relative path, resolve from project root (parent of scoring/ directory)
    if not path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        path = project_root / path

    if not path.exists():
        raise FileNotFoundError(output_file)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds: Dict[str, Dict[str, Set[str]]] = {}

    # Normalize to list of (instance_id, entry) tuples
    items = []
    if isinstance(data, list):
        # List format: each item has instance_id key
        for inst in data:
            instance_id = inst.get("instance_id")
            if instance_id:
                items.append((instance_id, inst))
    elif isinstance(data, dict):
        # Dict format: keys are instance_ids
        for instance_id, entry in data.items():
            items.append((instance_id, entry))

    for instance_id, inst in items:
        # Check for hierarchical_evaluation format (direct model evaluations)
        hierarchical_eval = inst.get("hierarchical_evaluation")
        if hierarchical_eval and isinstance(hierarchical_eval, list):
            # Hierarchical format: list of {model_name, errors, explanation}
            model_map: Dict[str, Set[str]] = {}
            for eval_item in hierarchical_eval:
                model_name = eval_item.get("model_name")
                errors = eval_item.get("errors", [])
                if model_name:
                    model_map[model_name] = set(errors) if errors else set()
            preds[instance_id] = model_map
            continue
        
        # Find judge evaluations (try multiple key names)
        judge_eval_raw = (
            inst.get("judge_evaluations") or 
            inst.get("judge_evaluation") or
            {}
        )
        
        # Get raw response string to parse
        raw_text = ""
        if isinstance(judge_eval_raw, str):
            raw_text = judge_eval_raw
        elif isinstance(judge_eval_raw, dict):
            # Check if it's a nested API response structure or already parsed
            # API response: {config_name: {model_name: {choices: [...]}}}
            # Parsed: {evaluations: [...]}
            has_api_structure = any(
                isinstance(v, dict) and any(
                    isinstance(vv, dict) and "choices" in vv 
                    for vv in v.values()
                )
                for v in judge_eval_raw.values() if isinstance(v, dict)
            )
            
            if has_api_structure:
                # Extract content from API response
                for config_results in judge_eval_raw.values():
                    if isinstance(config_results, dict):
                        for model_output in config_results.values():
                            if isinstance(model_output, dict) and "choices" in model_output:
                                raw_text = model_output["choices"][0]["message"]["content"]
                                break
                    if raw_text:
                        break
            else:
                # Already structured - convert to string for unified parsing
                raw_text = json.dumps(judge_eval_raw)
        
        # Also try raw_response field
        if not raw_text:
            raw_text = inst.get("raw_response", "")

        if not raw_text:
            preds[instance_id] = {}
            continue

        # Use unified parser
        parse_result = ResponseParser.parse(raw_text)
        
        model_map: Dict[str, Set[str]] = {}
        for eval_item in parse_result.evaluations:
            model_map[eval_item.model_name] = set(eval_item.errors)

        preds[instance_id] = model_map

    return preds


def _cohens_kappa_binary(y_true: List[int], y_pred: List[int]) -> float:
    """Compute Cohen's kappa for binary labels (0/1 lists)."""
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return float("nan")

    n = len(y_true)
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)

    po = (tp + tn) / n
    p_yes_true = (tp + fn) / n
    p_yes_pred = (tp + fp) / n
    pe = p_yes_true * p_yes_pred + (1 - p_yes_true) * (1 - p_yes_pred)

    denom = 1 - pe
    if denom == 0:
        return float("nan")

    kappa = (po - pe) / denom
    return kappa


def compute_model_label_metrics(
    ground_truth: Dict[str, Dict[str, Set[str]]],
    predictions: Dict[str, Dict[str, Set[str]]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Compute per-model, per-label metrics.

    Returns a nested dict: metrics[model_name][label] -> {tp, fp, fn, tn, precision, recall, f1, support, kappa}
    """
    # Collect all models and labels
    models = set()
    labels = set()
    samples: List[Tuple[str, str]] = []  # list of (instance_id, model_name)

    # Build sample list from matched instances only (where both ground truth and predictions exist)
    for inst_id, model_map in ground_truth.items():
        if inst_id not in predictions:
            continue
        for model_name, codes in model_map.items():
            models.add(model_name)
            labels.update(codes)
            samples.append((inst_id, model_name))

    # Also include labels that appear in predictions
    for inst_id, model_map in predictions.items():
        if inst_id not in ground_truth:
            continue
        for model_name, codes in model_map.items():
            models.add(model_name)
            labels.update(codes)

    metrics: Dict[str, Dict[str, Dict[str, Any]]] = {m: {} for m in models}

    # For each model and each label, build binary lists and compute metrics
    for model in models:
        for label in labels:
            y_true = []
            y_pred = []

            for inst_id, _ in samples:
                # only consider the sample if the ground truth contains this model
                gt_model_map = ground_truth.get(inst_id, {})
                if model not in gt_model_map:
                    continue

                gt_codes = gt_model_map.get(model, set())
                pred_codes = predictions.get(inst_id, {}).get(model, set())

                y_true.append(1 if label in gt_codes else 0)
                y_pred.append(1 if label in pred_codes else 0)

            n = len(y_true)
            if n == 0:
                # no samples for this model — skip
                continue

            tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
            fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
            fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
            tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            support = sum(y_true)
            kappa = _cohens_kappa_binary(y_true, y_pred)

            metrics[model][label] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "n_samples": n,
                "kappa": kappa,
            }

    return metrics


def summarize_metrics(metrics: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Produce a compact summary (macro averages) per model.

    Returns: { model_name: {avg_precision, avg_recall, avg_f1, avg_kappa, total_support} }
    """
    summary: Dict[str, Any] = {}
    for model, label_map in metrics.items():
        precisions = []
        recalls = []
        f1s = []
        kappas = []
        total_support = 0

        for label, stat in label_map.items():
            precisions.append(stat.get("precision", 0.0))
            recalls.append(stat.get("recall", 0.0))
            f1s.append(stat.get("f1", 0.0))
            k = stat.get("kappa")
            if k is not None and not (isinstance(k, float) and (k != k)):
                kappas.append(k)
            total_support += stat.get("support", 0)

        n_labels = max(1, len(label_map))
        summary[model] = {
            "avg_precision": sum(precisions) / n_labels,
            "avg_recall": sum(recalls) / n_labels,
            "avg_f1": sum(f1s) / n_labels,
            "avg_kappa": (sum(kappas) / len(kappas)) if kappas else float("nan"),
            "total_support": total_support,
            "n_labels": len(label_map),
        }

    return summary


def export_scoring_results(
    metrics: Dict[str, Dict[str, Dict[str, Any]]],
    summary: Dict[str, Any],
    output_file: str
) -> str:
    """Export scoring results to a JSON file.

    Args:
        metrics: Per-model, per-label metrics from compute_model_label_metrics()
        summary: Summarized metrics from summarize_metrics()
        output_file: Path to output JSON file

    Returns: Path to the exported file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert sets to lists for JSON serialization
    serializable_metrics = {
        model: {
            label: {k: (list(v) if isinstance(v, set) else v) for k, v in stats.items()}
            for label, stats in label_map.items()
        }
        for model, label_map in metrics.items()
    }

    results = {
        "metrics": serializable_metrics,
        "summary": summary
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return str(output_path)


def main(ground_path: str, pred_path: str, out_path: Optional[str] = None) -> Dict[str, Any]:
    """Main entry for scoring.

    Args:
        ground_path: path to prepared pipeline JSON (ground truth with error_codes)
        pred_path: path to predictions file (judge output from any workflow)
        out_path: optional path to save full metrics JSON

    Returns: dict with 'metrics' and 'summary'
    """
    print(f"Loading ground truth from: {ground_path}")
    ground = load_ground_truth(ground_path)
    print(f"  -> Loaded {len(ground)} instances with ground truth")

    preds: Dict[str, Dict[str, Set[str]]] = {}

    print(f"Loading predictions from: {pred_path}")

    # Load predictions from properly formatted JSON file
    preds = load_predictions_from_output(pred_path)

    print(f"  -> Loaded {len(preds)} instances with predictions")

    # Find matching instances
    matched_instances = set(ground.keys()) & set(preds.keys())
    print(f"  -> {len(matched_instances)} instances matched between ground truth and predictions")

    if not matched_instances:
        print("\nWARNING: No matching instances found! Check instance_id format.")
        print(f"  Ground truth instance IDs (first 5): {list(ground.keys())[:5]}")
        print(f"  Prediction instance IDs (first 5): {list(preds.keys())[:5]}")
        return {"metrics": {}, "summary": {}}

    metrics = compute_model_label_metrics(ground, preds)
    summary = summarize_metrics(metrics)

    print("\n" + "=" * 60)
    print("SCORING RESULTS")
    print("=" * 60)

    if out_path:
        export_scoring_results(metrics, summary, out_path)
        print(f"Saved detailed metrics to: {out_path}")

    # Print summary
    print("\nPer-Model Summary (macro-averaged):")
    print("-" * 60)
    for model, stats in summary.items():
        print(f"\n{model}:")
        print(f"  Precision: {stats['avg_precision']:.4f}")
        print(f"  Recall:    {stats['avg_recall']:.4f}")
        print(f"  F1:        {stats['avg_f1']:.4f}")
        print(f"  Kappa:     {stats['avg_kappa']:.4f}" if not (stats['avg_kappa'] != stats['avg_kappa']) else "  Kappa:     N/A")
        print(f"  Support:   {stats['total_support']} errors across {stats['n_labels']} labels")

    return {"metrics": metrics, "summary": summary}


def print_detailed_metrics(metrics: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    """Print detailed per-label metrics for debugging."""
    print("\n" + "=" * 80)
    print("DETAILED PER-LABEL METRICS")
    print("=" * 80)

    for model, label_map in metrics.items():
        print(f"\n[{model}]")
        print("-" * 60)
        print(f"{'Label':<12} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} {'Prec':>6} {'Rec':>6} {'F1':>6}")
        print("-" * 60)

        for label, stats in sorted(label_map.items()):
            print(f"{label:<12} {stats['tp']:>4} {stats['fp']:>4} {stats['fn']:>4} {stats['tn']:>4} "
                  f"{stats['precision']:>6.3f} {stats['recall']:>6.3f} {stats['f1']:>6.3f}")


# =============================================================================
# CONFIGURATION: Set paths here for easy testing
# =============================================================================

# Ground truth file (contains expert error_codes in model_predictions)
GROUND_TRUTH_PATH = "prepared_data_English/final_batch_pipeline.json"

# Prediction file (judge output from any workflow)
# The scorer automatically detects and handles different formats:
# - Workflow 4: Standard evaluation with error lists
# - Workflow 5: Chain-of-Thought with confidence scores
# - Workflow 6: Hierarchical with cluster consolidation
# - Workflow 7: Rubric with PRESENT/ABSENT format
# - Workflow 8: Final combined (all techniques)
#
# Uncomment the one you want to test:
# PREDICTIONS_PATH = "output/workflow_4_evaluation/output_1/judge_results.json"
# PREDICTIONS_PATH = "output/cot_evaluation/output_1/judge_results.json"
# PREDICTIONS_PATH = "output/hierarchical_evaluation/output_1/judge_results.json"
PREDICTIONS_PATH = "output/rubric_evaluation/output_1/judge_results.json"
# PREDICTIONS_PATH = "output/final_combined_evaluation/output_1/judge_results.json"

# Output file for detailed metrics (optional, set to None to skip)
OUTPUT_METRICS_PATH = "scoring/scoring_results.json"


if __name__ == "__main__":
    # Check if running with command-line arguments
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Score judge outputs against ground truth")
        parser.add_argument("--ground", required=True, help="Path to prepared pipeline JSON (ground truth)")
        parser.add_argument("--pred", required=True, help="Path to predictions file (judge output)")
        parser.add_argument("--out", required=False, help="Optional output JSON file for metrics")
        parser.add_argument("--detailed", action="store_true", help="Print detailed per-label metrics")
        args = parser.parse_args()

        result = main(args.ground, args.pred, args.out)

        if args.detailed:
            print_detailed_metrics(result["metrics"])
    else:
        # Use configured paths above
        print("Running scorer with configured paths...")
        print(f"  Ground truth: {GROUND_TRUTH_PATH}")
        print(f"  Predictions:  {PREDICTIONS_PATH}")
        print(f"  Output:       {OUTPUT_METRICS_PATH}")
        print()

        result = main(GROUND_TRUTH_PATH, PREDICTIONS_PATH, OUTPUT_METRICS_PATH)

        # Optionally print detailed metrics
        # print_detailed_metrics(result["metrics"])

