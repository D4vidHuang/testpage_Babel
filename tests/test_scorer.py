"""Tests for the scoring module."""

import json
from pathlib import Path

from scoring.scorer import (
    load_ground_truth,
    load_predictions_from_output,
    compute_model_label_metrics,
    summarize_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestLoadGroundTruth:
    """Test ground truth loading."""

    def test_load_example_pipeline(self):
        """Load ground truth from example pipeline file."""
        gt_path = REPO_ROOT / "prepared_data_examples" / "example_batch_pipeline.json"
        gt = load_ground_truth(str(gt_path))
        
        assert len(gt) > 0
        # Check structure: {instance_id: {model_name: set(errors)}}
        for inst_id, model_map in gt.items():
            assert isinstance(inst_id, str)
            assert isinstance(model_map, dict)


class TestLoadPredictions:
    """Test prediction loading from different formats."""

    def test_workflow4_list_format(self):
        """Load predictions from workflow 4 (list format)."""
        pred_path = REPO_ROOT / "output" / "workflow_4_evaluation" / "output_1" / "judge_results.json"
        if not pred_path.exists():
            return  # Skip if file doesn't exist
        
        preds = load_predictions_from_output(str(pred_path))
        
        assert len(preds) > 0
        for inst_id, model_map in preds.items():
            assert isinstance(model_map, dict)

    def test_rubric_dict_format(self):
        """Load predictions from rubric format (dict format)."""
        pred_path = REPO_ROOT / "output" / "experiments" / "rubric_evaluation" / "rubric_results.json"
        if not pred_path.exists():
            return  # Skip if file doesn't exist
        
        preds = load_predictions_from_output(str(pred_path))
        
        assert len(preds) > 0
        # Should have extracted PRESENT errors
        has_errors = any(
            errors for model_map in preds.values() for errors in model_map.values()
        )
        assert has_errors


class TestMetrics:
    """Test metric computation."""

    def test_compute_metrics(self):
        """Compute metrics on example data."""
        gt_path = REPO_ROOT / "prepared_data_examples" / "example_batch_pipeline.json"
        pred_path = REPO_ROOT / "output" / "workflow_4_evaluation" / "output_1" / "judge_results.json"
        
        if not pred_path.exists():
            return
        
        gt = load_ground_truth(str(gt_path))
        preds = load_predictions_from_output(str(pred_path))
        metrics = compute_model_label_metrics(gt, preds)
        
        # Should have metrics for at least one model
        assert len(metrics) > 0
        
        # Check metric structure
        for model, label_map in metrics.items():
            for label, stats in label_map.items():
                assert "precision" in stats
                assert "recall" in stats
                assert "f1" in stats

    def test_summarize_metrics(self):
        """Summarize metrics produces valid output."""
        gt_path = REPO_ROOT / "prepared_data_examples" / "example_batch_pipeline.json"
        pred_path = REPO_ROOT / "output" / "workflow_4_evaluation" / "output_1" / "judge_results.json"
        
        if not pred_path.exists():
            return
        
        gt = load_ground_truth(str(gt_path))
        preds = load_predictions_from_output(str(pred_path))
        metrics = compute_model_label_metrics(gt, preds)
        summary = summarize_metrics(metrics)
        
        for model, stats in summary.items():
            assert "avg_precision" in stats
            assert "avg_recall" in stats
            assert "avg_f1" in stats