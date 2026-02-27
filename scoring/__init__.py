"""Scoring utilities for EMSE-Babel.

Functions for computing evaluation metrics (precision, recall, F1, Cohen's kappa)
and analyzing results.

- scorer.py: Compute per-model/label metrics from ground truth vs predictions
- analysis.ipynb: All-in-one notebook for aggregation, metrics, and visualization

Ensures repo root is on sys.path for imports like `from data_loader import DataLoader`.
"""

from pathlib import Path
import sys

# Ensure the repository root is on sys.path so top-level modules (e.g. data_loader)
# can be imported when this package is used by tooling/test runners that don't
# already include the repo root on sys.path.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from .scorer import (
    load_ground_truth,
    load_predictions_from_output,
    compute_model_label_metrics,
    summarize_metrics,
    main,
)

__all__ = [
    "load_ground_truth",
    "load_predictions_from_output",
    "compute_model_label_metrics",
    "summarize_metrics",
    "main",
]