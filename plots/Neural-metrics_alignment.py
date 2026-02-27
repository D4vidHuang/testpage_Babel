"""
Neural Metrics — Human Alignment Visualization
===============================================
For each scoring model and context level, shows:
  • Weighted Cohen's κ with human expert labels
  • Macro F1

Iterates over all languages (Chinese, English, Greek, Dutch, Polish).
Uses Plotly for rich interactive charts.

Data: data/raw_{Language}.parquet
Output: interactive Plotly HTML per language.

NOTE: Fully self-contained — no external scoring module required.
      Accuracy is derived from continuous BARTScore/BERTScore values
      via tertile binning (matching original gen_accuracy logic):
        lower third  → Incorrect (0)
        middle third → Partial   (1)
        upper third  → Correct   (2)
"""

import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRIC_FAMILY      = "bartscore"   # "bartscore" | "bertscore"
METRIC_VALUE_SUFFIX = None         # None → auto-determined
SHOW_CONFUSION_MATRICES = True

DEFAULT_SUFFIX = {
    "bartscore": "bart_score_reverse",
    "bertscore": "F",
}

CONTEXT_COLORS  = {"none": "#00A6D6", "minimal": "#FFB81C", "full": "#009B77"}
CONTEXT_SYMBOLS = {"none": "circle",  "minimal": "diamond",  "full":   "square"}
CONTEXT_ORDER   = ["none", "minimal", "full"]

LANGUAGES = ["Chinese", "English", "Greek", "Dutch", "Polish"]

TARGET_LLMS = [
    "google/codegemma-7b",
    "meta-llama/CodeLlama-7b-hf",
    "Qwen/CodeQwen1.5-7B",
    "bigcode/starcoder2-7b",
    "ibm-granite/granite-8b-code-base",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _map_human_label(value):
    """Map expert accuracy string/int to 0/1/2."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)) and value in (0, 1, 2):
        return int(value)
    v = str(value).strip().lower()
    if v.startswith("correct"):
        return 2
    if v.startswith("partial"):
        return 1
    if v.startswith("incorrect"):
        return 0
    return np.nan


def _scores_to_labels(scores: np.ndarray) -> np.ndarray:
    """Convert continuous metric scores to 3-class labels via tertile binning.

    Tertile thresholds are computed on non-NaN values:
      lower third  → 0 (Incorrect)
      middle third → 1 (Partial)
      upper third  → 2 (Correct)
    """
    labels = np.full(len(scores), np.nan)
    valid = ~np.isnan(scores)
    if valid.sum() < 3:
        return labels
    q33, q67 = np.nanpercentile(scores[valid], [33.33, 66.67])
    labels[valid & (scores < q33)]              = 0
    labels[valid & (scores >= q33) & (scores < q67)] = 1
    labels[valid & (scores >= q67)]             = 2
    return labels


def _shorten(name: str) -> str:
    for prefix in ("microsoft__", "Salesforce__", "google__", "facebook__",
                   "distilbert__", "sdadas__"):
        name = name.replace(prefix, "")
    return name


# ── Scatter plot ──────────────────────────────────────────────────────────────

def _plot_metric(grouped, metric_col, y_label, title):
    scoring_model_order = list(pd.unique(grouped["scoring_model"]))
    x_positions = {m: i for i, m in enumerate(scoring_model_order)}

    fig = go.Figure()

    # Alternating band shading
    for i, model in enumerate(scoring_model_order):
        if i % 2 == 1:
            fig.add_vrect(x0=i - 0.5, x1=i + 0.5,
                          fillcolor="rgba(0,166,214,0.06)", line_width=0, layer="below")

    # Reference lines
    if "kappa" in metric_col:
        fig.add_hline(y=0,   line=dict(color="rgba(224,60,49,0.6)",  width=1.5, dash="dash"))
        fig.add_hline(y=0.6, line=dict(color="rgba(0,155,119,0.45)", width=1,   dash="dot"),
                      annotation_text="κ=0.6 (substantial)", annotation_position="right",
                      annotation_font_size=10)
        fig.add_hline(y=0.4, line=dict(color="rgba(255,184,28,0.45)", width=1,  dash="dot"),
                      annotation_text="κ=0.4 (moderate)",   annotation_position="right",
                      annotation_font_size=10)
    else:
        fig.add_hline(y=0.33, line=dict(color="rgba(224,60,49,0.45)", width=1, dash="dot"),
                      annotation_text="F1=0.33 (random baseline)", annotation_position="right",
                      annotation_font_size=10)

    for context in CONTEXT_ORDER:
        subset = grouped[grouped["context"] == context].dropna(subset=[metric_col])
        if subset.empty:
            continue
        x = [x_positions[m] for m in subset["scoring_model"].astype(str)]
        hovers = [
            f"<b>Context: {context}</b><br>Model: {_shorten(m)}<br>{y_label}: {v:.4f}"
            for m, v in zip(subset["scoring_model"], subset[metric_col])
        ]
        fig.add_trace(go.Scatter(
            x=x, y=subset[metric_col],
            mode="markers",
            name=f"Context: {context}",
            marker=dict(
                symbol=CONTEXT_SYMBOLS[context],
                color=CONTEXT_COLORS[context],
                size=11,
                line=dict(color="white", width=1.2),
                opacity=0.92,
            ),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hovers,
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family="Inter, Arial, sans-serif"),
                   x=0.5, xanchor="center"),
        xaxis=dict(
            tickmode="array",
            tickvals=list(x_positions.values()),
            ticktext=[_shorten(m) for m in scoring_model_order],
            tickangle=-55, tickfont=dict(size=9),
            showgrid=False, zeroline=False,
            title=dict(text="Scoring Model", font=dict(size=12)),
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(size=12)),
            gridcolor="rgba(200,200,200,0.35)", zeroline=False,
        ),
        legend=dict(title=dict(text="Context Level", font=dict(size=11)),
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="rgba(150,150,150,0.3)", borderwidth=1),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif"),
        width=1200, height=520,
        margin=dict(l=70, r=180, t=110, b=150),
        hovermode="closest",
    )
    return fig


# ── Confusion matrix grid ─────────────────────────────────────────────────────

def _plot_confusion_matrices(cm_records, title_prefix):
    if not cm_records:
        return None
    class_names = ["Incorrect", "Partial", "Correct"]
    n = len(cm_records)
    ncols = 4
    nrows = math.ceil(n / ncols)

    titles = [
        f"{_shorten(rec['scoring_model'])[:26]} | {rec['context']}<br>"
        f"F1={rec['macro_f1']:.3f}, κ={rec['kappa']:.3f}, n={rec['n']}"
        for rec in cm_records
    ] + [""] * (nrows * ncols - n)

    v_spacing = min(0.14, 0.9 / max(nrows, 2)) if nrows > 1 else 0.0
    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=titles,
                        horizontal_spacing=0.07,
                        vertical_spacing=v_spacing)

    for idx, rec in enumerate(cm_records):
        row, col = divmod(idx, ncols)
        cm = rec["confusion_matrix"]
        text_vals = [[f"{cm[i, j]:.1f}%" for j in range(3)] for i in range(3)]
        fig.add_trace(go.Heatmap(
            z=cm, x=class_names, y=class_names,
            colorscale="Blues", zmin=0, zmax=100,
            text=text_vals, texttemplate="%{text}", textfont=dict(size=9),
            showscale=(idx == 0),
            colorbar=dict(title="%", len=0.25, y=0.88) if idx == 0 else None,
            hovertemplate="Human: %{y}<br>Predicted: %{x}<br>%{z:.1f}%<extra></extra>",
        ), row=row + 1, col=col + 1)
        fig.update_xaxes(title_text="Predicted", row=row + 1, col=col + 1)
        fig.update_yaxes(title_text="Human",     row=row + 1, col=col + 1)

    fig.update_layout(
        title=dict(text=title_prefix,
                   font=dict(size=15, family="Inter, Arial, sans-serif"),
                   x=0.5, xanchor="center"),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=10),
        width=1200, height=420 * nrows + 100,
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    valid_families = {"bartscore", "bertscore"}
    if METRIC_FAMILY not in valid_families:
        raise ValueError(f"METRIC_FAMILY must be one of {sorted(valid_families)}")

    selected_suffix = METRIC_VALUE_SUFFIX or DEFAULT_SUFFIX[METRIC_FAMILY]
    suffix_re = re.escape(selected_suffix)

    col_pattern = re.compile(
        rf"^{METRIC_FAMILY}_(?P<scoring_model>.+?)_ctx-(?P<context>none|minimal|full)"
        rf"_noise-none_(?P<target_llm>.+?)_{suffix_re}$"
    )

    for language in LANGUAGES:
        path = os.path.join(DATA_DIR, f"raw_{language}.parquet")
        if not os.path.exists(path):
            print(f"  [warn] missing {path} — skipping")
            continue
        print(f"\nProcessing {language} …")
        df = pd.read_parquet(path)

        matched = [(col, m) for col in df.columns
                   if (m := col_pattern.match(col)) is not None]
        if not matched:
            print(f"  No columns matched for {language}.")
            continue

        rows    = []
        cm_store = {}

        for llm in TARGET_LLMS:
            expert_col = f"expert_accuracy_{llm}"
            if expert_col not in df.columns:
                continue

            llm_cols = [(col, m) for col, m in matched if m.group("target_llm") == llm]
            if not llm_cols:
                continue

            human_raw = df[expert_col].map(_map_human_label).to_numpy(dtype=float)

            for col, m in llm_cols:
                scoring_model = m.group("scoring_model")
                context       = m.group("context")

                scores = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
                pred   = _scores_to_labels(scores)

                mask = ~np.isnan(human_raw) & ~np.isnan(pred)
                if mask.sum() < 5:
                    continue

                y_true = human_raw[mask].astype(int)
                y_pred = pred[mask].astype(int)

                # Skip if both arrays are constant (kappa undefined)
                if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
                    macro_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2],
                                        average="macro", zero_division=0)
                    kappa = np.nan
                else:
                    macro_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2],
                                        average="macro", zero_division=0)
                    try:
                        kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
                    except Exception:
                        kappa = np.nan

                key = (scoring_model, context)
                if key not in cm_store:
                    cm_store[key] = {"y_true": [], "y_pred": []}
                cm_store[key]["y_true"].append(y_true)
                cm_store[key]["y_pred"].append(y_pred)

                rows.append({
                    "scoring_model":  scoring_model,
                    "context":        context,
                    "target_llm":     llm,
                    "alignment_kappa": kappa,
                    "macro_f1":        macro_f1,
                })

        if not rows:
            print(f"  No alignment data computed for {language}.")
            continue

        means_df = pd.DataFrame(rows)
        scoring_model_order = list(pd.unique(means_df["scoring_model"]))

        grouped = (
            means_df
            .groupby(["scoring_model", "context"], as_index=False)[["alignment_kappa", "macro_f1"]]
            .mean()
        )
        grouped["scoring_model"] = pd.Categorical(
            grouped["scoring_model"], categories=scoring_model_order, ordered=True)
        grouped["context"] = pd.Categorical(
            grouped["context"], categories=CONTEXT_ORDER, ordered=True)
        grouped = grouped.sort_values(["scoring_model", "context"])

        # ── Plot Macro F1 ──
        fig_f1 = _plot_metric(
            grouped, "macro_f1", "Macro F1",
            f"Neural Metric Macro F1 ({METRIC_FAMILY.upper()} · {selected_suffix}) — {language}<br>"
            "<sup>Tertile-binned predictions vs. human expert labels, averaged over target LLMs</sup>",
        )
        out_f1 = os.path.join(OUTPUT_DIR,
                              f"neural_metrics_alignment_f1_{METRIC_FAMILY}_{language.lower()}.html")
        fig_f1.write_html(out_f1)
        print(f"  → saved {out_f1}")

        # ── Plot Kappa ──
        fig_kappa = _plot_metric(
            grouped, "alignment_kappa", "Weighted Cohen's κ (quadratic)",
            f"Neural Metric Human Agreement ({METRIC_FAMILY.upper()} · {selected_suffix}) — {language}<br>"
            "<sup>Tertile-binned predictions vs. human expert labels, averaged over target LLMs</sup>",
        )
        out_kappa = os.path.join(OUTPUT_DIR,
                                 f"neural_metrics_alignment_kappa_{METRIC_FAMILY}_{language.lower()}.html")
        fig_kappa.write_html(out_kappa)
        print(f"  → saved {out_kappa}")

        # ── Confusion Matrices ──
        if SHOW_CONFUSION_MATRICES and cm_store:
            cm_records = []
            for (scoring_model, context), values in cm_store.items():
                y_true = np.concatenate(values["y_true"])
                y_pred = np.concatenate(values["y_pred"])
                if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
                    continue
                cm_val = confusion_matrix(y_true, y_pred,
                                          labels=[0, 1, 2], normalize="true") * 100
                try:
                    kappa_val = cohen_kappa_score(y_true, y_pred, weights="quadratic")
                except Exception:
                    kappa_val = np.nan
                cm_records.append({
                    "scoring_model":  scoring_model,
                    "context":        context,
                    "confusion_matrix": cm_val,
                    "macro_f1": f1_score(y_true, y_pred, labels=[0, 1, 2],
                                         average="macro", zero_division=0),
                    "kappa":  kappa_val,
                    "n":      int(len(y_true)),
                })
            if cm_records:
                cm_records.sort(key=lambda r: (r["scoring_model"],
                                               CONTEXT_ORDER.index(r["context"])))
                fig_cm = _plot_confusion_matrices(
                    cm_records,
                    f"Confusion Matrices ({language}) — {METRIC_FAMILY.upper()} · {selected_suffix}",
                )
                if fig_cm:
                    out_cm = os.path.join(
                        OUTPUT_DIR,
                        f"neural_metrics_alignment_cm_{METRIC_FAMILY}_{language.lower()}.html")
                    fig_cm.write_html(out_cm)
                    print(f"  → saved {out_cm}")

    print("\nDone.")


if __name__ == "__main__":
    main()
