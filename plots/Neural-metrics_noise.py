"""
Neural Metrics — Noise Detection Visualization
===============================================
For each scoring model and context level, fits a KDE-based binary classifier
to distinguish "real" comments from "noisy" ones and reports Macro F1.

Iterates over all languages. Produces Plotly dot-plots and optional confusion
matrix grids.

Data: data/raw_{Language}.parquet
Output: interactive Plotly HTML per language.
"""

import math
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, f1_score

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRIC_FAMILY = "bertscore"          # "bartscore" | "bertscore"
METRIC_VALUE_SUFFIX = None           # None → auto
NOISE_TYPES = ["uniform", "targeted"]
SHOW_CONFUSION_MATRICES = True

DEFAULT_SUFFIX = {"bartscore": "bart_score_forward", "bertscore": "F"}

CONTEXT_COLORS = {
    "none":    "#4285F4",
    "minimal": "#FBBC05",
    "full":    "#34A853",
}
CONTEXT_SYMBOLS = {
    "none":    "circle",
    "minimal": "diamond",
    "full":    "square",
}
CONTEXT_ORDER = ["none", "minimal", "full"]

LANGUAGES = ["Chinese", "English", "Greek", "Dutch", "Polish"]


# ── KDE-based binary classifier ───────────────────────────────────────────────

def _kde_density(samples, grid):
    samples = np.asarray(samples, dtype=float)
    n = len(samples)
    if n == 0:
        return np.zeros_like(grid)
    std = np.std(samples)
    if std == 0:
        return np.exp(-0.5 * ((grid - samples[0]) / 1e-6) ** 2)
    bw = max(1.06 * std * (n ** (-1 / 5)), 1e-6)
    diff = (grid[:, None] - samples[None, :]) / bw
    return np.mean(np.exp(-0.5 * diff ** 2), axis=1) / (bw * np.sqrt(2 * np.pi))


def _fit_binary_kde_segment_classifier(real_scores, noise_scores):
    real_scores = real_scores[~np.isnan(real_scores)]
    noise_scores = noise_scores[~np.isnan(noise_scores)]
    if len(real_scores) == 0 or len(noise_scores) == 0:
        return None

    x = np.concatenate([real_scores, noise_scores]).astype(float)
    lo, hi = float(np.min(x)), float(np.max(x))
    if lo == hi:
        return None

    class_scores = {1: real_scores, 0: noise_scores}
    grid = np.linspace(lo, hi, 1024)
    densities = {c: _kde_density(class_scores[c], grid) for c in (0, 1)}

    diff = densities[1] - densities[0]
    sign_changes = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]

    intersections = []
    for k in sign_changes:
        x0, x1 = grid[k], grid[k + 1]
        y0, y1 = diff[k], diff[k + 1]
        x_cross = x0 if y1 == y0 else x0 - y0 * (x1 - x0) / (y1 - y0)
        intersections.append(float(x_cross))

    if not intersections:
        return None

    intersections = np.array(sorted(set(round(t, 10) for t in intersections)))
    boundaries = np.concatenate(([-np.inf], intersections, [np.inf]))
    span = hi - lo
    margin = max(1e-6, 0.05 * span)

    segment_labels = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        seg_left = lo - margin if np.isneginf(left) else left
        seg_right = hi + margin if np.isposinf(right) else right
        if seg_right < seg_left:
            seg_left, seg_right = seg_right, seg_left
        probes = np.linspace(seg_left, seg_right, 9, dtype=float)
        best_class = max((0, 1), key=lambda c: float(np.mean(_kde_density(class_scores[c], probes))))
        segment_labels.append(int(best_class))

    return {"intersections": intersections, "segment_labels": np.array(segment_labels, dtype=int)}


def _predict_binary_kde(scores, classifier):
    scores = np.asarray(scores, dtype=float)
    pred = np.full(scores.shape, np.nan)
    if classifier is None:
        return pred
    valid = ~np.isnan(scores)
    seg_idx = np.searchsorted(classifier["intersections"], scores[valid], side="right")
    pred[valid] = classifier["segment_labels"][seg_idx]
    return pred


# ── Shared scatter plot ───────────────────────────────────────────────────────

def _plot_noise_f1(grouped, metric_family, selected_suffix, language, noise_types):
    scoring_model_order = list(pd.unique(grouped["scoring_model"]))
    x_positions = {m: i for i, m in enumerate(scoring_model_order)}

    def shorten(m):
        for prefix in ("microsoft__", "Salesforce__", "google__", "facebook__", "distilbert__"):
            m = m.replace(prefix, "")
        return m

    fig = go.Figure()

    for i in range(len(scoring_model_order)):
        if i % 2 == 1:
            fig.add_vrect(
                x0=i - 0.5, x1=i + 0.5,
                fillcolor="rgba(100,149,237,0.07)", line_width=0, layer="below"
            )

    # Baseline reference (guessing majority → F1 near 0.5 for balanced, but can vary)
    fig.add_hline(y=0.5, line=dict(color="rgba(150,150,150,0.5)", width=1, dash="dot"),
                  annotation_text="F1=0.5 (baseline)", annotation_position="right")

    for context in CONTEXT_ORDER:
        subset = grouped[grouped["context"] == context]
        if subset.empty:
            continue
        x = [x_positions[m] for m in subset["scoring_model"].astype(str)]
        hovers = [
            f"<b>Context: {context}</b><br>Model: {m}<br>Macro F1: {v:.4f}"
            for m, v in zip(subset["scoring_model"], subset["macro_f1"])
        ]
        fig.add_trace(go.Scatter(
            x=x,
            y=subset["macro_f1"],
            mode="markers",
            name=f"Context: {context}",
            marker=dict(
                symbol=CONTEXT_SYMBOLS[context],
                color=CONTEXT_COLORS[context],
                size=10,
                line=dict(color="white", width=1.2),
                opacity=0.92,
            ),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hovers,
        ))

    fig.update_layout(
        title=dict(
            text=(
                f"Noise Detection with KDE — {language}<br>"
                f"<sup>{metric_family.upper()} {selected_suffix} | "
                f"Noise types: {', '.join(noise_types)} | "
                "Averaged over target LLMs</sup>"
            ),
            font=dict(size=17, family="Inter, Arial, sans-serif"),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            tickmode="array",
            tickvals=list(x_positions.values()),
            ticktext=[shorten(m) for m in scoring_model_order],
            tickangle=-55, tickfont=dict(size=9),
            showgrid=False, zeroline=False,
            title=dict(text="Scoring Model", font=dict(size=12)),
        ),
        yaxis=dict(
            title=dict(text="Macro F1 (Noise vs Real)", font=dict(size=12)),
            gridcolor="rgba(200,200,200,0.35)", zeroline=False,
            range=[0, 1.05],
        ),
        legend=dict(
            title=dict(text="Context Level", font=dict(size=11)),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(150,150,150,0.3)", borderwidth=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif"),
        width=1200, height=500,
        margin=dict(l=70, r=160, t=110, b=140),
        hovermode="closest",
    )
    return fig


# ── Confusion matrix grid ─────────────────────────────────────────────────────

def _plot_confusion_matrices(cm_records, title_prefix):
    if not cm_records:
        return None

    class_names = ["Noise", "Real"]
    n = len(cm_records)
    ncols = 4
    nrows = math.ceil(n / ncols)

    titles = [
        f"{rec['scoring_model'].replace('microsoft__', '').replace('Salesforce__', '')[:24]} | "
        f"{rec['context']}<br>F1={rec['macro_f1']:.3f}, n={rec['n']}"
        for rec in cm_records
    ] + [""] * (nrows * ncols - n)

    v_spacing = min(0.14, 0.9 / max(nrows, 2)) if nrows > 1 else 0.0
    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=titles,
        horizontal_spacing=0.07,
        vertical_spacing=v_spacing,
    )

    for idx, rec in enumerate(cm_records):
        row, col = divmod(idx, ncols)
        cm = rec["confusion_matrix"]
        text_vals = [[f"{cm[i, j]:.1f}%" for j in range(2)] for i in range(2)]
        fig.add_trace(go.Heatmap(
            z=cm, x=class_names, y=class_names,
            colorscale="Blues", zmin=0, zmax=100,
            text=text_vals, texttemplate="%{text}", textfont=dict(size=10),
            showscale=(idx == 0),
            colorbar=dict(title="%", len=0.25, y=0.88) if idx == 0 else None,
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>%{z:.1f}%<extra></extra>",
        ), row=row + 1, col=col + 1)
        fig.update_xaxes(title_text="Predicted", row=row + 1, col=col + 1)
        fig.update_yaxes(title_text="True", row=row + 1, col=col + 1)

    fig.update_layout(
        title=dict(
            text=title_prefix,
            font=dict(size=15, family="Inter, Arial, sans-serif"),
            x=0.5, xanchor="center",
        ),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=10),
        width=1200, height=380 * nrows + 100,
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    valid_families = {"bartscore", "bertscore"}
    if METRIC_FAMILY not in valid_families:
        raise ValueError(f"METRIC_FAMILY must be one of {sorted(valid_families)}")

    selected_suffix = METRIC_VALUE_SUFFIX or DEFAULT_SUFFIX[METRIC_FAMILY]
    noise_expr = "|".join(re.escape(n) for n in NOISE_TYPES)

    metric_pattern = re.compile(
        rf"^{METRIC_FAMILY}_(?P<scoring_model>.+?)_ctx-(?P<context>none|minimal|full)"
        rf"_noise-(?P<noise>none|{noise_expr})_(?P<target_llm>.+?)_{re.escape(selected_suffix)}$"
    )

    for language in LANGUAGES:
        path = os.path.join(DATA_DIR, f"raw_{language}.parquet")
        if not os.path.exists(path):
            print(f"  [warn] missing {path} — skipping")
            continue
        print(f"\nProcessing {language} …")
        df = pd.read_parquet(path)

        parsed = []
        for col in df.columns:
            m = metric_pattern.match(col)
            if m is not None:
                parsed.append((col, m.groupdict()))

        if not parsed:
            print(f"  No columns matched for {language}.")
            continue

        col_map = {}
        for col, g in parsed:
            key = (g["scoring_model"], g["context"], g["target_llm"], g["noise"])
            col_map[key] = col

        rows = []
        cm_store = {}

        for scoring_model, context, target_llm, noise in sorted(col_map):
            if noise == "none":
                continue
            real_key = (scoring_model, context, target_llm, "none")
            if real_key not in col_map:
                continue

            real_scores = pd.to_numeric(df[col_map[real_key]], errors="coerce").to_numpy(dtype=float)
            noise_scores = pd.to_numeric(df[col_map[(scoring_model, context, target_llm, noise)]], errors="coerce").to_numpy(dtype=float)

            real_scores = real_scores[~np.isnan(real_scores)]
            noise_scores = noise_scores[~np.isnan(noise_scores)]

            if len(real_scores) == 0 or len(noise_scores) == 0:
                continue

            classifier = _fit_binary_kde_segment_classifier(real_scores, noise_scores)
            if classifier is None:
                continue

            y_true = np.concatenate([np.ones(len(real_scores), int), np.zeros(len(noise_scores), int)])
            all_scores = np.concatenate([real_scores, noise_scores])
            y_pred = _predict_binary_kde(all_scores, classifier)
            valid = ~np.isnan(y_pred)
            if valid.sum() == 0:
                continue

            y_true = y_true[valid]
            y_pred = y_pred[valid].astype(int)
            macro_f1 = f1_score(y_true, y_pred, labels=[0, 1], average="macro")
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1], normalize="true") * 100

            rows.append({
                "scoring_model": scoring_model,
                "context": context,
                "target_llm": target_llm,
                "noise_type": noise,
                "macro_f1": macro_f1,
            })

            key = (scoring_model, context)
            if key not in cm_store:
                cm_store[key] = {"y_true": [], "y_pred": []}
            cm_store[key]["y_true"].append(y_true)
            cm_store[key]["y_pred"].append(y_pred)

        if not rows:
            print(f"  No valid real-vs-noise comparisons for {language}.")
            continue

        rows_df = pd.DataFrame(rows)
        scoring_model_order = list(pd.unique(rows_df["scoring_model"]))
        grouped = rows_df.groupby(["scoring_model", "context"], as_index=False)["macro_f1"].mean()
        grouped["scoring_model"] = pd.Categorical(grouped["scoring_model"], categories=scoring_model_order, ordered=True)
        grouped["context"] = pd.Categorical(grouped["context"], categories=CONTEXT_ORDER, ordered=True)
        grouped = grouped.sort_values(["scoring_model", "context"])

        fig = _plot_noise_f1(grouped, METRIC_FAMILY, selected_suffix, language, NOISE_TYPES)
        out = os.path.join(OUTPUT_DIR, f"neural_metrics_noise_{METRIC_FAMILY}_{language.lower()}.html")
        fig.write_html(out)
        print(f"  → saved {out}")
        fig.show()

        if SHOW_CONFUSION_MATRICES and cm_store:
            cm_records = []
            for (sm, ctx), values in cm_store.items():
                y_true = np.concatenate(values["y_true"])
                y_pred = np.concatenate(values["y_pred"])
                cm_val = confusion_matrix(y_true, y_pred, labels=[0, 1], normalize="true") * 100
                cm_records.append({
                    "scoring_model": sm,
                    "context": ctx,
                    "confusion_matrix": cm_val,
                    "macro_f1": f1_score(y_true, y_pred, labels=[0, 1], average="macro"),
                    "n": int(len(y_true)),
                })
            cm_records.sort(key=lambda r: (r["scoring_model"], CONTEXT_ORDER.index(r["context"])))
            fig_cm = _plot_confusion_matrices(
                cm_records,
                f"Noise-vs-Real Confusion Matrices ({language}) — {METRIC_FAMILY.upper()} {selected_suffix}",
            )
            if fig_cm:
                out_cm = os.path.join(OUTPUT_DIR, f"neural_metrics_noise_cm_{METRIC_FAMILY}_{language.lower()}.html")
                fig_cm.write_html(out_cm)
                print(f"  → saved {out_cm}")

    print("\nDone.")


if __name__ == "__main__":
    main()
