"""
Neural Metrics — Mean Score Visualization
==========================================
Shows the mean BARTScore / BERTScore for each scoring model and context level
(none / minimal / full), averaged over all target LLMs.  Uses a premium Plotly
dot-plot with alternating band shading.

Data: data/raw_{Language}.parquet
Output: interactive Plotly HTML.
"""

import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRIC_FAMILY = "bertscore"          # "bartscore" | "bertscore"
METRIC_VALUE_SUFFIX = None           # None → auto (bartscore→bart_score_reverse, bertscore→F)
LANGUAGE = "Polish"                  # Default language

DEFAULT_SUFFIX = {"bartscore": "bart_score_reverse", "bertscore": "F"}

CONTEXT_COLORS = {
    "none":    "#4285F4",   # Google Blue
    "minimal": "#FBBC05",   # Google Yellow
    "full":    "#34A853",   # Google Green
}

CONTEXT_SYMBOLS = {
    "none":    "circle",
    "minimal": "diamond",
    "full":    "square",
}

CONTEXT_ORDER = ["none", "minimal", "full"]

TARGET_LLMS = [
    "google/codegemma-7b",
    "meta-llama/CodeLlama-7b-hf",
    "Qwen/CodeQwen1.5-7B",
    "bigcode/starcoder2-7b",
    "ibm-granite/granite-8b-code-base",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(language: str):
    path = os.path.join(DATA_DIR, f"raw_{language}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet not found: {path}")
    return pd.read_parquet(path)


def build_records(df: pd.DataFrame, metric_family: str, selected_suffix: str):
    suffix = re.escape(selected_suffix)
    col_pattern = re.compile(
        rf"^{metric_family}_(?P<scoring_model>.+?)_ctx-(?P<context>none|minimal|full)"
        rf"_noise-none_(?P<target_llm>.+?)_{suffix}$"
    )
    matched = [(col, col_pattern.match(col)) for col in df.columns]
    matched = [(col, m) for col, m in matched if m is not None]
    if not matched:
        return pd.DataFrame()

    metrics = df[[col for col, _ in matched]].apply(pd.to_numeric, errors="coerce")
    col_means = metrics.mean(axis=0)

    rows = []
    for col, m in matched:
        rows.append({
            "column": col,
            "scoring_model": m.group("scoring_model"),
            "context": m.group("context"),
            "target_llm": m.group("target_llm"),
            "column_mean": col_means[col],
        })
    return pd.DataFrame(rows)


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_mean_scores(grouped: pd.DataFrame, metric_family: str,
                     selected_suffix: str, language: str, use_log: bool = True):
    scoring_model_order = list(pd.unique(grouped["scoring_model"]))
    x_positions = {m: i for i, m in enumerate(scoring_model_order)}

    # Short model names for x-axis (strip common prefixes)
    def shorten(m):
        return m.replace("microsoft__", "").replace("Salesforce__", "").replace(
            "google__", "").replace("facebook__", "").replace("distilbert__", "")

    fig = go.Figure()

    # Alternating band shading
    for i, model in enumerate(scoring_model_order):
        if i % 2 == 1:
            fig.add_vrect(
                x0=i - 0.5, x1=i + 0.5,
                fillcolor="rgba(100,149,237,0.07)", line_width=0, layer="below"
            )

    # One trace per context level
    for context in CONTEXT_ORDER:
        subset = grouped[grouped["context"] == context].copy()
        if subset.empty:
            continue

        # For log scale we need absolute values
        plot_vals = subset["plot_value"] if "plot_value" in subset.columns else subset["column_mean"].abs()
        x = [x_positions[m] for m in subset["scoring_model"].astype(str)]

        hovers = [
            f"<b>{CONTEXT_ORDER[CONTEXT_ORDER.index(context)].capitalize()} context</b><br>"
            f"Model: {m}<br>"
            f"Mean {metric_family.upper()}: {v:.6f}"
            for m, v in zip(subset["scoring_model"], subset["column_mean"])
        ]

        fig.add_trace(go.Scatter(
            x=x,
            y=plot_vals,
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
                f"Mean {metric_family.upper()} Score ({selected_suffix}) "
                f"by Scoring Model and Context — {language}<br>"
                "<sup>Averaged over all 5 target LLMs</sup>"
            ),
            font=dict(size=18, family="Inter, Arial, sans-serif"),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            tickmode="array",
            tickvals=list(x_positions.values()),
            ticktext=[shorten(m) for m in scoring_model_order],
            tickangle=-55,
            tickfont=dict(size=9),
            showgrid=False,
            zeroline=False,
            title=dict(text="Scoring Model", font=dict(size=12)),
        ),
        yaxis=dict(
            title=dict(
                text=f"|Mean {selected_suffix} score|" if use_log else f"Mean {selected_suffix} score",
                font=dict(size=12)
            ),
            type="log" if use_log else "linear",
            gridcolor="rgba(200,200,200,0.35)",
            zeroline=False,
        ),
        legend=dict(
            title=dict(text="Context Level", font=dict(size=11)),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(150,150,150,0.3)", borderwidth=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif"),
        width=1200, height=520,
        margin=dict(l=70, r=40, t=110, b=130),
        hovermode="closest",
    )

    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    valid_families = {"bartscore", "bertscore"}
    if METRIC_FAMILY not in valid_families:
        raise ValueError(f"METRIC_FAMILY must be one of {sorted(valid_families)}")

    selected_suffix = METRIC_VALUE_SUFFIX or DEFAULT_SUFFIX[METRIC_FAMILY]
    print(f"Loading data for {LANGUAGE} | {METRIC_FAMILY} / {selected_suffix} …")

    df = load_data(LANGUAGE)
    records = build_records(df, METRIC_FAMILY, selected_suffix)

    if records.empty:
        print("No columns matched the selected pattern. Check METRIC_FAMILY / LANGUAGE.")
        return

    target_llms_found = sorted(records["target_llm"].dropna().unique())
    if len(target_llms_found) != 5:
        raise ValueError(
            f"Expected 5 target LLMs, found {len(target_llms_found)}: {target_llms_found}"
        )

    # Validate all groups have 5 LLMs
    per_group = (
        records.groupby(["scoring_model", "context"])["target_llm"]
        .nunique().reset_index(name="n_target_llms")
    )
    bad = per_group[per_group["n_target_llms"] != 5]
    if not bad.empty:
        raise ValueError(f"Incomplete groups:\n{bad.to_string(index=False)}")

    scoring_model_order = list(pd.unique(records["scoring_model"]))
    grouped = records.groupby(["scoring_model", "context"], as_index=False)["column_mean"].mean()
    grouped["scoring_model"] = pd.Categorical(grouped["scoring_model"], categories=scoring_model_order, ordered=True)
    grouped["context"] = pd.Categorical(grouped["context"], categories=CONTEXT_ORDER, ordered=True)
    grouped = grouped.sort_values(["scoring_model", "context"])
    grouped["plot_value"] = grouped["column_mean"].abs()

    # Remove zeros (cannot show on log scale)
    n_zeros = (grouped["plot_value"] == 0).sum()
    if n_zeros:
        print(f"  Skipping {n_zeros} points with |mean|=0 (log scale cannot show them).")
        grouped = grouped[grouped["plot_value"] > 0]

    print("Building plot…")
    fig = plot_mean_scores(grouped, METRIC_FAMILY, selected_suffix, LANGUAGE)
    out = os.path.join(OUTPUT_DIR, f"neural_metrics_scoring_{METRIC_FAMILY}_{LANGUAGE.lower()}.html")
    fig.write_html(out)
    print(f"  → saved {out}")
    fig.show()


if __name__ == "__main__":
    main()
