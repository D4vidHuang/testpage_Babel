"""
LLM-as-a-Judge Accuracy Visualization
======================================
Plots weighted Cohen's kappa (quadratic) between judge models and human expert
labels, broken down by language and prompt setting.

Data: data/raw_{Language}.parquet
Output: opens an interactive Plotly HTML in browser; also saves HTML file.
"""

import math
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SETTINGS = ["hierarchical", "cot", "standard", "rubric"]
LANGUAGES = ["en", "zh", "pl", "nl", "el"]
LANG_MAP = {"en": "English", "pl": "Polish", "zh": "Chinese", "nl": "Dutch", "el": "Greek"}
LANG_DISPLAY = {"en": "English", "zh": "Chinese", "pl": "Polish", "nl": "Dutch", "el": "Greek"}

TARGET_LLMS = [
    "google/codegemma-7b",
    "meta-llama/CodeLlama-7b-hf",
    "Qwen/CodeQwen1.5-7B",
    "bigcode/starcoder2-7b",
    "ibm-granite/granite-8b-code-base",
]

JUDGE_MODELS = [
    "google_gemini-3-flash-preview_floor",
    "meituan_longcat-flash-chat_floor",
    "openai_gpt-oss-20b_floor",
    "openai_gpt-oss-120b_floor",
    "anthropic_claude-haiku-4_5_floor",
    "qwen_qwen3-coder-next_floor",
    "qwen_qwen3-vl-235b-a22b-instruct_floor",
    "deepseek_deepseek-v3_2_floor",
    "x-ai_grok-4_1-fast_floor",
]

JUDGE_DISPLAY = {
    "google_gemini-3-flash-preview_floor": "Gemini Flash",
    "meituan_longcat-flash-chat_floor": "LongCat Flash",
    "openai_gpt-oss-20b_floor": "GPT-OSS 20B",
    "openai_gpt-oss-120b_floor": "GPT-OSS 120B",
    "anthropic_claude-haiku-4_5_floor": "Claude Haiku",
    "qwen_qwen3-coder-next_floor": "Qwen3 Coder",
    "qwen_qwen3-vl-235b-a22b-instruct_floor": "Qwen3 VL 235B",
    "deepseek_deepseek-v3_2_floor": "DeepSeek V3",
    "x-ai_grok-4_1-fast_floor": "Grok 4.1 Fast",
}

# Vibrant, accessible color palette
JUDGE_COLORS = {
    "google_gemini-3-flash-preview_floor": "#4285F4",
    "meituan_longcat-flash-chat_floor": "#FF6B9D",
    "openai_gpt-oss-20b_floor": "#FF9500",
    "openai_gpt-oss-120b_floor": "#34A853",
    "anthropic_claude-haiku-4_5_floor": "#EA4335",
    "qwen_qwen3-coder-next_floor": "#9B59B6",
    "qwen_qwen3-vl-235b-a22b-instruct_floor": "#00BCD4",
    "deepseek_deepseek-v3_2_floor": "#795548",
    "x-ai_grok-4_1-fast_floor": "#607D8B",
}

JUDGE_SYMBOLS = {
    "google_gemini-3-flash-preview_floor": "circle",
    "meituan_longcat-flash-chat_floor": "x",
    "openai_gpt-oss-20b_floor": "square",
    "openai_gpt-oss-120b_floor": "diamond",
    "anthropic_claude-haiku-4_5_floor": "triangle-up",
    "qwen_qwen3-coder-next_floor": "triangle-down",
    "qwen_qwen3-vl-235b-a22b-instruct_floor": "triangle-left",
    "deepseek_deepseek-v3_2_floor": "pentagon",
    "x-ai_grok-4_1-fast_floor": "star",
}

SHOW_CONFUSION_MATRICES = True

# ── Label mappers ─────────────────────────────────────────────────────────────

def _map_human_label(value):
    if pd.isna(value):
        return np.nan
    v = str(value).strip().lower()
    if v == "correct":
        return 2
    if v == "partial":
        return 1
    if v == "incorrect":
        return 0
    return np.nan


def _map_judge_label(value):
    if pd.isna(value):
        return np.nan
    v = str(value).strip().lower()
    if v == "correct":
        return 2
    if v == "partially_correct":
        return 1
    if v == "incorrect":
        return 0
    return np.nan


# ── Data loading ──────────────────────────────────────────────────────────────

def load_records():
    records = []
    for language in LANGUAGES:
        lang_name = LANG_MAP.get(language, language)
        path = os.path.join(DATA_DIR, f"raw_{lang_name}.parquet")
        if not os.path.exists(path):
            print(f"  [warn] missing {path}")
            continue
        df = pd.read_parquet(path)

        for judge_model in JUDGE_MODELS:
            for judge_setting in SETTINGS:
                y_true_all, y_pred_all = [], []

                for target_llm in TARGET_LLMS:
                    human_col = f"expert_accuracy_{target_llm}"
                    if human_col not in df.columns:
                        continue

                    judge_cols = df.filter(
                        regex=rf"^{re.escape(judge_model)}_{re.escape(judge_setting)}_"
                        rf"{re.escape(target_llm)}_accuracy$"
                    ).columns
                    if len(judge_cols) == 0:
                        continue

                    judge_col = judge_cols[0]
                    human = df[human_col].map(_map_human_label).to_numpy(dtype=float)
                    pred = df[judge_col].map(_map_judge_label).to_numpy(dtype=float)
                    mask = ~np.isnan(human) & ~np.isnan(pred)
                    if mask.sum() == 0:
                        continue

                    y_true_all.append(human[mask].astype(int))
                    y_pred_all.append(pred[mask].astype(int))

                if not y_true_all:
                    continue

                y_true = np.concatenate(y_true_all)
                y_pred = np.concatenate(y_pred_all)
                kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2], normalize="pred") * 100

                records.append({
                    "language": language,
                    "judge": judge_model,
                    "setting": judge_setting,
                    "kappa_weighted": kappa,
                    "n_samples": int(len(y_true)),
                    "confusion_matrix": cm,
                })

    return records


# ── Main scatter plot ─────────────────────────────────────────────────────────

def plot_kappa_scatter(records):
    group_width = len(SETTINGS)
    lang_to_x = {lang: idx for idx, lang in enumerate(LANGUAGES)}
    setting_offsets = {s: i for i, s in enumerate(SETTINGS)}

    # Build one trace per judge model (for clean legend)
    traces = {}
    for rec in records:
        jm = rec["judge"]
        x = lang_to_x[rec["language"]] * group_width + setting_offsets[rec["setting"]]
        y = rec["kappa_weighted"]
        hover = (
            f"<b>{JUDGE_DISPLAY.get(jm, jm)}</b><br>"
            f"Language: {LANG_DISPLAY.get(rec['language'], rec['language'])}<br>"
            f"Setting: {rec['setting']}<br>"
            f"κ (quadratic): <b>{y:.3f}</b><br>"
            f"n samples: {rec['n_samples']}"
        )
        if jm not in traces:
            traces[jm] = {"x": [], "y": [], "hover": []}
        traces[jm]["x"].append(x)
        traces[jm]["y"].append(y)
        traces[jm]["hover"].append(hover)

    fig = go.Figure()

    # Alternating language-group shading
    for i, lang in enumerate(LANGUAGES):
        base = i * group_width
        if i % 2 == 1:
            fig.add_vrect(
                x0=base - 0.5, x1=base + group_width - 0.5,
                fillcolor="rgba(100,149,237,0.06)", line_width=0,
                layer="below"
            )
        # Language divider lines
        if i > 0:
            fig.add_vline(
                x=base - 0.5,
                line=dict(color="rgba(150,150,150,0.4)", width=1, dash="dot")
            )

    # Reference line at κ = 0
    fig.add_hline(y=0, line=dict(color="rgba(200,50,50,0.5)", width=1.5, dash="dash"))
    # Guideline for "good" agreement (κ = 0.6)
    fig.add_hline(y=0.6, line=dict(color="rgba(50,180,50,0.4)", width=1, dash="dot"),
                  annotation_text="κ=0.6 (substantial)", annotation_position="right")
    fig.add_hline(y=0.4, line=dict(color="rgba(255,165,0,0.4)", width=1, dash="dot"),
                  annotation_text="κ=0.4 (moderate)", annotation_position="right")

    for jm, data in traces.items():
        fig.add_trace(go.Scatter(
            x=data["x"],
            y=data["y"],
            mode="markers",
            name=JUDGE_DISPLAY.get(jm, jm),
            marker=dict(
                symbol=JUDGE_SYMBOLS.get(jm, "circle"),
                color=JUDGE_COLORS.get(jm, "#888"),
                size=10,
                line=dict(color="white", width=1),
                opacity=0.9,
            ),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=data["hover"],
        ))

    # Bottom x-axis: settings
    tick_vals = []
    tick_text = []
    for lang in LANGUAGES:
        base = lang_to_x[lang] * group_width
        for setting in SETTINGS:
            tick_vals.append(base + setting_offsets[setting])
            tick_text.append(setting.capitalize())

    # Language annotations at the top
    for lang in LANGUAGES:
        base = lang_to_x[lang] * group_width + (group_width - 1) / 2
        fig.add_annotation(
            x=base, y=1.07, xref="x", yref="paper",
            text=f"<b>{LANG_DISPLAY.get(lang, lang)}</b>",
            showarrow=False, font=dict(size=12, color="#333"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(150,150,150,0.3)", borderwidth=1, borderpad=3,
        )

    fig.update_layout(
        title=dict(
            text="LLM-as-a-Judge Accuracy vs Human Expert<br>"
                 "<sup>Weighted Cohen's κ (quadratic) — higher is better</sup>",
            font=dict(size=20, family="Inter, Arial, sans-serif"),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            tickmode="array", tickvals=tick_vals, ticktext=tick_text,
            tickangle=-35, tickfont=dict(size=10),
            showgrid=False, zeroline=False,
            title=dict(text="Prompt Setting", font=dict(size=13)),
        ),
        yaxis=dict(
            title=dict(text="Weighted Cohen's κ (quadratic)", font=dict(size=13)),
            gridcolor="rgba(200,200,200,0.3)", zeroline=False,
            range=[-0.15, 1.05],
        ),
        legend=dict(
            title=dict(text="Judge Model", font=dict(size=12)),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(150,150,150,0.3)", borderwidth=1,
            font=dict(size=11),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif"),
        width=1200, height=560,
        margin=dict(l=70, r=200, t=100, b=90),
        hovermode="closest",
    )

    return fig


# ── Confusion matrices ────────────────────────────────────────────────────────

def plot_confusion_matrices(records):
    """One subplot grid per (language, setting) pair."""
    from itertools import groupby

    class_names = ["Incorrect", "Partial", "Correct"]
    grouped = {}
    for rec in records:
        grouped.setdefault((rec["language"], rec["setting"]), []).append(rec)

    figs = []
    for (lang, setting), recs in sorted(grouped.items()):
        recs = sorted(recs, key=lambda x: x["judge"])
        n = len(recs)
        if n == 0:
            continue

        ncols = 3
        nrows = math.ceil(n / ncols)
        titles = [
            f"{JUDGE_DISPLAY.get(r['judge'], r['judge'])}<br>κ={r['kappa_weighted']:.3f}"
            for r in recs
        ] + [""] * (nrows * ncols - n)

        v_spacing = min(0.14, 0.9 / max(nrows, 2)) if nrows > 1 else 0.0
        fig = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=titles,
            horizontal_spacing=0.08,
            vertical_spacing=v_spacing,
        )

        for idx, rec in enumerate(recs):
            row, col = divmod(idx, ncols)
            cm = rec["confusion_matrix"]
            text_vals = [[f"{cm[i, j]:.1f}%" for j in range(3)] for i in range(3)]

            fig.add_trace(go.Heatmap(
                z=cm,
                x=class_names,
                y=class_names,
                colorscale="Blues",
                zmin=0, zmax=100,
                text=text_vals,
                texttemplate="%{text}",
                textfont=dict(size=10),
                showscale=(idx == 0),
                colorbar=dict(title="% (col-norm)", len=0.3, y=0.85) if idx == 0 else None,
                hovertemplate="Human: %{y}<br>Judge: %{x}<br>%{z:.1f}%<extra></extra>",
            ), row=row + 1, col=col + 1)

            fig.update_xaxes(title_text="Judge", row=row + 1, col=col + 1)
            fig.update_yaxes(title_text="Human", row=row + 1, col=col + 1)

        fig.update_layout(
            title=dict(
                text=f"Confusion Matrices — {LANG_DISPLAY.get(lang, lang)} / {setting.capitalize()}",
                font=dict(size=16, family="Inter, Arial, sans-serif"),
                x=0.5, xanchor="center",
            ),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Inter, Arial, sans-serif", size=11),
            width=1100, height=380 * nrows + 80,
        )
        figs.append((lang, setting, fig))

    return figs


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    records = load_records()
    if not records:
        print("No accuracy data found. Check that parquet files exist in data/")
        return

    print(f"Loaded {len(records)} records. Building main scatter plot…")
    fig_scatter = plot_kappa_scatter(records)
    out_path = os.path.join(OUTPUT_DIR, "llm_judge_accuracy_kappa.html")
    fig_scatter.write_html(out_path)
    print(f"  → saved {out_path}")
    fig_scatter.show()

    if SHOW_CONFUSION_MATRICES:
        print("Building confusion matrix subplots…")
        cm_figs = plot_confusion_matrices(records)
        for lang, setting, fig in cm_figs:
            name = f"llm_judge_accuracy_cm_{lang}_{setting}.html"
            path = os.path.join(OUTPUT_DIR, name)
            fig.write_html(path)
            print(f"  → saved {path}")
        if cm_figs:
            cm_figs[0][2].show()


if __name__ == "__main__":
    main()
