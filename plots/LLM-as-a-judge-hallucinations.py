"""
LLM-as-a-Judge Hallucination Rate Visualization
================================================
Shows the fraction of error codes produced by judge models that are NOT in the
official taxonomy (i.e., hallucinated). Broken down by language, setting and
judge model.

Data: data/raw_{Language}.parquet, taxonomy/error_taxonomy.json
Output: interactive Plotly HTML.
"""

import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TAXONOMY_PATH = os.path.join(os.path.dirname(__file__), "..", "taxonomy", "error_taxonomy.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SETTINGS = ["hierarchical", "cot", "standard", "rubric"]
LANGUAGES = ["en", "zh", "pl", "nl", "el"]
LANG_MAP = {"en": "English", "pl": "Polish", "zh": "Chinese", "nl": "Dutch", "el": "Greek"}
LANG_DISPLAY = {"en": "English", "zh": "Chinese", "pl": "Polish", "nl": "Dutch", "el": "Greek"}

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

# ── Load taxonomy ─────────────────────────────────────────────────────────────

def load_error_codes():
    with open(TAXONOMY_PATH, "r") as f:
        taxonomy = json.load(f)
    codes = []
    for category_errors in taxonomy.values():
        for err in category_errors:
            codes.append(err["id"])
    return set(codes)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_records(error_codes):
    records = []
    overall_hallucination_counts = Counter()

    for language in LANGUAGES:
        lang_name = LANG_MAP.get(language, language)
        path = os.path.join(DATA_DIR, f"raw_{lang_name}.parquet")
        if not os.path.exists(path):
            print(f"  [warn] missing {path}")
            continue
        df = pd.read_parquet(path)

        for judge_model in JUDGE_MODELS:
            for judge_setting in SETTINGS:
                cols = df.filter(regex=f"{judge_model}.{judge_setting}.*_errors").columns
                hallucinations = 0
                total_errors = 0
                hallucination_counts = Counter()

                for col in cols:
                    for errors in df[col].values:
                        if errors is None:
                            continue
                        if isinstance(errors, float):
                            # NaN or other float — skip
                            continue
                        if isinstance(errors, str):
                            try:
                                errors = json.loads(errors)
                            except json.JSONDecodeError:
                                continue
                        if not isinstance(errors, (list, tuple)):
                            continue
                        if not errors:
                            continue
                        total_errors += len(errors)
                        for error in errors:
                            if error not in error_codes:
                                hallucinations += 1
                                hallucination_counts[error] += 1
                                overall_hallucination_counts[error] += 1

                rate = hallucinations / max(1, total_errors)
                records.append({
                    "language": language,
                    "judge": judge_model,
                    "setting": judge_setting,
                    "hallucinations": hallucinations,
                    "total_errors": total_errors,
                    "hallucination_rate": rate,
                    "hallucination_counts": dict(hallucination_counts),
                })

    return records, overall_hallucination_counts


# ── Main scatter plot ─────────────────────────────────────────────────────────

def plot_hallucination_scatter(records):
    group_width = len(SETTINGS)
    lang_to_x = {lang: idx for idx, lang in enumerate(LANGUAGES)}
    setting_offsets = {s: i for i, s in enumerate(SETTINGS)}

    traces = {}
    for rec in records:
        jm = rec["judge"]
        x = lang_to_x[rec["language"]] * group_width + setting_offsets[rec["setting"]]
        y = rec["hallucination_rate"] * 100

        top_halluc = sorted(rec["hallucination_counts"].items(), key=lambda kv: -kv[1])[:5]
        top_str = "<br>".join(f"  {k}: {v}" for k, v in top_halluc) or "none"
        hover = (
            f"<b>{JUDGE_DISPLAY.get(jm, jm)}</b><br>"
            f"Language: {LANG_DISPLAY.get(rec['language'], rec['language'])}<br>"
            f"Setting: {rec['setting']}<br>"
            f"Hallucination rate: <b>{y:.2f}%</b><br>"
            f"Hallucinations: {rec['hallucinations']} / {rec['total_errors']}<br>"
            f"Top hallucinated codes:<br>{top_str}"
        )
        if jm not in traces:
            traces[jm] = {"x": [], "y": [], "hover": []}
        traces[jm]["x"].append(x)
        traces[jm]["y"].append(y)
        traces[jm]["hover"].append(hover)

    fig = go.Figure()

    # Alternating shading and language dividers
    for i, lang in enumerate(LANGUAGES):
        base = i * group_width
        if i % 2 == 1:
            fig.add_vrect(
                x0=base - 0.5, x1=base + group_width - 0.5,
                fillcolor="rgba(255,165,0,0.05)", line_width=0, layer="below"
            )
        if i > 0:
            fig.add_vline(
                x=base - 0.5,
                line=dict(color="rgba(150,150,150,0.4)", width=1, dash="dot")
            )

    # Reference lines
    fig.add_hline(y=0, line=dict(color="rgba(60,60,60,0.3)", width=1))
    fig.add_hline(y=5, line=dict(color="rgba(255,165,0,0.5)", width=1.5, dash="dot"),
                  annotation_text="5% threshold", annotation_position="right")

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

    # Bottom x-axis ticks
    tick_vals, tick_text = [], []
    for lang in LANGUAGES:
        base = lang_to_x[lang] * group_width
        for setting in SETTINGS:
            tick_vals.append(base + setting_offsets[setting])
            tick_text.append(setting.capitalize())

    # Language annotations
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
            text="LLM-as-a-Judge Hallucination Rate<br>"
                 "<sup>Fraction of produced error codes not in taxonomy</sup>",
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
            title=dict(text="Hallucination Rate (%)", font=dict(size=13)),
            gridcolor="rgba(200,200,200,0.3)", zeroline=False,
            rangemode="tozero",
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


# ── Top hallucinated codes bar chart ─────────────────────────────────────────

def plot_top_hallucinations(overall_counts, top_n=20):
    if not overall_counts:
        return None

    top = overall_counts.most_common(top_n)
    codes = [t[0] for t in top]
    counts = [t[1] for t in top]

    # Color-code by count magnitude
    max_c = max(counts) if counts else 1
    colors = [f"rgba({int(220*(1-c/max_c)+50)},{int(100*(c/max_c)+50)},{int(200*(1-c/max_c)+50)},0.85)"
              for c in counts]

    fig = go.Figure(go.Bar(
        x=codes,
        y=counts,
        marker=dict(
            color=counts,
            colorscale="Reds",
            showscale=True,
            colorbar=dict(title="Count"),
        ),
        text=[str(c) for c in counts],
        textposition="outside",
        hovertemplate="Code: %{x}<br>Count: %{y}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=f"Top {top_n} Hallucinated Error Codes (across all languages & settings)",
            font=dict(size=18, family="Inter, Arial, sans-serif"),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title="Error Code", tickangle=-45,
            tickfont=dict(size=10), showgrid=False,
        ),
        yaxis=dict(
            title="Total Count",
            gridcolor="rgba(200,200,200,0.3)",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif"),
        width=1100, height=460,
        margin=dict(l=60, r=40, t=80, b=110),
    )

    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("Loading taxonomy …")
    error_codes = load_error_codes()
    print(f"  {len(error_codes)} valid error codes.")

    print("Loading data …")
    records, overall_counts = load_records(error_codes)
    if not records:
        print("No hallucination data found. Check that parquet files exist in data/")
        return

    print(f"Loaded {len(records)} records.")

    if overall_counts:
        print("\nTop hallucinated codes overall:")
        for code, cnt in overall_counts.most_common(10):
            print(f"  {code}: {cnt}")

    print("\nBuilding hallucination scatter plot…")
    fig_scatter = plot_hallucination_scatter(records)
    out = os.path.join(OUTPUT_DIR, "llm_judge_hallucinations.html")
    fig_scatter.write_html(out)
    print(f"  → saved {out}")
    fig_scatter.show()

    if overall_counts:
        print("Building top hallucination bar chart…")
        fig_bar = plot_top_hallucinations(overall_counts)
        if fig_bar:
            out2 = os.path.join(OUTPUT_DIR, "llm_judge_hallucinations_top_codes.html")
            fig_bar.write_html(out2)
            print(f"  → saved {out2}")
            fig_bar.show()


if __name__ == "__main__":
    main()
