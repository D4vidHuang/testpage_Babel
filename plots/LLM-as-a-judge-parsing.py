"""
LLM-as-a-Judge Parse Failure Rate Visualization
================================================
Plots the fraction of API calls that could not be parsed, by language, setting
and judge model. Reads run_stats.json files from the data directory.

Data: data/{judge}_-{lang}_-{lang}_-{setting}/run_stats.json
Output: interactive Plotly HTML.
"""

import json
import os

import plotly.graph_objects as go

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SETTINGS = ["hierarchical", "cot", "standard", "rubric"]
LANGUAGES = ["en", "zh", "pl", "el", "nl"]
LANG_DISPLAY = {"en": "English", "zh": "Chinese", "pl": "Polish", "el": "Greek", "nl": "Dutch"}

JUDGE_MODELS = [
    "meituan_longcat-flash-chat_floor",
    "anthropic_claude-haiku-4_5_floor",
    "deepseek_deepseek-v3_2_floor",
    "google_gemini-3-flash-preview_floor",
    "openai_gpt-oss-120b_floor",
    "openai_gpt-oss-20b_floor",
    "qwen_qwen3-coder-next_floor",
    "qwen_qwen3-vl-235b-a22b-instruct_floor",
    "x-ai_grok-4_1-fast_floor",
]

JUDGE_DISPLAY = {
    "meituan_longcat-flash-chat_floor": "LongCat Flash",
    "anthropic_claude-haiku-4_5_floor": "Claude Haiku",
    "deepseek_deepseek-v3_2_floor": "DeepSeek V3",
    "google_gemini-3-flash-preview_floor": "Gemini Flash",
    "openai_gpt-oss-120b_floor": "GPT-OSS 120B",
    "openai_gpt-oss-20b_floor": "GPT-OSS 20B",
    "qwen_qwen3-coder-next_floor": "Qwen3 Coder",
    "qwen_qwen3-vl-235b-a22b-instruct_floor": "Qwen3 VL 235B",
    "x-ai_grok-4_1-fast_floor": "Grok 4.1 Fast",
}

JUDGE_COLORS = {
    "meituan_longcat-flash-chat_floor": "#FF6B9D",
    "anthropic_claude-haiku-4_5_floor": "#EA4335",
    "deepseek_deepseek-v3_2_floor": "#795548",
    "google_gemini-3-flash-preview_floor": "#4285F4",
    "openai_gpt-oss-120b_floor": "#34A853",
    "openai_gpt-oss-20b_floor": "#FF9500",
    "qwen_qwen3-coder-next_floor": "#9B59B6",
    "qwen_qwen3-vl-235b-a22b-instruct_floor": "#00BCD4",
    "x-ai_grok-4_1-fast_floor": "#607D8B",
}

JUDGE_SYMBOLS = {
    "meituan_longcat-flash-chat_floor": "x",
    "anthropic_claude-haiku-4_5_floor": "triangle-up",
    "deepseek_deepseek-v3_2_floor": "pentagon",
    "google_gemini-3-flash-preview_floor": "circle",
    "openai_gpt-oss-120b_floor": "diamond",
    "openai_gpt-oss-20b_floor": "square",
    "qwen_qwen3-coder-next_floor": "triangle-down",
    "qwen_qwen3-vl-235b-a22b-instruct_floor": "triangle-left",
    "x-ai_grok-4_1-fast_floor": "star",
}


# ── Data loading ──────────────────────────────────────────────────────────────

import glob
import re as _re

_FOLDER_RE = _re.compile(
    r"^(?P<judge>.+?)_-(?P<lang>[a-z]{2})_-[a-z]{2}_-(?P<setting>\w+)$"
)

def load_records():
    records = []
    pattern = os.path.join(DATA_DIR, "*", "run_stats.json")
    for file_path in glob.glob(pattern):
        folder_name = os.path.basename(os.path.dirname(file_path))
        m = _FOLDER_RE.match(folder_name)
        if m is None:
            continue
        judge = m.group("judge")
        lang = m.group("lang")
        setting = m.group("setting")
        with open(file_path, "r") as f:
            data = json.load(f)
        total_calls = data.get("total_api_calls", 0)
        failed_parses = data.get("failed_parses", 0)
        parse_rate = failed_parses / max(1, total_calls)
        records.append({
            "judge": judge,
            "language": lang,
            "setting": setting,
            "parse_rate": parse_rate,
            "total_api_calls": total_calls,
            "failed_parses": failed_parses,
        })
    return records


# ── Main scatter plot ─────────────────────────────────────────────────────────

def plot_parse_failure(records):
    group_width = len(SETTINGS)
    lang_to_x = {lang: idx for idx, lang in enumerate(LANGUAGES)}
    setting_offsets = {s: i for i, s in enumerate(SETTINGS)}

    traces = {}
    for rec in records:
        jm = rec["judge"]
        x = lang_to_x[rec["language"]] * group_width + setting_offsets[rec["setting"]]
        y = rec["parse_rate"] * 100
        hover = (
            f"<b>{JUDGE_DISPLAY.get(jm, jm)}</b><br>"
            f"Language: {LANG_DISPLAY.get(rec['language'], rec['language'])}<br>"
            f"Setting: {rec['setting']}<br>"
            f"Parse failure rate: <b>{y:.2f}%</b><br>"
            f"Failed parses: {rec['failed_parses']} / {rec['total_api_calls']}"
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
                fillcolor="rgba(100,100,255,0.05)", line_width=0, layer="below"
            )
        if i > 0:
            fig.add_vline(
                x=base - 0.5,
                line=dict(color="rgba(150,150,150,0.4)", width=1, dash="dot")
            )

    # Reference lines
    fig.add_hline(y=0, line=dict(color="rgba(60,60,60,0.25)", width=1))
    fig.add_hline(y=5, line=dict(color="rgba(255,165,0,0.5)", width=1.5, dash="dot"),
                  annotation_text="5% threshold", annotation_position="right")
    fig.add_hline(y=10, line=dict(color="rgba(220,50,50,0.5)", width=1.5, dash="dot"),
                  annotation_text="10% threshold", annotation_position="right")

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

    tick_vals, tick_text = [], []
    for lang in LANGUAGES:
        base = lang_to_x[lang] * group_width
        for setting in SETTINGS:
            tick_vals.append(base + setting_offsets[setting])
            tick_text.append(setting.capitalize())

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
            text="LLM-as-a-Judge Parse Failure Rate<br>"
                 "<sup>Fraction of API calls that could not be parsed</sup>",
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
            title=dict(text="Parse Failure Rate (%)", font=dict(size=13)),
            gridcolor="rgba(200,200,200,0.3)", zeroline=False,
            range=[-0.5, 26],
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
        width=1200, height=540,
        margin=dict(l=70, r=200, t=100, b=90),
        hovermode="closest",
    )

    return fig


# ── Heatmap: average parse failure per judge × setting ────────────────────────

def plot_heatmap(records):
    import pandas as pd
    df = pd.DataFrame(records)
    if df.empty:
        return None

    df["judge_display"] = df["judge"].map(lambda j: JUDGE_DISPLAY.get(j, j))
    pivot = df.groupby(["judge_display", "setting"])["parse_rate"].mean().unstack(fill_value=0)
    pivot = pivot * 100  # to percent

    judges = [JUDGE_DISPLAY.get(j, j) for j in JUDGE_MODELS if JUDGE_DISPLAY.get(j, j) in pivot.index]
    settings = SETTINGS

    z = [[pivot.loc[j, s] if j in pivot.index and s in pivot.columns else 0
          for s in settings] for j in judges]

    text = [[f"{v:.2f}%" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[s.capitalize() for s in settings],
        y=judges,
        colorscale="YlOrRd",
        zmin=0,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="Judge: %{y}<br>Setting: %{x}<br>Avg failure rate: %{z:.2f}%<extra></extra>",
        colorbar=dict(title="Failure rate (%)"),
    ))

    fig.update_layout(
        title=dict(
            text="Average Parse Failure Rate — Judge × Setting Heatmap<br>"
                 "<sup>Averaged over all languages</sup>",
            font=dict(size=18, family="Inter, Arial, sans-serif"),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(title="Prompt Setting", side="bottom"),
        yaxis=dict(title="Judge Model", autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif"),
        width=700, height=500,
        margin=dict(l=200, r=100, t=100, b=60),
    )

    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("Loading run_stats.json files …")
    records = load_records()
    if not records:
        print("No run_stats.json files found. Check data directory structure.")
        return

    print(f"Loaded {len(records)} records.")

    print("Building parse failure scatter plot…")
    fig_scatter = plot_parse_failure(records)
    out = os.path.join(OUTPUT_DIR, "llm_judge_parsing.html")
    fig_scatter.write_html(out)
    print(f"  → saved {out}")
    fig_scatter.show()

    print("Building heatmap…")
    fig_heat = plot_heatmap(records)
    if fig_heat:
        out2 = os.path.join(OUTPUT_DIR, "llm_judge_parsing_heatmap.html")
        fig_heat.write_html(out2)
        print(f"  → saved {out2}")
        fig_heat.show()


if __name__ == "__main__":
    main()
