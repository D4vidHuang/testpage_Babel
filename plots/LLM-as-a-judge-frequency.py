"""
LLM-as-a-Judge Error Frequency Visualization
=============================================
Interactive Bokeh dashboard showing:
  1. Error-code frequency heatmap: judge × error-code (one tab per LLM / setting)
  2. LG-GR4 hallucination frequency bar chart per target LLM
  3. Accuracy distribution comparison: human vs all judge models

Data: data/raw_{Language}.parquet
Output: opens an interactive Bokeh HTML in browser; also saves it.
"""

import ast
import json
import os

import numpy as np
import pandas as pd
from bokeh.io import output_file, show
from bokeh.layouts import column, row, gridplot
from bokeh.models import (
    BasicTicker,
    ColorBar,
    ColumnDataSource,
    FactorRange,
    HoverTool,
    LinearColorMapper,
    PrintfTickFormatter,
    Tabs,
    TabPanel,
    Title,
)
from bokeh.palettes import Blues256, RdYlGn11, Spectral11
from bokeh.plotting import figure
from bokeh.transform import factor_cmap, linear_cmap

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LANGUAGE = "Polish"           # Default language for single-language plots
PARQUET_PATH = os.path.join(DATA_DIR, f"raw_{LANGUAGE}.parquet")

TARGET_LLMS = [
    "google/codegemma-7b",
    "meta-llama/CodeLlama-7b-hf",
    "Qwen/CodeQwen1.5-7B",
    "bigcode/starcoder2-7b",
    "ibm-granite/granite-8b-code-base",
]

LLM_SHORT = {
    "google/codegemma-7b": "CodeGemma",
    "meta-llama/CodeLlama-7b-hf": "CodeLlama",
    "Qwen/CodeQwen1.5-7B": "CodeQwen",
    "bigcode/starcoder2-7b": "StarCoder2",
    "ibm-granite/granite-8b-code-base": "Granite",
}

JUDGE_SETTINGS = ["cot", "hierarchical", "standard", "rubric"]

JUDGE_MODELS = [
    "openai_gpt-oss-120b_floor",
    "meituan_longcat-flash-chat_floor",
    "deepseek_deepseek-v3_2_floor",
    "google_gemini-3-flash-preview_floor",
    "openai_gpt-oss-20b_floor",
    "anthropic_claude-haiku-4_5_floor",
    "qwen_qwen3-coder-next_floor",
    "qwen_qwen3-vl-235b-a22b-instruct_floor",
    "x-ai_grok-4_1-fast_floor",
]

JUDGE_DISPLAY = {
    "openai_gpt-oss-120b_floor": "GPT-OSS 120B",
    "meituan_longcat-flash-chat_floor": "LongCat Flash",
    "deepseek_deepseek-v3_2_floor": "DeepSeek V3",
    "google_gemini-3-flash-preview_floor": "Gemini Flash",
    "openai_gpt-oss-20b_floor": "GPT-OSS 20B",
    "anthropic_claude-haiku-4_5_floor": "Claude Haiku",
    "qwen_qwen3-coder-next_floor": "Qwen3 Coder",
    "qwen_qwen3-vl-235b-a22b-instruct_floor": "Qwen3 VL 235B",
    "x-ai_grok-4_1-fast_floor": "Grok 4.1 Fast",
}

ERROR_CODES = [
    "MS-IG", "MS-CC", "MS-ME1", "MS-ME2", "MS-ME3", "MS-ET", "MS-LT", "MS-RE1", "MS-RE2",
    "LG-GR1", "LG-GR2", "LG-GR3", "LG-GR4", "LG-GR5", "LG-IS", "LG-WL",
    "SE-MD", "SE-TS", "SE-HA1", "SE-HA2", "SE-HA3", "SE-CS1", "SE-CS2", "SE-OI",
    "ST-IF",
]

BOKEH_THEME = {
    "background": "#FAFBFC",
    "grid_color": "#E8EBF0",
    "text_color": "#2C3E50",
    "accent": "#3498DB",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_errors(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "none" or text == "[]":
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
        return [x.strip() for x in text.split(",") if x.strip()]
    return []


def bokeh_figure_defaults(title, width=900, height=450, **kwargs):
    p = figure(
        title=title,
        width=width, height=height,
        background_fill_color=BOKEH_THEME["background"],
        border_fill_color="white",
        toolbar_location="above",
        **kwargs,
    )
    p.title.text_font_size = "15pt"
    p.title.text_color = BOKEH_THEME["text_color"]
    p.title.text_font = "Inter, Arial, sans-serif"
    p.xgrid.grid_line_color = BOKEH_THEME["grid_color"]
    p.ygrid.grid_line_color = BOKEH_THEME["grid_color"]
    p.axis.axis_label_text_font_size = "11pt"
    p.axis.major_label_text_font_size = "10pt"
    return p


# ── Plot 1: LG-GR4 frequency per target LLM ──────────────────────────────────

def plot_lg_gr4_frequency(df):
    freqs = []
    labels = []
    for llm in TARGET_LLMS:
        judge_error_cols = [
            col for col in df.columns
            if col.endswith("_errors") and llm in col
        ]
        if not judge_error_cols:
            continue

        lg_gr4_count = 0
        total_items = 0
        for col in judge_error_cols:
            error_lists = [normalize_errors(v) for v in df[col].values]
            lg_gr4_count += sum(1 for errs in error_lists if "LG-GR4" in errs)
            total_items += len(error_lists)

        freq = lg_gr4_count / total_items if total_items else 0.0
        freqs.append(freq * 100)
        labels.append(LLM_SHORT.get(llm, llm.split("/")[-1]))

    if not labels:
        return None

    palette = ["#4285F4", "#34A853", "#FBBC05", "#EA4335", "#9B59B6"]
    source = ColumnDataSource(dict(
        x=labels,
        top=freqs,
        color=palette[:len(labels)],
        freq_str=[f"{f:.2f}%" for f in freqs],
    ))

    p = bokeh_figure_defaults(
        "LG-GR4 Error Frequency per Target LLM",
        x_range=labels, height=420,
    )
    p.vbar(
        x="x", top="top", width=0.6,
        source=source,
        color="color",
        alpha=0.85,
        line_color="white",
        line_width=1.5,
    )
    p.add_tools(HoverTool(tooltips=[
        ("LLM", "@x"),
        ("LG-GR4 frequency", "@freq_str"),
    ]))
    p.xaxis.axis_label = "Target LLM"
    p.yaxis.axis_label = "Frequency (%)"
    p.xaxis.major_label_orientation = 0.4
    p.y_range.start = 0
    p.y_range.end = max(freqs) * 1.2 if freqs else 10
    p.toolbar.logo = None

    return p


# ── Plot 2: Error code frequency heatmap (judge × error-code) ────────────────

def plot_error_heatmap(df, target_llm, setting):
    llm_short = LLM_SHORT.get(target_llm, target_llm.split("/")[-1])

    judge_error_cols = [
        col for col in df.columns
        if col.endswith("_errors") and target_llm in col and setting in col
    ]
    human_error_cols = [
        col for col in df.columns
        if col.startswith("error_codes") and target_llm in col
    ]

    all_cols = judge_error_cols + human_error_cols
    rater_names = []
    for col in judge_error_cols:
        # Extract judge model name from column
        for jm in JUDGE_MODELS:
            if col.startswith(jm):
                rater_names.append(JUDGE_DISPLAY.get(jm, jm))
                break
        else:
            rater_names.append(col.split("_")[0])
    rater_names += ["Human Expert"] * len(human_error_cols)

    freq_matrix = []
    for col in all_cols:
        counts = {ec: 0 for ec in ERROR_CODES}
        total = len(df)
        for errors in df[col].values:
            for e in normalize_errors(errors):
                if e in counts:
                    counts[e] += 1
        row_freqs = [counts[ec] / max(1, total) * 100 for ec in ERROR_CODES]
        freq_matrix.append(row_freqs)

    if not freq_matrix:
        return None

    # Flatten for ColumnDataSource
    xs, ys, vals, val_strs = [], [], [], []
    for ri, rater in enumerate(rater_names):
        for ci, code in enumerate(ERROR_CODES):
            v = freq_matrix[ri][ci]
            xs.append(code)
            ys.append(rater)
            vals.append(v)
            val_strs.append(f"{v:.1f}%")

    mapper = LinearColorMapper(
        palette=Blues256[::-1],
        low=0, high=max(vals) if vals else 1,
    )

    source = ColumnDataSource(dict(x=xs, y=ys, vals=vals, val_strs=val_strs))

    p = bokeh_figure_defaults(
        f"Error Code Frequency — {llm_short} / {setting.capitalize()} ({LANGUAGE})",
        x_range=ERROR_CODES,
        y_range=list(reversed(rater_names)),
        width=1100, height=max(300, 50 * len(rater_names) + 80),
        x_axis_location="above",
    )
    p.rect(
        x="x", y="y", width=1, height=1,
        source=source,
        fill_color=linear_cmap("vals", Blues256[::-1], 0, max(vals) if vals else 1),
        line_color=None,
    )
    p.add_tools(HoverTool(tooltips=[
        ("Rater", "@y"),
        ("Error Code", "@x"),
        ("Frequency", "@val_strs"),
    ]))

    color_bar = ColorBar(
        color_mapper=mapper,
        ticker=BasicTicker(desired_num_ticks=6),
        formatter=PrintfTickFormatter(format="%.1f%%"),
        label_standoff=8,
        title="Frequency (%)",
    )
    p.add_layout(color_bar, "right")
    p.xaxis.major_label_orientation = 0.7
    p.toolbar.logo = None

    return p


# ── Plot 3: Accuracy distribution comparison ──────────────────────────────────

def plot_accuracy_distribution(df, target_llm, setting):
    llm_short = LLM_SHORT.get(target_llm, target_llm.split("/")[-1])
    human_accuracy_cols = [
        col for col in df.columns
        if col.startswith("expert_accuracy") and target_llm in col
    ]
    judge_accuracy_cols = [
        col for col in df.columns
        if col.endswith("_accuracy") and target_llm in col and setting in col
        and not col.startswith("expert")
    ]

    categories = ["Incorrect", "Partial", "Correct"]
    cat_map_human = {"Correct": "Correct", "Partial": "Partial", "Incorrect": "Incorrect"}
    cat_map_judge = {"correct": "Correct", "partially_correct": "Partial", "incorrect": "Incorrect"}

    groups = []
    counts_list = []
    colors_list = []
    palette = ["#EA4335", "#FBBC05", "#34A853"]  # wrong/partial/right

    if human_accuracy_cols:
        human_vals = df[human_accuracy_cols[0]].map(cat_map_human)
        counts = [int((human_vals == cat).sum()) for cat in categories]
        groups.append("Human Expert")
        counts_list.append(counts)
        colors_list.append(palette)

    judge_palette = [
        "#4285F4", "#FF6B9D", "#FF9500", "#9B59B6",
        "#00BCD4", "#795548", "#607D8B", "#E91E63", "#009688"
    ]

    for i, col in enumerate(judge_accuracy_cols[:8]):
        judge_vals = df[col].map(cat_map_judge)
        counts = [int((judge_vals == cat).sum()) for cat in categories]
        # Find judge display name
        rater_label = col
        for jm in JUDGE_MODELS:
            if col.startswith(jm):
                rater_label = JUDGE_DISPLAY.get(jm, jm)
                break
        groups.append(rater_label)
        counts_list.append(counts)
        colors_list.append(palette)

    if not groups:
        return None

    x = [(g, cat) for g in groups for cat in categories]
    vals = [cnt for counts in counts_list for cnt in counts]

    source = ColumnDataSource(dict(
        x=x,
        top=vals,
        category=[v[1] for v in x],
        group=[v[0] for v in x],
        color=[c for colors in colors_list for c in colors],
    ))

    p = bokeh_figure_defaults(
        f"Accuracy Distribution — {llm_short} / {setting.capitalize()} ({LANGUAGE})",
        x_range=FactorRange(*x),
        height=480, width=max(900, 120 * len(groups)),
    )
    p.vbar(
        x="x", top="top", width=0.8,
        source=source,
        color="color",
        alpha=0.85,
        line_color="white",
        line_width=1.2,
    )
    p.add_tools(HoverTool(tooltips=[
        ("Rater", "@group"),
        ("Category", "@category"),
        ("Count", "@top"),
    ]))
    p.xaxis.major_label_orientation = 0.6
    p.xaxis.major_label_text_font_size = "9pt"
    p.xaxis.group_text_font_size = "10pt"
    p.xaxis.group_text_font_style = "bold"
    p.yaxis.axis_label = "Count"
    p.toolbar.logo = None

    return p


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(PARQUET_PATH):
        print(f"Parquet file not found: {PARQUET_PATH}")
        return

    print(f"Loading {PARQUET_PATH} …")
    df = pd.read_parquet(PARQUET_PATH)

    out = os.path.join(OUTPUT_DIR, "llm_judge_frequency.html")
    output_file(out, title="LLM-as-a-Judge Error Frequency")

    panels = []

    # Tab 1: LG-GR4 frequency
    print("Building LG-GR4 frequency plot…")
    p_lgr4 = plot_lg_gr4_frequency(df)
    if p_lgr4:
        panels.append(TabPanel(child=p_lgr4, title="LG-GR4 Frequency"))

    # Tab 2: Accuracy distribution (default: first LLM, first setting)
    print("Building accuracy distribution plot…")
    p_acc = plot_accuracy_distribution(df, TARGET_LLMS[0], JUDGE_SETTINGS[0])
    if p_acc:
        panels.append(TabPanel(child=p_acc, title="Accuracy Distribution"))

    # Tab 3-N: Error heatmaps per LLM
    for llm in TARGET_LLMS[:2]:   # Limit to first 2 to keep file size manageable
        for setting in JUDGE_SETTINGS[:2]:
            print(f"Building error heatmap for {llm} / {setting}…")
            p_heat = plot_error_heatmap(df, llm, setting)
            if p_heat:
                label = f"{LLM_SHORT.get(llm, llm)} / {setting}"
                panels.append(TabPanel(child=p_heat, title=label))

    if not panels:
        print("No plots could be generated.")
        return

    tabs = Tabs(tabs=panels)
    show(tabs)
    print(f"  → saved {out}")


if __name__ == "__main__":
    main()
