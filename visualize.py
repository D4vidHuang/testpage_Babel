"""
EMSE-Babel Data Visualization Dashboard
========================================
Generates beautiful visualizations using Plotly and Bokeh for the multilingual
code comment quality evaluation dataset.
"""

import pandas as pd
import numpy as np
import json
import re
from collections import Counter
from pathlib import Path

# ── Plotly ──────────────────────────────────────────────────────────────────
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Bokeh ───────────────────────────────────────────────────────────────────
from bokeh.plotting import figure, output_file, save
from bokeh.models import (
    ColumnDataSource, HoverTool, ColorBar, LinearColorMapper,
    FactorRange, Legend, LegendItem, Div, Range1d
)
from bokeh.layouts import column, row, gridplot
from bokeh.palettes import Viridis256, Category10, Category20, Spectral6
from bokeh.transform import factor_cmap, dodge
from bokeh.io import export_png
from bokeh.themes import Theme

# ── Config ───────────────────────────────────────────────────────────────────
LANGUAGES   = ['Chinese', 'Dutch', 'English', 'Greek', 'Polish']
MODELS_FULL = [
    'Qwen/CodeQwen1.5-7B',
    'bigcode/starcoder2-7b',
    'ibm-granite/granite-8b-code-base',
    'meta-llama/CodeLlama-7b-hf',
    'google/codegemma-7b',
]
MODEL_SHORT = {
    'Qwen/CodeQwen1.5-7B':                'CodeQwen',
    'bigcode/starcoder2-7b':              'StarCoder2',
    'ibm-granite/granite-8b-code-base':   'Granite',
    'meta-llama/CodeLlama-7b-hf':         'CodeLlama',
    'google/codegemma-7b':                'CodeGemma',
}

# Colour palettes
LANG_COLORS   = ['#6E40AA', '#1FA8C9', '#27A888', '#F59B21', '#E23F31']
MODEL_COLORS  = ['#7C3AED', '#2563EB', '#059669', '#D97706', '#DC2626']
STATUS_COLORS = {'Correct': '#10B981', 'Partial': '#F59E0B', 'Incorrect': '#EF4444'}

OUT_DIR = Path("visualization_output")
OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
print("📦 Loading parquet files …")
dfs = {}
for lang in LANGUAGES:
    dfs[lang] = pd.read_parquet(f"data/raw_{lang}.parquet")
print(f"   Loaded {sum(len(v) for v in dfs.values()):,} total rows across {len(LANGUAGES)} languages.\n")

# Load taxonomy
with open("taxonomy/error_taxonomy.json") as f:
    taxonomy = json.load(f)

# Build error-code → full name map
error_name_map = {}
category_map   = {}   # code → category
for category, items in taxonomy.items():
    if category == "":
        continue
    for item in items:
        code = item["id"]
        error_name_map[code] = item["name"]
        category_map[code]   = category

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA PRE-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def accuracy_rates(df, models=MODELS_FULL):
    """Return dict {model: {Correct/Partial/Incorrect: fraction}}."""
    out = {}
    for m in models:
        col = f"expert_accuracy_{m}"
        if col not in df.columns:
            continue
        vc = df[col].value_counts(normalize=True)
        out[m] = {
            "Correct":   vc.get("Correct",   0.0),
            "Partial":   vc.get("Partial",   0.0),
            "Incorrect": vc.get("Incorrect", 0.0),
        }
    return out


def gather_errors(df, models=MODELS_FULL):
    """Collect all error codes from expert_error_codes columns."""
    errors = []
    for m in models:
        col = f"error_codes_{m}"
        if col not in df.columns:
            continue
        for val in df[col].dropna():
            codes = [e.strip() for e in str(val).split(",") if e.strip()]
            errors.extend(codes)
    return errors


def bartscore_summary(df, scorer="Salesforce__codet5-small",
                       ctx="ctx-full", noise="noise-none",
                       models=MODELS_FULL):
    """Return {model: mean_bart_score_avg}."""
    out = {}
    for m in models:
        col = f"bartscore_{scorer}_{ctx}_{noise}_{m}_bart_score_avg"
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals):
                out[m] = float(vals.mean())
    return out


# Pre-compute
lang_acc   = {lang: accuracy_rates(dfs[lang]) for lang in LANGUAGES}
lang_errs  = {lang: gather_errors(dfs[lang])  for lang in LANGUAGES}
lang_bart  = {lang: bartscore_summary(dfs[lang]) for lang in LANGUAGES}

# Correct-rate matrix  [language × model]
correct_matrix = pd.DataFrame(
    {lang: {MODEL_SHORT[m]: lang_acc[lang].get(m, {}).get("Correct", 0.0) * 100
            for m in MODELS_FULL}
     for lang in LANGUAGES}
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. PLOTLY FIGURES
# ─────────────────────────────────────────────────────────────────────────────

PLOTLY_TEMPLATE = dict(
    paper_bgcolor="#0F172A",
    plot_bgcolor="#1E293B",
    font=dict(family="Inter, Helvetica Neue, sans-serif", color="#E2E8F0"),
    title_font=dict(size=22, color="#F8FAFC"),
    legend=dict(bgcolor="rgba(30,41,59,0.8)", bordercolor="#334155",
                borderwidth=1, font=dict(color="#CBD5E1")),
    coloraxis_colorbar=dict(tickfont=dict(color="#CBD5E1"),
                            title_font=dict(color="#CBD5E1")),
)


# ── 3A. Grouped Stacked Bars – Accuracy per Language (ALL models) ─────────────
print("📊 [Plotly] Figure 1 – Accuracy stacked bars …")

fig1 = make_subplots(
    rows=1, cols=len(LANGUAGES),
    subplot_titles=[f"<b>{l}</b>" for l in LANGUAGES],
    shared_yaxes=True,
    horizontal_spacing=0.03,
)

STATUS_ORDER = ["Correct", "Partial", "Incorrect"]

for col_idx, lang in enumerate(LANGUAGES, 1):
    acc = lang_acc[lang]
    for si, status in enumerate(STATUS_ORDER):
        y_vals = [acc.get(m, {}).get(status, 0) * 100 for m in MODELS_FULL]
        fig1.add_trace(
            go.Bar(
                name=status,
                x=[MODEL_SHORT[m] for m in MODELS_FULL],
                y=y_vals,
                marker_color=list(STATUS_COLORS.values())[si],
                text=[f"{v:.0f}%" for v in y_vals],
                textposition="inside",
                textfont=dict(size=9, color="white"),
                showlegend=(col_idx == 1),
                legendgroup=status,
            ),
            row=1, col=col_idx,
        )

fig1.update_layout(
    barmode="stack",
    title_text="Expert-Evaluated Accuracy by Language & Model",
    height=520,
    **PLOTLY_TEMPLATE,
    xaxis_tickangle=-35,
)
for i in range(1, len(LANGUAGES) + 1):
    fig1.update_xaxes(tickfont=dict(size=9), row=1, col=i)
fig1.update_yaxes(title_text="Percentage (%)", row=1, col=1,
                  gridcolor="#334155")

fig1.write_html(str(OUT_DIR / "plotly_1_accuracy_stacked.html"))
print("   ✓ saved plotly_1_accuracy_stacked.html")


# ── 3B. Heatmap – Correct Rate [Language × Model] ────────────────────────────
print("📊 [Plotly] Figure 2 – Correct-rate heatmap …")

z  = correct_matrix.values
xl = list(correct_matrix.columns)
yl = list(correct_matrix.index)

annotations = []
for i, model in enumerate(yl):
    for j, lang in enumerate(xl):
        annotations.append(dict(
            x=lang, y=model,
            text=f"{z[i,j]:.1f}%",
            showarrow=False,
            font=dict(color="white" if z[i,j] < 60 else "#0F172A", size=13),
        ))

fig2 = go.Figure(data=go.Heatmap(
    z=z,
    x=xl,
    y=yl,
    colorscale="Viridis",
    text=np.round(z, 1),
    hovertemplate="Language: %{x}<br>Model: %{y}<br>Correct: %{z:.1f}%<extra></extra>",
    colorbar=dict(title="Correct %", ticksuffix="%", tickfont=dict(color="#CBD5E1")),
))
fig2.update_layout(
    title_text="Correct Rate Heatmap (Language × Model)",
    height=400,
    annotations=annotations,
    **PLOTLY_TEMPLATE,
)
fig2.write_html(str(OUT_DIR / "plotly_2_correctrate_heatmap.html"))
print("   ✓ saved plotly_2_correctrate_heatmap.html")


# ── 3C. Radar / Spider – per-language model performance ──────────────────────
print("📊 [Plotly] Figure 3 – Radar chart …")

fig3 = go.Figure()
theta_labels = LANGUAGES + [LANGUAGES[0]]   # close the loop

for mi, m in enumerate(MODELS_FULL):
    r_vals = [lang_acc[lang].get(m, {}).get("Correct", 0) * 100 for lang in LANGUAGES]
    r_vals += [r_vals[0]]
    fig3.add_trace(go.Scatterpolar(
        r=r_vals,
        theta=theta_labels,
        fill="toself",
        name=MODEL_SHORT[m],
        line_color=MODEL_COLORS[mi],
        fillcolor=MODEL_COLORS[mi].replace(")", ",0.15)").replace("rgb(", "rgba("),
        opacity=0.8,
    ))

fig3.update_layout(
    polar=dict(
        bgcolor="#1E293B",
        radialaxis=dict(visible=True, range=[0, 100], gridcolor="#334155",
                        ticksuffix="%", tickfont=dict(color="#94A3B8")),
        angularaxis=dict(gridcolor="#334155", tickfont=dict(color="#CBD5E1", size=13)),
    ),
    title_text="Model Correct Rate by Language (Spider Chart)",
    height=520,
    **PLOTLY_TEMPLATE,
)
fig3.write_html(str(OUT_DIR / "plotly_3_radar_chart.html"))
print("   ✓ saved plotly_3_radar_chart.html")


# ── 3D. Treemap – Error-code frequency across ALL languages ─────────────────
print("📊 [Plotly] Figure 4 – Error treemap …")

all_errs_combined = []
for lang in LANGUAGES:
    all_errs_combined.extend(lang_errs[lang])

err_counter = Counter(all_errs_combined)

# Build treemap records
records = []
for code, cnt in err_counter.items():
    cat = category_map.get(code, "Other")
    name = error_name_map.get(code, code)
    records.append(dict(id=code, parent=cat, value=cnt, label=f"{code}<br>{name}"))

# Add category level (parent = "")
categories = list({r["parent"] for r in records})
for cat in categories:
    records.append(dict(id=cat, parent="", value=0, label=cat))

ids, parents, values, labels = [], [], [], []
for r in records:
    ids.append(r["id"])
    parents.append(r["parent"])
    values.append(r["value"])
    labels.append(r["label"])

fig4 = go.Figure(go.Treemap(
    ids=ids,
    labels=labels,
    parents=parents,
    values=values,
    branchvalues="remainder",
    hovertemplate="<b>%{label}</b><br>Occurrences: %{value}<extra></extra>",
    marker=dict(colorscale="Plasma", line=dict(color="#0F172A", width=2)),
    textfont=dict(size=12),
    tiling=dict(pad=4),
))
fig4.update_layout(
    title_text="Error Code Frequency Treemap (All Languages)",
    height=550,
    margin=dict(t=60, l=10, r=10, b=10),
    **PLOTLY_TEMPLATE,
)
fig4.write_html(str(OUT_DIR / "plotly_4_error_treemap.html"))
print("   ✓ saved plotly_4_error_treemap.html")


# ── 3E. BARTScore distribution – violin per model (English) ──────────────────
print("📊 [Plotly] Figure 5 – BARTScore violins …")

# Use a common scorer/config that exists in most languages
SCORER   = "Salesforce__codet5-small"
CTX      = "ctx-full"
NOISE    = "noise-none"

fig5 = go.Figure()
for mi, m in enumerate(MODELS_FULL):
    col_name = f"bartscore_{SCORER}_{CTX}_{NOISE}_{m}_bart_score_avg"
    vals_all = []
    for lang in LANGUAGES:
        if col_name in dfs[lang].columns:
            v = dfs[lang][col_name].dropna().tolist()
            vals_all.extend(v)
    if vals_all:
        fig5.add_trace(go.Violin(
            y=vals_all,
            name=MODEL_SHORT[m],
            box_visible=True,
            meanline_visible=True,
            line_color=MODEL_COLORS[mi],
            fillcolor=MODEL_COLORS[mi],
            opacity=0.6,
            points=False,
        ))

fig5.update_layout(
    title_text="BARTScore Distribution by Model (All Languages, CodeT5-small scorer)",
    yaxis_title="BARTScore Average",
    height=480,
    violinmode="overlay",
    **PLOTLY_TEMPLATE,
    yaxis=dict(gridcolor="#334155"),
)
fig5.write_html(str(OUT_DIR / "plotly_5_bartscore_violin.html"))
print("   ✓ saved plotly_5_bartscore_violin.html")


# ── 3F. Sunburst – Error taxonomy by language ─────────────────────────────────
print("📊 [Plotly] Figure 6 – Sunburst error taxonomy …")

sun_ids, sun_parents, sun_values, sun_labels = ["ALL"], [""], [0], ["All Languages"]

for lang in LANGUAGES:
    lang_total = sum(Counter(lang_errs[lang]).values())
    sun_ids.append(lang)
    sun_parents.append("ALL")
    sun_values.append(lang_total)
    sun_labels.append(lang)

    cat_counts = Counter()
    for code in lang_errs[lang]:
        cat = category_map.get(code, "Other")
        cat_counts[cat] += 1

    for cat, cnt in cat_counts.items():
        uid = f"{lang}_{cat}"
        sun_ids.append(uid)
        sun_parents.append(lang)
        sun_values.append(cnt)
        sun_labels.append(cat)

fig6 = go.Figure(go.Sunburst(
    ids=sun_ids,
    labels=sun_labels,
    parents=sun_parents,
    values=sun_values,
    branchvalues="remainder",
    hovertemplate="<b>%{label}</b><br>Errors: %{value}<extra></extra>",
    marker=dict(colorscale="RdYlGn", line=dict(color="#0F172A", width=1)),
    insidetextfont=dict(size=11),
    leaf=dict(opacity=0.85),
))
fig6.update_layout(
    title_text="Error Category Distribution by Language (Sunburst)",
    height=550,
    **PLOTLY_TEMPLATE,
    margin=dict(t=60, l=0, r=0, b=0),
)
fig6.write_html(str(OUT_DIR / "plotly_6_sunburst.html"))
print("   ✓ saved plotly_6_sunburst.html")


# ── 3G. Line – BARTScore mean per language×model ─────────────────────────────
print("📊 [Plotly] Figure 7 – BARTScore per-language line chart …")

fig7 = go.Figure()
for mi, m in enumerate(MODELS_FULL):
    col_name = f"bartscore_{SCORER}_{CTX}_{NOISE}_{m}_bart_score_avg"
    y_means = []
    x_langs = []
    for lang in LANGUAGES:
        if col_name in dfs[lang].columns:
            v = dfs[lang][col_name].dropna()
            if len(v):
                y_means.append(float(v.mean()))
                x_langs.append(lang)
    if y_means:
        fig7.add_trace(go.Scatter(
            x=x_langs, y=y_means,
            mode="lines+markers",
            name=MODEL_SHORT[m],
            line=dict(color=MODEL_COLORS[mi], width=2.5),
            marker=dict(size=9, color=MODEL_COLORS[mi], line=dict(width=2, color="white")),
        ))

fig7.update_layout(
    title_text="Mean BARTScore per Language & Model",
    xaxis_title="Language",
    yaxis_title="Mean BARTScore",
    height=450,
    **PLOTLY_TEMPLATE,
    yaxis=dict(gridcolor="#334155"),
    xaxis=dict(gridcolor="#334155"),
)
fig7.write_html(str(OUT_DIR / "plotly_7_bartscore_lines.html"))
print("   ✓ saved plotly_7_bartscore_lines.html")


# ─────────────────────────────────────────────────────────────────────────────
# 4. BOKEH FIGURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n📊 [Bokeh] Building visualizations …")

# ── Helper theme settings ─────────────────────────────────────────────────────
BK_BG   = "#0F172A"
BK_PLT  = "#1E293B"
BK_GRID = "#334155"
BK_TEXT = "#E2E8F0"
BK_W, BK_H = 800, 420


def style_figure(p):
    """Apply dark theme to a bokeh figure."""
    p.background_fill_color = BK_PLT
    p.border_fill_color      = BK_BG
    p.outline_line_color     = None
    p.grid.grid_line_color   = BK_GRID
    p.grid.grid_line_alpha   = 0.5
    p.axis.axis_label_text_color   = BK_TEXT
    p.axis.major_label_text_color  = BK_TEXT
    p.axis.axis_line_color  = "#475569"
    p.title.text_color       = "#F8FAFC"
    p.title.text_font_size   = "15px"
    p.title.text_font        = "Inter, Helvetica Neue, sans-serif"
    return p


# ── 4A. Grouped Bar – per-model Correct rate across languages ─────────────────
print("   [Bokeh] Figure A – Grouped bar correct rates …")

bk_langs  = LANGUAGES
bk_models = [MODEL_SHORT[m] for m in MODELS_FULL]
x_range   = FactorRange(*[(lang, mod) for lang in bk_langs for mod in bk_models])

bk_correct = {}
for m in MODELS_FULL:
    bk_correct[MODEL_SHORT[m]] = [
        lang_acc[lang].get(m, {}).get("Correct", 0) * 100 for lang in LANGUAGES
    ]

data_bkA = dict(x=[(lang, mod) for lang in bk_langs for mod in bk_models])
for m in MODELS_FULL:
    data_bkA[MODEL_SHORT[m]] = bk_correct[MODEL_SHORT[m]]

source_A = ColumnDataSource(dict(
    x=data_bkA["x"],
    top=[bk_correct[MODEL_SHORT[m]][li] for li in range(len(LANGUAGES)) for m in MODELS_FULL],
    model=[mod for lang in bk_langs for mod in bk_models],
    lang=[lang for lang in bk_langs for mod in bk_models],
    color=[MODEL_COLORS[MODELS_FULL.index(m)] for li in range(len(LANGUAGES)) for m in MODELS_FULL],
))

pA = figure(
    x_range=x_range,
    title="Correct Rate by Language & Model",
    height=BK_H, width=BK_W + 100,
    toolbar_location="above",
    tools="pan,wheel_zoom,box_zoom,reset,save",
)
bars = pA.vbar(
    x="x", top="top",
    width=0.85,
    source=source_A,
    color="color",
    alpha=0.85,
    line_color="white",
    line_width=0.5,
)
hover_A = HoverTool(renderers=[bars], tooltips=[
    ("Language", "@lang"),
    ("Model",    "@model"),
    ("Correct",  "@top{0.1f}%"),
])
pA.add_tools(hover_A)
pA.xaxis.major_label_orientation = 0.9
pA.yaxis.axis_label = "Correct Rate (%)"
pA.xgrid.grid_line_color = None
style_figure(pA)

output_file(str(OUT_DIR / "bokeh_A_correct_rates.html"), title="Bokeh: Correct Rates")
save(pA)
print("   ✓ saved bokeh_A_correct_rates.html")


# ── 4B. Horizontal Bar – Top error codes (English) ───────────────────────────
print("   [Bokeh] Figure B – Top error codes …")

top_n    = 15
en_errs  = Counter(lang_errs["English"]).most_common(top_n)
codes_b  = [e[0] for e in reversed(en_errs)]
counts_b = [e[1] for e in reversed(en_errs)]
labels_b = [f"{c} – {error_name_map.get(c, c)}" for c in codes_b]
colors_b = [Spectral6[i % len(Spectral6)] for i in range(len(codes_b))]

source_B = ColumnDataSource(dict(
    y=labels_b, right=counts_b, code=codes_b, color=colors_b,
))
pB = figure(
    y_range=labels_b,
    title=f"Top {top_n} Error Codes – English",
    height=BK_H + 50, width=BK_W,
    toolbar_location="above",
    tools="pan,wheel_zoom,reset,save",
)
pB.hbar(y="y", right="right", height=0.6, source=source_B,
        color="color", alpha=0.85, line_color="white", line_width=0.5)
hover_B = HoverTool(tooltips=[("Code", "@code"), ("Count", "@right")])
pB.add_tools(hover_B)
pB.xaxis.axis_label = "Occurrences"
pB.ygrid.grid_line_color = None
style_figure(pB)

output_file(str(OUT_DIR / "bokeh_B_top_errors.html"), title="Bokeh: Top Errors")
save(pB)
print("   ✓ saved bokeh_B_top_errors.html")


# ── 4C. Heatmap-style scatter – BARTScore per language & model ───────────────
print("   [Bokeh] Figure C – BARTScore heatmap scatter …")

heat_records = []
for lang in LANGUAGES:
    for m in MODELS_FULL:
        col_name = f"bartscore_{SCORER}_{CTX}_{NOISE}_{m}_bart_score_avg"
        if col_name in dfs[lang].columns:
            v = dfs[lang][col_name].dropna()
            if len(v):
                heat_records.append(dict(
                    lang=lang,
                    model=MODEL_SHORT[m],
                    mean=float(v.mean()),
                    std=float(v.std()),
                ))

heat_df = pd.DataFrame(heat_records)
if not heat_df.empty:
    all_means = heat_df["mean"].tolist()
    mapper_C  = LinearColorMapper(
        palette=Viridis256,
        low=min(all_means),
        high=max(all_means),
    )
    source_C = ColumnDataSource(heat_df)
    pC = figure(
        x_range=LANGUAGES,
        y_range=list(reversed([MODEL_SHORT[m] for m in MODELS_FULL])),
        title="Mean BARTScore Heatmap (CodeT5-small scorer)",
        height=BK_H - 30, width=BK_W,
        toolbar_location="above",
        tools="hover,reset,save",
        tooltips=[("Language", "@lang"), ("Model", "@model"),
                  ("Mean BARTScore", "@mean{0.3f}"), ("Std", "@std{0.3f}")],
    )
    pC.rect(
        x="lang", y="model", width=0.98, height=0.98,
        source=source_C,
        fill_color={"field": "mean", "transform": mapper_C},
        line_color=None,
    )
    cb = ColorBar(color_mapper=mapper_C, location=(0, 0),
                  title="BARTScore",
                  title_text_color=BK_TEXT,
                  major_label_text_color=BK_TEXT)
    pC.add_layout(cb, "right")
    style_figure(pC)

    output_file(str(OUT_DIR / "bokeh_C_bartscore_heatmap.html"), title="Bokeh: BARTScore Heatmap")
    save(pC)
    print("   ✓ saved bokeh_C_bartscore_heatmap.html")


# ── 4D. Error categories comparison across languages ─────────────────────────
print("   [Bokeh] Figure D – Error categories across languages …")

categories_of_interest = ["Grammar", "Code Snippet", "Missing Info"]
cat_short = {"Grammar": "LG-GR", "Code Snippet": "SE-CS", "Missing Info": "MS"}

# Compute per-language cat counts (normalised by # errors)
cat_data = {cat: [] for cat in categories_of_interest}
for lang in LANGUAGES:
    errs = lang_errs[lang]
    total = max(len(errs), 1)
    for cat in categories_of_interest:
        prefix = cat_short.get(cat, "")
        cnt = sum(1 for e in errs if e.startswith(prefix))
        cat_data[cat].append(cnt / total * 100)

src_D = ColumnDataSource(dict(
    langs=LANGUAGES,
    **{cat: cat_data[cat] for cat in categories_of_interest},
))
cat_colors = ["#7C3AED", "#2563EB", "#059669"]

pD = figure(
    x_range=LANGUAGES,
    title="Error Category Rate by Language",
    height=BK_H, width=BK_W,
    toolbar_location="above",
    tools="pan,wheel_zoom,reset,save",
)
offsets = [-0.25, 0, 0.25]
renderers_D = []
for ci, (cat, offset, col) in enumerate(zip(categories_of_interest, offsets, cat_colors)):
    r = pD.vbar(
        x=dodge("langs", offset, range=pD.x_range),
        top=cat,
        width=0.22,
        source=src_D,
        color=col,
        alpha=0.85,
        legend_label=cat,
        line_color="white",
        line_width=0.5,
    )
    renderers_D.append(r)

hover_D = HoverTool(tooltips=[
    ("Language", "@langs"),
    ("Grammar",  "@Grammar{0.1f}%"),
    ("Code Snippet", "@{Code Snippet}{0.1f}%"),
    ("Missing Info", "@{Missing Info}{0.1f}%"),
])
pD.add_tools(hover_D)
pD.yaxis.axis_label = "% of All Errors"
pD.legend.location = "top_right"
pD.legend.background_fill_color = "rgba(30,41,59,0.8)"
pD.legend.label_text_color = BK_TEXT
pD.legend.border_line_color = "#334155"
style_figure(pD)

output_file(str(OUT_DIR / "bokeh_D_error_categories.html"), title="Bokeh: Error Categories")
save(pD)
print("   ✓ saved bokeh_D_error_categories.html")


# ─────────────────────────────────────────────────────────────────────────────
# 5. MASTER HTML DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
print("\n🌐 Building master dashboard HTML …")

PLOTLY_FILES = [
    ("plotly_1_accuracy_stacked.html",  "Expert Accuracy by Language & Model"),
    ("plotly_2_correctrate_heatmap.html","Correct Rate Heatmap"),
    ("plotly_3_radar_chart.html",       "Performance Radar Chart"),
    ("plotly_4_error_treemap.html",     "Error Taxonomy Treemap"),
    ("plotly_5_bartscore_violin.html",  "BARTScore Distributions"),
    ("plotly_6_sunburst.html",          "Error Category Sunburst"),
    ("plotly_7_bartscore_lines.html",   "BARTScore per Language"),
]
BOKEH_FILES = [
    ("bokeh_A_correct_rates.html",      "Correct Rates (Grouped Bar)"),
    ("bokeh_B_top_errors.html",         "Top Error Codes"),
    ("bokeh_C_bartscore_heatmap.html",  "BARTScore Heatmap"),
    ("bokeh_D_error_categories.html",   "Error Categories by Language"),
]

def make_iframe_section(title, fname):
    return f"""
    <section class="chart-section">
      <h2 class="chart-title">{title}</h2>
      <iframe
        src="{fname}"
        class="chart-frame"
        loading="lazy"
        title="{title}">
      </iframe>
    </section>"""

sections_plotly = "\n".join(make_iframe_section(t, f) for f, t in PLOTLY_FILES)
sections_bokeh  = "\n".join(make_iframe_section(t, f) for f, t in BOKEH_FILES)

# Quick-stats banner
total_samples = sum(len(dfs[l]) for l in LANGUAGES)
total_errors  = sum(len(lang_errs[l]) for l in LANGUAGES)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EMSE-Babel · Multilingual Code Comment Quality Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --bg-base:   #0F172A;
      --bg-card:   #1E293B;
      --bg-alt:    #162032;
      --border:    #334155;
      --accent1:   #7C3AED;
      --accent2:   #2563EB;
      --accent3:   #059669;
      --text-hi:   #F8FAFC;
      --text-mid:  #CBD5E1;
      --text-lo:   #64748B;
      --radius:    16px;
      --shadow:    0 8px 32px rgba(0,0,0,.5);
    }}

    html {{ scroll-behavior: smooth; }}

    body {{
      font-family: 'Inter', sans-serif;
      background: var(--bg-base);
      color: var(--text-mid);
      min-height: 100vh;
    }}

    /* ── Hero ────────────────────────────────────────────────────── */
    .hero {{
      background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #0F172A 100%);
      border-bottom: 1px solid var(--border);
      padding: 64px 40px 48px;
      text-align: center;
      position: relative;
      overflow: hidden;
    }}
    .hero::before {{
      content: '';
      position: absolute; inset: 0;
      background: radial-gradient(ellipse at 50% 0%, rgba(124,58,237,.25) 0%, transparent 70%);
      pointer-events: none;
    }}
    .hero-badge {{
      display: inline-block;
      background: rgba(124,58,237,.15);
      border: 1px solid rgba(124,58,237,.4);
      color: #A78BFA;
      font-size: .75rem;
      font-weight: 600;
      letter-spacing: .12em;
      text-transform: uppercase;
      padding: 4px 14px;
      border-radius: 999px;
      margin-bottom: 20px;
    }}
    .hero h1 {{
      font-size: clamp(2rem, 4vw, 3.25rem);
      font-weight: 900;
      color: var(--text-hi);
      line-height: 1.15;
      margin-bottom: 16px;
      background: linear-gradient(135deg, #F8FAFC, #A78BFA);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }}
    .hero p {{
      font-size: 1.1rem;
      color: var(--text-mid);
      max-width: 640px;
      margin: 0 auto 36px;
      line-height: 1.7;
    }}

    /* ── Stats bar ───────────────────────────────────────────────── */
    .stats-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      justify-content: center;
      margin-top: 8px;
    }}
    .stat-pill {{
      background: rgba(255,255,255,.05);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px 28px;
      text-align: center;
      min-width: 140px;
      backdrop-filter: blur(8px);
    }}
    .stat-pill .num {{
      font-size: 2rem;
      font-weight: 700;
      color: #A78BFA;
      display: block;
    }}
    .stat-pill .lbl {{
      font-size: .8rem;
      color: var(--text-lo);
      text-transform: uppercase;
      letter-spacing: .08em;
    }}

    /* ── Nav ─────────────────────────────────────────────────────── */
    .nav {{
      position: sticky;
      top: 0;
      z-index: 100;
      background: rgba(15,23,42,.9);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border);
      display: flex;
      gap: 4px;
      padding: 0 32px;
      overflow-x: auto;
    }}
    .nav a {{
      display: inline-block;
      color: var(--text-lo);
      text-decoration: none;
      font-size: .85rem;
      font-weight: 500;
      padding: 16px 14px;
      border-bottom: 2px solid transparent;
      white-space: nowrap;
      transition: color .2s, border-color .2s;
    }}
    .nav a:hover {{
      color: var(--text-hi);
      border-color: var(--accent1);
    }}

    /* ── Main layout ─────────────────────────────────────────────── */
    .container {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 48px 24px;
    }}

    .section-header {{
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 32px;
    }}
    .section-header .dot {{
      width: 10px; height: 10px;
      border-radius: 50%;
      flex-shrink: 0;
    }}
    .section-header h2 {{
      font-size: 1.5rem;
      font-weight: 700;
      color: var(--text-hi);
    }}
    .divider {{
      border: none;
      border-top: 1px solid var(--border);
      margin: 56px 0;
    }}

    /* ── Chart cards ─────────────────────────────────────────────── */
    .chart-section {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
      box-shadow: var(--shadow);
      margin-bottom: 28px;
      transition: transform .25s, box-shadow .25s;
    }}
    .chart-section:hover {{
      transform: translateY(-3px);
      box-shadow: 0 16px 48px rgba(0,0,0,.6);
    }}
    .chart-title {{
      font-size: 1rem;
      font-weight: 600;
      color: var(--text-hi);
      padding: 18px 24px;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(90deg, rgba(124,58,237,.08), transparent);
    }}
    .chart-frame {{
      width: 100%;
      height: 550px;
      border: none;
      display: block;
      background: transparent;
    }}

    /* ── Footer ──────────────────────────────────────────────────── */
    footer {{
      text-align: center;
      padding: 32px;
      border-top: 1px solid var(--border);
      color: var(--text-lo);
      font-size: .8rem;
    }}
    footer a {{ color: #A78BFA; text-decoration: none; }}
  </style>
</head>
<body>

<header class="hero">
  <div class="hero-badge">EMSE · Research Dashboard</div>
  <h1>Multilingual Code Comment<br>Quality Evaluation</h1>
  <p>
    Explore how five state-of-the-art code LLMs perform at generating accurate,
    well-formed code comments across <strong>five natural languages</strong>
    using the EMSE-Babel benchmark.
  </p>
  <div class="stats-bar">
    <div class="stat-pill"><span class="num">{total_samples:,}</span><span class="lbl">Code Snippets</span></div>
    <div class="stat-pill"><span class="num">5</span><span class="lbl">Languages</span></div>
    <div class="stat-pill"><span class="num">5</span><span class="lbl">LLMs Evaluated</span></div>
    <div class="stat-pill"><span class="num">{total_errors:,}</span><span class="lbl">Errors Logged</span></div>
    <div class="stat-pill"><span class="num">11</span><span class="lbl">Visualisations</span></div>
  </div>
</header>

<nav class="nav">
  <a href="#accuracy">Accuracy</a>
  <a href="#heatmap">Heatmap</a>
  <a href="#radar">Radar</a>
  <a href="#errors">Errors</a>
  <a href="#bartscore">BARTScore</a>
  <a href="#bokeh">Bokeh Charts</a>
</nav>

<main class="container">

  <!-- PLOTLY SECTION -->
  <div id="accuracy" class="section-header">
    <div class="dot" style="background:#7C3AED"></div>
    <h2>Expert-Evaluated Accuracy</h2>
  </div>
  {sections_plotly[:sections_plotly.find("plotly_2")]}
  {make_iframe_section("Correct Rate Heatmap (Language × Model)", "plotly_2_correctrate_heatmap.html")}

  <hr class="divider">

  <div id="radar" class="section-header">
    <div class="dot" style="background:#2563EB"></div>
    <h2>Multi-Language Radar</h2>
  </div>
  {make_iframe_section("Performance Radar Chart", "plotly_3_radar_chart.html")}

  <hr class="divider">

  <div id="errors" class="section-header">
    <div class="dot" style="background:#059669"></div>
    <h2>Error Analysis</h2>
  </div>
  {make_iframe_section("Error Taxonomy Treemap (All Languages)", "plotly_4_error_treemap.html")}
  {make_iframe_section("Error Category Sunburst by Language", "plotly_6_sunburst.html")}

  <hr class="divider">

  <div id="bartscore" class="section-header">
    <div class="dot" style="background:#D97706"></div>
    <h2>BARTScore Analysis</h2>
  </div>
  {make_iframe_section("BARTScore Distributions (Violin)", "plotly_5_bartscore_violin.html")}
  {make_iframe_section("Mean BARTScore per Language & Model", "plotly_7_bartscore_lines.html")}

  <hr class="divider">

  <!-- BOKEH SECTION -->
  <div id="bokeh" class="section-header">
    <div class="dot" style="background:#DC2626"></div>
    <h2>Bokeh Interactive Charts</h2>
  </div>
  {sections_bokeh}

</main>

<footer>
  Generated by <strong>EMSE-Babel visualize.py</strong>&nbsp;·&nbsp;
  Powered by <a href="https://plotly.com">Plotly</a> &amp;
  <a href="https://bokeh.org">Bokeh</a>
</footer>

</body>
</html>"""

dashboard_path = OUT_DIR / "dashboard.html"
dashboard_path.write_text(html, encoding="utf-8")
print(f"   ✓ saved dashboard.html")

print("\n✅ All done! Open `visualization_output/dashboard.html` in your browser.")
