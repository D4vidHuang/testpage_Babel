"""
LLM-as-a-Judge Error Frequency
================================
Generates:
  1. judge_frequency_all_codes.{pdf,png}
       Horizontal bar chart showing, for every error code in the taxonomy,
       the mean per-row frequency from the human expert vs. the average
       across all judge models × settings × languages.
       (This replaces the old LG-GR4-only chart.)

  2. judge_frequency_heatmap.{pdf,png}
       Code × (human | judge avg | Δ) heatmap for a configurable
       language + target LLM pair.

  3. judge_frequency_accumulated.{pdf,png}
       A single accumulated bar chart summarising TOTAL error-code counts
       (human vs. judge) aggregated over all languages, judge models,
       settings, and target LLMs.

Self-contained: reads  data/raw_{Language}.parquet
Output:         plots/out/
"""

import ast
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR      = Path("data")
TAXONOMY_PATH = Path("taxonomy/error_taxonomy.json")
OUT_DIR       = Path("plots/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS  = ["standard", "cot", "hierarchical", "rubric"]
LANGUAGES = ["en", "zh", "pl", "nl", "el"]
LANG_MAP  = {"en": "English", "zh": "Chinese", "pl": "Polish", "nl": "Dutch", "el": "Greek"}

TARGET_LLMS = [
    "google/codegemma-7b",
    "meta-llama/CodeLlama-7b-hf",
    "Qwen/CodeQwen1.5-7B",
    "bigcode/starcoder2-7b",
    "ibm-granite/granite-8b-code-base",
]
LLM_SHORT = {
    "google/codegemma-7b":                "CodeGemma",
    "meta-llama/CodeLlama-7b-hf":         "CodeLlama",
    "Qwen/CodeQwen1.5-7B":                "CodeQwen",
    "bigcode/starcoder2-7b":              "StarCoder2",
    "ibm-granite/granite-8b-code-base":   "Granite",
}
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

# Configurable detail-view settings (Fig 2)
HEATMAP_LANGUAGE  = "English"
HEATMAP_TARGET_LLM = "google/codegemma-7b"

# ── STYLE ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

# ── HELPERS ───────────────────────────────────────────────────────────────────
def load_taxonomy_codes():
    """Return list of leaf error-code IDs (codes that look like XX-YY or XX-YYn)."""
    with open(TAXONOMY_PATH) as f:
        tax = json.load(f)
    # Keep only leaf codes (skip parent group codes like 'LG', 'MS', 'SE', 'ST')
    all_ids = []
    for cat, items in tax.items():
        for item in items:
            cid = item["id"]
            # Leaf codes contain at least one dash and end in digits or single letters not alone
            if "-" in cid:
                all_ids.append(cid)
    # Preserve order from taxonomy, deduplicated
    seen = set()
    result = []
    for cid in all_ids:
        if cid not in seen:
            seen.add(cid)
            result.append(cid)
    return result


def parse_errors(value):
    """Robustly turn a column value into a list of error-code strings."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    s = str(value).strip()
    if not s or s.lower() in ("none", "nan", "[]"):
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    return [x.strip() for x in s.split(",") if x.strip()]


# ── DATA COLLECTION ───────────────────────────────────────────────────────────
def collect_all_frequencies(codes):
    """
    Returns two dicts: human_counts[code], judge_counts[code]
    aggregated over all languages × LLMs × (judges × settings for judge side).
    Also returns total_human_rows, total_judge_rows for normalisation.
    """
    human_counts  = {c: 0 for c in codes}
    judge_counts  = {c: 0 for c in codes}
    total_human_rows = 0
    total_judge_rows = 0

    for lang in LANGUAGES:
        lang_name = LANG_MAP[lang]
        path = DATA_DIR / f"raw_{lang_name}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)

        for llm in TARGET_LLMS:
            # Human
            hcol = f"error_codes_{llm}"
            if hcol in df.columns:
                for v in df[hcol]:
                    errs = parse_errors(v)
                    for c in errs:
                        if c in human_counts:
                            human_counts[c] += 1
                total_human_rows += len(df)

            # Judge
            for judge in JUDGE_MODELS:
                for setting in SETTINGS:
                    pat = rf"^{re.escape(judge)}_{re.escape(setting)}_{re.escape(llm)}_errors$"
                    cols = [c for c in df.columns if re.match(pat, c)]
                    for col in cols:
                        for v in df[col]:
                            errs = parse_errors(v)
                            for c in errs:
                                if c in judge_counts:
                                    judge_counts[c] += 1
                        total_judge_rows += len(df)

    return human_counts, judge_counts, total_human_rows, total_judge_rows


# ── PLOT 1: All-code frequency comparison (horizontal bar) ────────────────────
def plot_all_codes(codes, human_counts, judge_counts,
                   total_human_rows, total_judge_rows):
    h_freq = np.array([human_counts[c] / max(total_human_rows, 1) * 100 for c in codes])
    j_freq = np.array([judge_counts[c]  / max(total_judge_rows, 1)  * 100 for c in codes])

    # Sort by human frequency descending
    order = np.argsort(h_freq)[::-1]
    codes_s = [codes[i] for i in order]
    h_s = h_freq[order]
    j_s = j_freq[order]

    n = len(codes_s)
    y = np.arange(n)
    h_bar = 0.38

    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.35)))
    ax.barh(y + h_bar/2, h_s, height=h_bar, label="Human expert",
            color="#455A64", alpha=0.87, zorder=3)
    ax.barh(y - h_bar/2, j_s, height=h_bar, label="Judge avg.",
            color="#29B6F6", alpha=0.87, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(codes_s, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency per row  (%)", fontsize=9)
    ax.set_title("Error Code Frequency: Human Expert vs. Judge Average\n"
                 "(aggregated over all languages, target LLMs, judge models, and settings)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"judge_frequency_all_codes.{ext}", bbox_inches="tight", dpi=200)
    print(f"  Saved → {OUT_DIR}/judge_frequency_all_codes.{{pdf,png}}")
    plt.show()


# ── PLOT 2: Code × (human | judge | Δ) heatmap for one lang + LLM ─────────────
def compute_heatmap(codes, language, target_llm):
    path = DATA_DIR / f"raw_{language}.parquet"
    df = pd.read_parquet(path)
    n = len(df)

    hcol = f"error_codes_{target_llm}"
    human_freq = {}
    if hcol in df.columns:
        for c in codes:
            cnt = sum(1 for v in df[hcol] if c in parse_errors(v))
            human_freq[c] = cnt / max(n, 1) * 100
    else:
        human_freq = {c: 0.0 for c in codes}

    judge_freqs = {c: [] for c in codes}
    for judge in JUDGE_MODELS:
        for setting in SETTINGS:
            pat = rf"^{re.escape(judge)}_{re.escape(setting)}_{re.escape(target_llm)}_errors$"
            cols = [col for col in df.columns if re.match(pat, col)]
            for col in cols:
                for c in codes:
                    cnt = sum(1 for v in df[col] if c in parse_errors(v))
                    judge_freqs[c].append(cnt / max(n, 1) * 100)
    judge_freq = {c: (np.mean(judge_freqs[c]) if judge_freqs[c] else 0.0) for c in codes}
    return human_freq, judge_freq


def plot_heatmap(codes, human_freq, judge_freq, language, target_llm):
    h_vals = np.array([human_freq[c] for c in codes])
    j_vals = np.array([judge_freq[c] for c in codes])
    delta  = np.clip(
        np.where(h_vals + j_vals > 0,
                 (j_vals - h_vals) / (np.maximum(h_vals, 0.01)) * 100,
                 0.0),
        -300, 300
    )

    row_labels = ["Human (%)", "Judge avg (%)", "Δ judge/human (%)"]
    data = np.vstack([h_vals, j_vals, delta])

    fig, ax = plt.subplots(figsize=(max(12, len(codes) * 0.42), 3.2))
    for ci, c in enumerate(codes):
        for ri, (row, val) in enumerate(zip(data, [h_vals[ci], j_vals[ci], delta[ci]])):
            cmap  = plt.cm.RdBu_r if ri == 2 else plt.cm.Blues
            vmax  = 300 if ri == 2 else max(data[ri].max(), 1)
            vmin  = -300 if ri == 2 else 0
            norm  = mcolors.Normalize(vmin=vmin, vmax=vmax)
            color = cmap(norm(val))
            rect  = plt.Rectangle([ci - .5, ri - .5], 1, 1, color=color, zorder=1)
            ax.add_patch(rect)
            tx_color = "white" if abs(val) > (abs(vmax) * 0.55) else "black"
            ax.text(ci, ri, f"{val:.0f}", ha="center", va="center",
                    fontsize=6.5, color=tx_color, zorder=2)

    ax.set_xlim(-0.5, len(codes) - 0.5)
    ax.set_ylim(-0.5, len(row_labels) - 0.5)
    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels(codes, rotation=55, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8.5)
    ax.set_xlabel("Error code", fontsize=9)
    ax.set_title(
        f"Error Code Frequency Heatmap — {language}, {LLM_SHORT.get(target_llm, target_llm)}\n"
        "Rows: human expert rate | judge average rate | Δ relative change",
        fontsize=10, fontweight="bold",
    )
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"judge_frequency_heatmap.{ext}", bbox_inches="tight", dpi=200)
    print(f"  Saved → {OUT_DIR}/judge_frequency_heatmap.{{pdf,png}}")
    plt.show()


# ── PLOT 3: Accumulated single-bar chart ─────────────────────────────────────
def plot_accumulated(codes, human_counts, judge_counts,
                     total_human_rows, total_judge_rows):
    """Bar chart showing raw accumulated counts normalised per 1000 rows."""
    h_per_k = np.array([human_counts[c] / max(total_human_rows, 1) * 1000 for c in codes])
    j_per_k = np.array([judge_counts[c]  / max(total_judge_rows, 1)  * 1000 for c in codes])

    # Only show codes with any occurrence
    mask = (h_per_k + j_per_k) > 0
    codes_f = [c for c, m in zip(codes, mask) if m]
    h_f = h_per_k[mask]
    j_f = j_per_k[mask]

    order = np.argsort(h_f)[::-1]
    codes_s = [codes_f[i] for i in order]
    h_s = h_f[order]
    j_s = j_f[order]

    x = np.arange(len(codes_s))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(11, len(codes_s) * 0.55), 4.5))
    ax.bar(x - w/2, h_s, width=w, label="Human expert",  color="#455A64", alpha=0.87, zorder=3)
    ax.bar(x + w/2, j_s, width=w, label="Judge avg.",    color="#29B6F6", alpha=0.87, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(codes_s, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Occurrences per 1 000 rows", fontsize=9)
    ax.set_title(
        "Accumulated Error Code Frequency: Human Expert vs. Judge Average\n"
        "(all languages × target LLMs × judge models × settings)",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"judge_frequency_accumulated.{ext}", bbox_inches="tight", dpi=200)
    print(f"  Saved → {OUT_DIR}/judge_frequency_accumulated.{{pdf,png}}")
    plt.show()


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
def main():
    codes = load_taxonomy_codes()
    print(f"Taxonomy leaf codes ({len(codes)}): {codes}")

    print("\nCollecting error frequencies across all conditions …")
    human_counts, judge_counts, total_human, total_judge = collect_all_frequencies(codes)
    print(f"  Human rows: {total_human:,}  |  Judge rows: {total_judge:,}")

    print("\nPlot 1 — All-code frequency comparison …")
    plot_all_codes(codes, human_counts, judge_counts, total_human, total_judge)

    print("\nPlot 2 — Heatmap detail view …")
    h_freq, j_freq = compute_heatmap(codes, HEATMAP_LANGUAGE, HEATMAP_TARGET_LLM)
    plot_heatmap(codes, h_freq, j_freq, HEATMAP_LANGUAGE, HEATMAP_TARGET_LLM)

    print("\nPlot 3 — Accumulated bar chart …")
    plot_accumulated(codes, human_counts, judge_counts, total_human, total_judge)


if __name__ == "__main__":
    main()
