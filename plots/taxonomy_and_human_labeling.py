"""
Taxonomy & Human Labeling Results
====================================
Generates visualizations of:
  1. taxonomy_overview.{pdf,png}
       Grouped bar of error codes per category, with inclusion/exclusion
       criteria shown as annotated text.
  2. human_labeling_freq.{pdf,png}
       Human expert error-code frequency across all languages and target LLMs
       — a bar chart showing how often each code appeared.
  3. human_labeling_by_language.{pdf,png}
       Per-language human error-code distributions (stacked/grouped).
  4. human_labeling_by_llm.{pdf,png}
       Per-target-LLM human error-code frequencies.

Self-contained: reads  data/raw_{Language}.parquet
                        taxonomy/error_taxonomy.json
Output:         plots/out/
"""

import ast
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR      = Path("data")
TAXONOMY_PATH = Path("taxonomy/error_taxonomy.json")
OUT_DIR       = Path("plots/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGES = ["en", "zh", "pl", "nl", "el"]
LANG_MAP  = {"en": "English", "zh": "Chinese", "pl": "Polish",
             "nl": "Dutch",   "el": "Greek"}
LANG_LABELS = {"en": "EN", "zh": "ZH", "pl": "PL", "nl": "NL", "el": "EL"}
LANG_COLORS = {"en": "#4285F4", "zh": "#EA4335", "pl": "#FBBC05",
               "nl": "#34A853", "el": "#9C27B0"}

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
LLM_COLORS = {
    "google/codegemma-7b":               "#4285F4",
    "meta-llama/CodeLlama-7b-hf":        "#EA4335",
    "Qwen/CodeQwen1.5-7B":               "#FBBC05",
    "bigcode/starcoder2-7b":             "#34A853",
    "ibm-granite/granite-8b-code-base":  "#9C27B0",
}

# Category display colours
CAT_COLORS = {
    "Code Snippet":             "#26C6DA",
    "Grammar":                  "#66BB6A",
    "Hallucination":            "#EF5350",
    "Incorrect comment format": "#AB47BC",
    "Linguistic":               "#42A5F5",
    "Memorization":             "#FFA726",
    "Model-specific":           "#8D6E63",
    "Repetition":               "#EC407A",
    "Semantic":                 "#FF7043",
    "Syntax":                   "#78909C",
    "Wrong language":           "#26A69A",
    "":                         "#B0BEC5",
}

# ── STYLE ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

# ── HELPERS ───────────────────────────────────────────────────────────────────
def parse_errors(value):
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


def load_taxonomy():
    with open(TAXONOMY_PATH) as f:
        return json.load(f)


def leaf_codes(tax):
    """Return list of (category, item_dict) for all leaf-level codes."""
    result = []
    for cat, items in tax.items():
        for item in items:
            if "-" in item["id"]:
                result.append((cat, item))
    return result


# ── PLOT 1: Taxonomy overview ─────────────────────────────────────────────────
def plot_taxonomy_overview(tax):
    leaves = leaf_codes(tax)
    cats = list(dict.fromkeys(c for c, _ in leaves))
    cat_items = {c: [it for cat, it in leaves if cat == c] for c in cats}

    # Also include the special '' category (E, M)
    all_cats = [""] + [c for c in cats if c != ""]
    cat_items[""] = [it for cat, it in leaves if cat == ""]

    # Number of codes per category
    n_cats = len(all_cats)
    max_codes = max(len(cat_items.get(c, [])) for c in all_cats)

    fig, ax = plt.subplots(figsize=(max(10, n_cats * 1.5), 6))

    x = np.arange(n_cats)
    bars = ax.bar(x, [len(cat_items.get(c, [])) for c in all_cats],
                  color=[CAT_COLORS.get(c, "#B0BEC5") for c in all_cats],
                  alpha=0.85, edgecolor="white", linewidth=0.8, zorder=3)

    # Annotate bars with code IDs
    for xi, cat in enumerate(all_cats):
        items = cat_items.get(cat, [])
        codes_str = "\n".join(it["id"] for it in items)
        ax.text(xi, len(items) + 0.05, codes_str,
                ha="center", va="bottom", fontsize=6.5,
                color="#333333", linespacing=1.4)

    ax.set_xticks(x)
    cat_labels = [c if c else "Special" for c in all_cats]
    ax.set_xticklabels(cat_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Number of error codes", fontsize=9)
    ax.set_title("Error Taxonomy Overview\nError codes per category",
                 fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max_codes + 3)

    # Legend patches
    handles = [mpatches.Patch(color=CAT_COLORS.get(c, "#B0BEC5"),
                               label=c if c else "Special (E, M)")
               for c in all_cats]
    ax.legend(handles=handles, fontsize=7.5, loc="upper right",
              frameon=True, edgecolor="#DDDDDD", ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"taxonomy_overview.{ext}", bbox_inches="tight", dpi=200)
    print(f"  Saved → {OUT_DIR}/taxonomy_overview.{{pdf,png}}")
    plt.show()


# ── HUMAN LABELING DATA LOADING ────────────────────────────────────────────
def load_human_counts():
    """Returns (global_counter, lang_counter, llm_counter)."""
    global_counter = Counter()
    lang_counter   = {l: Counter() for l in LANGUAGES}
    llm_counter    = {llm: Counter() for llm in TARGET_LLMS}
    total_rows     = 0

    for lang in LANGUAGES:
        lang_name = LANG_MAP[lang]
        path = DATA_DIR / f"raw_{lang_name}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        for llm in TARGET_LLMS:
            col = f"error_codes_{llm}"
            if col not in df.columns:
                continue
            for v in df[col]:
                codes = parse_errors(v)
                for c in codes:
                    global_counter[c] += 1
                    lang_counter[lang][c] += 1
                    llm_counter[llm][c] += 1
            total_rows += len(df)

    return global_counter, lang_counter, llm_counter, total_rows


# ── PLOT 2: Human labeling global frequency ───────────────────────────────────
def plot_human_global(counter, total_rows, top_n=25):
    top = counter.most_common(top_n)
    codes = [c for c, _ in top]
    counts = [n for _, n in top]
    freq   = [n / max(total_rows, 1) * 100 for n in counts]

    y = np.arange(len(codes))
    fig, ax = plt.subplots(figsize=(9, max(5, len(codes) * 0.38)))
    bars = ax.barh(y, freq, color="#455A64", alpha=0.85, zorder=3)
    # Count labels
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f"n={cnt:,}", va="center", fontsize=7.5, color="#666666")
    ax.set_yticks(y)
    ax.set_yticklabels(codes, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency per row  (%)", fontsize=9)
    ax.set_title(f"Human Expert Error Code Frequency (top {top_n})\n"
                 "Aggregated over all 5 languages and 5 target LLMs",
                 fontsize=10, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"human_labeling_freq.{ext}", bbox_inches="tight", dpi=200)
    print(f"  Saved → {OUT_DIR}/human_labeling_freq.{{pdf,png}}")
    plt.show()


# ── PLOT 3: Human labeling by language ───────────────────────────────────────
def plot_human_by_language(lang_counter, top_n=15):
    # Use union of top-N codes across all languages
    all_codes = set()
    for lc in lang_counter.values():
        for c, _ in lc.most_common(top_n):
            all_codes.add(c)
    # Sort by global frequency
    global_counts = Counter()
    for lc in lang_counter.values():
        global_counts.update(lc)
    codes = [c for c, _ in global_counts.most_common() if c in all_codes][:top_n]

    x = np.arange(len(codes))
    n_langs = len(LANGUAGES)
    w = 0.15

    fig, ax = plt.subplots(figsize=(max(12, len(codes)*0.8), 5.5))
    for i, lang in enumerate(LANGUAGES):
        lc = lang_counter[lang]
        total = sum(lc.values()) or 1
        vals = [lc.get(c, 0) / total * 100 for c in codes]
        ax.bar(x + (i - n_langs/2 + 0.5) * w, vals, width=w,
               label=LANG_LABELS[lang], color=LANG_COLORS[lang],
               alpha=0.85, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("% of all errors in that language", fontsize=9)
    ax.set_title(f"Human Expert Error Codes by Language (top {top_n})\n"
                 "Each bar normalised to total errors in that language",
                 fontsize=10, fontweight="bold")
    ax.legend(title="Language", fontsize=8.5, title_fontsize=9)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"human_labeling_by_language.{ext}", bbox_inches="tight", dpi=200)
    print(f"  Saved → {OUT_DIR}/human_labeling_by_language.{{pdf,png}}")
    plt.show()


# ── PLOT 4: Human labeling by target LLM ─────────────────────────────────────
def plot_human_by_llm(llm_counter, top_n=15):
    global_counts = Counter()
    for lc in llm_counter.values():
        global_counts.update(lc)
    codes = [c for c, _ in global_counts.most_common(top_n)]

    x = np.arange(len(codes))
    n_llms = len(TARGET_LLMS)
    w = 0.15

    fig, ax = plt.subplots(figsize=(max(12, len(codes)*0.8), 5.5))
    for i, llm in enumerate(TARGET_LLMS):
        lc = llm_counter[llm]
        total = sum(lc.values()) or 1
        vals = [lc.get(c, 0) / total * 100 for c in codes]
        ax.bar(x + (i - n_llms/2 + 0.5) * w, vals, width=w,
               label=LLM_SHORT[llm], color=LLM_COLORS[llm],
               alpha=0.85, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("% of all errors for that target LLM", fontsize=9)
    ax.set_title(f"Human Expert Error Codes by Target LLM (top {top_n})\n"
                 "Each bar normalised to total errors in that LLM",
                 fontsize=10, fontweight="bold")
    ax.legend(title="Target LLM", fontsize=8.5, title_fontsize=9)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"human_labeling_by_llm.{ext}", bbox_inches="tight", dpi=200)
    print(f"  Saved → {OUT_DIR}/human_labeling_by_llm.{{pdf,png}}")
    plt.show()


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
def main():
    tax = load_taxonomy()
    print(f"Taxonomy: {sum(len(v) for v in tax.values())} entries across "
          f"{len(tax)} categories.")

    print("\nPlot 1 — Taxonomy overview …")
    plot_taxonomy_overview(tax)

    print("\nLoading human labeling data …")
    global_counter, lang_counter, llm_counter, total_rows = load_human_counts()
    print(f"  {total_rows:,} rows processed, {sum(global_counter.values()):,} error codes.")

    print("\nPlot 2 — Human global frequency …")
    plot_human_global(global_counter, total_rows)

    print("\nPlot 3 — Human by language …")
    plot_human_by_language(lang_counter)

    print("\nPlot 4 — Human by target LLM …")
    plot_human_by_llm(llm_counter)


if __name__ == "__main__":
    main()
