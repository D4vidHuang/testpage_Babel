"""
LLM-as-a-Judge Accuracy
========================
Plots and saves:
  1. judge_accuracy.{pdf,png}            — scatter of quadratic Cohen's κ
  2. judge_cm_{lang}_{setting}.{pdf,png} — per-language×setting confusion matrices
                                           (COLUMN-normalised: what did judge predict
                                            for each human label?)
  3. judge_cm_accumulated.{pdf,png}      — single aggregate CM across all conditions
     All CMs use label order: Correct / Partial / Incorrect  (per paper)

Self-contained: reads  data/raw_{Language}.parquet
Output:         plots/out/
"""

import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
OUT_DIR    = Path("plots/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS   = ["standard", "cot", "hierarchical", "rubric"]
LANGUAGES  = ["en", "zh", "pl", "nl", "el"]
LANG_MAP   = {"en": "English", "zh": "Chinese", "pl": "Polish", "nl": "Dutch", "el": "Greek"}
LANG_LABELS = {"en": "EN", "zh": "ZH", "pl": "PL", "nl": "NL", "el": "EL"}

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
    "google_gemini-3-flash-preview_floor":      "Gemini Flash",
    "meituan_longcat-flash-chat_floor":          "Longcat Flash",
    "openai_gpt-oss-20b_floor":                  "GPT 20B",
    "openai_gpt-oss-120b_floor":                 "GPT 120B",
    "anthropic_claude-haiku-4_5_floor":          "Claude Haiku",
    "qwen_qwen3-coder-next_floor":               "Qwen3 Coder",
    "qwen_qwen3-vl-235b-a22b-instruct_floor":    "Qwen3 VL 235B",
    "deepseek_deepseek-v3_2_floor":              "DeepSeek V3",
    "x-ai_grok-4_1-fast_floor":                 "Grok 4.1 Fast",
}
MODEL_COLORS = {
    "google_gemini-3-flash-preview_floor":      "#4285F4",
    "meituan_longcat-flash-chat_floor":          "#F06292",
    "openai_gpt-oss-20b_floor":                  "#FF9800",
    "openai_gpt-oss-120b_floor":                 "#4CAF50",
    "anthropic_claude-haiku-4_5_floor":          "#D32F2F",
    "qwen_qwen3-coder-next_floor":               "#7B1FA2",
    "qwen_qwen3-vl-235b-a22b-instruct_floor":    "#00BCD4",
    "deepseek_deepseek-v3_2_floor":              "#795548",
    "x-ai_grok-4_1-fast_floor":                 "#9E9E9E",
}
MODEL_MARKERS = {
    "google_gemini-3-flash-preview_floor":      "o",
    "meituan_longcat-flash-chat_floor":          "X",
    "openai_gpt-oss-20b_floor":                  "s",
    "openai_gpt-oss-120b_floor":                 "D",
    "anthropic_claude-haiku-4_5_floor":          "^",
    "qwen_qwen3-coder-next_floor":               "v",
    "qwen_qwen3-vl-235b-a22b-instruct_floor":    "<",
    "deepseek_deepseek-v3_2_floor":              "P",
    "x-ai_grok-4_1-fast_floor":                 ">",
}
SETTING_DISPLAY = {
    "standard": "Standard", "cot": "CoT",
    "hierarchical": "Hier.", "rubric": "Rubric"
}

# Paper label order: Correct=2, Partial=1, Incorrect=0
# But we show top-to-bottom as Correct / Partial / Incorrect (descending)
LABEL_ORDER   = [2, 1, 0]          # numeric
CLASS_NAMES   = ["Correct", "Partial", "Incorrect"]   # display
SHOW_CONFUSION_MATRICES = True     # set False to skip per-lang/setting CMs

# ── STYLE ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.4,
    "grid.alpha":        0.5,
    "figure.dpi":        150,
})

# ── HELPERS ───────────────────────────────────────────────────────────────────
def _map_human(v):
    if pd.isna(v): return np.nan
    s = str(v).strip().lower()
    return 2 if s == "correct" else 1 if s == "partial" else 0 if s == "incorrect" else np.nan

def _map_judge(v):
    if pd.isna(v): return np.nan
    s = str(v).strip().lower()
    return 2 if s == "correct" else 1 if s == "partially_correct" else 0 if s == "incorrect" else np.nan


# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_records():
    """Returns list of records and cm_store dict for CMs."""
    records  = []
    cm_store = defaultdict(lambda: {"y_true": [], "y_pred": []})

    for lang in LANGUAGES:
        lang_name = LANG_MAP[lang]
        path = DATA_DIR / f"raw_{lang_name}.parquet"
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        df = pd.read_parquet(path)

        for judge_model in JUDGE_MODELS:
            for setting in SETTINGS:
                y_true_all, y_pred_all = [], []

                for tlm in TARGET_LLMS:
                    human_col = f"expert_accuracy_{tlm}"
                    if human_col not in df.columns:
                        continue
                    pattern = rf"^{re.escape(judge_model)}_{re.escape(setting)}_{re.escape(tlm)}_accuracy$"
                    judge_cols = [c for c in df.columns if re.match(pattern, c)]
                    if not judge_cols:
                        continue
                    human = df[human_col].map(_map_human).to_numpy(float)
                    pred  = df[judge_cols[0]].map(_map_judge).to_numpy(float)
                    mask  = ~np.isnan(human) & ~np.isnan(pred)
                    if mask.sum() == 0:
                        continue
                    y_true_all.append(human[mask].astype(int))
                    y_pred_all.append(pred[mask].astype(int))

                if not y_true_all:
                    continue
                y_true = np.concatenate(y_true_all)
                y_pred = np.concatenate(y_pred_all)
                kappa  = cohen_kappa_score(y_true, y_pred, weights="quadratic")
                # Column-normalize: for each judge-predicted class, what fraction is each human label?
                cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER, normalize="pred") * 100

                records.append({
                    "language": lang,
                    "judge":    judge_model,
                    "setting":  setting,
                    "kappa":    kappa,
                    "n":        len(y_true),
                    "cm":       cm,
                })
                cm_store[(lang, setting)]["y_true"].append(y_true)
                cm_store[(lang, setting)]["y_pred"].append(y_pred)

    return records, cm_store


# ── KAPPA SCATTER PLOT ────────────────────────────────────────────────────────
def plot_kappa(records):
    n_langs    = len(LANGUAGES)
    n_settings = len(SETTINGS)
    group_w    = n_settings
    sep        = 0.5

    def xpos(li, si):
        return li * (group_w + sep) + si

    fig, ax = plt.subplots(figsize=(11, 4.5))

    for rec in records:
        li = LANGUAGES.index(rec["language"])
        si = SETTINGS.index(rec["setting"])
        x  = xpos(li, si)
        ax.scatter(x, rec["kappa"],
                   color=MODEL_COLORS.get(rec["judge"], "#888"),
                   marker=MODEL_MARKERS.get(rec["judge"], "o"),
                   s=55, alpha=0.92,
                   edgecolors="white", linewidths=0.5, zorder=3)

    for li in range(n_langs):
        ax.axvspan(xpos(li, 0) - 0.5, xpos(li, n_settings-1) + 0.5,
                   facecolor="#F5F5F5" if li % 2 == 0 else "white",
                   alpha=1.0, zorder=0)
    for li in range(1, n_langs):
        ax.axvline(xpos(li, 0) - 0.5 - sep/2, color="#CCCCCC", linewidth=0.8, zorder=1)

    for kref, lbl in [(0.2, "Fair"), (0.4, "Moderate"), (0.6, "Substantial")]:
        ax.axhline(kref, color="#AAAAAA", linewidth=0.7, linestyle=":", zorder=1)
        ax.text(xpos(n_langs-1, n_settings-1) + 0.6, kref, lbl,
                va="center", ha="left", fontsize=7, color="#888")

    xticks  = [xpos(li, si) for li in range(n_langs) for si in range(n_settings)]
    xlabels = [SETTING_DISPLAY[s] for _ in LANGUAGES for s in SETTINGS]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=8)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([xpos(li, (n_settings-1)/2) for li in range(n_langs)])
    ax2.set_xticklabels([LANG_LABELS[l] for l in LANGUAGES], fontsize=10, fontweight="bold")
    ax2.tick_params(axis="x", length=0, pad=6)
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax.set_ylabel("Quadratic Cohen's κ  (judge vs. human)", fontsize=9)
    ax.set_xlabel("Prompt setting", fontsize=9)
    ax.set_title("LLM-as-a-Judge Accuracy vs. Human Expert\n"
                 "(weighted Cohen's κ, aggregated over target LLMs)",
                 fontsize=10, fontweight="bold", pad=14)
    ax.set_ylim(-0.05, 1.0)

    handles = [
        plt.Line2D([0],[0], marker=MODEL_MARKERS[m], color="w",
                   markerfacecolor=MODEL_COLORS[m], markeredgecolor="white",
                   markersize=7, label=JUDGE_DISPLAY[m])
        for m in JUDGE_MODELS
    ]
    ax.legend(handles=handles, title="Judge model", fontsize=7.5,
              title_fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1),
              frameon=True, edgecolor="#DDDDDD")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"judge_accuracy.{ext}", bbox_inches="tight", dpi=200)
    print(f"  Saved → {OUT_DIR}/judge_accuracy.{{pdf,png}}")
    plt.show()


# ── CONFUSION MATRIX HELPERS ──────────────────────────────────────────────────
def _draw_cm(ax, cm, title, kappa=None):
    """Draw a single column-normalised CM on ax. cm is [n_labels × n_labels]."""
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=7.5)
    ax.set_yticklabels(CLASS_NAMES, fontsize=7.5)
    ax.set_xlabel("Judge prediction", fontsize=7.5)
    ax.set_ylabel("Human label", fontsize=7.5)
    kappa_str = f"  κ={kappa:.3f}" if kappa is not None else ""
    ax.set_title(f"{title}{kappa_str}", fontsize=7.5)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="white" if v > 50 else "black", fontsize=7.5)
    return im


def plot_confusion_matrices_per_lang_setting(records, cm_store):
    """One figure per (language, setting): one subplot per judge model."""
    by_ls = defaultdict(list)
    for rec in records:
        by_ls[(rec["language"], rec["setting"])].append(rec)

    for (lang, setting), recs in sorted(by_ls.items()):
        recs = sorted(recs, key=lambda r: r["judge"])
        n     = len(recs)
        ncols = 3
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5*ncols, 4.2*nrows))
        axes = np.array(axes).reshape(-1)
        im = None
        for ax, rec in zip(axes, recs):
            im = _draw_cm(ax, rec["cm"],
                          title=JUDGE_DISPLAY.get(rec["judge"], rec["judge"]),
                          kappa=rec["kappa"])
        for ax in axes[n:]:
            ax.axis("off")
        if im is not None:
            cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.6)
            cbar.set_label("Col-normalised (%)", fontsize=8)
        fig.suptitle(f"Confusion Matrices (col-norm) — "
                     f"{LANG_MAP[lang]}, {SETTING_DISPLAY[setting]}",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        fname = f"judge_cm_{lang}_{setting}"
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"{fname}.{ext}", bbox_inches="tight", dpi=150)
        print(f"  Saved → {OUT_DIR}/{fname}.{{pdf,png}}")
        plt.show()


def plot_accumulated_cm(cm_store):
    """Single aggregate CM over all languages × settings × judges × LLMs."""
    all_true, all_pred = [], []
    for vals in cm_store.values():
        all_true.extend(vals["y_true"])
        all_pred.extend(vals["y_pred"])

    if not all_true:
        print("  No data for accumulated CM.")
        return

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    kappa  = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    cm     = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER, normalize="pred") * 100

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = _draw_cm(ax, cm,
                  title="All languages, settings & judges",
                  kappa=kappa)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Col-normalised (%)", fontsize=9)
    ax.set_title(f"Accumulated Confusion Matrix (col-norm)\n"
                 f"n={len(y_true):,}  κ={kappa:.3f}", fontsize=10, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"judge_cm_accumulated.{ext}", bbox_inches="tight", dpi=200)
    print(f"  Saved → {OUT_DIR}/judge_cm_accumulated.{{pdf,png}}")
    plt.show()


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
def main():
    print("Loading data …")
    records, cm_store = load_records()
    if not records:
        print("No data found — check DATA_DIR.")
        return
    print(f"  {len(records)} records.")
    plot_kappa(records)
    plot_accumulated_cm(cm_store)
    if SHOW_CONFUSION_MATRICES:
        plot_confusion_matrices_per_lang_setting(records, cm_store)


if __name__ == "__main__":
    main()
