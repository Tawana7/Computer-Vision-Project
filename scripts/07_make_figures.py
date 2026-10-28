"""
Step 7: Build the report figures, tables and key numbers from the result CSVs.

Reads (project root by default)
  evaluation_results.csv            from 06_evaluate.py (mean +/- std per model)
  evaluation_results_by_seed.csv    from 06_evaluate.py (one row per model and seed)
  timing_log.csv                    from 03b_annotation_timer.py

Writes under ./report
  images/fig_time.pdf|png           annotation time per image, manual vs assisted
  images/fig_curve.pdf|png          test mAP50 vs number of training images
  images/fig_perclass.pdf|png       per-class AP50 for the main arms
  images/fig_effort.pdf|png         test mAP50 vs estimated human annotation minutes
  tables/results_table.tex          accuracy table (tabular only)
  tables/timing_table.tex           annotation-time table (tabular only)
  tables/key_numbers.tex            \\newcommand macros used by the report text (incl. box counts per
                                    class, raw vs corrected, if pseudo_annotations.json and
                                    corrected/train_pool_assisted.json are present)
  key_numbers.txt                   the same numbers, readable, for cross-checking

Usage (from the project root, after downloading the CSVs):
  python -m pip install matplotlib pandas numpy
  python scripts/07_make_figures.py

Every number in the report comes from these files, so rerun this script
whenever the CSVs change, then recompile the report.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(description="Build report figures and tables.")
ap.add_argument("--results", default="evaluation_results.csv")
ap.add_argument("--by-seed", default="evaluation_results_by_seed.csv")
ap.add_argument("--timing", default="timing_log.csv")
ap.add_argument("--out", default="report")
ap.add_argument("--pseudo", default="pseudo_annotations.json",
                help="raw YOLO-World labels from step 2 (used for box counts)")
ap.add_argument("--corrected", default="corrected/train_pool_assisted.json",
                help="corrected labels of the assisted batch (CVAT export, used for box counts)")
ap.add_argument("--raw-min-score", type=float, default=0.25,
                help="confidence cut-off of the raw arm; must equal RAW_MIN_SCORE in 04_coco_to_yolo.py")
ap.add_argument("--early-n", type=int, default=51,
                help="size of the 'early' assisted block used for the practice-effect check")
args = ap.parse_args()

IMG = os.path.join(args.out, "images")
TAB = os.path.join(args.out, "tables")
os.makedirs(IMG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
})
COL = {"assisted": "#0072B2", "manual": "#D55E00", "mixed": "#009E73",
       "raw": "#CC79A7", "zero_shot": "#555555"}   # colour-blind-safe palette
ARM_NAME = {"zero_shot": "Zero-shot", "assisted": "Assisted", "raw": "Raw",
            "manual": "Manual", "mixed": "Mixed"}
ARM_ORDER = ["zero_shot", "assisted", "raw", "manual", "mixed"]
W = 3.45  # IEEE single-column width in inches


def save(fig, name):
    fig.savefig(os.path.join(IMG, name + ".pdf"))
    fig.savefig(os.path.join(IMG, name + ".png"), dpi=200)
    plt.close(fig)


def short_class(col):
    n = col[len("AP50_"):].lower()
    if "boba" in n or "bubble" in n:
        return "Boba cup"
    if "coffee" in n:
        return "Coffee cup"
    if "straw" in n:
        return "Straw"
    return n


def model_label(row):
    if row["arm"] == "zero_shot":
        return "Zero-shot"
    return f"{ARM_NAME[row['arm']]}-{int(row['n_train_images'])}"


PM_TEX = " $\\pm$ "


def pm(mean, std, dec=3):
    """LaTeX 'mean +- std' (std omitted when there is only one seed)."""
    if pd.isna(mean):
        return "--"
    return f"{mean:.{dec}f}" + (f"{PM_TEX}{std:.{dec}f}" if pd.notna(std) else "")


def pm_plain(mean, std, dec=3):
    return pm(mean, std, dec).replace(PM_TEX, "+-")


# ----------------------------------------------------------------- load data
res = pd.read_csv(args.results)
seeds = pd.read_csv(args.by_seed)
res["arm"] = pd.Categorical(res["arm"], ARM_ORDER, ordered=True)
res = res.sort_values(["arm", "n_train_images"]).reset_index(drop=True)
res["arm"] = res["arm"].astype(str)
class_cols = [c for c in res.columns if c.startswith("AP50_") and not c.endswith("_std")]

t = pd.read_csv(args.timing)
t["seconds"] = t["seconds"].astype(float)
t = t[t["condition"] != "warmup"]            # practice batch is excluded from all results
man = t[t["condition"] == "manual"]
ass = t[t["condition"] == "assisted"]
ms, as_ = man["seconds"].to_numpy(), ass["seconds"].to_numpy()
man_mean, ass_mean = ms.mean(), as_.mean()

# ------------------------------------------------------------ timing numbers
rng = np.random.default_rng(0)
diffs = np.array([rng.choice(ms, len(ms)).mean() - rng.choice(as_, len(as_)).mean()
                  for _ in range(10000)])
ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
ass_sorted = ass.sort_values("image_index")
early = ass_sorted["seconds"].to_numpy()[:args.early_n]
late = ass_sorted["seconds"].to_numpy()[args.early_n:]


def stats(x):
    return dict(n=len(x), total=x.sum() / 60, mean=x.mean(), median=np.median(x),
                std=x.std(ddof=1) if len(x) > 1 else float("nan"))


rows_t = []
for batch, label in [("gold_test", "Manual, test set"), ("train_pool_manual", "Manual, training set")]:
    x = t[t["batch"] == batch]["seconds"].to_numpy()
    if len(x):
        rows_t.append((label, stats(x)))
rows_t.append(("Manual, pooled", stats(ms)))
rows_t.append(("Assisted (correction)", stats(as_)))

with open(os.path.join(TAB, "timing_table.tex"), "w") as f:
    f.write("\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n"
            "Batch & Images & Total (min) & Mean (s) & Median (s) & Std (s) \\\\\n\\midrule\n")
    for label, s in rows_t:
        if label == "Assisted (correction)":
            f.write("\\midrule\n")
        f.write(f"{label} & {s['n']} & {s['total']:.1f} & {s['mean']:.1f} & "
                f"{s['median']:.1f} & {s['std']:.1f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------ results table
with open(os.path.join(TAB, "results_table.tex"), "w") as f:
    f.write("\\begin{tabular}{@{}lrrcccccc@{}}\n\\toprule\n"
            "Model & Images & Seeds & mAP50 & mAP50--95 & "
            + " & ".join(f"AP50 {short_class(c)}" for c in class_cols) + " \\\\\n\\midrule\n")
    prev = None
    for _, r in res.iterrows():
        if prev is not None and r["arm"] != prev:
            f.write("\\midrule\n")
        prev = r["arm"]
        cells = [model_label(r), str(int(r["n_train_images"])), str(int(r["n_seeds"])),
                 pm(r["mAP50"], r.get("mAP50_std")), pm(r["mAP50_95"], r.get("mAP50_95_std"))]
        cells += [pm(r[c], r.get(c + "_std")) for c in class_cols]
        f.write(" & ".join(cells) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------- Fig 1: timing
fig, ax = plt.subplots(figsize=(W, 2.3))
jit = np.random.default_rng(1)
for i, (d, c) in enumerate([(ms, COL["manual"]), (as_, COL["assisted"])], 1):
    ax.boxplot(d, positions=[i], widths=0.5, showfliers=False, patch_artist=True,
               medianprops=dict(color="black", lw=1.2),
               boxprops=dict(facecolor=c, alpha=0.25, edgecolor=c),
               whiskerprops=dict(color=c), capprops=dict(color=c))
    ax.scatter(i + jit.uniform(-0.18, 0.18, len(d)), d, s=5, color=c, alpha=0.45,
               zorder=3, linewidths=0)
    ax.scatter([i], [d.mean()], marker="D", s=22, color="white", edgecolor="black", zorder=4)
ax.set_xticks([1, 2])
ax.set_xticklabels([f"Manual\n({len(ms)} images)", f"Assisted\n({len(as_)} images)"])
ax.set_xlim(0.4, 2.6)
ax.set_ylabel("Annotation time per image (s)")
save(fig, "fig_time")

# ---------------------------------------------------------- Fig 2: data curve
OFFSET = {"assisted": 0.0, "manual": -0.05, "mixed": 0.05, "raw": -0.05}
MARK = {"assisted": "o", "manual": "^", "mixed": "D", "raw": "s"}
LEGEND = {"assisted": "Assisted (corrected)", "manual": "Manual (from scratch)",
          "mixed": "Mixed", "raw": "Raw (uncorrected)"}
ft = res[res["arm"] != "zero_shot"]
zs = res[res["arm"] == "zero_shot"]

fig, ax = plt.subplots(figsize=(W, 2.7))
a = ft[ft["arm"] == "assisted"].sort_values("n_train_images")
ax.errorbar(a["n_train_images"], a["mAP50"], yerr=a["mAP50_std"].fillna(0), marker="o",
            color=COL["assisted"], capsize=2, lw=1, ms=4, label=LEGEND["assisted"], zorder=3)
for arm in ["manual", "mixed", "raw"]:
    d = ft[ft["arm"] == arm]
    if len(d):
        ax.errorbar(d["n_train_images"] * (1 + OFFSET[arm]), d["mAP50"],
                    yerr=d["mAP50_std"].fillna(0), fmt=MARK[arm], color=COL[arm],
                    capsize=2, ms=4, label=LEGEND[arm], zorder=3)
for arm, d in seeds[seeds["arm"] != "zero_shot"].groupby("arm"):
    ax.scatter(d["n_train_images"] * (1 + OFFSET.get(arm, 0)), d["mAP50"], s=6,
               color=COL[arm], alpha=0.35, zorder=2, linewidths=0)
if len(zs):
    ax.axhline(zs["mAP50"].iloc[0], ls="--", color=COL["zero_shot"], lw=0.9)
    ax.text(0.99, zs["mAP50"].iloc[0] + 0.006, "zero-shot baseline", ha="right", va="bottom",
            transform=ax.get_yaxis_transform(), fontsize=7, color=COL["zero_shot"])
ticks = sorted(ft["n_train_images"].unique())
ax.set_xscale("log")
ax.set_xticks(ticks)
ax.set_xticklabels([str(int(x)) for x in ticks])
ax.xaxis.set_minor_locator(mticker.NullLocator())
ax.set_xlabel("Training images")
ax.set_ylabel("Test mAP@0.5")
ax.legend(loc="upper left", frameon=False)
save(fig, "fig_curve")

# ---------------------------------------------------------- Fig 3: per class
sel = []
if len(zs):
    sel.append(zs.iloc[0])
for arm in ["raw", "assisted", "mixed"]:
    d = ft[ft["arm"] == arm]
    if len(d):
        sel.append(d.sort_values("n_train_images").iloc[-1])   # largest set of that arm
d = ft[ft["arm"] == "manual"]
if len(d):
    sel.append(d.sort_values("n_train_images").iloc[-1])

fig, ax = plt.subplots(figsize=(W, 2.5))
bw = 0.8 / len(sel)
for k, r in enumerate(sel):
    vals = [r[c] for c in class_cols]
    errs = [0 if pd.isna(r.get(c + "_std")) else r[c + "_std"] for c in class_cols]
    ax.bar(np.arange(len(class_cols)) + (k - (len(sel) - 1) / 2) * bw, vals, bw * 0.92,
           yerr=errs if any(errs) else None,   # single-seed models have no spread to draw
           capsize=1.5, color=COL[r["arm"]], label=model_label(r), error_kw=dict(lw=0.7))
ax.set_xticks(range(len(class_cols)))
ax.set_xticklabels([short_class(c) for c in class_cols])
ax.set_ylabel("Test AP@0.5")
ax.set_ylim(0, 1)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=False, fontsize=6.5)
save(fig, "fig_perclass")

# ---------------------------------------------------------- Fig 4: effort
man_min, ass_min = man_mean / 60, ass_mean / 60
m_manual = int(res[res["arm"] == "manual"]["n_train_images"].max()) if (res["arm"] == "manual").any() else 0
minutes = {}
for idx, r in ft.iterrows():
    n = int(r["n_train_images"])
    if r["arm"] == "assisted":
        minutes[idx] = n * ass_min
    elif r["arm"] == "manual":
        minutes[idx] = n * man_min
    elif r["arm"] == "raw":
        minutes[idx] = 0.0                      # no human time (inference only)
    elif r["arm"] == "mixed" and m_manual:
        n_man = m_manual // 2 if n == m_manual else m_manual
        minutes[idx] = n_man * man_min + (n - n_man) * ass_min
ft = ft.assign(minutes=pd.Series(minutes))

fig, ax = plt.subplots(figsize=(W, 2.7))
a = ft[ft["arm"] == "assisted"].sort_values("minutes")
ax.errorbar(a["minutes"], a["mAP50"], yerr=a["mAP50_std"].fillna(0), marker="o",
            color=COL["assisted"], capsize=2, lw=1, ms=4, label=LEGEND["assisted"])
for arm in ["manual", "mixed", "raw"]:
    d = ft[ft["arm"] == arm].dropna(subset=["minutes"])
    if len(d):
        ax.errorbar(d["minutes"], d["mAP50"], yerr=d["mAP50_std"].fillna(0), fmt=MARK[arm],
                    color=COL[arm], capsize=2, ms=4, label=LEGEND[arm])
if len(zs):
    ax.axhline(zs["mAP50"].iloc[0], ls="--", color=COL["zero_shot"], lw=0.9)
    ax.text(0.99, zs["mAP50"].iloc[0] + 0.006, "zero-shot baseline", ha="right", va="bottom",
            transform=ax.get_yaxis_transform(), fontsize=7, color=COL["zero_shot"])
ax.set_xlim(-5, None)
ax.set_xlabel("Estimated human annotation time (min)")
ax.set_ylabel("Test mAP@0.5")
ax.legend(loc="lower right", frameon=False)
save(fig, "fig_effort")

# ------------------------------------------------------------- key numbers
zs_row = zs.iloc[0] if len(zs) else None
macros = {
    "manN": len(ms), "manMean": f"{man_mean:.1f}", "manMedian": f"{np.median(ms):.1f}",
    "manTotalMin": f"{ms.sum() / 60:.1f}",
    "assN": len(as_), "assMean": f"{ass_mean:.1f}", "assMedian": f"{np.median(as_):.1f}",
    "assTotalMin": f"{as_.sum() / 60:.1f}",
    "redMean": f"{100 * (1 - ass_mean / man_mean):.0f}",
    "redMedian": f"{100 * (1 - np.median(as_) / np.median(ms)):.0f}",
    "handMin": f"{len(as_) * man_mean / 60:.0f}",
    "saveMin": f"{len(as_) * man_mean / 60 - as_.sum() / 60:.0f}",
    "earlyN": len(early), "earlyMean": f"{early.mean():.1f}",
    "lateN": len(late), "lateMean": f"{late.mean():.1f}" if len(late) else "--",
    "ciLow": f"{ci_low:.1f}", "ciHigh": f"{ci_high:.1f}",
}
if zs_row is not None:
    macros.update({
        "zsMap": f"{zs_row['mAP50']:.3f}", "zsMapx": f"{zs_row['mAP50_95']:.3f}",
        "zsBoba": f"{zs_row[[c for c in class_cols if short_class(c) == 'Boba cup'][0]]:.3f}",
        "zsCoffee": f"{zs_row[[c for c in class_cols if short_class(c) == 'Coffee cup'][0]]:.3f}",
        "zsStraw": f"{zs_row[[c for c in class_cols if short_class(c) == 'Straw'][0]]:.3f}",
    })


def count_boxes(path, min_score=None):
    """Boxes per short class name in a COCO json (optionally above a confidence cut-off)."""
    with open(path) as fh:
        coco = json.load(fh)
    names = {c["id"]: short_class("AP50_" + c["name"]) for c in coco["categories"]}
    counts = {"Boba cup": 0, "Coffee cup": 0, "Straw": 0}
    for a in coco["annotations"]:
        if min_score is not None and a.get("score", 1.0) < min_score:
            continue
        counts[names[a["category_id"]]] = counts.get(names[a["category_id"]], 0) + 1
    return counts


box_macros = {k: "??" for k in ["rawBoba", "rawCoffee", "rawStraw", "rawTotal",
                                "corrBoba", "corrCoffee", "corrStraw", "corrTotal"]}
box_note = "  (box counts skipped: pseudo/corrected json not found)\n"
if os.path.exists(args.pseudo) and os.path.exists(args.corrected):
    raw_c = count_boxes(args.pseudo, args.raw_min_score)
    cor_c = count_boxes(args.corrected)
    for tag, c in [("raw", raw_c), ("corr", cor_c)]:
        box_macros[tag + "Boba"], box_macros[tag + "Coffee"] = c["Boba cup"], c["Coffee cup"]
        box_macros[tag + "Straw"], box_macros[tag + "Total"] = c["Straw"], sum(c.values())
    box_note = (f"  raw (conf >= {args.raw_min_score}): {raw_c}   corrected: {cor_c}\n")
macros.update(box_macros)

with open(os.path.join(TAB, "key_numbers.tex"), "w") as f:
    f.write("% Generated by scripts/07_make_figures.py: do not edit by hand.\n")
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

with open(os.path.join(args.out, "key_numbers.txt"), "w") as f:
    f.write("TIMING (seconds per image unless stated)\n")
    for label, s in rows_t:
        f.write(f"  {label:<24} n={s['n']:>3}  total={s['total']:.1f} min  mean={s['mean']:.1f}  "
                f"median={s['median']:.1f}  std={s['std']:.1f}\n")
    f.write(f"  mean reduction {macros['redMean']}%, median reduction {macros['redMedian']}%\n")
    f.write(f"  bootstrap 95% CI of (manual mean - assisted mean): {ci_low:.1f} to {ci_high:.1f} s\n")
    f.write(f"  first {len(early)} assisted: {early.mean():.1f} s; remaining {len(late)}: "
            f"{late.mean() if len(late) else float('nan'):.1f} s\n")
    f.write(f"  assisted images done by hand would take ~{macros['handMin']} min; "
            f"assisted route took {macros['assTotalMin']} min; saving ~{macros['saveMin']} min\n")
    f.write("\nACCURACY (test set; mean +/- std over seeds)\n")
    for _, r in res.iterrows():
        per_class = "  ".join(f"{short_class(c)}={pm_plain(r[c], r.get(c + '_std'))}" for c in class_cols)
        f.write(f"  {model_label(r):<14} n={int(r['n_train_images']):>3} seeds={int(r['n_seeds'])}  "
                f"mAP50={pm_plain(r['mAP50'], r.get('mAP50_std'))}  "
                f"mAP50-95={pm_plain(r['mAP50_95'], r.get('mAP50_95_std'))}  {per_class}\n")
    f.write("\nBOX COUNTS PER CLASS, assisted batch\n" + box_note)
    f.write("\nESTIMATED ANNOTATION MINUTES (images x mean seconds per image of each method)\n")
    for idx, r in ft.iterrows():
        f.write(f"  {model_label(r):<14} {r['minutes']:.1f} min   mAP50={r['mAP50']:.3f}\n")

print(open(os.path.join(args.out, "key_numbers.txt")).read())
print(f"Figures -> {IMG}\nTables  -> {TAB}")
