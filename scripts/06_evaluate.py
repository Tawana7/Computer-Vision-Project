"""
This script evaluates every model on the SAME held-out gold test set.

Rows (one per model; mean +/- std over training seeds where several exist):
  zero_shot            YOLO-World with no fine-tuning (the "simpler thing" baseline)
  assisted_*           the data-size curve: corrected pseudo-labels, nested subsets
  raw_*                same images as assisted_*, but UNcorrected pseudo-labels
                       -> the gap between raw_all and assisted_all is what your
                          human correction effort bought
  manual_M / assisted_M / mixed_M   three-way comparison at the same size M
                       (different images; with so few images read it cautiously)
  mixed_all            manual + assisted labels together; same total size as
                       assisted_all unless MIXED_MATCH_ASSISTED_SIZE was changed

Outputs
  evaluation_results.csv           one row per model: metric means, <metric>_std, n_seeds
  evaluation_results_by_seed.csv   one row per (model, seed): the raw numbers

Std is the SAMPLE std over seeds (blank with a single seed). It reflects
training randomness only, not which images were chosen for each subset, so it
understates the real uncertainty at small N. Per-class AP50 columns are
included: report them, straws vs cups is the story. All scores are on the
human-annotated gold test set, never on pseudo-labels.
"""
import csv
import glob
import json
import os
import warnings
import numpy as np
import yaml
from ultralytics import YOLO, YOLOWorld

DATASET_DIR = "./yolo_dataset"
RUNS_DIR = os.path.abspath("runs_cup_detector")
PSEUDO_JSON = "./pseudo_annotations.json"
RESULTS_CSV = "./evaluation_results.csv"
BY_SEED_CSV = "./evaluation_results_by_seed.csv"
NAN = float("nan")

# every data_*.yaml has the same test: entry, so any of them points at the gold test set
yamls = sorted(glob.glob(os.path.join(DATASET_DIR, "data_*.yaml")))
if not yamls:
    raise SystemExit("No data_*.yaml found -- run step 4 first.")
eval_yaml = os.path.join(DATASET_DIR, "data_assisted_all.yaml")
if not os.path.exists(eval_yaml):
    eval_yaml = yamls[0]
with open(eval_yaml) as f:
    class_names = yaml.safe_load(f)["names"]



def labeller_prompts():
    """Class names YOLO-World was prompted with when it made the pseudo-labels, read from step 2's
    own output, so the zero-shot baseline is exactly the model that labelled the data (a hand-typed
    copy of the list can silently drift out of sync with 02_auto_annotate.py)."""
    if os.path.exists(PSEUDO_JSON):
        with open(PSEUDO_JSON) as f:
            cats = sorted(json.load(f)["categories"], key=lambda c: c["id"])
        return [c["name"] for c in cats]
    print(f"WARNING: {PSEUDO_JSON} not found; falling back to the class names in the dataset yaml.")
    return list(class_names)


PROMPTS = labeller_prompts()
if len(PROMPTS) != len(class_names):
    raise SystemExit(f"Class count mismatch: pseudo-labels have {PROMPTS}, dataset has {class_names}.")
if PROMPTS != class_names:
    print(f"Note: zero-shot prompts {PROMPTS} differ in wording from dataset names {class_names}; "
          f"scoring matches classes by INDEX, so the ORDER must be the same.")
print(f"Zero-shot baseline prompts: {PROMPTS}")

METRICS = ["mAP50", "mAP50_95", "precision", "recall"] + [f"AP50_{n}" for n in class_names]


def evaluate(model):
    m = model.val(data=eval_yaml, split="test", verbose=False)
    row = {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
           "precision": float(m.box.mp), "recall": float(m.box.mr)}
    row.update({f"AP50_{n}": NAN for n in class_names})  # nan = class absent from the test labels
    for i, c in enumerate(m.box.ap_class_index):
        row[f"AP50_{class_names[int(c)]}"] = float(m.box.ap50[i])
    return row


def aggregate(seed_rows):
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in METRICS:
            vals = np.array([r[k] for r in seed_rows], dtype=float)
            vals = vals[~np.isnan(vals)]
            out[k] = float(vals.mean()) if len(vals) else NAN
            out[k + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else NAN
    return out


def parse_run(ckpt):
    run = os.path.basename(os.path.dirname(os.path.dirname(ckpt)))   # e.g. assisted_30__s43
    tag, _, seed = run.partition("__s")
    return tag, (int(seed) if seed.isdigit() else None)


def tag_sort_key(tag):
    arm, size = tag.rsplit("_", 1)
    return (arm, int(size) if size.isdigit() else 10**9)


def n_train_images(tag):
    d = os.path.join(DATASET_DIR, f"train_{tag}", "images")
    return len(os.listdir(d)) if os.path.isdir(d) else None


by_seed = []   # raw per-(model, seed) rows
summary = []   # one aggregated row per model

print("=== Baseline: zero-shot YOLO-World (no fine-tuning) ===")
baseline = YOLOWorld("yolov8s-worldv2.pt")
baseline.set_classes(PROMPTS)
r = evaluate(baseline)
by_seed.append({"model": "zero_shot", "arm": "zero_shot", "seed": None, "n_train_images": 0, **r})
summary.append({"model": "zero_shot", "arm": "zero_shot", "n_train_images": 0, "n_seeds": 1, **aggregate([r])})

runs = {}
for ckpt in glob.glob(os.path.join(RUNS_DIR, "*", "weights", "best.pt")):
    tag, seed = parse_run(ckpt)
    runs.setdefault(tag, []).append((seed, ckpt))
if not runs:
    print(f"No trained checkpoints found under {RUNS_DIR} -- only the baseline was evaluated.")

for tag in sorted(runs, key=tag_sort_key):
    arm = tag.rsplit("_", 1)[0]
    n_train = n_train_images(tag)
    seed_rows = []
    for seed, ckpt in sorted(runs[tag], key=lambda x: (x[0] is None, x[0])):
        print(f"\n=== {tag}, seed {seed} ({n_train} training images) ===")
        r = evaluate(YOLO(ckpt))
        seed_rows.append(r)
        by_seed.append({"model": tag, "arm": arm, "seed": seed, "n_train_images": n_train, **r})
    summary.append({"model": tag, "arm": arm, "n_train_images": n_train, "n_seeds": len(seed_rows),
                    **aggregate(seed_rows)})

with open(BY_SEED_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "arm", "seed", "n_train_images"] + METRICS)
    w.writeheader()
    w.writerows(by_seed)

with open(RESULTS_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "arm", "n_train_images", "n_seeds"]
                       + METRICS + [k + "_std" for k in METRICS])
    w.writeheader()
    w.writerows(summary)

print(f"\nResults written to {RESULTS_CSV} (and per-seed numbers to {BY_SEED_CSV})\n")


def fmt(row, k):
    v, s = row[k], row[k + "_std"]
    if np.isnan(v):
        return "n/a"
    return f"{v:.3f}" + (f"+-{s:.3f}" if not np.isnan(s) else "")


cols = ["mAP50", "mAP50_95"] + [f"AP50_{n}" for n in class_names]
print(f"{'model':<14}{'n':>5}{'seeds':>6}" + "".join(f"{c[:15]:>17}" for c in cols))
for row in summary:
    print(f"{row['model']:<14}{str(row['n_train_images']):>5}{row['n_seeds']:>6}"
          + "".join(f"{fmt(row, c):>17}" for c in cols))

print("\nPair these with your annotation-time log from step 3 (seconds per image, manual vs assisted):"
      "\naccuracy per annotation-minute is the headline result of the project.")