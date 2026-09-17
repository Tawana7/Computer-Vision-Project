"""
Step 6: Evaluate every model on the SAME held-out gold test set.

Compares:
  - the zero-shot YOLO-World baseline (no fine-tuning at all -- this is
    the "something simpler sitting next to it" baseline the brief requires)
  - each fine-tuned checkpoint from step 5, one per training-set size

All against the human-annotated gold_test set from step 3/4, never the
pseudo-labels -- this is the number that's actually trustworthy.
"""
import csv
import glob
import os
from ultralytics import YOLO, YOLOWorld

DATASET_DIR = "./yolo_dataset"
RUNS_DIR = "./runs_cup_detector"
CUSTOM_CLASSES = ["boba cup", "disposable coffee cup", "straw"]  # must match 02_auto_annotate.py order
RESULTS_CSV = "./evaluation_results.csv"

# any data_*.yaml works for pointing at the same fixed test split
eval_yaml = os.path.join(DATASET_DIR, "data_all.yaml")
if not os.path.exists(eval_yaml):
    eval_yaml = sorted(glob.glob(os.path.join(DATASET_DIR, "data_*.yaml")))[0]

rows = []

# --- baseline: zero-shot, never fine-tuned ---
print("=== Baseline: zero-shot YOLO-World (no fine-tuning) ===")
baseline = YOLOWorld("yolov8s-worldv2.pt")
baseline.set_classes(CUSTOM_CLASSES)
metrics = baseline.val(data=eval_yaml, split="val", verbose=False)
rows.append({
    "model": "zero_shot_baseline",
    "train_images": 0,
    "mAP50": metrics.box.map50,
    "mAP50_95": metrics.box.map,
    "precision": metrics.box.mp,
    "recall": metrics.box.mr,
})

# --- each fine-tuned subset ---
checkpoints = sorted(glob.glob(os.path.join(RUNS_DIR, "subset_*", "weights", "best.pt")))
for ckpt in checkpoints:
    tag = ckpt.split(os.sep)[-3].replace("subset_", "")
    print(f"\n=== Fine-tuned on subset: {tag} ===")

    # count how many training images this subset actually had
    img_dir = os.path.join(DATASET_DIR, f"train_{tag}", "images")
    n_images = len(os.listdir(img_dir)) if os.path.isdir(img_dir) else None

    model = YOLO(ckpt)
    metrics = model.val(data=eval_yaml, split="val", verbose=False)
    rows.append({
        "model": f"finetuned_{tag}",
        "train_images": n_images,
        "mAP50": metrics.box.map50,
        "mAP50_95": metrics.box.map,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
    })

with open(RESULTS_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "train_images", "mAP50", "mAP50_95", "precision", "recall"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nResults written to {RESULTS_CSV}")
for r in rows:
    print(f"  {r['model']:<22} n={str(r['train_images']):<6} mAP50={r['mAP50']:.3f}  mAP50-95={r['mAP50_95']:.3f}")

print("\nRemember to also log your annotation-time numbers from step 3 (manual vs. assisted, "
      "minutes per image) next to this table -- that pairing is the actual headline result of the project.")
