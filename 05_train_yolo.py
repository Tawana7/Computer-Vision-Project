"""
Step 5: Fine-tune YOLOv8 on each dataset-size subset from step 4.

This is where your "accuracy vs. number of training images" curve comes
from -- one training run per data_*.yaml, all sharing the same seed,
starting weights, and hyperparameters, so the only thing that varies
between runs is how much data they saw.
"""
import glob
import os
from ultralytics import YOLO

DATASET_DIR = "./yolo_dataset"
SEED = 42
EPOCHS = 50          # small dataset -> more epochs is fine, watch for overfitting on the val curve
BASE_WEIGHTS = "yolov8n.pt"  # nano: fast enough to iterate on a free Colab GPU

yaml_files = sorted(glob.glob(os.path.join(DATASET_DIR, "data_*.yaml")))
if not yaml_files:
    raise SystemExit("No data_*.yaml found -- run 04_coco_to_yolo.py first.")

for yaml_path in yaml_files:
    tag = os.path.basename(yaml_path).replace("data_", "").replace(".yaml", "")
    print(f"\n=== Training on subset: {tag} ===")

    model = YOLO(BASE_WEIGHTS)
    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        seed=SEED,
        project="runs_cup_detector",
        name=f"subset_{tag}",
        exist_ok=True,
        patience=15,   # early stop if val metric plateaus -- small data overfits fast
        verbose=False,
    )

print("\nAll subsets trained. Weights are under runs_cup_detector/subset_<tag>/weights/best.pt")
print("Next: 06_evaluate.py compares each of these, plus the untouched zero-shot model, on the gold test set.")
