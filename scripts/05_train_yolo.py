"""
This script fine-tunes YOLOv8 on every dataset built in the previous script.

One training run per (data_<arm>_<size>.yaml, seed). All runs share the same
starting weights and hyperparameters, so the only things that differ between
runs are the data, its labels, and the training seed.

Usage
  python scripts/05_train_yolo.py                          # every dataset, seed 42
  python scripts/05_train_yolo.py --only assisted_30 manual_30 mixed_30 --seeds 43 44
                                                           # add two more seeds for the 30-image models
  python scripts/05_train_yolo.py --only assisted_all raw_all
  python scripts/05_train_yolo.py --force                  # retrain even if best.pt exists
  python scripts/05_train_yolo.py --epochs 200 --force     # longer training (all arms must use the same value!)
  python scripts/05_train_yolo.py --epochs 70 --equal-steps --force   # same iterations for every dataset (recommended)
Finished runs are skipped, so a dropped Colab session can simply be restarted
(use --force after rebuilding the dataset in step 4, or stale models get reused).

Why --equal-steps exists
  An "epoch" is one pass over the data, so a 30-image run gets ~2 optimiser
  iterations per epoch and a 195-image run ~13. With the same epoch count (or with
  early stopping, which ended the 30-image runs after ~16 epochs = ~32 iterations)
  small datasets get far less training, and, as I recall ultralytics' code, its
  learning-rate warm-up lasts at least 100 iterations, so those runs never left
  warm-up. A "more data" curve then mixes more data with more training.
  --equal-steps gives every dataset (about) the same number of iterations as the
  LARGEST dataset gets with --epochs epochs, and turns early stopping off
  (early stopping would break the equal-iterations guarantee).

What the seed does and does NOT cover: it changes training randomness (head
initialisation, augmentation, data order). It does NOT change WHICH images are
in each subset; those are fixed in step 4. At very small N the choice of images
matters at least as much as the seed, so seed spread is a lower bound on the
real uncertainty.

Runs are saved as runs_cup_detector/<tag>__s<seed>/.

Weights & Biases logging (optional)
  Setup (once):   python -m pip install wandb
                  python -m wandb login   (key is stored outside the repo; never commit it)
  Disable:        USE_WANDB=0             (Windows: set USE_WANDB=0)
  Offline:        WANDB_MODE=offline      (sync later with `wandb sync`)
If wandb is missing or login fails, training still runs normally. Runs are
grouped by arm; each run's config stores arm, seed and n_train_images. The
metrics shown while training are on the VALIDATION split; the gold-test numbers
for your report come from 06_evaluate.py.
"""
import argparse
import glob
import math
import os
from ultralytics import YOLO, settings

DATASET_DIR = "./yolo_dataset"
RUNS_DIR = os.path.abspath("runs_cup_detector")  # absolute, so ultralytics can't relocate it under runs/detect
SEED = 42
EPOCHS = 70          # small dataset -> more epochs is fine, watch for overfitting on the val curve
PATIENCE = 15        # early stopping: small data overfits fast
BATCH = 16           # explicit, so iterations per epoch is known exactly
BASE_WEIGHTS = "yolov8n.pt"  # nano: fast enough to iterate on a free Colab GPU
WANDB_PROJECT = "cup-detection-low-data"
LABEL_SOURCE = {"assisted": "pseudo-labels corrected by human",
                "raw": "uncorrected YOLO-World pseudo-labels",
                "manual": "human from scratch",
                "mixed": "human from scratch + corrected pseudo-labels"}

ap = argparse.ArgumentParser()
ap.add_argument("--only", nargs="+", help="train only these tags, e.g. assisted_all raw_all")
ap.add_argument("--seeds", nargs="+", type=int, default=[SEED], help="training seeds (default: 42)")
ap.add_argument("--force", action="store_true", help="retrain even if best.pt already exists")
ap.add_argument("--epochs", type=int, default=EPOCHS, help=f"training epochs (default: {EPOCHS})")
ap.add_argument("--equal-steps", action="store_true",
                help="give every dataset ~the same number of iterations as the largest one gets with --epochs "
                     "(early stopping off)")
ap.add_argument("--patience", type=int, default=PATIENCE,
                help=f"stop if val mAP hasn't improved for this many epochs (default: {PATIENCE})")
args = ap.parse_args()

# --- optional W&B setup: never let logging break training ---
wandb = None
if os.environ.get("USE_WANDB", "1") == "1":
    try:
        import wandb as _wandb
        if _wandb.login():          # stored key / WANDB_API_KEY; prompts once if neither exists
            wandb = _wandb
        else:
            print("W&B login failed; continuing without W&B logging.")
    except Exception as e:  # not installed, no network, no key, ...
        print(f"W&B unavailable ({type(e).__name__}: {e}); continuing without W&B logging.")
settings.update({"wandb": wandb is not None})  # ultralytics' built-in callback follows this

yaml_files = sorted(glob.glob(os.path.join(DATASET_DIR, "data_*.yaml")))
if not yaml_files:
    raise SystemExit("No data_*.yaml found -- run 04_coco_to_yolo.py first.")

def count_train(tag):
    d = os.path.join(DATASET_DIR, f"train_{tag}", "images")
    return len(os.listdir(d)) if os.path.isdir(d) else None


all_tags = [os.path.basename(y)[len("data_"):-len(".yaml")] for y in yaml_files]
max_n = max((count_train(t) or 0) for t in all_tags)
target_iters = math.ceil(max_n / BATCH) * args.epochs   # iterations the largest dataset gets

for yaml_path in yaml_files:
    tag = os.path.basename(yaml_path)[len("data_"):-len(".yaml")]   # e.g. assisted_50
    arm = tag.rsplit("_", 1)[0]                                      # assisted / raw / manual / mixed
    if args.only and tag not in args.only:
        continue

    n_train = count_train(tag)
    ipe = math.ceil(n_train / BATCH) if n_train else None   # iterations per epoch
    if args.equal_steps and ipe:
        epochs_i = math.ceil(target_iters / ipe)
        patience_i = epochs_i      # early stopping off: it would break equal iterations
    else:
        epochs_i, patience_i = args.epochs, args.patience

    for seed in args.seeds:
        run_name = f"{tag}__s{seed}"
        best = os.path.join(RUNS_DIR, run_name, "weights", "best.pt")
        if os.path.exists(best) and not args.force:
            print(f"\n=== {run_name}: already trained, skipping (use --force to retrain) ===")
            continue
        print(f"\n=== Training {run_name}: {n_train} images, labels = {LABEL_SOURCE.get(arm, arm)} ===")
        print(f"    epochs={epochs_i}, patience={patience_i}"
              + (f", ~{epochs_i * ipe} iterations" if ipe else ""))

        if wandb is not None:
            wandb.init(
                project=WANDB_PROJECT,
                group=arm,
                name=run_name,
                config={
                    "tag": tag,
                    "arm": arm,
                    "label_source": LABEL_SOURCE.get(arm, arm),
                    "n_train_images": n_train,
                    "seed": seed,
                    "epochs": epochs_i,
                    "patience": patience_i,
                    "equal_steps": args.equal_steps,
                    "iterations_per_epoch": ipe,
                    "batch": BATCH,
                    "base_weights": BASE_WEIGHTS,
                    "data_yaml": yaml_path,
                },
            )

        model = YOLO(BASE_WEIGHTS)
        model.train(
            data=yaml_path,
            epochs=epochs_i,
            batch=BATCH,
            seed=seed,
            project=RUNS_DIR,
            name=run_name,
            exist_ok=True,
            patience=patience_i,
            verbose=False,
            freeze=10
        )

        if wandb is not None:
            wandb.finish()  # no-op if ultralytics already closed the run

print(f"\nDone. Weights are under {RUNS_DIR}/<tag>__s<seed>/weights/best.pt")
print("Next: 06_evaluate.py scores every model, plus the untouched zero-shot baseline, on the gold test set.")