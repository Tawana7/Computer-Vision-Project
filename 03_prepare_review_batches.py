"""
Step 3: Split images into batches for MANUAL correction, and set up the
timing measurement your proposal promised.

Why this step exists (read this before running it):
YOLO-World's pseudo-labels are not ground truth. If you fine-tune on them
and then test on more of the same pseudo-labels, your "accuracy" number
only tells you how well the fine-tuned model imitates the zero-shot model
-- it says nothing about whether either of them is finding real cups. You
need at least one batch of images a human annotated from a blank slate.

This script creates three groups from your downloaded images:
  1. gold_test/   -- held out, ANNOTATE THESE FROM SCRATCH, no pseudo-label
                      preview. This is your only trustworthy evaluation set.
  2. train_pool_manual/  -- a small batch you annotate by hand, timed. This
                      gives you the "typical manual annotation time" number.
  3. train_pool_assisted/ -- the rest, where you correct the pseudo-labels
                      from step 2 instead of drawing boxes from scratch.
                      Time this too -- that gap IS your headline result.

Use CVAT, Roboflow, or labelImg (whichever Lab 5 used) to do the actual
annotating -- don't hand-roll a labelling GUI, that's not where your eight
weeks should go. CVAT has built-in per-task time tracking if you want it
for free; otherwise just note wall-clock start/end per batch.
"""
import os
import random
import shutil
import json

IMAGE_DIR = "./downloaded_cups"
OUT_DIR = "./review_batches"
SEED = 42

N_GOLD_TEST = 50          # untouched by auto-labels, your only real evaluation set
N_MANUAL_TIMED = 30       # annotated from scratch, timed -> "manual" data point
# everything else goes to train_pool_assisted, correcting step 2's boxes

random.seed(SEED)  # reproducibility: the brief's repo rubric checks for this

image_files = sorted(f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg")))
random.shuffle(image_files)

if len(image_files) < N_GOLD_TEST + N_MANUAL_TIMED + 20:
    print(f"Warning: only {len(image_files)} images found. Consider scraping more, "
          f"or shrink N_GOLD_TEST / N_MANUAL_TIMED -- a reduced curve is still a valid result.")

gold_test = image_files[:N_GOLD_TEST]
manual_timed = image_files[N_GOLD_TEST:N_GOLD_TEST + N_MANUAL_TIMED]
assisted_pool = image_files[N_GOLD_TEST + N_MANUAL_TIMED:]

for name, files in [("gold_test", gold_test),
                     ("train_pool_manual", manual_timed),
                     ("train_pool_assisted", assisted_pool)]:
    dest = os.path.join(OUT_DIR, name)
    os.makedirs(dest, exist_ok=True)
    for f in files:
        shutil.copy(os.path.join(IMAGE_DIR, f), os.path.join(dest, f))

manifest = {
    "seed": SEED,
    "gold_test": gold_test,
    "train_pool_manual": manual_timed,
    "train_pool_assisted": assisted_pool,
}
with open(os.path.join(OUT_DIR, "split_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"gold_test:            {len(gold_test)} images -> annotate blind, this is eval only")
print(f"train_pool_manual:    {len(manual_timed)} images -> annotate from scratch, TIME YOURSELF")
print(f"train_pool_assisted:  {len(assisted_pool)} images -> correct step 2's pseudo-labels, TIME YOURSELF")
print(f"\nManifest saved to {OUT_DIR}/split_manifest.json for reproducibility.")
print("\nNext: import these folders into CVAT/Roboflow/labelImg, annotate, "
      "export each as COCO json back into this project, then run 04_coco_to_yolo.py")
