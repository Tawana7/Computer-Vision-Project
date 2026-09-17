"""
Step 4: Convert corrected COCO annotations into ultralytics YOLO format,
and build nested training subsets so you can plot accuracy vs. dataset size.

Expects, after step 3's manual correction round-trips through CVAT/Roboflow:
  ./corrected/train_pool_manual.json      (COCO, exported from your tool)
  ./corrected/train_pool_assisted.json    (COCO, exported from your tool)
  ./corrected/gold_test.json              (COCO, exported from your tool)
plus the original images still in ./review_batches/<split>/

Produces ./yolo_dataset/ with images/ and labels/ for train, and the
subset sizes you asked for, each a strict superset of the smaller ones
(so the curve reflects "more data", not "different data").
"""
import os
import json
import random
import shutil

CORRECTED_DIR = "./corrected"
BATCHES_DIR = "./review_batches"
OUT_DIR = "./yolo_dataset"
SUBSET_SIZES = [50, 100, 150, None]  # None = use everything available
SEED = 42

random.seed(SEED)


def coco_to_yolo_labels(coco_path, image_dir, label_out_dir, image_out_dir):
    with open(coco_path) as f:
        coco = json.load(f)

    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}  # yolo classes are 0-indexed
    images_by_id = {img["id"]: img for img in coco["images"]}

    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    os.makedirs(label_out_dir, exist_ok=True)
    os.makedirs(image_out_dir, exist_ok=True)

    written = []
    for img_id, img in images_by_id.items():
        src_path = os.path.join(image_dir, img["file_name"])
        if not os.path.exists(src_path):
            print(f"  missing image file: {src_path}, skipping")
            continue

        w, h = img["width"], img["height"]
        lines = []
        for ann in anns_by_image.get(img_id, []):
            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            cls_idx = cat_id_to_idx[ann["category_id"]]
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        stem = os.path.splitext(img["file_name"])[0]
        with open(os.path.join(label_out_dir, f"{stem}.txt"), "w") as lf:
            lf.write("\n".join(lines))  # empty file if no boxes -- that's a valid negative example

        shutil.copy(src_path, os.path.join(image_out_dir, img["file_name"]))
        written.append(img["file_name"])

    return written, [c["name"] for c in cats]


# --- gold test set (fixed, never subsetted) ---
class_names = None
if os.path.exists(os.path.join(CORRECTED_DIR, "gold_test.json")):
    files, class_names = coco_to_yolo_labels(
        os.path.join(CORRECTED_DIR, "gold_test.json"),
        os.path.join(BATCHES_DIR, "gold_test"),
        os.path.join(OUT_DIR, "test", "labels"),
        os.path.join(OUT_DIR, "test", "images"),
    )
    print(f"test set: {len(files)} images")

# --- pooled training data (manual + assisted, corrected) ---
all_train_files = []
for split_name, json_name in [("train_pool_manual", "train_pool_manual.json"),
                               ("train_pool_assisted", "train_pool_assisted.json")]:
    path = os.path.join(CORRECTED_DIR, json_name)
    if not os.path.exists(path):
        print(f"  {path} not found yet, skipping")
        continue
    files, cn = coco_to_yolo_labels(
        path,
        os.path.join(BATCHES_DIR, split_name),
        os.path.join(OUT_DIR, "_pool", "labels"),
        os.path.join(OUT_DIR, "_pool", "images"),
    )
    class_names = class_names or cn
    all_train_files.extend(files)

random.shuffle(all_train_files)

# --- build nested subsets: subset(50) subset subset(100) etc. ---
for size in SUBSET_SIZES:
    n = len(all_train_files) if size is None else min(size, len(all_train_files))
    subset_files = all_train_files[:n]
    tag = "all" if size is None else str(size)

    img_out = os.path.join(OUT_DIR, f"train_{tag}", "images")
    lbl_out = os.path.join(OUT_DIR, f"train_{tag}", "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    for fname in subset_files:
        stem = os.path.splitext(fname)[0]
        shutil.copy(os.path.join(OUT_DIR, "_pool", "images", fname), os.path.join(img_out, fname))
        shutil.copy(os.path.join(OUT_DIR, "_pool", "labels", f"{stem}.txt"), os.path.join(lbl_out, f"{stem}.txt"))

    # per-subset data.yaml for ultralytics
    yaml_path = os.path.join(OUT_DIR, f"data_{tag}.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: {os.path.abspath(img_out)}\n")
        f.write(f"val: {os.path.abspath(os.path.join(OUT_DIR, 'test', 'images'))}\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")

    print(f"train_{tag}: {n} images -> {yaml_path}")

print("\nDone. Run 05_train_yolo.py next -- it trains once per data_*.yaml.")
