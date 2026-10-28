"""
This script converts annotations to ultralytics YOLO format and builds the datasets
for each experimental arm.

Inputs:
  ./corrected/gold_test.json            human annotated from scratch (CVAT export)
  ./corrected/train_pool_assisted.json  pseudo labels corrected by annotator (CVAT export)
  ./corrected/train_pool_manual.json    human annotated from scratch (CVAT export)
  ./pseudo_annotations.json             RAW YOLO-World output from automatic annotation before correction
  the original images in ./review_batches/<split>/

Outputs in ./yolo_dataset (deleted and rebuilt on every run):
  test/            test set, used ONLY for final evaluation.
  val/             fixed validation images taken from the assisted pool. Used
                   by training to pick the best epoch or early stop. Identical
                   for every arm, and never inside any training set.
  train_assisted_N nested subsets of the assisted pool (corrected labels).
                   Each is a strict superset of the smaller ones, so the curve
                   reflects "more data", not "different data".          [arm: assisted]
  train_raw_N      SAME images as train_assisted_N, but with the raw,
                   uncorrected pseudo-labels. Isolates what your correction
                   work bought.                                          [arm: raw]
  train_manual_M   all your from-scratch images (M of them).             [arm: manual]
  train_assisted_M the first M assisted images, so manual_M vs assisted_M is a
                   matched-size comparison of the two labelling methods.
  train_mixed_M    M images, half from-scratch and half assisted. Completes the
                   three-way comparison at size M: manual / assisted / mixed.
  train_mixed_all  all M from-scratch images plus assisted images, mixed into one
                   training set. By default the TOTAL equals assisted_all, so
                   mixed_all vs assisted_all only swaps some assisted images for
                   manual ones (see MIXED_MATCH_ASSISTED_SIZE).        [arm: mixed]

"""
import os
import json
import random
import shutil

CORRECTED_DIR = "./corrected"
BATCHES_DIR = "./review_batches"
PSEUDO_JSON = "./pseudo_annotations.json"
OUT_DIR = "./yolo_dataset"

SUBSET_SIZES = [50, 100, 150, None]  # assisted curve; None = everything left after the val split
RAW_SUBSET_SIZES = [None]            # sizes at which to also build the raw-pseudo-label arm
RAW_MIN_SCORE = 0.25                 # confidence cut-off for the raw arm (step 2 saved boxes down to 0.15).
                                     # This is a choice that affects the result: report it.
MIXED_MATCH_ASSISTED_SIZE = True     # True: mixed_all has the same total size as assisted_all (M manual +
                                     #   the rest assisted), so only the label source differs.
                                     # False: ALL manual + ALL assisted (bigger, the "use everything" model).
VAL_SIZE = 25
SEED = 42

random.seed(SEED)

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)


def abs_posix(path):
    # forward slashes keep the yaml valid on Windows too
    return os.path.abspath(path).replace("\\", "/")


def convert(coco_path, image_dir, out_root, min_score=None):
    """COCO json -> YOLO images/ + labels/ under out_root. Returns (filenames, class_names)."""
    with open(coco_path) as f:
        coco = json.load(f)

    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}  # yolo classes are 0-indexed
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    label_dir = os.path.join(out_root, "labels")
    image_out = os.path.join(out_root, "images")
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(image_out, exist_ok=True)

    written, missing = [], 0
    for img in coco["images"]:
        # CVAT stores file_name with a folder prefix ("train_pool_manual/xyz.jpg").
        # Keep only the bare filename: the images already live directly in image_dir.
        fname = os.path.basename(img["file_name"].replace("\\", "/"))
        src = os.path.join(image_dir, fname)
        if not os.path.exists(src):
            missing += 1
            if missing <= 5:
                print(f"  missing image file: {src}")
            continue

        w, h = img["width"], img["height"]
        lines = []
        for ann in anns_by_image.get(img["id"], []):
            if min_score is not None and ann.get("score", 1.0) < min_score:
                continue
            x, y, bw, bh = ann["bbox"]
            # clamp to the image so a box dragged slightly past the edge stays valid
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(w), x + bw), min(float(h), y + bh)
            if x1 <= x0 or y1 <= y0:
                continue
            lines.append(f"{cat_id_to_idx[ann['category_id']]} {(x0 + x1) / 2 / w:.6f} "
                         f"{(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")

        stem = os.path.splitext(fname)[0]
        with open(os.path.join(label_dir, f"{stem}.txt"), "w") as lf:
            lf.write("\n".join(lines))  # empty file if no boxes -- a valid negative example
        shutil.copy(src, os.path.join(image_out, fname))
        written.append(fname)

    if missing:
        print(f"  {missing} images listed in {os.path.basename(coco_path)} were not found in {image_dir}")
    return written, [c["name"] for c in cats]


def copy_pair(fname, pool_root, dest_root):
    stem = os.path.splitext(fname)[0]
    for sub, name in [("images", fname), ("labels", f"{stem}.txt")]:
        os.makedirs(os.path.join(dest_root, sub), exist_ok=True)
        shutil.copy(os.path.join(pool_root, sub, name), os.path.join(dest_root, sub, name))


def build_subset(tag, parts, class_names, note=""):
    """parts = [(filenames, pool_root), ...]: a training set may draw from several label sources."""
    dest = os.path.join(OUT_DIR, f"train_{tag}")
    n = 0
    for files, pool_root in parts:
        for fname in files:
            copy_pair(fname, pool_root, dest)
            n += 1
    yaml_path = os.path.join(OUT_DIR, f"data_{tag}.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: {abs_posix(os.path.join(dest, 'images'))}\n")
        f.write(f"val: {abs_posix(os.path.join(OUT_DIR, 'val', 'images'))}\n")
        f.write(f"test: {abs_posix(os.path.join(OUT_DIR, 'test', 'images'))}\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")
    print(f"train_{tag}: {n} images{note} -> {yaml_path}")


def pool_path(name):
    return os.path.join(OUT_DIR, f"_pool_{name}")


class_names = None

# --- gold test set (never trained on, never used for choosing anything) ---
gold_path = os.path.join(CORRECTED_DIR, "gold_test.json")
if os.path.exists(gold_path):
    files, class_names = convert(gold_path, os.path.join(BATCHES_DIR, "gold_test"),
                                 os.path.join(OUT_DIR, "test"))
    print(f"test set: {len(files)} images")
else:
    print(f"WARNING: {gold_path} not found. No test set yet, so step 6 cannot run until you export it.")

# --- assisted pool (corrected pseudo-labels) ---
assisted_json = os.path.join(CORRECTED_DIR, "train_pool_assisted.json")
if not os.path.exists(assisted_json):
    raise SystemExit(f"{assisted_json} not found: export train_pool_assisted from CVAT first.")
assisted_files, cn = convert(assisted_json, os.path.join(BATCHES_DIR, "train_pool_assisted"),
                             pool_path("assisted"))
class_names = class_names or cn
print(f"train_pool_assisted: {len(assisted_files)} images")
if not assisted_files:
    raise SystemExit("No assisted images were converted. Check the file names / folders above.")

random.shuffle(assisted_files)

# --- fixed validation split, taken first so it is identical for every arm ---
val_n = min(VAL_SIZE, len(assisted_files) // 5)
val_files, train_files = assisted_files[:val_n], assisted_files[val_n:]
for fname in val_files:
    copy_pair(fname, pool_path("assisted"), os.path.join(OUT_DIR, "val"))
print(f"val set: {len(val_files)} images (from the assisted pool, in no training subset)")
print(f"assisted training pool after val split: {len(train_files)} images\n")

# --- manual pool (from-scratch labels) ---
manual_json = os.path.join(CORRECTED_DIR, "train_pool_manual.json")
manual_files = []
if os.path.exists(manual_json):
    manual_files, _ = convert(manual_json, os.path.join(BATCHES_DIR, "train_pool_manual"),
                              pool_path("manual"))
    print(f"train_pool_manual: {len(manual_files)} images")
else:
    print(f"WARNING: {manual_json} not found: the manual arm will be skipped.")

# --- arm: assisted (nested subsets; includes size M for the matched comparison) ---
sizes = list(SUBSET_SIZES)
if manual_files:
    sizes.append(len(manual_files))
numeric = sorted({s for s in sizes if s is not None and s < len(train_files)})
for s in sizes:
    if s is not None and s >= len(train_files):
        print(f"skipping assisted_{s}: pool only has {len(train_files)} images (would duplicate 'all')")
for s in numeric + ([None] if None in sizes else []):
    n = len(train_files) if s is None else s
    build_subset(f"assisted_{'all' if s is None else s}", [(train_files[:n], pool_path("assisted"))], class_names)

# --- arm: manual (all of it) ---
if manual_files:
    build_subset(f"manual_{len(manual_files)}", [(sorted(manual_files), pool_path("manual"))], class_names)

# --- arm: mixed (manual + assisted labels in one training set) ---
if len(manual_files) >= 2:
    m = len(manual_files)
    rng = random.Random(SEED)
    manual_sorted = sorted(manual_files)

    # mixed_M: same total size as manual_M and assisted_M, half and half
    n_man = m // 2
    n_ass = m - n_man
    if n_ass <= len(train_files):
        build_subset(f"mixed_{m}",
                     [(rng.sample(manual_sorted, n_man), pool_path("manual")),
                      (train_files[:n_ass], pool_path("assisted"))],
                     class_names, note=f" ({n_man} manual + {n_ass} assisted)")

    # mixed_all: every manual image plus assisted ones
    n_ass_all = len(train_files) - m if MIXED_MATCH_ASSISTED_SIZE else len(train_files)
    if n_ass_all > 0:
        build_subset("mixed_all",
                     [(manual_sorted, pool_path("manual")),
                      (train_files[:n_ass_all], pool_path("assisted"))],
                     class_names, note=f" ({m} manual + {n_ass_all} assisted)")
    else:
        print("skipping mixed_all: not enough assisted images to match assisted_all's size")
elif manual_files:
    print("skipping mixed arms: need at least 2 manual images")

# --- arm: raw pseudo-labels on the SAME images as the assisted subsets ---
if os.path.exists(PSEUDO_JSON):
    raw_files, raw_names = convert(PSEUDO_JSON, os.path.join(BATCHES_DIR, "train_pool_assisted"),
                                   pool_path("raw"), min_score=RAW_MIN_SCORE)
    if raw_names != class_names:
        print(f"WARNING: class names differ between pseudo-labels {raw_names} and CVAT labels "
              f"{class_names}. Class INDICES are assumed to line up (same order); check this.")
    have = set(raw_files)
    for s in RAW_SUBSET_SIZES:
        if s is not None and s >= len(train_files):
            continue
        n = len(train_files) if s is None else s
        chosen = [f for f in train_files[:n] if f in have]
        if len(chosen) < n:
            print(f"  note: {n - len(chosen)} images have no raw pseudo-label entry and were left out of raw_{s}")
        build_subset(f"raw_{'all' if s is None else s}", [(chosen, pool_path("raw"))], class_names)
else:
    print(f"WARNING: {PSEUDO_JSON} not found: the raw-pseudo-label arm will be skipped.")

print("\nDone. Run 05_train_yolo.py next -- it trains once per data_*.yaml.")