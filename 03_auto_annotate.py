"""
Step 2: Zero-shot pseudo-labelling with YOLO-World.

Fixes vs. the original annotate.py + to_coco.py:
- Loads the model ONCE and runs inference ONCE per image, then derives both
  the COCO json and the QC preview images from that single pass (the old
  scripts ran inference twice, independently, which wastes time and risks
  the two outputs silently drifting apart if you tune the threshold in
  only one file).
- Uses ultralytics' native YOLO-World support instead of the separate
  `inference` + `supervision` stack, so the labelling step and the later
  fine-tuning step (03/04) share one library.
- CUSTOM_CLASSES order is the single source of truth for category ids --
  documented inline so nobody reorders it in one place and not the other.

"""
import os
import json
import cv2
from ultralytics import YOLOWorld

# --- CONFIGURATION ---
IMAGE_DIR = "./review_batches/train_pool_assisted"
PREVIEW_DIR = "./qc_previews"          
OUTPUT_JSON_PATH = "./pseudo_annotations.json"
CUSTOM_CLASSES = ["boba cup", "disposable coffee cup", "straw"]
CONFIDENCE_THRESHOLD = 0.15  # low on purpose: false negatives are invisible in QC, false positives are not

os.makedirs(PREVIEW_DIR, exist_ok=True)

model = YOLOWorld("yolov8s-worldv2.pt")
model.set_classes(CUSTOM_CLASSES)

coco_dataset = {
    "info": {
        "description": "Cup detection dataset (YOLO-World pseudo-labels, unverified)",
        "version": "0.1-pseudo",
        "contributor": "Auto-labelled, not yet human-verified",
    },
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [
        {"id": idx + 1, "name": name, "supercategory": "vessel"}
        for idx, name in enumerate(CUSTOM_CLASSES)
    ],
}

image_id = 1
annotation_id = 1
image_files = sorted(f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg")))

for filename in image_files:
    image_path = os.path.join(IMAGE_DIR, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read {filename}, skipping")
        continue

    height, width = image.shape[:2]
    coco_dataset["images"].append({
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": filename,
    })

    results = model.predict(image, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

    n_boxes = 0
    for box in results.boxes:
        xmin, ymin, xmax, ymax = map(float, box.xyxy[0].tolist())
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        coco_dataset["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": class_id + 1,
            "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
            "area": (xmax - xmin) * (ymax - ymin),
            "segmentation": [],
            "iscrowd": 0,
            "score": confidence,
        })
        annotation_id += 1
        n_boxes += 1

    # QC preview to eyeball box quality
    preview = image.copy()
    for box in results.boxes:
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
        label = f"{CUSTOM_CLASSES[int(box.cls[0])]} {float(box.conf[0]):.2f}"
        cv2.rectangle(preview, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(preview, label, (xmin, max(ymin - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(PREVIEW_DIR, f"preview_{filename}"), preview)

    print(f"[{image_id}/{len(image_files)}] {filename}: {n_boxes} boxes")
    image_id += 1

with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump(coco_dataset, f, indent=2)

print(f"\nPseudo-labels written to {OUTPUT_JSON_PATH}")
print(f"QC previews written to {PREVIEW_DIR}.")
