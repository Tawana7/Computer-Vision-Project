# Low-Data Cup Detection: Foundation-Model-Assisted vs. Manual Annotation

Measures whether YOLO-World-assisted pseudo-labelling reduces annotation
effort while still producing a dataset that trains a usable disposable-cup
detector, compared against fully manual annotation, across increasing
amounts of training data.

## Claim, experiment, number (per the brief)

- **Claim**: assisted annotation (correcting YOLO-World's proposed boxes)
  is faster per image than manual annotation, and the detector it trains
  is not meaningfully worse than one trained on manual annotations.
- **Experiment**: time both annotation methods on matched-size batches of
  cup images; fine-tune YOLOv8 on nested subsets of the resulting data
  (50/100/150/all); evaluate every checkpoint, plus the untouched
  zero-shot baseline, on one fixed, hand-annotated test set.
- **Number**: mAP@0.5 per subset size (the curve), minutes/image for
  manual vs. assisted annotation (the effort comparison), both in
  `evaluation_results.csv` and your own timing log.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `scripts/01_scrape_images.py` | Downloads cup images from Unsplash's search API, logs provenance |
| 2 | `scripts/02_auto_annotate.py` | YOLO-World zero-shot pseudo-labels + QC preview images |
| 3 | `scripts/03_prepare_review_batches.py` | Splits into gold-test / manual-timed / assisted-timed batches |
| — | *(manual, in CVAT/Roboflow/labelImg)* | Annotate/correct each batch, export as COCO json to `./corrected/` |
| 4 | `scripts/04_coco_to_yolo.py` | Converts corrected COCO → YOLO format, builds size subsets |
| 5 | `scripts/05_train_yolo.py` | Fine-tunes YOLOv8n on each subset |
| 6 | `scripts/06_evaluate.py` | Evaluates every checkpoint + zero-shot baseline on the gold test set |

## Setup

```bash
pip install ultralytics opencv-python requests
export UNSPLASH_ACCESS_KEY=your_key_here   # register a free app at unsplash.com/developers
```

Run steps in order:
```bash
python scripts/01_scrape_images.py
python scripts/02_auto_annotate.py
python scripts/03_prepare_review_batches.py
# -- manual step: import review_batches/* into your annotation tool, export corrected COCO json to ./corrected/, timing your own work --
python scripts/04_coco_to_yolo.py
python scripts/05_train_yolo.py
python scripts/06_evaluate.py
```

All randomness (data splitting, training) is seeded at `SEED = 42` for
reproducibility, per the repo rubric.

## Data and licensing

Images are from Unsplash via their documented search API (not scraped
outside it). Per-image photographer credit and source URL are logged to
`downloaded_cups/provenance.jsonl` at download time. No images contain
identifiable people by search design, but spot-check the QC previews in
step 2 and drop any that do before annotating.

## AI tool use

[Fill in: which parts were AI-assisted, e.g. "pipeline scripts were
drafted with Claude's help and reviewed/modified by the group; model
architecture choices and experiment design are our own."]

## Known limitations / next steps

- Test set size (50 images) is small; confidence intervals on mAP will be
  wide. Worth reporting rather than hiding.
- Only one open-vocabulary model (YOLO-World) is tested as the assistant;
  Grounding DINO is a natural comparison if time allows.
- "Straw" as a class is likely to be the hardest for both zero-shot and
  fine-tuned models (thin, low-contrast object) — worth a qualitative
  failure-case discussion in the report even if it drags mAP down.
