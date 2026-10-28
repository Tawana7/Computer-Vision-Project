# Low-Data Cup Detection: Foundation-Model-Assisted vs. Manual Annotation

Measures whether YOLO-World-assisted pseudo-labelling reduces annotation
effort while still producing a dataset that trains a usable disposable-cup
detector, compared against fully manual annotation, across increasing
amounts of training data.

## Claim, experiment, number (per the brief)

- **Claim**: correcting YOLO-World's proposed boxes is faster per image than
  drawing boxes from scratch, and the detector trained on the corrected labels
  is about as good as one trained on from-scratch labels. The correction step
  itself is also worth doing (raw pseudo-labels alone train a worse detector).
- **Experiment** (all models scored on one fixed, hand-annotated test set;
  each arm uses a single label source, never a mix):
  1. *Effort*: time per image for manual vs. assisted annotation
     (`timing_log.csv`, from `03b_annotation_timer.py`).
  2. *Data curve*: fine-tune YOLOv8n on nested subsets of the assisted images
     (50/100/150/all), each a superset of the smaller.
  3. *Value of correction*: same images as `assisted_all`, but with the raw
     uncorrected pseudo-labels (`raw_all`).
  4. *Labelling method*: `manual_M`, `assisted_M` and `mixed_M` (half and
     half) at the same size M = number of from-scratch images. Different
     images and few of them: read cautiously. `mixed_all` (all manual plus
     assisted, same total size as `assisted_all`) is the practical "use
     everything" model; vs. `assisted_all` it only swaps some assisted images
     for manual ones.
  5. *Baseline*: the zero-shot YOLO-World model, no fine-tuning.
- **Number**: test mAP@0.5 (overall and per class) per arm and size in
  `evaluation_results.csv`, paired with minutes/image from the timing log.

Validation images (carved out of the assisted pool, fixed seed) are used only
for choosing the best epoch; the gold test set is never trained on or used to
choose anything.

**Seeds.** Every model is trained with seed 42; the 30-image models
(`assisted_30`, `manual_30`, `mixed_30`) are additionally trained with seeds 43
and 44 (`python scripts/05_train_yolo.py --only assisted_30 manual_30 mixed_30
--seeds 43 44`) and reported as mean +/- std. The seed changes training
randomness only, not which images are in each subset, so the spread understates
the true uncertainty at small N. `evaluation_results_by_seed.csv` has the raw
per-seed numbers.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `scripts/01_scrape_images.py` | Downloads cup images from Unsplash's search API, logs provenance |
| 2 | `scripts/02_auto_annotate.py` | YOLO-World zero-shot pseudo-labels + QC preview images |
| 3 | `scripts/03_prepare_review_batches.py` | Splits into gold-test / manual-timed / assisted batches |
| 3b | `scripts/03b_annotation_timer.py` | Per-image stopwatch for the manual/assisted timing comparison |
| - | *(manual, in CVAT; see `ANNOTATION_RULES.md`)* | Annotate/correct each batch, export COCO json to `./corrected/` |
| 4 | `scripts/04_coco_to_yolo.py` | Builds the test/val sets and the assisted, raw, manual and mixed training arms |
| 5 | `scripts/05_train_yolo.py` | Fine-tunes YOLOv8n on every arm (`--only`, `--seeds`, `--force`; optional W&B logging) |
| 6 | `scripts/06_evaluate.py` | Scores every model + the zero-shot baseline on the gold test set, per class, mean +/- std over seeds |

## Setup

```bash
python -m pip install ultralytics opencv-python requests wandb
export UNSPLASH_ACCESS_KEY=your_key_here   # register a free app at unsplash.com/developers
python -m wandb login                      # optional: training metrics logged to Weights & Biases
```

Training logs to W&B by default; set `USE_WANDB=0` to turn that off (the
pipeline runs identically without it). The W&B key is stored by `wandb login`
outside the repo and must never be committed.

Run steps in order:
```bash
python scripts/01_scrape_images.py
python scripts/02_prepare_review_batches.py
python scripts/03_auto_annotate.py

# -- manual step: import review_batches/* into your annotation tool, export corrected COCO json to ./corrected/, timing your own work --
python scripts/04_coco_to_yolo.py
python scripts/05_train_yolo.py
python scripts/06_evaluate.py
```

All randomness (data splitting, training) is seeded at `SEED = 42` for
reproducibility.

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