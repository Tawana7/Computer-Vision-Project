# Annotation Rules

These rules were fixed **before** any manual annotation began and were applied
identically to all three batches (`gold_test`, `train_pool_manual`,
`train_pool_assisted`). They were not changed during annotation. If a rule had
to be added after starting, it is recorded in the Change Log at the bottom.

Why this file exists: bounding-box labels are human judgements. Two careful
people (or one person on two different days) will box the same image
differently unless the rules are written down. Every number in the report
depends on these judgements being consistent.

## Classes

Labels, in this exact order (this order is the class index everywhere in the
pipeline, see `CUSTOM_CLASSES` in `scripts/02_auto_annotate.py`):

| Index | Label | What it covers |
|---|---|---|
| 0 | `plastic boba cup` | Clear plastic cup holding bubble tea / milk tea, typically with a sealed film or dome lid and a wide straw. Tapioca pearls visible is a strong signal. |
| 1 | `disposable coffee cup` | Paper or plastic single-use takeaway cups, hot or iced, with or without a lid or sleeve. Includes clear plastic iced-coffee cups with a flat or dome lid. |
| 2 | `straw` | A drinking straw (plastic, paper, or similar), whether in a cup, held, or lying loose. |

**Boba vs. disposable coffee cup:** if pearls are visible, or it has a sealed
film lid with a wide straw, label it `boba cup`. Otherwise a single-use
takeaway cup is `disposable coffee cup`. If a cup is genuinely ambiguous
between the two, choose `disposable coffee cup`.

## What is NOT annotated (background)

- **Reusable vessels:** glass cups, ceramic mugs, metal or thick plastic
  tumblers, bottles. These are not disposable, so they belong to no class.
  They are deliberately left unboxed and act as hard negatives (the model must
  learn that a cup-shaped object is not automatically trash).
- Lids, sleeves, or straws lying alone are not cups. A lid alone gets no box.
- Cups shown on screens, posters, packaging artwork, or drawings.
- Anything too small or too blurry to identify (see below).
- If you are genuinely unsure whether something is a disposable cup, **do not
  box it.** A missed object costs less than a confidently wrong label.

## Drawing the box

1. **Tight box.** The rectangle touches the outermost visible edges of the
   object on all four sides. No padding, and no cutting off visible parts.
2. **Whole object, including its lid.** A cup with a lid is one object: the box
   includes the lid, and the sleeve if present. It does not include the shadow,
   reflection, hand holding it, or the straw sticking out of the top (the straw
   gets its own box).
3. **Visible extent only.** If part of an object is hidden behind something
   else, box only what is visible. Do not guess where the hidden part would be.
4. **Cut off by the image edge.** Box the visible part, ending at the image
   border.
5. **One box per object.** Several cups means several boxes. Overlapping boxes
   are fine when the objects overlap.
6. **Stacked or nested cups.** Box each cup whose body is clearly distinguishable.
   If a stack cannot be separated into individual cups, box the whole stack as
   one `disposable coffee cup`.
7. **Minimum size.** Skip objects smaller than about 20 pixels on their shorter
   side in the downloaded image.

## Straw-specific rules

- Straws are thin, so box placement matters a lot. Draw the box around the full
  visible length of the straw. It will often be a narrow diagonal rectangle
  containing a lot of background; that is expected.
- A straw sticking out of a cup is boxed on its own, separate from the cup.
- If the straw continues inside a **transparent** cup and is clearly visible,
  include that visible part in the same box. If it cannot be seen clearly
  through the cup, box only the part above the cup.
- Straws inside a reusable glass cup are still boxed. The straw is a class
  regardless of what holds it; the glass cup itself stays unboxed.
- Multiple straws in one cup: one box per straw if they can be told apart,
  otherwise one box around the bunch.

## Images with nothing to box

An image with no qualifying object is kept in the dataset with zero boxes. It is
a valid negative example. It still counts as annotated, and its time is logged
like any other image.

## Assisted batch (`train_pool_assisted`)

The pseudo-labels from YOLO-World are a starting point, not a suggestion to be
trusted. Each image is corrected to the **same standard as a from-scratch
annotation**: tighten loose boxes, delete false positives (including boxes on
reusable glass cups), fix wrong classes, and add anything missed. A box is
accepted only if you would have drawn it yourself under these rules.

## Gold test set (`gold_test`)

Annotated from a blank slate, before correcting any model output, with no
pseudo-labels loaded. No image from this set is ever used for training or for
choosing hyperparameters.

## Process rules
- Images are annotated in CVAT frame order, timed per image with
  `scripts/03b_annotation_timer.py`. A 15-image warm-up batch (not part of the
  300 and excluded from all results) was annotated first to learn the tool.
- Timing excludes breaks (paused with `p`) but includes time spent deciding
  that an image has nothing to box.

## Change log

Record of any rule added or altered after annotation started, with the date and the
number of images already annotated at that point.

| Date | Rule changed | Images already done |
|---|---|---|
| | | |