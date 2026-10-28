"""
This script acts as a per-image annotation stopwatch.

Run this script in a terminal next to CVAT. It records how long each image takes
and appends it to timing_log.csv, so per-image data (mean,
median, spread) is recorded as well one batch totals.

Usage:
  python scripts/03b_annotation_timer.py --batch gold_test --n-images 50
  python scripts/03b_annotation_timer.py --batch train_pool_manual --n-images 30
  python scripts/03b_annotation_timer.py --batch train_pool_assisted --n-images 220
  python scripts/03b_annotation_timer.py --batch warmup --n-images 15   # optional practice, which will be excluded from results

While running (type, then press Enter):
  <just Enter>  you finished the current image -> logs it, starts the next
  p             pause / resume (use it whenever you step away)
  u             undo the last logged image (e.g. you pressed Enter too early)
  q             quit (progress is saved; rerun the same command to resume)

"""
import argparse
import csv
import os
import statistics
import time
from datetime import datetime

CONDITION = {
    "warmup": "warmup",
    "gold_test": "manual",
    "train_pool_manual": "manual",
    "train_pool_assisted": "assisted",
}
FIELDS = ["batch", "condition", "image_index", "seconds", "finished_at"]


def load(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summary(rows, batch):
    secs = [float(r["seconds"]) for r in rows if r["batch"] == batch]
    if not secs:
        print("No images logged yet.")
        return
    print(f"\n{batch}: {len(secs)} images logged, total {sum(secs) / 60:.1f} min, "
          f"mean {statistics.mean(secs):.1f}s, median {statistics.median(secs):.1f}s"
          + (f", stdev {statistics.stdev(secs):.1f}s" if len(secs) > 1 else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, choices=list(CONDITION))
    ap.add_argument("--n-images", type=int, required=True)
    ap.add_argument("--log", default="timing_log.csv")
    args = ap.parse_args()

    rows = load(args.log)

    def n_done():
        return sum(r["batch"] == args.batch for r in rows)

    if n_done() >= args.n_images:
        print(f"{args.batch} already has {n_done()} images logged. Nothing to do.")
        summary(rows, args.batch)
        return

    print(f"Batch '{args.batch}' ({CONDITION[args.batch]}): {n_done()}/{args.n_images} already logged.")
    try:
        input(f"Open image {n_done() + 1} in CVAT, then press Enter to start the clock... ")
    except (EOFError, KeyboardInterrupt):
        return

    start = time.monotonic()
    paused_at = None
    paused_total = 0.0

    try:
        while n_done() < args.n_images:
            idx = n_done() + 1
            cmd = input(f"  image {idx}/{args.n_images} running | Enter=done  p=pause/resume  u=undo  q=quit > ").strip().lower()
            now = time.monotonic()

            if cmd == "p":
                if paused_at is None:
                    paused_at = now
                    print("  paused")
                else:
                    paused_total += now - paused_at
                    paused_at = None
                    print("  resumed")
            elif cmd == "q":
                break
            elif cmd == "u":
                for i in range(len(rows) - 1, -1, -1):
                    if rows[i]["batch"] == args.batch:
                        removed = rows.pop(i)
                        save(args.log, rows)
                        print(f"  removed image {removed['image_index']} ({float(removed['seconds']):.1f}s)")
                        break
                else:
                    print("  nothing to undo")
                start, paused_at, paused_total = now, None, 0.0
            elif cmd == "":
                if paused_at is not None:
                    paused_total += now - paused_at
                    paused_at = None
                secs = now - start - paused_total
                rows.append({
                    "batch": args.batch,
                    "condition": CONDITION[args.batch],
                    "image_index": idx,
                    "seconds": f"{secs:.2f}",
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                })
                save(args.log, rows)
                print(f"  logged image {idx}: {secs:.1f}s")
                start, paused_total = now, 0.0
            else:
                print("  unrecognised command")
    except (EOFError, KeyboardInterrupt):
        print()

    summary(rows, args.batch)
    print(f"Log: {args.log}")


if __name__ == "__main__":
    main()