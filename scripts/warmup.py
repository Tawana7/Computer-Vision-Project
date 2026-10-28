"""
This script prepares an optional warmup batch of images to practice annotating before the timed batches. 
It selects 15 random images from the downloaded_cups directory that are not already used in the 
gold_test, train_pool_manual, or train_pool_assisted batches. It should be noted that if no more than 300 
images are downloaded, the warmup batch will be empty.
"""

import json, os, random, shutil
m = json.load(open("review_batches/split_manifest.json"))
used = set(m["gold_test"] + m["train_pool_manual"] + m["train_pool_assisted"])
left = sorted(f for f in os.listdir("downloaded_cups")
              if f.lower().endswith((".jpg", ".jpeg", ".png")) and f not in used)
random.seed(42)
os.makedirs("review_batches/warmup", exist_ok=True)
for f in random.sample(left, 15):
    shutil.copy(f"downloaded_cups/{f}", f"review_batches/warmup/{f}")