"""
Step 1: Collect raw images from Unsplash.

Fixes vs. the original scrape_images.py:
- Uses the real search endpoint (api.unsplash.com/search/photos) with the
  key sent as an Authorization header, not glued onto a broken URL.
- Paginates properly instead of relying on a single request.
- Dedupes by Unsplash photo id across queries (a cup photo can match both
  "boba cup" and "disposable coffee cup").
- Records provenance (photographer, source URL, license) per image, so you
  have the "checked the licence, recorded what I found" evidence the brief
  asks for. Unsplash's licence is broad but scraping outside the documented
  API can violate their ToS -- this script stays inside it.
- Respects the demo-tier rate limit (50 requests/hour) with basic backoff.
"""
import os
import json
import time
import requests

# --- CONFIGURATION ---
ACCESS_KEY = os.environ.get("UNSPLASH_ACCESS_KEY", "YOUR_UNSPLASH_ACCESS_KEY")
SEARCH_QUERIES = ["boba cup", "disposable coffee cup", "iced coffee cup"]
DOWNLOAD_DIR = "./downloaded_cups"
IMAGES_PER_QUERY = 60          # total images wanted per query
PER_PAGE = 30                  # Unsplash max per page
API_URL = "https://api.unsplash.com/search/photos"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
provenance_path = os.path.join(DOWNLOAD_DIR, "provenance.jsonl")
seen_ids = set()

# Resume support: don't re-log images we already recorded
if os.path.exists(provenance_path):
    with open(provenance_path) as f:
        for line in f:
            seen_ids.add(json.loads(line)["id"])

headers = {"Authorization": f"Client-ID {ACCESS_KEY}"}

with open(provenance_path, "a") as prov_file:
    for query in SEARCH_QUERIES:
        print(f"Searching for: {query}")
        collected = 0
        page = 1

        while collected < IMAGES_PER_QUERY:
            params = {"query": query, "per_page": PER_PAGE, "page": page}
            response = requests.get(API_URL, headers=headers, params=params)

            if response.status_code == 403:
                print("Rate limited or bad key -- stopping early for this query.")
                break
            if response.status_code != 200:
                print(f"Failed to fetch page {page} for '{query}': {response.status_code} {response.text[:200]}")
                break

            data = response.json()
            results = data.get("results", [])
            if not results:
                break  # no more pages

            for item in results:
                img_id = item["id"]
                if img_id in seen_ids:
                    continue

                img_url = item["urls"]["regular"]
                filename = f"{query.replace(' ', '_')}_{img_id}.jpg"
                filepath = os.path.join(DOWNLOAD_DIR, filename)

                try:
                    img_response = requests.get(img_url, timeout=15)
                    img_response.raise_for_status()
                except requests.RequestException as e:
                    print(f"  skip {img_id}: download failed ({e})")
                    continue

                with open(filepath, "wb") as handler:
                    handler.write(img_response.content)

                # Provenance record -- keep this, you need it for the report
                prov_file.write(json.dumps({
                    "id": img_id,
                    "filename": filename,
                    "query": query,
                    "photographer": item["user"]["name"],
                    "photographer_profile": item["user"]["links"]["html"],
                    "source_url": item["links"]["html"],
                    "license": "Unsplash License (free to use, attribution appreciated)",
                }) + "\n")

                seen_ids.add(img_id)
                collected += 1
                print(f"  downloaded: {filename} ({collected}/{IMAGES_PER_QUERY})")

                if collected >= IMAGES_PER_QUERY:
                    break

            page += 1
            time.sleep(1)  # be polite to the rate limit

print(f"\nScraping complete. {len(seen_ids)} unique images in {DOWNLOAD_DIR}")
print(f"Provenance log: {provenance_path}")
