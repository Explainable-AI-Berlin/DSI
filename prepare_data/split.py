"""
Script to split dataset files into train/test sets.

Downloads all JSON files from the function_vectors GitHub repo:
  - dataset_files/abstractive/
  - dataset_files/extractive/

Splits each into train/test and saves results to:
  - data/train/
  - data/test/

Each output file is a JSON array of {input, output} objects.
"""

import json
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO = "ericwtodd/function_vectors"
COMMIT = "fb9eac7"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/dataset_files"
API_BASE = f"https://api.github.com/repos/{REPO}/contents/dataset_files"

seed = 0
split = 0.5

SOURCES = ["abstractive", "extractive"]

OUTPUT_DIRS = {
    "train": Path("data/train"),
    "test": Path("data/test"),
}

def list_json_files(category: str) -> list[str]:
    url = f"{API_BASE}/{category}?ref={COMMIT}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req) as resp:
        entries = json.loads(resp.read().decode())
    return sorted(e["name"] for e in entries if e["name"].endswith(".json"))

def fetch_json(category: str, filename: str) -> pd.DataFrame:
    url = f"{RAW_BASE}/{category}/{filename}"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode())
    return pd.DataFrame(data)[["input", "output"]]

def split_dataset(df: pd.DataFrame, test_size: float, seed: int):
    train, valid = train_test_split(df, test_size=test_size, random_state=seed)
    test, valid = train_test_split(valid, test_size=test_size, random_state=seed)
    return train, valid, test

def main():
    for split_dir in OUTPUT_DIRS.values():
        split_dir.mkdir(parents=True, exist_ok=True)

    total_files = 0
    split_counts = {}

    for category in SOURCES:
        filenames = list_json_files(category)
        print(f"\nProcessing {len(filenames)} files from {category}/")

        for filename in filenames:
            df = fetch_json(category, filename)
            train, valid, test = split_dataset(df, test_size=split, seed=seed)
            combined_test = pd.concat([valid, test], ignore_index=True)

            for split_name, split_df in [("train", train), ("test", combined_test)]:
                out_path = OUTPUT_DIRS[split_name] / filename
                records = split_df.to_dict(orient="records")
                with open(out_path, "w") as f:
                    json.dump(records, f, indent=2)

            task_name = Path(filename).stem
            split_counts[task_name] = {"train": len(train), "test": len(combined_test)}
            total_files += 1
            print(
                f"  {filename}: {len(df)} total -> "
                f"{len(train)} train / {len(combined_test)} test"
            )

    with open(Path("data") / "split_counts.json", "w") as f:
        json.dump(split_counts, f, indent=2)

    print(f"\nDone. Processed {total_files} files.")
    for split_name, split_dir in OUTPUT_DIRS.items():
        files = list(split_dir.glob("*.json"))
        print(f"  {split_dir}/  ({len(files)} files)")

if __name__ == "__main__":
    main()
