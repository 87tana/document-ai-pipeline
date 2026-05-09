# src/ingestion/download.py
"""
Download and prepare datasets.
Usage:
    python src/ingestion/download.py --dataset rvlcdip --sample
"""

import argparse
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"

RVL_CDIP_LABELS = [
    "letter", "form", "email", "handwritten", "advertisement",
    "scientific_report", "scientific_publication", "specification",
    "file_folder", "news_article", "budget", "invoice",
    "presentation", "questionnaire", "resume", "memo"
]

def download_rvlcdip(sample: bool = False):
    print("Downloading RVL-CDIP...")
    out_dir = SAMPLES_DIR / "rvlcdip" if sample else PROCESSED_DIR / "rvlcdip"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("dvgodoy/rvl_cdip_mini")

    if sample:
        # Only 100 images per split — fast for CPU development
        for split in ["train", "test"]:
            ds = dataset[split].shuffle(seed=42).select(range(min(500, len(dataset[split]))))
            ds.save_to_disk(str(out_dir / split))
        print(f"Sample saved to {out_dir}")
    else:
        for split in ["train", "validation", "test"]:
            dataset[split].save_to_disk(str(out_dir / split))
        print(f"Full dataset saved to {out_dir}")

    # Save label map
    with open(out_dir / "label_map.txt", "w") as f:
        for i, label in enumerate(RVL_CDIP_LABELS):
            f.write(f"{i}\t{label}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["rvlcdip", "funsd"], default="rvlcdip")
    parser.add_argument("--sample", action="store_true",
                        help="Download small sample only (recommended for CPU)")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset == "rvlcdip":
        download_rvlcdip(sample=args.sample)

    print("Done.")

if __name__ == "__main__":
    main()
