# src/ingestion/validate.py
import json
from pathlib import Path
from collections import Counter
import numpy as np
from datasets import load_from_disk
from PIL import Image

SAMPLES_DIR = Path("data/samples")#where the data stored
RESULTS_DIR = Path("data/validation_results")#where the validatin report are saved
NUM_CLASSES = 16

def validate(split="train"):#this function validates one dataset split
    data_dir = SAMPLES_DIR / "rvlcdip" / split
    print(f"Validating {split} split...")
    ds = load_from_disk(str(data_dir))

    results = {"split": split, "checks": []}

    def check(name, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
        results["checks"].append({"name": name, "passed": passed, "detail": detail})

    # 1. Non-empty
    check("Non-empty", len(ds) > 0, f"{len(ds)} samples")

    # 2. Required columns
    required = {"image", "label", "category", "ocr_words"}
    check("Required columns present", required.issubset(set(ds.column_names)))

    # 3. Label range
    labels = ds["label"]
    check("Labels in valid range [0,15]",
          min(labels) >= 0 and max(labels) <= 15,
          f"min={min(labels)}, max={max(labels)}")

    # 4. All classes represented
    counts = Counter(labels)
    missing = [i for i in range(NUM_CLASSES) if i not in counts]
    check("All 16 classes present", len(missing) == 0,
          f"missing: {missing}" if missing else "all 16 present")

    # 5. Images valid
    corrupt = 0
    for i in range(min(50, len(ds))):
        try:
            img = ds[i]["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))
            assert img.size[0] > 0 and img.size[1] > 0
        except Exception:
            corrupt += 1
    check("Images valid (50 sample check)", corrupt == 0,
          f"{corrupt} corrupt" if corrupt else "all valid")

    # 6. No null labels
    nulls = sum(1 for label in labels if label is None)
    check("No null labels", nulls == 0, f"{nulls} nulls")

    # 7. OCR words not all empty
    empty_ocr = sum(1 for i in range(len(ds)) if len(ds[i]["ocr_words"]) == 0)
    check("OCR words present", empty_ocr < len(ds) * 0.5,
          f"{empty_ocr}/{len(ds)} samples have empty OCR")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"rvlcdip_{split}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    passed = sum(1 for c in results["checks"] if c["passed"])
    total = len(results["checks"])
    print(f"\n  {passed}/{total} checks passed.")
    if passed == total:
        print("  Data is ready for training.")
    else:
        print("  WARNING: Fix issues before training.")

if __name__ == "__main__":
    validate("train")
    validate("test")
