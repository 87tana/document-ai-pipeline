# src/monitoring/monitor.py
"""
Drift monitoring using Evidently AI.

What this does:
- Loads reference data (training predictions — our baseline)
- Loads current data (recent production predictions)
- Compares them and generates an HTML drift report
- Flags if prediction distribution has shifted

Usage:
    python src/monitoring/monitor.py
"""

import json
import json as json_lib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset


# ── Config ────────────────────────────────────────────────────────────────────

REPORTS_DIR = Path("monitoring_reports")
LOGS_DIR = Path("prediction_logs")

LABELS = [
    "letter", "form", "email", "handwritten", "advertisement",
    "scientific_report", "scientific_publication", "specification",
    "file_folder", "news_article", "budget", "invoice",
    "presentation", "questionnaire", "resume", "memo"
]


# ── Reference data (training baseline) ───────────────────────────────────────

def get_reference_data() -> pd.DataFrame:
    """
    Reference data = what the model predicted during validation.
    Simulates a healthy uniform distribution across all 16 classes.
    In production: replace with real validation set predictions.
    """
    np.random.seed(42)
    n = 400
    predictions = np.random.choice(LABELS, size=n, p=[1/16]*16)
    confidences = np.random.uniform(0.3, 0.9, size=n)
    return pd.DataFrame({
        "predicted_class": predictions,
        "confidence": confidences,
    })


# ── Current production data ───────────────────────────────────────────────────

def get_current_data() -> pd.DataFrame:
    """
    Current data = recent predictions from the API.
    Reads from prediction_logs/predictions.jsonl if it exists.
    Otherwise simulates a drifted distribution for demonstration.
    """
    log_file = LOGS_DIR / "predictions.jsonl"

    if log_file.exists():
        records = []
        with open(log_file) as f:
            for line in f:
                records.append(json.loads(line))
        return pd.DataFrame(records)
    else:
        print("No production logs found. Simulating drifted distribution...")
        np.random.seed(99)
        n = 200
        # Simulate drift: invoice suddenly dominates at 70%
        weights = [0.02] * 16
        weights[11] = 0.70
        weights = [w / sum(weights) for w in weights]
        predictions = np.random.choice(LABELS, size=n, p=weights)
        confidences = np.random.uniform(0.2, 0.7, size=n)
        return pd.DataFrame({
            "predicted_class": predictions,
            "confidence": confidences,
        })


# ── Generate drift report ─────────────────────────────────────────────────────

def generate_drift_report():
    print("Loading reference data...")
    reference_df = get_reference_data()

    print("Loading current data...")
    current_df = get_current_data()

    print(f"Reference: {len(reference_df)} samples")
    print(f"Current:   {len(current_df)} samples")

    print("\nReference prediction distribution:")
    print(reference_df["predicted_class"].value_counts(normalize=True).round(3))

    print("\nCurrent prediction distribution:")
    print(current_df["predicted_class"].value_counts(normalize=True).round(3))

    # Build Evidently datasets
    definition = DataDefinition(
        categorical_columns=["predicted_class"],
        numerical_columns=["confidence"]
    )
    reference = Dataset.from_pandas(reference_df, data_definition=definition)
    current = Dataset.from_pandas(current_df, data_definition=definition)

    # Run drift report
    report = Report([DataDriftPreset()])
    result = report.run(reference, current)

    # Save HTML report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"drift_report_{timestamp}.html"
    result.save_html(str(report_path))

    # Save JSON report for automated alerting
    json_path = REPORTS_DIR / f"drift_report_{timestamp}.json"
    result.save_json(str(json_path))

    # Check drift programmatically from JSON
    with open(json_path) as f:
        report_json = json_lib.load(f)

    try:
        drift_share = report_json["metrics"][0]["value"]["share"]
        drift_detected = drift_share > 0.5
        if drift_detected:
            print("\nALERT: Drift detected! Notify the team.")
        else:
            print("\nOK: No drift detected.")
    except KeyError:
        print("\nCould not determine drift status from report.")

    print(f"\nDrift report saved to: {report_path}")
    return report_path


# ── Prediction logger (called from FastAPI) ───────────────────────────────────

def log_prediction(predicted_class: str, confidence: float):
    """
    Log a single prediction to the prediction log file.
    Called from the FastAPI /classify endpoint after each prediction.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "predictions.jsonl"
    record = {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "timestamp": datetime.now().isoformat()
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    generate_drift_report()
