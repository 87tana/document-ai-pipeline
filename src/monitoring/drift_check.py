"""
src/monitoring/drift_check.py

Reads recent prediction logs from CloudWatch, compares them against a
reference distribution, and produces an Evidently HTML drift report.

Usage:
    python src/monitoring/drift_check.py \
        --reference data/reference_predictions.csv \
        --output reports/drift_report.html
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PREDICTION_RE = re.compile(r'\{"event":\s*"prediction".*?\}')


def fetch_prediction_logs(log_group: str, hours: int, region: str) -> pd.DataFrame:
    client = boto3.client("logs", region_name=region)
    start_ms = int((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp() * 1000)
    log.info(f"Fetching logs from {log_group} (last {hours}h)")

    events = []
    paginator = client.get_paginator("filter_log_events")
    for page in paginator.paginate(logGroupName=log_group, startTime=start_ms, filterPattern='"event"'):
        events.extend(page.get("events", []))
    log.info(f"Pulled {len(events)} raw log events")

    records = []
    for evt in events:
        m = PREDICTION_RE.search(evt["message"])
        if not m:
            continue
        try:
            payload = json.loads(m.group(0))
            records.append({
                "timestamp": datetime.fromtimestamp(evt["timestamp"] / 1000, timezone.utc),
                "predicted_class": payload["class"],
                "confidence": float(payload["confidence"]),
            })
        except (json.JSONDecodeError, KeyError) as e:
            log.warning(f"Skipping malformed event: {e}")
    df = pd.DataFrame(records)
    log.info(f"Parsed {len(df)} prediction records")
    return df


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame, output: Path) -> dict:
    if current.empty:
        raise ValueError("No current data — no predictions in the time window.")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output))
    log.info(f"Report saved -> {output}")
    return report.as_dict()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log-group", default="/aws/lambda/document-ai-pipeline")
    p.add_argument("--hours", type=int, default=168)
    p.add_argument("--region", default="eu-central-1")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("reports/drift_report.html"))
    args = p.parse_args()

    if not args.reference.exists():
        log.error(f"Reference file not found: {args.reference}")
        return 1
    reference = pd.read_csv(args.reference)
    log.info(f"Loaded reference: {len(reference)} rows")

    current = fetch_prediction_logs(args.log_group, args.hours, args.region)
    if current.empty:
        log.warning("No predictions in window — skipping report.")
        return 0

    keep = ["predicted_class", "confidence"]
    reference = reference[keep]
    current = current[keep]

    summary = run_drift_report(reference, current, args.output)
    drifted = summary.get("metrics", [{}])[0].get("result", {}).get("number_of_drifted_columns", "?")
    total = summary.get("metrics", [{}])[0].get("result", {}).get("number_of_columns", "?")
    log.info(f"Drift summary: {drifted}/{total} columns drifted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
