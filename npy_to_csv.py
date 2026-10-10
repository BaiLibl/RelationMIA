"""Convert a trusted NumPy record cache to CSV: field names, then data rows."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def to_python(value):
    if isinstance(value, np.ndarray):
        return to_python(value.tolist())
    if isinstance(value, np.generic):
        return to_python(value.item())
    if isinstance(value, dict):
        return {to_python(key): to_python(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_python(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_python(item) for item in value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.input.with_suffix(".csv")
    if output.resolve() == args.input.resolve():
        parser.error("Input and output must be different files")

    # Object caches use pickle: load only files from a trusted source.
    records = to_python(np.load(args.input, allow_pickle=True))
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        parser.error("Expected an array of dictionaries")
    # Include every field, even if it appears only in later records.
    fields = list(dict.fromkeys(key for row in records for key in row))
    if "label" in fields:
        fields.remove("label")
        fields.insert(0, "label")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, tuple, dict)) else value
                for key, value in row.items()
            })
    print(f"Saved {len(records)} rows and {len(fields)} columns to {output}")


if __name__ == "__main__":
    main()
