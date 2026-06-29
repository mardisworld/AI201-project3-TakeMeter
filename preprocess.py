"""
preprocess.py — clean the raw Reddit export and build train/val/test splits.

The raw CSV (`Text,Label`) is a messy spreadsheet export:
  - non-UTF-8 bytes and "mojibake" smart quotes (Ò Ó Õ -> " " ')
  - stray tab characters sprinkled inside comment text
  - comments that contain commas and line breaks, so a naive comma split fails
  - the label always appears as the LAST field on each record

This script parses on the *label* (the only reliable anchor), normalizes the
text, maps each label to its integer id, and writes a stratified split so the
rare classes (analysis, question) appear in every split.

Usage:
    python preprocess.py
    python preprocess.py --input ../Reddit-thread-on-American-presidency.csv --outdir data
    python preprocess.py --seed 42 --val 0.15 --test 0.15

Outputs (written to --outdir, default ./data):
    data/train.csv, data/val.csv, data/test.csv   # columns: text,label,label_id
    data/clean.csv                                 # the full cleaned dataset
"""

import argparse
import csv
import os
import random
import re
from collections import defaultdict, Counter

from labels import LABEL_MAP  # single source of truth for the label taxonomy

# mojibake fixes: how mis-decoded smart punctuation shows up -> what it should be
MOJIBAKE = {
    "Ò": '"', "Ó": '"', "Õ": "'", "Ô": "'", "Ð": "-", "É": "...",
    "Ê": " ", "Ñ": "-", "Â": "", "â€™": "'", "â€œ": '"', "â€\x9d": '"',
}

# a record ends with: ,<optional quote><label><optional quote><trailing tabs/spaces><newline-or-EOF>
RECORD_RE = re.compile(
    r'(?P<text>.*?),\s*"?(?P<label>analysis|op_response|response|question)"?[\t "]*(?:\r?\n|\Z)',
    re.DOTALL,
)


def read_text(path):
    """Read the file as bytes and decode leniently; the export is not valid UTF-8."""
    with open(path, "rb") as f:
        raw = f.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def clean_text(text):
    """Normalize a single comment's text."""
    for bad, good in MOJIBAKE.items():
        text = text.replace(bad, good)
    text = text.replace("\t", " ")            # stray tabs -> spaces
    text = text.strip().strip('"').strip()    # drop wrapping quotes
    text = text.replace('""', '"')            # un-double CSV-escaped quotes
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text


def parse_records(content):
    """Yield (clean_text, label) for every record anchored on a trailing label."""
    # drop a leading header line if present
    content = re.sub(r"^\s*\"?Text\s*,\s*Label.*?(?:\r?\n)", "", content, count=1, flags=re.IGNORECASE)
    for m in RECORD_RE.finditer(content):
        text = clean_text(m.group("text"))
        label = m.group("label")
        if text:  # skip empties
            yield text, label


def stratified_split(rows, val_frac, test_frac, seed):
    """Split per-label so every class is represented in train/val/test."""
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for label, items in by_label.items():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * test_frac)) if n > 2 else 0
        n_val = max(1, round(n * val_frac)) if n > 2 else 0
        test += items[:n_test]
        val += items[n_test:n_test + n_val]
        train += items[n_test + n_val:]
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "label_id"])
        for r in rows:
            w.writerow([r["text"], r["label"], r["label_id"]])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Clean the Reddit export and build splits.")
    ap.add_argument("--input", default=os.path.join(here, "..", "Reddit-thread-on-American-presidency.csv"))
    ap.add_argument("--outdir", default=os.path.join(here, "data"))
    ap.add_argument("--val", type=float, default=0.15, help="validation fraction")
    ap.add_argument("--test", type=float, default=0.15, help="test fraction")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    content = read_text(args.input)
    rows = []
    for text, label in parse_records(content):
        label_id = LABEL_MAP[label][0]
        rows.append({"text": text, "label": label, "label_id": label_id})

    if not rows:
        raise SystemExit(f"No records parsed from {args.input!r} — check the file path/format.")

    os.makedirs(args.outdir, exist_ok=True)
    write_csv(os.path.join(args.outdir, "clean.csv"), rows)

    train, val, test = stratified_split(rows, args.val, args.test, args.seed)
    write_csv(os.path.join(args.outdir, "train.csv"), train)
    write_csv(os.path.join(args.outdir, "val.csv"), val)
    write_csv(os.path.join(args.outdir, "test.csv"), test)

    # report
    dist = Counter(r["label"] for r in rows)
    print(f"Parsed {len(rows)} labeled comments from {os.path.basename(args.input)}\n")
    print(f"{'label':<12}{'total':>7}{'train':>7}{'val':>6}{'test':>6}")
    for label in LABEL_MAP:
        t = sum(1 for r in train if r["label"] == label)
        v = sum(1 for r in val if r["label"] == label)
        s = sum(1 for r in test if r["label"] == label)
        print(f"{label:<12}{dist.get(label, 0):>7}{t:>7}{v:>6}{s:>6}")
    print(f"{'TOTAL':<12}{len(rows):>7}{len(train):>7}{len(val):>6}{len(test):>6}")
    print(f"\nWrote: {args.outdir}/clean.csv, train.csv, val.csv, test.csv")


if __name__ == "__main__":
    main()
