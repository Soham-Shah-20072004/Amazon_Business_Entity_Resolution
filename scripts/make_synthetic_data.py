#!/usr/bin/env python3
"""Generate a SMALL synthetic dataset for pipeline smoke-testing ONLY.

This creates obviously-fake toy records (e.g. "Test Bakery 12") so the team
can verify installation and EDA plumbing WITHOUT the real challenge data.
It is NOT external data augmentation — never mix it with real data, never
train on it for submissions.

Usage:
    python scripts/make_synthetic_data.py --out dataset-synth
    python scripts/run_eda.py --data-root dataset-synth

Multilingual smoke tests: pass --with-devanagari K to append K script-mix
records (Latin<->Devanagari cross-script positives, Devanagari same-script
positives, Devanagari distractors). Default 0 leaves output unchanged.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

FIRST = ["Sharma", "Royal", "Metro", "Green", "Sunrise", "National", "City", "Lucky"]
CORE = ["Bakery", "Pharmacy", "Auto Parts", "Textiles", "Cafe", "Hotel", "Motors", "Traders"]
SUFFIX = ["Pvt Ltd", "LLC", "Inc", "Corporation", "Co", ""]
STREETS = ["MG Road", "Park Street", "Main Street", "Gandhi Nagar", "5th Avenue", "Rue de la Paix"]
CITIES = ["Bengaluru 560001", "Mumbai 400001", "New York 10001", "Austin 73301", "Paris 75001"]

# Script-mix pools for --with-devanagari (fake toy records, plumbing only).
DEVA_FIRST = ["राम", "श्याम", "मोहन", "आदित्य", "प्रिया", "सुनीता", "विकास", "अनिल"]
DEVA_CORE = ["मार्केटिंग", "प्रॉपर्टीज", "टेक्सटाइल्स", "फार्मेसी", "होटल", "मोटर्स", "ट्रेडर्स", "बेकरी"]
DEVA_SUFFIX = ["प्राइवेट लिमिटेड", "लिमिटेड", "एलएलपी", ""]
LATIN_MIRROR_FIRST = ["Ram", "Shyam", "Mohan", "Aditya", "Priya", "Sunita", "Vikas", "Anil"]
LATIN_MIRROR_CORE = ["Marketing", "Properties", "Textiles", "Pharmacy", "Hotel", "Motors", "Traders", "Bakery"]
LATIN_MIRROR_SUFFIX = ["Pvt Ltd", "Ltd", "LLP", ""]
DEVA_STREETS = ["एमजी रोड", "पार्क स्ट्रीट", "गांधी नगर", "मुख्य मार्ग"]
DEVA_CITIES = ["बेंगलुरु 560001", "मुंबई 400001", "दिल्ली 110001"]


def synth_name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(CORE)} {rng.choice(SUFFIX)}".strip()


def noisy_copy(text: str, rng: random.Random) -> str:
    if rng.random() < 0.25 and len(text) > 6:  # abbrev-style noise
        text = text.replace("Corporation", "Corp").replace("Private", "Pvt").replace("Limited", "Ltd")
    if rng.random() < 0.20 and len(text) > 4:  # single-char typo
        i = rng.randrange(len(text))
        text = text[:i] + rng.choice("aeiourtns") + text[i + 1:]
    return text


def synth_addr(rng: random.Random) -> str:
    return f"{rng.randint(1, 499)}{rng.choice(['', 'A', 'B'])} {rng.choice(STREETS)}, {rng.choice(CITIES)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset-synth", help="Output dataset root.")
    ap.add_argument("--n-s1", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--with-devanagari", type=int, default=0,
                    help="Append K script-mix records (cross-script + same-script "
                         "Devanagari positives, Devanagari distractors). Default 0.")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    out = Path(args.out)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)

    # train: S1 reference + S2/S3 fragments with known matches (incl. singletons + multi)
    s1_rows, s2_rows, s3_rows, gt_rows = [], [], [], []
    s2_i = s3_i = 0
    for i in range(1, args.n_s1 + 1):
        sid = f"S1-{i:05d}"
        country = rng.choice(["US", "India"])
        name, addr = synth_name(rng), synth_addr(rng)
        s1_rows.append((sid, name, addr, country))
        matched: list[str] = []
        r = rng.random()
        n_match = 0 if r < 0.30 else (1 if r < 0.65 else (2 if r < 0.9 else 3))
        for _ in range(n_match):
            if rng.random() < 0.5:
                s2_i += 1
                cid = f"S2-{s2_i:05d}"
                s2_rows.append((cid, noisy_copy(name, rng), noisy_copy(addr, rng), country))
            else:
                s3_i += 1
                cid = f"S3-{s3_i:05d}"
                s3_rows.append((cid, noisy_copy(name, rng), noisy_copy(addr, rng), country))
            matched.append(cid)
        gt_rows.append((sid, ",".join(matched)))
    # distractors (records with no match)
    for _ in range(120):
        s2_i += 1
        s2_rows.append((f"S2-{s2_i:05d}", synth_name(rng), synth_addr(rng), rng.choice(["US", "India"])))
    for _ in range(120):
        s3_i += 1
        s3_rows.append((f"S3-{s3_i:05d}", synth_name(rng), synth_addr(rng), rng.choice(["US", "India"])))

    # Optional script-mix injection (separate RNG stream: base output is
    # byte-identical when --with-devanagari 0).
    n_deva = max(0, int(args.with_devanagari))
    if n_deva:
        drng = random.Random(args.seed + 7919)
        s1_next = args.n_s1
        for j in range(n_deva):
            fi, ci, si = (drng.randrange(len(DEVA_FIRST)), drng.randrange(len(DEVA_CORE)),
                          drng.randrange(len(DEVA_SUFFIX)))
            deva_name = f"{DEVA_FIRST[fi]} {DEVA_CORE[ci]} {DEVA_SUFFIX[si]}".strip()
            lat_name = (f"{LATIN_MIRROR_FIRST[fi]} {LATIN_MIRROR_CORE[ci]} "
                        f"{LATIN_MIRROR_SUFFIX[si]}").strip()
            deva_addr = (f"{drng.randint(1, 499)} {drng.choice(DEVA_STREETS)}, "
                         f"{drng.choice(DEVA_CITIES)}")
            pat = j % 3
            if pat == 0:  # cross-script positive: Latin S1 <-> Devanagari S2
                s1_next += 1
                sid = f"S1-{s1_next:05d}"
                s1_rows.append((sid, lat_name, synth_addr(drng), "India"))
                s2_i += 1
                cid = f"S2-{s2_i:05d}"
                s2_rows.append((cid, deva_name, deva_addr, "India"))
                gt_rows.append((sid, cid))
            elif pat == 1:  # same-script Devanagari positive (exact copy)
                s1_next += 1
                sid = f"S1-{s1_next:05d}"
                s1_rows.append((sid, deva_name, deva_addr, "India"))
                s2_i += 1
                cid = f"S2-{s2_i:05d}"
                s2_rows.append((cid, deva_name, deva_addr, "India"))
                gt_rows.append((sid, cid))
            else:  # Devanagari distractor in S2 only (script mix, no match)
                s2_i += 1
                cid = f"S2-{s2_i:05d}"
                s2_rows.append((cid, deva_name, deva_addr, "India"))

    cols = ["entity_id", "business_name", "business_address", "country"]
    pd.DataFrame(s1_rows, columns=cols).to_csv(out / "train/train_source1.tsv", sep="\t", index=False)
    pd.DataFrame(s2_rows, columns=cols).to_csv(out / "train/train_source2.tsv", sep="\t", index=False)
    pd.DataFrame(s3_rows, columns=cols).to_csv(out / "train/train_source3.tsv", sep="\t", index=False)
    pd.DataFrame(gt_rows, columns=["source1_entity_id", "matched_entity_ids"]).to_csv(
        out / "train/train_ground_truth.tsv", sep="\t", index=False)

    # test: includes unseen France
    t1, t2, t3 = [], [], []
    for i in range(1, 121):
        t1.append((f"S1-{i:05d}", synth_name(rng), synth_addr(rng),
                   "France" if i % 3 == 0 else rng.choice(["US", "India"])))
    for i in range(1, 101):
        t2.append((f"S2-{i:05d}", synth_name(rng), synth_addr(rng),
                   "France" if i % 4 == 0 else rng.choice(["US", "India"])))
        t3.append((f"S3-{i:05d}", synth_name(rng), synth_addr(rng),
                   "France" if i % 5 == 0 else rng.choice(["US", "India"])))
    pd.DataFrame(t1, columns=cols).to_csv(out / "test/test_source1.tsv", sep="\t", index=False)
    pd.DataFrame(t2, columns=cols).to_csv(out / "test/test_source2.tsv", sep="\t", index=False)
    pd.DataFrame(t3, columns=cols).to_csv(out / "test/test_source3.tsv", sep="\t", index=False)
    print(f"Synthetic smoke-test dataset written to {out} (FAKE DATA — plumbing only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
