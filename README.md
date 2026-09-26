# Business Entity Resolution — Amazon ML Challenge 2026

**Stage 1: dataset audit + entity-resolution EDA** (this repo state).
Later stages (blocking pipeline, pair classifier, thresholding, inference,
submission files) will extend this layout without restructuring it.

## 1. The challenge

Link noisy business records across 3 sources with **no common identifiers**:

- **Source 1 is the deduplicated reference table.** For each S1 entity, find all
  matching records in Source 2 and Source 3. Each S1 may match **zero, one, or
  many** records.
- Train: `train_source1/2/3.tsv` + `train_ground_truth.tsv`
  (`matched_entity_ids` = comma-separated S2/S3 ids, empty for singletons).
- Test: `test_source1/2/3.tsv`, no labels. **France appears in test but not in
  train** → `country` is an **open set** (never hard-code `{US, India}`).
- Metric: **macro F0.5 per S1 entity** (precision-weighted; correct empty lists
  score 1.0, any false merge on a singleton scores 0.0).
- **Strictly prohibited:** external lookup — no entity APIs, government
  registries, geocoding APIs, or internet augmentation. This repo is fully
  **offline**: no network calls anywhere.

All files are **tab-separated** — the code always reads with `sep="\t", dtype=str`.

## 2. Repository structure

```text
business_entity_resolution/          <- project root (run commands from here)
├── README.md
├── requirements.txt                 <- pinned, Windows-friendly
├── .gitignore
├── config/config.yaml               <- all EDA knobs (seeds, top-k, thresholds)
├── data/README.md                   <- where to put the challenge dataset/
├── src/business_entity_resolution/  <- the package (imported by scripts)
│   ├── __init__.py
│   ├── config.py        # YAML + BER_DATA_ROOT + CLI precedence, path resolution
│   ├── io.py            # TSV loading (sep="\t"), ground-truth parsing
│   ├── normalization.py # conservative (+aggressive-for-analysis) normalization
│   ├── features.py      # pair similarity features (RapidFuzz + numeric/country)
│   ├── validation.py    # TSV/ID/ground-truth integrity checks
│   ├── eda.py           # Stage-1 analyses -> eda/*.csv, figures, casebook, summary
│   └── utils.py         # seeding, logging, automatic run-history recording
├── scripts/
│   ├── run_eda.py               # THE pipeline: load->validate->EDA->summary->log
│   ├── validate_data.py         # validation only (exit 0/1)
│   ├── log_experiment.py        # human experiment logger -> logs/experiment_log.csv
│   └── make_synthetic_data.py   # toy fake data for plumbing smoke-tests only
├── eda/                         # generated artifacts 01..09 CSVs + 10 HTML casebook
│   └── figures/                 # generated PNGs
├── logs/
│   ├── experiment_log.csv       # human experiment history (append-only!)
│   ├── run_history.jsonl        # automatic per-execution log (JSON lines)
│   └── README.md                # team workflow: new vs rerun vs submission
├── notebooks/01_eda.ipynb       # thin companion: calls the SAME functions
├── output/                      # submission TSVs go here in later stages
└── reports/eda_summary.md       # generated Finding->Evidence->Implication summary
```

## 3. Configuring the dataset location (no hard-coded paths)

Precedence: `--data-root` flag > `BER_DATA_ROOT` env var > `config/config.yaml`
(default `dataset/` inside the repo root).

```bat
:: A. copy/symlink dataset/ into the repo root, then no flag is needed
mklink /D dataset "C:\Users\panum\Downloads\amazon\...\dataset"

:: B. explicit flag (recommended on your machine)
python scripts/run_eda.py --data-root "C:\Users\panum\Downloads\amazon\Amazon ML challenge\student_resource\dataset"

:: C. environment variable
set BER_DATA_ROOT=C:\Users\panum\Downloads\amazon\Amazon ML challenge\student_resource\dataset
python scripts/run_eda.py
```

Expected layout under the dataset root:

```text
dataset/train/train_source1.tsv  dataset/test/test_source1.tsv
dataset/train/train_source2.tsv  dataset/test/test_source2.tsv
dataset/train/train_source3.tsv  dataset/test/test_source3.tsv
dataset/train/train_ground_truth.tsv
```

## 4. Installation (Windows / VS Code)

```bat
cd path\to\business_entity_resolution
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Python ≥ 3.10. All dependencies ship Windows wheels (no compiler needed).
Verify with the synthetic smoke test (fake data, plumbing only):

```bat
python scripts/make_synthetic_data.py --out dataset-synth
python scripts/run_eda.py --data-root dataset-synth
```

## 5. Data validation

```bat
python scripts/validate_data.py --data-root "C:\path\to\dataset"
```

Checks: tab parsing, expected columns, `S1-/S2-/S3-` prefixes, duplicate IDs,
unexpected prefixes, embedded tabs/newlines, ground-truth referential integrity
(S1 ids exist, matched ids exist in S2/S3, no self-matches, no dup ids in a list).
Exit code `0` = safe to proceed (warnings ok), `1` = errors to fix.

## 6. Running the EDA (the one command)

```bat
python scripts/run_eda.py --data-root "C:\path\to\dataset"
```

It (1) loads data, (2) validates, (3) runs all EDA sections, (4) writes CSVs,
(5) writes figures, (6) writes the HTML casebook, (7) writes
`reports/eda_summary.md`, (8) appends to `logs/run_history.jsonl`.
Deterministic via fixed seeds (`config/config.yaml: eda.random_seed`); offline.

## 7. Outputs

| Artifact | Content |
|---|---|
| `eda/01_data_audit.csv` | rows, unique/dup IDs, countries, missingness, length stats per table |
| `eda/02_ground_truth_match_distribution.csv` | 0/1/2/3/4+ matches, S2 vs S3, singleton/S2-only/S3-only/mixed |
| `eda/03_positive_pair_feature_summary.csv` | positive similarity stats by pair/country/missingness/exactness |
| `eda/04_hard_negative_feature_summary.csv` | same for random/name/address/hybrid negatives |
| `eda/05_normalization_collision_report.csv` | uniqueness collapse per representation + cluster sizes |
| `eda/06_blocking_metrics.csv` | per-blocker S1–S2/S1–S3/overall recall + candidates/S1 + burden |
| `eda/07_blocking_positive_coverage.csv` | per-positive rescue map (`rescued_only_by`) |
| `eda/08_country_shift_report.csv` | per-country stats + char/token coverage vs train vocab |
| `eda/09_candidate_graph_diagnostics.csv` | degrees, hubs, components of the bipartite candidate graph |
| `eda/10_casebook_train_pairs.html` | sortable/filterable difficult-pair review (self-contained) |
| `eda/11_script_distribution.csv` | FULL-data script mix (Latin/Devanagari/Cyrillic/Arabic/mixed/…) per source × field |
| `eda/12_transliteration_collision_report.csv` | SAMPLED raw-vs-norm-vs-transliterated uniqueness + cross-script groups |
| `eda/13_cross_script_positive_pairs.csv` | cross-script positives by source-pair × country × script-pair + translit agreement |
| `eda/14_character_statistics.csv` | per table × country char stats, script shares, trigram vocab (FULL `n`, SAMPLED stats) |
| `eda/15_token_assumption_audit.csv` | word/token-feature behavior by script bucket × class (suffix/stopword overlap measured) |
| `eda/16_char_ngram_comparison.csv` | word-token vs char-n-gram similarity by script bucket + hard/random negatives |
| `eda/figures/*.png` | audit, match counts, ECDFs, 2-D scatter, heatmaps, frontier, degrees, script/translit/ngram |
| `reports/eda_summary.md` | **Finding → Evidence → Implication** per section + next-stage decisions |

Open the casebook by double-clicking the HTML file (works offline — no CDNs).

### Multilingual / multi-script (§28)

Raw Unicode is always preserved; transliteration is an **additional feature only, never
canonical**. The pipeline measures script mix (11), character load (14), transliteration
recall-vs-risk (12/13), token-assumption behavior across scripts (15, no stopword/suffix
stripping defaults), and word-vs-char-n-gram tradeoffs (16) — then keeps every
representation side by side. Backend: `config/config.yaml: eda.multilingual`
(`auto` = Unidecode when installed, else deterministic builtin; `none` skips
transliteration). Smoke-test with non-Latin data:
`python scripts/make_synthetic_data.py --out dataset-dev --with-devanagari 60`.

### Scale & sampling (why EDA stays fast on 2.2M S1 / 7.6M pairs)

- **FULL-data:** audit (01), ground-truth counts (02, vectorized), raw + conservative
  collisions (05), country counts + missing rates (08), script distribution (11),
  character-stat denominators (14 `n` + non-ASCII rates).
- **SAMPLED (closed world, forced truth):** positive features (03), hard negatives (04),
  aggressive collisions (05 `aggr_*__sample*` rows), blocking (06/07), country string-stats
  + vocab coverage (08 `n_sampled`), graph (09), transliteration collisions (12),
  character stats (14 `n_sampled`), cross-script positives (13), token audit (15),
  char-n-gram comparison (16). Sampled pools always contain every anchor's
  true matches; every sampled artifact records its sample size.
- Sizes live in `config/config.yaml: eda.sampling`; `reports/eda_summary.md` has a
  **Sampling & scale** disclosure section. Quick smoke run:
  `python scripts/run_eda.py --data-root dataset --fast-mode` (~5x smaller samples).

## 8–10. Experiment logging workflow (3 people / 3 days / 5 submissions)

Full rules: [`logs/README.md`](logs/README.md). Short version:

```bat
python scripts/log_experiment.py                 :: interactive Q&A -> appends a row
python scripts/log_experiment.py --list 10       :: review recent history
python scripts/log_experiment.py --rerun EXP-003 --team-member Asha   :: rerun record
```

- **NEW experiment** (`new`): normalization/blocker/top-k/feature/model/
  threshold/negative-sampling/hyperparameter changed → new `EXP-###` row.
- **RERUN** (`rerun`): identical code/config/data executed again → new row with
  `experiment_type=rerun` + `parent_experiment_id` pointing at the original.
  The pipeline tells you when a run "looks IDENTICAL to the previous run".
- **SUBMISSION** (`submission`): leaderboard upload → row with
  `public_submission=yes`, `submission_number=1..5`, score filled in afterwards.
- **Never edit/delete rows.** History is append-only.

## 11. Intentionally NOT implemented yet (do not jump ahead)

- Final multi-pass blocking pipeline (only blocker *diagnostics* exist)
- Pair classifier (LightGBM/CatBoost), hard-negative training miner
- Entity-level thresholding / macro-F0.5 optimization
- Test inference, `output/matching_results.tsv`, `output/candidate_pairs.tsv`
- Submission validator + final packaging

`output/` stays empty until Stage 8/9.

## 12. Roadmap

1. ✅ **EDA (here)** — audit, positives vs hard negatives, collisions, blocking
   diagnostics, country shift, graph diagnostics, casebook
2. Normalization + feature engineering (from EDA findings)
3. Multi-pass blocking (recall ceiling + burden budget)
4. Training-pair construction / hard negatives
5. LightGBM/CatBoost pair classifier
6. Entity-level thresholding (singleton abstention)
7. Validation + macro-F0.5 optimization
8. Test inference
9. `candidate_pairs.tsv` + `matching_results.tsv`
10. Leaderboard experiment tracking (this log system already supports it)

Fair-play reminder: everything must be learned from the provided TSVs only —
no external business registries, geocoders, or web augmentation, ever.
