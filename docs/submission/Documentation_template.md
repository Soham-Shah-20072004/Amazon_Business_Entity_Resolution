# Methodology: Business Entity Resolution (Amazon ML Challenge 2026)

**Team:** `<team name>` · **Members:** `<names>`
**Final leaderboard submission:** `<public score>` · **Out-of-fold macro F0.5 (60k held-out train S1):** 0.9569
**Code:** `code/business_entity_resolution/` (reproduction steps in its `README.md`)

---

## 1. Summary

A cascade that looks at fewer pairs with more expensive evidence at each step:

1. **Normalisation**: several text views per record (Unicode NFKC with combining marks kept, transliteration to Latin, canonical abbreviations, core name, sound skeleton, numbers, postcodes).
2. **Blocking (candidate generation)**, per country and per target source:
   - approximate and exact nearest-neighbour search on character n-gram TF-IDF + LSA vectors (full record, address, sound skeleton, name);
   - a rare-token inverted index and exact core-name blocks;
   - a reverse search in which every S2/S3 record proposes its nearest S1s.
3. **Pre-ranker**: LightGBM on 24 cheap features trims ~164 blocked candidates per S1.
4. **Matcher**: LightGBM on ~80 string, token, number and context features.
5. **Cross-encoder**: a fine-tuned multilingual MiniLM reads each pair jointly. Its score, plus "sibling" features (similarity to the S1's other strong candidates), feeds a **stacked LightGBM**.
6. **Decision**: per-S1 expected-F0.5 rule with a one-owner constraint (each S2/S3 record goes to at most one S1). Empty lists are allowed, which matters for singletons.

## 2. Problem understanding and data facts

| Fact (from EDA on train) | Consequence for the design |
|---|---|
| Train: 2.21M S1, 5.03M S2, 5.29M S3; test: 1.73M S1, 4.89M S2, 5.08M S3 | No all-pairs comparison; blocking plus sampling for training, batched inference |
| 7.64M positive pairs; 5.6% of S1 are singletons, 48% have 4+ matches | No top-1 rule; each candidate judged on its own; empty prediction allowed |
| No true pair crosses countries | Blocking runs within each country; France (test only) is one more group |
| No S2/S3 record belongs to two S1s | One-owner rule and reverse-rank features |
| Names identical in 21% of true pairs, addresses in 8% | Fuzzy, token and character features rather than exact keys |
| ~6.5% of true pairs are cross-script (Devanagari, Gujarati, Malayalam, …) | Transliteration, a sound-skeleton view, a multilingual cross-encoder |
| Test has France (15%), which is absent from train | No country features; universal character features; per-country sanity checks |

**Metric:** macro F0.5 over S1. Precision weighs more, so one false match costs about 3× a missed match. A true singleton scores 1 only with an empty list.

## 3. Normalisation (`er/text.py`, `er/prepare.py`)

- **norm**: NFKC, lowercase, `&`→`and`, punctuation to spaces. Unicode combining marks are kept (dropping them split Indic words into letters).
- **ascii**: Unidecode transliteration with a fix for the Devanagari candra-O, plus digit/letter splitting (`12a` → `12 a`).
- **canon**: separate maps for names (pvt→private, ltd→limited, …) and addresses (rd→road; US and Indian state names → codes; city aliases such as Bombay→Mumbai).
- **core name**: the canonical name without legal tokens. Also kept: its acronym, a sound skeleton (inner vowels and *h* dropped, doubles collapsed), the address numbers (leading zeros stripped) and postcode-like tokens.
- Implementation: chunked TSV reading (400k rows), parallel normalisation, one parquet file per split. The raw text is never overwritten.

## 4. Blocking: candidate generation (`er/retrieval.py`)

Everything runs **per country × per target source** (S2 and S3 separately, so neither crowds out the other).

| Blocker | How it works | Settings |
|---|---|---|
| `ann_full` / `ann_addr` / `ann_skel` / `ann_name` | Character n-gram TF-IDF (char_wb 3–4; skeleton 2–4; 262k features; sublinear tf; vocabulary and IDF fitted on a 200k-row sample) → truncated SVD to 128 dims (LSA), L2-normalised, float16 memory-mapped → FAISS inner-product search | Top 25 per view and source. Exact brute-force search on GPU; IVF-Flat (nlist = 2√n, nprobe 32) as the CPU fallback |
| `rare` | Hashed name, address and postcode tokens (2²⁴ buckets). A token counts as rare if it appears in ≤ max(100, 300 per million) pool records. Score = sum of IDF of shared rare tokens, computed as a sparse matrix product | Top 20 per source |
| `exact` | Identical core name inside the country | Blocks with more than 50 records skipped |
| `rev` | Every S2/S3 record of the country searches all S1 of the country (full view); its 2 nearest S1 become candidates | Adds pairs that a crowded S1's own top-k misses |

- **Union**: about 164 candidates per S1. Every pair then gets the cosine of every view, the rare score, blocker flags, its rank and gap within the S1's list, and reverse-check features (is this S1 the record's best or second-best S1, and by how much).
- **Measured** on 60,000 train S1 against the full pool: union recall **94.25%** of true pairs; after the pre-ranker, 93.58%.
- **Pre-ranker (candidate pruning)**:
  - LightGBM on 24 cheap features (flags, cosines, rare score, ranks, gaps, reverse features), 3-fold out-of-fold.
  - Cut: top-k with p ≥ t, the smallest list within a 0.5% loss of the positives the union found (~18 per S1).
  - Pairs found only by the reverse search or the name view bypass the pre-ranker, because it was trained before those blockers existed.
- **Focus set**: pairs with matcher p ≥ 0.01, at most 8 per S1. This is the input of the final stacked matcher and the content of **`candidate_pairs.tsv`** (4.30 candidates per test S1).

## 5. Pair matching

**Matcher** (`er/features.py`, `er/model.py`): LightGBM binary (lr 0.05, 127 leaves, 5-fold out-of-fold grouped by S1, early stopping). About 80 features:
- the cheap set;
- rapidfuzz ratio, partial, token-sort, token-set and Jaro-Winkler on name, core name, skeleton and address;
- IDF-weighted token overlap;
- house-number and postcode agreement / conflict / missing / dropped-digit;
- legal-suffix mismatch, acronym match, name contained in the address, lengths;
- rank and gap of the name/address similarity within the S1's list.

Negatives are the blocker's own wrong candidates, i.e. hard negatives.

**Cross-encoder** (`scripts/gpu/cross_encoder.py`):
- Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 1-logit head.
- Input: `[CLS] name | address [SEP] name | address [SEP]` in the original script, max 96 tokens.
- Training: binary cross-entropy, AdamW 3e-5, 2 epochs, fp16, on the focus set (229k train pairs).
- Two folds by S1 give out-of-fold scores; test pairs get the mean of both models. Out-of-fold AUC 0.980.
- Pairs added later by retrieval v2 are scored by one model fine-tuned on all train pairs.

**Sibling features** (`features.sibling_features`): similarity (token-set on name, address and skeleton) of each candidate to the S1's three strongest other candidates, weighted by their probability, plus the number of confident siblings. A business's records often resemble each other more than they resemble the S1.

**Stacked matcher** (`scripts/run_combine.py`): LightGBM on the matcher features plus the cross-encoder score and its rank and gap, plus the sibling features. 5-fold out-of-fold.

## 6. Decision rule (`er/evaluate.py`)

- **One owner**: if several S1 claim a record, only the highest-probability S1 keeps it.
- **Per-S1 expected F0.5**: for each S1, keep the top-k candidates maximising 1.25·Σp₁..ₖ / (0.25·(Σp + λ) + k), or keep nothing when P(no match) = Π(1−p)·e^(−λ) is higher. λ and a calibration power are tuned on out-of-fold scores (final: λ = 0.35, power 1.25). A global threshold was also evaluated; the better rule on out-of-fold data is used.

## 7. Validation

- Training and every report use random train S1s searched against the **complete** S2/S3 pool, so the density of look-alikes matches test.
- Folds are grouped by S1 (a hash of the row id): an S1 is always scored by a model that never saw it.
- The exact competition metric is computed over all sampled S1, singletons included.
- **Error funnel** (`scripts/diagnose.py`), out-of-fold before stacking. Of 4.73 points lost:
  - 2.28 from true pairs that never became candidates;
  - 1.50 from true candidates scored below the rule;
  - 0.95 from false matches.
- France has no labels; it was checked without labels per country (`scripts/inspect_country.py`).

## 8. Results

| Version | Out-of-fold macro F0.5 | Public leaderboard | Candidates / S1 |
|---|---|---|---|
| Matcher + threshold (submission 1) | 0.9527 | 0.935 | 18.1 |
| + cross-encoder, sibling features, expected-F0.5 rule | 0.9569 | `<lb v2>` | 4.16 |
| + retrieval v2 (exact GPU search, reverse search, name view) | 0.9569* | `<lb v3>` | 4.30 |

\*Training pairs unchanged, so the out-of-fold score cannot show the retrieval gain on test.

## 9. Compute and runtime

Kaggle notebooks (4 vCPU / ~30 GB RAM; 2 × T4 for GPU steps):

| Step | Time |
|---|---|
| Preparation | ~25 min |
| Stage B (CPU) | ~8 h |
| Cross-encoder | ~1 h 45 m |
| Retrieval v2 + new-pair scoring + stacking | ~3 h 30 m |

Scale tricks:
- the TF-IDF vocabulary is fitted on a sample and rows are transformed in parallel chunks, never holding the full sparse matrix;
- vectors are stored as float16 memory maps;
- indexes are partitioned by country and source;
- cosines are computed in threads;
- grouped ranks use Polars;
- intermediate tables are cached so later stages rerun in minutes.

## 10. Models, libraries and licences

| Component | Licence | Size |
|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 (only pretrained model) | Apache-2.0 | 118M parameters (< 8B) |
| LightGBM | MIT | — |
| FAISS | MIT | — |
| rapidfuzz | MIT | — |
| polars | MIT | — |
| PyTorch | BSD-3 | — |
| transformers | Apache-2.0 | — |
| pyarrow | Apache-2.0 | — |
| scikit-learn, numpy, pandas, scipy | BSD | — |
| Unidecode (transliteration library, not a model) | GPL-2.0+ | — |

## 11. Fair play and external data

- No external datasets, APIs, geocoding or lookups.
- Only the provided train and test files are used, plus the pretrained weights listed above.
- Test labels are never used. Test text is used only unsupervised, as model input.
- Abbreviation, state-code and city-alias tables are hand-written domain knowledge kept in `er/text.py`.

## 12. What we learned and what we would do next

- **Retrieval recall was the ceiling.** 94% recall cost ~2.3 points; missed rows were mostly other-script records, records with empty addresses, and crowded look-alikes.
- **Built but not part of the submitted result:**
  - a fine-tuned bi-encoder retriever with sibling propagation and a gated merge (`scripts/gpu/biencoder.py`, `scripts/merge_bienc.py`);
  - a full retrain with the new blockers.
- **Next steps:**
  - retrain with the new blockers and more training S1;
  - learned (bi-encoder) retrieval as an additional view;
  - transliteration for more Indic scripts;
  - France-specific address handling (generic names, house numbers).
