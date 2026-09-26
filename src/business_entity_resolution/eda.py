"""Stage 1 EDA: dataset audit + entity-resolution diagnostics.

Every function below produces CSV artifacts under ``eda/``, figures under
``eda/figures/``, and returns a JSON-serializable ``results`` dict that feeds
``reports/eda_summary.md`` (Finding -> Evidence -> Implication).

Scale rework (7.6M pairs must never materialize):
  FULL-data: audit, ground-truth counts (vectorized), raw + conservative
    collisions, country counts / missing rates, script distribution (11),
    character-stat denominators (14).
  SAMPLED: positive features, hard negatives, blocking, graph, vocab overlap,
    aggressive-norm collisions, transliteration collisions (12), character
    stats (14), cross-script positives (13), token audit (15), char-n-gram
    comparison (16). Every sampled artifact records its sample size;
    ``reports/eda_summary.md`` discloses full vs sampled.

Sections:
    1. run_dataset_audit                 -> 01_data_audit.csv + audit figures
    2. run_ground_truth_eda              -> 02_ground_truth_match_distribution.csv
    3. sampled positives / negatives     -> 03_*, 04_* + similarity figures
    4. run_normalization_collision_eda   -> 05_normalization_collision_report.csv
    5. run_blocking_eda (sampled)        -> 06_*, 07_* + recall-vs-burden figure
    6. run_country_shift_eda (sampled)   -> 08_country_shift_report.csv
    7. run_graph_diagnostics (sampled)   -> 09_candidate_graph_diagnostics.csv
    8. build_casebook                    -> 10_casebook_train_pairs.html
    9. multilingual EDA (§28)            -> 11_script_distribution.csv,
                                            12_transliteration_collision_report.csv,
                                            13_cross_script_positive_pairs.csv,
                                            14_character_statistics.csv,
                                            15_token_assumption_audit.csv,
                                            16_char_ngram_comparison.csv
                                            (+ 4 figures; FULL: 11 + 14 denominators)
   10. generate_summary_md               -> reports/eda_summary.md

Determinism: all sampling uses ``np.random.RandomState(cfg seed)``.
Offline: matplotlib Agg backend (headless-safe); seaborn optional.
"""

from __future__ import annotations

import gc
import html
import logging
import math
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Collection, Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import AppConfig
from .features import (
    FEATURE_COLUMNS,
    add_pair_features,
    fuzz_ratio,
    summarize_feature_frame,
)
from .io import anchor_truth_to_basic, ground_truth_stats, parse_anchor_truth
from .normalization import (
    BUSINESS_TYPE_TOKENS,
    EN_STOPWORDS_MEASURE,
    INDIC_SUFFIX_TOKENS,
    LEGAL_SUFFIX_TOKENS,
    SCRIPT_CATEGORIES,
    SCRIPT_REGEX,
    char_ngram_jaccard,
    char_ngram_set,
    detect_script_category,
    extract_numeric_tokens,
    jaccard_similarity,
    normalize_aggressive,
    normalize_basic,
    pair_script_bucket,
    resolve_transliteration_backend,
    string_stats,
    tokenize,
    transliterate_text,
    word_jaccard,
)

logger = logging.getLogger("ber")

try:
    import seaborn as sns

    _HAS_SEABORN = True
except Exception:  # pragma: no cover
    _HAS_SEABORN = False

try:
    from scipy.sparse import coo_matrix, csr_matrix
    from scipy.sparse.csgraph import connected_components

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _apply_style(cfg: AppConfig) -> None:
    try:
        plt.style.use(cfg.eda.get("figures", {}).get("style", "default"))
    except Exception:
        plt.style.use("default")


def _dpi(cfg: AppConfig) -> int:
    try:
        return int(cfg.eda.get("figures", {}).get("dpi", 150))
    except Exception:
        return 150


def _savefig(fig: plt.Figure, path: str | Path, cfg: AppConfig) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=_dpi(cfg))
    plt.close(fig)
    return str(path)


def _rng(cfg: AppConfig, salt: int = 0) -> np.random.RandomState:
    base = int(cfg.eda.get("random_seed", cfg.seed))
    return np.random.RandomState((base + salt) % (2**31 - 1))


def _ecdf(vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.array([]), np.array([])
    x = np.sort(vals)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def _pct(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


# ---------------------------------------------------------------------------
# normalization columns (raw is never overwritten)
# ---------------------------------------------------------------------------

def ensure_normalized_columns(df: pd.DataFrame, cols: Dict[str, str]) -> pd.DataFrame:
    """Return a copy with ``name_norm/address_norm`` (+aggressive) columns added."""
    out = df.copy()
    c_name, c_addr = cols["business_name"], cols["business_address"]
    if c_name in out.columns:
        out["name_norm"] = out[c_name].astype(str).map(normalize_basic)
        out["name_aggr"] = out[c_name].astype(str).map(normalize_aggressive)
    else:
        out["name_norm"] = ""
        out["name_aggr"] = ""
    if c_addr in out.columns:
        out["address_norm"] = out[c_addr].astype(str).map(normalize_basic)
        out["address_aggr"] = out[c_addr].astype(str).map(normalize_aggressive)
    else:
        out["address_norm"] = ""
        out["address_aggr"] = ""
    out["name_address_norm"] = out["name_norm"] + " || " + out["address_norm"]
    return out


# ---------------------------------------------------------------------------
# sampling helpers (scale rework: small lookups only, forced truth)
# ---------------------------------------------------------------------------

def _sampling_cfg(cfg: AppConfig) -> Dict[str, Any]:
    return dict(cfg.eda.get("sampling", {}) or {})


def _stable_salt(cfg: AppConfig, key: str, base: int = 0) -> int:
    """Deterministic salt per group key (crc32, immune to hash randomization)."""
    digest = zlib.crc32(str(key).encode("utf-8")) % 100000
    return int(base + digest)


def _sample_ids(ids: List[str], n: int, rng: np.random.RandomState) -> List[str]:
    """Deterministic sample of ids (sorted output)."""
    if len(ids) <= n:
        return sorted(ids)
    return sorted(rng.choice(ids, int(n), replace=False).tolist())


def _filter_raw_to_ids(
    df: Optional[pd.DataFrame], ids: Collection[str], id_col: str
) -> pd.DataFrame:
    """Filter a raw table to ``ids`` (vectorized isin, sorted by id)."""
    if df is None or df.empty or id_col not in df.columns or not ids:
        return pd.DataFrame(columns=list(df.columns) if df is not None else [id_col])
    want = {str(x) for x in ids}
    try:
        mask = df[id_col].astype(str).isin(want)
    except Exception:
        return df.iloc[0:0].copy()
    out = df.loc[mask].copy()
    if not out.empty:
        out = out.sort_values(id_col).reset_index(drop=True)
    return out


def _sample_pool_with_forced_truth(
    s2_raw: Optional[pd.DataFrame],
    s3_raw: Optional[pd.DataFrame],
    forced_ids: Collection[str],
    n_sample: int,
    rng: np.random.RandomState,
    id_col: str,
) -> pd.DataFrame:
    """Sample ``n_sample`` pool rows + FORCE every true match id.

    Sampling is proportional to table sizes; forced ids are fetched via
    vectorized ``isin`` scans (no 10M-row dict/set of the full pool). Output is
    deduplicated and sorted by id for determinism.
    """
    forced_set = {str(x) for x in (forced_ids or []) if str(x)}
    parts_raw: List[pd.DataFrame] = []
    for df in (s2_raw, s3_raw):
        if df is not None and not df.empty and id_col in df.columns:
            parts_raw.append(df)
    if not parts_raw:
        return pd.DataFrame(columns=[id_col])
    total = sum(len(df) for df in parts_raw)
    if total <= int(n_sample):
        combined = pd.concat(parts_raw, ignore_index=True)
        if id_col in combined.columns:
            combined = combined.drop_duplicates(subset=[id_col], keep="first")
            combined = combined.sort_values(id_col).reset_index(drop=True)
        return combined
    # proportional sampling without ever concatenating the full 10M pool
    sampled_parts: List[pd.DataFrame] = []
    remaining = int(n_sample)
    for i, df in enumerate(parts_raw):
        if i == len(parts_raw) - 1:
            n_take = remaining
        else:
            n_take = int(int(n_sample) * len(df) / total)
            remaining -= n_take
        n_take = max(0, min(n_take, len(df)))
        if n_take <= 0:
            continue
        idx = rng.choice(len(df), n_take, replace=False)
        sampled_parts.append(df.iloc[sorted(idx.tolist())].copy())
    sampled = (
        pd.concat(sampled_parts, ignore_index=True)
        if sampled_parts
        else pd.DataFrame(columns=list(parts_raw[0].columns))
    )
    if forced_set and not sampled.empty:
        try:
            have = set(sampled[id_col].astype(str).tolist())
        except Exception:
            have = set()
        missing = sorted(forced_set - have)
        if missing:
            missing_set = set(missing)
            extra: List[pd.DataFrame] = []
            for df in parts_raw:
                try:
                    mask = df[id_col].astype(str).isin(missing_set)
                except Exception:
                    continue
                hit = df.loc[mask].copy()
                if not hit.empty:
                    extra.append(hit)
            if extra:
                sampled = pd.concat([sampled] + extra, ignore_index=True)
                sampled = sampled.drop_duplicates(subset=[id_col], keep="first")
    elif forced_set and sampled.empty:
        extra = []
        missing_set = set(forced_set)
        for df in parts_raw:
            try:
                mask = df[id_col].astype(str).isin(missing_set)
            except Exception:
                continue
            hit = df.loc[mask].copy()
            if not hit.empty:
                extra.append(hit)
        if extra:
            sampled = pd.concat(extra, ignore_index=True).drop_duplicates(
                subset=[id_col], keep="first"
            )
    if not sampled.empty and id_col in sampled.columns:
        sampled = sampled.sort_values(id_col).reset_index(drop=True)
    return sampled


# ===========================================================================
# 1. DATASET AUDIT -> 01_data_audit.csv
# ===========================================================================

def _audit_one_table(
    df: pd.DataFrame, table: str, cols: Dict[str, str]
) -> List[Dict[str, object]]:
    c_id, c_name, c_addr, c_cty = (
        cols["entity_id"], cols["business_name"], cols["business_address"], cols["country"]
    )
    rows: List[Dict[str, object]] = []
    n = len(df)

    def add(metric: str, value: object) -> None:
        rows.append({"table": table, "metric": metric, "value": value})

    add("rows", n)
    if c_id in df.columns:
        ids = df[c_id].astype(str)
        add("unique_entity_ids", int(ids.nunique()))
        add("duplicate_entity_id_rows", int(ids.duplicated(keep=False).sum()))
    if c_cty in df.columns:
        countries = df[c_cty].astype(str)
        add("n_countries_distinct_raw", int(countries.nunique()))
        for country, cnt in countries.value_counts(dropna=False).items():
            add(f"country_count::{country}", int(cnt))
            add(f"country_pct::{country}", round(_pct(cnt, n), 6))
        add("missing_country_n", int((countries.str.strip() == "").sum()))
        add("missing_country_rate", round(float((countries.str.strip() == "").mean()) if n else 0.0, 6))
    for col, label in ((c_name, "name"), (c_addr, "address")):
        if col not in df.columns:
            add(f"missing_{label}_n", "NA-no-column")
            continue
        vals = df[col].astype(str)
        empty = vals.str.strip() == ""
        add(f"missing_{label}_n", int(empty.sum()))
        add(f"missing_{label}_rate", round(float(empty.mean()) if n else 0.0, 6))
        add(f"empty_string_{label}_n", int((vals == "").sum()))
        lens = vals.str.len().astype(float)
        add(f"{label}_len_mean", round(float(lens.mean()) if n else 0.0, 3))
        for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.99):
            tag = f"p{int(q * 100)}"
            if tag == "p50":
                tag = "median"
            add(f"{label}_len_{tag}", round(float(lens.quantile(q)) if n else 0.0, 3))
        add(f"{label}_len_max", int(lens.max()) if n else 0)
    return rows


def run_dataset_audit(
    tables: Dict[str, Optional[pd.DataFrame]],
    cfg: AppConfig,
) -> Dict[str, Any]:
    """Audit every available table; write 01_data_audit.csv + audit figures."""
    _apply_style(cfg)
    cols = cfg.columns
    labels = {
        "train_s1": "train_source1", "train_s2": "train_source2",
        "train_s3": "train_source3", "test_s1": "test_source1",
        "test_s2": "test_source2", "test_s3": "test_source3",
    }
    all_rows: List[Dict[str, object]] = []
    present: Dict[str, pd.DataFrame] = {}
    for key, df in tables.items():
        if df is None:
            all_rows.append({"table": labels.get(key, key), "metric": "rows", "value": "NA-missing-file"})
            continue
        present[key] = df
        all_rows += _audit_one_table(df, labels.get(key, key), cols)

    audit = pd.DataFrame(all_rows, columns=["table", "metric", "value"])
    out_csv = cfg.eda_dir / "01_data_audit.csv"
    audit.to_csv(out_csv, index=False)

    artifacts = [str(out_csv)]
    # -- figures --
    if present:
        order = [k for k in labels if k in present]
        # rows per source
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.bar([labels[k] for k in order], [len(present[k]) for k in order])
        ax.set_title("Rows per source table")
        ax.set_ylabel("rows")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        artifacts.append(_savefig(fig, cfg.figures_dir / "rows_per_source.png", cfg))

        # missingness
        c_name, c_addr, c_cty = cols["business_name"], cols["business_address"], cols["country"]
        miss = pd.DataFrame(
            [
                {
                    "table": labels[k],
                    "name": float((present[k][c_name].astype(str).str.strip() == "").mean()) if c_name in present[k].columns else 0.0,
                    "address": float((present[k][c_addr].astype(str).str.strip() == "").mean()) if c_addr in present[k].columns else 0.0,
                    "country": float((present[k][c_cty].astype(str).str.strip() == "").mean()) if c_cty in present[k].columns else 0.0,
                }
                for k in order
            ]
        ).set_index("table")
        fig, ax = plt.subplots(figsize=(8, 4.2))
        miss.plot(kind="bar", ax=ax)
        ax.set_title("Missing-value rate per field (blank after strip)")
        ax.set_ylabel("rate")
        ax.set_ylim(0, max(0.05, float(miss.values.max()) * 1.25 if miss.size else 0.05))
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        artifacts.append(_savefig(fig, cfg.figures_dir / "missingness.png", cfg))

        # lengths
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
        rng = _rng(cfg, salt=11)
        for ax, col, title in zip(axes, (c_name, c_addr), ("Business-name length", "Business-address length")):
            data, tick = [], []
            for k in order:
                if col not in present[k].columns:
                    continue
                lens = present[k][col].astype(str).str.len().to_numpy(dtype=float)
                if len(lens) > 20000:
                    lens = rng.choice(lens, 20000, replace=False)
                data.append(lens)
                tick.append(labels[k])
            ax.boxplot(data, labels=tick, showfliers=False)
            ax.set_title(title)
            ax.set_ylabel("characters")
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        artifacts.append(_savefig(fig, cfg.figures_dir / "name_address_lengths.png", cfg))

    results: Dict[str, Any] = {
        "audit_csv": str(out_csv),
        "artifacts": artifacts,
        "tables_present": sorted(present.keys()),
        "n_rows": {labels.get(k, k): int(len(v)) for k, v in present.items()},
    }
    return results


# ===========================================================================
# 2. GROUND-TRUTH EDA -> 02_ground_truth_match_distribution.csv
# ===========================================================================

def _bucket_total(n: int) -> str:
    if n == 0:
        return "0 matches"
    if n == 1:
        return "1 match"
    if n == 2:
        return "2 matches"
    if n == 3:
        return "3 matches"
    return "4+ matches"


def _bucket_series(n: pd.Series) -> pd.Series:
    """Vectorized _bucket_total (C-speed select, no per-row Python)."""
    arr = pd.to_numeric(n, errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    out = np.select(
        [arr == 0, arr == 1, arr == 2, arr == 3],
        ["0 matches", "1 match", "2 matches", "3 matches"],
        default="4+ matches",
    )
    return pd.Series(out, index=n.index)


def run_ground_truth_eda(
    gt_df: Optional[pd.DataFrame],
    cfg: AppConfig,
) -> Dict[str, Any]:
    """Match-count distributions per S1 (FULL data, vectorized, no pair expansion)."""
    _apply_style(cfg)
    out_csv = cfg.eda_dir / "02_ground_truth_match_distribution.csv"
    artifacts = [str(out_csv)]
    if gt_df is None or gt_df.empty:
        pd.DataFrame(columns=["breakdown", "category", "n_s1", "pct_of_s1"]).to_csv(out_csv, index=False)
        logger.warning("Ground truth missing — wrote empty 02 CSV.")
        empty_stats, _ = ground_truth_stats(None, cfg.columns)
        return {"gt_csv": str(out_csv), "artifacts": artifacts, "skipped": True,
                "gt_stats": empty_stats, "n_s1_in_gt": 0, "n_positive_pairs": 0}

    gt_stats, summary = ground_truth_stats(gt_df, cfg.columns)
    n_s1 = int(summary["n_s1_in_gt"])
    rows: List[Dict[str, object]] = []

    def add_block(breakdown: str, series: pd.Series) -> None:
        vc = series.value_counts()
        for cat, cnt in vc.items():
            rows.append({"breakdown": breakdown, "category": str(cat),
                         "n_s1": int(cnt), "pct_of_s1": round(_pct(cnt, n_s1), 6)})

    add_block("total_matches", _bucket_series(gt_stats["n_matches"]))
    add_block("s2_matches", _bucket_series(gt_stats["n_s2_matches"]))
    add_block("s3_matches", _bucket_series(gt_stats["n_s3_matches"]))
    pattern = pd.Series(
        np.select(
            [gt_stats["is_singleton"] == 1, gt_stats["is_s2_only"] == 1,
             gt_stats["is_s3_only"] == 1, gt_stats["is_mixed"] == 1],
            ["singleton (no match)", "S2-only", "S3-only", "mixed S2+S3"],
            default="other",
        ),
        index=gt_stats.index,
    )
    add_block("match_pattern", pattern)

    dist = pd.DataFrame(rows, columns=["breakdown", "category", "n_s1", "pct_of_s1"])
    dist.to_csv(out_csv, index=False)

    # figure: total match counts + pattern side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    order = ["0 matches", "1 match", "2 matches", "3 matches", "4+ matches"]
    tot = dist[dist.breakdown == "total_matches"].set_index("category").reindex(order).fillna(0)
    axes[0].bar(tot.index, tot["n_s1"].to_numpy(dtype=float))
    axes[0].set_title("Matches per Source-1 entity (total)")
    axes[0].set_ylabel("# S1 entities")
    plt.setp(axes[0].get_xticklabels(), rotation=20, ha="right")
    pat = dist[dist.breakdown == "match_pattern"].set_index("category")["n_s1"].sort_values(ascending=False)
    axes[1].bar(pat.index, pat.to_numpy(dtype=float))
    axes[1].set_title("Match pattern per Source-1 entity")
    plt.setp(axes[1].get_xticklabels(), rotation=20, ha="right")
    artifacts.append(_savefig(fig, cfg.figures_dir / "match_count_distribution.png", cfg))

    results = {
        "gt_csv": str(out_csv),
        "artifacts": artifacts,
        "skipped": False,
        "n_s1_in_gt": int(summary["n_s1_in_gt"]),
        "n_positive_pairs": int(summary["n_positive_pairs"]),
        "n_s1_s2": int(summary["n_s1_s2"]),
        "n_s1_s3": int(summary["n_s1_s3"]),
        "singleton_rate": round(float(summary["singleton_rate"]), 6),
        "multi_match_rate": round(float(summary["multi_match_rate"]), 6),
        "pattern_counts": dict(summary["pattern_counts"]),
        "total_counts": dict(summary["total_counts"]),
        "gt_stats": gt_stats,  # internal reuse for sampled sections (popped before return)
    }
    return results


# ===========================================================================
# 5. NORMALIZATION COLLISION EDA -> 05_normalization_collision_report.csv
# ===========================================================================

def _collision_stats(values: pd.Series) -> Dict[str, float]:
    n = len(values)
    n_unique = int(values.nunique())
    vc = values.value_counts()
    groups = vc[vc > 1]
    return {
        "n_records": float(n),
        "n_unique": float(n_unique),
        "pct_unique": round(_pct(n_unique, n), 6),
        "n_collision_groups": float(len(groups)),
        "largest_cluster": float(int(groups.max())) if len(groups) else 0.0,
        "mean_cluster_size": round(float(groups.mean()), 3) if len(groups) else 0.0,
    }


def _aggressive_sample_label(n_config: int) -> str:
    if n_config >= 1000000 and n_config % 1000000 == 0:
        return f"__sample{n_config // 1000000}M"
    if n_config >= 1000 and n_config % 1000 == 0:
        return f"__sample{n_config // 1000}k"
    return f"__sample{n_config}"


def run_normalization_collision_eda(
    tables: Dict[str, Optional[pd.DataFrame]],
    cfg: AppConfig,
) -> Dict[str, Any]:
    """Compare uniqueness: FULL raw+conservative, SAMPLED aggressive.

    Aggressive normalization is computed on ``collision_aggressive_sample``
    rows per table only (1M default) and labeled
    ``aggr_name__sample1M`` / ``aggr_address__sample1M`` so the CSV discloses
    sampling via both the representation name and ``n_records``.
    """
    cols = cfg.columns
    c_name, c_addr, c_cty = cols["business_name"], cols["business_address"], cols["country"]
    labels = {
        "train_s1": "train_source1", "train_s2": "train_source2",
        "train_s3": "train_source3", "test_s1": "test_source1",
        "test_s2": "test_source2", "test_s3": "test_source3",
    }
    aggr_n = int(_sampling_cfg(cfg).get("collision_aggressive_sample", 1000000))
    aggr_suffix = _aggressive_sample_label(aggr_n)
    rows: List[Dict[str, object]] = []
    largest_examples: Dict[str, Any] = {}
    cluster_sizes: Dict[str, Dict[str, np.ndarray]] = {}
    aggressive_actual: Dict[str, int] = {}
    order_keys = [k for k in ("train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3")
                  if k in tables]
    for table_idx, key in enumerate(order_keys):
        df = tables.get(key)
        if df is None:
            continue
        if c_name not in df.columns or c_addr not in df.columns:
            logger.warning("Collision EDA skipping %s (missing name/address columns).", key)
            continue
        n = len(df)
        raw_name = df[c_name].astype(str)
        raw_addr = df[c_addr].astype(str)

        def _record(rep_name: str, vals: pd.Series) -> None:
            st = _collision_stats(vals)
            rows.append({"table": labels[key], "representation": rep_name, **st})
            vc = vals.value_counts()
            groups = vc[vc > 1]
            cluster_sizes.setdefault(key, {})[rep_name] = (
                groups.to_numpy(dtype=float) if len(groups) else np.array([])
            )
            if len(groups):
                largest_examples.setdefault(labels[key], {})[rep_name] = {
                    "value_preview": str(groups.index[0])[:120],
                    "size": int(groups.iloc[0]),
                }

        # FULL-data raw + conservative (sequential to bound memory)
        _record("raw_name", raw_name)
        name_norm = raw_name.map(normalize_basic)
        _record("norm_name", name_norm)
        _record("raw_address", raw_addr)
        addr_norm = raw_addr.map(normalize_basic)
        _record("norm_address", addr_norm)
        _record("raw_name_address", raw_name + " || " + raw_addr)
        name_addr_norm = name_norm + " || " + addr_norm
        _record("norm_name_address", name_addr_norm)
        if c_cty in df.columns:
            cty_norm = df[c_cty].astype(str).str.strip().str.lower()
            _record("norm_name_address_country", name_addr_norm + " || " + cty_norm)
            del cty_norm
        else:
            _record("norm_name_address_country", name_addr_norm)
        del name_addr_norm
        # SAMPLED aggressive (bounded to aggr_n rows)
        rng = _rng(cfg, salt=601 + table_idx)
        if n > aggr_n:
            idx = sorted(rng.choice(n, aggr_n, replace=False).tolist())
            samp_name = raw_name.iloc[idx]
            samp_addr = raw_addr.iloc[idx]
        else:
            samp_name, samp_addr = raw_name, raw_addr
        aggressive_actual[labels[key]] = int(len(samp_name))
        _record(f"aggr_name{aggr_suffix}", samp_name.map(normalize_aggressive))
        _record(f"aggr_address{aggr_suffix}", samp_addr.map(normalize_aggressive))
        del raw_name, raw_addr, name_norm, addr_norm, samp_name, samp_addr
        gc.collect()

    report = pd.DataFrame(
        rows,
        columns=["table", "representation", "n_records", "n_unique", "pct_unique",
                 "n_collision_groups", "largest_cluster", "mean_cluster_size"],
    )
    out_csv = cfg.eda_dir / "05_normalization_collision_report.csv"
    report.to_csv(out_csv, index=False)
    artifacts = [str(out_csv)]

    # figure: collision-group size distributions (log scale) for key representations
    _apply_style(cfg)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, rep in zip(axes, ("norm_name", "norm_address", "norm_name_address")):
        plotted = False
        for key in ("train_s1", "train_s2", "train_s3"):
            sizes = cluster_sizes.get(key, {}).get(rep, np.array([]))
            sizes = sizes[sizes > 1]
            if len(sizes):
                x, y = _ecdf(np.log10(sizes))
                ax.plot(x, y, label=f"{labels[key]} (n={len(sizes)})")
                plotted = True
        ax.set_title(f"Collision sizes: {rep}")
        ax.set_xlabel("log10(cluster size)")
        if not plotted:
            ax.text(0.5, 0.5, "no collisions", ha="center", transform=ax.transAxes)
    axes[0].set_ylabel("ECDF")
    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    artifacts.append(_savefig(fig, cfg.figures_dir / "normalization_collision_sizes.png", cfg))

    return {
        "collision_csv": str(out_csv),
        "artifacts": artifacts,
        "largest_examples": largest_examples,
        "aggressive_sample_config": int(aggr_n),
        "aggressive_sample_label": aggr_suffix,
        "aggressive_actual": dict(aggressive_actual),
        "train_norm_name_groups": {
            t: float(report[(report.table == t) & (report.representation == "norm_name")]["n_collision_groups"].sum())
            for t in ("train_source1", "train_source2", "train_source3")
        },
    }


# ===========================================================================
# 3. SAMPLED POSITIVE PAIRS + 4. HARD NEGATIVES -> 03_*, 04_* + figures
# ===========================================================================

def build_sampled_positive_pairs(
    s1_raw: Optional[pd.DataFrame],
    s2_raw: Optional[pd.DataFrame],
    s3_raw: Optional[pd.DataFrame],
    gt_df: Optional[pd.DataFrame],
    gt_stats: Optional[pd.DataFrame],
    cfg: AppConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Sample S1 anchors, expand ONLY their truth, and featurize (bounded).

    Returns (positives_basic, positives_feat, info). Normalization touches
    only the small anchor/candidate lookups; the global 7.6M pair table is
    never built. Output is sorted by (s1, candidate) for determinism.
    """
    cols = cfg.columns
    c_id = cols["entity_id"]
    sampling = _sampling_cfg(cfg)
    n_positive_s1 = int(sampling.get("n_positive_s1", 10000))
    max_positive_pairs = int(sampling.get("max_positive_pairs", 30000))
    rng = _rng(cfg, salt=301)
    empty_basic = pd.DataFrame(
        columns=["source1_entity_id", "candidate_entity_id", "source_pair"]
    )
    if gt_df is None or gt_df.empty or s1_raw is None or s1_raw.empty:
        return empty_basic, pd.DataFrame(), {
            "n_anchors_sampled": 0, "n_positive_rows": 0,
            "n_dangling_skipped": 0, "n_s1_lookup": 0, "n_pool_lookup": 0,
        }
    if gt_stats is not None and not gt_stats.empty and cols["gt_source1"] in gt_stats.columns:
        matched = gt_stats.loc[gt_stats["n_matches"] > 0, cols["gt_source1"]].astype(str).tolist()
    else:
        matched = []
    anchor_ids = _sample_ids(sorted({str(x) for x in matched}), n_positive_s1, rng)
    anchor_truth = parse_anchor_truth(gt_df, anchor_ids, cols)
    positives_basic = anchor_truth_to_basic(anchor_truth)
    n_other = 0
    if not positives_basic.empty:
        before = len(positives_basic)
        positives_basic = positives_basic[
            positives_basic["source_pair"].isin(["S1_S2", "S1_S3"])
        ].reset_index(drop=True)
        n_other = int(before - len(positives_basic))
    if len(positives_basic) > max_positive_pairs:
        idx = sorted(rng.choice(len(positives_basic), max_positive_pairs, replace=False).tolist())
        positives_basic = positives_basic.iloc[idx].sort_values(
            ["source1_entity_id", "candidate_entity_id"]).reset_index(drop=True)
    elif not positives_basic.empty:
        positives_basic = positives_basic.sort_values(
            ["source1_entity_id", "candidate_entity_id"]).reset_index(drop=True)
    if positives_basic.empty:
        return positives_basic, pd.DataFrame(), {
            "n_anchors_sampled": int(len(anchor_ids)),
            "n_positive_rows": 0, "n_dangling_skipped": 0,
            "n_other_dropped": int(n_other), "n_s1_lookup": 0, "n_pool_lookup": 0,
        }
    needed_s1 = sorted(set(positives_basic["source1_entity_id"].astype(str).tolist()))
    needed_cand = sorted(set(positives_basic["candidate_entity_id"].astype(str).tolist()))
    want_s1, want_cand = set(needed_s1), set(needed_cand)
    s1_small_raw = _filter_raw_to_ids(s1_raw, want_s1, c_id)
    pool_parts: List[pd.DataFrame] = []
    for part in (s2_raw, s3_raw):
        f = _filter_raw_to_ids(part, want_cand, c_id)
        if not f.empty:
            pool_parts.append(f)
    pool_small_raw = (
        pd.concat(pool_parts, ignore_index=True).drop_duplicates(subset=[c_id])
        .sort_values(c_id).reset_index(drop=True)
        if pool_parts else pd.DataFrame(columns=list(s1_raw.columns))
    )
    found_cand = set(pool_small_raw[c_id].astype(str).tolist()) if not pool_small_raw.empty else set()
    found_s1 = set(s1_small_raw[c_id].astype(str).tolist()) if not s1_small_raw.empty else set()
    before = len(positives_basic)
    positives_basic = positives_basic[
        positives_basic["source1_entity_id"].astype(str).isin(found_s1)
        & positives_basic["candidate_entity_id"].astype(str).isin(found_cand)
    ].reset_index(drop=True)
    n_dangling = int(before - len(positives_basic))
    s1_small = ensure_normalized_columns(s1_small_raw, cols) if not s1_small_raw.empty else s1_small_raw
    pool_small = ensure_normalized_columns(pool_small_raw, cols) if not pool_small_raw.empty else pool_small_raw
    positives_feat = add_pair_features(positives_basic, s1_small, pool_small, cols, label=1, neg_type="")
    info = {
        "n_anchors_sampled": int(len(anchor_ids)),
        "n_anchors_with_pairs": int(len(needed_s1)),
        "n_positive_rows": int(len(positives_basic)),
        "n_dangling_skipped": int(n_dangling),
        "n_other_dropped": int(n_other),
        "n_s1_lookup": int(len(s1_small)),
        "n_pool_lookup": int(len(pool_small)),
    }
    del s1_small, pool_small, s1_small_raw, pool_small_raw
    gc.collect()
    return positives_basic, positives_feat, info


# ---------------------------------------------------------------------------
# char TF-IDF retrieval index (shared by hard negatives + blocking diagnostics)
# ---------------------------------------------------------------------------

def build_tfidf_index(
    texts: List[str], cfg: AppConfig
) -> Tuple[Optional[Any], Optional[Any]]:
    """Fit a char TF-IDF index; returns (vectorizer, matrix) or (None, None)."""
    if not _HAS_SKLEARN:
        logger.warning("scikit-learn missing — TF-IDF retrieval disabled.")
        return None, None
    params = cfg.eda.get("retrieval", {})
    try:
        vec = TfidfVectorizer(
            analyzer=params.get("analyzer", "char_wb"),
            ngram_range=(int(params.get("ngram_min", 3)), int(params.get("ngram_max", 5))),
            max_features=int(params.get("max_features", 30000)),
            sublinear_tf=bool(params.get("sublinear_tf", True)),
            lowercase=False,  # inputs are already normalized
        )
        mat = vec.fit_transform(texts)
        return vec, mat
    except ValueError as exc:  # e.g. empty vocabulary (all strings blank)
        logger.warning("TF-IDF fit failed (%s) — retrieval disabled for this field.", exc)
        return None, None


def topk_from_sparse(
    scores: "csr_matrix", k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-row top-k from a sparse score matrix; missing slots are -1 / 0.0."""
    scores = scores.tocsr()
    nrows = scores.shape[0]
    top_idx = np.full((nrows, k), -1, dtype=np.int64)
    top_val = np.zeros((nrows, k), dtype=np.float32)
    indptr, indices, data = scores.indptr, scores.indices, scores.data
    for i in range(nrows):
        s, e = indptr[i], indptr[i + 1]
        if e <= s:
            continue
        row_idx = indices[s:e]
        row_val = data[s:e]
        if len(row_val) > k:
            part = np.argpartition(row_val, -k)[-k:]
            order = part[np.argsort(row_val[part])[::-1]]
        else:
            order = np.argsort(row_val)[::-1]
        m = len(order)
        top_idx[i, :m] = row_idx[order]
        top_val[i, :m] = row_val[order]
    return top_idx, top_val


def retrieve_topk(
    query_texts: List[str],
    vectorizer: Optional[Any],
    pool_matrix: Optional[Any],
    k: int,
    chunk_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cosine top-k pool neighbours for each query (sparse, chunked)."""
    n = len(query_texts)
    top_idx = np.full((n, k), -1, dtype=np.int64)
    top_val = np.zeros((n, k), dtype=np.float32)
    if vectorizer is None or pool_matrix is None or n == 0 or k <= 0:
        return top_idx, top_val
    pool_t = pool_matrix.T.tocsr()
    for s in range(0, n, chunk_size):
        e = min(n, s + chunk_size)
        q = vectorizer.transform(query_texts[s:e])
        sims = (q @ pool_t).tocsr()
        ti, tv = topk_from_sparse(sims, k)
        top_idx[s:e] = ti
        top_val[s:e] = tv
    return top_idx, top_val


# ---------------------------------------------------------------------------
# negative sampling framework
# ---------------------------------------------------------------------------

def build_negative_sets(
    s1_raw: Optional[pd.DataFrame],
    s2_raw: Optional[pd.DataFrame],
    s3_raw: Optional[pd.DataFrame],
    gt_df: Optional[pd.DataFrame],
    gt_stats: Optional[pd.DataFrame],
    cfg: AppConfig,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any], Dict[str, List[str]], pd.DataFrame, pd.DataFrame]:
    """Anchor-based negatives on a SAMPLED pool with forced truth (bounded).

    Samples matched + singleton S1 anchors and a candidate pool that is FORCED
    to contain every anchor's true matches, normalizes only those small
    lookups, and mines random + TF-IDF hard negatives that exclude truth via
    the SMALL anchor-truth dict (never a global 7.6M pair set).

    Returns (neg_basic_by_type, info, anchor_truth, s1_small, pool_small).
    Callers featurize via the returned SMALL normalized lookups.
    """
    cols = cfg.columns
    c_id = cols["entity_id"]
    sampling = _sampling_cfg(cfg)
    n_match_anchors = int(sampling.get("n_negative_match_anchors", 3000))
    n_single_anchors = int(sampling.get("n_negative_singleton_anchors", 2000))
    pool_sample_n = int(sampling.get("retrieval_pool_sample", 150000))
    neg_cfg = cfg.eda.get("negatives", {})
    n_random = int(neg_cfg.get("n_random", 3000))
    n_name = int(neg_cfg.get("n_name_hard", 3000))
    n_addr = int(neg_cfg.get("n_address_hard", 3000))
    n_hyb = int(neg_cfg.get("n_hybrid_hard", 2000))
    topk = int(neg_cfg.get("retrieval_topk", 20))
    max_anchors = neg_cfg.get("max_s1_for_mining", 4000)
    chunk = int(cfg.eda.get("retrieval", {}).get("query_chunk_size", 256))

    rng_anchors = _rng(cfg, salt=302)
    rng_pool = _rng(cfg, salt=303)
    rng_neg = _rng(cfg, salt=304)

    def _frame(rows: List[Tuple[str, str, str, str]]) -> pd.DataFrame:
        df = pd.DataFrame(
            rows, columns=["source1_entity_id", "candidate_entity_id", "source_pair", "neg_type"]
        )
        if not df.empty:
            df = df.sort_values(["source1_entity_id", "candidate_entity_id"]).reset_index(drop=True)
        return df

    def _pair_type(cid: str) -> str:
        return "S1_S2" if cid.startswith("S2-") else ("S1_S3" if cid.startswith("S3-") else "S1_OTHER")

    empty_out = {"random": _frame([]), "name_hard": _frame([]),
                 "address_hard": _frame([]), "hybrid_hard": _frame([])}
    if s1_raw is None or s1_raw.empty or gt_df is None or gt_df.empty:
        info = {"n_random": 0, "n_name_hard": 0, "n_address_hard": 0,
                "n_hybrid_hard": 0, "n_anchors_mined": 0,
                "n_anchors_match": 0, "n_anchors_singleton": 0, "n_pool_sampled": 0}
        return empty_out, info, {}, pd.DataFrame(), pd.DataFrame()

    if gt_stats is not None and not gt_stats.empty and cols["gt_source1"] in gt_stats.columns:
        matched_all = sorted({str(x) for x in
            gt_stats.loc[gt_stats["n_matches"] > 0, cols["gt_source1"]].astype(str).tolist()})
        single_all = sorted({str(x) for x in
            gt_stats.loc[gt_stats["n_matches"] == 0, cols["gt_source1"]].astype(str).tolist()})
    else:
        matched_all, single_all = [], []
    match_anchors = _sample_ids(matched_all, n_match_anchors, rng_anchors)
    # reuse the same RNG stream deterministically for singletons
    single_anchors = _sample_ids(
        [x for x in single_all if x not in set(match_anchors)], n_single_anchors, rng_anchors)
    anchor_ids = sorted(set(match_anchors) | set(single_anchors))
    if not anchor_ids:
        info = {"n_random": 0, "n_name_hard": 0, "n_address_hard": 0,
                "n_hybrid_hard": 0, "n_anchors_mined": 0,
                "n_anchors_match": len(match_anchors),
                "n_anchors_singleton": len(single_anchors), "n_pool_sampled": 0}
        return empty_out, info, {}, pd.DataFrame(), pd.DataFrame()

    anchor_truth = parse_anchor_truth(gt_df, anchor_ids, cols)
    anchor_sets: Dict[str, Set[str]] = {a: set(v) for a, v in anchor_truth.items()}
    forced: Set[str] = set()
    for v in anchor_truth.values():
        forced.update(map(str, v))

    pool_sample_raw = _sample_pool_with_forced_truth(
        s2_raw, s3_raw, forced, pool_sample_n, rng_pool, c_id)
    s1_small_raw = _filter_raw_to_ids(s1_raw, set(anchor_ids), c_id)
    s1_small = ensure_normalized_columns(s1_small_raw, cols) if not s1_small_raw.empty else pd.DataFrame()
    pool_small = ensure_normalized_columns(pool_sample_raw, cols) if not pool_sample_raw.empty else pd.DataFrame()

    s1_ids = s1_small[c_id].astype(str).tolist() if (not s1_small.empty and c_id in s1_small.columns) else []
    pool_ids = pool_small[c_id].astype(str).tolist() if (not pool_small.empty and c_id in pool_small.columns) else []

    out: Dict[str, pd.DataFrame] = {}
    info: Dict[str, Any] = {}
    # -- 1. random negatives (closed-world exclusion via small anchor sets) --
    rows: List[Tuple[str, str, str, str]] = []
    seen: set = set()
    attempts = 0
    budget = max(n_random * 30, 10000)
    while len(rows) < n_random and attempts < budget and s1_ids and pool_ids:
        attempts += 1
        a = s1_ids[rng_neg.randint(len(s1_ids))]
        b = pool_ids[rng_neg.randint(len(pool_ids))]
        if b in anchor_sets.get(a, set()) or (a, b) in seen:
            continue
        seen.add((a, b))
        rows.append((a, b, _pair_type(b), "random"))
    out["random"] = _frame(rows)
    info["n_random"] = len(rows)

    # -- retrieval-backed hard negatives on the SMALL pool --
    mining_ids = sorted(set(s1_ids))
    if max_anchors and len(mining_ids) > int(max_anchors):
        mining_ids = _sample_ids(mining_ids, int(max_anchors), _rng(cfg, salt=305))
    can_retrieve = (
        not s1_small.empty and not pool_small.empty
        and "name_norm" in s1_small.columns and "name_norm" in pool_small.columns
        and mining_ids and pool_ids
    )
    a_text_name: List[str] = []
    a_text_addr: List[str] = []
    name_vec = name_mat = addr_vec = addr_mat = None
    if can_retrieve:
        try:
            indexed = s1_small.set_index(c_id)
            a_text_name = indexed.loc[mining_ids, "name_norm"].astype(str).tolist()
            a_text_addr = indexed.loc[mining_ids, "address_norm"].astype(str).tolist()
        except KeyError:
            keep = [a for a in mining_ids if a in set(s1_ids)]
            mining_ids = keep
            indexed = s1_small.set_index(c_id)
            a_text_name = indexed.loc[mining_ids, "name_norm"].astype(str).tolist() if keep else []
            a_text_addr = indexed.loc[mining_ids, "address_norm"].astype(str).tolist() if keep else []
        p_text_name = pool_small["name_norm"].astype(str).tolist()
        p_text_addr = pool_small["address_norm"].astype(str).tolist()
        name_vec, name_mat = build_tfidf_index(p_text_name, cfg)
        addr_vec, addr_mat = build_tfidf_index(p_text_addr, cfg)
    else:
        p_text_name, p_text_addr = [], []

    def _collect(idx: np.ndarray, n_target: int, tag: str) -> pd.DataFrame:
        """Round-robin over anchors so hard negatives cover many S1s."""
        picked: List[Tuple[str, str, str, str]] = []
        used: set = set()
        rank = 0
        if idx.size == 0 or not mining_ids:
            return _frame(picked)
        while len(picked) < n_target and rank < idx.shape[1]:
            progressed = False
            for ai, a in enumerate(mining_ids):
                if len(picked) >= n_target:
                    break
                if ai >= idx.shape[0]:
                    continue
                j = int(idx[ai, rank])
                if j < 0 or j >= len(pool_ids):
                    continue
                b = pool_ids[j]
                if b in anchor_sets.get(a, set()) or (a, b) in used:
                    continue
                used.add((a, b))
                picked.append((a, b, _pair_type(b), tag))
                progressed = True
            if not progressed:
                rank += 1
            else:
                rank += 1 if len(picked) >= (rank + 1) * max(1, len(mining_ids)) // 4 else 0
                if rank >= idx.shape[1]:
                    break
                if rank < 0:
                    rank = 0
        return _frame(picked)

    if name_vec is not None and a_text_name:
        ni, _ = retrieve_topk(a_text_name, name_vec, name_mat, topk, chunk)
        out["name_hard"] = _collect(ni, n_name, "name_hard")
    else:
        out["name_hard"] = _frame([])
    info["n_name_hard"] = len(out["name_hard"])

    if addr_vec is not None and a_text_addr:
        ai_, _ = retrieve_topk(a_text_addr, addr_vec, addr_mat, topk, chunk)
        out["address_hard"] = _collect(ai_, n_addr, "address_hard")
    else:
        out["address_hard"] = _frame([])
    info["n_address_hard"] = len(out["address_hard"])

    if name_vec is not None and addr_vec is not None and a_text_name and a_text_addr:
        nq = len(a_text_name)
        hyb_idx = np.full((nq, topk), -1, dtype=np.int64)
        try:
            pool_n_t = name_mat.T.tocsr()
            pool_a_t = addr_mat.T.tocsr()
            for s in range(0, nq, chunk):
                e = min(nq, s + chunk)
                qn = name_vec.transform(a_text_name[s:e])
                qa = addr_vec.transform(a_text_addr[s:e])
                sims = ((qn @ pool_n_t) + (qa @ pool_a_t)).tocsr()
                ti, _ = topk_from_sparse(sims, topk)
                hyb_idx[s:e] = ti
            out["hybrid_hard"] = _collect(hyb_idx, n_hyb, "hybrid_hard")
        except Exception as exc:
            logger.warning("Hybrid retrieval failed (%s) — hybrid_hard empty.", exc)
            out["hybrid_hard"] = _frame([])
    else:
        out["hybrid_hard"] = _frame([])
    info["n_hybrid_hard"] = len(out["hybrid_hard"])
    info["n_anchors_mined"] = len(mining_ids)
    info["n_anchors_match"] = len(match_anchors)
    info["n_anchors_singleton"] = len(single_anchors)
    info["n_pool_sampled"] = int(len(pool_small))
    info["n_s1_lookup"] = int(len(s1_small))
    return out, info, anchor_truth, s1_small, pool_small


def _add_pair_breakdowns(
    feat: pd.DataFrame, gt_stats: Optional[pd.DataFrame], cols: Dict[str, str]
) -> pd.DataFrame:
    """Add missingness/exactness/match-count buckets (small-map, no 2.2M dict)."""
    out = feat.copy()
    if out.empty:
        out["addr_missing_bucket"] = pd.Series(dtype=str)
        out["name_exact_bucket"] = pd.Series(dtype=str)
        out["s1_match_bucket"] = pd.Series(dtype=str)
        return out
    out["addr_missing_bucket"] = np.select(
        [
            (out["s1_addr_missing"] == 1) & (out["cand_addr_missing"] == 1),
            (out["s1_addr_missing"] == 1),
            (out["cand_addr_missing"] == 1),
        ],
        ["both_missing", "s1_missing", "cand_missing"],
        default="both_present",
    )
    out["name_exact_bucket"] = np.where(out["name_exact"] == 1.0, "exact_name", "non_exact_name")
    if gt_stats is not None and not gt_stats.empty and cols.get("gt_source1") in gt_stats.columns:
        try:
            needed = set(out["source1_entity_id"].astype(str).unique().tolist())
            if len(gt_stats) > 50000 and len(needed) < len(gt_stats):
                small = gt_stats[gt_stats[cols["gt_source1"]].astype(str).isin(needed)]
                m = dict(zip(small[cols["gt_source1"]].astype(str),
                             small["n_matches"].astype(int)))
            else:
                m = dict(zip(gt_stats[cols["gt_source1"]].astype(str),
                             gt_stats["n_matches"].astype(int)))
            out["s1_match_bucket"] = out["source1_entity_id"].map(m).fillna(-1).astype(int).map(
                lambda n: "1" if n == 1 else ("2-3" if 2 <= n <= 3 else ("4+" if n >= 4 else "unknown"))
            )
        except Exception:
            out["s1_match_bucket"] = "unknown"
    else:
        out["s1_match_bucket"] = "unknown"
    return out


def run_positive_pair_eda(
    pos_feat: pd.DataFrame,
    gt_stats: Optional[pd.DataFrame],
    cfg: AppConfig,
) -> Dict[str, Any]:
    """Write 03 CSV + positive-only figures (split by pair/country/missingness)."""
    _apply_style(cfg)
    cols = cfg.columns
    out_csv = cfg.eda_dir / "03_positive_pair_feature_summary.csv"
    artifacts = [str(out_csv)]
    if pos_feat.empty:
        summarize_feature_frame(pos_feat, ["source_pair"], FEATURE_COLUMNS).to_csv(out_csv, index=False)
        return {"positive_csv": str(out_csv), "artifacts": artifacts, "skipped": True}

    feat = _add_pair_breakdowns(pos_feat, gt_stats, cols)
    breakdowns = [
        ("overall", []),
        ("source_pair", ["source_pair"]),
        ("s1_country", ["s1_country"]),
        ("addr_missing", ["addr_missing_bucket"]),
        ("name_exact", ["name_exact_bucket"]),
        ("s1_match_count", ["s1_match_bucket"]),
        ("source_pair_x_country", ["source_pair", "s1_country"]),
    ]
    parts = []
    for name, gcols in breakdowns:
        s = summarize_feature_frame(feat, gcols, FEATURE_COLUMNS)
        if not s.empty:
            s.insert(0, "breakdown", name)
            parts.append(s)
    summary = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    summary.to_csv(out_csv, index=False)

    # figures: ECDFs of name/address similarity by source pair
    for feature, fname, title in [
        ("name_ratio", "positive_name_similarity.png", "Positive name similarity (name_ratio)"),
        ("addr_token_jaccard", "positive_address_similarity.png", "Positive address similarity (token Jaccard)"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.4))
        for pair in ("S1_S2", "S1_S3"):
            vals = pd.to_numeric(feat.loc[feat.source_pair == pair, feature], errors="coerce").to_numpy(dtype=float)
            x, y = _ecdf(vals)
            if len(x):
                ax.plot(x, y, label=f"{pair} (n={(~np.isnan(vals)).sum()})")
        ax.set_title(title)
        ax.set_xlabel(feature)
        ax.set_ylabel("ECDF")
        ax.legend()
        artifacts.append(_savefig(fig, cfg.figures_dir / fname, cfg))

    # source-pair comparison panel
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, feature in zip(
        axes.ravel(), ["name_ratio", "name_token_set", "addr_ratio", "addr_token_jaccard"]
    ):
        data = [
            pd.to_numeric(feat.loc[feat.source_pair == p, feature], errors="coerce").dropna().to_numpy(dtype=float)
            for p in ("S1_S2", "S1_S3")
        ]
        ax.boxplot([d for d in data if len(d)], labels=[p for p, d in zip(("S1_S2", "S1_S3"), data) if len(d)], showfliers=False)
        ax.set_title(feature)
    artifacts.append(_savefig(fig, cfg.figures_dir / "source_pair_comparison.png", cfg))

    def _rate(mask: pd.Series) -> float:
        return round(float(mask.mean()), 6) if len(mask) else 0.0

    results = {
        "positive_csv": str(out_csv),
        "artifacts": artifacts,
        "skipped": False,
        "n_positives": int(len(feat)),
        "name_exact_rate": _rate(feat["name_exact"] == 1.0),
        "addr_exact_rate": _rate(feat["addr_exact"] == 1.0),
        "country_equal_rate": _rate(feat["country_equal"] == 1.0),
        "either_addr_missing_rate": _rate(feat["either_addr_missing"] == 1.0),
        "name_ratio_p50": round(float(feat["name_ratio"].median()), 4),
        "addr_jaccard_p50": round(float(feat["addr_token_jaccard"].median()), 4),
        "by_source_pair": {
            p: {
                "n": int((feat.source_pair == p).sum()),
                "name_exact_rate": _rate(feat.loc[feat.source_pair == p, "name_exact"] == 1.0),
                "addr_exact_rate": _rate(feat.loc[feat.source_pair == p, "addr_exact"] == 1.0),
                "name_ratio_p50": round(float(pd.to_numeric(feat.loc[feat.source_pair == p, "name_ratio"], errors="coerce").median()), 4) if (feat.source_pair == p).any() else None,
                "addr_jaccard_p50": round(float(pd.to_numeric(feat.loc[feat.source_pair == p, "addr_token_jaccard"], errors="coerce").median()), 4) if (feat.source_pair == p).any() else None,
            }
            for p in ("S1_S2", "S1_S3")
        },
    }
    return results


def run_hard_negative_eda(
    pos_feat: pd.DataFrame,
    negs_feat: pd.DataFrame,
    cfg: AppConfig,
) -> Dict[str, Any]:
    """Write 04 CSV + positive-vs-negative figures (ECDFs, 2D scatter, heatmap)."""
    _apply_style(cfg)
    out_csv = cfg.eda_dir / "04_hard_negative_feature_summary.csv"
    artifacts = [str(out_csv)]
    if negs_feat.empty and pos_feat.empty:
        summarize_feature_frame(negs_feat, ["neg_type"], FEATURE_COLUMNS).to_csv(out_csv, index=False)
        return {"negative_csv": str(out_csv), "artifacts": artifacts, "skipped": True}

    parts = []
    for name, gcols in (("neg_type", ["neg_type"]), ("neg_type_x_pair", ["neg_type", "source_pair"])):
        s = summarize_feature_frame(negs_feat, gcols, FEATURE_COLUMNS)
        if not s.empty:
            s.insert(0, "breakdown", name)
            parts.append(s)
    summary = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    summary.to_csv(out_csv, index=False)

    combined = pd.concat(
        [pos_feat.assign(cls="positive")] +
        [negs_feat.assign(cls=negs_feat["neg_type"]) if not negs_feat.empty else pos_feat.iloc[0:0].assign(cls=[])],
        ignore_index=True,
    )
    order = ["positive", "random", "name_hard", "address_hard", "hybrid_hard"]
    order = [c for c in order if (combined["cls"] == c).any()]

    # ECDF grid
    grid_feats = ["name_ratio", "name_token_set", "name_jaro_winkler",
                  "addr_ratio", "addr_token_jaccard", "numeric_jaccard"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, feature in zip(axes.ravel(), grid_feats):
        for cls in order:
            vals = pd.to_numeric(combined.loc[combined.cls == cls, feature], errors="coerce").to_numpy(dtype=float)
            x, y = _ecdf(vals)
            if len(x):
                ax.plot(x, y, label=cls)
        ax.set_title(feature)
        ax.set_xlabel(feature)
    axes[0, 0].set_ylabel("ECDF")
    fig.legend(loc="lower center", ncol=len(order), bbox_to_anchor=(0.5, -0.02))
    artifacts.append(_savefig(fig, cfg.figures_dir / "positive_vs_negative_similarity.png", cfg))

    # 2D scatter: name vs address similarity; marker encodes numeric agreement
    rng = _rng(cfg, salt=202)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    markers = {1.0: ("o", "numeric overlap"), 0.0: ("x", "no numeric overlap")}
    for cls in order:
        sub = combined[combined.cls == cls]
        if len(sub) > 4000:
            sub = sub.sample(4000, random_state=int(rng.randint(2**31 - 1)))
        for flag, (mk, lab) in markers.items():
            ss = sub[sub["numeric_any_overlap"] == flag]
            if len(ss):
                ax.scatter(ss["name_ratio"], ss["addr_token_jaccard"], s=12, alpha=0.45,
                           marker=mk, label=f"{cls} / {lab} (n={len(ss)})")
    ax.set_title("Name vs address similarity (marker = numeric-token overlap)")
    ax.set_xlabel("name_ratio (normalized)")
    ax.set_ylabel("addr_token_jaccard (normalized)")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.92)
    artifacts.append(_savefig(fig, cfg.figures_dir / "name_vs_address_scatter.png", cfg))

    # correlation heatmap over key features
    heat_feats = ["name_ratio", "name_token_set", "name_jaro_winkler", "name_wratio",
                  "addr_ratio", "addr_token_jaccard", "addr_containment_s1_in_cand",
                  "numeric_any_overlap", "numeric_conflict", "postcode_any_overlap",
                  "country_equal", "label"]
    heat_df = combined[[c for c in heat_feats if c in combined.columns]].copy()
    for c in heat_df.columns:
        heat_df[c] = pd.to_numeric(heat_df[c], errors="coerce")
    heat_df = heat_df.dropna()
    if len(heat_df) > 20000:
        heat_df = heat_df.sample(20000, random_state=int(rng.randint(2**31 - 1)))
    corr = heat_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    if _HAS_SEABORN and not corr.empty:
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                    square=True, ax=ax, cbar_kws={"shrink": 0.8})
    else:
        im = ax.imshow(corr.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(corr.columns, fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Pair-feature correlation (positives + sampled negatives)")
    artifacts.append(_savefig(fig, cfg.figures_dir / "feature_correlation_heatmap.png", cfg))

    results = {
        "negative_csv": str(out_csv),
        "artifacts": artifacts,
        "skipped": False,
        "n_by_neg_type": {str(k): int(v) for k, v in negs_feat["neg_type"].value_counts().items()} if not negs_feat.empty else {},
    }
    return results


# ===========================================================================
# ADDRESS-SPECIFIC + NAME-SPECIFIC EDA (figures + summary inputs)
# ===========================================================================

def _class_rates(df: pd.DataFrame, cols_: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for cls, grp in df.groupby("cls"):
        out[str(cls)] = {c: round(float((pd.to_numeric(grp[c], errors="coerce") == 1.0).mean()), 6) for c in cols_}
    return out


def run_address_eda(
    pos_feat: pd.DataFrame, negs_feat: pd.DataFrame, cfg: AppConfig
) -> Dict[str, Any]:
    """Numeric/postcode agreement: positives vs hard negatives (+ 2x2 table)."""
    _apply_style(cfg)
    combined = pd.concat(
        [pos_feat.assign(cls="positive"), negs_feat.assign(cls=negs_feat["neg_type"])],
        ignore_index=True,
    ) if not pos_feat.empty else negs_feat.assign(cls=negs_feat["neg_type"])
    num_cols = ["numeric_exact_set", "numeric_any_overlap", "numeric_conflict",
                "numeric_one_side_missing", "numeric_both_missing",
                "postcode_any_overlap", "house_number_agree"]
    rates = _class_rates(combined, num_cols) if not combined.empty else {}

    artifacts: List[str] = []
    if rates:
        classes = [c for c in ("positive", "random", "name_hard", "address_hard", "hybrid_hard") if c in rates]
        labels = ["exact set", "any overlap", "CONFLICT", "one-side missing",
                  "both missing", "postcode overlap", "house-no agree"]
        x = np.arange(len(labels))
        width = 0.8 / max(1, len(classes))
        fig, ax = plt.subplots(figsize=(12, 4.6))
        for i, cls in enumerate(classes):
            ax.bar(x + (i - (len(classes) - 1) / 2) * width,
                   [rates[cls][c] for c in num_cols], width=width, label=cls)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylabel("rate")
        ax.set_title("Address numeric agreement: positives vs negatives")
        ax.legend(fontsize=8)
        artifacts.append(_savefig(fig, cfg.figures_dir / "address_numeric_agreement.png", cfg))

    # 2x2: house-number agrees vs conflicts (positives vs all hard negatives)
    table_2x2: Dict[str, Any] = {}
    if not pos_feat.empty and not negs_feat.empty:
        hard = negs_feat[negs_feat["neg_type"].isin(["name_hard", "address_hard", "hybrid_hard"])]
        table_2x2 = {
            "positive_house_agree": int((pos_feat["house_number_agree"] == 1.0).sum()),
            "positive_conflict": int((pos_feat["numeric_conflict"] == 1.0).sum()),
            "positive_n": int(len(pos_feat)),
            "hardneg_house_agree": int((hard["house_number_agree"] == 1.0).sum()),
            "hardneg_conflict": int((hard["numeric_conflict"] == 1.0).sum()),
            "hardneg_n": int(len(hard)),
        }
    return {"artifacts": artifacts, "numeric_rates_by_class": rates, "house_2x2": table_2x2}


def _suffix_of(norm_name: str) -> str:
    toks = tokenize(norm_name or "")
    return toks[-1] if toks else ""


def _core_name(norm_name: str) -> str:
    toks = [t for t in tokenize(norm_name or "")
            if t not in LEGAL_SUFFIX_TOKENS and t not in BUSINESS_TYPE_TOKENS]
    return " ".join(toks)


def _first_last_match(a: str, b: str) -> Tuple[float, float]:
    ta, tb = tokenize(a or ""), tokenize(b or "")
    if not ta or not tb:
        return 0.0, 0.0
    return float(ta[0] == tb[0]), float(ta[-1] == tb[-1])


def _acronym_like(a: str, b: str) -> float:
    ta, tb = tokenize(a or ""), tokenize(b or "")
    if len(ta) >= 2 and len(tb) == 1:
        return float("".join(t[0] for t in ta if t) == tb[0])
    if len(tb) >= 2 and len(ta) == 1:
        return float("".join(t[0] for t in tb if t) == ta[0])
    return 0.0


def run_name_eda(
    pos_feat: pd.DataFrame, negs_feat: pd.DataFrame, cfg: AppConfig
) -> Dict[str, Any]:
    """Suffix / business-type / order / acronym behaviour on THIS dataset."""
    _apply_style(cfg)
    combined = pd.concat(
        [pos_feat.assign(cls="positive"), negs_feat.assign(cls=negs_feat["neg_type"])],
        ignore_index=True,
    ) if not pos_feat.empty else negs_feat.assign(cls=negs_feat["neg_type"])
    artifacts: List[str] = []
    summary: Dict[str, Any] = {"artifacts": artifacts}
    if combined.empty:
        summary["suffix_rates_by_class"] = {}
        return summary

    suf_a = combined["s1_name_norm"].astype(str).map(_suffix_of)
    suf_b = combined["cand_name_norm"].astype(str).map(_suffix_of)
    same_suffix = (suf_a == suf_b) & (suf_a != "")
    one_missing = ((suf_a == "") != (suf_b == ""))
    both_missing = (suf_a == "") & (suf_b == "")
    suffix_in_lex = suf_a.isin(LEGAL_SUFFIX_TOKENS | BUSINESS_TYPE_TOKENS) | suf_b.isin(
        LEGAL_SUFFIX_TOKENS | BUSINESS_TYPE_TOKENS
    )
    core_a = combined["s1_name_norm"].astype(str).map(_core_name)
    core_b = combined["cand_name_norm"].astype(str).map(_core_name)
    core_exact = (core_a == core_b) & (core_a != "")
    fl = [_first_last_match(a, b) for a, b in
          zip(combined["s1_name_norm"].astype(str), combined["cand_name_norm"].astype(str))]
    first_match = pd.Series([x[0] for x in fl], index=combined.index)
    last_match = pd.Series([x[1] for x in fl], index=combined.index)
    acr = pd.Series(
        [_acronym_like(a, b) for a, b in
         zip(combined["s1_name_norm"].astype(str), combined["cand_name_norm"].astype(str))],
        index=combined.index,
    )
    combined = combined.assign(
        _same_suffix=same_suffix.astype(float), _one_missing=one_missing.astype(float),
        _both_missing=both_missing.astype(float), _suffix_in_lex=suffix_in_lex.astype(float),
        _core_exact=core_exact.astype(float), _first_match=first_match,
        _last_match=last_match, _acronym_like=acr,
        _sorted_equal=pd.to_numeric(combined["name_sorted_token_equal"], errors="coerce"),
    )
    rates = _class_rates(
        combined,
        ["_same_suffix", "_one_missing", "_both_missing", "_suffix_in_lex",
         "_core_exact", "_first_match", "_last_match", "_acronym_like", "_sorted_equal"],
    )
    summary["suffix_rates_by_class"] = rates

    classes = [c for c in ("positive", "random", "name_hard", "address_hard", "hybrid_hard") if c in rates]
    if classes:
        labels = ["same suffix", "suffix one-side missing", "both empty",
                  "suffix in lexicon", "core-name exact",
                  "first-token match", "last-token match", "acronym-like", "sorted-token equal"]
        keys = ["_same_suffix", "_one_missing", "_both_missing", "_suffix_in_lex",
                "_core_exact", "_first_match", "_last_match", "_acronym_like", "_sorted_equal"]
        x = np.arange(len(labels))
        width = 0.8 / len(classes)
        fig, ax = plt.subplots(figsize=(13, 4.8))
        for i, cls in enumerate(classes):
            ax.bar(x + (i - (len(classes) - 1) / 2) * width,
                   [rates[cls][k] for k in keys], width=width, label=cls)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("rate")
        ax.set_title("Name structure: suffix / order / acronym behaviour")
        ax.legend(fontsize=8)
        artifacts.append(_savefig(fig, cfg.figures_dir / "name_suffix_agreement.png", cfg))
    return summary


# ===========================================================================
# 6. BLOCKING DIAGNOSTICS -> 06_blocking_metrics.csv, 07_blocking_positive_coverage.csv
# ===========================================================================

def _cap_per_s1(
    pairs: pd.DataFrame, cap: int
) -> Tuple[pd.DataFrame, int]:
    """Deterministically cap candidates per S1 (sorted cand ids); returns (pairs, n_truncated_s1)."""
    if pairs.empty or cap <= 0:
        return pairs, 0
    counts = pairs.groupby("source1_entity_id").size()
    big = set(counts[counts > cap].index.tolist())
    if not big:
        return pairs, 0
    pairs = pairs.sort_values(["source1_entity_id", "candidate_entity_id"])
    pairs["_rank"] = pairs.groupby("source1_entity_id").cumcount()
    pairs = pairs[pairs["_rank"] < cap].drop(columns=["_rank"]).reset_index(drop=True)
    return pairs, len(big)


def _exact_join_blocker(
    s1_df: pd.DataFrame, pool_df: pd.DataFrame, key: str, id_col: str
) -> pd.DataFrame:
    left = s1_df.loc[s1_df[key] != "", [id_col, key]]
    right = pool_df.loc[pool_df[key] != "", [id_col, key]]
    if left.empty or right.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
    m = left.merge(right, on=key, suffixes=("_s1", "_cand"))
    return pd.DataFrame(
        {"source1_entity_id": m[f"{id_col}_s1"], "candidate_entity_id": m[f"{id_col}_cand"]}
    ).drop_duplicates().reset_index(drop=True)


def _inverted_index_blocker(
    s1_tokens: List[set], pool_tokens: List[set],
    s1_ids: List[str], pool_ids: List[str],
) -> pd.DataFrame:
    """Union candidates sharing >=1 (pre-filtered, e.g. rare) token."""
    postings: Dict[str, List[int]] = {}
    for j, toks in enumerate(pool_tokens):
        for t in toks:
            postings.setdefault(t, []).append(j)
    rows: List[Tuple[str, str]] = []
    for a, toks in zip(s1_ids, s1_tokens):
        cands: set = set()
        for t in toks:
            for j in postings.get(t, ()):
                cands.add(pool_ids[j])
        for b in cands:
            rows.append((a, b))
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])


def _topk_blocker(
    query_texts: List[str], s1_ids: List[str], pool_ids: List[str],
    vectorizer: Optional[Any], pool_matrix: Optional[Any], k: int, chunk: int,
) -> pd.DataFrame:
    idx, _ = retrieve_topk(query_texts, vectorizer, pool_matrix, k, chunk)
    rows: List[Tuple[str, str]] = []
    for ai, a in enumerate(s1_ids):
        for j in idx[ai]:
            if int(j) >= 0:
                rows.append((a, pool_ids[int(j)]))
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])


def _mark_retrieved(positives: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
    """Boolean mask over positives: was each pair retrieved by this blocker?"""
    if positives.empty:
        return np.array([], dtype=bool)
    if pairs.empty:
        return np.zeros(len(positives), dtype=bool)
    if len(pairs) <= 600_000:
        hit = set(zip(pairs["source1_entity_id"].astype(str), pairs["candidate_entity_id"].astype(str)))
        return np.array(
            [(a, b) in hit for a, b in zip(
                positives["source1_entity_id"].astype(str),
                positives["candidate_entity_id"].astype(str))],
            dtype=bool,
        )
    key_p = positives["source1_entity_id"].astype(str) + "\x00" + positives["candidate_entity_id"].astype(str)
    key_b = pairs["source1_entity_id"].astype(str) + "\x00" + pairs["candidate_entity_id"].astype(str)
    return key_p.isin(set(key_b.unique())).to_numpy(dtype=bool)


def run_blocking_eda(
    s1_raw: Optional[pd.DataFrame],
    s2_raw: Optional[pd.DataFrame],
    s3_raw: Optional[pd.DataFrame],
    gt_df: Optional[pd.DataFrame],
    gt_stats: Optional[pd.DataFrame],
    cfg: AppConfig,
) -> Dict[str, Any]:
    """Anchor-based blocking diagnostics on a SAMPLED pool with forced truth.

    Samples ``n_blocking_s1`` S1 anchors and a ``blocking_pool_sample`` pool
    FORCED to contain every anchor's true matches, normalizes only those small
    frames, and evaluates diagnostic blockers in that closed world. Returns
    ``union_pairs`` + ``s1_small`` + ``pool_small`` + ``anchor_truth`` for the
    graph and casebook stages (small frames only, never the full 10M pool).
    """
    _apply_style(cfg)
    cols = cfg.columns
    c_id = cols["entity_id"]
    ret_cfg = cfg.eda.get("retrieval", {})
    blk_cfg = cfg.eda.get("blocking", {})
    sampling = _sampling_cfg(cfg)
    n_blocking_s1 = int(sampling.get("n_blocking_s1", 8000))
    blocking_pool_n = int(sampling.get("blocking_pool_sample", 250000))
    enabled = blk_cfg.get("enabled", []) or []
    per_s1_cap = int(blk_cfg.get("max_candidates_per_s1_cap", 5000))
    chunk = int(ret_cfg.get("query_chunk_size", 256))
    out_metrics = cfg.eda_dir / "06_blocking_metrics.csv"
    out_coverage = cfg.eda_dir / "07_blocking_positive_coverage.csv"

    if s1_raw is None or s1_raw.empty or c_id not in s1_raw.columns:
        pd.DataFrame(columns=["blocker", "n_candidate_pairs", "s1_s2_recall",
                              "s1_s3_recall", "overall_recall", "avg_candidates_per_s1",
                              "p50_candidates_per_s1", "p95_candidates_per_s1",
                              "max_candidates_per_s1", "candidate_burden",
                              "s1_with_zero_candidates", "truncated_s1"]).to_csv(out_metrics, index=False)
        pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "source_pair",
                              "n_blockers_hit", "rescued_only_by"]).to_csv(out_coverage, index=False)
        return {"metrics_csv": str(out_metrics), "coverage_csv": str(out_coverage),
                "artifacts": [str(out_metrics), str(out_coverage)], "metrics": [],
                "rescue_counts": {}, "union_pairs": pd.DataFrame(
                    columns=["source1_entity_id", "candidate_entity_id"]),
                "s1_small": pd.DataFrame(), "pool_small": pd.DataFrame(),
                "anchor_truth": {}, "positives_small": pd.DataFrame(
                    columns=["source1_entity_id", "candidate_entity_id", "source_pair"]),
                "n_s1": 0, "n_pool": 0, "n_positives_sampled": 0}

    rng_anchors = _rng(cfg, salt=401)
    rng_pool = _rng(cfg, salt=402)
    all_s1_ids = sorted({str(x) for x in s1_raw[c_id].astype(str).tolist()})
    anchor_ids = _sample_ids(all_s1_ids, n_blocking_s1, rng_anchors)
    anchor_truth = parse_anchor_truth(gt_df, anchor_ids, cols) if gt_df is not None else {a: [] for a in anchor_ids}
    positives_small = anchor_truth_to_basic(anchor_truth)
    if not positives_small.empty:
        positives_small = positives_small[
            positives_small["source_pair"].isin(["S1_S2", "S1_S3"])].reset_index(drop=True)
    forced: Set[str] = set()
    for v in anchor_truth.values():
        forced.update(map(str, v))
    pool_sample_raw = _sample_pool_with_forced_truth(
        s2_raw, s3_raw, forced, blocking_pool_n, rng_pool, c_id)
    s1_small_raw = _filter_raw_to_ids(s1_raw, set(anchor_ids), c_id)
    s1_df = ensure_normalized_columns(s1_small_raw, cols) if not s1_small_raw.empty else pd.DataFrame()
    pool_df = ensure_normalized_columns(pool_sample_raw, cols) if not pool_sample_raw.empty else pd.DataFrame()
    if not positives_small.empty and not s1_df.empty and not pool_df.empty:
        found_s1 = set(s1_df[c_id].astype(str).tolist())
        found_pool = set(pool_df[c_id].astype(str).tolist())
        positives_small = positives_small[
            positives_small["source1_entity_id"].astype(str).isin(found_s1)
            & positives_small["candidate_entity_id"].astype(str).isin(found_pool)
        ].reset_index(drop=True)

    s1_ids = s1_df[c_id].astype(str).tolist() if (not s1_df.empty and c_id in s1_df.columns) else []
    pool_ids = pool_df[c_id].astype(str).tolist() if (not pool_df.empty and c_id in pool_df.columns) else []
    all_s1 = pd.DataFrame({"source1_entity_id": s1_ids})

    # retrieval indices (fit once on the SMALL pool, reused)
    if not pool_df.empty and "name_norm" in pool_df.columns:
        name_vec, name_mat = build_tfidf_index(pool_df["name_norm"].astype(str).tolist(), cfg)
        addr_vec, addr_mat = build_tfidf_index(pool_df["address_norm"].astype(str).tolist(), cfg)
    else:
        name_vec = name_mat = addr_vec = addr_mat = None

    # rare-token prep (SMALL frames only)
    rare_df = int(ret_cfg.get("rare_token_max_df", 25))
    rare_min_len = int(ret_cfg.get("rare_token_min_len", 4))
    if not s1_df.empty and not pool_df.empty and "name_norm" in s1_df.columns and "name_norm" in pool_df.columns:
        pool_name_toks = [set(t for t in tokenize(t) if len(t) >= rare_min_len)
                          for t in pool_df["name_norm"].astype(str).tolist()]
        df_counter: Counter = Counter()
        for toks in pool_name_toks:
            df_counter.update(toks)
        rare_vocab = {t for t, c in df_counter.items() if c <= rare_df}
        s1_name_toks = [set(t for t in tokenize(t) if t in rare_vocab)
                        for t in s1_df["name_norm"].astype(str).tolist()]
        pool_rare_toks = [toks & rare_vocab for toks in pool_name_toks]
    else:
        s1_name_toks, pool_rare_toks = [], []

    # postcode prep (SMALL frames only)
    from .normalization import extract_postcode_like_tokens as _pl

    if not s1_df.empty and cols["business_address"] in s1_df.columns:
        s1_post = [set(_pl(a)) for a in s1_df[cols["business_address"]].astype(str).tolist()]
    else:
        s1_post = [set() for _ in s1_ids]
    if not pool_df.empty and cols["business_address"] in pool_df.columns:
        pool_post = [set(_pl(a)) for a in pool_df[cols["business_address"]].astype(str).tolist()]
    else:
        pool_post = [set() for _ in pool_ids]

    blockers: Dict[str, pd.DataFrame] = {}
    truncations: Dict[str, int] = {}

    def _register(name: str, pairs: pd.DataFrame) -> None:
        pairs, n_trunc = _cap_per_s1(pairs, per_s1_cap)
        blockers[name] = pairs.drop_duplicates().reset_index(drop=True)
        truncations[name] = n_trunc

    if "exact_norm_name" in enabled:
        if not s1_df.empty and not pool_df.empty and "name_norm" in s1_df.columns and "name_norm" in pool_df.columns:
            _register("exact_norm_name", _exact_join_blocker(s1_df, pool_df, "name_norm", c_id))
        else:
            _register("exact_norm_name", pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"]))
    if "exact_rare_name_token" in enabled:
        _register("exact_rare_name_token",
                  _inverted_index_blocker(s1_name_toks, pool_rare_toks, s1_ids, pool_ids))
    if "exact_postcode" in enabled:
        _register("exact_postcode",
                  _inverted_index_blocker(s1_post, pool_post, s1_ids, pool_ids))
    s1_name_texts = s1_df["name_norm"].astype(str).tolist() if (not s1_df.empty and "name_norm" in s1_df.columns) else []
    s1_addr_texts = s1_df["address_norm"].astype(str).tolist() if (not s1_df.empty and "address_norm" in s1_df.columns) else []
    for name in enabled:
        if name.startswith("name_tfidf_top"):
            try:
                k = int(name.rsplit("top", 1)[1])
            except ValueError:
                continue
            if s1_name_texts and name_vec is not None:
                _register(name, _topk_blocker(s1_name_texts, s1_ids,
                                              pool_ids, name_vec, name_mat, k, chunk))
            else:
                _register(name, pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"]))
        elif name.startswith("address_tfidf_top"):
            try:
                k = int(name.rsplit("top", 1)[1])
            except ValueError:
                continue
            if s1_addr_texts and addr_vec is not None:
                _register(name, _topk_blocker(s1_addr_texts, s1_ids,
                                              pool_ids, addr_vec, addr_mat, k, chunk))
            else:
                _register(name, pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"]))

    # union of all blockers
    if blockers:
        union_pairs = pd.concat(list(blockers.values()), ignore_index=True).drop_duplicates().reset_index(drop=True)
    else:
        union_pairs = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
    blockers["union_all"] = union_pairs
    truncations["union_all"] = 0

    # metrics (sampled closed world: recall measured fairly, burden is per-sampled-S1)
    n_space = len(s1_ids) * len(pool_ids)
    pos = positives_small.reset_index(drop=True) if positives_small is not None else pd.DataFrame(
        columns=["source1_entity_id", "candidate_entity_id", "source_pair"])
    if not pos.empty:
        pos = pos.sort_values(["source1_entity_id", "candidate_entity_id"]).reset_index(drop=True)
    pos_s2 = pos[pos.source_pair == "S1_S2"] if not pos.empty else pos
    pos_s3 = pos[pos.source_pair == "S1_S3"] if not pos.empty else pos
    metric_rows: List[Dict[str, object]] = []
    coverage = pos.copy() if not pos.empty else pd.DataFrame(
        columns=["source1_entity_id", "candidate_entity_id", "source_pair"])
    for name, pairs in blockers.items():
        if not pairs.empty and not all_s1.empty:
            cnt = all_s1.merge(
                pairs.groupby("source1_entity_id").size().rename("c"), on="source1_entity_id", how="left"
            )["c"].fillna(0).to_numpy(dtype=float)
        else:
            cnt = np.zeros(len(s1_ids), dtype=float)
        hit = _mark_retrieved(pos, pairs)
        hit_s2 = _mark_retrieved(pos_s2, pairs)
        hit_s3 = _mark_retrieved(pos_s3, pairs)
        metric_rows.append({
            "blocker": name,
            "n_candidate_pairs": int(len(pairs)),
            "s1_s2_recall": round(float(hit_s2.mean()) if len(pos_s2) else 0.0, 6),
            "s1_s3_recall": round(float(hit_s3.mean()) if len(pos_s3) else 0.0, 6),
            "overall_recall": round(float(hit.mean()) if len(pos) else 0.0, 6),
            "avg_candidates_per_s1": round(float(cnt.mean()) if len(cnt) else 0.0, 3),
            "p50_candidates_per_s1": round(float(np.median(cnt)) if len(cnt) else 0.0, 3),
            "p95_candidates_per_s1": round(float(np.percentile(cnt, 95)) if len(cnt) else 0.0, 3),
            "max_candidates_per_s1": int(cnt.max()) if len(cnt) else 0,
            "candidate_burden": round(_pct(len(pairs), n_space), 8),
            "s1_with_zero_candidates": int((cnt == 0).sum()),
            "truncated_s1": int(truncations.get(name, 0)),
        })
        if not pos.empty:
            coverage[f"hit_{name}"] = hit.astype(int)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_metrics, index=False)

    # rescue map: which blocker(s) retrieved each positive
    if not pos.empty:
        hit_cols = [c for c in coverage.columns if c.startswith("hit_") and c != "hit_union_all"]
        coverage["n_blockers_hit"] = coverage[hit_cols].sum(axis=1) if hit_cols else 0
        only: List[str] = []
        for _, r in coverage[hit_cols].iterrows() if hit_cols else []:
            hits = [c[4:] for c in hit_cols if r[c] == 1]
            only.append(hits[0] if len(hits) == 1 else ("multiple" if hits else "none"))
        coverage["rescued_only_by"] = only if hit_cols else "none"
    else:
        coverage["n_blockers_hit"] = []
        coverage["rescued_only_by"] = []
    coverage.to_csv(out_coverage, index=False)

    artifacts = [str(out_metrics), str(out_coverage)]
    # recall vs burden figure
    if not metrics.empty:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        xs = metrics["avg_candidates_per_s1"].to_numpy(dtype=float)
        ys = metrics["overall_recall"].to_numpy(dtype=float)
        xs_plot = np.where(xs <= 0, 0.4, xs)  # zero-candidate blockers stay visible on log axis
        ax.scatter(xs_plot, ys, s=80)
        for xp, (_, r) in zip(xs_plot, metrics.iterrows()):
            ax.annotate(str(r["blocker"]), (float(xp), float(r["overall_recall"])),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("avg candidates per S1 (log scale)")
        ax.set_ylabel("overall positive-pair recall")
        ax.set_title("Blocking recall vs candidate burden (train)")
        ax.set_ylim(-0.03, 1.03)
        artifacts.append(_savefig(fig, cfg.figures_dir / "blocking_recall_vs_burden.png", cfg))

    rescue_counts: Dict[str, int] = {}
    if "rescued_only_by" in coverage.columns and not coverage.empty:
        rescue_counts = {str(k): int(v) for k, v in coverage["rescued_only_by"].value_counts().items()}

    return {
        "metrics_csv": str(out_metrics),
        "coverage_csv": str(out_coverage),
        "artifacts": artifacts,
        "metrics": metrics.to_dict(orient="records"),
        "rescue_counts": rescue_counts,
        "union_pairs": union_pairs,  # passed to graph diagnostics (not serialized here)
        "s1_small": s1_df,  # SMALL normalized lookups for graph + casebook
        "pool_small": pool_df,
        "anchor_truth": anchor_truth,
        "positives_small": pos,
        "n_s1": len(s1_ids),
        "n_pool": len(pool_ids),
        "n_positives_sampled": int(len(pos)),
        "n_anchors_sampled": int(len(anchor_ids)),
    }


# ===========================================================================
# 7. CANDIDATE GRAPH DIAGNOSTICS -> 09_candidate_graph_diagnostics.csv
# ===========================================================================

def run_graph_diagnostics(
    candidates_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    cfg: AppConfig,
) -> Dict[str, Any]:
    """Bipartite S1<->candidate diagnostics (on SAMPLED blocking union + SMALL frames)."""
    cols = cfg.columns
    c_id, c_name, c_cty = cols["entity_id"], cols["business_name"], cols["country"]
    out_csv = cfg.eda_dir / "09_candidate_graph_diagnostics.csv"
    artifacts = [str(out_csv)]
    _apply_style(cfg)

    if s1_df is None or s1_df.empty or c_id not in s1_df.columns:
        s1_ids: List[str] = []
    else:
        s1_ids = s1_df[c_id].astype(str).tolist()
    if pool_df is None or pool_df.empty or c_id not in pool_df.columns:
        pool_ids: List[str] = []
    else:
        pool_ids = pool_df[c_id].astype(str).tolist()
    edges = candidates_df.copy() if candidates_df is not None else pd.DataFrame(
        columns=["source1_entity_id", "candidate_entity_id"])
    truncated_for_graph = 0
    # safety: bound total edges for component analysis (deterministic per-S1 truncation)
    max_edges = 6_000_000
    if len(edges) > max_edges:
        per_s1 = max(10, max_edges // max(1, len(s1_ids)))
        edges, truncated_for_graph = _cap_per_s1(
            edges.sort_values(["source1_entity_id", "candidate_entity_id"]), per_s1)
    edges = edges[edges["source1_entity_id"].isin(set(s1_ids))].reset_index(drop=True)

    s1_deg = edges.groupby("source1_entity_id").size()
    s1_deg_full = pd.Series(0, index=pd.Index(s1_ids, name="source1_entity_id"))
    s1_deg_full.update(s1_deg)
    cand_deg = edges.groupby("candidate_entity_id").size()

    try:
        s1_country = dict(zip(s1_df[c_id].astype(str), s1_df[c_cty].astype(str))) \
            if (s1_df is not None and not s1_df.empty and c_id in s1_df.columns and c_cty in s1_df.columns) else {}
        cand_country = dict(zip(pool_df[c_id].astype(str), pool_df[c_cty].astype(str))) \
            if (pool_df is not None and not pool_df.empty and c_id in pool_df.columns and c_cty in pool_df.columns) else {}
        cand_name = dict(zip(pool_df[c_id].astype(str), pool_df[c_name].astype(str))) \
            if (pool_df is not None and not pool_df.empty and c_id in pool_df.columns and c_name in pool_df.columns) else {}
    except Exception:
        s1_country, cand_country, cand_name = {}, {}, {}
    cand_source = {cid: ("S2" if cid.startswith("S2-") else ("S3" if cid.startswith("S3-") else "OTHER"))
                   for cid in pool_ids}

    rows: List[Dict[str, object]] = []

    def add(section: str, group: str, metric: str, value: object) -> None:
        rows.append({"section": section, "group": group, "metric": metric, "value": value})

    d = s1_deg_full.to_numpy(dtype=float)
    add("s1_degree", "all", "n_s1", len(s1_ids))
    add("s1_degree", "all", "mean", round(float(d.mean()) if len(d) else 0.0, 3))
    add("s1_degree", "all", "p50", round(float(np.median(d)) if len(d) else 0.0, 3))
    add("s1_degree", "all", "p95", round(float(np.percentile(d, 95)) if len(d) else 0.0, 3))
    add("s1_degree", "all", "max", int(d.max()) if len(d) else 0)
    add("s1_degree", "all", "n_zero", int((d == 0).sum()))
    dc = cand_deg.to_numpy(dtype=float)
    add("cand_degree", "all", "n_candidates_with_edges", int(len(cand_deg)))
    add("cand_degree", "all", "n_pool_total", len(pool_ids))
    add("cand_degree", "all", "mean", round(float(dc.mean()) if len(dc) else 0.0, 3))
    add("cand_degree", "all", "p95", round(float(np.percentile(dc, 95)) if len(dc) else 0.0, 3))
    add("cand_degree", "all", "max", int(dc.max()) if len(dc) else 0)
    add("edges", "all", "n_edges", int(len(edges)))
    add("edges", "all", "truncated_s1_for_graph", int(truncated_for_graph))

    # degree by country / source
    s1_by_cty: Dict[str, List[float]] = {}
    for sid, deg in zip(s1_ids, d):
        s1_by_cty.setdefault(s1_country.get(sid, ""), []).append(float(deg))
    for cty, vals in sorted(s1_by_cty.items()):
        add("s1_degree_by_country", cty or "(blank)", "mean", round(float(np.mean(vals)), 3))
        add("s1_degree_by_country", cty or "(blank)", "max", int(max(vals)) if vals else 0)
    cand_by_src: Dict[str, List[float]] = {}
    cand_by_cty: Dict[str, List[float]] = {}
    for cid, deg in cand_deg.items():
        cand_by_src.setdefault(cand_source.get(cid, "OTHER"), []).append(float(deg))
        cand_by_cty.setdefault(cand_country.get(cid, ""), []).append(float(deg))
    for src, vals in sorted(cand_by_src.items()):
        add("cand_degree_by_source", src, "mean", round(float(np.mean(vals)), 3))
        add("cand_degree_by_source", src, "max", int(max(vals)) if vals else 0)
    for cty, vals in sorted(cand_by_cty.items()):
        add("cand_degree_by_country", cty or "(blank)", "mean", round(float(np.mean(vals)), 3))
        add("cand_degree_by_country", cty or "(blank)", "max", int(max(vals)) if vals else 0)

    # hubs
    top_cand = cand_deg.sort_values(ascending=False).head(20)
    for cid, deg in top_cand.items():
        add("hub_candidate", cid,
            f"degree={int(deg)} src={cand_source.get(cid)} country={cand_country.get(cid)}",
            (cand_name.get(cid, "") or "")[:120])
    top_s1 = s1_deg.sort_values(ascending=False).head(10)
    for sid, deg in top_s1.items():
        add("hub_s1", sid, f"degree={int(deg)} country={s1_country.get(sid)}", "")

    # connected components (diagnostic only — never used as predictions)
    comp_info: Dict[str, Any] = {"computed": False}
    if _HAS_SCIPY and len(edges):
        try:
            left_index = {sid: i for i, sid in enumerate(s1_ids)}
            right_index = {cid: len(s1_ids) + j for j, cid in enumerate(pool_ids)}
            r = edges["source1_entity_id"].map(left_index)
            c = edges["candidate_entity_id"].map(right_index)
            keep = r.notna() & c.notna()
            rr = r[keep].to_numpy(dtype=np.int64)
            cc = c[keep].to_numpy(dtype=np.int64)
            n_nodes = len(s1_ids) + len(pool_ids)
            data = np.ones(len(rr) * 2, dtype=np.float32)
            rows_i = np.concatenate([rr, cc])
            cols_j = np.concatenate([cc, rr])
            adj = coo_matrix((data, (rows_i, cols_j)), shape=(n_nodes, n_nodes)).tocsr()
            n_comp, labels_cc = connected_components(adj, directed=False)
            sizes = Counter(labels_cc.tolist())
            largest = max(sizes.values()) if sizes else 0
            comp_info = {"computed": True, "n_components": int(n_comp),
                         "largest_component": int(largest),
                         "n_singleton_components": int(sum(1 for v in sizes.values() if v == 1))}
            add("components", "all", "n_components", int(n_comp))
            add("components", "all", "largest_component_nodes", int(largest))
            add("components", "all", "singleton_components", int(comp_info["n_singleton_components"]))
            for i, (lab, sz) in enumerate(sorted(sizes.items(), key=lambda kv: -kv[1])[:10]):
                add("components", f"rank_{i+1}", "size_nodes", int(sz))
        except Exception as exc:  # pragma: no cover
            add("components", "all", "error", str(exc)[:200])
    else:
        add("components", "all", "note", "scipy unavailable or no edges — skipped")

    pd.DataFrame(rows, columns=["section", "group", "metric", "value"]).to_csv(out_csv, index=False)

    # degree figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].hist(d, bins=50, log=True)
    axes[0].set_title("Candidates per S1 (union blockers)")
    axes[0].set_xlabel("degree")
    axes[0].set_ylabel("count (log)")
    axes[1].hist(dc if len(dc) else [0], bins=50, log=True)
    axes[1].set_title("S1 neighbours per candidate")
    axes[1].set_xlabel("degree")
    artifacts.append(_savefig(fig, cfg.figures_dir / "candidate_degree_distribution.png", cfg))

    return {
        "graph_csv": str(out_csv),
        "artifacts": artifacts,
        "mean_s1_degree": round(float(d.mean()) if len(d) else 0.0, 3),
        "max_s1_degree": int(d.max()) if len(d) else 0,
        "max_cand_degree": int(dc.max()) if len(dc) else 0,
        "components": comp_info,
        "top_hub": {"id": str(top_cand.index[0]), "degree": int(top_cand.iloc[0])} if len(top_cand) else {},
    }


# ===========================================================================
# 8. COUNTRY SHIFT -> 08_country_shift_report.csv (+ country_distribution.png)
# ===========================================================================

def _char_ngrams(text: str, n: int) -> List[str]:
    t = (text or "").lower()
    if len(t) < n:
        return [t] if t.strip() else []
    return [t[i:i + n] for i in range(len(t) - n + 1)]


def _sample_group_indices(n: int, per_group: int, rng: np.random.RandomState) -> np.ndarray:
    if n <= per_group:
        return np.arange(n, dtype=np.int64)
    return np.array(sorted(rng.choice(n, int(per_group), replace=False).tolist()), dtype=np.int64)


def run_country_shift_eda(
    train_tables: Dict[str, Optional[pd.DataFrame]],
    test_tables: Dict[str, Optional[pd.DataFrame]],
    cfg: AppConfig,
) -> Dict[str, Any]:
    """Train vs test shift (FULL counts/missing, SAMPLED string stats + vocab).

    Full-data (vectorized): ``n``, missing rates, and country counts for the
    figure. Sampled: all string-stat means + char/token vocab coverage on
    ``country_shift_per_group`` rows per table x country. Adds ``n_sampled``.
    Country stays open-set throughout.
    """
    _apply_style(cfg)
    cols = cfg.columns
    c_name, c_addr, c_cty = cols["business_name"], cols["business_address"], cols["country"]
    ngram_n = int(cfg.eda.get("country_shift", {}).get("char_ngram_n", 3))
    char_cap = int(cfg.eda.get("country_shift", {}).get("max_vocab_chars", 4000000))
    per_group = int(_sampling_cfg(cfg).get("country_shift_per_group", 20000))

    tables: Dict[str, Optional[pd.DataFrame]] = {
        "train_source1": train_tables.get("train_s1"),
        "train_source2": train_tables.get("train_s2"),
        "train_source3": train_tables.get("train_s3"),
        "test_source1": test_tables.get("test_s1"),
        "test_source2": test_tables.get("test_s2"),
        "test_source3": test_tables.get("test_s3"),
    }
    table_order = ["train_source1", "train_source2", "train_source3",
                   "test_source1", "test_source2", "test_source3"]

    # ---- FULL-data per-country n + missing rates (vectorized) ----
    full_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
    dist_rows: List[Dict[str, object]] = []
    for table in table_order:
        df = tables.get(table)
        if df is None or df.empty:
            continue
        cty = df[c_cty].astype(str) if c_cty in df.columns else pd.Series(
            [""] * len(df), index=df.index)
        if c_name in df.columns:
            name_empty = (df[c_name].astype(str).str.strip() == "")
        else:
            name_empty = pd.Series(np.zeros(len(df), dtype=bool), index=df.index)
        if c_addr in df.columns:
            addr_empty = (df[c_addr].astype(str).str.strip() == "")
        else:
            addr_empty = pd.Series(np.zeros(len(df), dtype=bool), index=df.index)
        frame = pd.DataFrame({"_cty": cty, "_ne": name_empty, "_ae": addr_empty})
        agg = frame.groupby("_cty", dropna=False).agg(
            n=("_ne", "size"), miss_name=("_ne", "mean"), miss_addr=("_ae", "mean"))
        per_cty: Dict[str, Dict[str, Any]] = {}
        for cty_val, row in agg.iterrows():
            label = str(cty_val)
            per_cty[label] = {"n": int(row["n"]),
                              "missing_name_rate": round(float(row["miss_name"]), 6),
                              "missing_addr_rate": round(float(row["miss_addr"]), 6)}
            dist_rows.append({"table": table, "country": label, "n": int(row["n"])})
        full_stats[table] = per_cty

    # ---- SAMPLED train vocabs (bounded, deterministic per-group sampling) ----
    train_char_vocab: set = set()
    train_tok_vocab: set = set()
    used_chars = 0
    for table in ("train_source1", "train_source2", "train_source3"):
        df = tables.get(table)
        if df is None or df.empty or table not in full_stats:
            continue
        cty = df[c_cty].astype(str).to_numpy() if c_cty in df.columns else np.array([""] * len(df))
        names = df[c_name].astype(str).to_numpy() if c_name in df.columns else np.array([""] * len(df))
        addrs = df[c_addr].astype(str).to_numpy() if c_addr in df.columns else np.array([""] * len(df))
        for country in sorted(full_stats[table].keys()):
            if used_chars >= char_cap:
                break
            pos = np.where(cty == country)[0]
            if len(pos) == 0:
                continue
            rng = _rng(cfg, salt=_stable_salt(cfg, f"{table}::{country}::vocab", base=501))
            take = _sample_group_indices(len(pos), per_group, rng)
            sel = pos[take]
            for i in sel:
                if used_chars >= char_cap:
                    break
                blob = f"{names[i]} {addrs[i]}"
                used_chars += len(blob)
                train_char_vocab.update(_char_ngrams(blob, ngram_n))
                train_tok_vocab.update(t for t in str(blob).lower().split() if t)
            del pos, take, sel
        del cty, names, addrs
        if used_chars >= char_cap:
            break

    # ---- per-group rows: FULL n/missing + SAMPLED means/coverage ----
    rows: List[Dict[str, object]] = []
    for table in table_order:
        df = tables.get(table)
        if df is None or df.empty or table not in full_stats:
            continue
        cty = df[c_cty].astype(str).to_numpy() if c_cty in df.columns else np.array([""] * len(df))
        names = df[c_name].astype(str).to_numpy() if c_name in df.columns else np.array([""] * len(df))
        addrs = df[c_addr].astype(str).to_numpy() if c_addr in df.columns else np.array([""] * len(df))
        for country in sorted(full_stats[table].keys()):
            pos = np.where(cty == country)[0]
            rng = _rng(cfg, salt=_stable_salt(cfg, f"{table}::{country}::stats", base=1501))
            take = _sample_group_indices(len(pos), per_group, rng)
            sel = pos[take]
            n_sampled = int(len(sel))
            samp_names = [str(names[i]) for i in sel] if n_sampled else []
            samp_addrs = [str(addrs[i]) for i in sel] if n_sampled else []
            name_stats = [string_stats(v) for v in samp_names]
            addr_stats = [string_stats(v) for v in samp_addrs]

            def _mean(key_: str, arr: List[Dict[str, float]]) -> float:
                return round(float(np.mean([d[key_] for d in arr])) if arr else 0.0, 4)

            c_vocab: set = set()
            t_vocab: set = set()
            for a, b in zip(samp_names, samp_addrs):
                blob = f"{a} {b}"
                c_vocab.update(_char_ngrams(blob, ngram_n))
                t_vocab.update(t for t in str(blob).lower().split() if t)
            char_cov = _pct(len(c_vocab & train_char_vocab), len(c_vocab)) if c_vocab else 1.0
            tok_cov = _pct(len(t_vocab & train_tok_vocab), len(t_vocab)) if t_vocab else 1.0
            full = full_stats[table][country]
            rows.append({
                "table": table,
                "country": country,
                "n": int(full["n"]),
                "n_sampled": int(n_sampled),
                "name_len_mean": _mean("char_len", name_stats),
                "addr_len_mean": _mean("char_len", addr_stats),
                "name_digit_mean": _mean("digit_count", name_stats),
                "addr_digit_mean": _mean("digit_count", addr_stats),
                "name_punct_mean": _mean("punct_count", name_stats),
                "addr_punct_mean": _mean("punct_count", addr_stats),
                "name_nonascii_mean": _mean("non_ascii_ratio", name_stats),
                "addr_nonascii_mean": _mean("non_ascii_ratio", addr_stats),
                "name_tok_mean": _mean("token_count", name_stats),
                "addr_tok_mean": _mean("token_count", addr_stats),
                "addr_numtok_mean": _mean("numeric_token_count", addr_stats),
                "missing_name_rate": float(full["missing_name_rate"]),
                "missing_addr_rate": float(full["missing_addr_rate"]),
                "char_ngram_coverage_vs_train": round(char_cov, 6),
                "char_ngram_oov_vs_train": round(1.0 - char_cov, 6),
                "token_coverage_vs_train": round(tok_cov, 6),
            })
            del pos, take, sel
        del cty, names, addrs
    rows.sort(key=lambda r: (str(r["table"]), str(r["country"])))

    cols_out = ["table", "country", "n", "n_sampled", "name_len_mean", "addr_len_mean",
                "name_digit_mean", "addr_digit_mean", "name_punct_mean", "addr_punct_mean",
                "name_nonascii_mean", "addr_nonascii_mean", "name_tok_mean", "addr_tok_mean",
                "addr_numtok_mean", "missing_name_rate", "missing_addr_rate",
                "char_ngram_coverage_vs_train", "char_ngram_oov_vs_train",
                "token_coverage_vs_train"]
    report = pd.DataFrame(rows, columns=cols_out) if rows else pd.DataFrame(columns=cols_out)
    out_csv = cfg.eda_dir / "08_country_shift_report.csv"
    report.to_csv(out_csv, index=False)
    artifacts = [str(out_csv)]

    # country distribution figure from FULL counts (open set: whatever labels exist)
    if dist_rows:
        dist = pd.DataFrame(dist_rows)
        pivot = dist.pivot_table(index="table", columns="country", values="n", aggfunc="sum", fill_value=0)
        order = [t for t in table_order if t in pivot.index]
        pivot = pivot.loc[order]
        fig, ax = plt.subplots(figsize=(10, 4.6))
        pivot.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Country composition per table (raw labels, open set)")
        ax.set_ylabel("rows")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        ax.legend(title="country", fontsize=8)
        artifacts.append(_savefig(fig, cfg.figures_dir / "country_distribution.png", cfg))

    train_countries = sorted({str(r["country"]) for r in rows if str(r["table"]).startswith("train_")})
    test_countries = sorted({str(r["country"]) for r in rows if str(r["table"]).startswith("test_")})
    unseen = sorted(set(test_countries) - set(train_countries))
    return {
        "shift_csv": str(out_csv),
        "artifacts": artifacts,
        "train_countries": train_countries,
        "test_countries": test_countries,
        "unseen_in_train": unseen,
        "n_train_char_vocab": len(train_char_vocab),
        "n_train_tok_vocab": len(train_tok_vocab),
        "per_group_config": int(per_group),
        "n_groups": int(len(rows)),
    }


# ===========================================================================
# 9. CASEBOOK -> 10_casebook_train_pairs.html
# ===========================================================================

# ===========================================================================
# 11. MULTILINGUAL / MULTI-SCRIPT EDA (§28) -> 11_*, 12_*, 13_*, 14_*, 15_*, 16_*
# ===========================================================================
#
# Approach: keep MULTIPLE representations side by side and compare their
# behavior on the SAME sampled data — (a) raw (authoritative, never
# overwritten), (b) unicode-norm (NFKC), (c) conservative script-aware norm
# (normalize_basic, Unicode-preserving), (d) transliteration (ADDITIONAL
# feature only, never canonical), (e) character-level (stats + n-grams).
# Transliteration is judged by its recall-vs-risk tradeoff and is NEVER
# assumed beneficial just because it creates more matches.

_NONASCII_RE = r"[^\x00-\x7F]"
_SCRIPT_PLOT_COLORS = {
    "Latin": "#4C78A8",
    "Devanagari": "#F58518",
    "Cyrillic": "#54A24B",
    "Arabic": "#E45756",
    "mixed": "#79706E",
    "other Unicode": "#BAB0AC",
    "empty/missing": "#DCDCDC",
}
_ML_TABLE_ORDER = ("train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3")
_ML_TABLE_LABELS = {
    "train_s1": "train_source1", "train_s2": "train_source2",
    "train_s3": "train_source3", "test_s1": "test_source1",
    "test_s2": "test_source2", "test_s3": "test_source3",
}
_ML_PAIR_COLUMNS = (
    "s1_name_script", "cand_name_script", "name_script_bucket",
    "translit_name_agree_exact", "translit_name_ratio", "translit_addr_ratio",
)
_HARD_NEG_TYPES = ("name_hard", "address_hard", "hybrid_hard")


def _multilingual_cfg(cfg: AppConfig) -> Dict[str, Any]:
    return dict(cfg.eda.get("multilingual", {}) or {})


def _script_category_series(values: pd.Series) -> pd.Series:
    """Vectorized script categories (C-speed regex; mirrors per-row detect).

    Missing values (-> "") become "empty/missing"; pure digit/punct ASCII
    becomes "Latin" (documented scriptless rule). Never raises on odd input.
    """
    s = values.fillna("").astype(str)
    if s.empty:
        return pd.Series([], dtype=object, index=values.index)
    blank = (s.str.strip() == "").to_numpy()
    nonascii = s.str.contains(_NONASCII_RE, regex=True, na=False).to_numpy()
    has_latin = s.str.contains(SCRIPT_REGEX["Latin"], regex=True, na=False).to_numpy()
    has_deva = s.str.contains(SCRIPT_REGEX["Devanagari"], regex=True, na=False).to_numpy()
    has_cyrl = s.str.contains(SCRIPT_REGEX["Cyrillic"], regex=True, na=False).to_numpy()
    has_arab = s.str.contains(SCRIPT_REGEX["Arabic"], regex=True, na=False).to_numpy()
    has_other = s.str.contains(SCRIPT_REGEX["other"], regex=True, na=False).to_numpy()
    n = (has_latin.astype(np.int8) + has_deva.astype(np.int8)
         + has_cyrl.astype(np.int8) + has_arab.astype(np.int8)
         + has_other.astype(np.int8))
    cats = np.select(
        [
            blank,
            n >= 2,
            has_latin & (n == 1),
            has_deva & (n == 1),
            has_cyrl & (n == 1),
            has_arab & (n == 1),
            has_other & (n == 1),
            (n == 0) & ~nonascii,
        ],
        [
            "empty/missing", "mixed", "Latin", "Devanagari", "Cyrillic",
            "Arabic", "other Unicode", "Latin",
        ],
        default="other Unicode",
    )
    return pd.Series(cats, index=values.index)


# ---------------------------------------------------------------------------
# 11. Script distribution (FULL data) -> 11_script_distribution.csv
# ---------------------------------------------------------------------------

def run_script_distribution_eda(
    tables: Dict[str, Optional[pd.DataFrame]],
    cfg: AppConfig,
) -> Dict[str, Any]:
    """FULL-data script mix per source x field (+ stacked figure)."""
    cols = cfg.columns
    c_name, c_addr = cols["business_name"], cols["business_address"]
    fields = ((c_name, "business_name"), (c_addr, "business_address"))
    rows: List[Dict[str, object]] = []
    for key in _ML_TABLE_ORDER:
        df = (tables or {}).get(key)
        if df is None or df.empty:
            continue
        n = len(df)
        for col, field in fields:
            if col not in df.columns:
                continue
            cats = _script_category_series(df[col])
            vc = cats.value_counts()
            for cat in SCRIPT_CATEGORIES:
                cnt = int(vc.get(cat, 0))
                rows.append({
                    "source": _ML_TABLE_LABELS[key],
                    "field": field,
                    "script_category": cat,
                    "row_count": cnt,
                    "percentage": round(cnt / n, 6) if n else 0.0,
                })
            del cats
        gc.collect()
    report = pd.DataFrame(
        rows,
        columns=["source", "field", "script_category", "row_count", "percentage"],
    )
    out_csv = cfg.eda_dir / "11_script_distribution.csv"
    report.to_csv(out_csv, index=False)
    artifacts = [str(out_csv)]

    # figure: stacked percentage bars per source, one panel per field
    _apply_style(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, field in zip(axes, ("business_name", "business_address")):
        sub = report[report.field == field]
        sources = [s for s in
                   ("train_source1", "train_source2", "train_source3",
                    "test_source1", "test_source2", "test_source3")
                   if s in set(sub.source.tolist())]
        bottom = np.zeros(len(sources))
        plotted = False
        for cat in SCRIPT_CATEGORIES:
            vals = np.array([
                float(sub[(sub.source == s) & (sub.script_category == cat)]["percentage"].sum())
                for s in sources
            ])
            if (vals > 0).any():
                plotted = True
            ax.bar(sources, vals * 100.0, bottom=bottom,
                   label=cat, color=_SCRIPT_PLOT_COLORS.get(cat))
            bottom = bottom + vals * 100.0
        ax.set_title(f"Script mix: {field} (FULL data)")
        ax.set_ylabel("% of rows")
        ax.set_xticklabels(sources, rotation=25, ha="right")
        if not plotted:
            ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
    fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    artifacts.append(_savefig(fig, cfg.figures_dir / "script_distribution.png", cfg))

    # non-Latin share among ALL rows per source x field (for the summary:
    # what a Latin-only approach could not even see)
    non_latin_rates: Dict[str, Dict[str, float]] = {}
    non_latin_cats = {"Devanagari", "Cyrillic", "Arabic", "mixed", "other Unicode"}
    for (source, field), grp in report.groupby(["source", "field"]):
        share = float(grp[grp.script_category.isin(non_latin_cats)]["percentage"].sum())
        non_latin_rates.setdefault(str(source), {})[str(field)] = round(share, 6)
    return {
        "script_csv": str(out_csv),
        "artifacts": artifacts,
        "non_latin_rates": non_latin_rates,
    }


# ---------------------------------------------------------------------------
# 14. Character-level statistics (FULL n + FULL non-ASCII rates; SAMPLED rest)
# ---------------------------------------------------------------------------

def _extended_char_stats(text: str) -> Dict[str, float]:
    n = len(text)
    n_alpha = sum(1 for ch in text if ch.isalpha())
    n_digit = sum(1 for ch in text if ch.isdigit())
    n_space = sum(1 for ch in text if ch.isspace())
    n_punct = n - n_alpha - n_digit - n_space  # remainder partition
    return {
        "len": float(n),
        "alpha": float(n_alpha),
        "digit": float(n_digit),
        "punct": float(n_punct),
        "space": float(n_space),
    }


def run_character_stats_eda(
    train: Dict[str, Optional[pd.DataFrame]],
    test: Dict[str, Optional[pd.DataFrame]],
    cfg: AppConfig,
) -> Dict[str, Any]:
    """Per table x country: FULL n + non-ASCII rates; SAMPLED char stats.

    Also records sampled script shares and lowercase char-3-gram vocabulary
    sizes so S1/S2/S3 and train/test can be compared on representation load.
    """
    cols = cfg.columns
    c_name, c_addr, c_cty = cols["business_name"], cols["business_address"], cols["country"]
    per_group = int(_sampling_cfg(cfg).get("country_shift_per_group", 20000))
    tables: List[Tuple[str, Optional[pd.DataFrame]]] = [
        ("train_source1", (train or {}).get("train_s1")),
        ("train_source2", (train or {}).get("train_s2")),
        ("train_source3", (train or {}).get("train_s3")),
        ("test_source1", (test or {}).get("test_s1")),
        ("test_source2", (test or {}).get("test_s2")),
        ("test_source3", (test or {}).get("test_s3")),
    ]
    rows: List[Dict[str, object]] = []
    actual: Dict[str, int] = {}
    for table_idx, (label, df) in enumerate(tables):
        if df is None or df.empty:
            continue
        if c_name not in df.columns and c_addr not in df.columns:
            logger.warning("Character-stats EDA skipping %s (no name/address columns).", label)
            continue
        name_s = df[c_name].fillna("").astype(str) if c_name in df.columns else pd.Series([""] * len(df))
        addr_s = df[c_addr].fillna("").astype(str) if c_addr in df.columns else pd.Series([""] * len(df))
        if c_cty in df.columns:
            cty_s = df[c_cty].fillna("").astype(str)
        else:
            cty_s = pd.Series(["__missing_country_column__"] * len(df))
        counts = cty_s.value_counts()
        # FULL-data non-ASCII rates per country (vectorized groupby)
        try:
            name_na = name_s.str.contains(_NONASCII_RE, regex=True, na=False).groupby(cty_s).mean()
            addr_na = addr_s.str.contains(_NONASCII_RE, regex=True, na=False).groupby(cty_s).mean()
        except Exception:
            name_na = addr_na = pd.Series(dtype=float)
        for country in sorted(counts.index.tolist()):
            n_full = int(counts.loc[country])
            mask = (cty_s == country).to_numpy()
            positions = np.flatnonzero(mask)
            rng = _rng(cfg, salt=_stable_salt(cfg, f"chstats::{label}::{country}", base=800 + table_idx))
            if len(positions) > per_group:
                sel = np.sort(rng.choice(positions, per_group, replace=False))
            else:
                sel = positions
            samp_names = name_s.iloc[sel]
            samp_addrs = addr_s.iloc[sel]
            name_scripts = _script_category_series(samp_names).value_counts(normalize=True)
            addr_scripts = _script_category_series(samp_addrs).value_counts(normalize=True)
            n_len = a_len = n_alpha = n_digit = n_punct = n_space = 0.0
            a_alpha = a_digit = a_punct = a_space = 0.0
            name_chars: Set[str] = set()
            addr_chars: Set[str] = set()
            name_tri: Set[str] = set()
            addr_tri: Set[str] = set()
            for val in samp_names.tolist():
                st = _extended_char_stats(val)
                n_len += st["len"]; n_alpha += st["alpha"]; n_digit += st["digit"]
                n_punct += st["punct"]; n_space += st["space"]
                name_chars.update(val)
                name_tri.update(char_ngram_set(val.lower(), (3,)))
            for val in samp_addrs.tolist():
                st = _extended_char_stats(val)
                a_len += st["len"]; a_alpha += st["alpha"]; a_digit += st["digit"]
                a_punct += st["punct"]; a_space += st["space"]
                addr_chars.update(val)
                addr_tri.update(char_ngram_set(val.lower(), (3,)))
            m = float(len(sel)) or 1.0
            script_cols: Dict[str, object] = {}
            for prefix, shares in (("name", name_scripts), ("addr", addr_scripts)):
                script_cols[f"{prefix}_latin_share"] = round(float(shares.get("Latin", 0.0)), 6)
                script_cols[f"{prefix}_deva_share"] = round(float(shares.get("Devanagari", 0.0)), 6)
                script_cols[f"{prefix}_cyrl_share"] = round(float(shares.get("Cyrillic", 0.0)), 6)
                script_cols[f"{prefix}_arab_share"] = round(float(shares.get("Arabic", 0.0)), 6)
                script_cols[f"{prefix}_mixed_share"] = round(float(shares.get("mixed", 0.0)), 6)
                script_cols[f"{prefix}_other_share"] = round(float(shares.get("other Unicode", 0.0)), 6)
            rows.append({
                "table": label,
                "country": country,
                "n": n_full,
                "n_sampled": int(len(sel)),
                "name_nonascii_rate": round(float(name_na.get(country, 0.0)), 6),
                "addr_nonascii_rate": round(float(addr_na.get(country, 0.0)), 6),
                "name_len_mean": round(n_len / m, 3),
                "addr_len_mean": round(a_len / m, 3),
                "name_alpha_mean": round(n_alpha / m, 3),
                "name_digit_mean": round(n_digit / m, 3),
                "name_punct_mean": round(n_punct / m, 3),
                "name_space_mean": round(n_space / m, 3),
                "addr_alpha_mean": round(a_alpha / m, 3),
                "addr_digit_mean": round(a_digit / m, 3),
                "addr_punct_mean": round(a_punct / m, 3),
                "addr_space_mean": round(a_space / m, 3),
                "name_unique_chars": len(name_chars),
                "addr_unique_chars": len(addr_chars),
                **script_cols,
                "name_trigram_vocab": len(name_tri),
                "addr_trigram_vocab": len(addr_tri),
            })
            actual[f"{label}::{country}"] = int(len(sel))
        del name_s, addr_s, cty_s
        gc.collect()
    columns = [
        "table", "country", "n", "n_sampled",
        "name_nonascii_rate", "addr_nonascii_rate",
        "name_len_mean", "addr_len_mean",
        "name_alpha_mean", "name_digit_mean", "name_punct_mean", "name_space_mean",
        "addr_alpha_mean", "addr_digit_mean", "addr_punct_mean", "addr_space_mean",
        "name_unique_chars", "addr_unique_chars",
        "name_latin_share", "name_deva_share", "name_cyrl_share", "name_arab_share",
        "name_mixed_share", "name_other_share",
        "addr_latin_share", "addr_deva_share", "addr_cyrl_share", "addr_arab_share",
        "addr_mixed_share", "addr_other_share",
        "name_trigram_vocab", "addr_trigram_vocab",
    ]
    report = pd.DataFrame(rows, columns=columns)
    if not report.empty:
        report = report.sort_values(["table", "country"]).reset_index(drop=True)
    out_csv = cfg.eda_dir / "14_character_statistics.csv"
    report.to_csv(out_csv, index=False)
    return {
        "char_stats_csv": str(out_csv),
        "artifacts": [str(out_csv)],
        "per_group_config": int(per_group),
        "actual": dict(actual),
    }


# ---------------------------------------------------------------------------
# 12. Transliteration collision study (SAMPLED, fair raw/norm/translit)
# ---------------------------------------------------------------------------

def run_transliteration_collision_eda(
    tables: Dict[str, Optional[pd.DataFrame]],
    cfg: AppConfig,
) -> Dict[str, Any]:
    """Compare raw vs conservative-norm vs transliterated on ONE sample.

    All three representations share the same sampled rows per table, so
    uniqueness differences are representation effects, not sampling effects.
    Counts cross-script collision groups (same value, mixed source scripts)
    as the false-merge risk of each representation. Transliteration is one
    more representation here — never canonical.
    """
    cols = cfg.columns
    c_name, c_addr = cols["business_name"], cols["business_address"]
    backend = resolve_transliteration_backend(
        _multilingual_cfg(cfg).get("transliteration_backend", "auto"))
    sample_n = int(_sampling_cfg(cfg).get("transliteration_sample", 500000))
    if backend == "none":
        logger.warning("Transliteration backend is 'none' — writing raw/norm sample rows only.")
    rows: List[Dict[str, object]] = []
    largest_cross: Dict[str, Any] = {}
    cluster_sizes: Dict[str, Dict[str, np.ndarray]] = {}
    actual: Dict[str, int] = {}
    for table_idx, key in enumerate(_ML_TABLE_ORDER):
        df = (tables or {}).get(key)
        if df is None or df.empty:
            continue
        if c_name not in df.columns or c_addr not in df.columns:
            logger.warning("Transliteration-collision EDA skipping %s (missing columns).", key)
            continue
        n = len(df)
        rng = _rng(cfg, salt=701 + table_idx)
        if n > sample_n:
            idx = sorted(rng.choice(n, sample_n, replace=False).tolist())
        else:
            idx = list(range(n))
        label = _ML_TABLE_LABELS[key]
        raw_name = df[c_name].fillna("").astype(str).iloc[idx]
        raw_addr = df[c_addr].fillna("").astype(str).iloc[idx]
        name_scripts = _script_category_series(raw_name)
        addr_scripts = _script_category_series(raw_addr)
        actual[label] = int(len(raw_name))
        reps: List[Tuple[str, Optional[pd.Series], pd.Series]] = [
            ("raw_name", raw_name, name_scripts),
            ("norm_name", raw_name.map(normalize_basic), name_scripts),
            ("raw_address", raw_addr, addr_scripts),
            ("norm_address", raw_addr.map(normalize_basic), addr_scripts),
        ]
        if backend != "none":
            reps.append(("translit_name",
                         raw_name.map(lambda s: transliterate_text(s, backend)),
                         name_scripts))
            reps.append(("translit_address",
                         raw_addr.map(lambda s: transliterate_text(s, backend)),
                         addr_scripts))
        for rep_name, vals, scripts in reps:
            assert vals is not None
            st = _collision_stats(vals)
            n_empty = int((vals == "").sum())
            # Cross-script groups: identical value from >= 2 script buckets.
            # (raw rows always score 0 — identical strings share one bucket —
            # but we compute uniformly for a fair table.)
            n_cross, largest_cross_size = 0, 0
            try:
                frame = pd.DataFrame({"v": vals.to_numpy(), "s": scripts.to_numpy()})
                grp = frame.groupby("v")["s"]
                sizes = grp.size()
                nunique = grp.nunique()
                mask = (sizes > 1) & (nunique > 1)
                n_cross = int(mask.sum())
                if n_cross:
                    largest_cross_size = int(sizes[mask].max())
                    if rep_name.startswith("translit_"):
                        top = sizes[mask].sort_values(ascending=False).head(3)
                        examples = []
                        for val_key, size in top.items():
                            mix = sorted(frame.loc[frame.v == val_key, "s"].unique().tolist())
                            examples.append({
                                "value_preview": str(val_key)[:80],
                                "size": int(size),
                                "scripts": mix,
                            })
                        largest_cross.setdefault(label, {})[rep_name] = examples
            except Exception as exc:
                logger.warning("Cross-script grouping failed for %s/%s: %s", label, rep_name, exc)
            rows.append({
                "table": label,
                "representation": rep_name,
                "backend": backend,
                "n_records": st["n_records"],
                "n_unique": st["n_unique"],
                "pct_unique": st["pct_unique"],
                "n_collision_groups": st["n_collision_groups"],
                "largest_cluster": st["largest_cluster"],
                "mean_cluster_size": st["mean_cluster_size"],
                "n_empty": n_empty,
                "n_cross_script_groups": n_cross,
                "largest_cross_script_group": largest_cross_size,
            })
            vc = vals.value_counts()
            groups = vc[vc > 1]
            cluster_sizes.setdefault(key, {})[rep_name] = (
                groups.to_numpy(dtype=float) if len(groups) else np.array([]))
        del raw_name, raw_addr, name_scripts, addr_scripts
        gc.collect()
    report = pd.DataFrame(
        rows,
        columns=["table", "representation", "backend", "n_records", "n_unique",
                 "pct_unique", "n_collision_groups", "largest_cluster",
                 "mean_cluster_size", "n_empty", "n_cross_script_groups",
                 "largest_cross_script_group"],
    )
    out_csv = cfg.eda_dir / "12_transliteration_collision_report.csv"
    report.to_csv(out_csv, index=False)
    artifacts = [str(out_csv)]

    # figure: pooled train collision-size ECDFs (raw vs norm vs translit)
    _apply_style(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    rep_sets = (
        ("raw_name", "norm_name", "translit_name"),
        ("raw_address", "norm_address", "translit_address"),
    )
    for ax, rep_set in zip(axes, rep_sets):
        plotted = False
        for rep in rep_set:
            pooled = np.concatenate([
                cluster_sizes.get(k, {}).get(rep, np.array([]))
                for k in ("train_s1", "train_s2", "train_s3")
            ]) if any(len(cluster_sizes.get(k, {}).get(rep, []))
                       for k in ("train_s1", "train_s2", "train_s3")) else np.array([])
            pooled = pooled[pooled > 1]
            if len(pooled):
                x, y = _ecdf(np.log10(pooled))
                ax.plot(x, y, label=f"{rep} (n={len(pooled)})")
                plotted = True
        ax.set_title(f"Collision sizes (train pooled): {rep_set[0].split('_')[-1]}")
        ax.set_xlabel("log10(cluster size)")
        if not plotted:
            ax.text(0.5, 0.5, "no collisions", ha="center", transform=ax.transAxes)
    axes[0].set_ylabel("ECDF")
    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    artifacts.append(_savefig(
        fig, cfg.figures_dir / "transliteration_collision_sizes.png", cfg))

    per_table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for (table, rep), grp in report.groupby(["table", "representation"]):
        per_table.setdefault(str(table), {})[str(rep)] = {
            "n_unique": float(grp["n_unique"].iloc[0]),
            "n_collision_groups": float(grp["n_collision_groups"].iloc[0]),
            "n_cross_script_groups": float(grp["n_cross_script_groups"].iloc[0]),
            "largest_cross_script_group": float(grp["largest_cross_script_group"].iloc[0]),
        }
    return {
        "translit_csv": str(out_csv),
        "artifacts": artifacts,
        "backend": backend,
        "sample_config": int(sample_n),
        "sample_actual": dict(actual),
        "per_table": per_table,
        "largest_cross_script_examples": largest_cross,
    }


# ---------------------------------------------------------------------------
# Shared pair annotation for §28.5-§28.8 (sampled pairs only, in place)
# ---------------------------------------------------------------------------

def _ensure_multilingual_pair_columns(
    df: Optional[pd.DataFrame],
    cfg: AppConfig,
) -> Optional[pd.DataFrame]:
    """Add script + transliteration-agreement columns to a pair frame.

    Mutates in place (shared across sections 13/15/16 + casebook so the work
    happens once). No-op on empty frames or when columns already exist.
    Transliteration here is an agreement MEASURE on raw strings, never a
    replacement representation.
    """
    if df is None or df.empty:
        return df
    if all(c in df.columns for c in _ML_PAIR_COLUMNS):
        return df
    backend = resolve_transliteration_backend(
        _multilingual_cfg(cfg).get("transliteration_backend", "auto"))
    get = lambda c: df[c].fillna("").astype(str) if c in df.columns else pd.Series([""] * len(df))
    s1_raw, cand_raw = get("s1_name_raw"), get("cand_name_raw")
    s1_addr, cand_addr = get("s1_addr_raw"), get("cand_addr_raw")
    s1_script = _script_category_series(s1_raw)
    cand_script = _script_category_series(cand_raw)
    df["s1_name_script"] = s1_script.to_numpy()
    df["cand_name_script"] = cand_script.to_numpy()
    df["name_script_bucket"] = [
        pair_script_bucket(a, b) for a, b in zip(s1_script.tolist(), cand_script.tolist())
    ]
    t_s1 = s1_raw.map(lambda s: transliterate_text(s, backend))
    t_cand = cand_raw.map(lambda s: transliterate_text(s, backend))
    agree = ((t_s1.str.lower() == t_cand.str.lower()) & (t_s1 != "") & (t_cand != ""))
    df["translit_name_agree_exact"] = agree.astype(float).to_numpy()
    df["translit_name_ratio"] = np.array(
        [fuzz_ratio(a, b) for a, b in zip(t_s1.tolist(), t_cand.tolist())], dtype=float)
    if backend == "none":
        df["translit_addr_ratio"] = np.zeros(len(df), dtype=float)
    else:
        t_s1a = s1_addr.map(lambda s: transliterate_text(s, backend))
        t_canda = cand_addr.map(lambda s: transliterate_text(s, backend))
        df["translit_addr_ratio"] = np.array(
            [fuzz_ratio(a, b) for a, b in zip(t_s1a.tolist(), t_canda.tolist())], dtype=float)
    return df


# ---------------------------------------------------------------------------
# 13. Cross-script positive-pair analysis (SAMPLED positives)
# ---------------------------------------------------------------------------

_CROSS_SCRIPT_BUCKETS = ("cross_latin_deva", "cross_other")

def run_cross_script_positive_eda(
    pos_feat: Optional[pd.DataFrame],
    cfg: AppConfig,
) -> Dict[str, Any]:
    """Where do cross-script positives live, and does translit agree?"""
    columns = ["source_pair", "s1_country", "cand_country", "s1_script",
               "cand_script", "n_pairs", "pct_of_source_pair",
               "translit_exact_rate", "translit_ratio_p50",
               "name_exact_rate", "name_ratio_p50"]
    pos_feat = _ensure_multilingual_pair_columns(pos_feat, cfg)
    out_csv = cfg.eda_dir / "13_cross_script_positive_pairs.csv"
    if pos_feat is None or pos_feat.empty or "source_pair" not in pos_feat.columns:
        pd.DataFrame(columns=columns).to_csv(out_csv, index=False)
        return {"xscript_csv": str(out_csv), "artifacts": [str(out_csv)],
                "n_pairs": 0, "cross_script_rate": 0.0,
                "nonlatin_involved_rate": 0.0, "same_after_translit_rate": 0.0,
                "by_source_pair": {}}
    grp = pos_feat.groupby(["source_pair", "s1_country", "cand_country",
                            "s1_name_script", "cand_name_script"], dropna=False)
    pair_totals = pos_feat.groupby("source_pair").size().to_dict()
    rows: List[Dict[str, object]] = []
    for keys, sub in grp:
        sp = str(keys[0])
        rows.append({
            "source_pair": sp,
            "s1_country": "" if pd.isna(keys[1]) else str(keys[1]),
            "cand_country": "" if pd.isna(keys[2]) else str(keys[2]),
            "s1_script": str(keys[3]),
            "cand_script": str(keys[4]),
            "n_pairs": int(len(sub)),
            "pct_of_source_pair": round(len(sub) / pair_totals.get(keys[0], len(sub)), 6),
            "translit_exact_rate": round(float(sub["translit_name_agree_exact"].mean()), 6),
            "translit_ratio_p50": round(float(sub["translit_name_ratio"].median()), 6),
            "name_exact_rate": round(float(sub["name_exact"].mean()), 6)
            if "name_exact" in sub.columns else 0.0,
            "name_ratio_p50": round(float(sub["name_ratio"].median()), 6)
            if "name_ratio" in sub.columns else 0.0,
        })
    report = pd.DataFrame(rows, columns=columns)
    report = report.sort_values(["source_pair", "n_pairs"],
                                ascending=[True, False]).reset_index(drop=True)
    report.to_csv(out_csv, index=False)
    n = len(pos_feat)
    by_pair: Dict[str, Dict[str, float]] = {}
    for sp, sub in pos_feat.groupby("source_pair"):
        cross = sub["name_script_bucket"].isin(_CROSS_SCRIPT_BUCKETS).mean()
        by_pair[str(sp)] = {
            "n": int(len(sub)),
            "cross_script_rate": round(float(cross), 6),
            "translit_exact_rate": round(float(sub["translit_name_agree_exact"].mean()), 6),
        }
    return {
        "xscript_csv": str(out_csv),
        "artifacts": [str(out_csv)],
        "n_pairs": int(n),
        "cross_script_rate": round(
            float(pos_feat["name_script_bucket"].isin(_CROSS_SCRIPT_BUCKETS).mean()), 6),
        "nonlatin_involved_rate": round(float(
            ((pos_feat["s1_name_script"] != "Latin")
             | (pos_feat["cand_name_script"] != "Latin")).mean()), 6),
        "same_after_translit_rate": round(
            float(pos_feat["translit_name_agree_exact"].mean()), 6),
        "by_source_pair": by_pair,
    }


# ---------------------------------------------------------------------------
# 15. English-token assumption audit (SAMPLED positives + negatives)
# ---------------------------------------------------------------------------

def _suffix_shared_flags(
    s1_norm: pd.Series, cand_norm: pd.Series
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-pair word-Jaccard + suffix/stopword overlap flags (measure only)."""
    word_jac = np.zeros(len(s1_norm), dtype=float)
    en_suffix = np.zeros(len(s1_norm), dtype=float)
    en_stop = np.zeros(len(s1_norm), dtype=float)
    indic_suffix = np.zeros(len(s1_norm), dtype=float)
    stopword_only = np.zeros(len(s1_norm), dtype=float)
    s1_list = s1_norm.fillna("").astype(str).tolist()
    cand_list = cand_norm.fillna("").astype(str).tolist()
    for i, (a, b) in enumerate(zip(s1_list, cand_list)):
        set_a = set(tokenize(a))
        set_b = set(tokenize(b))
        word_jac[i] = jaccard_similarity(set_a, set_b)
        overlap = set_a & set_b
        if not overlap:
            continue
        if overlap & LEGAL_SUFFIX_TOKENS:
            en_suffix[i] = 1.0
        if overlap & EN_STOPWORDS_MEASURE:
            en_stop[i] = 1.0
        if overlap & INDIC_SUFFIX_TOKENS:
            indic_suffix[i] = 1.0
        if overlap <= EN_STOPWORDS_MEASURE:
            stopword_only[i] = 1.0
    return word_jac, en_suffix, en_stop, indic_suffix, stopword_only


def run_token_assumption_audit_eda(
    pos_feat: Optional[pd.DataFrame],
    negs_feat: Optional[pd.DataFrame],
    cfg: AppConfig,
) -> Dict[str, Any]:
    """How do token-based features behave per script bucket and class?

    Measures word-Jaccard, token_set_ratio, suffix sharing (English legal
    suffixes AND Indic suffix candidates — both measure-only, never
    stripped, never equated) and stopword-only overlap. No stopword removal
    anywhere: this section audits what such removal would destroy.
    """
    columns = ["script_bucket", "class", "n", "word_jaccard_mean",
               "word_jaccard_p50", "token_set_p50", "en_suffix_shared_rate",
               "en_stopword_shared_rate", "indic_suffix_shared_rate",
               "stopword_only_overlap_rate", "name_exact_rate"]
    pos_feat = _ensure_multilingual_pair_columns(pos_feat, cfg)
    negs_feat = _ensure_multilingual_pair_columns(negs_feat, cfg)
    frames: List[Tuple[str, Optional[pd.DataFrame]]] = [("positive", pos_feat)]
    if negs_feat is not None and not negs_feat.empty and "neg_type" in negs_feat.columns:
        for neg_type, sub in negs_feat.groupby("neg_type"):
            frames.append((str(neg_type), sub))
    rows: List[Dict[str, object]] = []
    for class_name, frame in frames:
        if frame is None or frame.empty:
            continue
        wj, en_suf, en_stop, in_suf, sw_only = _suffix_shared_flags(
            frame["s1_name_norm"] if "s1_name_norm" in frame.columns else pd.Series([""] * len(frame)),
            frame["cand_name_norm"] if "cand_name_norm" in frame.columns else pd.Series([""] * len(frame)),
        )
        work = frame.copy()
        work["_word_jaccard"] = wj
        work["_en_suffix"] = en_suf
        work["_en_stop"] = en_stop
        work["_indic_suffix"] = in_suf
        work["_sw_only"] = sw_only
        for bucket, sub in work.groupby("name_script_bucket"):
            rows.append({
                "script_bucket": str(bucket),
                "class": class_name,
                "n": int(len(sub)),
                "word_jaccard_mean": round(float(sub["_word_jaccard"].mean()), 6),
                "word_jaccard_p50": round(float(sub["_word_jaccard"].median()), 6),
                "token_set_p50": round(float(sub["name_token_set"].median()), 6)
                if "name_token_set" in sub.columns else 0.0,
                "en_suffix_shared_rate": round(float(sub["_en_suffix"].mean()), 6),
                "en_stopword_shared_rate": round(float(sub["_en_stop"].mean()), 6),
                "indic_suffix_shared_rate": round(float(sub["_indic_suffix"].mean()), 6),
                "stopword_only_overlap_rate": round(float(sub["_sw_only"].mean()), 6),
                "name_exact_rate": round(float(sub["name_exact"].mean()), 6)
                if "name_exact" in sub.columns else 0.0,
            })
    report = pd.DataFrame(rows, columns=columns)
    if not report.empty:
        class_order = {c: i for i, c in enumerate(
            ["positive", "name_hard", "address_hard", "hybrid_hard", "random"])}
        report["_o"] = report["class"].map(lambda c: class_order.get(c, 99))
        report = report.sort_values(["script_bucket", "_o"]).drop(columns="_o").reset_index(drop=True)
    out_csv = cfg.eda_dir / "15_token_assumption_audit.csv"
    report.to_csv(out_csv, index=False)
    artifacts = [str(out_csv)]

    # figure: word-Jaccard p50 + suffix-share rates by script bucket
    _apply_style(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    buckets = sorted(report.script_bucket.unique().tolist()) if not report.empty else []
    classes = [c for c in ("positive", "name_hard", "address_hard", "hybrid_hard", "random")
               if c in set(report["class"].tolist())] if not report.empty else []
    x = np.arange(len(buckets))
    width = 0.8 / max(len(classes), 1)
    for i, class_name in enumerate(classes):
        vals = [float(report[(report.script_bucket == b) & (report["class"] == class_name)]
                        ["word_jaccard_p50"].sum()) for b in buckets]
        axes[0].bar(x + (i - len(classes) / 2 + 0.5) * width, vals, width, label=class_name)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(buckets, rotation=25, ha="right")
    axes[0].set_title("word-Jaccard p50 by script bucket")
    axes[0].set_ylabel("p50")
    axes[0].legend(fontsize=8)
    pos_only = report[report["class"] == "positive"] if not report.empty else report
    x2 = np.arange(len(buckets))
    for j, col in enumerate(("en_suffix_shared_rate", "indic_suffix_shared_rate")):
        vals = [float(pos_only[pos_only.script_bucket == b][col].sum()) for b in buckets]
        axes[1].bar(x2 + (j - 0.5) * 0.35, vals, 0.35, label=col)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(buckets, rotation=25, ha="right")
    axes[1].set_title("suffix sharing on positives (measure only)")
    axes[1].set_ylabel("rate")
    axes[1].legend(fontsize=8)
    if report.empty:
        axes[0].text(0.5, 0.5, "no data", ha="center", transform=axes[0].transAxes)
    fig.tight_layout()
    artifacts.append(_savefig(fig, cfg.figures_dir / "token_assumption_by_script.png", cfg))

    headline: Dict[str, Any] = {}
    for bucket in ("latin_latin", "deva_deva", "cross_latin_deva"):
        sub = report[(report.script_bucket == bucket) & (report["class"] == "positive")]
        if not sub.empty:
            headline[bucket] = {
                "n": int(sub["n"].iloc[0]),
                "word_jaccard_p50": float(sub["word_jaccard_p50"].iloc[0]),
                "token_set_p50": float(sub["token_set_p50"].iloc[0]),
                "indic_suffix_shared_rate": float(sub["indic_suffix_shared_rate"].iloc[0]),
            }
    return {
        "token_csv": str(out_csv),
        "artifacts": artifacts,
        "headline": headline,
    }


# ---------------------------------------------------------------------------
# 16. Character n-gram vs word-token comparison (SAMPLED pairs)
# ---------------------------------------------------------------------------

def _granularity_label(sizes: Tuple[int, ...]) -> str:
    sizes = tuple(sorted(int(s) for s in sizes))
    if not sizes:
        return "char_?"
    if len(sizes) > 1 and all(b - a == 1 for a, b in zip(sizes, sizes[1:])):
        return f"char_{sizes[0]}-{sizes[-1]}"
    return "char_" + "_".join(str(s) for s in sizes)


def _ngram_bucket(row_bucket: str, is_positive: bool, neg_type: str) -> str:
    if is_positive:
        if row_bucket == "latin_latin":
            return "pos_latin_latin"
        if row_bucket == "deva_deva":
            return "pos_deva_deva"
        if row_bucket in _CROSS_SCRIPT_BUCKETS:
            return "pos_cross"
        return "pos_other"
    if neg_type in _HARD_NEG_TYPES:
        return "hard_neg"
    return "random_neg"


def run_char_ngram_eda(
    pos_feat: Optional[pd.DataFrame],
    negs_feat: Optional[pd.DataFrame],
    cfg: AppConfig,
) -> Dict[str, Any]:
    """Word-token vs char-n-gram similarity across script buckets + cost.

    Similarities are computed on conservative normalized names. Runtime per
    representation is measured (summary only, never in the CSV — timings are
    environment-dependent and must not break determinism checks).
    """
    columns = ["representation", "granularity", "bucket", "n",
               "mean", "p10", "p50", "p90"]
    pos_feat = _ensure_multilingual_pair_columns(pos_feat, cfg)
    negs_feat = _ensure_multilingual_pair_columns(negs_feat, cfg)
    raw_grans = _multilingual_cfg(cfg).get(
        "char_ngram_granularities", [[2, 3, 4], [3, 4, 5], [3, 4, 5, 6]])
    granularities: List[Tuple[int, ...]] = []
    for g in raw_grans:
        try:
            granularities.append(tuple(sorted(int(x) for x in g)))
        except (TypeError, ValueError):
            continue
    if not granularities:
        granularities = [(3, 4, 5)]
    out_csv = cfg.eda_dir / "16_char_ngram_comparison.csv"
    pairs: List[Tuple[str, str, str, bool, str]] = []
    if pos_feat is not None and not pos_feat.empty:
        for row in pos_feat.itertuples(index=False):
            d = row._asdict() if hasattr(row, "_asdict") else {}
            def _g(key: str, default: str = "") -> str:
                try:
                    v = d[key] if isinstance(d, dict) else getattr(row, key)
                except Exception:
                    return default
                return "" if v is None else str(v)
            pairs.append((_g("s1_name_norm"), _g("cand_name_norm"),
                          _g("name_script_bucket"), True, ""))
    if negs_feat is not None and not negs_feat.empty:
        for row in negs_feat.itertuples(index=False):
            d = row._asdict() if hasattr(row, "_asdict") else {}
            def _g2(key: str, default: str = "") -> str:
                try:
                    v = d[key] if isinstance(d, dict) else getattr(row, key)
                except Exception:
                    return default
                return "" if v is None else str(v)
            pairs.append((_g2("s1_name_norm"), _g2("cand_name_norm"),
                          _g2("name_script_bucket"), False, _g2("neg_type")))
    if not pairs:
        pd.DataFrame(columns=columns).to_csv(out_csv, index=False)
        return {"ngram_csv": str(out_csv), "artifacts": [str(out_csv)],
                "timings_sec": {}, "granularities": [], "headline": {}}
    rep_defs: List[Tuple[str, object]] = [("word_token", None)]
    rep_defs += [(_granularity_label(g), g) for g in granularities]
    sims: Dict[str, List[float]] = {name: [] for name, _ in rep_defs}
    timings: Dict[str, float] = {}
    for rep_name, gran in rep_defs:
        t0 = time.perf_counter()
        if gran is None:
            sims[rep_name] = [word_jaccard(a, b) for a, b, _, _, _ in pairs]
        else:
            assert isinstance(gran, tuple)
            sims[rep_name] = [char_ngram_jaccard(a, b, gran) for a, b, _, _, _ in pairs]
        timings[rep_name] = round(time.perf_counter() - t0, 3)
    bucket_of = [_ngram_bucket(b, is_pos, nt) for _, _, b, is_pos, nt in pairs]
    bucket_order = ["pos_latin_latin", "pos_deva_deva", "pos_cross", "pos_other",
                    "hard_neg", "random_neg"]
    rows: List[Dict[str, object]] = []
    for rep_name, gran in rep_defs:
        vals = np.array(sims[rep_name], dtype=float)
        gran_label = "1-grams(words)" if gran is None else ",".join(str(x) for x in gran)
        for bucket in bucket_order:
            mask = np.array([b == bucket for b in bucket_of])
            if not mask.any():
                continue
            v = vals[mask]
            rows.append({
                "representation": rep_name,
                "granularity": gran_label,
                "bucket": bucket,
                "n": int(mask.sum()),
                "mean": round(float(v.mean()), 6),
                "p10": round(float(np.percentile(v, 10)), 6),
                "p50": round(float(np.percentile(v, 50)), 6),
                "p90": round(float(np.percentile(v, 90)), 6),
            })
    report = pd.DataFrame(rows, columns=columns)
    report.to_csv(out_csv, index=False)
    artifacts = [str(out_csv)]

    # figure: ECDF per representation for pos_all / pos_cross / hard / random
    _apply_style(cfg)
    n_panels = len(rep_defs)
    fig, axes = plt.subplots(1, max(n_panels, 1), figsize=(4.2 * max(n_panels, 1), 4.2),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]
    line_defs = [
        ("pos_all", np.array([p[3] for p in pairs])),
        ("pos_cross", np.array([b == "pos_cross" for b in bucket_of])),
        ("hard_neg", np.array([b == "hard_neg" for b in bucket_of])),
        ("random_neg", np.array([b == "random_neg" for b in bucket_of])),
    ]
    for ax, (rep_name, _) in zip(axes, rep_defs):
        vals = np.array(sims[rep_name], dtype=float)
        for line_name, mask in line_defs:
            if mask.any():
                x, y = _ecdf(vals[mask])
                ax.plot(x, y, label=f"{line_name} (n={int(mask.sum())})")
        ax.set_title(rep_name)
        ax.set_xlabel("similarity")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("ECDF")
    fig.suptitle("Word-token vs char-n-gram similarity (sampled pairs)")
    fig.tight_layout()
    artifacts.append(_savefig(fig, cfg.figures_dir / "char_ngram_comparison.png", cfg))

    headline: Dict[str, Dict[str, float]] = {}
    for rep_name, _ in rep_defs:
        sub = report[report.representation == rep_name]
        entry: Dict[str, float] = {}
        for bucket in ("pos_cross", "hard_neg", "random_neg"):
            sel = sub[sub.bucket == bucket]
            if not sel.empty:
                entry[bucket] = float(sel["p50"].iloc[0])
        headline[rep_name] = entry
    return {
        "ngram_csv": str(out_csv),
        "artifacts": artifacts,
        "timings_sec": dict(timings),
        "granularities": [list(g) for g in granularities],
        "headline": headline,
    }


CASEBOOK_DISPLAY_FEATURES = [
    "name_exact", "name_ratio", "name_token_set", "addr_exact", "addr_ratio",
    "addr_token_jaccard", "numeric_any_overlap", "numeric_conflict",
    "postcode_any_overlap", "house_number_agree", "country_equal",
    "s1_name_script", "cand_name_script", "name_script_bucket",
    "translit_name_agree_exact", "translit_name_ratio",
]


def _suffix_overlap_mask(df: pd.DataFrame, tokens: Collection[str]) -> pd.Series:
    """Rows whose norm-name token overlap includes any of `tokens` (measure)."""
    want = set(tokens)
    if df.empty or "s1_name_norm" not in df.columns or "cand_name_norm" not in df.columns:
        return pd.Series([], dtype=bool)
    s1_list = df["s1_name_norm"].fillna("").astype(str).tolist()
    cand_list = df["cand_name_norm"].fillna("").astype(str).tolist()
    flags = [bool(set(tokenize(a)) & set(tokenize(b)) & want)
             for a, b in zip(s1_list, cand_list)]
    return pd.Series(flags, index=df.index)


def _case_records(df: pd.DataFrame, bucket: str, label_text: str) -> List[Dict[str, str]]:
    recs: List[Dict[str, str]] = []
    for r in df.itertuples(index=False):
        d = r._asdict() if hasattr(r, "_asdict") else r
        def g(k: str, default: str = "") -> str:
            try:
                v = d[k] if isinstance(d, dict) else getattr(r, k)
            except Exception:
                return default
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return ""
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)
        recs.append({
            "bucket": bucket,
            "s1": g("source1_entity_id"), "cand": g("candidate_entity_id"),
            "pair": g("source_pair"), "s1_cty": g("s1_country"), "cand_cty": g("cand_country"),
            "s1_name": g("s1_name_raw"), "cand_name": g("cand_name_raw"),
            "s1_addr": g("s1_addr_raw"), "cand_addr": g("cand_addr_raw"),
            "s1_name_norm": g("s1_name_norm"), "cand_name_norm": g("cand_name_norm"),
            "label": label_text, "neg_type": g("neg_type"),
            "feats": "; ".join(f"{f}={g(f)}" for f in CASEBOOK_DISPLAY_FEATURES),
        })
    return recs


def build_casebook(
    pos_feat: pd.DataFrame,
    negs_feat: pd.DataFrame,
    union_pairs: Optional[pd.DataFrame],
    known_positives: set,
    s1_small: Optional[pd.DataFrame],
    pool_small: Optional[pd.DataFrame],
    test_tables: Dict[str, Optional[pd.DataFrame]],
    cfg: AppConfig,
) -> Dict[str, Any]:
    """Collect difficult buckets (SMALL lookups + SMALL known-positive set).

    ``known_positives`` holds only sampled + blocking-anchor pairs (small, never
    the global 7.6M set). ``s1_small``/``pool_small`` are the blocking SMALL
    normalized lookups; the large-group bucket stays within blocking anchors
    and the France-nearest bucket searches the blocking pool sample only.
    """
    cols = cfg.columns
    cb_cfg = cfg.eda.get("casebook", {})
    k = int(cb_cfg.get("per_bucket", 25))
    lo_name = float(cb_cfg.get("low_name_threshold", 0.55))
    lo_addr = float(cb_cfg.get("low_address_threshold", 0.45))
    hi_name = float(cb_cfg.get("high_neg_name_threshold", 0.80))
    hi_addr = float(cb_cfg.get("high_neg_address_threshold", 0.60))
    b_lo = float(cb_cfg.get("boundary_lo", 0.45))
    b_hi = float(cb_cfg.get("boundary_hi", 0.65))

    buckets: Dict[str, List[Dict[str, str]]] = {}

    def _pos_bucket(name: str, mask: pd.Series, sort_by: str, ascending: bool) -> None:
        if pos_feat.empty:
            buckets[name] = []
            return
        sub = pos_feat[mask].copy()
        if sort_by in sub.columns:
            sub = sub.sort_values(sort_by, ascending=ascending)
        buckets[name] = _case_records(sub.head(k), name, "positive(1)")

    def _neg_bucket(name: str, mask: pd.Series, sort_by: Optional[str], ascending: bool) -> None:
        if negs_feat.empty:
            buckets[name] = []
            return
        sub = negs_feat[mask].copy()
        if sort_by and sort_by in sub.columns:
            sub = sub.sort_values(sort_by, ascending=ascending)
        buckets[name] = _case_records(sub.head(k), name, "negative(0)")

    if not pos_feat.empty:
        _pos_bucket("pos_low_name", pos_feat["name_ratio"] <= lo_name, "name_ratio", True)
        _pos_bucket("pos_low_address", pos_feat["addr_token_jaccard"] <= lo_addr, "addr_token_jaccard", True)
        _pos_bucket("pos_conflicting_numbers", pos_feat["numeric_conflict"] == 1.0, "name_ratio", True)
        _pos_bucket("pos_country_mismatch", pos_feat["country_equal"] == 0.0, "name_ratio", False)
    else:
        for b in ("pos_low_name", "pos_low_address", "pos_conflicting_numbers", "pos_country_mismatch"):
            buckets[b] = []
    if not negs_feat.empty:
        hard = negs_feat[negs_feat["neg_type"].isin(["name_hard", "address_hard", "hybrid_hard"])].copy()
        hard["_combo"] = pd.to_numeric(hard["name_ratio"], errors="coerce").fillna(0) + \
            pd.to_numeric(hard["addr_token_jaccard"], errors="coerce").fillna(0)
        _neg_bucket("hardneg_high_score",
                    negs_feat.index.isin(hard[(hard["name_ratio"] >= hi_name) | (hard["addr_token_jaccard"] >= hi_addr)].index),
                    None, False)
        # re-sort the high-score bucket by combo
        if buckets["hardneg_high_score"]:
            pass  # order preserved from selection; combo sort applied below on raw frame instead
        _neg_bucket("neg_exact_name", negs_feat["name_exact"] == 1.0, "addr_token_jaccard", False)
        _neg_bucket("neg_exact_address", negs_feat["addr_exact"] == 1.0, "name_ratio", False)
        mid = (pd.to_numeric(negs_feat["name_ratio"], errors="coerce").fillna(0)
               + pd.to_numeric(negs_feat["addr_token_jaccard"], errors="coerce").fillna(0)) / 2
        _neg_bucket("boundary_negatives", (mid >= b_lo) & (mid <= b_hi), None, False)
    else:
        for b in ("hardneg_high_score", "neg_exact_name", "neg_exact_address", "boundary_negatives"):
            buckets[b] = []

    # Multilingual buckets (§28.8) over the SAME sampled pairs (annotated once).
    try:
        _ensure_multilingual_pair_columns(pos_feat, cfg)
        _ensure_multilingual_pair_columns(negs_feat, cfg)
    except Exception as exc:
        logger.warning("Multilingual pair annotation failed: %s", exc)
    ml_cfg = _multilingual_cfg(cfg)
    helpful_thr = float(ml_cfg.get("translit_helpful_threshold", 0.8))
    danger_thr = float(ml_cfg.get("translit_dangerous_threshold", 0.9))

    def _ml_bucket(name: str, df: pd.DataFrame, mask, sort_by: str,
                   ascending: bool, label_text: str) -> None:
        try:
            need = {"s1_name_script", "cand_name_script", "name_script_bucket",
                    "translit_name_agree_exact", "translit_name_ratio"}
            if df.empty or not need.issubset(set(df.columns)):
                buckets[name] = []
                return
            m = mask.reindex(df.index).fillna(False).to_numpy(dtype=bool)
            sub = df[m].copy()
            if sort_by in sub.columns:
                sub = sub.sort_values(sort_by, ascending=ascending, kind="mergesort")
            buckets[name] = _case_records(sub.head(k), name, label_text)
        except Exception as exc:
            logger.warning("Casebook bucket %s skipped: %s", name, exc)
            buckets[name] = []

    _all_false_pos = pd.Series(False, index=pos_feat.index) if not pos_feat.empty else pd.Series([], dtype=bool)
    _all_false_neg = pd.Series(False, index=negs_feat.index) if not negs_feat.empty else pd.Series([], dtype=bool)
    pos_ml_ok = (not pos_feat.empty
                 and {"s1_name_script", "cand_name_script", "name_script_bucket",
                      "translit_name_agree_exact", "translit_name_ratio"}.issubset(pos_feat.columns))
    neg_ml_ok = (not negs_feat.empty
                 and {"s1_name_script", "cand_name_script", "name_script_bucket",
                      "translit_name_agree_exact", "translit_name_ratio"}.issubset(negs_feat.columns))
    hard_neg = (negs_feat[negs_feat["neg_type"].isin(list(_HARD_NEG_TYPES))]
                if neg_ml_ok and "neg_type" in negs_feat.columns else negs_feat.head(0))

    # 1. Devanagari–Devanagari positives (same-script non-Latin behavior)
    _ml_bucket("ml_deva_deva_positive", pos_feat,
               (pos_feat["s1_name_script"] == "Devanagari")
               & (pos_feat["cand_name_script"] == "Devanagari") if pos_ml_ok else _all_false_pos,
               "name_ratio", True, "positive(1)")
    # 2. Latin–Devanagari cross-script positives
    _ml_bucket("ml_latin_deva_positive", pos_feat,
               pos_feat["name_script_bucket"] == "cross_latin_deva" if pos_ml_ok else _all_false_pos,
               "name_ratio", True, "positive(1)")
    # 3. Positives that agree ONLY after transliteration
    _ml_bucket("ml_same_after_transliteration", pos_feat,
               (pos_feat["translit_name_agree_exact"] == 1.0)
               & (pos_feat["name_exact"] == 0.0)
               if pos_ml_ok and "name_exact" in pos_feat.columns else _all_false_pos,
               "translit_name_ratio", False, "positive(1)")
    # 4. Positives where transliteration disagrees (recall risk if trusted)
    _ml_bucket("ml_transliteration_disagreement", pos_feat,
               (pos_feat["translit_name_agree_exact"] == 0.0)
               & (pos_feat["translit_name_ratio"] < helpful_thr) if pos_ml_ok else _all_false_pos,
               "translit_name_ratio", True, "positive(1)")
    # 5. Hard negatives with HIGH transliteration similarity (false-merge risk)
    _ml_bucket("ml_high_translit_hard_negative", hard_neg,
               ((hard_neg["translit_name_agree_exact"] == 1.0)
                | (hard_neg["translit_name_ratio"] >= danger_thr))
               if neg_ml_ok and not hard_neg.empty else _all_false_neg.reindex(
                   hard_neg.index).fillna(False) if not hard_neg.empty else pd.Series([], dtype=bool),
               "translit_name_ratio", False, "negative(0)")
    # 6. Positives with low token overlap but high character similarity
    _ml_bucket("ml_low_token_overlap_positive", pos_feat,
               (pos_feat["name_token_set"] < 0.3)
               if pos_ml_ok and "name_token_set" in pos_feat.columns else _all_false_pos,
               "name_ratio", False, "positive(1)")
    # 7. Positives sharing an English legal suffix (measure-only, never stripped)
    _ml_bucket("ml_suffix_shared_positive", pos_feat,
               _suffix_overlap_mask(pos_feat, LEGAL_SUFFIX_TOKENS) if pos_ml_ok else _all_false_pos,
               "name_ratio", True, "positive(1)")
    # 8. Positives whose overlap is stopwords only (what removal would destroy)
    if pos_ml_ok and "s1_name_norm" in pos_feat.columns and "cand_name_norm" in pos_feat.columns:
        _sw_flags = []
        for a, b in zip(pos_feat["s1_name_norm"].fillna("").astype(str).tolist(),
                        pos_feat["cand_name_norm"].fillna("").astype(str).tolist()):
            overlap = set(tokenize(a)) & set(tokenize(b))
            _sw_flags.append(bool(overlap) and overlap <= EN_STOPWORDS_MEASURE)
        _sw_mask = pd.Series(_sw_flags, index=pos_feat.index)
    else:
        _sw_mask = _all_false_pos
    _ml_bucket("ml_stopword_only_overlap", pos_feat, _sw_mask,
               "name_ratio", False, "positive(1)")
    # 9. Positives sharing an Indic suffix candidate (no equivalence assumed)
    _ml_bucket("ml_indic_suffix_positive", pos_feat,
               _suffix_overlap_mask(pos_feat, INDIC_SUFFIX_TOKENS) if pos_ml_ok else _all_false_pos,
               "name_ratio", True, "positive(1)")
    # 10. Cross-script hard negatives
    _ml_bucket("ml_cross_script_hard_negative", hard_neg,
               hard_neg["name_script_bucket"].isin(list(_CROSS_SCRIPT_BUCKETS))
               if neg_ml_ok and not hard_neg.empty else _all_false_neg,
               "translit_name_ratio", False, "negative(0)")

    # large candidate groups (blocking anchors only; closed-world labels apply)
    large_group_recs: List[Dict[str, str]] = []
    try:
        can_large = (
            union_pairs is not None and not union_pairs.empty
            and s1_small is not None and not s1_small.empty
            and pool_small is not None and not pool_small.empty
            and "name_norm" in s1_small.columns and "name_norm" in pool_small.columns
        )
        if can_large:
            assert s1_small is not None and pool_small is not None
            deg = union_pairs.groupby("source1_entity_id").size().sort_values(ascending=False).head(3)
            for sid, _ in deg.items():
                cands = union_pairs[union_pairs.source1_entity_id == sid]["candidate_entity_id"].astype(str).tolist()
                sample = sorted(cands)[:15]
                basic = pd.DataFrame({
                    "source1_entity_id": [sid] * len(sample),
                    "candidate_entity_id": sample,
                    "source_pair": ["S1_S2" if c.startswith("S2-") else "S1_S3" for c in sample],
                })
                feat = add_pair_features(basic, s1_small, pool_small, cols, label=0, neg_type="blocking_candidate")
                for rec in _case_records(feat, "large_candidate_group", "negative(0)"):
                    if (rec["s1"], rec["cand"]) in (known_positives or set()):
                        rec["label"] = "positive(1)"
                        rec["neg_type"] = ""
                    large_group_recs.append(rec)
    except Exception as exc:
        logger.warning("Large-group bucket skipped: %s", exc)
    buckets["large_candidate_group"] = large_group_recs

    # France test records nearest to the BLOCKING pool sample (unlabeled illustration)
    fr_recs: List[Dict[str, str]] = []
    try:
        test_s1 = (test_tables or {}).get("test_s1")
        can_fr = (
            test_s1 is not None and not test_s1.empty
            and pool_small is not None and not pool_small.empty
            and "name_norm" in pool_small.columns
        )
        if can_fr:
            assert test_s1 is not None and pool_small is not None
            c_cty = cols["country"]
            c_id = cols["entity_id"]
            if c_cty in test_s1.columns:
                fr = test_s1[test_s1[c_cty].astype(str).str.strip().str.lower() == "france"].copy()
            else:
                fr = test_s1.head(0).copy()
            # test_small may already be a France-only preview; fall back to head rows
            if fr.empty and not test_s1.empty and len(test_s1) <= 50:
                fr = test_s1.copy()
            if not fr.empty:
                fr = ensure_normalized_columns(fr, cols)
                fr_sample = fr.head(min(10, len(fr)))
                chunk = int(cfg.eda.get("retrieval", {}).get("query_chunk_size", 256))
                vec, mat = build_tfidf_index(pool_small["name_norm"].astype(str).tolist(), cfg)
                if vec is not None and c_id in pool_small.columns and c_id in fr_sample.columns:
                    idx, _ = retrieve_topk(fr_sample["name_norm"].astype(str).tolist(), vec, mat, 3, chunk)
                    pool_ids = pool_small[c_id].astype(str).tolist()
                    rows = []
                    for ai, sid in enumerate(fr_sample[c_id].astype(str).tolist()):
                        for j in idx[ai]:
                            if 0 <= int(j) < len(pool_ids):
                                rows.append((sid, pool_ids[int(j)]))
                    if rows:
                        basic = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])
                        basic["source_pair"] = basic["candidate_entity_id"].map(
                            lambda c: "S1_S2" if str(c).startswith("S2-") else "S1_S3")
                        ffeat = add_pair_features(basic, fr_sample, pool_small, cols, label=-1, neg_type="unlabeled")
                        fr_recs = _case_records(ffeat, "france_test_nearest_train", "unknown")
    except Exception as exc:  # pragma: no cover — illustrative bucket must not break EDA
        logger.warning("France-nearest bucket skipped: %s", exc)
    buckets["france_test_nearest_train"] = fr_recs

    # -- render HTML (fully self-contained: inline CSS/JS, no network) --
    all_recs = [r for rs in buckets.values() for r in rs]
    empty_buckets = sorted(b for b, rs in buckets.items() if not rs)
    empty_line = ("No examples found: " + ", ".join(empty_buckets) + "."
                  if empty_buckets else "No examples found: none — every bucket has rows.")
    options = "\n".join(
        f'<option value="{html.escape(b)}">{html.escape(b)} (n={len(rs)})</option>'
        for b, rs in buckets.items()
    )
    body_rows = []
    for r in all_recs:
        body_rows.append(
            "<tr>" + "".join(f"<td>{html.escape(str(r[c]))}</td>" for c in
                             ("bucket", "s1", "cand", "pair", "s1_cty", "cand_cty", "label",
                              "neg_type", "s1_name", "cand_name", "s1_addr", "cand_addr",
                              "s1_name_norm", "cand_name_norm", "feats")) + "</tr>"
        )
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Casebook — train pairs (Stage 1 EDA)</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;margin:16px;font-size:13px}}
h1{{font-size:20px}} .meta{{color:#444;margin-bottom:12px}}
.controls{{margin:12px 0;display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
input,select{{padding:6px;font-size:13px}}
table{{border-collapse:collapse;width:100%;font-size:12px}}
th,td{{border:1px solid #bbb;padding:5px 7px;vertical-align:top;max-width:340px;overflow-wrap:anywhere}}
th{{background:#f0f0f0;cursor:pointer;position:sticky;top:0}}
tr:nth-child(even){{background:#fafafa}}
</style></head><body>
<h1>Casebook — difficult train pairs</h1>
<div class="meta">Thresholds: low_name&le;{lo_name}, low_addr&le;{lo_addr}, high_neg_name&ge;{hi_name},
high_neg_addr&ge;{hi_addr}, boundary=[{b_lo},{b_hi}], per-bucket cap={k}. Click a header to sort;
use the search box / bucket filter to narrow rows. France bucket is unlabeled retrieval illustration.
Multilingual buckets (ml_*): translit helpful&lt;{helpful_thr}, dangerous&ge;{danger_thr}.
{html.escape(empty_line)}</div>
<div class="controls">
<label>Search <input id="q" type="text" placeholder="type to filter..." size="40"></label>
<label>Bucket <select id="bucket"><option value="">(all)</option>
{options}</select></label>
<span id="count"></span>
</div>
<table id="cb"><thead><tr>
<th>bucket</th><th>s1_id</th><th>cand_id</th><th>pair</th><th>s1_cty</th><th>cand_cty</th>
<th>label</th><th>neg_type</th><th>s1_name_raw</th><th>cand_name_raw</th>
<th>s1_addr_raw</th><th>cand_addr_raw</th><th>s1_name_norm</th><th>cand_name_norm</th><th>features</th>
</tr></thead><tbody>
{''.join(body_rows)}
</tbody></table>
<script>
const q=document.getElementById('q'), b=document.getElementById('bucket'),
      tbl=document.getElementById('cb'), cnt=document.getElementById('count');
function applyFilter(){{
  const needle=q.value.toLowerCase(), bk=b.value;
  let n=0;
  for(const tr of tbl.tBodies[0].rows){{
    const okB=!bk||tr.cells[0].textContent===bk;
    const okQ=!needle||tr.textContent.toLowerCase().includes(needle);
    tr.style.display=(okB&&okQ)?'':'none';
    if(okB&&okQ)n++;
  }}
  cnt.textContent=n+' rows shown';
}}
q.addEventListener('input',applyFilter); b.addEventListener('change',applyFilter);
for(const th of tbl.tHead.rows[0].cells){{
  let asc=true;
  th.addEventListener('click',()=>{{
    const i=th.cellIndex, rows=[...tbl.tBodies[0].rows];
    rows.sort((a,b)=>asc
      ? a.cells[i].textContent.localeCompare(b.cells[i].textContent)
      : b.cells[i].textContent.localeCompare(a.cells[i].textContent));
    asc=!asc;
    for(const r of rows)tbl.tBodies[0].appendChild(r);
  }});
}}
applyFilter();
</script></body></html>
"""
    out_html = cfg.eda_dir / "10_casebook_train_pairs.html"
    out_html.write_text(page, encoding="utf-8")
    return {
        "casebook_html": str(out_html),
        "artifacts": [str(out_html)],
        "bucket_counts": {b: len(rs) for b, rs in buckets.items()},
        "n_rows": len(all_recs),
        "empty_buckets": empty_buckets,
    }


# ===========================================================================
# 10. EDA SUMMARY -> reports/eda_summary.md
# ===========================================================================

def _fmt_pct(x: Optional[float]) -> str:
    return "NA" if x is None else f"{100 * float(x):.1f}%"


def _render_sampling_section(R: Dict[str, Any]) -> str:
    samp = R.get("sampling", {}) or {}
    fast = samp.get("fast_mode", False)
    cfg_s = samp.get("sampling_config", {}) or {}
    actual = samp.get("actual", {}) or {}
    lines = [
        f"- Fast mode: `{fast}` (divides sampling sizes ~5x, floor 500).",
        "- FULL-data sections (no sampling): dataset audit (01), ground-truth counts (02), "
        "raw + conservative-norm collisions (05), country counts + missing rates (08), "
        "script distribution (11), character-stat denominators (14 `n` + non-ASCII rates).",
        "- SAMPLED sections: positive features (03), hard negatives (04), aggressive-norm "
        "collisions (05 `__sample*` rows), blocking recall/burden (06/07), country string-stats "
        "+ vocab coverage (08 `n_sampled`), candidate graph (09, on the blocking union), "
        "transliteration collisions (12), character stats (14 `n_sampled`), cross-script "
        "positives (13), token audit (15), char-n-gram comparison (16).",
        f"- Transliteration backend: `{samp.get('transliteration_backend', 'NA')}` "
        "(additional feature only, never canonical).",
    ]
    if cfg_s:
        order = ["n_positive_s1", "max_positive_pairs", "n_negative_match_anchors",
                 "n_negative_singleton_anchors", "retrieval_pool_sample", "n_blocking_s1",
                 "blocking_pool_sample", "country_shift_per_group", "collision_aggressive_sample",
                 "transliteration_sample"]
        cfg_line = "; ".join(f"{k}={cfg_s.get(k, 'NA')}" for k in order if k in cfg_s)
        lines.append(f"- Config sample sizes: {cfg_line}.")
    bits: List[str] = []
    pos_a = actual.get("positives", {}) or {}
    if pos_a:
        bits.append(f"positives anchors={pos_a.get('n_anchors_sampled', 'NA')} "
                    f"pairs={pos_a.get('n_positive_rows', 'NA')}")
    neg_a = actual.get("negatives", {}) or {}
    if neg_a:
        bits.append(f"negatives match_anchors={neg_a.get('n_anchors_match', 'NA')} "
                    f"singleton_anchors={neg_a.get('n_anchors_singleton', 'NA')} "
                    f"pool={neg_a.get('n_pool_sampled', 'NA')}")
    blk_a = actual.get("blocking", {}) or {}
    if blk_a:
        bits.append(f"blocking s1={blk_a.get('n_s1', blk_a.get('n_anchors_sampled', 'NA'))} "
                    f"pool={blk_a.get('n_pool', 'NA')} "
                    f"positives={blk_a.get('n_positives_sampled', 'NA')}")
    coll_a = samp.get("collisions_aggressive_actual", {}) or samp.get("collisions", {}) or {}
    if isinstance(coll_a, dict) and coll_a:
        preview = "; ".join(f"{t}={n}" for t, n in sorted(coll_a.items())[:6])
        bits.append(f"aggressive collisions per-table n: {preview}")
    if bits:
        lines.append("- Actual sampled sizes: " + "; ".join(bits) + ".")
    lines.append("- Closed world: every sampled pool is FORCED to contain all true matches "
                 "of its anchors, so recall/exclusion is measured fairly; labels apply only "
                 "within each sample.")
    return "\n".join(lines)


def generate_summary_md(
    R: Dict[str, Any],
    cfg: AppConfig,
    meta: Dict[str, Any],
) -> str:
    """Render reports/eda_summary.md from computed results (no invented numbers)."""
    gt = R.get("ground_truth", {})
    pos = R.get("positives", {})
    neg = R.get("negatives", {})
    addr = R.get("address", {})
    name = R.get("names", {})
    blk = R.get("blocking", {})
    geo = R.get("country_shift", {})
    graph = R.get("graph", {})
    case = R.get("casebook", {})
    audit = R.get("audit", {})
    sampling_section = _render_sampling_section(R)

    by_pair = (pos.get("by_source_pair", {}) or {})
    s2 = by_pair.get("S1_S2", {}) or {}
    s3 = by_pair.get("S1_S3", {}) or {}

    # blocking table rows
    blk_rows = []
    for m in (blk.get("metrics", []) or []):
        blk_rows.append(
            f"| {m.get('blocker')} | {_fmt_pct(m.get('s1_s2_recall'))} | {_fmt_pct(m.get('s1_s3_recall'))} "
            f"| {_fmt_pct(m.get('overall_recall'))} | {m.get('avg_candidates_per_s1')} "
            f"| {m.get('p95_candidates_per_s1')} | {m.get('max_candidates_per_s1')} |")
    blk_table = "\n".join(blk_rows) if blk_rows else "| (blocking skipped — no positives) | | | | | | |"

    num_rates = (addr.get("numeric_rates_by_class", {}) or {})
    num_line = "; ".join(
        f"{c}: conflict={v.get('numeric_conflict', 'NA')}, overlap={v.get('numeric_any_overlap', 'NA')}"
        for c, v in sorted(num_rates.items())
    ) or "NA"

    n_rows = audit.get("n_rows", {}) or {}
    dataset_line = "; ".join(f"{t}={n}" for t, n in sorted(n_rows.items())) or "NA"

    # multilingual locals (all NA-safe; sections may be skipped)
    script_d = R.get("script_dist", {}) or {}
    nlr = script_d.get("non_latin_rates", {}) or {}
    nlr_line = "; ".join(
        f"{src}: name={_fmt_pct((rates or {}).get('business_name'))}, "
        f"addr={_fmt_pct((rates or {}).get('business_address'))}"
        for src, rates in sorted(nlr.items())
    ) or "NA"
    tr = R.get("translit", {}) or {}
    tr_backend = tr.get("backend", "NA")
    tr_tab = tr.get("per_table", {}) or {}
    tr_lines = []
    for t in ("train_source1", "train_source2", "train_source3",
              "test_source1", "test_source2", "test_source3"):
        reps = tr_tab.get(t, {}) or {}
        raw_u = (reps.get("raw_name", {}) or {}).get("n_unique")
        tr_u = (reps.get("translit_name", {}) or {}).get("n_unique")
        cross = (reps.get("translit_name", {}) or {}).get("n_cross_script_groups")
        delta = ("NA" if raw_u is None or tr_u is None
                 else f"{int(raw_u - tr_u):+} merges")
        tr_lines.append(f"{t}: raw_name_unique={raw_u if raw_u is not None else 'NA'} vs "
                        f"translit_name_unique={tr_u if tr_u is not None else 'NA'} "
                        f"({delta}); translit cross-script groups="
                        f"{cross if cross is not None else 'NA'}")
    tr_line = "; ".join(tr_lines) if tr_tab else "NA (transliteration-collision section skipped)"
    xs = R.get("xscript", {}) or {}
    xs_pair = "; ".join(
        f"{sp}: n={v.get('n', 'NA')}, cross={_fmt_pct(v.get('cross_script_rate'))}, "
        f"same-after-translit={_fmt_pct(v.get('translit_exact_rate'))}"
        for sp, v in sorted((xs.get("by_source_pair", {}) or {}).items())
    ) or "NA"
    tok = R.get("token_audit", {}) or {}
    tok_head = tok.get("headline", {}) or {}
    tok_line = "; ".join(
        f"{b}: n={v.get('n', 'NA')}, word-jaccard-p50={v.get('word_jaccard_p50', 'NA')}, "
        f"token-set-p50={v.get('token_set_p50', 'NA')}, "
        f"indic-suffix-share={_fmt_pct(v.get('indic_suffix_shared_rate'))}"
        for b, v in sorted(tok_head.items())
    ) or "NA"
    ng = R.get("char_ngram", {}) or {}
    ng_head = ng.get("headline", {}) or {}
    ng_line = "; ".join(
        f"{rep}: pos_cross_p50={v.get('pos_cross', 'NA')}, "
        f"hard_neg_p50={v.get('hard_neg', 'NA')}, random_neg_p50={v.get('random_neg', 'NA')}"
        for rep, v in sorted(ng_head.items())
    ) or "NA"
    ng_time = "; ".join(f"{rep}={sec}s" for rep, sec in
                        sorted((ng.get("timings_sec", {}) or {}).items())) or "NA"
    ml_empty = (case.get("empty_buckets", []) or [])
    ml_empty_line = ", ".join(ml_empty) if ml_empty else "none — every bucket has rows"

    md = f"""# EDA Summary — Business Entity Resolution (Stage 1)

> Auto-generated by `python scripts/run_eda.py` — do not hand-edit numbers here;
> re-run the pipeline instead. All figures referenced live under `eda/figures/`.

- Generated (UTC): {meta.get('timestamp_utc', 'NA')}
- Data root: `{meta.get('data_root', 'NA')}`
- Config: `{meta.get('config_path', 'NA')}` (hash `{meta.get('config_hash', 'NA')}`)
- Git commit: `{meta.get('git_commit', 'NA')}`
- Tables present: {', '.join(audit.get('tables_present', []) or ['NA'])}

## Sampling & scale

{sampling_section}

## Dataset

- Rows: {dataset_line}.
- Full per-table metrics (missingness, lengths, country counts): `eda/01_data_audit.csv`;
  figures `rows_per_source.png`, `missingness.png`, `name_address_lengths.png`.

**Finding → Evidence → Implication:** field missingness and length differences across
sources set the prior for which signals are even available — e.g. sources with high
address-missing rates force name-led retrieval, while boilerplate-long addresses inflate
character similarity without adding evidence.

## Ground-truth structure

- S1 entities in ground truth: {gt.get('n_s1_in_gt', 'NA')}; positive pairs: {gt.get('n_positive_pairs', 'NA')}
  (S1–S2: {gt.get('n_s1_s2', 'NA')}, S1–S3: {gt.get('n_s1_s3', 'NA')}).
- Singleton rate: {_fmt_pct(gt.get('singleton_rate'))}; multi-match (2+) rate: {_fmt_pct(gt.get('multi_match_rate'))}.
- Patterns: {gt.get('pattern_counts', 'NA')}.
- Detail: `eda/02_ground_truth_match_distribution.csv`, `match_count_distribution.png`.

**Finding → Evidence → Implication:** the task explicitly includes zero-match entities and
multi-match entities, so a top-1 policy is invalid and the final per-entity decision must
support abstention (empty list). Because macro-F0.5 scores singletons 1.0/0.0, the
operating threshold must be tuned per-entity with a precision bias — to be set in Stage 6/7.

## Name behavior

- Positive name_exact rate: {_fmt_pct(pos.get('name_exact_rate'))}; median name_ratio: {pos.get('name_ratio_p50', 'NA')}.
- S1–S2: exact={_fmt_pct(s2.get('name_exact_rate'))}, p50={s2.get('name_ratio_p50', 'NA')}, n={s2.get('n', 'NA')};
  S1–S3: exact={_fmt_pct(s3.get('name_exact_rate'))}, p50={s3.get('name_ratio_p50', 'NA')}, n={s3.get('n', 'NA')}.
- Suffix/order/acronym breakdown: `name_suffix_agreement.png` (+ `eda/03_*.csv` for exact/non-exact splits).

**Finding → Evidence → Implication:** exact normalized names cover a measurable slice of
positives but (unless ~100%) cannot be the sole strategy; low-similarity positives define
the retrieval gap that address/char-n-gram blockers and a nonlinear pair model must close.
Suffix and business-type tokens are analyzed, never blindly stripped, because they
separate co-located distinct businesses.

## Address behavior

- Positive addr_exact rate: {_fmt_pct(pos.get('addr_exact_rate'))}; median token-Jaccard: {pos.get('addr_jaccard_p50', 'NA')};
  either-side-missing rate: {_fmt_pct(pos.get('either_addr_missing_rate'))}.
- Numeric agreement by class [{num_line}]; 2×2 house table: {addr.get('house_2x2', 'NA')}.
- Figures: `positive_address_similarity.png`, `address_numeric_agreement.png`.

**Finding → Evidence → Implication:** missing numbers and contradictory numbers are
different evidence and must stay separate features. If conflicts are rare among positives
but common among hard negatives, a numeric-conflict penalty will be one of the strongest
precision features; if numbers are often missing, the model must not treat absence as
contradiction.

## Normalization behavior

- Collision report: `eda/05_normalization_collision_report.csv`, `normalization_collision_sizes.png`.
- Train normalized-name collision groups: {R.get('collisions', {}).get('train_norm_name_groups', 'NA')}.

**Finding → Evidence → Implication:** representations are judged by whether their
collisions predict matches, not by how much they collapse. Large normalized-name/address
groups are false-merge risks: exact normalized name is a high-confidence blocker/feature,
while aggressive normalization is recall-only signal and must never be a direct match rule.

## Hard negatives

- Sampled negatives: {neg.get('n_by_neg_type', 'NA')} (see `eda/04_hard_negative_feature_summary.csv`).
- Separation: `positive_vs_negative_similarity.png`; 2-D map: `name_vs_address_scatter.png`;
  correlations: `feature_correlation_heatmap.png`.

**Finding → Evidence → Implication:** random negatives are a sanity baseline only — the
decision-relevant comparison is positives vs name/address/hybrid hard negatives. Strong
overlap in the 2-D name×address map justifies a nonlinear pair classifier with
missingness/country/source-pair features over any fixed weighted rule (Stage 5).

## S1–S2 vs S1–S3

- Counts and exact-match/median splits are tabulated above (`by_source_pair`) and in
  `eda/03_positive_pair_feature_summary.csv` (`source_pair` breakdowns); panel: `source_pair_comparison.png`.

**Finding → Evidence → Implication:** any material gap between S1–S2 and S1–S3 (noise,
missingness, exact rates) must become an explicit `source_pair` feature and, if large,
separate calibration/thresholds per pair type rather than one global cutoff.

## Country shift

- Train countries: {geo.get('train_countries', 'NA')}; test countries: {geo.get('test_countries', 'NA')};
  unseen in train: {geo.get('unseen_in_train', 'NA')}.
- Per-country stats + char/token coverage vs train vocab: `eda/08_country_shift_report.csv`;
  composition: `country_distribution.png`.

**Finding → Evidence → Implication:** country is an open-set string — the pipeline must
never filter or one-hot to a fixed list, and every test entity (France included) needs a
prediction. Country-specific parsing (e.g. PIN heuristics) stays optional; universal
character/token features carry the unseen-country load.

## Multilingual observations

- Script mix (FULL data): non-Latin row shares [{nlr_line}]; detail:
  `eda/11_script_distribution.csv`, `script_distribution.png`. A Latin-only approach
  could not even *see* these rows — the shares above bound its maximum loss.
- Character load per table × country (FULL `n` + non-ASCII rates, SAMPLED char/shape
  stats, script shares, trigram vocab): `eda/14_character_statistics.csv`.
- Transliteration backend: `{tr_backend}` (additional feature only, never canonical).
  Same-sample raw-vs-transliterated name uniqueness [{tr_line}]; detail:
  `eda/12_transliteration_collision_report.csv`, `transliteration_collision_sizes.png`.
  Extra merges are recall opportunity AND false-merge risk — the cross-script-group
  counts are the risk side of that ledger.
- Cross-script positives (SAMPLED, n={xs.get('n_pairs', 'NA')}): overall cross-script
  rate {_fmt_pct(xs.get('cross_script_rate'))}; non-Latin-involved rate
  {_fmt_pct(xs.get('nonlatin_involved_rate'))}; same-after-transliteration rate
  {_fmt_pct(xs.get('same_after_translit_rate'))} [{xs_pair}]; detail:
  `eda/13_cross_script_positive_pairs.csv`.
- Token-assumption audit on positives [{tok_line}]; detail:
  `eda/15_token_assumption_audit.csv`, `token_assumption_by_script.png`. English legal
  suffixes and Indic suffix candidates are measured, never stripped, never equated —
  and no English stopword removal is applied (the stopword-only-overlap column shows
  what such removal would destroy).
- Word-token vs char-n-gram p50 [{ng_line}] with single-thread timings [{ng_time}];
  detail: `eda/16_char_ngram_comparison.csv`, `char_ngram_comparison.png`. No
  representation is crowned here — keep raw, normalized, char-level, token-level and
  optional-transliteration signals side by side into Stage 2/3.

**Finding → Evidence → Implication:** keep every representation (raw Unicode,
normalized Unicode, char-level, token-level, transliteration-as-feature) because each
fails on a different slice — Latin-only matching, ASCII folding, stopword stripping and
suffix stripping each delete evidence this section measures. Transliteration earns its
place only through the measured recall-vs-risk tradeoff, never by assumption.

## Blocking observations

| blocker | S1–S2 recall | S1–S3 recall | overall | avg/S1 | p95/S1 | max/S1 |
|---|---:|---:|---:|---:|---:|---:|
{blk_table}

- Per-positive rescue map: `eda/07_blocking_positive_coverage.csv`
  (`rescued_only_by` counts: {blk.get('rescue_counts', 'NA')}); frontier: `blocking_recall_vs_burden.png`.

**Finding → Evidence → Implication:** blocking sets the recall ceiling — fix retrieval
before touching classifier architecture. Keep blockers that add unique recall at modest
cost (especially for address-only / transliterated positives); drop or tighten blockers
that only duplicate coverage while exploding candidates.

## Candidate graph observations

- Mean/max S1 degree: {graph.get('mean_s1_degree', 'NA')} / {graph.get('max_s1_degree', 'NA')};
  max candidate degree: {graph.get('max_cand_degree', 'NA')}; components: {graph.get('components', 'NA')};
  top hub: {graph.get('top_hub', 'NA')}.
- Detail: `eda/09_candidate_graph_diagnostics.csv`, `candidate_degree_distribution.png`.

**Finding → Evidence → Implication:** hubs (generic names/addresses, blank records,
over-broad keys) are where precision dies — downweight common tokens and tighten those
blocks. Components are diagnostic only; never use transitive closure as predictions under
F0.5.

## Important difficult cases

- Casebook: `eda/10_casebook_train_pairs.html` — buckets {case.get('bucket_counts', 'NA')}.
- Empty buckets (No examples found): {ml_empty_line}.
- Review at least: low-name positives, low-address positives, conflicting-number
  positives, country-mismatch positives, exact-name/exact-address negatives, the
  largest candidate groups, and the ml_* cross-script/transliteration buckets
  before finalizing features and thresholds.

## Modeling decisions suggested by EDA

| # | EDA signal | Suggested response (later stage) |
|---|-----------|----------------------------------|
| 1 | Exact-name coverage vs hard-negative exact-name collisions | High-confidence feature + blocker; never sole matcher (Stage 3/5) |
| 2 | Numeric conflict rare in positives, common in hard negatives | Strong learned conflict penalty; missing≠conflict (Stage 2/5) |
| 3 | Low-name-similarity positives exist | Address/char-n-gram retrieval + nonlinear model (Stage 3/5) |
| 4 | S1–S2 vs S1–S3 gap | `source_pair` feature ± separate calibration (Stage 2/6) |
| 5 | Singleton prevalence | Per-entity abstention threshold; optimize macro-F0.5 directly (Stage 6/7) |
| 6 | France unseen / formatting shift | Universal char/token features; optional country-specific rules (Stage 2) |
| 7 | Large graph hubs | Downweight common tokens; tighten broad blocks (Stage 3) |
| 8 | Blocking recall gaps | Fix candidate generation before classifier tuning (Stage 3 first) |
| 9 | Non-Latin/cross-script positives exist | Keep raw+normalized Unicode, char-level and token-level signals; no Latin-only/ASCII/stopword/suffix stripping defaults (Stage 2/3) |
| 10 | Transliteration recall-vs-risk tradeoff | Use transliteration as an optional feature gated by measured agreement, never canonical (Stage 2/3) |

## Artifacts

{chr(10).join('- `' + a + '`' for a in sorted(meta.get('artifacts', [])))}
"""
    out_md = cfg.reports_dir / "eda_summary.md"
    out_md.write_text(md, encoding="utf-8")
    return str(out_md)


# ===========================================================================
# ORCHESTRATOR (scale-safe order: full-data first, then sampled, test freed)
# ===========================================================================

def run_full_eda(
    cfg: AppConfig,
    train: Dict[str, Optional[pd.DataFrame]],
    test: Dict[str, Optional[pd.DataFrame]],
    run_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run every Stage-1 section in scale-safe order; return results + artifacts.

    Order: (1) audit FULL, (2) ground-truth stats FULL, (3) collisions
    FULL raw+conservative / SAMPLED aggressive, (4) country shift FULL counts /
    SAMPLED stats, then free test frames, (5) sampled positives, (6) sampled
    negatives, (7) address/name deep-dives, (8) sampled blocking, (9) graph on
    the blocking union, (10) casebook with small frames, (11) summary.
    """
    cols = cfg.columns
    artifacts: List[str] = []
    R: Dict[str, Any] = {}

    # source tables only — ground truth has different columns and is handled separately
    source_tables = {k: v for k, v in {**train, **test}.items() if k != "train_gt"}

    # 1. audit (FULL data)
    logger.info("EDA 1/9: dataset audit (FULL data)")
    R["audit"] = run_dataset_audit(source_tables, cfg)
    artifacts += R["audit"].get("artifacts", [])

    # 2. ground truth (FULL counts, vectorized — no pair expansion)
    logger.info("EDA 2/9: ground-truth EDA (FULL counts)")
    R["ground_truth"] = run_ground_truth_eda(train.get("train_gt"), cfg)
    artifacts += R["ground_truth"].get("artifacts", [])
    gt_stats = R["ground_truth"].get("gt_stats")

    # 3. normalization collisions (FULL raw+conservative, SAMPLED aggressive)
    logger.info("EDA 3/9: normalization collisions (FULL raw+conservative, SAMPLED aggressive)")
    R["collisions"] = run_normalization_collision_eda(source_tables, cfg)
    artifacts += R["collisions"].get("artifacts", [])

    # 4. country shift (FULL counts/missing, SAMPLED stats/vocab)
    logger.info("EDA 4/9: country shift (FULL counts, SAMPLED stats/vocab)")
    R["country_shift"] = run_country_shift_eda(train, test, cfg)
    artifacts += R["country_shift"].get("artifacts", [])

    # 4b. multilingual table-level sections (need train+test frames alive).
    # 11 script distribution (FULL), 14 char stats (FULL n, SAMPLED stats),
    # 12 transliteration collisions (SAMPLED, fair raw/norm/translit).
    logger.info("EDA 4b/9: script distribution (FULL data)")
    R["script_dist"] = run_script_distribution_eda(source_tables, cfg)
    artifacts += R["script_dist"].get("artifacts", [])
    logger.info("EDA 4c/9: character statistics (FULL n, SAMPLED stats)")
    R["char_stats"] = run_character_stats_eda(train, test, cfg)
    artifacts += R["char_stats"].get("artifacts", [])
    logger.info("EDA 4d/9: transliteration collisions (SAMPLED)")
    R["translit"] = run_transliteration_collision_eda(source_tables, cfg)
    artifacts += R["translit"].get("artifacts", [])

    # Free test frames (keep only a tiny France preview for the casebook).
    france_preview: Optional[pd.DataFrame] = None
    try:
        test_s1_full = test.get("test_s1") if test is not None else None
        if test_s1_full is not None and not test_s1_full.empty and cols["country"] in test_s1_full.columns:
            mask = test_s1_full[cols["country"]].astype(str).str.strip().str.lower() == "france"
            france_preview = test_s1_full.loc[mask].head(10).copy()
    except Exception:
        france_preview = None
    try:
        del source_tables
    except Exception:
        pass
    if test is not None:
        for k in list(test.keys()):
            test[k] = None
    gc.collect()

    s1_raw = train.get("train_s1")
    s2_raw = train.get("train_s2")
    s3_raw = train.get("train_s3")
    gt_df = train.get("train_gt")
    has_pool = not (
        (s2_raw is None or s2_raw.empty) and (s3_raw is None or s3_raw.empty)
    )
    supervised = (
        gt_df is not None and s1_raw is not None and has_pool
        and not gt_df.empty and not s1_raw.empty
    )
    pos_feat = pd.DataFrame()
    negs_feat = pd.DataFrame()
    positives_basic = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "source_pair"])
    union_pairs: Optional[pd.DataFrame] = None
    s1_small_blocking: pd.DataFrame = pd.DataFrame()
    pool_small_blocking: pd.DataFrame = pd.DataFrame()
    blocking_positives_small: pd.DataFrame = pd.DataFrame(
        columns=["source1_entity_id", "candidate_entity_id", "source_pair"])
    actual: Dict[str, Any] = {}

    if supervised:
        assert gt_df is not None and s1_raw is not None
        # 5. sampled positives
        logger.info("EDA 5/9: positive pairs (SAMPLED anchors)")
        positives_basic, pos_feat, pos_info = build_sampled_positive_pairs(
            s1_raw, s2_raw, s3_raw, gt_df, gt_stats, cfg)
        logger.info("positives sampled: %s", pos_info)
        R["positives"] = run_positive_pair_eda(pos_feat, gt_stats, cfg)
        artifacts += R["positives"].get("artifacts", [])
        actual["positives"] = pos_info

        # 6. sampled negatives + featurize via returned SMALL lookups
        logger.info("EDA 6/9: hard negatives (SAMPLED anchors + pool, forced truth)")
        neg_sets, neg_info, _neg_truth, neg_s1_small, neg_pool_small = build_negative_sets(
            s1_raw, s2_raw, s3_raw, gt_df, gt_stats, cfg)
        logger.info("negatives mined: %s", neg_info)
        neg_frames = []
        for tag, basic in neg_sets.items():
            if basic is None or basic.empty:
                continue
            if neg_s1_small.empty or neg_pool_small.empty:
                continue
            try:
                f = add_pair_features(basic, neg_s1_small, neg_pool_small, cols, label=0, neg_type=tag)
            except Exception as exc:
                logger.warning("Featurizing negatives '%s' failed: %s", tag, exc)
                continue
            if not f.empty:
                neg_frames.append(f)
        negs_feat = pd.concat(neg_frames, ignore_index=True) if neg_frames else pd.DataFrame()
        del neg_sets, neg_s1_small, neg_pool_small
        gc.collect()
        R["negatives"] = run_hard_negative_eda(pos_feat, negs_feat, cfg)
        artifacts += R["negatives"].get("artifacts", [])
        actual["negatives"] = neg_info

        # 6b. multilingual pair-level sections on the SAME sampled pairs
        # (annotation is shared/in-place; 03/04 were already written above).
        logger.info("EDA 6b/9: cross-script positives + token audit + char n-grams (SAMPLED pairs)")
        R["xscript"] = run_cross_script_positive_eda(pos_feat, cfg)
        artifacts += R["xscript"].get("artifacts", [])
        R["token_audit"] = run_token_assumption_audit_eda(pos_feat, negs_feat, cfg)
        artifacts += R["token_audit"].get("artifacts", [])
        R["char_ngram"] = run_char_ngram_eda(pos_feat, negs_feat, cfg)
        artifacts += R["char_ngram"].get("artifacts", [])

        # 7. address/name deep-dives on the SAMPLED pairs
        logger.info("EDA 7/9: address + name deep-dives (SAMPLED pairs)")
        R["address"] = run_address_eda(pos_feat, negs_feat, cfg)
        artifacts += R["address"].get("artifacts", [])
        R["names"] = run_name_eda(pos_feat, negs_feat, cfg)
        artifacts += R["names"].get("artifacts", [])

        # 8. sampled blocking (pool forced to contain anchor truth)
        logger.info("EDA 8/9: blocking diagnostics (SAMPLED anchors + pool)")
        R["blocking"] = run_blocking_eda(s1_raw, s2_raw, s3_raw, gt_df, gt_stats, cfg)
        union_pairs = R["blocking"].pop("union_pairs", None)
        s1_small_blocking = R["blocking"].pop("s1_small", pd.DataFrame())
        pool_small_blocking = R["blocking"].pop("pool_small", pd.DataFrame())
        R["blocking"].pop("anchor_truth", {})
        blocking_positives_small = R["blocking"].pop("positives_small", blocking_positives_small)
        artifacts += R["blocking"].get("artifacts", [])
        actual["blocking"] = {
            k: R["blocking"].get(k) for k in
            ("n_s1", "n_pool", "n_positives_sampled", "n_anchors_sampled")
            if k in R["blocking"]
        }

        # free heavy train frames + gt_stats before graph/casebook (small only now)
        try:
            for k in list(train.keys()):
                train[k] = None
        except Exception:
            pass
        try:
            R["ground_truth"].pop("gt_stats", None)
        except Exception:
            pass
        gt_stats = None
        gc.collect()

        # 9. graph diagnostics on the SAMPLED union with SMALL frames
        logger.info("EDA 9/9: candidate graph diagnostics (SAMPLED union)")
        R["graph"] = run_graph_diagnostics(
            union_pairs if union_pairs is not None else pd.DataFrame(
                columns=["source1_entity_id", "candidate_entity_id"]),
            s1_small_blocking if s1_small_blocking is not None else pd.DataFrame(),
            pool_small_blocking if pool_small_blocking is not None else pd.DataFrame(),
            cfg)
        artifacts += R["graph"].get("artifacts", [])
    else:
        logger.warning("Supervised EDA sections skipped (need train S1 + S2/S3 + ground truth).")
        for _key, fname in (("positives", "03_positive_pair_feature_summary.csv"),
                            ("negatives", "04_hard_negative_feature_summary.csv"),
                            ("blocking_m", "06_blocking_metrics.csv"),
                            ("blocking_c", "07_blocking_positive_coverage.csv"),
                            ("graph", "09_candidate_graph_diagnostics.csv")):
            p = cfg.eda_dir / fname
            if not p.exists():
                pd.DataFrame().to_csv(p, index=False)
            artifacts.append(str(p))
        for fname, header in (
            ("13_cross_script_positive_pairs.csv",
             ["source_pair", "s1_country", "cand_country", "s1_script",
              "cand_script", "n_pairs", "pct_of_source_pair",
              "translit_exact_rate", "translit_ratio_p50",
              "name_exact_rate", "name_ratio_p50"]),
            ("15_token_assumption_audit.csv",
             ["script_bucket", "class", "n", "word_jaccard_mean",
              "word_jaccard_p50", "token_set_p50", "en_suffix_shared_rate",
              "en_stopword_shared_rate", "indic_suffix_shared_rate",
              "stopword_only_overlap_rate", "name_exact_rate"]),
            ("16_char_ngram_comparison.csv",
             ["representation", "granularity", "bucket", "n",
              "mean", "p10", "p50", "p90"]),
        ):
            p = cfg.eda_dir / fname
            if not p.exists():
                pd.DataFrame(columns=header).to_csv(p, index=False)
            artifacts.append(str(p))
        R["positives"] = {"skipped": True, "artifacts": []}
        R["negatives"] = {"skipped": True, "artifacts": []}
        R["xscript"] = {"skipped": True, "artifacts": []}
        R["token_audit"] = {"skipped": True, "artifacts": []}
        R["char_ngram"] = {"skipped": True, "artifacts": []}
        R["address"] = {"artifacts": []}
        R["names"] = {"artifacts": []}
        R["blocking"] = {"metrics": [], "artifacts": []}
        R["graph"] = {"artifacts": []}
        try:
            R["ground_truth"].pop("gt_stats", None)
        except Exception:
            pass
        gt_stats = None

    # 10. casebook with SMALL frames + SMALL known positives
    logger.info("Casebook + summary")
    known_positives: set = set()
    try:
        if positives_basic is not None and not positives_basic.empty:
            known_positives.update(zip(
                positives_basic["source1_entity_id"].astype(str).tolist(),
                positives_basic["candidate_entity_id"].astype(str).tolist()))
        if blocking_positives_small is not None and not blocking_positives_small.empty:
            known_positives.update(zip(
                blocking_positives_small["source1_entity_id"].astype(str).tolist(),
                blocking_positives_small["candidate_entity_id"].astype(str).tolist()))
    except Exception:
        pass
    test_small: Dict[str, Optional[pd.DataFrame]] = {"test_s1": france_preview}
    R["casebook"] = build_casebook(
        pos_feat, negs_feat,
        union_pairs if union_pairs is not None else pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id"]),
        known_positives,
        s1_small_blocking if s1_small_blocking is not None else pd.DataFrame(),
        pool_small_blocking if pool_small_blocking is not None else pd.DataFrame(),
        test_small, cfg)
    artifacts += R["casebook"].get("artifacts", [])

    # 11. sampling disclosure + summary
    R["sampling"] = {
        "fast_mode": bool(cfg.eda.get("fast_mode", False)),
        "sampling_config": dict(_sampling_cfg(cfg)),
        "actual": dict(actual),
        "collisions_aggressive_actual": dict(
            R.get("collisions", {}).get("aggressive_actual", {})),
        "full_data_sections": [
            "dataset audit (01)", "ground-truth counts (02)",
            "raw + conservative collisions (05)",
            "country counts + missing rates (08)",
            "script distribution (11)",
            "character-stat denominators (14 n + non-ASCII rates)"],
        "sampled_sections": [
            "positive features (03)", "hard negatives (04)",
            "aggressive collisions (05 __sample*)",
            "blocking (06/07)", "country stats/vocab (08 n_sampled)",
            "candidate graph (09)",
            "transliteration collisions (12, transliteration_sample/table)",
            "character stats (14 n_sampled)",
            "cross-script positives (13, sampled positives)",
            "token audit (15, sampled pairs)",
            "char n-gram comparison (16, sampled pairs)"],
        "transliteration_backend": R.get("translit", {}).get("backend", "NA"),
        "transliteration_sample_actual": dict(
            R.get("translit", {}).get("sample_actual", {})),
    }
    meta = dict(run_meta or {})
    meta["artifacts"] = sorted(set(artifacts))
    summary_path = generate_summary_md(R, cfg, meta)
    artifacts.append(summary_path)

    return {"results": R, "artifacts": sorted(set(artifacts))}
    pass
    test_small: Dict[str, Optional[pd.DataFrame]] = {"test_s1": france_preview}
    R["casebook"] = build_casebook(
        pos_feat, negs_feat,
        union_pairs if union_pairs is not None else pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id"]),
        known_positives,
        s1_small_blocking if s1_small_blocking is not None else pd.DataFrame(),
        pool_small_blocking if pool_small_blocking is not None else pd.DataFrame(),
        test_small, cfg)
    artifacts += R["casebook"].get("artifacts", [])

    # 11. sampling disclosure + summary
    R["sampling"] = {
        "fast_mode": bool(cfg.eda.get("fast_mode", False)),
        "sampling_config": dict(_sampling_cfg(cfg)),
        "actual": dict(actual),
        "collisions_aggressive_actual": dict(
            R.get("collisions", {}).get("aggressive_actual", {})),
        "full_data_sections": [
            "dataset audit (01)", "ground-truth counts (02)",
            "raw + conservative collisions (05)",
            "country counts + missing rates (08)",
            "script distribution (11)",
            "character-stat denominators (14 n + non-ASCII rates)"],
        "sampled_sections": [
            "positive features (03)", "hard negatives (04)",
            "aggressive collisions (05 __sample*)",
            "blocking (06/07)", "country stats/vocab (08 n_sampled)",
            "candidate graph (09)",
            "transliteration collisions (12, transliteration_sample/table)",
            "character stats (14 n_sampled)",
            "cross-script positives (13, sampled positives)",
            "token audit (15, sampled pairs)",
            "char n-gram comparison (16, sampled pairs)"],
        "transliteration_backend": R.get("translit", {}).get("backend", "NA"),
        "transliteration_sample_actual": dict(
            R.get("translit", {}).get("sample_actual", {})),
    }
    meta = dict(run_meta or {})
    meta["artifacts"] = sorted(set(artifacts))
    summary_path = generate_summary_md(R, cfg, meta)
    artifacts.append(summary_path)

    return {"results": R, "artifacts": sorted(set(artifacts))}
