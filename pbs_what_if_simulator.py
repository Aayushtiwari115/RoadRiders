#!/usr/bin/env python3
# pbs_what_if_simulator.py
# ---------------------------------------------------------------------------------
# PBS What‑if Simulator + Vetted Chat (Retrieval‑Only, No Hallucinations)
#  + NLP/NER-assisted intent & scope extraction (spaCy EntityRuler, regex fallback)
#  + Deterministic analyzers with trust scoring & tamper-evident audit trail
#  + Streamlit UI with guardrails, evidence tables & provenance
#
# Run:  streamlit run pbs_what_if_simulator.py
# ---------------------------------------------------------------------------------
from __future__ import annotations
from hf_client import hf_text_generation, hf_question_answering, hf_sentence_embedding, semantic_search
import hashlib
import io
import json
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

# Optional NLP (no external downloads; uses spaCy blank+EntityRuler built from data)
try:
    import spacy
    from spacy.language import Language
    from spacy.pipeline import EntityRuler
except Exception:  # spaCy not installed in some environments
    spacy = None
    Language = None
    EntityRuler = None

# ---------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------
DATA_DIR: Path = Path(".")
PBS_CSV_NAME: str = "2024-25-pbs-program-expense-line-items.csv"
PROC_CSV_NAME: str = "sample_procurements.csv"
AUDIT_LOG: Path = DATA_DIR / "audit_logs_chat.jsonl"
PBS_CSV: Path = DATA_DIR / PBS_CSV_NAME
PROC_CSV: Path = DATA_DIR / PROC_CSV_NAME

# Official data.gov.au resource for the 2024–25 PBS line items (Finance)
PBS_DOWNLOAD_URL: str = (
    "https://data.gov.au/data/dataset/6aeeefa5-65c4-45fb-bc7e-c8f33c7ffa44/"
    "resource/935c6502-fff0-48a0-bb20-3939687a5eda/download/2024-25-pbs-program-expense-line-items.csv"
)

# Category thresholds used for split-purchase flags
DEFAULT_SPLIT_PURCHASE_THRESHOLDS: Dict[str, float] = {
    "ICT": 10_000.0,
    "Construction": 80_000.0,
    "Consulting": 20_000.0,
    "Office Supplies": 5_000.0,
    "Travel": 5_000.0,
}

# Refuse to answer when trust < LOW_EVIDENCE_REFUSAL
LOW_EVIDENCE_REFUSAL: float = 0.30

# Constants for the FY
FY_START = pd.Timestamp("2024-07-01")
FY_END = pd.Timestamp("2025-06-30")
FY_MONTHS = pd.period_range(pd.Period("2024-07", "M"), pd.Period("2025-06", "M"))

# A tiny epsilon to avoid division-by-zero / product-of-zero edge cases
EPS = 1e-12

# Security / privacy options (default ON)
DEFAULT_REDACT_VENDOR_IDS = True
DEFAULT_TAMPER_EVIDENT_AUDIT = True

# ---------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------

def _safe_toast(msg: str, icon: str = "ℹ️") -> None:
    """Toast helper that won't crash on older Streamlit versions."""
    try:
        st.toast(msg, icon=icon)
    except Exception:
        st.info(msg)

def _section_divider(label: str, emoji: str = "•") -> None:
    st.markdown(
        f"""
        <div style="padding-top:0.6rem;padding-bottom:0.3rem;">
          <span style="font-weight:600;letter-spacing:.2px">{emoji} {label}</span>
        </div>
        <hr style="margin-top:0.4rem;margin-bottom:0.8rem;opacity:.2">
        """,
        unsafe_allow_html=True,
    )

def _mask_vendor_id(s: str) -> str:
    if not isinstance(s, str):
        return s
    if len(s) <= 4:
        return "***"
    return s[:4] + "-***"

# ---------------------------------------------------------------------------------
# IO & Validation
# ---------------------------------------------------------------------------------

def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    """Read CSV with several encodings before falling back to python engine."""
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1", engine="python", low_memory=False)

def try_download_pbs_csv(dest: Path) -> bool:
    """Attempt to download the PBS CSV to `dest`."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(PBS_DOWNLOAD_URL, timeout=30, stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        _safe_toast(f"PBS auto-download failed: {e}", icon="⚠️")
        return False

def ensure_procurement_csv(proc_path: Path, force_generate: bool = False) -> None:
    """Create a synthetic procurement CSV if one does not exist."""
    if proc_path.exists() and not force_generate:
        return
    rng = pd.date_range("2024-07-01", "2025-06-30", freq="D")
    categories = ["ICT", "Construction", "Consulting", "Office Supplies", "Travel"]
    vendors = [f"VEND-{i:03d}" for i in range(1, 61)]
    rows: List[Dict[str, object]] = []
    rs = np.random.RandomState(42)
    base_map = {
        "ICT": 12_000,
        "Construction": 45_000,
        "Consulting": 16_000,
        "Office Supplies": 2_200,
        "Travel": 3_800,
    }
    for d in rng:
        for _ in range(6):  # ~6 purchases/day
            cat = rs.choice(categories, p=[0.30, 0.22, 0.20, 0.18, 0.10])
            base = base_map[cat]
            amount = float(rs.lognormal(mean=np.log(base), sigma=0.6))
            rows.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "vendor_id": rs.choice(vendors),
                    "category": cat,
                    "amount": round(amount, 2),
                }
            )
    pd.DataFrame(rows).to_csv(proc_path, index=False)

def get_first_match(df: pd.DataFrame, keywords: List[str]) -> str:
    """Return the first column name where all keywords appear in its lowercase name."""
    for col in df.columns:
        name = col.lower()
        if all(k in name for k in keywords):
            return col
    raise KeyError(f"PBS CSV missing expected column with keywords={keywords!r}")

@st.cache_data(show_spinner=False)
def load_pbs(path: Path) -> Tuple[pd.DataFrame, str]:
    """
    Load and normalize the PBS CSV.
    Returns (df, original_budget_col_name_for_2024_25)
    """
    if not path.exists():
        ok = try_download_pbs_csv(path)
        if not ok:
            st.error(
                f"Could not auto-download the PBS CSV. Please place it manually as **{PBS_CSV_NAME}**."
            )
            st.stop()
    try:
        df = read_csv_with_fallback(path)
    except Exception as e:
        st.error(f"Failed to read PBS CSV: {e}")
        st.stop()

    # Normalize numbers in year columns
    year_cols = [c for c in df.columns if any(y in c for y in ["2023", "2024", "2025", "2026", "2027"])]
    for c in year_cols:
        s = (
            df[c]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip()
            .replace({"nan": np.nan, "": np.nan, "nfp": np.nan, "NFP": np.nan})
        )
        df[c] = pd.to_numeric(s, errors="coerce")

    # Canonical columns with guarded lookups
    try:
        df["portfolio"] = df[get_first_match(df, ["portfolio"])]
        df["program"] = df[get_first_match(df, ["program"])]
        df["outcome"] = df[get_first_match(df, ["outcome"])]
        if any("department" in c.lower() for c in df.columns):
            df["entity"] = df[get_first_match(df, ["department"])]
        else:
            df["entity"] = df[get_first_match(df, ["entity"])]
        df["appropriation_type"] = df[get_first_match(df, ["appropriation"])]
        df["expense_type"] = df[get_first_match(df, ["expense", "type"])]
    except KeyError as e:
        st.error(
            "The PBS CSV does not contain required columns.\n\n"
            f"Missing: {e}\n\n"
            "Tip: Ensure the official Finance CSV is used without altering column headers."
        )
        st.stop()

    # Locate the 2024–25 budget column (allowing various dash types)
    y24 = [k for k in df.columns if ("2024-25" in k) or ("2024/25" in k) or ("2024–25" in k)]
    if not y24:
        st.error("PBS CSV missing a 2024–25 budget column.")
        st.stop()
    df["budget_2024_25"] = df[y24[0]]
    return df, y24[0]

@st.cache_data(show_spinner=False)
def load_procurement(path: Path) -> pd.DataFrame:
    """Load procurement CSV (creating a synthetic one if necessary)."""
    if not path.exists():
        ensure_procurement_csv(path, force_generate=True)
    try:
        df = pd.read_csv(path, parse_dates=["date"])
    except Exception as e:
        st.error(f"Failed to read procurement CSV: {e}")
        st.stop()
    required = ["date", "category", "vendor_id", "amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Procurement CSV missing required columns: {missing}")
        st.stop()
    # Sanity: coerce amount numeric
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date", "category", "vendor_id", "amount"]).copy()
    return df

def dataset_fingerprint(df: pd.DataFrame, src_path: Path) -> dict:
    """Produce a minimal fingerprint for provenance."""
    buf = io.BytesIO()
    df.head(1000).to_csv(buf, index=False)
    h = hashlib.md5(buf.getvalue()).hexdigest()
    mtime = None
    if src_path.exists():
        mtime = datetime.utcfromtimestamp(src_path.stat().st_mtime).isoformat() + "Z"
    return {
        "file": str(src_path),
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "sample_md5": h,
        "mtime": mtime,
    }

# ---------------------------------------------------------------------------------
# Math & Analytics
# ---------------------------------------------------------------------------------

def modified_z_scores(x: np.ndarray) -> np.ndarray:
    """Robust modified z-scores (\n z \n >= 3.5 => outlier)."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad

def crosscheck_sum(
    df: pd.DataFrame,
    value_col: str,
    group_cols: Optional[List[str]] = None,
) -> Tuple[float, float, float]:
    """Return (direct_sum, group_sum, relative_difference)."""
    direct = float(df[value_col].sum()) if not df.empty else 0.0
    if group_cols:
        gsum = float(df.groupby(group_cols)[value_col].sum().sum()) if not df.empty else direct
    else:
        gsum = direct
    denom = max(abs(direct), 1.0)
    rel_diff = abs(direct - gsum) / denom
    return direct, gsum, rel_diff

def compute_procurement_baseline(procs_df: pd.DataFrame) -> Tuple[float, float, pd.Series, pd.Period]:
    """YTD, projected future (simple seasonal avg), monthly series, last completed period."""
    df = procs_df[(procs_df["date"] >= FY_START) & (procs_df["date"] <= FY_END)].copy()
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month")["amount"].sum().sort_index()
    if monthly.empty:
        return 0.0, 0.0, monthly, pd.Period("2024-06", "M")
    last_completed = monthly.index.max()
    ytd = float(monthly[monthly.index <= last_completed].sum())
    recent = monthly[monthly.index <= last_completed].tail(3)
    avg = float(recent.mean()) if len(recent) else float(monthly.mean())
    months_left = [m for m in FY_MONTHS if m > last_completed]
    proj_future = avg * len(months_left)
    return ytd, proj_future, monthly, last_completed

def backtest_forecast_accuracy(monthly: pd.Series, k_holdout: int = 2) -> float:
    """Return a [0..1] backtest score using a naive tail-3 average vs holdout WAPE."""
    if len(monthly) <= k_holdout + 3:
        return 0.6  # conservative when data is sparse
    train = monthly.iloc[: -k_holdout]
    holdout = monthly.iloc[-k_holdout:]
    avg = float(train.tail(3).mean()) if len(train) >= 3 else float(train.mean())
    forecast = pd.Series([avg] * len(holdout), index=holdout.index)
    denom = max(float(holdout.sum()), 1.0)
    wape = float(np.abs(holdout - forecast).sum()) / denom
    return float(max(0.0, 1.0 - min(wape, 1.0)))

def trust_score_base(procs_df: pd.DataFrame) -> Tuple[float, Dict[str, float], pd.Series]:
    """Compute a composite trust score for procurement data."""
    df = procs_df[(procs_df["date"] >= FY_START) & (procs_df["date"] <= FY_END)].copy()
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month")["amount"].sum().sort_index()

    last_date = procs_df["date"].max()
    if pd.isna(last_date):
        freshness = 0.0
    else:
        days = (pd.Timestamp.today().normalize() - last_date.normalize()).days
        freshness = max(0.0, min(1.0, 1 - days / 90.0))

    today_m = pd.Period(pd.Timestamp.today(), "M")
    elapsed = len([m for m in FY_MONTHS if m <= today_m])
    observed = df["month"].nunique()
    coverage = observed / max(elapsed, 1)

    _, _, rel = crosscheck_sum(df if not df.empty else procs_df, "amount", ["category"])
    consistency = 1.0 if rel < 1e-6 else max(0.0, 1 - min(rel, 1.0))

    strength = float(min(1.0, np.log10(max(len(procs_df), 10)) / 5.0))
    backtest = backtest_forecast_accuracy(monthly, k_holdout=2)

    components = np.array([freshness, coverage, consistency, strength, backtest], dtype=float)
    components = np.clip(components, 0.0, 1.0)
    score = float(np.exp(np.mean(np.log(np.maximum(components, EPS)))))  # geometric mean
    comp = {
        "freshness": float(freshness),
        "coverage": float(coverage),
        "consistency": float(consistency),
        "statistical_strength": float(strength),
        "backtest_accuracy": float(backtest),
    }
    return score, comp, monthly

# ---------------------------------------------------------------------------------
# NLP/NER: Build domain-specific recognisers from the data (no external downloads)
# ---------------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def build_domain_nlp(pbs: pd.DataFrame, procs: pd.DataFrame):
    """Return an NLP pipeline (spaCy blank + EntityRuler) and quick-lookups.

    - Recognises: PORTFOLIO, ENTITY, OUTCOME, PROGRAM, APPRO, CATEGORY, VENDOR
    - Also regex-detects PERCENT, AMOUNT, OUTCOME numbers like "Outcome 2"
    - If spaCy is unavailable, returns (None, dictionaries) and we will fallback
    """
    lookups = {
        "portfolio": sorted({str(x) for x in pbs["portfolio"].dropna().unique()}),
        "entity": sorted({str(x) for x in pbs["entity"].dropna().unique()}),
        "outcome": sorted({str(x) for x in pbs["outcome"].dropna().unique()}),
        "program": sorted({str(x) for x in pbs["program"].dropna().unique()}),
        "appropriation_type": sorted({str(x) for x in pbs["appropriation_type"].dropna().unique()}),
        "category": sorted({str(x) for x in procs["category"].dropna().unique()}),
        "vendor": sorted({str(x) for x in procs["vendor_id"].dropna().unique()}),
    }

    if spacy is None:
        return None, lookups

    nlp: Language = spacy.blank("en")
    ruler: EntityRuler = nlp.add_pipe("entity_ruler")
    patterns = []

    def add_list(values: List[str], label: str):
        for v in values:
            if not v or v.lower() == "nan":
                continue
            patterns.append({"label": label, "pattern": v})

    add_list(lookups["portfolio"], "PORTFOLIO")
    add_list(lookups["entity"], "ENTITY")
    add_list(lookups["program"], "PROGRAM")
    add_list(lookups["appropriation_type"], "APPRO")
    add_list(lookups["category"], "CATEGORY")
    add_list(lookups["vendor"], "VENDOR")

    # Outcomes: often like "Outcome 1" or just numeric
    for o in lookups["outcome"]:
        patterns.append({"label": "OUTCOME", "pattern": str(o)})
        patterns.append({"label": "OUTCOME", "pattern": f"Outcome {o}"})

    # Intent cue phrases to aid routing (cheap and optional)
    for p in ["budget", "variance", "forecast", "meet my budget"]:
        patterns.append({"label": "INTENT_BUDGET", "pattern": p})
    for p in ["outlier", "outliers", "anomaly", "spend spike", "irregular"]:
        patterns.append({"label": "INTENT_OUTLIER", "pattern": p})
    for p in ["red flag", "red flags", "split purchase", "threshold", "spike"]:
        patterns.append({"label": "INTENT_FLAGS", "pattern": p})

    ruler.add_patterns(patterns)
    return nlp, lookups

def parse_user_query(q: str, nlp, lookups) -> dict:
    """Extract scope + knobs from an NL query using NER + regex fallback."""
    out = {
        "portfolio": None,
        "entity": None,
        "outcome": None,
        "program": None,
        "appropriation_type": None,
        "intent_hint": None,
        "percent": None,  # first % found
        "amount": None,   # first amount found
        "category": None, # a category mentioned (for overrides)
    }

    ql = q.lower()

    # Regex amounts and percents
    m = re.search(r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)(?:\s*(million|bn|billion|m|b))?", ql)
    if m:
        num = float(m.group(1).replace(",", ""))
        scale = m.group(2)
        if scale in {"m", "million"}:
            num *= 1_000_000
        elif scale in {"b", "bn", "billion"}:
            num *= 1_000_000_000
        out["amount"] = num
    pm = re.search(r"(\d{1,3})\s*%", ql)
    if pm:
        out["percent"] = int(pm.group(1))

    # Outcome by pattern "outcome <n>"
    om = re.search(r"outcome\s+(\d+)", ql)
    if om:
        out["outcome"] = om.group(1)

    # NLP pass
    if nlp is not None:
        doc = nlp(q)
        for ent in doc.ents:
            label = ent.label_.upper()
            val = ent.text.strip()
            if label == "PORTFOLIO":
                out["portfolio"] = val
            elif label == "ENTITY":
                out["entity"] = val
            elif label == "PROGRAM":
                out["program"] = val
            elif label == "OUTCOME":
                out["outcome"] = re.sub(r"^Outcome\s+", "", val, flags=re.I)
            elif label == "APPRO":
                out["appropriation_type"] = val
            elif label == "CATEGORY":
                out["category"] = val
            elif label == "INTENT_BUDGET":
                out["intent_hint"] = "budget"
            elif label == "INTENT_OUTLIER":
                out["intent_hint"] = "outlier"
            elif label == "INTENT_FLAGS":
                out["intent_hint"] = "flags"

    # Fallback keyword-based intent if none
    if not out["intent_hint"]:
        if any(k in ql for k in ["budget", "variance", "forecast"]):
            out["intent_hint"] = "budget"
        elif any(k in ql for k in ["outlier", "anomaly", "spike"]):
            out["intent_hint"] = "outlier"
        elif any(k in ql for k in ["red flag", "split purchase", "threshold"]):
            out["intent_hint"] = "flags"

    return out

# ---------------------------------------------------------------------------------
# Chat Analyzers
# ---------------------------------------------------------------------------------

@dataclass
class ChatContext:
    portfolio: Optional[str]
    entity: Optional[str]
    outcome: Optional[str]
    program: Optional[str]
    appropriation_type: Optional[str]

@dataclass
class ChatAnswer:
    ok: bool
    intent: str
    summary: str
    tables: Dict[str, pd.DataFrame]
    trust: Dict[str, float]
    provenance: Dict[str, dict]
    code_path: List[str]
    scope_guard_reason: Optional[str] = None

class Analyzer:
    key: str
    label: str
    keywords: List[str]

    def match(self, q: str) -> bool:
        ql = q.lower()
        return any(k in ql for k in self.keywords)

    def run(self, q: str, pbs: pd.DataFrame, procs: pd.DataFrame, ctx: ChatContext) -> ChatAnswer:
        raise NotImplementedError

class BudgetOutlookAnalyzer(Analyzer):
    key = "budget_outlook"
    label = "Budget outlook vs PBS (Finance)"
    keywords = ["budget", "meet my budget", "variance", "pbs", "forecast"]

    def run(self, q: str, pbs: pd.DataFrame, procs: pd.DataFrame, ctx: ChatContext) -> ChatAnswer:
        code_steps: List[str] = []
        scope = pbs.copy()
        # Apply scope filters
        for label, field, value in [
            ("portfolio", "portfolio", ctx.portfolio),
            ("entity", "entity", ctx.entity),
            ("outcome", "outcome", ctx.outcome),
            ("program", "program", ctx.program),
            ("appropriation_type", "appropriation_type", ctx.appropriation_type),
        ]:
            if value and value != "All":
                scope = scope[scope[field] == value]
                code_steps.append(f"Filter PBS: {label} == {value!r}")

        pbs_total = float(scope["budget_2024_25"].fillna(0).sum())
        code_steps.append("pbs_total = sum(scope['budget_2024_25'])")

        ytd_proc, proj_future_proc, monthly, last_completed = compute_procurement_baseline(procs)
        code_steps.append("Compute procurement baseline (YTD + projected future)")

        procurement_share_pct = st.session_state.get("proc_share_pct", 60)
        share_f = max(procurement_share_pct / 100.0, 0.01)
        baseline_total = (ytd_proc + proj_future_proc) / share_f
        variance_vs_pbs = pbs_total - baseline_total
        code_steps.append(f"baseline_total=(ytd+proj)/{share_f:.2f}; variance=pbs_total-baseline_total")

        score, comp, _ = trust_score_base(procs)

        summary = textwrap.dedent(
            f"""
            **Budget outlook (scope applied)**
            - PBS baseline 2024–25: **${pbs_total:,.0f}**
            - Forecast from procurement (YTD + seasonal avg, share {procurement_share_pct}%): **${baseline_total:,.0f}**
            - **Variance to PBS** *(positive = PBS above forecast)*: **${variance_vs_pbs:,.0f}**
            """
        ).strip()

        tables = {"Monthly procurement (YTD)": monthly.rename("amount").to_timestamp().to_frame()}
        provenance = {
            "pbs": dataset_fingerprint(scope, PBS_CSV),
            "procurement": dataset_fingerprint(procs, PROC_CSV),
            "filters": {
                "portfolio": ctx.portfolio,
                "entity": ctx.entity,
                "outcome": ctx.outcome,
                "program": ctx.program,
                "appropriation_type": ctx.appropriation_type,
            },
        }
        trust = {"score": float(score), **comp}
        return ChatAnswer(True, self.label, summary, tables, trust, provenance, code_steps)

class VendorOutliersAnalyzer(Analyzer):
    key = "vendor_outliers"
    label = "Vendor payment outliers (Finance/Operations)"
    keywords = ["outlier", "outliers", "vendor", "payment", "spend spike", "anomaly"]

    def run(self, q: str, pbs: pd.DataFrame, procs: pd.DataFrame, ctx: ChatContext) -> ChatAnswer:
        code_steps = ["df = procs.copy()"]
        df = procs.copy()
        z = modified_z_scores(df["amount"].to_numpy())
        df["mod_z"] = z
        outliers = df[(np.abs(df["mod_z"]) >= 3.5)].sort_values("mod_z", ascending=False)
        code_steps.append("Compute modified z-scores; select \n z \n >= 3.5")

        by_vendor = (
            df.groupby("vendor_id")["amount"].agg(["count", "sum", "mean", "max"]).sort_values("sum", ascending=False)
        )

        score, comp, _ = trust_score_base(procs)

        summary = (
            f"**Vendor payment outliers** detected via robust modified z‑score (\n z \n ≥ 3.5). "
            f"Flagged **{len(outliers)}** transactions of **{len(df)}** total."
        )

        # Privacy: redact vendor_id if enabled
        if st.session_state.get("redact_vendor_ids", DEFAULT_REDACT_VENDOR_IDS):
            outliers = outliers.copy()
            outliers["vendor_id"] = outliers["vendor_id"].astype(str).map(_mask_vendor_id)
            by_vendor = by_vendor.copy()
            by_vendor.index = [
                _mask_vendor_id(str(v)) for v in by_vendor.index.to_list()
            ]

        provenance = {"procurement": dataset_fingerprint(procs, PROC_CSV)}
        tables = {
            "Outlier transactions": outliers[["date", "vendor_id", "category", "amount", "mod_z"]].head(200),
            "Vendor totals (context)": by_vendor.head(50),
        }
        trust = {"score": float(score), **comp}
        return ChatAnswer(True, self.label, summary, tables, trust, provenance, code_steps)

class ProcurementRedFlagsAnalyzer(Analyzer):
    key = "proc_red_flags"
    label = "Procurement red flags (Operations)"
    keywords = ["red flag", "red flags", "procurement risk", "split purchase", "threshold"]

    def run(self, q: str, pbs: pd.DataFrame, procs: pd.DataFrame, ctx: ChatContext) -> ChatAnswer:
        code_steps = ["df = procs.copy()"]
        df = procs.copy()
        df["day"] = df["date"].dt.date

        # Thresholds can be adjusted in the sidebar; fall back to defaults
        thresholds: Dict[str, float] = st.session_state.get("split_thresholds", DEFAULT_SPLIT_PURCHASE_THRESHOLDS)

        grp = (
            df.groupby(["day", "vendor_id", "category"]).agg(
                count=("amount", "count"), total=("amount", "sum"), max_amt=("amount", "max")
            ).reset_index()
        )
        grp["threshold"] = grp["category"].map(thresholds).fillna(10_000.0)

        split_flags = grp[(grp["count"] >= 3) & (grp["total"] > grp["threshold"]) & (grp["max_amt"] < grp["threshold"])].copy()
        split_flags["rule"] = "Possible split purchases (sum > threshold, many small txns)"

        df = df.sort_values("date")
        cat_daily = (
            df.groupby(["category", df["date"].dt.to_period("D")])["amount"].sum().rename("amount").to_timestamp()
        )
        cat_df = cat_daily.reset_index().rename(columns={"date": "day"})

        # Rolling median by category
        cat_df["rolling_median"] = cat_df.groupby("category")["amount"].transform(lambda s: s.rolling(90, min_periods=30).median())
        cat_df["spike_ratio"] = cat_df["amount"] / (cat_df["rolling_median"].replace(0, np.nan))
        spike_flags = cat_df[(cat_df["rolling_median"].notna()) & (cat_df["spike_ratio"] >= 3.0)].copy()
        spike_flags["rule"] = "Category daily spend ≥ 3× 90-day rolling median"
        code_steps.append("Apply Rule 1 (split purchases) and Rule 2 (category spikes)")

        score, comp, _ = trust_score_base(procs)
        density = min(1.0, (len(split_flags) + len(spike_flags)) / max(1, len(procs)) * 100)
        trust_score = float(np.exp(np.mean(np.log(np.maximum([score, max(0.5, density)], EPS)))))  # geo-mean

        # Privacy: redact vendor_id if enabled
        if st.session_state.get("redact_vendor_ids", DEFAULT_REDACT_VENDOR_IDS):
            split_flags = split_flags.copy()
            split_flags["vendor_id"] = split_flags["vendor_id"].astype(str).map(_mask_vendor_id)

        summary = textwrap.dedent(
            f"""
            **Procurement red flags (rule‑based)**
            - Split‑purchase candidates: **{len(split_flags)}** groups
            - Category spikes: **{len(spike_flags)}** days
            """
        ).strip()

        tables = {
            "Split purchase groups": split_flags.sort_values("total", ascending=False).head(100),
            "Category spikes": spike_flags.sort_values("spike_ratio", ascending=False).head(100),
        }
        provenance = {
            "procurement": dataset_fingerprint(procs, PROC_CSV),
            "rules": {
                "split_thresholds": thresholds,
                "spike_ratio_min": 3.0,
                "rolling_window_days": 90,
                "min_days": 30,
            },
        }
        trust = {"score": trust_score, **comp, "rule_density": float(density)}
        return ChatAnswer(True, self.label, summary, tables, trust, provenance, code_steps)

ANALYZERS: List[Analyzer] = [
    BudgetOutlookAnalyzer(),
    VendorOutliersAnalyzer(),
    ProcurementRedFlagsAnalyzer(),
]

# ---------------------------------------------------------------------------------
# Tamper-evident audit log (simple hash chaining)
# ---------------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _audit_salt() -> str:
    return hashlib.sha256(("pbs-audit-salt" + str(Path.cwd())).encode("utf-8")).hexdigest()

def _audit_prev_hash() -> str:
    if not AUDIT_LOG.exists():
        return "0" * 64
    try:
        with open(AUDIT_LOG, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            back = min(size, 8192)
            f.seek(-back, 1)
            chunk = f.read().decode("utf-8", errors="ignore")
        *_, last = [line.strip() for line in chunk.splitlines() if line.strip()]
        rec = json.loads(last)
        return rec.get("hash", "0" * 64)
    except Exception:
        return "0" * 64

def write_audit(entry: dict) -> None:
    """Append an entry to the audit log with hash chaining."""
    entry = {**entry, "timestamp": datetime.utcnow().isoformat() + "Z"}
    prev = _audit_prev_hash()
    salt = _audit_salt()
    body = json.dumps(entry, sort_keys=True)
    h = hashlib.sha256((salt + prev + body).encode("utf-8")).hexdigest()
    wrapped = {**entry, "prev": prev, "hash": h}
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(wrapped) + "\n")

# ---------------------------------------------------------------------------------
# Chat orchestration (now uses NLP/NER to enrich scope + intent)
# ------------------------------------------------------------------
# ---------------



def run_chat(query: str, pbs: pd.DataFrame, procs: pd.DataFrame, ctx: ChatContext, nlp, lookups) -> ChatAnswer:
    q = (query or "").strip()
    if not q:
        return ChatAnswer(False, "", "Please type a supported question or use a suggestion.", {}, {"score": 0.0}, {}, [])

    # Parse NL to extract scope/intent hints
    extracted = parse_user_query(q, nlp, lookups)
    # Merge with current context (NLQ overrides sidebar when present)
    merged_ctx = ChatContext(
        portfolio=extracted["portfolio"] or ctx.portfolio,
        entity=extracted["entity"] or ctx.entity,
        outcome=extracted["outcome"] or ctx.outcome,
        program=extracted["program"] or ctx.program,
        appropriation_type=extracted["appropriation_type"] or ctx.appropriation_type,
    )

    # Analyzer routing (hint first, then keywords fallback)
    chosen: Optional[Analyzer] = None
    if extracted["intent_hint"]:
        hint = extracted["intent_hint"]
        for a in ANALYZERS:
            if ((hint == "budget" and isinstance(a, BudgetOutlookAnalyzer)) or
                (hint == "outlier" and isinstance(a, VendorOutliersAnalyzer)) or
                (hint == "flags" and isinstance(a, ProcurementRedFlagsAnalyzer))):
                chosen = a
                break

    if chosen is None:
        matched: List[Tuple[int, Analyzer]] = []
        ql = q.lower()
        for a in ANALYZERS:
            hits = sum(k in ql for k in a.keywords)
            if hits:
                matched.append((hits, a))
        if not matched:
            supported = ", ".join(a.label for a in ANALYZERS)
            return ChatAnswer(
                False,
                "unsupported",
                "Out of scope. I can answer vetted, retrieval‑based questions such as:\n"
                f"- {supported}\n\nTip: use the suggestion chips below.",
                {},
                {"score": 0.0},
                {},
                [],
                "No analyzer matched",
            )
        matched.sort(key=lambda x: x[0], reverse=True)
        chosen = matched[0][1]

    ans = chosen.run(q, pbs=pbs, procs=procs, ctx=merged_ctx)

    if ans.trust.get("score", 0.0) < LOW_EVIDENCE_REFUSAL:
        refusal = (
            "**Insufficient evidence to answer confidently.**\n\n"
            "- Data freshness/coverage/consistency appear low.\n"
            "- Please update the procurement data, then try again."
        )
        ans = ChatAnswer(
            False,
            chosen.label,
            refusal,
            {},
            ans.trust,
            ans.provenance,
            ans.code_path,
            "Low trust score",
        )

    write_audit(
        {
            "type": "chat_answer",
            "query": q,
            "intent": ans.intent,
            "ok": ans.ok,
            "trust": ans.trust,
            "provenance": ans.provenance,
            "code_path": ans.code_path,
            "nlp": extracted,
            "scope_guard": ans.scope_guard_reason,
        }
    )
    return ans

# ---------------------------------------------------------------------------------
# UI: Sidebar Scope & Controls
# ---------------------------------------------------------------------------------

def sidebar_scope(pbs: pd.DataFrame, key_prefix: str = "scope_") -> Tuple[pd.DataFrame, ChatContext]:
    st.sidebar.header("PBS Scope (baseline)")
    scope = pbs.copy()

    def pick(label: str, choices: List[str], key_suffix: str) -> str:
        opts = ["All"] + sorted([c for c in choices if pd.notna(c)])
        return st.sidebar.selectbox(label, opts, key=f"{key_prefix}{key_suffix}")

    portfolio = pick("Portfolio", scope["portfolio"].unique().tolist(), "portfolio")
    if portfolio != "All":
        scope = scope[scope["portfolio"] == portfolio]

    entity = pick("Entity", scope["entity"].unique().tolist(), "entity")
    if entity != "All":
        scope = scope[scope["entity"] == entity]

    outcome = pick("Outcome", scope["outcome"].unique().tolist(), "outcome")
    if outcome != "All":
        scope = scope[scope["outcome"] == outcome]

    program = pick("Program", scope["program"].unique().tolist(), "program")
    if program != "All":
        scope = scope[scope["program"] == program]

    appr = pick("Appropriation type", scope["appropriation_type"].unique().tolist(), "appr")
    if appr != "All":
        scope = scope[scope["appropriation_type"] == appr]

    # Editable thresholds for red-flags
    st.sidebar.markdown("---")
    with st.sidebar.expander("Red‑flag thresholds (split‑purchase)"):
        current = st.session_state.get("split_thresholds", DEFAULT_SPLIT_PURCHASE_THRESHOLDS.copy())
        # include dynamic expense types if present
        dynamic_types = scope["expense_type"].dropna().unique().tolist() if "expense_type" in scope.columns else []
        for k in sorted(set(list(DEFAULT_SPLIT_PURCHASE_THRESHOLDS.keys()) + dynamic_types)):
            current[k] = float(
                st.number_input(f"{k} threshold ($)", value=float(current.get(k, 10_000.0)), step=1000.0, key=f"thr_{k}")
            )
        st.session_state["split_thresholds"] = current

    # Privacy & compliance
    st.sidebar.markdown("---")
    with st.sidebar.expander("Privacy & Compliance"):
        st.session_state["redact_vendor_ids"] = st.checkbox("Redact vendor IDs in tables", value=DEFAULT_REDACT_VENDOR_IDS)
        st.session_state["tamper_evident_audit"] = st.checkbox("Tamper‑evident audit log (hash chain)", value=DEFAULT_TAMPER_EVIDENT_AUDIT)
        st.caption("Note: No personal information is processed. Vendor identifiers may be masked.")

    ctx = ChatContext(
        portfolio=None if portfolio == "All" else portfolio,
        entity=None if entity == "All" else entity,
        outcome=None if outcome == "All" else outcome,
        program=None if program == "All" else program,
        appropriation_type=None if appr == "All" else appr,
    )
    return scope, ctx

# ---------------------------------------------------------------------------------
# UI: Tabs
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------
# UI styling (drop-in)
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --gov-blue: #0e5a8a;
        --gov-sky: #f2f8fc;
        --gov-green: #2d6a4f;
        --gov-grey: #6b7280;
    }
    /* Global reset */
    .stApp { background-color: var(--gov-sky); }
    header, .stApp header {visibility: hidden;}

    /* App banner */
    .app-banner {
        background: var(--gov-blue);
        color: white;
        padding: 1rem 1.25rem;
        border-radius: .5rem;
        margin-bottom: 1rem;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.15);
    }

    /* Classification badge */
    .classif {
        font-size: .8rem;
        font-weight: 600;
        letter-spacing: .5px;
        background: var(--gov-green);
        padding: .2rem .6rem;
        border-radius: .25rem;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: white;
        padding: 1rem;
        border-radius: .75rem;
        box-shadow: 0px 2px 4px rgba(0,0,0,0.08);
    }

    /* Chat messages */
    .stChatMessage {
        background: white;
        padding: .8rem 1rem;
        border-radius: .75rem;
        margin: .5rem 0;
        box-shadow: 0px 1px 3px rgba(0,0,0,0.1);
    }

    /* Expander */
    [data-testid="stExpander"] {
        border-radius: .5rem;
        border: 1px solid #ddd;
    }

    /* Buttons */
    .stButton button {
        border-radius: .5rem;
        font-weight: 600;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Theme toggle in sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    theme_choice = st.radio("🎨 Theme", ["Gov Default", "Light", "Dark"], index=0)
    if theme_choice == "Dark":
        st.markdown("<style> .stApp {background:#111; color:#eee;} </style>", unsafe_allow_html=True)
    elif theme_choice == "Light":
        st.markdown("<style> .stApp {background:#fff; color:#111;} </style>", unsafe_allow_html=True)


def simulator_tab(pbs_scope: pd.DataFrame, procs: pd.DataFrame) -> None:
    st.subheader("What‑if Simulator: Procurement → PBS Budget Impact (2024–25)")
    pbs_total = float(pbs_scope["budget_2024_25"].fillna(0).sum())

    # Sidebar: forecast assumptions
    st.sidebar.markdown("---")
    st.sidebar.caption("Forecast assumptions")
    overall_pct = st.sidebar.slider("Overall procurement spend change (%)", -50, 100, 10, step=1, key="sim_overall_pct")
    st.sidebar.markdown("**Per‑category overrides**")
    categories = sorted(procs["category"].dropna().unique().tolist())
    cat_overrides = {c: st.sidebar.slider(f"{c} change (%)", -50, 150, 0, step=5, key=f"sim_cat_{c}") for c in categories}
    price_infl = st.sidebar.slider("Price inflation (%)", -5, 20, 3, step=1, key="sim_price_infl")
    savings = st.sidebar.slider("Efficiency savings (%)", 0, 30, 5, step=1, key="sim_savings")

    # Run
    
    if st.sidebar.button("Run Simulation", type="primary", key="sim_run_btn"):
        st.metric("PBS baseline (2024–25)", f"${pbs_total:,.0f}")
        ytd, proj_future, monthly, last_completed = compute_procurement_baseline(procs)
        share_f = max(st.session_state.get("proc_share_pct", 60) / 100.0, 0.01)
        baseline_total = (ytd + proj_future) / share_f

        def factor(cat: str) -> float:
            cat_pct = cat_overrides.get(cat, 0.0) / 100.0
            return (1 + cat_pct) * (1 + price_infl / 100.0) * (1 + overall_pct / 100.0) * (1 - savings / 100.0)

        fy_mask = (procs["date"] >= FY_START) & (procs["date"] <= FY_END)
        future_mask = procs["date"].dt.to_period("M") > last_completed
        future = procs[fy_mask & future_mask].copy()
        if not future.empty:
            future["scenario_amount"] = future.apply(lambda r: r["amount"] * factor(r["category"]), axis=1)
            scen_future_total = float(future["scenario_amount"].sum())
        else:
            scen_future_total = 0.0
        scen_total = (ytd + scen_future_total) / share_f
        variance_to_pbs = pbs_total - scen_total

        # KPI Row
        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline forecast", f"${baseline_total:,.0f}")
        c2.metric("Scenario forecast", f"${scen_total:,.0f}")
        c3.metric(
            "Variance to PBS",
            f"${variance_to_pbs:,.0f}",
            delta=f"{(variance_to_pbs / max(pbs_total, 1.0)) * 100:+.1f}%",
            delta_color="normal",
        )

        score, comp, _ = trust_score_base(procs)

        _section_divider("Trust & Auditability", "🔎")
        st.progress(score, text=f"Composite trust score: {score:.2f}")
        st.json(comp, expanded=False)

        with st.expander("Show calculation details"):
            st.write("**Assumptions & formulas**")
            st.markdown(
                "- **PBS baseline** = sum of 2024–25 expenses in scope \n"
                "- **Forecast total** = (YTD + projected) ÷ procurement_share \n"
                "- **Scenario factor** = (1+category%) × (1+inflation%) × (1+overall%) × (1−savings%) \n"
                "- **Trust score** = geometric mean of freshness, coverage, consistency, statistical strength, back‑test accuracy"
            )

        # Plotly chart for monthly series
        if not monthly.empty:
            df_plot = monthly.rename("amount").to_timestamp().reset_index()
            fig = px.line(df_plot, x="month", y="amount", markers=True, title="Monthly Procurement (YTD)")
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=360)
            st.plotly_chart(fig, use_container_width=True)

        # Download scoped PBS
        with st.expander("Download scoped PBS lines"):
            buf = io.StringIO()
            pbs_scope.to_csv(buf, index=False)
            st.download_button(
                "Download CSV",
                data=buf.getvalue(),
                file_name="pbs_scope.csv",
                mime="text/csv",
                use_container_width=True,
            )

        write_audit(
            {
                "type": "what_if_simulation",
                "pbs_budget_2024_25": pbs_total,
                "inputs": {
                    "procurement_share_pct": st.session_state.get("proc_share_pct", 60),
                    "overall_pct": overall_pct,
                    "category_overrides": cat_overrides,
                    "price_inflation_pct": price_infl,
                    "savings_offset_pct": savings,
                },
                "scenario": {
                    "baseline_total_from_proc": baseline_total,
                    "scenario_total_from_proc": scen_total,
                    "variance_to_pbs": variance_to_pbs,
                },
                "trust": {"score": score, **comp},
            }
        )
        st.success("Simulation complete. Audit log updated.")

def chat_tab(pbs_scope: pd.DataFrame, procs: pd.DataFrame, ctx: ChatContext, nlp, lookups) -> None:
    st.subheader("Chat (Vetted, Retrieval‑Only, NLP‑assisted)")
    st.caption(
        "Answers are computed from the loaded datasets with rule‑based analyzers, "
        "evidence tables, and trust scoring. Natural‑language queries are parsed to auto‑apply scope where possible."
    )

    # Persist scope selections in session for audit
    st.session_state["portfolio"] = ctx.portfolio
    st.session_state["entity"] = ctx.entity
    st.session_state["outcome"] = ctx.outcome
    st.session_state["program"] = ctx.program
    st.session_state["appropriation_type"] = ctx.appropriation_type

    # Suggestion chips
    chips = st.columns(3)
    if chips[0].button("Am I going to meet my budget this year?", key="chip_budget"):
        st.session_state["chat_prefill"] = "Am I going to meet my budget this year?"
    if chips[1].button("Show me vendor payment outliers", key="chip_outliers"):
        st.session_state["chat_prefill"] = "Show me vendor payment outliers"
    if chips[2].button("Are there any red flags in our procurement data?", key="chip_flags"):
        st.session_state["chat_prefill"] = "Are there any red flags in our procurement data?"

    # History
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("trust") is not None:
            st.caption(f"Trust score: {msg['trust']:.2f}")

    # Chat input
    default_prefill = st.session_state.pop("chat_prefill", None)
    user_q = st.chat_input("Ask a supported question…", key="chat_input")
    if user_q is None and default_prefill:
        user_q = default_prefill

    if user_q:
        st.session_state["chat_history"].append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        ans = run_chat(user_q, pbs=pbs_scope, procs=procs, ctx=ctx, nlp=nlp, lookups=lookups)

        with st.chat_message("assistant"):
            st.markdown(ans.summary)
            if ans.ok:
                st.caption(f"Trust score: {ans.trust.get('score', 0.0):.2f}")
                # NER extraction preview
                with st.expander("NLP-extracted scope & knobs"):
                    last = None
                    try:
                        # get last audit entry to show nlp extractions
                        with open(AUDIT_LOG, "r", encoding="utf-8") as f:
                            for line in f:
                                pass
                            last = json.loads(line)
                    except Exception:
                        pass
                    if last and "nlp" in last:
                        st.json(last["nlp"], expanded=False)
                for name, tdf in ans.tables.items():
                    with st.expander(f"{name}"):
                        st.dataframe(tdf, use_container_width=True)
                with st.expander("Provenance & Scope"):
                    st.json(ans.provenance, expanded=False)
                with st.expander("How this was computed (code path)"):
                    st.code("\n".join(ans.code_path), language="text")

        st.session_state["chat_history"].append(
            {"role": "assistant", "content": ans.summary, "trust": ans.trust.get("score", 0.0)}
        )

    # Audit log download
    st.markdown("---")
    if AUDIT_LOG.exists():
        with open(AUDIT_LOG, "r", encoding="utf-8") as f:
            content = f.read()
        st.download_button(
            "Download audit log (JSONL)",
            data=content,
            file_name=AUDIT_LOG.name,
            mime="application/jsonl",
            key="dl_audit",
        )

# ---------------------------------------------------------------------------------
# App Init & Layout polish (government-friendly look)
# ---------------------------------------------------------------------------------

def data_health_panel(pbs: pd.DataFrame, procs: pd.DataFrame) -> None:
    """Small panel that shows dataset health (rows, date ranges, missing %)."""
    _section_divider("Data Health", "🩺")
    c1, c2, c3 = st.columns(3)
    c1.metric("PBS rows", f"{len(pbs):,}")
    c2.metric("Procurement rows", f"{len(procs):,}")
    if not procs.empty:
        dmin = pd.to_datetime(procs["date"]).min()
        dmax = pd.to_datetime(procs["date"]).max()
        c3.metric("Procurement date range", f"{dmin.date()} → {dmax.date()}")

    # Missing value overview (quick)
    with st.expander("Missing values overview (top 10 columns)"):
        mv_pbs = pbs.isna().mean().sort_values(ascending=False).head(10).rename("missing_ratio").to_frame()
        mv_procs = procs.isna().mean().sort_values(ascending=False).head(10).rename("missing_ratio").to_frame()
        st.write("**PBS**")
        st.dataframe((mv_pbs * 100).round(2), use_container_width=True)
        st.write("**Procurement**")
        st.dataframe((mv_procs * 100).round(2), use_container_width=True)

def main() -> None:
    st.set_page_config(
        page_title="PBS What‑if + Vetted Chat (Retrieval‑Only)",
        layout="wide",
        page_icon="📊",
    )

    # Subtle CSS polish / sober AU Gov style
    st.markdown(
        """
        <style>
        :root {
          --gov-blue: #0e5a8a;
          --gov-sky: #e6f0f6;
        }
        header, .stApp header {visibility: hidden;}
        .app-banner {background: var(--gov-blue); color: white; padding: .6rem 1rem; border-radius: .25rem;}
        .classif {font-size: .75rem; letter-spacing: .5px; background:#2d6a4f; color:#fff; padding:.15rem .5rem; border-radius:.15rem;}
        .footer-note {color:#6b7280; font-size:.8rem;}
        [data-testid="stMetricValue"] { font-weight: 700; }
        .st-emotion-cache-1r4qj8v p, .stMarkdown p { line-height: 1.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="app-banner">
          <div><span class="classif">OFFICIAL</span> PBS What‑if + Vetted Chat</div>
          <div style="opacity:.9;font-size:.9rem;">Retrieval‑only analytics with auditability. No free‑text generation.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.title("📊 PBS What‑if + Vetted Chat (Retrieval‑Only, NLP‑assisted)")
    st.caption("Prototype for analytical decision support. For internal use only.")

    with st.sidebar:
        st.subheader("Settings")
        st.session_state.setdefault("proc_share_pct", 60)
        st.session_state["proc_share_pct"] = st.slider(
            "Procurement share of total spend (%)", 10, 90, st.session_state["proc_share_pct"], key="share_slider"
        )
        if st.button("Generate synthetic procurement CSV", key="btn_gen_sample"):
            ensure_procurement_csv(PROC_CSV, force_generate=True)
            st.success("Generated sample_procurements.csv.")
            st.rerun()
        st.caption("Provide a procurement CSV named **sample_procurements.csv** with columns: date, category, vendor_id, amount.")
        st.markdown("---")
        st.caption("This app is retrieval‑only (no free‑text LLM). Outliers use modified z‑score (\n z \n ≥ 3.5).")

    # Load data
    with st.spinner("Loading PBS and procurement data..."):
        pbs, _ = load_pbs(PBS_CSV)
        procs = load_procurement(PROC_CSV)

    # Build NLP from data (cached). If spaCy missing, nlp will be None but lookups still returned.
    nlp, lookups = build_domain_nlp(pbs, procs)

    # Data health
    data_health_panel(pbs, procs)

    # Shared scope
    pbs_scope, ctx = sidebar_scope(pbs, key_prefix="scope_")

    # Tabs
    tab1, tab2 = st.tabs(["What‑if Simulator", "Chat (Vetted)"])
    with tab1:
        simulator_tab(pbs_scope, procs)
    with tab2:
        chat_tab(pbs_scope, procs, ctx, nlp, lookups)

    st.markdown("---")
    st.markdown(
        "<div class='footer-note'>© Australian Government (prototype). This tool provides indicative analytics only and does not replace Finance advice. Data remains on-device; audit logs are tamper‑evident.</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
