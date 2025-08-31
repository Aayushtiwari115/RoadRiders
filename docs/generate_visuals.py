# docs/generate_visuals.py
# ---------------------------------------------------------------------
# Build gorgeous, publication-ready visuals from your PBS (2024–25)
# and Procurement CSVs. All plots use Australian currency (A$) and
# an accessibility-conscious palette with clear, annotated graphics.
#
# Run:  python docs/generate_visuals.py
# Deps: pandas numpy matplotlib seaborn requests
# ---------------------------------------------------------------------

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as patches
import seaborn as sns
import requests

# -----------------------
# Paths & constants
# -----------------------
ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DOCS.mkdir(parents=True, exist_ok=True)

PBS_CSV = ROOT / "2024-25-pbs-program-expense-line-items.csv"
PROC_CSV = ROOT / "sample_procurements.csv"

PBS_DOWNLOAD_URL = (
    "https://data.gov.au/data/dataset/6aeeefa5-65c4-45fb-bc7e-c8f33c7ffa44/"
    "resource/935c6502-fff0-48a0-bb20-3939687a5eda/download/2024-25-pbs-program-expense-line-items.csv"
)

FY_START = pd.Timestamp("2024-07-01")
FY_END = pd.Timestamp("2025-06-30")
FY_MONTHS = pd.period_range(pd.Period("2024-07","M"), pd.Period("2025-06","M"))
EPS = 1e-12

DPI = 220               # crisp
EXPORT_SVG = False      # set True if you want scalable vector files too

# Palette (accessible)
COL = {
    "brand": "#0F62FE",
    "brand_2": "#2F80ED",
    "ink": "#0B0C0C",
    "muted": "#5f6d7a",
    "good": "#1e8e3e",
    "good_bg": "#E6F4EA",
    "bad": "#B00020",
    "bad_bg": "#FDECEA",
    "grid": "#E5E7EB",
    "pbs": "#93c5fd",
    "proc": "#0F62FE",
    "accent": "#6F52ED",
}

# Matplotlib/Seaborn theme
sns.set_theme(
    style="whitegrid",
    rc={
        "figure.titlesize": 18,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.edgecolor": COL["grid"],
        "grid.color": COL["grid"],
        "grid.alpha": 0.35,
        "text.color": COL["ink"],
        "axes.labelcolor": COL["ink"],
        "axes.titleweight": "bold",
    },
)
plt.rcParams["font.family"] = "DejaVu Sans"  # broadly available


# -----------------------
# Utilities
# -----------------------
def a_dollar(x: float) -> str:
    try:
        return f"A${x:,.0f}"
    except Exception:
        return "A$0"

def savefig(fig: plt.Figure, name: str):
    png = DOCS / f"{name}.png"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", facecolor="white")
    if EXPORT_SVG:
        fig.savefig(DOCS / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)

def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1", engine="python", low_memory=False)

def ensure_pbs(path: Path) -> pd.DataFrame:
    if not path.exists():
        with requests.get(PBS_DOWNLOAD_URL, stream=True, timeout=30) as r:
            r.raise_for_status()
            path.write_bytes(r.content)
            print("Downloaded PBS CSV.")
    df = read_csv_with_fallback(path)
    # normalise numeric year columns
    year_cols = [c for c in df.columns if any(y in c for y in ["2023","2024","2025","2026","2027"])]
    for c in year_cols:
        s = (df[c].astype(str).str.replace(",","",regex=False).str.replace("$","",regex=False)
             .str.strip().replace({"":"nan","nfp":"nan","NFP":"nan"}))
        df[c] = pd.to_numeric(s, errors="coerce")

    def first_match(keys: List[str]) -> str:
        for col in df.columns:
            if all(k in col.lower() for k in keys):
                return col
        raise KeyError(keys)

    df["portfolio"] = df[first_match(["portfolio"])]
    df["program"] = df[first_match(["program"])]
    df["outcome"] = df[first_match(["outcome"])]
    df["entity"] = df[first_match(["department"])] if any("department" in c.lower() for c in df.columns) else df[first_match(["entity"])]
    y24 = [c for c in df.columns if ("2024-25" in c) or ("2024/25" in c) or ("2024–25" in c)]
    if not y24:
        raise SystemExit("PBS CSV missing a 2024–25 budget column.")
    df["budget_2024_25"] = df[y24[0]]
    return df

def ensure_proc(path: Path) -> pd.DataFrame:
    if not path.exists():
        rng = pd.date_range(FY_START, FY_END, freq="D")
        cats = ["ICT","Construction","Consulting","Office Supplies","Travel"]
        vendors = [f"VEND-{i:03d}" for i in range(1,61)]
        rs = np.random.RandomState(42)
        base={"ICT":12000,"Construction":45000,"Consulting":16000,"Office Supplies":2200,"Travel":3800}
        rows=[]
        for d in rng:
            for _ in range(6):
                c = rs.choice(cats, p=[0.30,0.22,0.20,0.18,0.10])
                amt = float(rs.lognormal(np.log(base[c]), 0.6))
                rows.append({"date":d.strftime("%Y-%m-%d"),"vendor_id":rs.choice(vendors),"category":c,"amount":round(amt,2)})
        pd.DataFrame(rows).to_csv(path, index=False)
        print("Generated synthetic procurement CSV.")
    df = pd.read_csv(path, parse_dates=["date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df.dropna(subset=["date","category","vendor_id","amount"]).copy()

def compute_baseline(procs: pd.DataFrame) -> Tuple[float,float,pd.Series,pd.Period]:
    df = procs[(procs["date"]>=FY_START)&(procs["date"]<=FY_END)].copy()
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month")["amount"].sum().sort_index()
    if monthly.empty:
        return 0.0,0.0,monthly,pd.Period("2024-06","M")
    last = monthly.index.max()
    ytd = float(monthly.loc[monthly.index<=last].sum())
    recent = monthly.loc[monthly.index<=last].tail(3)
    avg = float(recent.mean()) if len(recent) else float(monthly.mean())
    months_left = [m for m in FY_MONTHS if m > last]
    proj = avg * len(months_left)
    return ytd, proj, monthly, last

def modified_z_scores(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad

def trust_components(procs: pd.DataFrame) -> Dict[str,float]:
    df = procs[(procs["date"]>=FY_START)&(procs["date"]<=FY_END)].copy()
    df["month"]=df["date"].dt.to_period("M")
    monthly=df.groupby("month")["amount"].sum().sort_index()

    last = procs["date"].max()
    freshness = 0.0 if pd.isna(last) else max(0.0, min(1.0, 1 - (pd.Timestamp.today().normalize() - last.normalize()).days/90.0))

    today_m = pd.Period(pd.Timestamp.today(),"M")
    elapsed = len([m for m in FY_MONTHS if m<=today_m])
    coverage = df["month"].nunique()/max(elapsed,1)

    direct = float(df["amount"].sum()) if not df.empty else 0.0
    by_cat = float(df.groupby("category")["amount"].sum().sum()) if not df.empty else direct
    denom=max(abs(direct),1.0); rel=abs(direct-by_cat)/denom
    consistency = 1.0 if rel < 1e-6 else max(0.0, 1-min(rel,1.0))

    strength = float(min(1.0, np.log10(max(len(procs),10))/5.0))
    if len(monthly)<=5:
        back=0.6
    else:
        train=monthly.iloc[:-2]; hold=monthly.iloc[-2:]
        avg=float(train.tail(3).mean()) if len(train)>=3 else float(train.mean())
        fc=pd.Series([avg]*len(hold), index=hold.index)
        denom=max(float(hold.sum()),1.0); wape=float(np.abs(hold-fc).sum())/denom
        back=float(max(0.0, 1.0-min(wape,1.0)))

    comps = dict(freshness=freshness, coverage=coverage, consistency=consistency,
                 statistical_strength=strength, backtest_accuracy=back)
    score = float(np.exp(np.mean(np.log(np.maximum(np.array(list(comps.values())), EPS)))))
    comps["score"]=score
    return comps


# -----------------------
# Plots (beautiful & annotated)
# -----------------------
def plot_monthly_line(monthly: pd.Series, ytd: float, proj: float, last: pd.Period):
    ts = monthly.rename("amount").to_timestamp()
    fig, ax = plt.subplots(figsize=(12,5.6))
    # past
    ax.plot(ts.index, ts.values, marker="o", color=COL["brand"], lw=2.2, label="Actual (YTD)")
    # projection band
    all_months = pd.period_range(pd.Period("2024-07","M"), pd.Period("2025-06","M")).to_timestamp()
    future = all_months[all_months > last.to_timestamp()]
    if len(future):
        avg = float(ts.tail(3).mean()) if len(ts)>=3 else float(ts.mean())
        ax.plot(future, [avg]*len(future), color=COL["accent"], lw=2.0, ls="--", label="Projection (avg of last 3 months)")
        ax.axvspan(future.min(), future.max(), color=COL["accent"], alpha=0.08)
    # labels & grid
    ax.set_title("Monthly procurement trend (YTD)")
    ax.set_ylabel("Amount (A$)")
    ax.set_xlabel("Month")
    ax.grid(True, axis="y", alpha=.35)
    # annotations
    ax.text(0.01, 1.06, f"YTD: {a_dollar(ytd)}   Projected remainder: {a_dollar(proj)}",
            transform=ax.transAxes, fontsize=11, color=COL["muted"])
    ax.legend(frameon=False, loc="upper left")
    savefig(fig, "monthly_line")

def plot_results_table(pbs: pd.DataFrame, procs: pd.DataFrame, ytd: float, proj: float):
    pbs_total = float(pbs["budget_2024_25"].fillna(0).sum()) or 1.0
    observed_share = (ytd+proj)/pbs_total
    scen_total = (ytd+proj)/max(observed_share,0.01)

    top = pbs.groupby("program")["budget_2024_25"].sum().sort_values(ascending=False).head(4)
    total_scope = float(pbs["budget_2024_25"].sum()) or 1.0
    alloc = (top/total_scope)*scen_total
    variance = top - alloc

    df = pd.DataFrame({"PBS Baseline": top, "Procurement Scenario": alloc, "Variance": variance})
    df = df.sort_values("PBS Baseline", ascending=True)

    fig, ax = plt.subplots(figsize=(12,6))
    y = np.arange(len(df))
    ax.barh(y-0.22, df["PBS Baseline"], height=0.44, label="PBS Baseline", color=COL["pbs"])
    ax.barh(y+0.22, df["Procurement Scenario"], height=0.44, label="Procurement", color=COL["proc"])
    ax.set_yticks(y); ax.set_yticklabels(df.index)
    ax.set_xlabel("A$")
    ax.set_title("Results — PBS vs Procurement (Top Programs)")
    ax.legend(frameon=False, loc="lower right")
    # annotate variance badges
    for i, (pb, sc, var) in enumerate(zip(df["PBS Baseline"], df["Procurement Scenario"], df["Variance"])):
        sign = "pos" if var >= 0 else "neg"
        txt = f"{'+' if var>=0 else ''}{a_dollar(var)}"
        ax.text(max(pb, sc)*1.02, y[i], txt,
                va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.2",
                         fc=COL["good_bg"] if sign=="pos" else COL["bad_bg"],
                         ec=COL["good"] if sign=="pos" else COL["bad"]),
                color=COL["good"] if sign=="pos" else COL["bad"], fontsize=10)
    savefig(fig, "results_table")

def plot_outliers(procs: pd.DataFrame):
    df = procs.copy()
    z = modified_z_scores(df["amount"].to_numpy())
    df["mod_z"] = z
    # pick top-N outliers for callouts
    top = df.reindex(np.argsort(-np.abs(df["mod_z"].values))).head(12)

    fig, ax = plt.subplots(figsize=(11.5,6.2))
    ax.scatter(df["mod_z"], df["amount"], s=12, alpha=0.5, color=COL["brand_2"], edgecolors="none", label="Transactions")
    ax.scatter(top["mod_z"], top["amount"], s=30, color=COL["bad"], alpha=0.9, label="Top outliers")
    ax.axvline(3.5, color=COL["bad"], linestyle="--", lw=1.2)
    ax.axvline(-3.5, color=COL["bad"], linestyle="--", lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Modified z‑score (|z| ≥ 3.5 flagged)")
    ax.set_ylabel("Amount (A$, log scale)")
    ax.set_title("Vendor outliers — amount vs modified z‑score")
    # Callouts
    for _, r in top.iterrows():
        ax.annotate(r["vendor_id"],
                    xy=(r["mod_z"], r["amount"]),
                    xytext=(5,5), textcoords="offset points",
                    fontsize=9, color=COL["ink"])
    ax.legend(frameon=False, loc="lower right")
    savefig(fig, "outliers_scatter")

def plot_redflags_heatmap(procs: pd.DataFrame):
    df = procs.copy()
    df["day"] = df["date"].dt.to_period("D")
    cat_daily = df.groupby(["category", "day"])["amount"].sum().rename("amount").reset_index()
    cat_daily["day"] = cat_daily["day"].dt.to_timestamp()
    cat_daily["rolling_median"] = cat_daily.groupby("category")["amount"].transform(lambda s: s.rolling(90, min_periods=30).median())
    cat_daily["ratio"] = cat_daily["amount"]/(cat_daily["rolling_median"].replace(0, np.nan))
    spikes = cat_daily[(cat_daily["rolling_median"].notna()) & (cat_daily["ratio"] >= 3.0)].copy()
    if spikes.empty:
        # build an empty grid for consistent output
        cats = sorted(df["category"].unique())
        piv = pd.DataFrame(index=cats, columns=[], data=[])
    else:
        spikes["week"] = spikes["day"].dt.to_period("W").dt.start_time
        piv = spikes.pivot_table(index="category", columns="week", values="ratio", aggfunc="count").fillna(0.0)

    fig, ax = plt.subplots(figsize=(12.5,5.8))
    if piv.size > 0:
        sns.heatmap(piv, ax=ax, cmap="YlOrRd", cbar_kws={"label":"Spike count"})
    ax.set_title("Red flags heatmap — category spikes by week (≥ 3× rolling median)")
    ax.set_xlabel("Week"); ax.set_ylabel("Category")
    savefig(fig, "redflags_spikes_heatmap")

def plot_trust_card(procs: pd.DataFrame):
    comps = trust_components(procs)
    score = comps.pop("score", 0.0)

    # Figure with donut + bars
    fig = plt.figure(figsize=(11.5,7.0))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.1, 1.4], width_ratios=[1.1, 1.0], hspace=0.3, wspace=0.25)

    # Donut gauge
    ax0 = fig.add_subplot(gs[:,0])
    ax0.axis("equal")
    theta = score * 360.0
    # background ring
    ring_bg = patches.Wedge((0,0), 1.0, 0, 360, width=0.28, facecolor="#F2F4F7", edgecolor="none")
    ax0.add_patch(ring_bg)
    # filled arc
    ring_fg = patches.Wedge((0,0), 1.0, 0, theta, width=0.28, facecolor=COL["brand"], edgecolor="none")
    ax0.add_patch(ring_fg)
    ax0.text(0, 0.05, f"{score*100:.0f}", ha="center", va="center", fontsize=36, fontweight="bold", color=COL["ink"])
    ax0.text(0, -0.22, "Trust Score", ha="center", va="center", fontsize=12, color=COL["muted"])
    ax0.set_xlim(-1.2,1.2); ax0.set_ylim(-1.2,1.2)
    ax0.axis("off")

    # Component bars
    ax1 = fig.add_subplot(gs[0,1])
    labels = [k.replace("_"," ").title() for k in comps.keys()]
    vals = np.array(list(comps.values())) * 100.0
    y = np.arange(len(labels))
    ax1.barh(y, vals, color=COL["brand"])
    ax1.set_yticks(y); ax1.set_yticklabels(labels)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Score (0–100)")
    ax1.grid(True, axis="x", alpha=.25)
    for i, v in enumerate(vals):
        ax1.text(v + 1.5, i, f"{v:.0f}", va="center", fontsize=10, color=COL["muted"])

    # Legend / explainer
    ax2 = fig.add_subplot(gs[1,1])
    ax2.axis("off")
    expl = (
        "Freshness • Coverage • Consistency • Statistical Strength • Back‑test\n"
        "Geometric mean of components; evidence derived from your datasets only."
    )
    ax2.text(0.0, 0.8, "Why this score?", fontsize=13, fontweight="bold")
    ax2.text(0.0, 0.55, expl, fontsize=11, color=COL["muted"])
    ax2.text(0.0, 0.25, "Provenance and audit logs are available in the app.", fontsize=11, color=COL["muted"])
    savefig(fig, "trust_card")

def plot_vendor_pareto(procs: pd.DataFrame):
    fy = procs[(procs["date"]>=FY_START)&(procs["date"]<=FY_END)]
    s = fy.groupby("vendor_id")["amount"].sum().sort_values(ascending=False)
    if s.empty:
        return
    n = min(20, len(s))
    top = s.head(n)
    cumshare = top.cumsum() / s.sum()

    fig, ax1 = plt.subplots(figsize=(12.5,6))
    ax1.bar(top.index, top.values, color=COL["brand_2"])
    ax1.set_ylabel("Spend (A$)")
    ax1.tick_params(axis='x', rotation=45, ha='right')
    ax1.set_title("Top vendors — Pareto chart with cumulative share")

    ax2 = ax1.twinx()
    ax2.plot(top.index, cumshare.values, color=COL["accent"], marker="o", lw=2)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.set_ylabel("Cumulative share")
    ax2.set_ylim(0,1.05)
    ax1.grid(True, axis="y", alpha=.35)
    savefig(fig, "vendor_pareto")

def plot_category_quarter(procs: pd.DataFrame):
    fy = procs[(procs["date"]>=FY_START)&(procs["date"]<=FY_END)].copy()
    if fy.empty:
        return
    this_q = pd.Timestamp.today().quarter
    fy["quarter"]=fy["date"].dt.quarter
    qdf=fy[fy["quarter"]==this_q]
    by_cat=qdf.groupby("category")["amount"].sum().sort_values(ascending=True)
    total=float(by_cat.sum()) or 1.0
    share=(by_cat/total)*100.0

    fig, ax = plt.subplots(figsize=(11.5,6))
    ax.barh(by_cat.index, by_cat.values, color=COL["brand"])
    ax.set_xlabel("Spend (A$)")
    ax.set_title(f"Category drivers — Quarter {this_q}")
    for i,(v,p) in enumerate(zip(by_cat.values, share.values)):
        ax.text(v*1.01, i, f"{p:.1f}%", va="center", fontsize=10, color=COL["muted"])
    savefig(fig, "category_quarter")


# -----------------------
# Main
# -----------------------
def main():
    print("Loading datasets…")
    pbs = ensure_pbs(PBS_CSV)
    procs = ensure_proc(PROC_CSV)

    print("Computing baseline…")
    ytd, proj, monthly, last = compute_baseline(procs)

    print("Rendering visuals…")
    if not monthly.empty:
        plot_monthly_line(monthly, ytd, proj, last)
    plot_results_table(pbs, procs, ytd, proj)
    plot_outliers(procs)
    plot_redflags_heatmap(procs)
    plot_trust_card(procs)
    plot_vendor_pareto(procs)
    plot_category_quarter(procs)

    print(f"✅ All visuals saved to {DOCS.resolve()} (DPI={DPI}, SVG={EXPORT_SVG})")

if __name__ == "__main__":
    main()