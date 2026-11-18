"""
Momentum Stock Picker
=====================
Scans the S&P 500 + Nasdaq-100, scores every stock on intermediate-term
momentum (built for a 6-12 month hold), backtests the strategy, and writes:

  output/picks_YYYY-MM-DD.csv   every scored stock, ranked
  output/dashboard.html         self-contained dashboard (open it, or publish_dashboard.bat pushes it to GitHub Pages)

Usage
-----
  python momentum_picker.py                      # defaults
  python momentum_picker.py --top 15 --sector-cap 3
  python momentum_picker.py --tickers my_list.txt  # one ticker per line
  python momentum_picker.py --no-backtest          # faster

Not investment advice. Momentum is a well-documented factor, but it has sharp
crashes (e.g. 2009, 2020 rebounds) and the backtest here has survivorship bias.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "dashboard_template.html"
UA = {"User-Agent": "Mozilla/5.0 (StockAnalyzer personal research script)"}

# ---------------------------------------------------------------------------
# Scoring configuration - tweak these to change the strategy
# ---------------------------------------------------------------------------
WEIGHTS = {
    "ret_12_1": 0.30,      # 12-month return, skipping the latest month (classic momentum)
    "ret_6_1": 0.20,       # 6-month return, skipping the latest month
    "risk_adj_mom": 0.20,  # 12-1 return divided by 1-yr volatility (smoother trends score higher)
    "rel_strength_6m": 0.10,  # 6-month return vs SPY
    "pct_off_high": 0.10,  # closeness to 52-week high (0 = at the high)
    "ret_3m": 0.10,        # recent 3-month trend confirmation
}
MIN_HISTORY = 260          # trading days needed to score a stock
MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 20e6   # 20-day average $ volume
TRADING_COST_BPS = 10      # per side, used in the backtest


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
def _wiki_tables(url: str) -> list[pd.DataFrame]:
    import requests

    html = requests.get(url, headers=UA, timeout=30).text
    return pd.read_html(StringIO(html))


ICB_TO_GICS = {"Technology": "Information Technology", "Telecommunications": "Communication Services",
               "Basic Materials": "Materials"}


def _nasdaq100() -> pd.DataFrame:
    """Nasdaq-100 members. Wikipedia has used both GICS and ICB sector columns,
    so find the components table by shape rather than exact headers."""
    for t in _wiki_tables("https://en.wikipedia.org/wiki/Nasdaq-100"):
        cols = {str(c).strip(): c for c in t.columns}
        tick = next((cols[c] for c in cols if c.lower() in ("ticker", "symbol")), None)
        if tick is None or len(t) < 90:
            continue
        name = next((cols[c] for c in cols if c.lower() in ("company", "security", "name")), tick)
        sec = next((cols[c] for c in cols if "sector" in c.lower() or "industry" in c.lower()), None)
        out = pd.DataFrame({"ticker": t[tick], "name": t[name],
                            "sector": t[sec].replace(ICB_TO_GICS) if sec is not None else "Unknown"})
        out["index"] = "Nasdaq-100"
        return out
    raise ValueError("components table not found")


def get_universe(cache: Path) -> pd.DataFrame:
    """S&P 500 + Nasdaq-100 constituents with name and GICS sector.
    Falls back to the last cached copy if Wikipedia can't be reached."""
    try:
        sp = _wiki_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        sp = sp.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"})
        sp = sp[["ticker", "name", "sector"]]

        sp["index"] = "S&P 500"
        parts = [sp]
        try:
            parts.append(_nasdaq100())
        except Exception as exc:
            print(f"  ! Skipping Nasdaq-100 list ({exc}); using S&P 500 only")
        uni = pd.concat(parts)
        uni["ticker"] = uni["ticker"].astype(str).str.strip().str.replace(".", "-", regex=False)
        uni = (uni.groupby("ticker", as_index=False)
                  .agg(name=("name", "first"), sector=("sector", "first"),
                       index=("index", lambda s: " + ".join(sorted(set(s))))))
        cache.parent.mkdir(parents=True, exist_ok=True)
        uni.to_csv(cache, index=False)
        return uni
    except Exception as exc:  # network / parsing issue
        if cache.exists():
            print(f"  ! Could not refresh universe ({exc}); using cached {cache.name}")
            return pd.read_csv(cache)
        raise


def universe_from_file(path: Path) -> pd.DataFrame:
    tickers = [t.strip().upper().replace(".", "-") for t in path.read_text().splitlines()
               if t.strip() and not t.startswith("#")]
    return pd.DataFrame({"ticker": tickers, "name": tickers, "sector": "Unknown", "index": "Watchlist"})


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------
def download_prices(tickers: list[str], years: int, include_open: bool = False):
    """Returns (adjusted close, volume) as date x ticker frames,
    plus adjusted open when include_open=True."""
    import yfinance as yf

    closes, vols, opens = [], [], []
    chunk = 100
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        print(f"  downloading {i + 1}-{i + len(batch)} of {len(tickers)}")
        data = yf.download(batch, period=f"{years}y", auto_adjust=True,
                           progress=False, threads=True, group_by="column")
        c, v, o = data["Close"], data["Volume"], data["Open"]
        if isinstance(c, pd.Series):  # single ticker
            c, v, o = c.to_frame(batch[0]), v.to_frame(batch[0]), o.to_frame(batch[0])
        closes.append(c)
        vols.append(v)
        opens.append(o)
    close = pd.concat(closes, axis=1).sort_index()
    volume = pd.concat(vols, axis=1).sort_index()
    close = close.loc[:, ~close.columns.duplicated()].dropna(how="all", axis=1)
    volume = volume.loc[:, ~volume.columns.duplicated()].reindex(columns=close.columns)
    if include_open:
        opn = pd.concat(opens, axis=1).sort_index()
        opn = opn.loc[:, ~opn.columns.duplicated()].reindex(columns=close.columns)
        return close, volume, opn
    return close, volume


# ---------------------------------------------------------------------------
# Factors & scoring
# ---------------------------------------------------------------------------
def _zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    z = (s - s.mean()) / s.std(ddof=0)
    return z.clip(-3, 3).fillna(0)


def compute_factors(close: pd.DataFrame, volume: pd.DataFrame, spy: pd.Series) -> pd.DataFrame:
    """Factor values as of the last row of `close`."""
    c = close.ffill(limit=5)
    n_obs = close.notna().sum()
    p = c.iloc[-1]

    def back(n):  # price n trading days ago
        return c.iloc[-1 - n] if len(c) > n else pd.Series(np.nan, index=c.columns)

    rets = c.pct_change(fill_method=None)
    vol_1y = rets.iloc[-252:].std() * np.sqrt(252)
    spy = spy.ffill()
    spy_6m = spy.iloc[-1] / spy.iloc[-127] - 1 if len(spy) > 127 else np.nan

    f = pd.DataFrame({
        "price": p,
        "ret_1m": p / back(21) - 1,
        "ret_3m": p / back(63) - 1,
        "ret_6_1": back(21) / back(126) - 1,
        "ret_12_1": back(21) / back(252) - 1,
        "ret_12m": p / back(252) - 1,
        "vol_1y": vol_1y,
        "pct_off_high": p / c.iloc[-252:].max() - 1,
        "ma50": c.iloc[-50:].mean(),
        "ma200": c.iloc[-200:].mean(),
        "dollar_vol_20d": (c.iloc[-20:] * volume.reindex(c.index).iloc[-20:]).mean(),
        "history_days": n_obs,
    })
    f["risk_adj_mom"] = f["ret_12_1"] / f["vol_1y"]
    f["rel_strength_6m"] = (p / back(126)) / (1 + spy_6m) - 1
    f["above_200dma"] = f["price"] > f["ma200"]
    f["ma50_over_200"] = f["ma50"] > f["ma200"]
    return f


def score(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    f["passes_filters"] = (
        (f["history_days"] >= MIN_HISTORY)
        & (f["price"] >= MIN_PRICE)
        & (f["dollar_vol_20d"] >= MIN_DOLLAR_VOLUME)
        & f[list(WEIGHTS)].notna().all(axis=1)
    )
    ok = f[f["passes_filters"]]
    composite = sum(_zscore(ok[k]) * w for k, w in WEIGHTS.items())
    f["composite"] = composite
    f["score"] = (composite.rank(pct=True) * 100).round(1)
    f["eligible"] = f["passes_filters"] & f["above_200dma"]  # trend filter
    return f


def pick(f: pd.DataFrame, top: int, sector_cap: int, sectors: pd.Series | None) -> pd.Index:
    """Top-N eligible names by score, at most `sector_cap` per sector."""
    cand = f[f["eligible"]].sort_values("composite", ascending=False)
    chosen, per_sector = [], {}
    for t in cand.index:
        sec = sectors.get(t, "Unknown") if sectors is not None else "Unknown"
        if sector_cap and per_sector.get(sec, 0) >= sector_cap:
            continue
        chosen.append(t)
        per_sector[sec] = per_sector.get(sec, 0) + 1
        if len(chosen) == top:
            break
    return pd.Index(chosen)


# ---------------------------------------------------------------------------
# Backtest: monthly rebalance into the top-N, equal weight
# ---------------------------------------------------------------------------
def backtest(close, volume, spy, sectors, top, sector_cap) -> dict:
    c = close.ffill(limit=5)
    month_ends = c.groupby(c.index.to_period("M")).tail(1).index
    month_ends = [d for d in month_ends if c.index.get_loc(d) >= MIN_HISTORY]
    if len(month_ends) < 13:
        return {}

    rets = c.pct_change(fill_method=None).fillna(0)
    spy_ret = spy.ffill().pct_change().fillna(0)
    daily, holdings_log = [], []
    prev = pd.Index([])
    cost = TRADING_COST_BPS / 10_000

    for i, d in enumerate(month_ends[:-1]):
        loc = c.index.get_loc(d)
        f = score(compute_factors(close.iloc[: loc + 1], volume.iloc[: loc + 1], spy.iloc[: loc + 1]))
        hold = pick(f, top, sector_cap, sectors)
        turnover = 1 - len(hold.intersection(prev)) / max(len(hold), 1) if len(prev) else 1.0
        prev = hold
        nxt = month_ends[i + 1]
        window = rets.loc[(rets.index > d) & (rets.index <= nxt), hold]
        port = window.mean(axis=1) if len(hold) else window.sum(axis=1) * 0
        if len(port):
            port.iloc[0] -= turnover * cost * 2
        daily.append(port)
        holdings_log.append({"date": str(d.date()), "turnover": round(turnover, 2)})

    strat = pd.concat(daily)
    bench = spy_ret.reindex(strat.index).fillna(0)
    eq = pd.DataFrame({"strategy": (1 + strat).cumprod(), "spy": (1 + bench).cumprod()})

    def stats(r: pd.Series, curve: pd.Series) -> dict:
        yrs = len(r) / 252
        cagr = curve.iloc[-1] ** (1 / yrs) - 1
        vol = r.std() * np.sqrt(252)
        dd = (curve / curve.cummax() - 1).min()
        return {"cagr": round(cagr, 4), "vol": round(vol, 4),
                "sharpe": round((r.mean() * 252) / vol, 2) if vol else None,
                "max_drawdown": round(dd, 4), "total_return": round(curve.iloc[-1] - 1, 4)}

    weekly = eq.resample("W-FRI").last().dropna()
    monthly_s = (1 + strat).resample("ME").prod() - 1
    monthly_b = (1 + bench).resample("ME").prod() - 1
    return {
        "start": str(eq.index[0].date()), "end": str(eq.index[-1].date()),
        "strategy": stats(strat, eq["strategy"]), "spy": stats(bench, eq["spy"]),
        "pct_months_beat_spy": round(float((monthly_s > monthly_b).mean()), 3),
        "avg_turnover": round(float(np.mean([h["turnover"] for h in holdings_log])), 2),
        "curve": [{"d": str(i.date()), "s": round(r.strategy, 4), "b": round(r.spy, 4)}
                  for i, r in weekly.iterrows()],
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def build_dashboard(table: pd.DataFrame, bt: dict, meta: dict, out: Path) -> None:
    cols = ["rank", "ticker", "name", "sector", "index", "pick", "eligible", "score", "price",
            "ret_1m", "ret_3m", "ret_6_1", "ret_12_1", "ret_12m", "rel_strength_6m",
            "vol_1y", "pct_off_high", "above_200dma", "ma50_over_200", "dollar_vol_20d"]
    rows = table[cols].replace({np.nan: None}).to_dict(orient="records")
    payload = {"meta": meta, "rows": rows, "backtest": bt}
    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", json.dumps(payload, default=float))
    out.write_text(html, encoding="utf-8")


def run(close, volume, uni, args, out_dir: Path) -> pd.DataFrame:
    uni = uni.drop_duplicates("ticker").set_index("ticker")
    spy = close["SPY"]
    stocks = [t for t in close.columns if t in uni.index]
    close_s, vol_s = close[stocks], volume[stocks]

    print("Scoring...")
    f = score(compute_factors(close_s, vol_s, spy))
    picks = pick(f, args.top, args.sector_cap, uni["sector"])

    table = f.join(uni, how="left")
    table["pick"] = table.index.isin(picks)
    table = table.sort_values(["passes_filters", "composite"], ascending=[False, False])
    table["rank"] = pd.Series(np.arange(1, len(table) + 1), index=table.index).where(table["passes_filters"]).astype("Int64")
    table.index.name = "ticker"
    table = table.reset_index()

    bt = {}
    if not args.no_backtest:
        print("Backtesting (monthly rebalance)...")
        bt = backtest(close_s, vol_s, spy, uni["sector"], args.top, args.sector_cap)

    as_of = str(close.index[-1].date())
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"picks_{as_of}.csv"
    table.to_csv(csv_path, index=False)

    meta = {"as_of": as_of, "generated": str(date.today()), "universe": int(len(stocks)),
            "scored": int(table["passes_filters"].sum()), "eligible": int(table["eligible"].sum()),
            "top": args.top, "sector_cap": args.sector_cap, "weights": WEIGHTS,
            "min_dollar_volume": MIN_DOLLAR_VOLUME, "cost_bps": TRADING_COST_BPS}
    build_dashboard(table, bt, meta, out_dir / "dashboard.html")

    print(f"\nTop {args.top} momentum picks as of {as_of}:")
    show = table[table["pick"]][["rank", "ticker", "name", "sector", "score", "ret_12_1", "ret_6_1", "vol_1y"]]
    with pd.option_context("display.width", 140, "display.max_colwidth", 28):
        print(show.to_string(index=False, formatters={
            "ret_12_1": "{:+.0%}".format, "ret_6_1": "{:+.0%}".format, "vol_1y": "{:.0%}".format}))
    if bt:
        s, b = bt["strategy"], bt["spy"]
        print(f"\nBacktest {bt['start']} -> {bt['end']}: CAGR {s['cagr']:.1%} vs SPY {b['cagr']:.1%}, "
              f"max drawdown {s['max_drawdown']:.1%} vs {b['max_drawdown']:.1%}")
    print(f"\nWrote {csv_path}\nWrote {out_dir / 'dashboard.html'}")
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description="Momentum stock picker")
    ap.add_argument("--top", type=int, default=20, help="number of picks (default 20)")
    ap.add_argument("--sector-cap", type=int, default=5, help="max picks per sector, 0 = no cap")
    ap.add_argument("--years", type=int, default=6, help="years of price history to download")
    ap.add_argument("--tickers", type=Path, help="text file of tickers instead of S&P 500 + Nasdaq-100")
    ap.add_argument("--out", type=Path, default=HERE / "output")
    ap.add_argument("--no-backtest", action="store_true")
    args = ap.parse_args(argv)

    print("Loading universe...")
    uni = universe_from_file(args.tickers) if args.tickers else get_universe(args.out / "universe.csv")
    print(f"  {len(uni)} tickers")
    print("Downloading prices (yfinance)...")
    close, volume = download_prices(sorted(set(uni["ticker"]) | {"SPY"}), args.years)
    if "SPY" not in close:
        sys.exit("Could not download SPY - check your connection.")
    run(close, volume, uni, args, args.out)


if __name__ == "__main__":
    main()
