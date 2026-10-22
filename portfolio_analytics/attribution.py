"""Where the money was made and lost, and what trading cost."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .instruments import Instrument


def pnl_by_sleeve(pnl: pd.DataFrame, instruments: dict[str, Instrument]) -> pd.DataFrame:
    final = pd.DataFrame({"pnl": pnl.iloc[-1]})
    final["sleeve"] = [instruments[s].sleeve for s in final.index]
    final["asset_class"] = [instruments[s].asset_class for s in final.index]
    by_sleeve = final.groupby(["asset_class", "sleeve"])["pnl"].sum().reset_index()
    total = by_sleeve["pnl"].sum()
    by_sleeve["share"] = by_sleeve["pnl"] / total if total else np.nan
    return by_sleeve.sort_values("pnl", ascending=False).reset_index(drop=True)


def pnl_by_instrument(pnl: pd.DataFrame, instruments, top: int = 10) -> pd.DataFrame:
    final = pd.DataFrame({"pnl": pnl.iloc[-1].sort_values()})
    final["asset_class"] = [instruments[s].asset_class for s in final.index]
    best, worst = final.tail(top).iloc[::-1], final.head(top)
    return pd.concat([best, worst]).drop_duplicates()


def monthly_pnl_by_class(pnl: pd.DataFrame, instruments) -> pd.DataFrame:
    daily = pnl.diff()
    daily.iloc[0] = pnl.iloc[0]
    asset_class = pd.Series({s: instruments[s].asset_class for s in pnl.columns})
    monthly = daily.T.groupby(asset_class).sum().T.resample("ME").sum()
    monthly.index = monthly.index.strftime("%Y-%m")
    monthly["Total"] = monthly.sum(axis=1)
    return monthly


def trading_stats(ledger, nav: pd.Series) -> dict:
    trades = ledger.trades
    years = max(len(nav) / 252, 1e-9)
    if trades is None or trades.empty:
        return {"trades": 0, "trades_per_month": 0.0, "turnover_annual": 0.0, "fees_total": 0.0,
                "fees_pct_nav_annual": 0.0, "gross_pnl": 0.0, "net_pnl": 0.0,
                "cost_share_of_gross": np.nan}

    average_nav = nav.mean()
    # F&O turnover is counted on the underlying's value, not the premium
    is_equity = trades["asset_class"] == "equity"
    turnover = trades["underlying_notional"].where(~is_equity, trades["notional"]).sum()
    fees = float(trades["fees"].sum())
    net = float(ledger.pnl.iloc[-1].sum())
    gross = net + fees
    return {
        "trades": int(len(trades)),
        "trades_per_month": len(trades) / (years * 12),
        "turnover_annual": turnover / average_nav / years,
        "fees_total": fees,
        "fees_pct_nav_annual": fees / average_nav / years,
        "gross_pnl": gross,
        "net_pnl": net,
        "cost_share_of_gross": fees / gross if gross > 0 else np.nan,
    }
