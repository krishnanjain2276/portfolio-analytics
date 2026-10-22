"""Command line interface.

    portfolio-analytics demo                                   synthetic data, full report
    portfolio-analytics init         --data my_data            Excel input template
    portfolio-analytics add          --data my_data            add trades, step by step
    portfolio-analytics add          --data my_data --symbol RELIANCE --side BUY --qty 10 --price 2900
    portfolio-analytics add-cash     --data my_data --amount 500000 --date 2026-01-01
    portfolio-analytics import       --data my_data --file tradebook.csv
    portfolio-analytics fetch-prices --data my_data
    portfolio-analytics validate     --data my_data
    portfolio-analytics run          --data my_data --trials 3
    portfolio-analytics verdict      --data my_data            verdicts in the terminal only

``python -m portfolio_analytics ...`` works the same way.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import textwrap
import warnings
from pathlib import Path

from . import config

WIDTH = 100
TONE_MARK = {"good": "+", "warn": "!", "bad": "x", "neutral": "."}


def print_verdicts(results: dict) -> None:
    overall = results["overall"]
    rule = "=" * WIDTH
    print(f"\n{rule}\nOVERALL VERDICT: {overall['headline']}   "
          f"(as of {results['as_of']}, NAV Rs {results['nav_last']:,.0f})\n{rule}")
    for action in overall["actions"]:
        print(textwrap.fill(action, WIDTH, initial_indent="  > ", subsequent_indent="    "))

    print(f"\n  {'ANALYSIS':<22}VERDICT")
    for v in results["section_verdicts"]:
        print(f"{TONE_MARK[v.tone]} {v.area:<22}{v.verdict}")
        print(textwrap.fill(v.action, WIDTH, initial_indent=" " * 24, subsequent_indent=" " * 24))

    idea = results["index_view"]
    print(f"\nNIFTY: {idea['view']} (score {idea['score']:+d}), implied vol {idea['iv']:.1%} vs realised "
          f"{idea['rv20']:.1%} [{idea['vol_regime']}]  ->  {idea['strategy']}")
    for leg in idea.get("legs", []):
        print(f"    {leg['action']:<4} {idea.get('lots', 0)} lot(s) {leg['symbol']:<22} "
              f"est. {leg['premium_est']:>8.1f} pts")
    if idea.get("legs"):
        breakevens = ", ".join(f"{b:,.0f}" for b in idea["breakevens"])
        print(f"    max loss Rs {idea['max_loss_per_lot']:,.0f} a lot, breakeven {breakevens}")

    if len(results["position_verdicts"]):
        print("\nPOSITIONS")
        for p in results["position_verdicts"].itertuples():
            print(f"  {p.verdict:<24}{p.position[:48]:<50}{p.action[:120]}")
    if len(results["signals"]):
        print("\nSTOCKS")
        for s in results["signals"].itertuples():
            held = f"held {s.held_units:,.0f}" if s.held_units else "watch"
            print(f"  {s.action:<30}{s.symbol:<12}{held:<12}score {s.score:+4d}  "
                  f"last {s.last:,.1f}  stop {s.stop:,.1f}")
    print(f"\n{textwrap.fill(config.DISCLAIMER, WIDTH)}\n")


def analyse(data_dir, out_dir, rf, trials, synthetic=False, write_reports=True, mc_paths=config.MC_PATHS):
    from .analysis import run_all
    from .data import load_folder
    from .excel import to_excel
    from .report import to_html

    data = load_folder(data_dir, rf=rf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_all(data, rf=rf, n_trials=trials, mc_paths=mc_paths)
    print_verdicts(results)
    if write_reports:
        out = Path(out_dir)
        title = "Portfolio Tearsheet" + (" (synthetic demo)" if synthetic else "")
        html_path = to_html(results, out / "tearsheet.html", title, synthetic=synthetic)
        excel_path = to_excel(results, out / "portfolio_analytics.xlsx")
        print(f"Report:   {html_path}\nWorkbook: {excel_path}")
    return results



def cmd_demo(args):
    from .demo import generate_demo

    generate_demo(args.data)
    print(f"Synthetic data written to {args.data}/")
    analyse(args.data, args.out, config.RISK_FREE, trials=3, synthetic=True)


def cmd_run(args):
    analyse(args.data, args.out, args.rf, args.trials)


def cmd_verdict(args):
    analyse(args.data, args.out, args.rf, args.trials, write_reports=False)


def cmd_init(args):
    from .inputs import init_folder

    path = init_folder(args.data, args.force)
    print(f"Created {path}. Fill in Cashflows and Trades (the 'Start here' sheet explains how), then:\n"
          f"  python -m portfolio_analytics fetch-prices --data {args.data}\n"
          f"  python -m portfolio_analytics run --data {args.data}")


def cmd_add(args):
    from .inputs import add_trade, wizard

    if not (args.symbol and args.side and args.qty and args.price):
        wizard(args.data)
        return
    where, estimate = add_trade(args.data, args.date, args.symbol, args.side, args.qty, args.price,
                                args.fees, args.notes)
    extra = "" if args.fees is not None else f" Estimated charges Rs {estimate:,.2f}."
    print(f"Saved to {where}.{extra}")


def cmd_add_cash(args):
    from .inputs import add_cash

    print(f"Saved to {add_cash(args.data, args.date, args.amount, args.notes)}.")


def cmd_import(args):
    from .tradebook import import_tradebook

    column_map = dict(item.split("=", 1) for item in args.map)
    result = import_tradebook(args.data, args.file, column_map, fo_quantity_in_units=not args.fo_in_lots)
    print(f"Added {result['added']} trades to {result['written_to']}; "
          f"skipped {result['duplicates_skipped']} already there.")
    for warning in result["warnings"]:
        print("WARN", warning)


def cmd_fetch_prices(args):
    from .inputs import fetch_prices

    print(f"Prices written to {fetch_prices(args.data, args.start, args.end)}")


def cmd_validate(args):
    from .inputs import validate

    for level, message in validate(args.data):
        print(f"[{level}] {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="portfolio-analytics",
                                     description="Portfolio analytics with verdicts for Indian equities and NIFTY F&O")
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="generate synthetic data and run the full report")
    demo.add_argument("--data", default="data/demo")
    demo.add_argument("--out", default="reports")
    demo.set_defaults(handler=cmd_demo)

    for name, handler, help_text in (("run", cmd_run, "full report: HTML tearsheet and Excel workbook"),
                                     ("verdict", cmd_verdict, "verdicts in the terminal only")):
        p = commands.add_parser(name, help=help_text)
        p.add_argument("--data", required=True)
        p.add_argument("--out", default="reports")
        p.add_argument("--rf", type=float, default=config.RISK_FREE, help="risk-free rate, e.g. 0.065")
        p.add_argument("--trials", type=int, default=1,
                       help="how many strategy variants you tried (feeds the Deflated Sharpe)")
        p.set_defaults(handler=handler)

    init = commands.add_parser("init", help="create a data folder with the Excel input template")
    init.add_argument("--data", required=True)
    init.add_argument("--force", action="store_true", help="overwrite an existing input.xlsx")
    init.set_defaults(handler=cmd_init)

    add = commands.add_parser("add", help="add a trade (asks step by step if details are missing)")
    add.add_argument("--data", required=True)
    add.add_argument("--date", default=dt.date.today().isoformat())
    add.add_argument("--symbol")
    add.add_argument("--side", type=str.upper, choices=["BUY", "SELL"])
    add.add_argument("--qty", type=float, help="shares, or lots for F&O")
    add.add_argument("--price", type=float)
    add.add_argument("--fees", type=float, help="from the contract note; leave out to estimate")
    add.add_argument("--notes", default="")
    add.set_defaults(handler=cmd_add)

    cash = commands.add_parser("add-cash", help="record a deposit (+) or withdrawal (-)")
    cash.add_argument("--data", required=True)
    cash.add_argument("--amount", type=float, required=True)
    cash.add_argument("--date", default=dt.date.today().isoformat())
    cash.add_argument("--notes", default="")
    cash.set_defaults(handler=cmd_add_cash)

    imp = commands.add_parser("import", help="import a broker tradebook (Zerodha Console and most CSV/XLSX exports)")
    imp.add_argument("--data", required=True)
    imp.add_argument("--file", required=True)
    imp.add_argument("--map", nargs="*", default=[], help="column names, e.g. date='Trade Date' symbol=Scrip")
    imp.add_argument("--fo-in-lots", action="store_true", help="the file's F&O quantities are already in lots")
    imp.set_defaults(handler=cmd_import)

    fetch = commands.add_parser("fetch-prices", help="download stock and index closes (needs yfinance)")
    fetch.add_argument("--data", required=True)
    fetch.add_argument("--start")
    fetch.add_argument("--end")
    fetch.set_defaults(handler=cmd_fetch_prices)

    check = commands.add_parser("validate", help="check your data and explain any problems")
    check.add_argument("--data", required=True)
    check.set_defaults(handler=cmd_validate)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0
