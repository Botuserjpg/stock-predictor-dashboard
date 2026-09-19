#!/usr/bin/env python3
"""Precompute cached analysis for the public /demo route.

Run this OFFLINE (on a machine with more memory/time than the free-tier Render
worker) because it trains real models:

    python scripts/precompute_demo_cache.py

One JSON file is written per ticker into the demo cache directory
(``data_cache/demo_precomputed`` or ``DEMO_CACHE_DIR``), overwriting any
previous run, so re-running weekly just refreshes the results. The web app's
/demo route only READS these files — it never trains.

Examples:
    python scripts/precompute_demo_cache.py
    python scripts/precompute_demo_cache.py --tickers AAPL MSFT
    python scripts/precompute_demo_cache.py --days 30 --period 1y --model-type AUTO
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("precompute_demo_cache")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="*", default=None,
                        help="Tickers to precompute (defaults to the full demo set).")
    parser.add_argument("--days", type=int, default=30, help="Forecast horizon in days (default 30).")
    parser.add_argument("--period", default="1y", help="History period fed to the model (default 1y).")
    parser.add_argument("--risk", default="medium", help="Risk profile for the strategy (default medium).")
    parser.add_argument("--model-type", default="AUTO",
                        help="Model: AUTO, LSTM, GRU or ENSEMBLE (default AUTO).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Symbols processed in parallel (default 1).")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    from stockpredictor.services.demo_cache import (
        DEMO_CACHE_DIR,
        DEMO_TICKERS,
        build_demo_result,
        demo_write,
        list_demo_tickers,
    )

    tickers = [ticker.strip().upper() for ticker in (args.tickers or list_demo_tickers())]
    bad = [t for t in tickers if t not in DEMO_TICKERS]
    if bad:
        parser_error = f"Tickers not in the demo set (won't be served by /demo): {', '.join(bad)}"
        logger.error(parser_error)
        return 2
    if not tickers:
        logger.info("No tickers selected.")
        return 0

    DEMO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Writing precomputed results to %s", DEMO_CACHE_DIR)
    logger.info("Using %d-day horizon, period=%s, risk=%s, model=%s",
                args.days, args.period, args.risk, args.model_type)

    ok_count = 0

    def _one(symbol: str) -> tuple[str, bool, str]:
        try:
            result = build_demo_result(
                symbol, period=args.period, days=args.days,
                risk=args.risk, model_type=args.model_type,
            )
        except Exception as exc:  # noqa: BLE001 - report every ticker
            logger.error("Unexpected failure for %s: %s", symbol, exc)
            return symbol, False, str(exc)
        if isinstance(result, dict) and result.get("error"):
            message = result.get("error") or "unknown error"
            logger.error("Pipeline error for %s: %s", symbol, message)
            if result.get("forecast_rows") or result.get("report"):
                logger.info("Caching degraded result for %s (has forecast data).", symbol)
                try:
                    demo_write(symbol, result)
                    return symbol, True, ""
                except Exception as exc:  # noqa: BLE001
                    return symbol, False, str(exc)
            return symbol, False, str(message)
        try:
            demo_write(symbol, result)
        except Exception as exc:  # noqa: BLE001 - report every ticker
            logger.error("Failed to write cache for %s: %s", symbol, exc)
            return symbol, False, str(exc)
        logger.info("Cached %s under %s", symbol, DEMO_CACHE_DIR)
        return symbol, True, ""

    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_one, tickers))
    else:
        results = [_one(symbol) for symbol in tickers]

    ok_count = sum(1 for _, ok, _ in results if ok)
    failed_symbols = [symbol for symbol, ok, _ in results if not ok]
    if failed_symbols:
        logger.error("Failed %d/%d: %s", len(failed_symbols), len(tickers), ", ".join(failed_symbols))
        return 1
    logger.info("Precomputed demo cache: %d/%d tickers saved under %s",
                ok_count, len(tickers), DEMO_CACHE_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())