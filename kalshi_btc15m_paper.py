#!/usr/bin/env python3
"""Paper trade Kalshi's BTC Up or Down series using Python 3.10 or later.

Run python3 kalshi_btc15m_paper.py --once to sample the current window.
Run python3 kalshi_btc15m_paper.py to follow new windows until Ctrl+C.
Run python3 kalshi_btc15m_paper.py --summary to read local results.

The model uses Bitstamp spot prices as a proxy for the BRTI settlement index.
Paper fills assume one whole contract at the displayed ask. The baseline
assumes zero drift and uses up to 30 completed one-minute returns for
volatility. Entry cost estimates use standard 1x taker fees and cent balances.

Kalshi API docs at https://docs.kalshi.com/
Bitstamp public data docs at https://www.bitstamp.net/api/
"""

import argparse
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
from email.utils import parsedate_to_datetime
import json
import math
import sqlite3
from statistics import NormalDist, stdev
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


KALSHI = "https://external-api.kalshi.com/trade-api/v2"
BITSTAMP = "https://www.bitstamp.net/api/v2"
SERIES = "KXBTC15M"
DOLLAR = Decimal("1")
MIN_SECONDS_LEFT = 65
SCHEMA = """
CREATE TABLE IF NOT EXISTS windows (
    ticker TEXT PRIMARY KEY,
    next_result_check REAL NOT NULL,
    yes_payout TEXT
);
CREATE TABLE IF NOT EXISTS quotes (
    observed_at TEXT NOT NULL,
    ticker TEXT NOT NULL,
    seconds_left REAL NOT NULL,
    spot_usd REAL NOT NULL,
    target_usd REAL NOT NULL,
    minute_volatility REAL NOT NULL,
    probability_yes REAL NOT NULL,
    yes_ask TEXT NOT NULL,
    no_ask TEXT NOT NULL,
    yes_ask_size TEXT NOT NULL,
    no_ask_size TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_trades (
    ticker TEXT PRIMARY KEY,
    entered_at TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('yes', 'no')),
    ask TEXT NOT NULL,
    cost_with_fee TEXT NOT NULL,
    expected_value REAL NOT NULL
);
"""


class SkipSample(Exception):
    """The current inputs are unsuitable for a paper trade."""


def get_json(url, max_age_seconds=None):
    request = Request(
        url,
        headers={"Accept": "application/json", "Cache-Control": "no-cache",
                 "User-Agent": "kalshi-btc15m-paper/1.0"},
    )
    started = time.monotonic()
    with urlopen(request, timeout=6) as response:
        payload = json.load(response)
        if max_age_seconds is not None:
            server_date = response.headers.get("Date")
            ages = [time.monotonic() - started, int(response.headers.get("Age", "0"))]
            if server_date:
                ages.append(time.time() - parsedate_to_datetime(server_date).timestamp())
            if max(ages) > max_age_seconds:
                raise SkipSample("Order book response was stale or slow.")
    return payload


def epoch(iso_time):
    return datetime.fromisoformat(iso_time.replace("Z", "+00:00")).timestamp()


def current_market():
    data = get_json(f"{KALSHI}/markets?series_ticker={SERIES}&status=open&limit=100")
    now = time.time()
    active = [
        market for market in data["markets"]
        if market["status"] == "active"
        and epoch(market["open_time"]) <= now < epoch(market["close_time"])
    ]
    return min(active, key=lambda market: epoch(market["close_time"]), default=None)


def recent_minute_volatility():
    url = f"{BITSTAMP}/ohlc/btcusd/?step=60&limit=31&exclude_current_candle=true"
    rows = get_json(url)["data"]["ohlc"]
    now = time.time()
    candles = sorted(
        (int(row["timestamp"]), float(row["close"]))
        for row in rows if int(row["timestamp"]) + 60 <= now
    )
    if len(candles) < 21 or now - (candles[-1][0] + 60) > 150:
        raise SkipSample("Waiting for a recent set of completed BTC candles.")
    returns = [
        math.log(price / prior_price)
        for (prior_ts, prior_price), (ts, price) in zip(candles, candles[1:])
        if ts - prior_ts == 60 and 0 < prior_price < math.inf and 0 < price < math.inf
    ]
    if len(returns) != len(candles) - 1:
        raise SkipSample("BTC candles have a gap or invalid price.")
    sigma = stdev(returns)
    if not math.isfinite(sigma) or sigma <= 0:
        raise SkipSample("BTC volatility estimate is invalid.")
    return sigma


def best_asks(ticker):
    url = f"{KALSHI}/markets/{ticker}/orderbook?depth=1"
    book = get_json(url, max_age_seconds=6)["orderbook_fp"]
    yes_bids = book.get("yes_dollars")
    no_bids = book.get("no_dollars")
    if not yes_bids or not no_bids:
        raise SkipSample("Waiting for quotes on both sides.")
    yes_bid, yes_size = max(tuple(map(Decimal, row)) for row in yes_bids)
    no_bid, no_size = max(tuple(map(Decimal, row)) for row in no_bids)
    if min(yes_bid, no_bid) <= 0 or yes_bid + no_bid >= DOLLAR:
        raise SkipSample("Order book prices are invalid or crossed.")
    return {"yes": (DOLLAR - no_bid, no_size), "no": (DOLLAR - yes_bid, yes_size)}


def yes_probability(spot, target, seconds_left, minute_volatility):
    # Before the final minute starts, the variance of the final-minute mean
    # under a Brownian log-price model is sigma squared times (T - 2/3).
    average_volatility = minute_volatility * math.sqrt(seconds_left / 60 - 2 / 3)
    return NormalDist().cdf(math.log(spot / target) / average_volatility)


def one_contract_cost(ask):
    trade_fee = Decimal("0.07") * ask * (DOLLAR - ask)
    trade_fee = trade_fee.quantize(Decimal("0.000001"), rounding=ROUND_CEILING)
    return (ask + trade_fee).quantize(Decimal("0.01"), rounding=ROUND_CEILING)


def observe(db, market, sigma, min_edge):
    ticker = market["ticker"]
    close = epoch(market["close_time"])
    target = float(market.get("floor_strike") or 0)
    if not math.isfinite(target) or target <= 0:
        raise SkipSample(f"{ticker} is waiting for the opening benchmark.")
    if close - time.time() < MIN_SECONDS_LEFT:
        raise SkipSample(f"{ticker} is nearing the close. Waiting for the next window.")

    spot_data = get_json(f"{BITSTAMP}/ticker/btcusd/")
    spot = float(spot_data["last"])
    asks = best_asks(ticker)
    now = time.time()
    seconds_left = close - now
    if seconds_left < MIN_SECONDS_LEFT:
        raise SkipSample("Market reached the entry cutoff while fetching quotes.")
    if not 0 <= now - int(spot_data["timestamp"]) <= 20 or not 0 < spot < math.inf:
        raise SkipSample("BTC spot quote is stale or invalid.")

    p_yes = yes_probability(spot, target, seconds_left, sigma)
    quoted_at = datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="milliseconds")
    yes_ask, yes_size = asks["yes"]
    no_ask, no_size = asks["no"]

    candidates = []
    for side, probability in (("yes", p_yes), ("no", 1 - p_yes)):
        ask, size = asks[side]
        cost = one_contract_cost(ask)
        edge = probability - float(cost)
        if size >= 1 and edge >= min_edge:
            candidates.append((edge, side, ask, cost))

    inserted = 0
    with db:
        db.execute(
            "INSERT INTO windows VALUES (?, ?, NULL) ON CONFLICT(ticker) DO NOTHING",
            (ticker, close + 10),
        )
        db.execute(
            "INSERT INTO quotes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (quoted_at, ticker, seconds_left, spot, target, sigma, p_yes,
             str(yes_ask), str(no_ask), str(yes_size), str(no_size)),
        )
        if candidates:
            edge, side, ask, cost = max(candidates)
            inserted = db.execute(
                "INSERT INTO paper_trades VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(ticker) DO NOTHING",
                (ticker, quoted_at, side, str(ask), str(cost), edge),
            ).rowcount
    if inserted:
        print(f"Paper buy {side.upper()} at ${ask} with ${cost} all-in "
              f"and ${edge:.3f} modeled edge.", flush=True)
    print(f"{ticker} | {seconds_left:.0f}s left | BTC ${spot:,.2f} "
          f"| target ${target:,.2f} | P(YES) {p_yes:.1%} "
          f"| asks YES ${yes_ask} / NO ${no_ask}", flush=True)


def resolve_closed_windows(db):
    due = db.execute(
        "SELECT ticker FROM windows WHERE yes_payout IS NULL AND next_result_check <= ? "
        "ORDER BY next_result_check LIMIT 3",
        (time.time(),),
    ).fetchall()
    for (ticker,) in due:
        with db:
            db.execute("UPDATE windows SET next_result_check = ? WHERE ticker = ?",
                       (time.time() + 30, ticker))
        try:
            market = get_json(f"{KALSHI}/markets/{ticker}")["market"]
        except HTTPError as error:
            if error.code != 404:
                raise
            market = get_json(f"{KALSHI}/historical/markets/{ticker}")["market"]
        if market["status"] != "finalized":
            continue
        raw_payout = market.get("settlement_value_dollars")
        if raw_payout is None or raw_payout == "":
            raise ValueError(f"{ticker} has no published settlement payout yet.")
        payout = Decimal(str(raw_payout))
        if not 0 <= payout <= DOLLAR:
            raise ValueError(f"{ticker} returned an invalid settlement payout.")
        with db:
            db.execute("UPDATE windows SET yes_payout = ? WHERE ticker = ?",
                       (str(payout), ticker))
        print(f"{ticker} settled. YES payout ${payout}.", flush=True)


def summarize(db):
    rows = db.execute(
        "SELECT side, cost_with_fee, yes_payout FROM paper_trades JOIN windows USING (ticker)"
    ).fetchall()
    settled = 0
    pnl = Decimal("0")
    for side, cost, yes_payout in rows:
        if yes_payout is None:
            continue
        payout = Decimal(yes_payout)
        if side == "no":
            payout = DOLLAR - payout
        pnl += payout - Decimal(cost)
        settled += 1
    print(f"{settled} settled paper trades, {len(rows) - settled} pending. "
          f"Net paper P&L {pnl:+.2f} USD.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run one polling cycle.")
    mode.add_argument("--summary", action="store_true", help="Show saved paper P&L.")
    parser.add_argument("--db", default="kalshi_btc15m.sqlite3")
    parser.add_argument("--poll", type=float, default=8, help="Seconds between polls.")
    parser.add_argument("--min-edge", type=float, default=0.05,
                        help="Minimum modeled net gain in dollars per contract.")
    args = parser.parse_args()
    if not 3 <= args.poll <= 60 or not 0 <= args.min_edge <= 1:
        parser.error("Use a poll interval of 3 to 60 seconds and a minimum edge of 0 to 1.")

    sigma = None
    next_volatility_refresh = 0
    delay = args.poll
    with closing(sqlite3.connect(args.db, timeout=5)) as db:
        db.executescript(SCHEMA)
        if args.summary:
            summarize(db)
            return 0
        try:
            while True:
                failed = False
                try:
                    resolve_closed_windows(db)
                    market = current_market()
                    if market is None:
                        raise SkipSample("No active BTC 15-minute window yet.")
                    if time.monotonic() >= next_volatility_refresh:
                        sigma = recent_minute_volatility()
                        next_volatility_refresh = time.monotonic() + 60
                    observe(db, market, sigma, args.min_edge)
                except SkipSample as reason:
                    print(reason, flush=True)
                except (OSError, ValueError, KeyError, ArithmeticError) as error:
                    print(f"Could not sample market data. {error}", flush=True)
                    failed = True
                if args.once:
                    return int(failed)
                delay = min(delay * 2, 60) if failed else args.poll
                time.sleep(delay)
        except KeyboardInterrupt:
            summarize(db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
