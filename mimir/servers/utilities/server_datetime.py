"""
MCP DateTime Server
===================
Date and time utilities.  All tools return structured dicts.

Workflow examples:
  - "What time is it in Tokyo?" → current_datetime(tz="Asia/Tokyo")
  - "How many days until 2027-01-01?" → days_between("2026-03-23", "2027-01-01")
  - "What day of the week is my birthday?" → day_of_week("1990-07-14")
  - "Add 30 days to today" → add_days("2026-03-23", 30)
  - "Reformat '14/07/1990' to 'July 14, 1990'" → format_date("14/07/1990", "%d/%m/%Y", "%B %d, %Y")
"""

from datetime import date, datetime, timedelta

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    from backports.zoneinfo import ZoneInfo, ZoneInfoNotFoundError  # type: ignore

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import FastMCP
from responses import err, ok
from capabilities import tool_caps

mcp = FastMCP(
    "DateTimeServer",
    debug=False,
    log_level="ERROR",
)

_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ── tools ─────────────────────────────────────────────────────────────────────

_DATE_OPS = (
    "current_datetime", "days_between", "add_days", "day_of_week",
    "unix_to_date", "format_date",
)


@mcp.tool(**tool_caps(label="Date {op}"))
def date_op(
    op: str,
    tz: str = "UTC",
    date1: str = "",
    date2: str = "",
    date_str: str = "",
    n: int = 0,
    timestamp: int = 0,
    input_fmt: str = "",
    output_fmt: str = "",
) -> dict:
    """Apply a single date/time operation, selected by ``op``.

    Operations (set ``op`` to one of these):
      current_datetime -> now in `tz`  -> {datetime, date, time, weekday, tz}
      days_between     -> days from `date1` to `date2` (ISO) -> {days, from, to, direction}
      add_days         -> `date_str` (ISO) + `n` days -> {result, weekday, input, n}
      day_of_week      -> weekday of `date_str` (ISO) -> {date, weekday, weekday_index, is_weekend}
      unix_to_date     -> `timestamp` (epoch secs) in `tz` -> {datetime, date, time, weekday, tz}
      format_date      -> reformat `date_str` from `input_fmt` to `output_fmt`

    Args:
        op:          The operation to perform (see list above).
        tz:          IANA timezone (current_datetime / unix_to_date). Default 'UTC'.
        date1, date2: ISO YYYY-MM-DD dates (days_between).
        date_str:    ISO date (add_days / day_of_week) or the value to reformat (format_date).
        n:           Days to add, negative to subtract (add_days).
        timestamp:   Unix timestamp in seconds (unix_to_date).
        input_fmt, output_fmt: strptime/strftime formats (format_date), e.g. '%d/%m/%Y' -> '%B %d, %Y'.
    """
    try:
        if op == "current_datetime":
            now = datetime.now(ZoneInfo(tz))
            return ok({
                "datetime": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "date":     now.strftime("%Y-%m-%d"),
                "time":     now.strftime("%H:%M:%S"),
                "weekday":  _WEEKDAYS[now.weekday()],
                "tz":       tz,
            })
        if op == "days_between":
            d1 = date.fromisoformat(date1)
            d2 = date.fromisoformat(date2)
            delta = (d2 - d1).days
            direction = "future" if delta > 0 else ("past" if delta < 0 else "same")
            return ok({"days": abs(delta), "from": date1, "to": date2, "direction": direction})
        if op == "add_days":
            d = date.fromisoformat(date_str)
            result = d + timedelta(days=n)
            return ok({
                "result":  result.isoformat(),
                "weekday": _WEEKDAYS[result.weekday()],
                "input":   date_str,
                "n":       n,
            })
        if op == "day_of_week":
            d = date.fromisoformat(date_str)
            idx = d.weekday()
            return ok({
                "date":          date_str,
                "weekday":       _WEEKDAYS[idx],
                "weekday_index": idx,
                "is_weekend":    idx >= 5,
            })
        if op == "unix_to_date":
            dt = datetime.fromtimestamp(timestamp, tz=ZoneInfo(tz))
            return ok({
                "datetime": dt.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "date":     dt.strftime("%Y-%m-%d"),
                "time":     dt.strftime("%H:%M:%S"),
                "weekday":  _WEEKDAYS[dt.weekday()],
                "tz":       tz,
            })
        if op == "format_date":
            dt = datetime.strptime(date_str, input_fmt)
            return ok({
                "result":     dt.strftime(output_fmt),
                "input":      date_str,
                "output_fmt": output_fmt,
            })
        return err(
            f"Unknown date op '{op}'.",
            hint=f"Use one of: {', '.join(_DATE_OPS)}.",
        )
    except ZoneInfoNotFoundError:
        return err(f"Unknown timezone '{tz}'.",
                   hint="Use IANA names like 'Europe/Paris', 'America/New_York', 'Asia/Tokyo'.")
    except ValueError as e:
        return err(str(e), hint="Use ISO format YYYY-MM-DD, e.g. '2026-03-23'.")
    except Exception as e:
        return err(str(e))


if __name__ == "__main__":
    mcp.run()
