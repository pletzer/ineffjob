"""Shared sacct helpers: run with retry, and query a date range in windows.

A single sacct over weeks of cluster-wide data can overload slurmdbd and fail
with "Connection refused". Querying in small windows and retrying keeps each
request light.

Caveat when merging windows: a job appears in every window it was pending or
running in, but sacct CLIPS TotalCPU/UserCPU/SystemCPU to the query window
(roughly proportionally, not exactly: summing windows overshoots by ~2%).
Elapsed, State and ReqTRES are not clipped. So CPU times are only exact in a
window that fully contains the job (see contained()); for jobs that cross a
window boundary, ask for them by ID (job_rows(): `sacct -j` has no window and
returns the exact totals, ~0.4 s per 300 jobs).
"""
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta


def run(cmd, retries=3, env=None):
    """Run a command; retry with backoff if slurmdbd refuses the connection."""
    for attempt in range(retries + 1):
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if not p.returncode:
            return p.stdout
        if "onnection" in p.stderr and attempt < retries:
            wait = 10 * 3 ** attempt  # 10s, 30s, 90s
            print(f"slurmdbd unavailable; retrying in {wait}s "
                  f"({attempt + 1}/{retries})...", file=sys.stderr)
            time.sleep(wait)
            continue
        sys.exit(f"{' '.join(cmd[:3])} ... failed: {p.stderr.strip()}")


_REL = re.compile(r"now(?:-(\d+)(minute|hour|day|week)s?)?$")
_UNIT = {"minute": 60, "hour": 3600, "day": 86400, "week": 604800}


def parse_when(text, now=None):
    """ISO date/time or sacct-style 'now[-Ndays|hours|weeks|minutes]' -> datetime.
    Raises ValueError for anything else."""
    now = now or datetime.now().replace(microsecond=0)
    m = _REL.match(text)
    if m:
        return now - timedelta(seconds=int(m.group(1) or 0) * _UNIT.get(m.group(2), 0))
    return datetime.fromisoformat(text)


def date_chunks(start, end, days):
    """Split [start, end] into windows of `days` days. Falls back to a single
    window when a date cannot be parsed."""
    try:
        now = datetime.now().replace(microsecond=0)
        s = parse_when(start, now)
        e = parse_when(end, now) if end else now
    except ValueError:
        return [(start, end)]
    fmt = "%Y-%m-%dT%H:%M:%S"
    out = []
    while s < e:
        nxt = min(s + timedelta(days=days), e)
        out.append((s.strftime(fmt), nxt.strftime(fmt)))
        s = nxt
    return out or [(start, end)]


def _hms(sec):
    sec = int(sec)
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 \
        else f"{sec // 60}:{sec % 60:02d}"


def windows(base_cmd, start, end, days, env=None):
    """Yield (sacct output, lo, hi) for each window. lo/hi are the window
    bounds as datetimes, or None when dates are not ISO (single query, where
    contained() is always True). Runs sacct with SLURM_TIME_FORMAT=standard so
    Start/End columns parse. Progress (rows, time, ETA) goes to stderr."""
    env = env or {**os.environ, "SLURM_TIME_FORMAT": "standard"}
    chunks = date_chunks(start, end, days)
    n_total, t0 = len(chunks), time.time()
    for n, (cs, ce) in enumerate(chunks, 1):
        label = f"{cs[:10]} .. {(ce or 'now')[:10]}"
        print(f"[{n}/{n_total}] querying {label} ...", end=" ", flush=True,
              file=sys.stderr)
        t1 = time.time()
        text = run(base_cmd + ["-S", cs] + (["-E", ce] if ce else []), env=env)
        done, took = time.time() - t0, time.time() - t1
        eta = f", ETA {_hms(done / n * (n_total - n))}" if n < n_total else ""
        print(f"{text.count(chr(10)):,} rows in {_hms(took)} "
              f"(elapsed {_hms(done)}{eta})", file=sys.stderr)
        try:
            lo, hi = datetime.fromisoformat(cs), datetime.fromisoformat(ce)
        except (ValueError, TypeError):
            lo = hi = None
        yield text, lo, hi


def contained(start, end, lo, hi):
    """True if a job (ISO Start/End strings) lies fully inside [lo, hi], so its
    CPU times were not clipped. Unknown End (still running) -> False."""
    if lo is None:
        return True
    try:
        return datetime.fromisoformat(start) >= lo and datetime.fromisoformat(end) <= hi
    except (ValueError, TypeError):
        return False


def job_rows(ids, fields, batch=300, env=None):
    """Exact (unclipped) rows by job ID: yields sacct -P lines for `ids`."""
    env = env or {**os.environ, "SLURM_TIME_FORMAT": "standard"}
    ids = list(ids)
    for i in range(0, len(ids), batch):
        if len(ids) > batch:
            print(f"  fetching exact data: {min(i + batch, len(ids)):,}/{len(ids):,} jobs",
                  file=sys.stderr)
        out = run(["sacct", "-n", "-P", "-o", ",".join(fields), "-j",
                   ",".join(ids[i:i + batch])], env=env)
        yield from out.splitlines()
