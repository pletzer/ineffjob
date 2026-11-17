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


def date_chunks(start, end, days):
    """Split [start, end] into windows of `days` days. Falls back to a single
    window when the dates are not ISO (e.g. 'now-7days')."""
    if end is None:
        end = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    try:
        s, e = datetime.fromisoformat(start), datetime.fromisoformat(end)
    except ValueError:
        return [(start, end)]
    fmt = "%Y-%m-%dT%H:%M:%S"
    out = []
    while s < e:
        nxt = min(s + timedelta(days=days), e)
        out.append((s.strftime(fmt), nxt.strftime(fmt)))
        s = nxt
    return out or [(start, end)]


def windows(base_cmd, start, end, days, env=None):
    """Yield (sacct output, lo, hi) for each window. lo/hi are the window
    bounds as datetimes, or None when dates are not ISO (single query, where
    contained() is always True). Runs sacct with SLURM_TIME_FORMAT=standard so
    Start/End columns parse."""
    env = env or {**os.environ, "SLURM_TIME_FORMAT": "standard"}
    chunks = date_chunks(start, end, days)
    for n, (cs, ce) in enumerate(chunks, 1):
        if len(chunks) > 1:
            print(f"querying {cs[:10]} .. {ce[:10]} ({n}/{len(chunks)})",
                  file=sys.stderr)
        text = run(base_cmd + ["-S", cs] + (["-E", ce] if ce else []), env=env)
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
        out = run(["sacct", "-n", "-P", "-o", ",".join(fields), "-j",
                   ",".join(ids[i:i + batch])], env=env)
        yield from out.splitlines()
