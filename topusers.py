#!/usr/bin/env python3
"""topusers: rank the biggest users of the cluster over a time window.

Per user it reports job count, requested CPU-hours, GPU-hours, memory-hours
and the CPU efficiency (CPU time used / CPU-hours requested, finished
jobs only).
Uses `sacct -a` and keeps job-level rows only: with -X, sacct reports
TotalCPU as 0, so the steps must be queried and then skipped.
"""
import argparse
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass

from ineffjob import parse_duration, parse_mem_mb, parse_reqtres

IN_PROGRESS = {"RUNNING", "PENDING", "SUSPENDED", "REQUEUED", "RESIZING",
               "COMPLETING", "CONFIGURING", "SIGNALING", "STAGE_OUT"}
FIELDS = ["JobID", "User", "Account", "Elapsed", "ReqTRES", "TotalCPU", "State", "NNodes"]


@dataclass
class Usage:
    jobs: int = 0
    cpu_h: float = 0.0      # requested CPUs x elapsed
    done_cpu_h: float = 0.0  # same, finished jobs only (TotalCPU is final)
    used_cpu_h: float = 0.0  # TotalCPU of finished jobs
    gpu_h: float = 0.0
    mem_gb_h: float = 0.0
    wall_h: float = 0.0
    accounts: set = None

    def __post_init__(self):
        self.accounts = set()

    @property
    def eff(self):
        return self.used_cpu_h / self.done_cpu_h if self.done_cpu_h else 0.0


def parse_gpus(tres):
    # 'gres/gpu=2' is the total; typed entries ('gres/gpu:a100=2') duplicate it
    m = re.fullmatch(r"\d+", tres.get("gres/gpu", ""))
    return int(m.group()) if m else 0


def aggregate(text, by="User", account=None):
    """Parse sacct output into ({user-or-account: Usage}, cluster CPU-hours).

    The cluster total covers every job in `text`, even when `account` limits
    the rows reported, so the %cpu column is always a share of the cluster.
    """
    out = defaultdict(Usage)
    cluster_cpu_h = 0.0
    for line in text.splitlines():
        cols = line.split("|")
        if len(cols) != len(FIELDS):
            continue
        r = dict(zip(FIELDS, cols))
        if "." in r["JobID"]:  # job step; the job row has the totals
            continue
        elapsed = parse_duration(r["Elapsed"]) or 0.0
        if not r[by] or elapsed <= 0:  # never started
            continue
        tres = parse_reqtres(r["ReqTRES"])
        ncpu = int(tres["cpu"]) if tres.get("cpu", "").isdigit() else 0
        # ReqMem in TRES is the total for the job (e.g. mem=32G)
        mem_mb = parse_mem_mb(tres.get("mem")) or 0.0
        h = elapsed / 3600
        cluster_cpu_h += ncpu * h
        if account and r["Account"] != account:
            continue
        u = out[r[by]]
        u.jobs += 1
        u.wall_h += h
        u.cpu_h += ncpu * h
        # RUNNING etc. jobs report TotalCPU=0 until they end; counting them
        # would make efficiency look far worse than it is.
        if r["State"].split()[0] not in IN_PROGRESS:
            u.done_cpu_h += ncpu * h
            u.used_cpu_h += (parse_duration(r["TotalCPU"]) or 0.0) / 3600
        u.gpu_h += parse_gpus(tres) * h
        u.mem_gb_h += mem_mb / 1024 * h
        u.accounts.add(r["Account"])
    return out, cluster_cpu_h


SORT_KEYS = {"cpu": "cpu_h", "gpu": "gpu_h", "mem": "mem_gb_h",
             "jobs": "jobs", "wall": "wall_h"}


def render(usage, key, top, label, total_cpu):
    total_cpu = total_cpu or 1.0
    rows = sorted(usage.items(), key=lambda kv: getattr(kv[1], SORT_KEYS[key]),
                  reverse=True)[:top]
    lines = [f"{label:<16}{'jobs':>8}{'cpu-h':>11}{'%cpu':>6}{'gpu-h':>10}"
             f"{'mem-GBh':>11}{'cpu-eff':>9}  accounts"]
    for name, u in rows:
        lines.append(f"{name:<16}{u.jobs:>8}{u.cpu_h:>11.0f}"
                     f"{u.cpu_h / total_cpu:>6.0%}{u.gpu_h:>10.0f}"
                     f"{u.mem_gb_h:>11.0f}"
                     f"{(f'{u.eff:.0%}' if u.done_cpu_h else '-'):>9}  "
                     f"{','.join(sorted(u.accounts))}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-S", "--start", default="now-7days")
    ap.add_argument("-E", "--end")
    ap.add_argument("-A", "--account", help="only count this account's jobs "
                    "(%%cpu stays relative to the whole cluster)")
    ap.add_argument("-n", "--top", type=int, default=10)
    ap.add_argument("--sort", choices=SORT_KEYS, default="cpu",
                    help="ranking metric (default cpu = requested CPU-hours)")
    ap.add_argument("--by-account", action="store_true",
                    help="group by account instead of user")
    args = ap.parse_args(argv)

    cmd = ["sacct", "-a", "-n", "-P", "-S", args.start,
           "-o", ",".join(FIELDS)]
    if args.end:
        cmd += ["-E", args.end]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        sys.exit(f"sacct failed: {p.stderr.strip()}")

    by = "Account" if args.by_account else "User"
    usage, cluster_cpu_h = aggregate(p.stdout, by, args.account)
    if not usage:
        sys.exit("no jobs found")
    print(render(usage, args.sort, args.top, by.lower(), cluster_cpu_h))
    print(f"\n{sum(u.jobs for u in usage.values())} jobs, "
          f"{len(usage)} {by.lower()}s since {args.start}", file=sys.stderr)


if __name__ == "__main__":
    main()
