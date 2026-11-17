#!/usr/bin/env python3
"""cpudrill: explain where the CPU inefficiency of a user/account comes from.

For every job it splits the requested CPU-hours (requested CPUs x elapsed)
into:
  used (user)    CPU time spent in user code
  used (system)  CPU time spent in the kernel (I/O, MPI busy-wait, ...)
  idle: no step  wall time when no srun step was running (serial setup /
                 post-processing, or waiting) x requested CPUs
  idle: in step  everything else: CPUs idle while a step was running
                 (load imbalance, serial sections, I/O waits, too many CPUs)
  not accounted  jobs still running: Slurm reports TotalCPU=0 until the end,
                 so their efficiency is unknown (excluded from percentages)

Run from the repo directory (it imports helpers from ineffjob.py).
"""
import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime

from slurmq import job_rows, windows
from ineffjob import parse_duration, parse_reqtres, physical_cores, add_threads_arg, set_threads_per_core
from topusers import IN_PROGRESS

FIELDS = ["JobID", "JobName", "State", "Elapsed", "Start", "End", "ReqTRES",
          "TotalCPU", "UserCPU", "SystemCPU", "NNodes", "NTasks", "User", "AllocCPUS"]


@dataclass
class Step:
    name: str
    start: float
    end: float
    total_cpu: float


@dataclass
class Job:
    jobid: str
    name: str = ""
    user: str = ""
    state: str = ""
    elapsed: float = 0.0
    start: float = 0.0
    end: float = 0.0
    req_cpus: float = 0.0   # physical cores (AllocCPUS / 2)
    nnodes: int = 1
    total_cpu: float = 0.0
    user_cpu: float = 0.0
    sys_cpu: float = 0.0
    steps: list = field(default_factory=list)

    @property
    def finished(self):
        return self.state not in IN_PROGRESS

    @property
    def req_cpu_s(self):
        return self.req_cpus * self.elapsed


def ts(s):
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:  # Unknown / None
        return None


def parse(text):
    jobs = {}
    for line in text.splitlines():
        cols = line.split("|")
        if len(cols) != len(FIELDS):
            continue
        r = dict(zip(FIELDS, cols))
        base, _, step = r["JobID"].partition(".")
        j = jobs.setdefault(base, Job(base))
        if not step:
            tres = parse_reqtres(r["ReqTRES"])
            j.name, j.user = r["JobName"], r["User"]
            j.state = r["State"].split()[0]
            j.elapsed = parse_duration(r["Elapsed"]) or 0.0
            j.start, j.end = ts(r["Start"]), ts(r["End"])
            j.req_cpus = physical_cores(r["AllocCPUS"], tres.get("cpu"))
            j.nnodes = int(r["NNodes"] or 1)
            j.total_cpu = parse_duration(r["TotalCPU"]) or 0.0
            j.user_cpu = parse_duration(r["UserCPU"]) or 0.0
            j.sys_cpu = parse_duration(r["SystemCPU"]) or 0.0
        elif step not in ("batch", "extern"):
            s, e = ts(r["Start"]), ts(r["End"])
            if s is not None and e is not None:
                j.steps.append(Step(step, s, e, parse_duration(r["TotalCPU"]) or 0.0))
    return jobs


def busy_wall(job):
    """Seconds covered by the union of srun step intervals."""
    spans = sorted((s.start, s.end) for s in job.steps)
    total, cur_s, cur_e = 0.0, None, None
    for s, e in spans:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return min(total, job.elapsed)


HOUR = 3600.0


def breakdown(job):
    """CPU-hours split for a finished job."""
    req = job.req_cpu_s / HOUR
    used_sys = min(job.sys_cpu, job.total_cpu) / HOUR
    used_usr = job.total_cpu / HOUR - used_sys
    if job.steps:
        no_step = job.req_cpus * (job.elapsed - busy_wall(job)) / HOUR
    else:  # nothing ran under srun: the batch step did all the work
        no_step = 0.0
    waste = max(req - job.total_cpu / HOUR, 0.0)
    no_step = min(max(no_step, 0.0), waste)
    return dict(req=req, usr=used_usr, sys=used_sys, no_step=no_step,
                in_step=waste - no_step)


def verdict(job, b):
    notes = []
    if not job.steps and job.req_cpus > 1:
        notes.append("no srun steps (batch step only)")
    if b["req"] and b["no_step"] / b["req"] > 0.1:
        notes.append(f"{b['no_step'] / b['req']:.0%} of CPU-h with no step running")
    if job.total_cpu and job.sys_cpu / job.total_cpu > 0.10:
        notes.append(f"system time {job.sys_cpu / job.total_cpu:.0%} of CPU time")
    if job.state in ("TIMEOUT", "FAILED", "OUT_OF_MEMORY", "CANCELLED", "NODE_FAIL"):
        notes.append(job.state.lower())
    return "; ".join(notes)


def pct(x, tot):
    return f"{x / tot:.0%}" if tot else "-"


def render(jobs, top):
    done = [j for j in jobs.values() if j.finished and j.req_cpus]
    live = [j for j in jobs.values() if not j.finished and j.req_cpus]
    rows = [(j, breakdown(j)) for j in done]
    tot = {k: sum(b[k] for _, b in rows) for k in ("req", "usr", "sys", "no_step", "in_step")}
    out = []

    out.append(f"Finished jobs: {len(done)}   still running/pending: {len(live)}")
    out.append(f"{'':<22}{'CPU-h':>12}{'share':>8}")
    for label, k in (("used (user)", "usr"), ("used (system)", "sys"),
                     ("idle: no step", "no_step"), ("idle: in step", "in_step")):
        out.append(f"{label:<22}{tot[k]:>12.0f}{pct(tot[k], tot['req']):>8}")
    out.append(f"{'requested':<22}{tot['req']:>12.0f}")
    if live:
        lh = sum(j.req_cpu_s for j in live) / HOUR
        out.append(f"{'not accounted (live)':<22}{lh:>12.0f}"
                   f"   <- excluded above: TotalCPU is 0 until jobs end")

    out.append("")
    out.append(f"{'job':<10}{'state':<11}{'elapsed':>10}{'cores':>6}{'req-h':>9}"
               f"{'eff':>6}{'sys%':>6}{'idle-nostep':>12}{'idle-instep':>12}  notes")
    rows.sort(key=lambda jb: jb[1]["no_step"] + jb[1]["in_step"], reverse=True)
    for j, b in rows[:top]:
        eff = (b["usr"] + b["sys"]) / b["req"] if b["req"] else 0
        sysp = b["sys"] / (b["usr"] + b["sys"]) if (b["usr"] + b["sys"]) else 0
        el = f"{int(j.elapsed // 3600)}:{int(j.elapsed % 3600 // 60):02d}"
        out.append(f"{j.jobid:<10}{j.state:<11}{el:>10}{j.req_cpus:>6.0f}"
                   f"{b['req']:>9.0f}{eff:>6.0%}{sysp:>6.0%}"
                   f"{b['no_step']:>12.0f}{b['in_step']:>12.0f}  {verdict(j, b)}")
    for j in sorted(live, key=lambda j: -j.req_cpu_s)[:top]:
        el = f"{int(j.elapsed // 3600)}:{int(j.elapsed % 3600 // 60):02d}"
        out.append(f"{j.jobid:<10}{j.state:<11}{el:>10}{j.req_cpus:>6.0f}"
                   f"{j.req_cpu_s / HOUR:>9.0f}{'?':>6}{'?':>6}{'?':>12}{'?':>12}"
                   f"  still running: efficiency unknown")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-S", "--start", default="now-7days")
    ap.add_argument("-E", "--end")
    ap.add_argument("-u", "--user")
    ap.add_argument("-A", "--account")
    ap.add_argument("-j", "--jobs")
    ap.add_argument("--chunk-days", type=int, default=3,
                    help="query sacct in windows of this many days (default 3)")
    ap.add_argument("-n", "--top", type=int, default=15,
                    help="jobs to list (default 15)")
    add_threads_arg(ap)
    args = ap.parse_args(argv)
    set_threads_per_core(args.threads_per_core)
    if not (args.user or args.account or args.jobs):
        ap.error("give at least one of -u, -A, -j")

    # Windows only enumerate job IDs (-X: light rows, avoids overloading
    # slurmdbd); the numbers then come from `sacct -j`, which is not clipped
    # to a window (see slurmq.py).
    enum = ["sacct", "-X", "-n", "-P", "-o", "JobID"]
    enum += ["-u", args.user] if args.user else ["-a"]
    for flag, val in (("-A", args.account), ("-j", args.jobs)):
        if val:
            enum += [flag, val]
    ids = set()
    for text, _, _ in windows(enum, args.start, args.end, args.chunk_days):
        ids.update(l.strip() for l in text.splitlines() if l.strip())
    jobs = parse("\n".join(job_rows(sorted(ids), FIELDS))) if ids else {}
    if not jobs:
        sys.exit("no jobs found")
    print(render(jobs, args.top))


if __name__ == "__main__":
    main()
