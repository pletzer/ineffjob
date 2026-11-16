#!/usr/bin/env python3
"""ineffjob: find mismatches between requested and used resources in Slurm jobs.

Combines two sources:
  * `sacct -B`  -> the submitted batch script (its #SBATCH header)
  * `sacct -P`  -> accounting data (Elapsed, TotalCPU, MaxRSS, ...)

Checks (all on finished jobs):
  memory  peak RSS is a small fraction of the requested memory
  time    elapsed is a small fraction of the requested time limit
  cpu     CPU time used is a small fraction of (requested CPUs x elapsed)
"""
import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field

FIELDS = ["JobID", "State", "Elapsed", "Timelimit", "ReqMem", "TotalCPU",
          "MaxRSS", "NNodes", "NTasks", "ReqTRES", "User", "Account", "JobName"]


# --------------------------------------------------------------------------
# parsing helpers
# --------------------------------------------------------------------------
def parse_duration(s):
    """Slurm duration ([D-]HH:MM:SS, MM:SS, MM:SS.mmm) -> seconds, or None."""
    if not s or s in ("UNLIMITED", "Partition_Limit", "INVALID"):
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [float(p) for p in s.split(":")]
    while len(parts) < 3:  # MM:SS -> 0:MM:SS
        parts.insert(0, 0.0)
    h, m, sec = parts
    return days * 86400 + h * 3600 + m * 60 + sec


_UNITS = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 ** 2}


def parse_mem_mb(s):
    """Memory string ('4G', '4000M', '4634216K') -> MB, ignoring c/n suffix."""
    m = re.fullmatch(r"([\d.]+)\s*([KMGT]?)[a-zA-Z]?", (s or "").strip())
    if not m or not m.group(1):
        return None
    return float(m.group(1)) * _UNITS.get(m.group(2) or "M", 1)


def parse_reqtres(s):
    out = {}
    for kv in (s or "").split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = v
    return out


def parse_header(script):
    """Return {long-option: value} from #SBATCH lines (before first command)."""
    short = {"t": "time", "c": "cpus-per-task", "n": "ntasks", "N": "nodes",
             "J": "job-name", "A": "account", "p": "partition"}
    opts = {}
    for line in script.splitlines():
        line = line.strip()
        if not line or line.startswith("#!"):
            continue
        if not line.startswith("#"):
            break  # header ends at first non-comment line
        m = re.match(r"#SBATCH\s+(.*)", line)
        if not m:
            continue
        # drop trailing comments; handle "--opt=val", "--opt val", "-x val", "-xval"
        body = m.group(1).split(" #")[0].strip()
        m2 = re.match(r"--([\w-]+)(?:[=\s]\s*(\S+))?", body)
        if m2:
            opts[m2.group(1)] = m2.group(2) or ""
            continue
        m2 = re.match(r"-(\w)\s*=?\s*(\S+)?", body)
        if m2:
            val = m2.group(2) or ""
            opts[short.get(m2.group(1), m2.group(1))] = val
    return opts


def parse_batch_scripts(text):
    """Split `sacct -B` output into {jobid: script}."""
    scripts, cur, lines = {}, None, []
    for line in text.splitlines():
        m = re.match(r"Batch Script for (\S+)$", line)
        if m:
            if cur:
                scripts[cur] = "\n".join(lines).strip("\n")
            cur, lines = m.group(1), []
        elif cur and not re.fullmatch(r"-{20,}", line):
            lines.append(line)
    if cur:
        scripts[cur] = "\n".join(lines).strip("\n")
    return scripts


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------
@dataclass
class Job:
    jobid: str
    name: str = ""
    user: str = ""
    account: str = ""
    state: str = ""
    elapsed: float = 0.0
    timelimit: float = None
    req_mem_mb: float = None
    req_cpus: int = None
    total_cpu: float = 0.0
    max_rss_mb: float = 0.0
    nnodes: int = 1
    script: str = ""
    header: dict = field(default_factory=dict)


def parse_accounting(text):
    """Parse `sacct -P -n -o FIELDS` output into Jobs (steps folded in)."""
    jobs = {}
    for line in text.splitlines():
        cols = line.split("|")
        if len(cols) != len(FIELDS):
            continue
        r = dict(zip(FIELDS, cols))
        base, _, step = r["JobID"].partition(".")
        if "_" in base and base.split("_")[1].startswith("["):
            continue  # pending array placeholder
        if not step:
            tres = parse_reqtres(r["ReqTRES"])
            ncpu = tres.get("cpu")
            j = jobs.setdefault(base, Job(base))
            j.name, j.user, j.account = r["JobName"], r["User"], r["Account"]
            j.state = r["State"].split()[0]
            j.elapsed = parse_duration(r["Elapsed"]) or 0.0
            j.timelimit = parse_duration(r["Timelimit"])
            j.req_mem_mb = parse_mem_mb(r["ReqMem"])
            j.req_cpus = int(ncpu) if ncpu and ncpu.isdigit() else None
            j.total_cpu = parse_duration(r["TotalCPU"]) or 0.0
            j.nnodes = int(r["NNodes"] or 1)
        else:
            j = jobs.setdefault(base, Job(base))
            rss = parse_mem_mb(r["MaxRSS"])
            if rss:
                # MaxRSS is per task; scale by task count as an estimate of
                # the step's total footprint.
                ntasks = int(r["NTasks"]) if r["NTasks"].isdigit() else 1
                j.max_rss_mb = max(j.max_rss_mb, rss * ntasks)
    return jobs


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
@dataclass
class Finding:
    jobid: str
    kind: str
    message: str
    suggestion: str


def _fmt_dur(sec):
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    return f"{h}:{rem // 60:02d}:{rem % 60:02d}"


def _fmt_mem(mb):
    return f"{mb / 1024:.1f}G" if mb >= 1024 else f"{mb:.0f}M"


def _directive(job, *keys):
    for k in keys:
        if k in job.header:
            return f"#SBATCH --{k}={job.header[k]}"
    return None


def check_memory(job, thr):
    if job.state != "COMPLETED" or not job.req_mem_mb or job.max_rss_mb <= 0:
        return None
    frac = job.max_rss_mb / (job.req_mem_mb * 1.0)
    if frac >= thr:
        return None
    sugg = _fmt_mem(max(job.max_rss_mb * 1.5, 256))
    where = _directive(job, "mem", "mem-per-cpu") or "memory request"
    return Finding(job.jobid, "memory",
                   f"peak RSS {_fmt_mem(job.max_rss_mb)} of "
                   f"{_fmt_mem(job.req_mem_mb)} requested ({frac:.0%}) [{where}]",
                   f"--mem={sugg} (peak + 50%)")


def check_time(job, thr, min_limit):
    if job.state != "COMPLETED" or not job.timelimit:
        return None
    if job.timelimit < min_limit:
        return None
    frac = job.elapsed / job.timelimit
    if frac >= thr:
        return None
    sugg = _fmt_dur(max(job.elapsed * 1.5, 60))
    where = _directive(job, "time") or "time limit"
    return Finding(job.jobid, "time",
                   f"ran {_fmt_dur(job.elapsed)} of {_fmt_dur(job.timelimit)} "
                   f"requested ({frac:.0%}) [{where}]",
                   f"--time={sugg} (elapsed + 50%)")


def check_cpu(job, thr, min_elapsed):
    if (job.state != "COMPLETED" or not job.req_cpus or job.req_cpus < 2
            or job.elapsed < min_elapsed):
        return None
    eff = job.total_cpu / (job.elapsed * job.req_cpus)
    if eff >= thr:
        return None
    used = job.total_cpu / job.elapsed
    where = _directive(job, "cpus-per-task", "ntasks") or "CPU request"
    sugg = max(1, round(used + 0.5))
    return Finding(job.jobid, "cpu",
                   f"avg {used:.1f} CPUs busy of {job.req_cpus} requested "
                   f"({eff:.0%}) [{where}]",
                   f"request ~{sugg} CPU(s), or check the code is parallel")


def analyse(jobs, args):
    out = []
    for j in jobs.values():
        for f in (check_memory(j, args.mem_threshold),
                  check_time(j, args.time_threshold, args.min_timelimit * 60),
                  check_cpu(j, args.cpu_threshold, args.min_elapsed * 60)):
            if f:
                out.append(f)
    return out


# --------------------------------------------------------------------------
# sacct wrappers
# --------------------------------------------------------------------------
def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        sys.exit(f"{' '.join(cmd)} failed: {p.stderr.strip()}")
    return p.stdout


def sacct_filters(args):
    cmd = ["-S", args.start]
    if args.end:
        cmd += ["-E", args.end]
    if args.user:
        cmd += ["-u", args.user]
    if args.account:
        cmd += ["-A", args.account]
    if args.jobs:
        cmd += ["-j", args.jobs]
    return cmd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-S", "--start", default="now-7days")
    ap.add_argument("-E", "--end")
    ap.add_argument("-u", "--user")
    ap.add_argument("-A", "--account")
    ap.add_argument("-j", "--jobs", help="comma-separated job ids")
    ap.add_argument("--mem-threshold", type=float, default=0.5,
                    help="flag if peak RSS / requested < this (default 0.5)")
    ap.add_argument("--time-threshold", type=float, default=0.25,
                    help="flag if elapsed / time limit < this (default 0.25)")
    ap.add_argument("--cpu-threshold", type=float, default=0.5,
                    help="flag if CPU efficiency < this (default 0.5)")
    ap.add_argument("--min-timelimit", type=float, default=10,
                    help="ignore time check for limits under N minutes")
    ap.add_argument("--min-elapsed", type=float, default=1,
                    help="ignore cpu check for jobs shorter than N minutes")
    args = ap.parse_args(argv)

    filt = sacct_filters(args)
    acct = run(["sacct", "-P", "-n", "-o", ",".join(FIELDS)] + filt)
    jobs = parse_accounting(acct)
    if not jobs:
        sys.exit("no jobs found")
    scripts = {}
    ids = sorted(jobs)
    for i in range(0, len(ids), 200):  # -B needs explicit -j; chunk the list
        out = run(["sacct", "-B", "-X", "-j", ",".join(ids[i:i + 200])])
        scripts.update(parse_batch_scripts(out))
    for jid, script in scripts.items():
        if jid in jobs:
            jobs[jid].script = script
            jobs[jid].header = parse_header(script)

    findings = analyse(jobs, args)
    for f in sorted(findings, key=lambda f: (f.jobid, f.kind)):
        j = jobs[f.jobid]
        print(f"{f.jobid} ({j.name}) {f.kind.upper()}: {f.message}\n"
              f"    -> {f.suggestion}")
    print(f"\n{len(findings)} finding(s) across {len(jobs)} job(s)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
