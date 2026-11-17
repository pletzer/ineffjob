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
          "MaxRSS", "NNodes", "NTasks", "ReqTRES", "User", "Account", "JobName",
          "AllocCPUS"]

# Nodes run several hardware threads per core and Slurm allocates whole cores,
# so AllocCPUS counts threads. Efficiency is measured against physical cores
# (AllocCPUS / threads per core); using every hyperthread can exceed 100%
# (up to 100% x threads per core). Set with --threads-per-core.
THREADS_PER_CORE = 2


def set_threads_per_core(n):
    global THREADS_PER_CORE
    if n < 1:
        raise ValueError("threads per core must be >= 1")
    THREADS_PER_CORE = n


def add_threads_arg(ap):
    ap.add_argument("--threads-per-core", type=int, default=THREADS_PER_CORE,
                    help="hardware threads per core: AllocCPUS is divided by "
                    "this to get physical cores (default %(default)s)")


def physical_cores(alloc_cpus, reqtres_cpu=None):
    """Physical cores of an allocation (falls back to ReqTRES cpu)."""
    if str(alloc_cpus or "").isdigit() and int(alloc_cpus) > 0:
        return int(alloc_cpus) / THREADS_PER_CORE
    return float(reqtres_cpu) if str(reqtres_cpu or "").isdigit() else 0.0


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
    req_cpus: float = None   # physical cores (AllocCPUS / 2)
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
            j.req_cpus = physical_cores(r["AllocCPUS"], ncpu) or None
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
    kind: str       # cpu | memory | time
    name: str
    where: str      # the #SBATCH directive responsible
    req: float      # requested: CPUs / MB / seconds
    used: float     # used: avg busy CPUs / peak MB / elapsed seconds
    waste: float    # CPU-h idle / GB-h reserved but unused / hours of excess limit


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
    if job.max_rss_mb / job.req_mem_mb >= thr:
        return None
    waste = (job.req_mem_mb - job.max_rss_mb) / 1024 * job.elapsed / 3600
    return Finding(job.jobid, "memory", job.name,
                   _directive(job, "mem", "mem-per-cpu") or "memory request",
                   job.req_mem_mb, job.max_rss_mb, waste)


def check_time(job, thr, min_limit):
    if job.state != "COMPLETED" or not job.timelimit:
        return None
    if job.timelimit < min_limit or job.elapsed / job.timelimit >= thr:
        return None
    return Finding(job.jobid, "time", job.name,
                   _directive(job, "time") or "time limit",
                   job.timelimit, job.elapsed,
                   (job.timelimit - job.elapsed) / 3600)


def check_cpu(job, thr, min_elapsed):
    if (job.state != "COMPLETED" or not job.req_cpus or job.req_cpus < 2
            or job.elapsed < min_elapsed):
        return None
    if job.total_cpu / (job.elapsed * job.req_cpus) >= thr:
        return None
    used = job.total_cpu / job.elapsed
    return Finding(job.jobid, "cpu", job.name,
                   _directive(job, "cpus-per-task", "ntasks") or "CPU request",
                   job.req_cpus, used,
                   (job.req_cpus - used) * job.elapsed / 3600)


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
# report
# --------------------------------------------------------------------------
KIND_ORDER = {"cpu": 0, "memory": 1, "time": 2}
CHECK_LABEL = {"cpu": "CPU", "memory": "MEM", "time": "TIME"}


def _id_range(ids):
    ids = sorted(ids, key=lambda i: [int(p) if p.isdigit() else 0
                                     for p in re.split(r"[_.]", i)])
    if len(ids) == 1:
        return ids[0]
    bases = {i.split("_")[0] for i in ids}
    if len(bases) == 1 and all("_" in i for i in ids):  # array tasks
        return f"{ids[0].split('_')[0]}_[{ids[0].split('_')[1]}-{ids[-1].split('_')[1]}]"
    return f"{ids[0]}..{ids[-1]}"


def summarise(findings):
    """Group findings of the same check, job name and directive into rows.

    Suggestions are sized for the worst job in the group, so they are safe
    to apply to all of them.
    """
    groups = {}
    for f in findings:
        # drop trailing "(params)" so sweeps of one job group together
        base = re.sub(r"[_\s]*\(.*\)$", "", f.name) or f.name
        groups.setdefault((f.kind, base, f.where), []).append(f)
    rows = []
    for (kind, name, where), fs in groups.items():
        req = max(f.req for f in fs)
        used = max(f.used for f in fs)   # worst case across the group
        typical = sorted(f.used / f.req for f in fs)[len(fs) // 2]
        if kind == "cpu":
            r, u = f"{req:g} cores", f"{used:.1f} cores"
            sugg = f"{max(1, round(used + 0.5))} core(s), or check the code is parallel"
        elif kind == "memory":
            r, u = _fmt_mem(req), _fmt_mem(used)
            sugg = f"--mem={_fmt_mem(max(used * 1.5, 256))}"
        else:
            r, u = _fmt_dur(req), _fmt_dur(used)
            sugg = f"--time={_fmt_dur(max(used * 1.5, 60))}"
        rows.append(dict(kind=kind, name=name, ids=_id_range([f.jobid for f in fs]),
                         n=len(fs), req=r, used=u, typ=typical,
                         waste=sum(f.waste for f in fs), sugg=sugg, where=where))
    # most costly first: CPU waste, then memory, then time limits
    rows.sort(key=lambda r: (KIND_ORDER[r["kind"]], -r["waste"]))
    return rows


def render_report(findings, njobs, top=25):
    if not findings:
        return f"No inefficiencies found across {njobs} job(s)."
    rows = summarise(findings)
    hidden = max(len(rows) - top, 0)
    rows = rows[:top]
    tot = {k: sum(f.waste for f in findings if f.kind == k) for k in KIND_ORDER}
    out = [f"{len(findings)} finding(s) in {len({f.jobid for f in findings})} of "
           f"{njobs} job(s), {len(rows)} distinct problem(s)",
           f"  cores idle:        {tot['cpu']:>10,.0f} core-hours",
           f"  memory unused:     {tot['memory']:>10,.0f} GB-hours reserved",
           f"  time limit unused: {tot['time']:>10,.0f} hours (hurts backfill, not charged)",
           ""]
    unit = {"cpu": "core-h", "memory": "GB-h", "time": "h"}
    hdr = ["check", "jobs", "name", "job ids", "requested", "used(max)",
           "typ.eff", "waste", "suggested change"]
    table = [hdr]
    for r in rows:
        table.append([CHECK_LABEL[r["kind"]], str(r["n"]), r["name"][:32], r["ids"],
                      r["req"], r["used"], f"{r['typ']:.0%}",
                      f"{r['waste']:,.0f} {unit[r['kind']]}", r["sugg"]])
    w = [max(len(row[i]) for row in table) for i in range(len(hdr))]
    for i, row in enumerate(table):
        out.append("  ".join(c.ljust(w[j]) if j in (0, 2, 3, 8) else c.rjust(w[j])
                             for j, c in enumerate(row)).rstrip())
        if i == 0:
            out.append("  ".join("-" * x for x in w))
    if hidden:
        out.append(f"... {hidden} smaller problem(s) not shown (use --top)")
    return "\n".join(out)


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
    ap.add_argument("--top", type=int, default=25,
                    help="max rows in the report (default 25)")
    add_threads_arg(ap)
    args = ap.parse_args(argv)
    set_threads_per_core(args.threads_per_core)

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

    print(render_report(analyse(jobs, args), len(jobs), args.top))


if __name__ == "__main__":
    main()
