#!/usr/bin/env python3
"""scriptjobs: find big jobs that run Python, R or Julia scripts.

1. `sacct -a -X` lists jobs between two dates and keeps those that used at
   least -C core-hours (requested CPUs x elapsed, same convention as the
   other ineffjob tools).
2. `sacct -B` fetches the batch script of each of those jobs only.
3. The script body (comments, #SBATCH lines and `module ...` lines ignored)
   is scanned for interpreter invocations and script file names.

Limitations: only the submitted batch script is inspected, so an interpreter
launched indirectly (wrapper script, make, snakemake, container entrypoint)
is not detected. With -X (exclusive), jobs that also run any other program
(e.g. a compiled executable after a Python data-preparation step) are dropped:
every command must be an interpreter call or housekeeping (cd, mkdir, cp,
module, echo, control flow, ...). Unknown commands count as "other program", so
-X errs on the side of excluding. Scripts are only available if the site stores them
(AccountingStoreFlags=job_script); jobs without one are counted separately.

Run from the repo directory (it imports helpers from ineffjob.py).
"""
import argparse
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from ineffjob import parse_batch_scripts, parse_duration, parse_reqtres

FIELDS = ["JobID", "User", "Account", "JobName", "State", "Elapsed", "ReqTRES",
          "WorkDir"]

# Patterns are applied to command lines (a command position or any word).
LANG_PATTERNS = {
    "python": re.compile(
        r"(?:^|[\s;&|(`'\"/])(?:python[0-9.]*|ipython|pypy[0-9]*)(?=$|[\s;&|)`'\"])"
        r"|\.py\b"),
    "R": re.compile(
        r"(?:^|[\s;&|(`'\"/])(?:Rscript|R\s+(?:CMD|--\w+|-f|-e|<))(?=$|[\s;&|)`'\"])"
        r"|\.[Rr](?:md)?(?=$|[\s;&|)`'\"])"),
    "julia": re.compile(r"(?:^|[\s;&|(`'\"/])julia(?=$|[\s;&|)`'\"])|\.jl\b"),
}


@dataclass
class Hit:
    jobid: str
    user: str
    account: str
    name: str
    state: str
    core_hours: float
    langs: dict = field(default_factory=dict)  # lang -> [matching lines]
    other: list = field(default_factory=list)  # non-interpreter commands
    workdir: str = ""


HOUSEKEEPING = set("""
cd mkdir cp mv rm ln echo printf export unset set shopt module ml source . eval
ls cat date hostname pwd touch chmod chown tar gzip gunzip zip unzip sleep wait
exit return true false test [ [[ if then else elif fi for do done while until
case esac in function local read declare readonly trap sed awk grep cut sort
head tail wc tee find rsync scp basename dirname mktemp nvidia-smi squeue sacct
seff mail curl wget git pip pip3 uv conda mamba micromamba activate deactivate
: { } ! getopts shift
tr uniq paste break continue seq expr bc cmp diff du df stat readlink realpath ps kill nproc
flock lscpu free printenv env xargs ulimit umask mkfifo md5sum sha256sum column
""".split())
WRAPPERS = {"conda", "mamba", "micromamba", "uv", "poetry", "pipenv", "pixi"}
LAUNCHERS = {"srun", "mpirun", "mpiexec", "mpiexec.hydra", "mpiexec.openmpi",
             "time", "nohup", "exec", "command", "nice", "env", "stdbuf",
             "taskset", "numactl"}
CONTAINERS = {"apptainer", "singularity"}
VALUE_OPTS = {"-n", "-N", "-c", "-np", "-t", "-p", "-J", "-A", "-w", "-x", "-o",
              "-e", "-m", "-C", "-d", "-D", "-u"}
INTERPRETER_WORD = re.compile(r"(?:python[0-9.]*|ipython|pypy[0-9]*|Rscript|R|julia)$")
CONTROL_PREFIX = re.compile(r"^(?:then|do|else|elif|if|while|until|!|\(|\{)\s+")


def logical_lines(script):
    """Join backslash continuations, drop comments and heredoc bodies."""
    out, buf, heredoc = [], "", None
    for raw in script.splitlines():
        if heredoc:
            if raw.strip() == heredoc:
                heredoc = None
            continue
        if not buf and raw.strip().startswith("#"):
            continue  # comment line (may contain stray quotes)
        line = buf + raw.strip()
        if line.endswith("\\"):
            buf = line[:-1] + " "
            continue
        if (line.count("'") % 2 or line.count('"') % 2
                or line.count("$(") > line.count(")")
                or line.count("((") > line.count("))")):
            buf = line + " "  # inside a multi-line quoted string
            continue
        buf = ""
        if not line or line.startswith("#"):
            continue
        if '"' not in line and "'" not in line:
            line = re.sub(r"\s+#.*$", "", line)
        m = re.search(r"<<-?\s*['\"]?(\w+)['\"]?", line)
        if m:
            heredoc = m.group(1)
        out.append(line)
    if buf:
        out.append(buf)
    return out


def split_commands(line):
    """Split on ; && || | outside quotes, $(...) and ${...}."""
    segs, cur, quote, depth, i = [], [], None, 0, 0
    while i < len(line):
        c = line[i]
        if quote:
            if c == "\\" and quote == '"':
                cur.append(line[i:i + 2]); i += 2; continue
            if c == quote:
                quote = None
        elif c in "'\"":
            quote = c
        elif c in "({" and i and line[i - 1] == "$" or (c == "(" and depth):
            depth += 1
        elif c in ")}" and depth:
            depth -= 1
        elif not depth and (c == ";" or c == "|" or line.startswith("&&", i)):
            segs.append("".join(cur)); cur = []
            i += 2 if line.startswith(("&&", "||"), i) else 1
            continue
        cur.append(c); i += 1
    segs.append("".join(cur))
    return segs


def command_word(tokens):
    """Strip env assignments and launcher/container prefixes; return the
    remaining token list (the real command first)."""
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if re.match(r"^[A-Za-z_]\w*=", t):
            i += 1
        elif t in LAUNCHERS:
            i += 1
            while i < len(tokens) and (tokens[i].startswith("-")
                                       or re.match(r"^[A-Za-z_]\w*=", tokens[i])):
                skip = tokens[i] in VALUE_OPTS and "=" not in tokens[i]
                i += 2 if skip else 1
        elif t in CONTAINERS and i + 1 < len(tokens) and tokens[i + 1] in ("exec", "run"):
            i += 2
            while i < len(tokens) and tokens[i].startswith("-"):
                i += 2 if tokens[i] in ("--bind", "-B", "--env", "--pwd") else 1
            i += 1  # the image
        else:
            break
    if i < len(tokens) and re.match(r"^\d*[<>]", tokens[i]):
        return []  # e.g. `exec 9>lock`: redirection only, no program
    return tokens[i:]


def function_aliases(lines, langs):
    """Shell functions whose body runs an interpreter -> {name: [langs]}."""
    aliases = {}
    for n, line in enumerate(lines):
        m = re.match(r"(?:function\s+)?([A-Za-z_]\w*)\s*\(\)", line)
        if m:
            body = " ".join(lines[n:n + 6])
            hit = [lang for lang in langs if LANG_PATTERNS[lang].search(body)]
            if hit:
                aliases[m.group(1)] = hit
    return aliases


def defined_functions(lines):
    out = set()
    for line in lines:
        m = re.match(r"(?:function\s+)?([A-Za-z_]\w*)\s*\(\)", line)
        if m:
            out.add(m.group(1))
    return out


def classify_script(script, langs):
    """Return ({lang: [lines]}, [other-program lines]) for the script body."""
    found, other = {}, []
    lines = logical_lines(script)
    aliases = function_aliases(lines, langs)
    functions = defined_functions(lines)  # bodies are scanned in place
    for line in lines:
        line = re.sub(r"\[\[.*?\]\]|(?<![\w$])\[ .*? \]", "true", line)
        for seg in split_commands(line):
            seg = CONTROL_PREFIX.sub("", seg.strip())
            seg = re.sub(r"^[\w*|\"'.-]+\)\s*", "", seg)  # case label `a|b)`
            if seg.startswith("(("):  # arithmetic
                continue
            if not seg:
                continue
            seg = re.sub(r"\$\(\(.*?\)\)", "0", seg)  # arithmetic expansion
            sub = re.search(r"\$\((.*)\)|`(.*?)`", seg)
            if re.match(r"^(?:export\s+)?[A-Za-z_]\w*=", seg) and sub:
                f2, o2 = classify_script(sub.group(1) or sub.group(2), langs)
                for lang, ls in f2.items():
                    found.setdefault(lang, []).extend(ls)
                other.extend(o2)
                continue
            try:
                toks = shlex.split(seg)
            except ValueError:
                toks = seg.split()
            toks = command_word(toks)
            if not toks:
                continue
            cmd = toks[0]
            base = cmd.rsplit("/", 1)[-1]
            hit = [lang for lang in langs if LANG_PATTERNS[lang].search(seg)]
            if base in aliases:
                for lang in aliases[base]:
                    found.setdefault(lang, []).append(seg)
            elif base in functions:
                pass
            elif INTERPRETER_WORD.match(base) or (
                    base in WRAPPERS and hit and base not in ("conda",) or
                    base == "conda" and "run" in toks and hit):
                for lang in hit:
                    found.setdefault(lang, []).append(seg)
                if not hit and INTERPRETER_WORD.match(base):
                    other.append(seg)  # interpreter of a language not asked for
            elif re.search(r"\.(py|R|r|jl)$", cmd) and hit:
                for lang in hit:  # executable script, e.g. ./train.py
                    found.setdefault(lang, []).append(seg)
            elif base in HOUSEKEEPING or cmd.endswith("()") or cmd.endswith("=") \
                    or re.match(r"^[A-Za-z_]\w*=", cmd):
                pass  # housekeeping
            else:
                other.append(seg)
    return found, other


def parse_jobs(text, min_core_hours):
    """Job rows from `sacct -X -P -n` with core-hours >= threshold."""
    jobs = {}
    for line in text.splitlines():
        cols = line.split("|")
        if len(cols) != len(FIELDS):
            continue
        r = dict(zip(FIELDS, cols))
        tres = parse_reqtres(r["ReqTRES"])
        ncpu = int(tres["cpu"]) if tres.get("cpu", "").isdigit() else 0
        ch = ncpu * (parse_duration(r["Elapsed"]) or 0.0) / 3600
        if ch >= min_core_hours:
            jobs[r["JobID"]] = Hit(r["JobID"], r["User"], r["Account"],
                                   r["JobName"], r["State"].split()[0], ch,
                                   workdir=r["WorkDir"])
    return jobs


def run(cmd, retries=3):
    """Run sacct; retry with backoff if slurmdbd refuses the connection."""
    for attempt in range(retries + 1):
        p = subprocess.run(cmd, capture_output=True, text=True)
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


def fetch_jobs(start, end, min_core_hours, user=None, account=None,
               chunk_days=3):
    """Jobs in [start, end] with >= min_core_hours, keyed by JobID.

    Queried in small windows: one huge cluster-wide sacct can overload
    slurmdbd. Jobs spanning windows repeat but are identical (Elapsed is not
    clipped to the window), so merging by JobID is safe.
    """
    jobs = {}
    chunks = date_chunks(start, end, chunk_days)
    for n, (cs, ce) in enumerate(chunks, 1):
        cmd = ["sacct", "-X", "-n", "-P", "-S", cs, "-E", ce,
               "-o", ",".join(FIELDS)]
        cmd += ["-u", user] if user else ["-a"]
        if account:
            cmd += ["-A", account]
        if len(chunks) > 1:
            print(f"querying {cs[:10]} .. {ce[:10]} ({n}/{len(chunks)})",
                  file=sys.stderr)
        jobs.update(parse_jobs(run(cmd), min_core_hours))
    return jobs


def fetch_scripts(ids):
    """{jobid: batch script} via sacct -B (needs explicit -j, so chunked)."""
    scripts = {}
    for i in range(0, len(ids), 200):
        scripts.update(parse_batch_scripts(
            run(["sacct", "-B", "-X", "-j", ",".join(ids[i:i + 200])])))
    return scripts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-S", "--start", required=True, help="start date")
    ap.add_argument("-E", "--end", required=True, help="end date")
    ap.add_argument("-C", "--min-core-hours", type=float, required=True,
                    help="only jobs that used at least this many core-hours")
    ap.add_argument("-X", "--exclusive", action="store_true",
                    help="only jobs that run nothing but scripts (no compiled "
                         "or other executables besides housekeeping commands)")
    ap.add_argument("-u", "--user")
    ap.add_argument("-A", "--account")
    ap.add_argument("-l", "--lang", action="append", choices=list(LANG_PATTERNS),
                    help="language(s) to look for (repeatable; default all)")
    ap.add_argument("--chunk-days", type=int, default=3,
                    help="query sacct in windows of this many days (default 3)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print the matching script lines")
    args = ap.parse_args(argv)
    langs = args.lang or list(LANG_PATTERNS)

    jobs = fetch_jobs(args.start, args.end, args.min_core_hours, args.user,
                      args.account, args.chunk_days)
    print(f"{len(jobs)} job(s) over {args.min_core_hours:g} core-hours; "
          f"inspecting scripts...", file=sys.stderr)
    ids = sorted(jobs)
    scripts = fetch_scripts(ids)

    hits, noscript, mixed = [], 0, 0
    for jid in ids:
        script = scripts.get(jid)
        if not script:
            noscript += 1
            continue
        found, other = classify_script(script, langs)
        if not found:
            continue
        if args.exclusive and other:
            mixed += 1
            continue
        jobs[jid].langs, jobs[jid].other = found, other
        hits.append(jobs[jid])

    hits.sort(key=lambda h: h.core_hours, reverse=True)
    print(f"{'jobid':<14}{'user':<14}{'account':<12}{'core-h':>10}  "
          f"{'state':<11}{'lang':<14}name")
    for h in hits:
        print(f"{h.jobid:<14}{h.user:<14}{h.account:<12}{h.core_hours:>10.0f}  "
              f"{h.state:<11}{','.join(h.langs):<14}{h.name}")
        if args.verbose:
            for lang, lines in h.langs.items():
                for ln in lines[:3]:
                    print(f"{'':<12}[{lang}] {ln[:110]}")
            for ln in h.other[:3]:
                print(f"{'':<12}[other] {ln[:110]}")
    print(f"\n{len(hits)} of {len(jobs)} job(s) run {'/'.join(langs)}; "
          f"{noscript} had no stored script"
          + (f"; {mixed} excluded for also running other programs" if args.exclusive else ""),
          file=sys.stderr)


if __name__ == "__main__":
    main()
