#!/usr/bin/env python3
"""buildcheck: is a compiled executable a debug / unoptimised build?

Two ways to use it:

  buildcheck.py EXE [EXE ...]                    inspect executables directly
  buildcheck.py -S 2026-11-01 -E 2026-11-16 -C 5000 [-u USER] [-A ACCOUNT]
      find big jobs, extract the executables their batch scripts run, locate
      them (absolute path, job WorkDir, `module show` PATHs, current PATH) and
      inspect each one, ranked by the core-hours of the jobs that ran it.

Evidence, strongest first (the optimisation level is NOT stored in an ELF
file, so this is inference):
  1. build records next to the install: EasyBuild (*-easybuild-devel, .env,
     .eb), Spack (.spack/spack-build-env.txt), CMake build type. These hold the
     real CFLAGS/FFLAGS and are authoritative for that install.
  2. recorded flags: compilers that record their command line in the DWARF
     producer string (GCC -grecord-gcc-switches, default on many distros)
     give the literal -O level. Needs debug info (.debug_str).
  3. disassembly heuristics (x86-64), WEAK: -O0 code addresses nearly every
     local through the frame pointer, e.g. -0x14(%rbp); optimised GCC code
     does not. Validated on GCC only: Intel-compiled code and hand-written
     assembly (MKL) can look -O0 when they are not, so for Intel binaries the
     result is reported as inconclusive.
  4. markers: assertions left enabled (__assert_fail), sanitizers, gprof.

Executables are only read with readelf/objdump, never run. Run from the repo
directory (imports helpers from scriptjobs.py / ineffjob.py).
"""
import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

import scriptjobs
from scriptjobs import classify_script, command_word, fetch_jobs, fetch_scripts

# fraction of instructions addressing memory via %rbp above which code looks -O0
O0_RBP_FRACTION = 0.12


# --------------------------------------------------------------------------
# build records (EasyBuild / Spack / CMake)
# --------------------------------------------------------------------------
FLAG_VARS = r"(?:EBVAR)?(?:CFLAGS|CXXFLAGS|FFLAGS|F90FLAGS|FCFLAGS|F77FLAGS|OFLAG|CPPFLAGS)"
RECORD_GLOBS = ("easybuild/*-easybuild-devel", "easybuild/reprod/*.env",
                "easybuild/*.eb", ".spack/spack-build-env.txt",
                ".spack/spack-build-out.txt", "CMakeCache.txt")


def opt_of(flags):
    opts = re.findall(r"(?<!\S)-O(\d|s|g|fast|z)?(?!\S)", flags)
    return "O" + (opts[-1] or "1") if opts else None


def parse_record(text):
    """Return (flags-string, level) from an EasyBuild/Spack/CMake record."""
    flags = []
    for m in re.finditer(r'(?:setenv\(\s*"|export\s+|^|\s)' + FLAG_VARS
                         + r'"?\s*[,=]\s*"?([^"\n]*)', text, re.M):
        if "-O" in m.group(1) or "-g" in m.group(1):
            flags.append(m.group(1).strip())
    level = opt_of(" ".join(flags)) if flags else None
    if re.search(r"'noopt':\s*True", text):
        level = "O0"
    m = re.search(r"CMAKE_BUILD_TYPE(?::\w+)?\s*[=:]\s*\"?(\w+)", text)
    if m and not level:
        level = {"Debug": "O0", "Release": "O3", "RelWithDebInfo": "O2",
                 "MinSizeRel": "Os"}.get(m.group(1))
        flags.append(f"CMAKE_BUILD_TYPE={m.group(1)}")
    return (flags[0] if flags else ""), level


def find_record(path, levels_up=5):
    """Look for build records in the install tree around the executable."""
    import glob
    d = os.path.dirname(os.path.realpath(path))
    for _ in range(levels_up):
        for g in RECORD_GLOBS:
            for f in sorted(glob.glob(os.path.join(d, g))):
                try:
                    text = open(f, errors="replace").read(2_000_000)
                except OSError:
                    continue
                flags, level = parse_record(text)
                if level:
                    return {"source": f, "flags": flags[:160], "level": level}
        if d == "/":
            break
        d = os.path.dirname(d)
    return {}


# --------------------------------------------------------------------------
# inspecting one binary
# --------------------------------------------------------------------------
@dataclass
class Report:
    path: str
    error: str = ""
    arch: str = ""
    compilers: list = field(default_factory=list)    # from .comment
    debug_info: bool = False
    stripped: bool = False
    producers: dict = field(default_factory=dict)    # producer string -> -O level
    markers: list = field(default_factory=list)
    insns: int = 0
    rbp_fraction: float = None
    avx: int = 0
    avx512: int = 0
    fma: int = 0
    regions: list = field(default_factory=list)  # (insns, rbp_frac, avx_frac)
    o0_funcs: list = field(default_factory=list)  # (name, insns, rbp_frac)
    o0_share: float = None   # --deep: share of instructions in -O0-like functions
    intel: bool = False      # Intel compiler runtime seen (heuristics unreliable)
    record: dict = field(default_factory=dict)  # build record: source, flags, level
    verdict: str = ""
    level: str = "unknown"   # O0 | optimised | suspect | no-sign | unknown
    evidence: str = ""       # record | recorded flags | heuristic


def sh(cmd, timeout=120):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return ""
    return p.stdout


def is_elf(path):
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


PRODUCER = re.compile(
    r"^(GNU (?:C|C\+\+|Fortran|Objective-C|GIMPLE)\S* \d[^\n]*"
    r"|clang version [^\n]*|Intel\(R\)[^\n]*|AMD clang[^\n]*|flang[^\n]*"
    r"|NVIDIA[^\n]*|Cray[^\n]*)$")


def parse_opt(producer):
    """-O level recorded in a producer string, or None if no flags recorded."""
    flags = [t for t in producer.split() if t.startswith("-")]
    opts = [t for t in flags if re.fullmatch(r"-O(\d|s|g|fast|z)?", t)]
    if opts:
        return "O" + (opts[-1][2:] or "1")
    # GCC records the flags it was given; no -O among them means -O0.
    # A bare "GNU C17 13.2.0" (no flags at all) tells us nothing.
    return "O0" if flags else None


def section_strings(path, section):
    out = sh(["readelf", "-p", section, path])
    return [m.group(1) for m in re.finditer(r"^\s*\[\s*[0-9a-f]+\]\s+(.*)$", out, re.M)]


WINDOW = 256 * 1024      # bytes of .text per sampled region
MAX_REGIONS = 24


def text_section(path):
    out = sh(["readelf", "-S", "-W", path])
    m = re.search(r"\]\s+\.text\s+PROGBITS\s+([0-9a-f]+)\s+[0-9a-f]+\s+([0-9a-f]+)", out)
    return (int(m.group(1), 16), int(m.group(2), 16)) if m else (0, 0)


def count_lines(lines):
    """-> insns, rbp-relative, AVX (ymm), AVX-512 (zmm), FMA."""
    n = rbp = avx = avx512 = fma = 0
    for line in lines:
        if "\t" not in line:
            continue  # headers / labels
        n += 1
        if "(%rbp)" in line:
            rbp += 1
        if "%zmm" in line:
            avx512 += 1
        elif "%ymm" in line:
            avx += 1
        if "vfmadd" in line or "vfnmadd" in line or "vfmsub" in line:
            fma += 1
    return n, rbp, avx, avx512, fma


def disasm_regions(path, timeout=120):
    """Sample evenly spaced windows of .text (the whole thing if it is small).
    Returns per-region (insns, rbp_frac, avx_frac) plus totals."""
    base, size = text_section(path)
    if not size:
        return [], (0, 0, 0, 0, 0)
    if size <= 6 * WINDOW:
        wins = [(base, size)]
    else:
        k = min(MAX_REGIONS, size // WINDOW)
        wins = [(base + int(size * (i + 0.5) / k) - WINDOW // 2, WINDOW)
                for i in range(k)]
    regions, tot = [], [0, 0, 0, 0, 0]
    for start, length in wins:
        out = sh(["objdump", "-d", "--no-show-raw-insn", f"--start-address={start}",
                  f"--stop-address={start + length}", path], timeout)
        c = count_lines(out.splitlines())
        if c[0]:
            regions.append((c[0], c[1] / c[0], (c[2] + c[3]) / c[0]))
        tot = [a + b for a, b in zip(tot, c)]
    return regions, tuple(tot)


FUNC = re.compile(r"^[0-9a-f]+ <(.+)>:$")


def deep_scan(path, timeout=300, min_insns=200):
    """Per-function scan of all of .text. Returns (o0-like functions, share)."""
    proc = subprocess.Popen(["objdump", "-d", "-C", "-j", ".text", "--no-show-raw-insn",
                             path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, errors="replace")
    funcs, name, n, rbp, total = [], None, 0, 0, 0
    mov = simd = 0
    t0 = time.time()

    def close():
        nonlocal name, n, rbp, mov, simd
        # skip hand-written SIMD kernels and long constant-init blocks
        if (name and n >= min_insns and rbp / n >= O0_RBP_FRACTION
                and simd / n < 0.05 and mov / n < 0.9):
            funcs.append((name, n, rbp / n))
        name, n, rbp, mov, simd = None, 0, 0, 0, 0

    try:
        for line in proc.stdout:
            m = FUNC.match(line)
            if m:
                close()
                name = m.group(1)
            elif "\t" in line:
                n += 1
                total += 1
                rbp += "(%rbp)" in line
                body = line.split("\t", 1)[1].lstrip()
                mov += body.startswith("mov")
                simd += "%ymm" in body or "%zmm" in body
                if total % 500000 == 0 and time.time() - t0 > timeout:
                    break
        close()
    finally:
        proc.kill()
        proc.wait()
    o0 = sum(f[1] for f in funcs)
    funcs.sort(key=lambda f: -f[1])
    return funcs, (o0 / total if total else None)


def inspect(path, disasm=True, deep=False):
    r = Report(path)
    if not is_elf(path):
        r.error = "not an ELF executable (script or unreadable)"
        return r
    hdr = sh(["readelf", "-h", "-W", path])
    m = re.search(r"Machine:\s+(.*)", hdr)
    r.arch = m.group(1).strip() if m else "?"
    secs = sh(["readelf", "-S", "-W", path])
    r.debug_info = bool(re.search(r"\.z?debug_info", secs))
    r.stripped = ".symtab" not in secs

    comp = []
    for s in section_strings(path, ".comment"):
        if s and s not in comp:
            comp.append(s)
    r.compilers = comp[:4]

    if r.debug_info:
        for sec in (".debug_str", ".debug_line_str"):
            for s in section_strings(path, sec):
                if PRODUCER.match(s):
                    r.producers[s] = parse_opt(s)

    r.record = find_record(path)

    syms = sh(["readelf", "-s", "-W", path])
    r.intel = bool(re.search(r"\b__intel_|\b_intel_|\bfor_(?:write|read|open|close|"
                             r"alloc|dealloc|stop)\w*|\bifort_", syms))
    for marker, pat in (("assert() calls present (may come from libraries)", r"\b__assert_fail\b|\b__assert_rtn\b"),
                        ("sanitizer", r"\b__[amtu]?san_init\b|\b__ubsan_|\b__asan_"),
                        ("gprof (-pg)", r"\b_?mcount\b|\b__gnu_mcount_nc\b|\b_mcleanup\b")):
        if re.search(pat, syms):
            r.markers.append(marker)

    if disasm and "x86-64" in r.arch.lower():
        r.regions, (n, rbp, r.avx, r.avx512, r.fma) = disasm_regions(path)
        r.insns = n
        r.rbp_fraction = rbp / n if n else None
        if deep:
            r.o0_funcs, r.o0_share = deep_scan(path)

    decide(r)
    return r


def decide(r):
    # 1. build record: what the site's build system says
    if r.record:
        lvl = r.record["level"]
        r.level = "O0" if lvl == "O0" else "optimised"
        r.evidence = "record"
        r.verdict = (f"built with {lvl} per build record "
                     f"({os.path.basename(r.record['source'])}: {r.record['flags'] or '-'})")
        return
    # 2. flags recorded in the binary itself
    levels = {v for v in r.producers.values() if v}
    n_o0 = sum(1 for v in r.producers.values() if v == "O0")
    if levels:
        r.evidence = "recorded flags"
        if n_o0:
            r.level = "O0"
            r.verdict = (f"-O0 recorded in {n_o0} of {len(r.producers)} compiler "
                         f"producer(s)" + ("" if n_o0 == len(r.producers)
                                           else " (mixed build)"))
        else:
            r.level = "optimised"
            r.verdict = "built with " + "/".join(sorted(levels)) + " (recorded flags)"
        return
    # 3. heuristics: weak evidence, validated on GCC only
    if r.regions and r.insns >= 100:
        r.evidence = "heuristic"
        o0 = [g for g in r.regions if g[1] >= O0_RBP_FRACTION]
        k = len(r.regions)
        if r.intel and o0:
            r.verdict = (f"inconclusive: {len(o0)} of {k} sampled regions look -O0, "
                         f"but this is Intel-compiled code, for which the heuristic "
                         f"is unreliable; look for a build record or log")
        elif len(o0) == k:
            r.level = "suspect"
            r.verdict = (f"suspect -O0 (heuristic): {r.rbp_fraction:.0%} of instructions "
                         f"are %rbp-relative; no flags recorded")
        elif not o0:
            r.level = "no-sign"
            r.verdict = (f"no sign of -O0 ({r.rbp_fraction:.1%} rbp-relative); "
                         f"flags not recorded")
        else:
            r.level = "suspect"
            r.verdict = (f"partly suspect: {len(o0)} of {k} sampled regions look -O0 "
                         f"(heuristic); some component may be unoptimised")
        return
    r.verdict = "cannot tell (no build record or recorded flags; disassembly unavailable)"


def notes(r):
    out = list(r.markers)
    if r.intel:
        out.append("Intel compiler runtime")
    if r.debug_info:
        out.append("debug info")
    if r.insns and r.arch and "x86-64" in r.arch.lower() and not (r.avx or r.avx512):
        out.append("no AVX/AVX-512 in sampled code (generic x86-64 target?)")
    return out


# --------------------------------------------------------------------------
# finding the executables a job runs
# --------------------------------------------------------------------------
SYSTEM_DIRS = ("/usr/bin/", "/bin/", "/usr/sbin/", "/sbin/", "/usr/lib/",
               "/usr/lib64/", "/usr/libexec/", "/usr/local/bin/")
SHELLS = {"bash", "sh", "zsh", "csh", "tcsh", "dash", "source", "python", "python3"}
_module_cache = {}


def module_names(script):
    names = []
    for line in scriptjobs.logical_lines(script):
        m = re.match(r"(?:module|ml)\s+(?:(?:load|add)\s+)?(.*)$", line)
        if m and not re.match(r"(?:module|ml)\s+(?:purge|unload|swap|list|"
                              r"avail|show|use|restore|save|reset)\b", line):
            names += [t for t in shlex.split(m.group(1)) if not t.startswith("-")]
    return names


def module_bin_dirs(name):
    if name not in _module_cache:
        out = sh(["bash", "-lc", f"module --redirect show {shlex.quote(name)} 2>&1"], 60)
        _module_cache[name] = re.findall(
            r'(?:prepend|append)_path\(\s*"PATH"\s*,\s*"([^"]+)"', out)
    return _module_cache[name]


def script_vars(script):
    """Literal VAR=value assignments in the script (for $VAR expansion)."""
    env = {}
    for line in scriptjobs.logical_lines(script):
        m = re.match(r"(?:export\s+)?([A-Za-z_]\w*)=(\S+)\s*$", line)
        if m and "$(" not in m.group(2) and "`" not in m.group(2):
            env[m.group(1)] = m.group(2).strip("\"'")
    return env


def expand(token, env):
    for _ in range(3):
        new = re.sub(r"\$\{(\w+)\}|\$(\w+)",
                     lambda m: env.get(m.group(1) or m.group(2), m.group(0)), token)
        if new == token:
            break
        token = new
    return token


def executables(script):
    """Command words of non-interpreter commands in the script."""
    _, other = classify_script(script, ["python", "R", "julia"])
    exes = []
    for seg in other:
        try:
            toks = shlex.split(seg)
        except ValueError:
            toks = seg.split()
        toks = command_word(toks)
        if (toks and re.match(r"[A-Za-z_./~$]", toks[0])
                and toks[0].rsplit("/", 1)[-1] not in SHELLS
                and toks[0] not in exes):
            exes.append(toks[0])
    return exes


def locate(token, script, workdir, extra_dirs):
    """Return (path, reason). path is None if not found."""
    env = script_vars(script)
    tok = os.path.expanduser(expand(token, env))
    if "$" in tok:
        return None, "unresolved variable"
    if "/" in tok:
        cands = [tok if tok.startswith("/") else os.path.join(workdir or ".", tok)]
    else:
        dirs = []
        for name in module_names(script):
            dirs += module_bin_dirs(name)
        dirs += extra_dirs
        cands = [os.path.join(d, tok) for d in dirs]
        w = shutil.which(tok)
        if w:
            cands.append(w)
    for c in cands:
        if os.path.isfile(c):
            if os.access(c, os.R_OK):
                return os.path.realpath(c), ""
            return None, "not readable"
    return None, "not found (module not in script, or in a container?)"


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def short(s, n):
    return s if len(s) <= n else "..." + s[-(n - 3):]


def print_report(r, verbose=True):
    print(f"{r.path}")
    if r.error:
        print(f"    {r.error}")
        return
    print(f"    verdict : {r.verdict}")
    if r.evidence:
        print(f"    evidence: {r.evidence}")
    print(f"    arch    : {r.arch}   compilers: {'; '.join(r.compilers) or '-'}")
    print(f"    debug   : {'yes' if r.debug_info else 'no'}"
          f"   symbols: {'stripped' if r.stripped else 'present'}")
    for p, lvl in list(r.producers.items())[:6]:
        print(f"    producer: [{lvl or 'flags not recorded'}] {p[:110]}")
    if r.rbp_fraction is not None:
        print(f"    disasm  : {r.insns} insns sampled in {len(r.regions)} region(s), "
              f"{r.rbp_fraction:.1%} rbp-relative, AVX {r.avx}, AVX-512 {r.avx512}, "
              f"FMA {r.fma}")
        if len(r.regions) > 1:
            print("    regions : " + " ".join(
                "O" if g[1] >= O0_RBP_FRACTION else "." for g in r.regions)
                + "   (O = looks -O0, . = looks optimised; low to high address)")
    if r.o0_share is not None:
        print(f"    deep    : {len(r.o0_funcs)} function(s) look -O0, "
              f"{r.o0_share:.0%} of all instructions")
        for name, n, frac in r.o0_funcs[:12]:
            print(f"              {n:>8} insns {frac:>4.0%} rbp  {name[:90]}")
    for mk in r.markers:
        print(f"    marker  : {mk}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("exes", nargs="*", help="executables to inspect directly")
    ap.add_argument("-S", "--start")
    ap.add_argument("-E", "--end")
    ap.add_argument("-C", "--min-core-hours", type=float)
    ap.add_argument("-u", "--user")
    ap.add_argument("-A", "--account")
    ap.add_argument("--chunk-days", type=int, default=3)
    ap.add_argument("--path", action="append", default=[], metavar="DIR",
                    help="extra directory to search for bare executable names")
    ap.add_argument("--no-disasm", action="store_true",
                    help="skip the disassembly heuristics (faster)")
    ap.add_argument("--deep", action="store_true",
                    help="direct mode: scan every function and list those that look "
                         "-O0 (slow on big binaries, ~1 min for 100 MB)")
    ap.add_argument("-n", "--top", type=int, default=25)
    ap.add_argument("--only-suspect", action="store_true",
                    help="job mode: list only -O0 / suspect builds")
    args = ap.parse_args(argv)

    disasm = not args.no_disasm
    if args.exes:
        for e in args.exes:
            print_report(inspect(e, disasm, args.deep))
            print()
        return
    if not (args.start and args.end and args.min_core_hours is not None):
        ap.error("give executables, or -S, -E and -C to scan jobs")

    jobs = fetch_jobs(args.start, args.end, args.min_core_hours, args.user,
                      args.account, args.chunk_days)
    print(f"{len(jobs)} job(s) over {args.min_core_hours:g} core-hours; "
          f"reading scripts...", file=sys.stderr)
    scripts = fetch_scripts(sorted(jobs))

    usage = defaultdict(lambda: {"ch": 0.0, "jobs": 0, "users": set(), "names": set()})
    missing = defaultdict(lambda: {"ch": 0.0, "jobs": 0, "why": ""})
    for jid, script in scripts.items():
        job = jobs[jid]
        for tok in executables(script):
            path, why = locate(tok, script, job.workdir, args.path)
            if path and path.startswith(SYSTEM_DIRS):
                continue  # system utilities (which, scancel, ...), not science codes
            if path:
                u = usage[path]
                u["ch"] += job.core_hours
                u["jobs"] += 1
                u["users"].add(job.user)
            else:
                m = missing[tok]
                m["ch"] += job.core_hours
                m["jobs"] += 1
                m["why"] = why
    print(f"{len(usage)} distinct executable(s) located; inspecting...",
          file=sys.stderr)

    rows = []
    for path, u in sorted(usage.items(), key=lambda kv: -kv[1]["ch"]):
        r = inspect(path, disasm)
        if r.error:  # script, not a binary
            continue
        rows.append((u, r))

    print(f"{'core-h':>9}{'jobs':>6}  {'level':<10}{'dbg':<5}  executable / verdict")
    shown = 0
    for u, r in rows:
        if args.only_suspect and r.level not in ("O0", "suspect"):
            continue
        if shown >= args.top:
            break
        shown += 1
        print(f"{u['ch']:>9.0f}{u['jobs']:>6}  {r.level:<10}"
              f"{'yes' if r.debug_info else 'no':<5}  {short(r.path, 70)}")
        extra = notes(r)
        print(f"{'':<24}{r.verdict}" + (f"  [{'; '.join(extra)}]" if extra else ""))
    if missing:
        print("\nnot located (core-hours of jobs that ran them):")
        for tok, m in sorted(missing.items(), key=lambda kv: -kv[1]["ch"])[:10]:
            print(f"{m['ch']:>9.0f}{m['jobs']:>6}  {short(tok, 50):<50} {m['why']}")


if __name__ == "__main__":
    main()
