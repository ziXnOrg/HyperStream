import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Tuple


def load_compile_commands(build_dir: str) -> List[dict]:
    path = os.path.join(build_dir, "compile_commands.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_tus(entries: Iterable[dict]) -> Iterable[str]:
    seen: set[str] = set()
    for e in entries:
        f = e.get("file")
        if not isinstance(f, str):
            continue
        if not (f.endswith(".cc") or f.endswith(".cpp")):
            continue
        n = os.path.normpath(f)
        if n in seen:
            continue
        seen.add(n)
        yield n


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--build-dir", required=True)
    p.add_argument("--output", default="clang_tidy_tus.txt")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2), help="Parallel jobs")
    args = p.parse_args()

    try:
        entries = load_compile_commands(args.build_dir)
    except FileNotFoundError:
        print(f"compile_commands.json not found in {args.build_dir}", file=sys.stderr)
        return 2

    msvc_include = os.environ.get("INCLUDE", "")
    extra_args: List[str] = [
        "--extra-arg-before=--driver-mode=cl",
        "--extra-arg=-Wno-unknown-warning-option",
        "--warnings-as-errors=*",
        "--system-headers",
    ]
    for inc in [p for p in msvc_include.split(";") if p]:
        extra_args.append(f"--extra-arg=-isystem{inc}")

    # Collected diagnostics
    naming_lines: List[str] = []
    naming_counts: Dict[str, int] = {}

    cppcore_cert_lines: List[str] = []
    cppcore_cert_counts: Dict[str, int] = {}

    check_re = re.compile(r"\[([^\]]+)\]$")

    # Filters
    name_checks_prefixes = (
        "readability-identifier-",
        "readability-magic-numbers",
    )
    cppcore_cert_prefixes = (
        "cppcoreguidelines-",
        "cert-",
    )

    # Minimal fix suggestions by check prefix (best-effort)
    fix_suggestions: Dict[str, str] = {
        "cppcoreguidelines-narrowing-conversions": "Avoid implicit narrowing casts; use explicit casts with checks or wider types.",
        "cppcoreguidelines-avoid-magic-numbers": "Replace literals with named constants or enums.",
        "cppcoreguidelines-owning-memory": "Prefer smart pointers/RAII; avoid raw owning pointers.",
        "cppcoreguidelines-pro-type-reinterpret-cast": "Avoid reinterpret_cast; use safer alternatives or refactor API.",
        "cppcoreguidelines-pro-type-const-cast": "Avoid const_cast; redesign interfaces to respect const-correctness.",
        "cppcoreguidelines-pro-bounds-pointer-arithmetic": "Avoid pointer arithmetic; use span/array indexing.",
        "cppcoreguidelines-pro-type-vararg": "Avoid C varargs; use variadic templates or overloads.",
        "cert-err34-c": "Check return values from library calls and handle errors explicitly.",
        "cert-dcl16-c": "Do not declare identifiers in the global namespace if avoidable; limit scope.",
        "cert-env33-c": "Validate environment-variable usage and bounds before conversion.",
        "cert-flp30-c": "Avoid floating-point equality; compare within tolerances.",
    }

    def run_one(tu_path: str) -> Tuple[str, List[str]]:
        cmd = ["clang-tidy", tu_path, "-p", args.build_dir, "--use-color", *extra_args]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return tu_path, proc.stdout.splitlines()

    tus = list(iter_tus(entries))
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futures = [ex.submit(run_one, tu) for tu in tus]
        for fut in as_completed(futures):
            _, lines = fut.result()
            for line in lines:
                # Naming/magic number collection
                if any(pref in line for pref in name_checks_prefixes):
                    m = check_re.search(line)
                    if m:
                        naming_lines.append(line)
                        naming_counts[m.group(1)] = naming_counts.get(m.group(1), 0) + 1
                # cppcoreguidelines / cert collection
                if any(pref in line for pref in cppcore_cert_prefixes):
                    m = check_re.search(line)
                    if m:
                        cppcore_cert_lines.append(line)
                        cppcore_cert_counts[m.group(1)] = cppcore_cert_counts.get(m.group(1), 0) + 1

    # Write naming diagnostics (legacy output)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(naming_lines))

    summary_path = os.path.splitext(args.output)[0] + "_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        for k, v in sorted(naming_counts.items(), key=lambda kv: kv[1], reverse=True):
            f.write(f"{k}: {v}\n")

    # Write cppcoreguidelines/cert diagnostics and summary with suggestions
    cppcore_out = os.path.splitext(args.output)[0] + "_cppcore_cert.txt"
    with open(cppcore_out, "w", encoding="utf-8") as f:
        f.write("\n".join(cppcore_cert_lines))

    cppcore_summary = os.path.splitext(args.output)[0] + "_cppcore_cert_summary.txt"
    with open(cppcore_summary, "w", encoding="utf-8") as f:
        for k, v in sorted(cppcore_cert_counts.items(), key=lambda kv: kv[1], reverse=True):
            # Find best suggestion for this check by prefix
            suggestion = "Review guideline and refactor accordingly."
            for pref, hint in fix_suggestions.items():
                if k.startswith(pref):
                    suggestion = hint
                    break
            f.write(f"{k}: {v} | Suggestion: {suggestion}\n")

    print(f"Wrote TU tidy naming diagnostics to {args.output} ({len(naming_lines)} lines)")
    print(f"Wrote cppcore/cert diagnostics to {cppcore_out} ({len(cppcore_cert_lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
