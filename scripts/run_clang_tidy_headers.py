import argparse
import json
import os
import re
import subprocess
import sys
from typing import Iterable, List, Dict


def load_compile_commands(build_dir: str) -> List[dict]:
    compile_commands_path = os.path.join(build_dir, "compile_commands.json")
    with open(compile_commands_path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_translation_units(entries: Iterable[dict]) -> Iterable[str]:
    seen: set[str] = set()
    for entry in entries:
        path = entry.get("file")
        if not isinstance(path, str):
            continue
        if not (path.endswith(".cc") or path.endswith(".cpp") or path.endswith(".c")):
            continue
        norm = os.path.normpath(path)
        if norm in seen:
            continue
        seen.add(norm)
        yield norm


def main() -> int:
    parser = argparse.ArgumentParser(description="Run clang-tidy and capture header diagnostics only.")
    parser.add_argument("--build-dir", default="build_ninja_msvc", help="Directory containing compile_commands.json")
    parser.add_argument("--output", default="clang_tidy_headers.txt", help="Output log file for header diagnostics")
    args = parser.parse_args()

    try:
        entries = load_compile_commands(args.build_dir)
    except FileNotFoundError:
        print(f"compile_commands.json not found in {args.build_dir}", file=sys.stderr)
        return 2

    # Header filter: lines that reference files under include/ (both slash styles)
    # Only keep diagnostics referencing our project headers under include/hyperstream/
    project_header_substrings = ("include/hyperstream/", "include\\hyperstream\\")

    header_only_lines: List[str] = []
    # Add MSVC include directories from environment, if present
    extra_args: List[str] = ["--extra-arg-before=--driver-mode=cl"]
    msvc_include = os.environ.get("INCLUDE", "")
    for inc in [p for p in msvc_include.split(";") if p]:
        # Pass each INCLUDE path as -isystem to help clang find headers
        extra_args.append(f"--extra-arg=-isystem{inc}")
    tus = list(iter_translation_units(entries))
    if not tus:
        print("No translation units found in compilation database.", file=sys.stderr)
        return 3

    for tu in tus:
        # Run clang-tidy for each TU; rely on repository .clang-tidy HeaderFilterRegex
        try:
            cmd = ["clang-tidy", tu, "-p", args.build_dir, "--use-color", *extra_args]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            print("clang-tidy not found on PATH", file=sys.stderr)
            return 4

        # Keep only lines that clearly reference headers under include/
        for line in proc.stdout.splitlines():
            if any(s in line for s in project_header_substrings):
                header_only_lines.append(line)

    with open(args.output, "w", encoding="utf-8") as out:
        out.write("\n".join(header_only_lines))

    # Emit simple summary of check counts
    counts: Dict[str, int] = {}
    check_re = re.compile(r"\[([^\]]+)\]$")
    for line in header_only_lines:
        m = check_re.search(line)
        if not m:
            continue
        counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    summary_path = os.path.splitext(args.output)[0] + "_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        for check, n in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
            f.write(f"{check}: {n}\n")

    print(f"Wrote header diagnostics to {args.output} ({len(header_only_lines)} matching lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


