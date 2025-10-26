import argparse
import os
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run clang-tidy on a single file using compile_commands.json")
    parser.add_argument("file", help="Path to .cpp/.cc/.c/.hpp/.h file")
    parser.add_argument("--build-dir", required=True, help="Build directory with compile_commands.json")
    parser.add_argument("--fix", action="store_true", help="Apply fixes")
    parser.add_argument("--extra", nargs="*", default=[], help="Extra args to pass after --")
    args = parser.parse_args()

    file_path = os.path.abspath(args.file)
    build_dir = os.path.abspath(args.build_dir)

    if not os.path.exists(os.path.join(build_dir, "compile_commands.json")):
        print(f"compile_commands.json not found in {build_dir}", file=sys.stderr)
        return 2

    cmd = [
        "clang-tidy",
        file_path,
        "-p",
        build_dir,
        "--use-color",
    ]
    if args.fix:
        cmd.append("--fix")

    # MSVC environment include paths when available
    msvc_include = os.environ.get("INCLUDE", "")
    # Ensure C++20 parsing so std::span/std::bit_cast/std::popcount are recognized
    extra_driver_args = [
        "--extra-arg-before=--driver-mode=cl",
        "--extra-arg=/std:c++20",
        # MSVC: ensure __cplusplus reflects the selected standard version when headers gate on it
        "--extra-arg=/Zc:__cplusplus",
    ]
    # Add project include root so headers can resolve internal includes
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    project_include = os.path.join(repo_root, "include")
    if os.path.isdir(project_include):
        extra_driver_args.append(f"--extra-arg=-I{project_include}")
    for inc in [p for p in msvc_include.split(";") if p]:
        extra_driver_args.append(f"--extra-arg=-isystem{inc}")

    # Append extras after --
    proc = subprocess.run(cmd + extra_driver_args + ["--"] + args.extra, text=True)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())


