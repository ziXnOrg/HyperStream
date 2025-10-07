# Contributing to HyperStream

Thank you for your interest in contributing! HyperStream is a high‑performance, header‑only C++17 library for Hyperdimensional Computing. We welcome contributions that improve correctness, performance, documentation, and developer experience.

- Build system: CMake 3.16+
- Compilers: MSVC 2022, Clang 12+, GCC 10+
- Warnings-as-errors are enabled for tests; keep builds clean.
- Please run the full test suite before submitting PRs: `ctest -C Release --output-on-failure`.

## Development workflow

1. Fork the repository and create a feature branch.
2. Make changes with small, focused commits.
3. Write or update tests to cover your changes.
4. Run the full test suite on Windows/Linux/macOS if possible.
5. Open a Pull Request with a clear description and motivation.

## Coding standards

- Prefer simple, clear, and defensive code for safety-critical paths.
- Follow Google C++ style where practical; keep lines ≤ 100 characters.
- Use RAII and avoid raw new/delete.
- Add brief Doxygen-style comments to public APIs and non-trivial functions.

## Runtime Dispatch and SIMD Backends

HyperStream uses runtime dispatch to select SIMD backends (SSE2/AVX2) at runtime and avoid illegal instructions on unsupported CPUs. If you are adding a new SIMD backend or modifying dispatch logic, please consult the detailed guide:

- See Runtime Dispatch Architecture: Docs/Runtime_Dispatch.md

That document explains:
- GCC/Clang function-level target attributes (no global -m flags)
- MSVC separate translation units compiled with per-file /arch flags
- The raw kernel API (BindWords/HammingWords) and thin templated wrappers
- Transitive linkage patterns and safety invariants (unaligned load/store)

## Testing and CI

- Write unit tests for all new functionality. Prefer deterministic patterns and fixed seeds.
- Keep tests fast; avoid long-running microbenchmarks in the default suite.
- CI builds and runs tests across Windows, Linux, and macOS.

## Reporting issues

Please include:
- Environment (OS, compiler, commit hash)
- Reproduction steps and minimal code example
- Expected vs actual behavior and any logs

