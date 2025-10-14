---
type: "always_apply"
---

# PR & Commit Guidelines (HyperStream)

- Commits
  - Conventional Commits: `feat|fix|refactor|docs|test|chore|perf|ci(scope): message`.
  - Keep commits small and logically scoped; include rationale when non-obvious.

- Pull Requests
  - Description: what/why, benchmarks (if perf-sensitive), links to Issues/ADRs.
  - Checklist (must):
    - [ ] All tests green across OS matrix
    - [ ] Coverage ≥ 90% (including failure paths)
    - [ ] Perf regressions checked; attach NDJSON or summary
    - [ ] Lint clean (clang-tidy/format)
    - [ ] Docs updated (API/ADR/README) when applicable
    - [ ] ABI stability reviewed if public headers changed
  - Approvals: 2 for public API changes; 1 otherwise.