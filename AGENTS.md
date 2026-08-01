# BARAM2026 Project Rules

- Work only from the single root session. Do not create subagents, worktrees, or background agent sessions.
- Treat `/Users/um-yunsang/Downloads/open.zip` and `/Users/um-yunsang/Downloads/baseline.ipynb` as immutable inputs. Verify their frozen SHA-256 values before every full run.
- Use only competition-supplied data. Do not add external data, pretrained weights, remote inference, reanalysis, or test-period observations.
- Use the project-local Python 3.12 environment and at most six model workers. Do not modify system packages.
- Keep operating year 2024 as a one-use lockbox. Freeze development decisions before reading its scores.
- Do not upload to Dacon, mutate a browser/account/team, push a remote repository, or deploy anything without separate explicit authority.
- Stage paths explicitly; never use `git add .`.
- Label PASS narrowly: a unit test, contract suite, lockbox decision, or reproduction receipt proves only its stated scope.
- Generated competition data, models, predictions, and CSV files stay untracked; reviewer-facing manifests and reports may be committed.
