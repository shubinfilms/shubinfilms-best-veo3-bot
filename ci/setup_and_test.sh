#!/usr/bin/env bash
set -euo pipefail

# Ensure we are operating from repo root
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Normalize git workspace
git config core.autocrlf input
git config apply.whitespace nowarn
# Avoid ownership warnings on GitHub-hosted runners
if command -v git >/dev/null 2>&1; then
  git config --global --add safe.directory "$(pwd)" || true
fi
git reset --hard
git clean -xfd

# Apply patches via git am if present
if compgen -G "patches/*.patch" > /dev/null; then
  git am --3way --whitespace=nowarn patches/*.patch
fi

# Setup Python environment
python -m pip install -U pip
pip install -r requirements.txt

# Run lint and tests
ruff check --select F821 handlers scripts tests
pytest -q
