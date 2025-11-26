#!/usr/bin/env bash
set -euo pipefail

# Ensure we are operating from repo root
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Normalize git workspace
git config core.autocrlf input
git config apply.whitespace nowarn
git reset --hard
git clean -xfd

# Optionally apply a patch from base64-encoded content
if [[ -n "${PATCH_B64:-}" ]]; then
  tmp_patch="$(mktemp)"
  echo "$PATCH_B64" | base64 -d > "$tmp_patch"
  if ! git am --3way "$tmp_patch"; then
    git am --abort || true
    echo "Patch application failed" >&2
    exit 1
  fi
  rm -f "$tmp_patch"
fi

# Setup Python environment
python -m pip install -U pip
pip install -r requirements.txt

# Run lint and tests
ruff check --select F821 handlers scripts tests
pytest -q
