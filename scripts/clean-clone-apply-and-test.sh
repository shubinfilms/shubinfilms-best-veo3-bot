#!/usr/bin/env bash
set -euo pipefail
REPO_URL="${REPO_URL:-git@github.com:org/project.git}"
WORKDIR="${WORKDIR:-/tmp/project-clean}"
PATCH_FILE="${PATCH_FILE:-/tmp/change.patch}"

rm -rf "$WORKDIR" && mkdir -p "$WORKDIR" && cd "$WORKDIR"
git clone "$REPO_URL" .
git config core.autocrlf input
git config apply.whitespace nowarn

[ -f "$PATCH_FILE" ] && git am --3way "$PATCH_FILE" || true

if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
if [ -d tests ] || [ -f pytest.ini ]; then pytest -q; else echo "No tests"; fi
