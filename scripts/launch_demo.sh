#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

cd "${PROJECT_ROOT}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

echo "Running competition demo preflight check..."
if ! "${PYTHON_BIN}" scripts/preflight_check.py; then
    echo "ERROR: Demo preflight failed. No packages or data were downloaded." >&2
    exit 1
fi

echo "Preflight passed. Launching GUI..."
exec "${PYTHON_BIN}" gui/main.py
