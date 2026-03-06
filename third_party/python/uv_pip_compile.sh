#!/usr/bin/env bash
set -euo pipefail

MODE="$1"           # "update" or "test"
PYTHON_VERSION="$2" # e.g. "3.12"
IN_FILE="$3"        # path to requirements_X_Y.in
OUT_FILE="$4"       # path to requirements_X_Y.txt

if ! command -v uv &> /dev/null; then
    echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

TMPFILE=$(mktemp)
trap "rm -f ${TMPFILE}" EXIT

uv pip compile \
    --python-version "${PYTHON_VERSION}" \
    -o "${TMPFILE}" \
    --no-header \
    --format requirements.txt \
    --index-strategy unsafe-best-match \
    --generate-hashes \
    --emit-index-url \
    "${IN_FILE}"

if [ "${MODE}" = "update" ]; then
    if [ -n "${BUILD_WORKSPACE_DIRECTORY:-}" ]; then
        cp "${TMPFILE}" "${BUILD_WORKSPACE_DIRECTORY}/${OUT_FILE}"
        echo "Updated ${BUILD_WORKSPACE_DIRECTORY}/${OUT_FILE}"
    else
        cp "${TMPFILE}" "${OUT_FILE}"
        echo "Updated ${OUT_FILE}"
    fi
elif [ "${MODE}" = "test" ]; then
    if ! diff -q "${TMPFILE}" "${OUT_FILE}" > /dev/null 2>&1; then
        echo "ERROR: ${OUT_FILE} is out of date. Run: bazel run //third_party/python:requirements_${PYTHON_VERSION//./_}.update"
        diff "${TMPFILE}" "${OUT_FILE}" || true
        exit 1
    fi
    echo "OK: ${OUT_FILE} is up to date."
fi
