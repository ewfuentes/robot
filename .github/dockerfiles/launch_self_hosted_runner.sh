#! /usr/bin/env bash

set -euo pipefail

# Load .env file
if [ -f .env ]; then
    # Export variables from .env
    export $(grep -v '^#' .env | xargs)
else
    echo ".env file not found!"
    exit 1
fi

# Check if token is set
if [ -z "${GITHUB_ACCESS_TOKEN:-}" ]; then
    echo "GITHUB_ACCESS_TOKEN is not set in .env"
    exit 1
fi

GITHUB_RUNNER_TOKEN=$(curl -s -X POST \
            -H "Authorization: token ${GITHUB_ACCESS_TOKEN}" \
            -H "Accept: application/vnd.github+json" \
            https://api.github.com/repos/ewfuentes/robot/actions/runners/registration-token \
            | jq -r .token)

GITHUB_RUNNER_TOKEN=${GITHUB_RUNNER_TOKEN} docker compose up -d
