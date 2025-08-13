#! /usr/bin/env bash

set -euo pipefail

GITHUB_RUNNER_TOKEN=$(curl -s -X POST \
            -H "Authorization: token ${GITHUB_ACCESS_TOKEN}" \
            -H "Accept: application/vnd.github+json" \
            https://api.github.com/repos/ewfuentes/robot/actions/runners/registration-token \
            | jq -r .token)

./config.sh remove --token ${GITHUB_RUNNER_TOKEN} || true
./config.sh --url https://github.com/ewfuentes/robot --token ${GITHUB_RUNNER_TOKEN} --unattended
./run.sh
