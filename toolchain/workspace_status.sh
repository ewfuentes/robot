#!/bin/bash
# toolchain/workspace_status.sh

echo "STABLE_GIT_COMMIT $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "STABLE_GIT_DIFF $(git diff --no-color 2>/dev/null | base64 -w 0 || echo '')"