#!/usr/bin/env bash
# Bake .mbtiles for every VIGOR city, then render OSM tiles into satellite_osm/.
# Run from the repo root. Skips cities whose mbtiles or tiles already exist.

set -euo pipefail

CITIES=(Seattle SanFrancisco NewYork Chicago Boston)

# Use xvfb-run if available (needed on truly headless hosts where pymgl cannot
# obtain a GL context). On hosts with a working EGL/X setup pymgl renders
# directly and we save the wrapper overhead.
RENDER_PREFIX=()
if command -v xvfb-run > /dev/null; then
    RENDER_PREFIX=(xvfb-run -a)
fi

for city in "${CITIES[@]}"; do
    echo "=== baking $city ==="
    bazel run //experimental/overhead_matching/baseline/dataset:bake_mbtiles -- \
        --city "$city"

    echo "=== rendering $city ==="
    "${RENDER_PREFIX[@]}" bazel run //experimental/overhead_matching/baseline/dataset:render_osm_tiles -- \
        --city "$city"
done
