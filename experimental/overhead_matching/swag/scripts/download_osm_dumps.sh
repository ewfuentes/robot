#!/bin/bash

# Download the dated Geofabrik OSM extracts the paper's landmark tables and
# rasterized-OSM tiles are derived from (one entry per benchmark region; the
# region -> city mapping lives in
# experimental/overhead_matching/baseline/dataset/city_pbf_map.py).
# Geofabrik archives dated extracts as <region>-<YYMMDD>.osm.pbf.

BASE_URL="https://download.geofabrik.de"
OUTPUT_DIR="/data/overhead_matching/datasets/osm_dumps"

mkdir -p "$OUTPUT_DIR"

DUMPS=(
    "north-america/us/illinois-200101.osm.pbf"           # Chicago (train)
    "north-america/us/washington-200101.osm.pbf"         # Seattle (calibration/val)
    "north-america/us/new-york-200101.osm.pbf"           # New York
    "north-america/us/massachusetts-260101.osm.pbf"      # Boston Snowy/Night, Framingham
    "north-america/us/connecticut-250101.osm.pbf"        # Middletown
    "north-america/us/california/norcal-220101.osm.pbf"  # San Francisco (Mapillary)
    "north-america/us/florida-220101.osm.pbf"            # Fort Myers (post_hurricane_ian_sw)
    "europe/netherlands-250101.osm.pbf"                  # Noordoostpolder, Veluwe
)

for rel in "${DUMPS[@]}"; do
    filename=$(basename "$rel")
    url="${BASE_URL}/${rel}"
    output_path="${OUTPUT_DIR}/${filename}"

    if [[ -f "$output_path" ]]; then
        echo "Skipping $filename (already exists)"
        continue
    fi

    echo "Downloading $filename..."
    if wget -q --show-progress -O "$output_path" "$url"; then
        echo "Downloaded $filename successfully"
    else
        echo "Failed to download $filename"
        rm -f "$output_path"  # Remove partial download
    fi
done

echo "Done!"
