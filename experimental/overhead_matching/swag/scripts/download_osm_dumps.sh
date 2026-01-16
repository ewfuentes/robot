#!/bin/bash

# Download OpenStreetMap dumps for all US states from Geofabrik

BASE_URL="https://download.geofabrik.de/north-america/us"
OUTPUT_DIR="/data/overhead_matching/datasets/osm_dumps"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# List of US states
STATES=(
    "Alabama"
    "Alaska"
    "Arizona"
    "Arkansas"
    "California"
    "Colorado"
    "Connecticut"
    "Delaware"
    "Florida"
    "Georgia"
    "Hawaii"
    "Idaho"
    "Illinois"
    "Indiana"
    "Iowa"
    "Kansas"
    "Kentucky"
    "Louisiana"
    "Maine"
    "Maryland"
    "Massachusetts"
    "Michigan"
    "Minnesota"
    "Mississippi"
    "Missouri"
    "Montana"
    "Nebraska"
    "Nevada"
    "New Hampshire"
    "New Jersey"
    "New Mexico"
    "New York"
    "North Carolina"
    "North Dakota"
    "Ohio"
    "Oklahoma"
    "Oregon"
    "Pennsylvania"
    "Rhode Island"
    "South Carolina"
    "South Dakota"
    "Tennessee"
    "Texas"
    "Utah"
    "Vermont"
    "Virginia"
    "Washington"
    "West Virginia"
    "Wisconsin"
    "Wyoming"
)

for state in "${STATES[@]}"; do
    # Convert to lowercase and replace spaces with hyphens
    state_slug=$(echo "$state" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')

    filename="${state_slug}-200101.osm.pbf"
    url="${BASE_URL}/${filename}"
    output_path="${OUTPUT_DIR}/${filename}"

    if [[ -f "$output_path" ]]; then
        echo "Skipping $state (already exists)"
        continue
    fi

    echo "Downloading $state..."
    if wget -q --show-progress -O "$output_path" "$url"; then
        echo "Downloaded $state successfully"
    else
        echo "Failed to download $state"
        rm -f "$output_path"  # Remove partial download
    fi
done

echo "Done!"
