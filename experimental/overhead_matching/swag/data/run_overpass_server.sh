#! /usr/bin/env bash

echo "Hello World!"

# Pull the image
docker pull wiktorn/overpass-api

mkdir -p ~/.cache/robot/overpass_db

# Run with a specific area (e.g., Monaco - small for testing)
docker run -d \
  --name overpass-api \
  -p 12345:80 \
  -e OVERPASS_META=yes \
  -e OVERPASS_MODE=init \
  -e OVERPASS_PLANET_URL=https://download.geofabrik.de/north-america/us-latest.osm.bz2 \
  -v ~/.cache/robot/overpass_db:/db \
  wiktorn/overpass-api
