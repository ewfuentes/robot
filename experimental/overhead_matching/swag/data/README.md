
# How To Get Data

## VIGOR

Follow the instructions [here](https://github.com/Jeff-Zilence/VIGOR/blob/main/data/DATASET.md)

## AlphaEarth

Sign up for Google Earth Engine and use this [snippet](https://code.earthengine.google.com/906473d86339524e701237f2d64381fd) to create the extraction tasks.

## Landmarks

Run the `experimental/overhead_matching/swag/scripts:extract_landmarks_for_vigor_dataset` target and place the generated landmark table in a `landmarks/` subdirectory of the dataset (next to the satellite and panorama folders), named after the landmark version: `<dataset>/landmarks/<landmark_version>.feather` (the loader falls back to `<landmark_version>.geojson`). The version string is what `--landmark_version` selects at load time.
