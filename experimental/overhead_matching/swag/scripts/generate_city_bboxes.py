"""Generate YAML file of US cities with bounding boxes.

Reads the simplemaps US cities CSV and generates a YAML file containing
state, city, and bounding box for cities above a population threshold.

Data source: https://simplemaps.com/data/us-cities
"""

import argparse
import csv
from pathlib import Path

import yaml


def generate_bbox(lat: float, lng: float, radius_km: float = 10.0) -> dict:
    """Generate a bounding box around a lat/lng point.

    Args:
        lat: Latitude in degrees
        lng: Longitude in degrees
        radius_km: Radius in kilometers (default 10km)

    Returns:
        Dictionary with min_lat, max_lat, min_lng, max_lng
    """
    # Approximate degrees per km
    # 1 degree latitude ≈ 111 km
    # 1 degree longitude ≈ 111 * cos(lat) km
    import math

    lat_delta = radius_km / 111.0
    lng_delta = radius_km / (111.0 * math.cos(math.radians(lat)))

    return {
        "min_lat": round(lat - lat_delta, 6),
        "max_lat": round(lat + lat_delta, 6),
        "min_lng": round(lng - lng_delta, 6),
        "max_lng": round(lng + lng_delta, 6),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate YAML file of US cities with bounding boxes"
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path("/data/overhead_matching/datasets/simplemaps/uscities.csv"),
        help="Path to simplemaps uscities.csv",
    )
    parser.add_argument(
        "--output_yaml",
        type=Path,
        default=Path("/data/overhead_matching/datasets/us_city_bboxes.yaml"),
        help="Output YAML file path",
    )
    parser.add_argument(
        "--min_population",
        type=int,
        default=100000,
        help="Minimum population threshold (default: 100000)",
    )
    parser.add_argument(
        "--radius_km",
        type=float,
        default=10.0,
        help="Bounding box radius in km (default: 10.0)",
    )
    args = parser.parse_args()

    cities = []

    with open(args.input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows with missing population
            if not row["population"]:
                continue

            population = int(float(row["population"]))
            if population < args.min_population:
                continue

            lat = float(row["lat"])
            lng = float(row["lng"])
            bbox = generate_bbox(lat, lng, args.radius_km)

            cities.append(
                {
                    "state": row["state_name"],
                    "state_id": row["state_id"],
                    "city": row["city"],
                    "population": population,
                    "bbox": bbox,
                }
            )

    # Sort by state, then by population descending
    cities.sort(key=lambda x: (x["state"], -x["population"]))

    output_data = {
        "metadata": {
            "source": "https://simplemaps.com/data/us-cities",
            "min_population": args.min_population,
            "radius_km": args.radius_km,
            "num_cities": len(cities),
        },
        "cities": cities,
    }

    with open(args.output_yaml, "w", encoding="utf-8") as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    print(f"Generated {args.output_yaml} with {len(cities)} cities")


if __name__ == "__main__":
    main()
