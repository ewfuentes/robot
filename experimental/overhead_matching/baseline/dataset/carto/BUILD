load("@pip//:requirements.bzl", "requirement")

# render_carto.py runs inside the docker container (python-mapnik); run_carto_render.sh and
# import_region.sh are run from the source tree. See README.md.

py_binary(
    name = "make_carto_jobs",
    srcs = ["make_carto_jobs.py"],
    deps = [
        "//common/gps:web_mercator",
        "//experimental/overhead_matching/baseline/dataset:city_pbf_map",
        "//experimental/overhead_matching/baseline/dataset:tile_geometry",
    ],
)

py_binary(
    name = "check_carto",
    srcs = ["check_carto.py"],
    deps = [requirement("Pillow")],
)
