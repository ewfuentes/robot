
load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

exports_files([".marimo.toml"])

refresh_compile_commands(
    name = "refresh_compile_commands",

    # Specify the targets of interest.
    # For example, specify a dict of targets and any flags required to build.
    targets = { "//..." : "",  
      "//experimental/beacon_dist:render_ycb_scene_test": "",
    },
    # No need to add flags already in .bazelrc. They're automatically picked up.
    # If you don't need flags, a list of targets is also okay, as is a single target string.
    # Wildcard patterns, like //... for everything, *are* allowed here, just like a build.
      # As are additional targets (+) and subtractions (-), like in bazel query https://docs.bazel.build/versions/main/query.html#expressions
    # And if you're working on a header-only library, specify a test or binary target that compiles it.
)

load("@rules_gtsam//bzl:gtsam.bzl", "gtsam_config")
load("@rules_gtsam//bzl:gtsam.bzl", "gtsam_dllexport")

gtsam_config()
gtsam_dllexport(library_name="gtsam")
gtsam_dllexport(library_name="gtsam_unstable")
