"""Makes `import rerun` resolve for the rerun-sdk wheel.

The wheel does not put its top-level package at the root of site-packages. It ships
`site-packages/rerun_sdk/rerun/` plus a `site-packages/rerun_sdk.pth` file holding the single
line `rerun_sdk`, relying on the interpreter's site machinery to append that subdirectory to
`sys.path` at startup.

Bazel's py rules build `sys.path` from `imports` attributes and never read `.pth` files, so with
only the wheel's own `:pkg` target the import fails outright -- `site-packages` is on the path
but `site-packages/rerun_sdk` is not, and `rerun` lives one level below what is visible.

`:rerun` adds the missing directory. Depend on it through
`extra_requirement("rerun-sdk", "rerun")` rather than `requirement("rerun-sdk")`, which resolves
to `:pkg` and does not carry the extra import path.
"""

load("@rules_python//python:defs.bzl", "py_library")

def extra_rerun_targets():
    py_library(
        name = "rerun",
        imports = ["site-packages/rerun_sdk"],
        deps = [":pkg"],
        visibility = ["//visibility:public"],
    )
