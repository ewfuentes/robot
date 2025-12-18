load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

def extra_torch_targets():
    cc_import(
        name = "libtorch.so",
        shared_library = "site-packages/torch/lib/libtorch.so",
    )

    cc_import(
        name = "libtorch_python.so",
        shared_library = "site-packages/torch/lib/libtorch_python.so",
    )

    cc_library(
        name = "libtorch",
        hdrs = native.glob(["site-packages/torch/include/**/*.h"]),
        includes = [
            "site-packages/torch/include/",
            "site-packages/torch/include/torch/csrc/api/include",
        ],
        deps = [
            ":libtorch.so",
            ":libtorch_python.so",
        ],
        visibility = ["//visibility:public"],
    )
