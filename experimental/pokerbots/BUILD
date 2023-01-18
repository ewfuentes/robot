load("@pip//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_runtime")
load("@rules_pkg//pkg:zip.bzl", "pkg_zip")

pkg_zip(
    name="zipped_bot",
    tags=["manual"],
    srcs=[":pokerbot_zip", "commands.json"],
)

filegroup(
    name="pokerbot_zip",
    tags=["manual"],
    srcs=[
        ":pokerbot",
    ],
    output_group="python_zip_file",
)

py_runtime(
    name="system_runtime",
    python_version="PY3",
    interpreter_path="/usr/bin/python3",
)

pybind_extension(
  name = "hand_evaluator_python",
  srcs = ["hand_evaluator_python.cc"],
  copts = ["-fPIC"],
  deps = [
  ":hand_evaluator",
  ]
)

py_test(
  name = "hand_evaluator_python_test",
  srcs = ["hand_evaluator_python_test.py"],
  data = [":hand_evaluator_python.so"],
)

py_binary(
    name="pokerbot",
    srcs=["pokerbot.py"],
    tags=["manual"],
    data = [
      ":hand_evaluator_python.so",
      "pokerbot_checkpoint_001270000.pb",
    ],
    deps=[
        requirement("numpy"),
        "@mit_pokerbots//:framework",
        "//learning:min_regret_strategy_proto_py",
    ],
)

py_test(
    name = "pokerbot_test",
    srcs = ["pokerbot_test.py"],
    deps = [
      ":pokerbot",
    ]
)

cc_binary(
  name = "pokerbot_master",
  srcs = ["pokerbot_master.cc"],
  deps = [
    ":generate_infoset_id",
    "//domain:rob_poker",
    "//learning:cfr",
    "//learning:min_regret_strategy_to_proto",
    "@cxxopts//:cxxopts",
  ]
)

py_binary(
    name = "compute_preflop_equity",
    srcs= ["compute_preflop_equity.py"],
    data = [":hand_evaluator_python.so"],
)

genrule(
    name = "preflop_equities",
    outs = ["preflop_equities.p"],
    tools = [":compute_preflop_equity"],
    cmd = "$(location :compute_preflop_equity) --output $(OUTS)",
    tags = ["manual"],
)

cc_library(
  name = "hand_evaluator",
  hdrs = ["hand_evaluator.hh"],
  srcs = ["hand_evaluator.cc"],
  deps = [
    "@ompeval//:ompeval_lib",
    "//domain:deck",
    "//domain:rob_poker",
    "//common/time:robot_time",
    ]
)

cc_library(
  name = "generate_infoset_id",
  hdrs = ["generate_infoset_id.hh"],
  srcs = ["generate_infoset_id.cc"],
  deps = [
    "//domain:deck",
    "//domain:rob_poker",
    ":hand_evaluator",
  ],
)