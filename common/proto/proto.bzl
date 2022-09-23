
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

UNDERLYING_TAG = "__"

def underlying_proto_target_from_label(label):
    head, tail = label.split(':')
    return head + ":" + UNDERLYING_TAG + tail

def multi_proto_library(name, **kwargs):
    srcs = kwargs["srcs"]
    deps = kwargs.get("deps", {})
    kwargs.pop("srcs")
    if deps:
      kwargs.pop("deps")

    underlying_deps = [underlying_proto_target_from_label(label) for label in deps]
    
    # Create the proto rule for the current srcs, adding the underlying deps
    underlying_proto_name = UNDERLYING_TAG + name

    proto_library(
      name = underlying_proto_name,
      srcs = srcs,
      deps = underlying_deps,
    )

    native.cc_proto_library(
      name = name,
      deps = [":" + underlying_proto_name],
      features=["-warning_compile_flags"],
      **kwargs,
    )

    py_proto_library(
      name = name + "_py",
      srcs = srcs,
      **kwargs,
    )