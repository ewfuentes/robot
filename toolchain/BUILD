
load(":clang_toolchain_config.bzl", "clang_toolchain_config")
load(":gcc_toolchain_config.bzl", "gcc_toolchain_config")

package(default_visibility = ["//visibility:public"])

filegroup(name="empty")

clang_toolchain_config(
  name="k8_clang_toolchain_config",
)

gcc_toolchain_config(
  name="k8_gcc_11_toolchain_config",
  gcc_version="11",
)

gcc_toolchain_config(
  name="k8_gcc_10_toolchain_config",
  gcc_version="10",
)

cc_toolchain(
  name="k8_clang_toolchain",
  all_files = ":empty",
  compiler_files = ":empty",
  strip_files = ":empty",
  objcopy_files = ":empty",
  linker_files = ":empty",
  dwp_files = ":empty",
  toolchain_config = ":k8_clang_toolchain_config",
  supports_param_files=0,
)

toolchain(
  name = "clang_toolchain_for_linux_x84_64",
  toolchain = ":k8_clang_toolchain",
  toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
  exec_compatible_with = [
    "@platforms//cpu:x86_64",
    "@platforms//os:linux",
  ],
  target_compatible_with = [
    "@platforms//cpu:x86_64",
    "@platforms//os:linux",
    ":clang",
  ],
)

cc_toolchain(
  name="k8_gcc_10_toolchain",
  all_files = ":empty",
  compiler_files = ":empty",
  strip_files = ":empty",
  objcopy_files = ":empty",
  linker_files = ":empty",
  dwp_files = ":empty",
  toolchain_config = ":k8_gcc_10_toolchain_config",
  supports_param_files=0,
)

toolchain(
  name = "gcc_10_toolchain_for_linux_x84_64",
  toolchain = ":k8_gcc_10_toolchain",
  toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
  exec_compatible_with = [
    "@platforms//cpu:x86_64",
    "@platforms//os:linux",
  ],
  target_compatible_with = [
    "@platforms//cpu:x86_64",
    "@platforms//os:linux",
    ":gcc10",
  ],
)

cc_toolchain(
  name="k8_gcc_11_toolchain",
  all_files = ":empty",
  compiler_files = ":empty",
  strip_files = ":empty",
  objcopy_files = ":empty",
  linker_files = ":empty",
  dwp_files = ":empty",
  toolchain_config = ":k8_gcc_11_toolchain_config",
  supports_param_files=0,
)

toolchain(
  name = "gcc_11_toolchain_for_linux_x84_64",
  toolchain = ":k8_gcc_11_toolchain",
  toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
  exec_compatible_with = [
    "@platforms//cpu:x86_64",
    "@platforms//os:linux",
  ],
  target_compatible_with = [
    "@platforms//cpu:x86_64",
    "@platforms//os:linux",
    ":gcc11",
  ],
)

constraint_setting(name = "compiler")
constraint_value(
  name="clang",
  constraint_setting=":compiler"
)

constraint_value(
  name="gcc10",
  constraint_setting=":compiler"
)

constraint_value(
  name="gcc11",
  constraint_setting=":compiler"
)

platform(
  name = "clang_x86_64",
  constraint_values = [
    ":clang",
    "@platforms//cpu:x86_64",
    "@platforms//os:linux",
  ]
)

platform(
  name = "gcc10_x86_64",
  constraint_values = [
    ":gcc10",
    "@platforms//cpu:x86_64",
    "@platforms//os:linux",
  ]
)

platform(
  name = "gcc11_x86_64",
  constraint_values = [
    ":gcc11",
    "@platforms//cpu:x86_64",
    "@platforms//os:linux",
  ]
)
