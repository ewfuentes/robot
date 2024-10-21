
load(":clang_toolchain_config.bzl", "clang_toolchain_config")
load(":gcc_toolchain_config.bzl", "gcc_toolchain_config")
load(":gcc_aarch64_toolchain_config.bzl", "gcc_aarch64_toolchain_config")

package(default_visibility = ["//visibility:public"], features=["warning_compile_flags"])

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

gcc_aarch64_toolchain_config(
  name="aarch64_gcc_toolchain_config",
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

filegroup(
  name = "aarch64_wrapper_scripts",
  srcs = [
    ":aarch64-linux-ar.sh",
    ":aarch64-linux-gcc.sh",
    ":aarch64-linux-ld.sh",
    "@aarch64-none-linux-gnu//:all",
    "@jetson_sysroot"
  ],
)

cc_toolchain(
  name="aarch64_gcc_toolchain",
  all_files = ":aarch64_wrapper_scripts",
  ar_files = ":aarch64_wrapper_scripts",
  as_files = ":aarch64_wrapper_scripts",
  compiler_files = ":aarch64_wrapper_scripts",
  strip_files = ":aarch64_wrapper_scripts",
  objcopy_files = ":aarch64_wrapper_scripts",
  linker_files = ":aarch64_wrapper_scripts",
  dwp_files = ":empty",
  toolchain_config = ":aarch64_gcc_toolchain_config",
  supports_param_files=0,
)

toolchain(
  name = "gcc_toolchain_for_linux_aarch64",
  toolchain = ":aarch64_gcc_toolchain",
  toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
  exec_compatible_with = [
    "@platforms//cpu:x86_64",
    "@platforms//os:linux",
  ],
  target_compatible_with = [
    "@platforms//cpu:aarch64",
    "@platforms//os:linux",
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

platform(
  name = "gcc_aarch64",
  constraint_values = [
    "@platforms//cpu:aarch64",
    "@platforms//os:linux",
  ]
)

constraint_setting(name="ubuntu_version")
constraint_value(
  name = "noble_24.04",
  constraint_setting= ":ubuntu_version"
)

constraint_value(
  name = "jammy_22.04",
  constraint_setting= ":ubuntu_version"
)

platform(
  name="jammy_clang",
  constraint_values = [
    "@platforms//cpu:x86_64",
    "@platforms//os:linux",
    ":jammy_22.04",
    ":clang",
  ]
)

platform(
  name="jammy_gcc",
  constraint_values = [
    "@platforms//cpu:x86_64",
    "@platforms//os:linux",
    ":jammy_22.04",
    ":gcc11",
  ]
)


platform(
  name="noble_clang",
  constraint_values = [
    "@platforms//cpu:x86_64",
    "@platforms//os:linux",
    ":noble_24.04",
    ":clang",
  ]
)

platform(
  name="noble_gcc",
  constraint_values = [
    "@platforms//cpu:x86_64",
    "@platforms//os:linux",
    ":noble_24.04",
    ":gcc11",
  ]
)

