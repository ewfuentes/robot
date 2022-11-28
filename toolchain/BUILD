
load(":clang_toolchain_config.bzl", "clang_toolchain_config")
load(":gcc_toolchain_config.bzl", "gcc_toolchain_config")

package(default_visibility = ["//visibility:public"])

cc_toolchain_suite(
  name = "cpp_suite",
  toolchains = {
    "k8|clang": ":k8_clang_toolchain",
    "k8|gcc": ":k8_gcc_toolchain",
  }
)

filegroup(name="empty")

clang_toolchain_config(
  name="k8_clang_toolchain_config",
)

gcc_toolchain_config(
  name="k8_gcc_toolchain_config",
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

cc_toolchain(
  name="k8_gcc_toolchain",
  all_files = ":empty",
  compiler_files = ":empty",
  strip_files = ":empty",
  objcopy_files = ":empty",
  linker_files = ":empty",
  dwp_files = ":empty",
  toolchain_config = ":k8_gcc_toolchain_config",
  supports_param_files=0,
)