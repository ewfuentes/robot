
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl", "feature", "flag_group", "flag_set", "tool_path")

def _impl(ctx):
    gcc_version = ctx.attr.gcc_version
    tool_paths = [
      tool_path(
        name = "gcc",
        path = "/usr/bin/gcc-{}".format(gcc_version),
      ),
      tool_path(
        name = "ar",
        path = "/usr/bin/ar",
      ),
      tool_path(
        name = "ld",
        path = "/usr/bin/ld.gold",
      ),
      tool_path(
        name = "cpp",
        path = "/bin/false",
      ),
      tool_path(
        name = "gcov",
        path = "/bin/false",
      ),
      tool_path(
        name = "nm",
        path = "/bin/false",
      ),
      tool_path(
        name = "objdump",
        path = "/bin/false",
      ),
      tool_path(
        name = "strip",
        path = "/bin/false",
      ),
    ]
    all_link_actions = [
        ACTION_NAMES.cpp_link_executable,
        ACTION_NAMES.cpp_link_dynamic_library,
        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
    ]
    all_compile_actions = [
        ACTION_NAMES.c_compile,
        ACTION_NAMES.cpp_compile,
    ]
    features = [
      feature(
        name = "default_linker_flags",
        enabled=True,
        flag_sets = [
          flag_set(
            actions = all_link_actions,
            flag_groups = [
              flag_group(
                flags=["-lstdc++", "-lm", "-Wl,--disable-new-dtags"],
              ),
            ],
          ),
        ],
      ),
      feature(
          name="warning_compile_flags",
          enabled=True,
          flag_sets = [
            flag_set(
              actions = all_compile_actions,
              flag_groups = [
                flag_group(
                  flags=[
                    "-Wall", "-Wextra", "-Werror",
                    "-Woverlength-strings",
                    "-Wpointer-arith",
                    "-Wunused-local-typedefs",
                    "-Wunused-result",
                    "-Wvarargs",
                    "-Wvla",
                  ]
                )
              ]
            )
          ]
      ),
      feature(
          name="cpp_compile_flags",
          enabled=True,
          flag_sets = [
            flag_set(
              actions = [ACTION_NAMES.cpp_compile],
              flag_groups = [
                flag_group(
                  flags=["-std=c++20", "-fPIC"],
                )
              ]
            )
          ]
      ),
      feature(
          name="dbg",
          enabled=False,
          flag_sets = [
            flag_set(
              actions = all_compile_actions,
              flag_groups = [
                flag_group(
                  flags=["-ggdb"],
                )
              ]
            )
          ]
      ),
      feature(
          name="opt",
          enabled=False,
          flag_sets = [
            flag_set(
              actions = all_compile_actions,
              flag_groups = [
                flag_group(
                  flags=["-O2"],
                )
              ]
            )
          ]
      ),
      feature(
          name="fastbuild",
          enabled=False,
          flag_sets = [
            flag_set(
              actions = all_compile_actions,
              flag_groups = [
                flag_group(
                  flags=["-O1"],
                )
              ]
            )
          ]
      ),
    ]

    return cc_common.create_cc_toolchain_config_info(
      ctx=ctx,
      features = features,
      cxx_builtin_include_directories = [
        "/usr/include",
        "/usr/include/c++/{}".format(gcc_version),
        "/usr/lib/gcc/x86_64-linux-gnu/{}/include".format(gcc_version),
      ],
      toolchain_identifier="k8-gcc-toolchain",
      host_system_name="local",
      target_system_name="local",
      target_cpu="k8",
      target_libc="unknown",
      compiler="gcc",
      abi_version="unknown",
      abi_libc_version="unknown",
      tool_paths = tool_paths
    )

gcc_toolchain_config = rule(
    implementation = _impl,
    attrs = {
      "gcc_version": attr.string(),
    },
    provides = [CcToolchainConfigInfo]
)
