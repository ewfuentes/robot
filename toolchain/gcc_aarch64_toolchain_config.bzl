
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl", "feature", "flag_group", "flag_set", "tool_path")

def _impl(ctx):
    tool_paths = [
      tool_path(
        name = "gcc",
        path = "aarch64-linux-gcc.sh",
      ),
      tool_path(
        name = "ar",
        path = "aarch64-linux-ar.sh",
      ),
      tool_path(
        name = "ld",
        path = "aarch64-linux-ld.sh",
      ),
      tool_path(
        name = "cpp",
        path = "aarch64-linux-gcc.sh",
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
        name = "toolchain_include_directories",
        enabled = True,
        flag_sets = [
          flag_set(
            actions = all_compile_actions,
            flag_groups = [
              flag_group(
                flags=[
                  "--sysroot", 
                  "external/aarch64-none-linux-gnu/aarch64-buildroot-linux-gnu/sysroot",
                  "-isystem",
                  "external/aarch64-none-linux-gnu/aarch64-buildroot-linux-gnu/include",
                  "-isystem",
                  "external/aarch64-none-linux-gnu/aarch64-buildroot-linux-gnu/include/c++/10.2.0",
                  "-isystem",
                  "external/aarch64-none-linux-gnu/aarch64-buildroot-linux-gnu/include/c++/10.2.0/aarch64-buildroot-linux-gnu",
                  "-isystem",
                  "external/aarch64-none-linux-gnu/lib/gcc/aarch64-buildroot-linux-gnu/10.2.0/include",
                  "-isystem",
                  "external/aarch64-none-linux-gnu/lib/gcc/aarch64-buildroot-linux-gnu/10.2.0/include-fixed",
                ]
              )
            ]
          )
        ]
      ),
      feature(
        name = "default_linker_flags",
        enabled=True,
        flag_sets = [
          flag_set(
            actions = all_link_actions,
            flag_groups = [
              flag_group(
                flags=[
                  "-lstdc++",
                  "-lm", 
                ],
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
                  flags=["-std=c++2a", "-fPIC", "-no-canonical-prefixes", "-fno-canonical-system-headers"],
                )
              ]
            )
          ]
      ),
      feature(
          name="c_compile_flags",
          enabled=True,
          flag_sets = [
            flag_set(
              actions = [ACTION_NAMES.c_compile],
              flag_groups = [
                flag_group(
                  flags=["-fPIC", "-no-canonical-prefixes", "-fno-canonical-system-headers"],
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
      ],
      toolchain_identifier="aarch64-gcc-toolchain",
      host_system_name="local",
      target_system_name="local",
      target_cpu="aarch64",
      target_libc="unknown",
      compiler="gcc",
      abi_version="unknown",
      abi_libc_version="unknown",
      tool_paths = tool_paths
    )

gcc_aarch64_toolchain_config = rule(
    implementation = _impl,
    attrs = {
    },
    provides = [CcToolchainConfigInfo]
)
