
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl", "feature", "flag_group", "flag_set", "tool_path")

def _impl(ctx):
    tool_paths = [
      tool_path(
        name = "gcc",
        path = "/usr/bin/clang-15",
      ),
      tool_path(
        name = "ar",
        path = "/usr/bin/ar",
      ),
      tool_path(
        name = "ld",
        path = "/usr/bin/ld.lld-15",
      ),
      tool_path(
        name = "cpp",
        path = "/usr/bin/clang++-15",
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
                flags=["-lstdc++", "-lm"],
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
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-Wno-gcc-compat",
                    "-Wno-c++98-compat",
                    "-Wno-c++98-compat-pedantic",
                    "-Wconversion-null",
                    "-Woverlength-strings",
                    "-Wpointer-arith",
                    "-Wunused-local-typedefs",
                    "-Wunused-result",
                    "-Wvarargs",
                    "-Wvla",
                    "-Wwrite-strings",
                    "-Wno-incompatible-pointer-types-discards-qualifiers",
                    "-Wno-deprecated-builtins",
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
                  flags=["-stdlib=libstdc++", "-std=c++20", "-fno-omit-frame-pointer", "-g", "-fPIC", "-gdwarf-4"],
                )
              ]
            )
          ]
      ),
      feature(
          name="valgrind_info",
          enabled=False,
          flag_sets = [
            flag_set(
              actions = [ACTION_NAMES.cpp_compile],
              flag_groups = [
                flag_group(
                  flags=["-gdwarf-4"],
                )
              ]
            )
          ]
      ),
      feature(
          name="ubsan",
          enabled=False,
          flag_sets = [
            flag_set(
              actions = all_compile_actions + all_link_actions,
              flag_groups = [
                flag_group(
                  flags=["-fsanitize=undefined"],
                )
              ]
            ),
            flag_set(
              actions = all_link_actions,
              flag_groups = [
                flag_group(
                  flags=["-fuse-ld=lld", "-lubsan"],
                )
              ]
            )
          ]
      ),
      feature(
          name="asan",
          enabled=False,
          flag_sets = [
            flag_set(
              actions = all_compile_actions + all_link_actions,
              flag_groups = [
                flag_group(
                  flags=["-fsanitize=address"],
                )
              ]
            ),
            flag_set(
              actions = all_link_actions,
              flag_groups = [
                flag_group(
                  flags=["-fuse-ld=lld"],
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
        "/usr/lib/llvm-15/lib/clang/15.0.6/include",
        "/usr/lib/llvm-15/lib/clang/15.0.7/include",
        "/usr/include",
        "/usr/include/c++/12",
        "/usr/include/x86_64-linux-gnu/c++/12",
        "/usr/lib/llvm-15/lib/clang/15.0.6/share",
        "/usr/lib/llvm-15/lib/clang/15.0.7/share",
        "/usr/lib/llvm-15/include/c++/v1/",
      ],
      toolchain_identifier="k8-clang-toolchain",
      host_system_name="local",
      target_system_name="local",
      target_cpu="k8",
      target_libc="unknown",
      compiler="clang",
      abi_version="unknown",
      abi_libc_version="unknown",
      tool_paths = tool_paths
    )

clang_toolchain_config = rule(
    implementation = _impl,
    attrs = {},
    provides = [CcToolchainConfigInfo]
)
