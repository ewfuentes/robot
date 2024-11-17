
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "action_config", "feature", "flag_group", "flag_set", "tool_path", "tool", "variable_with_value")

def _impl(ctx):
    gcc_version = ctx.attr.gcc_version
    all_link_actions = [
        ACTION_NAMES.cpp_link_executable,
        ACTION_NAMES.cpp_link_dynamic_library,
        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
    ]
    all_compile_actions = [
        ACTION_NAMES.c_compile,
        ACTION_NAMES.cpp_compile,
    ]
    action_configs = [
      action_config(
        action_name = ACTION_NAMES.preprocess_assemble,
        tools = [tool(path="/usr/bin/gcc-{}".format(gcc_version))],
      ),
      action_config(
        action_name = ACTION_NAMES.assemble,
        tools = [tool(path="/usr/bin/gcc-{}".format(gcc_version))],
      ),
      action_config(
        action_name = ACTION_NAMES.c_compile,
        tools = [tool(path="/usr/bin/gcc-{}".format(gcc_version))],
        implies = [
          "c_compile_flags"
        ],
      ),
      action_config(
        action_name = ACTION_NAMES.cpp_compile,
        tools = [tool(path="/usr/bin/g++-{}".format(gcc_version))],
        # implies = [
        #   "cpp_compile_flags"
        # ],
      ),
      action_config(
        action_name = ACTION_NAMES.cpp_link_executable,
        tools = [tool(path="/usr/bin/g++-{}".format(gcc_version))],
        # implies = [
        #   "default_linker_flags"
        # ],
      ),
      action_config(
        action_name = ACTION_NAMES.cpp_link_dynamic_library,
        tools = [tool(path="/usr/bin/g++-{}".format(gcc_version))],
        # implies = [
        #   "default_linker_flags"
        # ],
      ),
      action_config(
        action_name = ACTION_NAMES.cpp_link_static_library,
        tools = [tool(path="/usr/bin/ar")],
        implies = [
          "archiver_flags"
        ],
      ),
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
          enabled=False,
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
                  flags=["-std=c++20", "-fPIC", "-no-canonical-prefixes", "-fno-canonical-system-headers"],
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
                  flags=["-fPIC"],
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
      feature(
          name = "archiver_flags",
          flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        flags = [
                            "rcsD",
                            "%{output_execpath}",
                        ],
                        expand_if_available = "output_execpath",
                    ),
                ],
            ),
            flag_set(
                actions = [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        iterate_over = "libraries_to_link",
                        flag_groups = [
                            flag_group(
                                flags = ["%{libraries_to_link.name}"],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "object_file",
                                ),
                            ),
                            flag_group(
                                flags = ["%{libraries_to_link.object_files}"],
                                iterate_over = "libraries_to_link.object_files",
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "object_file_group",
                                ),
                            ),
                        ],
                        expand_if_available = "libraries_to_link",
                    ),
                ],
            ),
            flag_set(
                actions = [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        flags = ["%{user_archiver_flags}"],
                        iterate_over = "user_archiver_flags",
                        expand_if_available = "user_archiver_flags",
                    ),
                ],
            ),
        ],
      )
    ]

    return cc_common.create_cc_toolchain_config_info(
      ctx=ctx,
      features = features,
      action_configs = action_configs,
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
    )

gcc_toolchain_config = rule(
    implementation = _impl,
    attrs = {
      "gcc_version": attr.string(),
    },
    provides = [CcToolchainConfigInfo]
)
