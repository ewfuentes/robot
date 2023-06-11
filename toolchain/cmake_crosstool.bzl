
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")

def _cmake_crosstool_impl(ctx):
  toolchain = find_cpp_toolchain(ctx)
  output_file = ctx.actions.declare_file(ctx.attr.name + ".cmake")

  cxx = toolchain.preprocessor_executable
  if cxx == "/bin/false":
    cxx = toolchain.compiler_executable

  ctx.actions.write(
    output_file,
    '''set(CMAKE_AR "{ar}" CACHE FILEPATH "Archiver")
set(CMAKE_CXX_COMPILER "{cxx}")
set(CMAKE_CXX_FLAGS_INIT "-O1")
set(CMAKE_C_COMPILER "{cc}")
set(CMAKE_C_FLAGS_INIT "-O1")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-shared")
'''.format(ar=toolchain.ar_executable,
      cc=toolchain.compiler_executable,
      cxx=cxx)
  )

  return [
    DefaultInfo(
      files=depset([output_file]),
    )
  ]

cmake_crosstool = rule(
  implementation = _cmake_crosstool_impl,
  attrs = {
    "_cc_toolchain": attr.label(default = Label("@bazel_tools//tools/cpp:current_cc_toolchain")),
  },
)
