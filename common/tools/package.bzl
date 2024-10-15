load("@rules_pkg//pkg:zip.bzl", "pkg_zip")
load("@rules_pkg//pkg:mappings.bzl", "pkg_files", "pkg_filegroup")
load("@rules_pkg//pkg:providers.bzl", "PackageFilesInfo")
def _cc_library_static_libs_impl(ctx):
  files = []
  dest_src_map = {}
  print(ctx.workspace_name)
  for lib in ctx.attr.libs:
    for f in lib[DefaultInfo].files.to_list():
      dest_src_map['lib/' + f.short_path] = f
      files.append(f)
    for f in lib[CcInfo].compilation_context.headers.to_list():
      if not f.is_source:
        continue
      dest_src_map['include/' + f.path] = f
      files.append(f)
  return [DefaultInfo(files = depset(files)),
  PackageFilesInfo(
    attributes = {},
    dest_src_map=dest_src_map,
  )]
_do_cc_library_static_libs = rule(
  implementation = _cc_library_static_libs_impl,
  attrs = {
    'libs': attr.label_list(
      providers = [CcInfo],
    ),
  },
)
def cc_package_library(name, libs):
  _do_cc_library_static_libs(
    name = name + '__static_libs',
    visibility = ['//visibility:private'],
    libs = libs,
  )
  pkg_zip(
    name = name,
    srcs = [
	":%s__static_libs" % name,
    ],
  )
