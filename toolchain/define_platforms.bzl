
mappings = {
  "jammy": ":jammy_22.04",
  "noble": ":noble_24.04",
  "clang": {"jammy": ":clang15", "noble": ":clang18"},
  "gcc": {"jammy": ":clang15", "noble": ":clang18"},
  "x86_64": "@platforms//cpu:x86_64",
  "aarch64": "@platforms//cpu:aarch64",
}

def define_platforms():
  for os in ["jammy", "noble"]:
    for compiler in ["clang", "gcc"]:
      for arch in ["x86_64", "aarch64"]:
        native.platform(
          name = "_".join([os, compiler, arch]),
          constraint_values = [
            "@platforms//os:linux",
            mappings[os],
            mappings[compiler][os],
            mappings[arch],
          ]
        )
