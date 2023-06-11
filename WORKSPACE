workspace(name = "robot")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_pkg",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.8.0/rules_pkg-0.8.0.tar.gz",
        "https://github.com/bazelbuild/rules_pkg/releases/download/0.8.0/rules_pkg-0.8.0.tar.gz",
    ],
    sha256 = "eea0f59c28a9241156a47d7a8e32db9122f3d50b505fae0f33de6ce4d9b61834",
)
load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")
rules_pkg_dependencies()

http_archive(
  name = "absl",
  urls = ["https://github.com/abseil/abseil-cpp/archive/refs/tags/20220623.1.zip"],
  strip_prefix = "abseil-cpp-20220623.1",
  sha256 = "54707f411cb62a26a776dad5fd60829098c181700edcd022ea5c2ca49e9b7ef1",
)

http_archive(
  name = "fmt",
  urls = ["https://github.com/fmtlib/fmt/releases/download/10.0.0/fmt-10.0.0.zip"],
  strip_prefix="fmt-10.0.0",
  patch_cmds=[
    "mv support/bazel/.bazelversion .bazelversion",
    "mv support/bazel/BUILD.bazel BUILD.bazel",
    "mv support/bazel/WORKSPACE.bazel WORKSPACE.bazel",
  ],
  sha256 = "4943cb165f3f587f26da834d3056ee8733c397e024145ca7d2a8a96bb71ac281",
)

http_archive(
  name = "pybind11_bazel",
  strip_prefix = "pybind11_bazel-faf56fb3df11287f26dbc66fdedf60a2fc2c6631",
  urls = ["https://github.com/pybind/pybind11_bazel/archive/faf56fb3df11287f26dbc66fdedf60a2fc2c6631.zip"],
  sha256 = "a185aa68c93b9f62c80fcb3aadc3c83c763854750dc3f38be1dadcb7be223837",
)

# We still require the pybind library.
http_archive(
  name = "pybind11",
  build_file = "@pybind11_bazel//:pybind11.BUILD",
  strip_prefix = "pybind11-2.10.2",
  urls = ["https://github.com/pybind/pybind11/archive/v2.10.2.tar.gz"],
  sha256 = "93bd1e625e43e03028a3ea7389bba5d3f9f2596abc074b068e70f4ef9b1314ae",
)

http_archive(
    name = "zlib",
    build_file = "@//third_party:BUILD.zlib",
    sha256 = "629380c90a77b964d896ed37163f5c3a34f6e6d897311f1df2a7016355c45eff",
    strip_prefix = "zlib-1.2.11",
    urls = ["https://github.com/madler/zlib/archive/v1.2.11.tar.gz"],
)

# Note that rules_python must be loaded before protobuf
http_archive(
    name = "rules_python",
    sha256 = "8c8fe44ef0a9afc256d1e75ad5f448bb59b81aba149b8958f02f7b3a98f5d9b4",
    strip_prefix = "rules_python-0.13.0",
    url = "https://github.com/bazelbuild/rules_python/archive/refs/tags/0.13.0.tar.gz",
)

load("@rules_python//python:repositories.bzl", "python_register_toolchains")
python_register_toolchains(
    name = "python3_10",
    python_version = "3.10",
)


load("@python3_10//:defs.bzl", "interpreter")
load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python", python_interpreter_target=interpreter)
bind(
  name = "python_headers",
  actual = "@python3_10//:python_headers",
)

load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
  name = "pip",
  python_interpreter_target = interpreter,
  requirements = "@//third_party/python:requirements.txt",
)

load("@pip//:requirements.bzl", "install_deps")
install_deps()

http_archive(
    name = "rules_foreign_cc",
    sha256 = "bdaa954208400a084cb19f8271dd81524a8d1fb341da3e2694e2781244670ed9",
    strip_prefix = "rules_foreign_cc-62230f8a2123a64b411a3888f75128a10bfe0755",
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/62230f8a2123a64b411a3888f75128a10bfe0755.tar.gz",
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
rules_foreign_cc_dependencies()


http_archive(
    name = "rules_proto",
    sha256 = "80d3a4ec17354cccc898bfe32118edd934f851b03029d63ef3fc7c8663a7415c",
    strip_prefix = "rules_proto-5.3.0-21.5",
    urls = [
        "https://github.com/bazelbuild/rules_proto/archive/refs/tags/5.3.0-21.5.tar.gz",
    ],
)
load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")
rules_proto_dependencies()
rules_proto_toolchains()

http_archive(
  name = "com_google_protobuf",
  urls = ["https://github.com/protocolbuffers/protobuf/archive/refs/tags/v21.6.tar.gz"],
  strip_prefix="protobuf-21.6",
  sha256 = "dbb16fdbca8f277c9a194d9a837395cde408ca136738d94743130dd0de015efd",
  patches = [
    "@//third_party:protobuf-0001-disable-warning-flags.patch",
  ],
  patch_args=["-p1"],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()

http_archive(
  name = "com_google_googletest",
  urls = ["https://github.com/google/googletest/archive/5f467ec04df33024e3c6760fa403b5cd5d8e9ace.zip"],
  strip_prefix = "googletest-5f467ec04df33024e3c6760fa403b5cd5d8e9ace",
  sha256 = "aff2e98fd8fb11becd00d3820f33839eb16370c82693bb8552118c93d963773a"
) 

http_archive(
  name = "glfw",
  urls = ["https://github.com/glfw/glfw/releases/download/3.3.8/glfw-3.3.8.zip"],
  strip_prefix = "glfw-3.3.8",
  build_file="@//third_party:BUILD.glfw",
  sha256 = "4d025083cc4a3dd1f91ab9b9ba4f5807193823e565a5bcf4be202669d9911ea6",
)

http_archive(
  name = "eigen",
  urls = ["https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"],
  strip_prefix = "eigen-3.4.0",
  build_file="@//third_party:BUILD.eigen",
  sha256 = "1ccaabbfe870f60af3d6a519c53e09f3dcf630207321dffa553564a8e75c4fc8",
)

http_archive(
  name = "sophus_lie",
  urls = ["https://github.com/strasdat/Sophus/archive/refs/tags/1.22.4.zip"],
  strip_prefix= "Sophus-1.22.4",
  build_file="@//third_party:BUILD.sophus",
  patches = [
    "@//third_party:sophus-0001-cpp20-va-args-changes.patch",
    "@//third_party:sophus-0002-quote-eigen-includes.patch",
  ],
  patch_args=["-p1"],
  sha256 = "7748c82c21f29a71b0f22c529bfb3ec468f52b313b26041257aca3dc330ddb90"
)

http_archive(
  name = "cxxopts",
  urls = ["https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.0.0.tar.gz"],
  strip_prefix = "cxxopts-3.0.0",
  sha256 = "36f41fa2a46b3c1466613b63f3fa73dc24d912bc90d667147f1e43215a8c6d00"
)

http_archive(
  name = "wise_enum",
  urls = ["https://github.com/quicknir/wise_enum/archive/34ac79f7ea2658a148359ce82508cc9301e31dd3.zip"],
  strip_prefix = "wise_enum-34ac79f7ea2658a148359ce82508cc9301e31dd3",
  build_file="@//third_party:BUILD.wise_enum",
  sha256 = "9bd940ecb810aa6af15372436731a8fb898abf170342acf95850e8ce2747eda8",
)

# Hedron's Compile Commands Extractor for Bazel
# https://github.com/hedronvision/bazel-compile-commands-extractor
http_archive(
    name = "hedron_compile_commands",
    url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/f4546f123cd4717483b059b3dfa80a8abab1c9b7.tar.gz",
    strip_prefix = "bazel-compile-commands-extractor-f4546f123cd4717483b059b3dfa80a8abab1c9b7",
    sha256 = "c535b6d61e0b99b0c44af9146eb98daec64f920102ee978f9a1d3c45d607c7d7",
)
load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")
hedron_compile_commands_setup()

# Poker hand evaluator
http_archive(
  name = "ompeval",
  url = "https://github.com/zekyll/OMPEval/archive/4aec210ff75b0851af0ee170b35a7899e1a4fe8f.zip",
  strip_prefix="OMPEval-4aec210ff75b0851af0ee170b35a7899e1a4fe8f",
  build_file="@//third_party:BUILD.ompeval",
  patches = [
    "@//third_party:ompeval-0001-build-patches.patch",
    "@//third_party:ompeval-0002-fix-unused-parameter-issue.patch",
  ],
  patch_args=["-p1"],
  sha256 = "816febbcd7f1c014cfe57fae7b73732c7938a26355582a63666fcb96457949d0",
)

# MIT pokerbot skeleton
http_archive(
  name = "mit_pokerbots",
  url = "https://github.com/mitpokerbots/engine-2023/archive/efa09ffc8f89a612f49eabc252acface3fda52a4.zip",
  strip_prefix="engine-2023-efa09ffc8f89a612f49eabc252acface3fda52a4",
  build_file="@//third_party:BUILD.mit_pokerbots",
  sha256 = "4acd3c8d68305c6f3fb18bca2e2ef043fd938fe4b09ae8c2759b5cce545f8aaf",
)

# http_archive(
#   name = "drake_lib",
#   url = "https://github.com/RobotLocomotion/drake/releases/download/v1.16.0/drake-20230518-jammy.tar.gz",
#   strip_prefix="drake",
#   build_file_content="#",
#   sha256="811c24a328ffc0113c5b8127de2276f2d2d98f68cf036dc552766a11984f0b01",
# )
http_archive(
  name = "drake_lib",
  url = "https://drake-packages.csail.mit.edu/drake/nightly/drake-20230609-jammy.tar.gz",
  strip_prefix="drake",
  build_file_content="#",
  sha256 = "e2320ff066e5797df54e884395c7bfde26823d6c3532790180063f2ce7d438fa",
)
load("@drake_lib//:share/drake/repo.bzl", "drake_repository")
drake_repository(name="drake", excludes=["eigen", "fmt"])

http_archive(
  name = "opencv",
  url = "https://github.com/opencv/opencv/archive/refs/tags/4.7.0.zip",
  strip_prefix="opencv-4.7.0",
  build_file="//third_party:BUILD.opencv",
  sha256 = "db6fb5e7dc76829d738fdbcdba11a810c66ca0a4752e531eaf3d793361e96de8",
)

