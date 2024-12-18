workspace(name = "robot")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
  name = "aarch64-none-linux-gnu",
  urls = [
    "https://toolchains.bootlin.com/downloads/releases/toolchains/aarch64/tarballs/aarch64--glibc--bleeding-edge-2020.08-1.tar.bz2"
  ],
  strip_prefix="aarch64--glibc--bleeding-edge-2020.08-1",
  build_file="//third_party:BUILD.aarch64-none-linux-gnu",
  sha256 = "212f3c05f3b2263b0e2f902d055aecc2755eba10c0011927788a38faee8fc9aa"
)

http_archive(
  name = "jetson_sysroot",
  urls = ["https://www.dropbox.com/scl/fi/39qmmgn3mdnhj14sa21zl/cleaned_jetson.tar?rlkey=2fq8ynle042p6ojhprz3vjis3&dl=1"],
  build_file = "//third_party:BUILD.jetson_sysroot",
  integrity = "sha256-CVtgQSRXe8o2wMjNcJrxtlUxiHZY9ZQJ0kde8BkrTPw="
)

http_archive(
  name = "symphony_lake_snippet",
  urls = ["https://www.dropbox.com/scl/fi/gh7855qdos5mspinst77k/symphony_lake_snippet.zip?rlkey=mg7qe0gq51r446dwhr9g2wiut&st=774393fx&dl=1"],
  build_file = "//third_party:BUILD.symphony_lake_snippet",
  sha256 = "f16211fb370c9471153c9ed4a345b9fb848d292dbb8b7dc26fea24cb30ba5c15",
)

http_archive(
    name = "bazel_skylib",
    sha256 = "bc283cdfcd526a52c3201279cda4bc298652efa898b10b4db0837dc51652756f",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
    ],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

http_archive(
    name = "platforms",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.8/platforms-0.0.8.tar.gz",
        "https://github.com/bazelbuild/platforms/releases/download/0.0.8/platforms-0.0.8.tar.gz",
    ],
    sha256 = "8150406605389ececb6da07cbcb509d5637a3ab9a24bc69b1101531367d89d74",
)

register_toolchains(
  "//toolchain:clang_15_toolchain_for_linux_x84_64",
  "//toolchain:clang_18_toolchain_for_linux_x84_64",
  "//toolchain:gcc_10_toolchain_for_linux_x84_64",
  "//toolchain:gcc_11_toolchain_for_linux_x84_64",
  "//toolchain:gcc_toolchain_for_linux_aarch64",
)

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
    name = "com_github_gflags_gflags",
    sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
    strip_prefix = "gflags-2.2.2",
    urls = ["https://github.com/gflags/gflags/archive/v2.2.2.tar.gz"],
)

http_archive(
    name = "com_github_google_glog",
    sha256 = "c17d85c03ad9630006ef32c7be7c65656aba2e7e2fbfc82226b7e680c771fc88",
    strip_prefix = "glog-0.7.1",
    urls = ["https://github.com/google/glog/archive/v0.7.1.zip"],
)

http_archive(
  name = "com_google_absl",
  urls = ["https://github.com/abseil/abseil-cpp/releases/download/20240116.1/abseil-cpp-20240116.1.tar.gz"],
  strip_prefix = "abseil-cpp-20240116.1",
  sha256 = "3c743204df78366ad2eaf236d6631d83f6bc928d1705dd0000b872e53b73dc6a",
)

http_archive(
    name = "org_bzip_bzip2",
    sha256 = "ab5a03176ee106d3f0fa90e381da478ddae405918153cca248e682cd0c4a2269",
    strip_prefix = "bzip2-1.0.8",
    build_file = "@//third_party:BUILD.bz2.bazel",
    urls = ["https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz"],
)

http_archive(
  name = "fmt",
  urls = ["https://github.com/fmtlib/fmt/releases/download/11.0.2/fmt-11.0.2.zip"],
  strip_prefix="fmt-11.0.2",
  patch_cmds=[
    "mv support/bazel/BUILD.bazel BUILD.bazel",
    "mv support/bazel/WORKSPACE.bazel WORKSPACE.bazel",
  ],
  integrity = "sha256-QPxYvrzzjHWeEae9j9wWNQfSQj71BYu6fyYoDFucVGU=",
)

http_archive(
  name = "pybind11_bazel",
  strip_prefix = "pybind11_bazel-2.11.1.bzl.2",
  urls = ["https://github.com/pybind/pybind11_bazel/releases/download/v2.11.1.bzl.2/pybind11_bazel-2.11.1.bzl.2.zip"],
  sha256 = "d911ef169750491c9ddb4e6630bae882b127425627af10e59d499f0f7ff90a48"
)

# We still require the pybind library.
http_archive(
  name = "pybind11",
  build_file = "@pybind11_bazel//:pybind11.BUILD",
  strip_prefix = "pybind11-2.11.1",
  urls = ["https://github.com/pybind/pybind11/archive/v2.11.1.tar.gz"],
  sha256 = "d475978da0cdc2d43b73f30910786759d593a9d8ee05b1b6846d1eb16c6d2e0c"
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
    sha256 = "690e0141724abb568267e003c7b6d9a54925df40c275a870a4d934161dc9dd53",
    strip_prefix = "rules_python-0.40.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.40.0/rules_python-0.40.0.tar.gz",
    patch_args = ["-p1"],
    patches = ["//third_party:rules_python_0001-disable-user-site-package.patch"],
)

load("@rules_python//python:repositories.bzl", "py_repositories", "python_register_multi_toolchains")
py_repositories()

DEFAULT_PYTHON_VERSION = "3.10"
python_register_multi_toolchains(
  name="python",
  python_versions = ['3.10', '3.8'],
  default_version = DEFAULT_PYTHON_VERSION
)

load("@python//:pip.bzl", "multi_pip_parse")

multi_pip_parse(
  name="pip",
  default_version = DEFAULT_PYTHON_VERSION,
  python_interpreter_target = {
    "3.8": "@python_3_8_host//:python",
    "3.10": "@python_3_10_host//:python"
  },
  requirements_lock = {
    "3.8": "//third_party/python:requirements_3_8.txt",
    "3.10": "//third_party/python:requirements_3_10.txt"
  },
)

load("@pip//:requirements.bzl", install_pip_deps = "install_deps")
install_pip_deps()

http_archive(
  name="embag",
  url="https://github.com/embarktrucks/embag/archive/74c0b5f9d50bd45bcb6ed8e44718cd60924c13d0.zip",
  strip_prefix="embag-74c0b5f9d50bd45bcb6ed8e44718cd60924c13d0",
  build_file="@//third_party:BUILD.embag",
  sha256 = "d1715dc5887b6a9cbf67ddd3fe05a3507c0528101d41cf9858d1408597f01536",
  patches = [
    "@//third_party:embag_0001-delete-lib-build-file.patch",
    "@//third_party:embag_0002-use-std-span.patch",
    "@//third_party:embag_0003-handle-tabs-in-message-definition-of-constants.patch",
    "@//third_party:embag_0004-fix-build-warnings.patch",
    "@//third_party:embag_0005-display-primitive-arrays.patch",
    "@//third_party:embag_0006-delete-python-build.patch",
  ],
  patch_args=["-p1"],
)

http_archive(
    name = "com_github_nelhage_rules_boost",
    urls = ["https://github.com/nelhage/rules_boost/archive/5160325dbdc8c9e499f9d9917d913f35f1785d52.zip"],
    strip_prefix="rules_boost-5160325dbdc8c9e499f9d9917d913f35f1785d52",
    integrity = "sha256-/rSxKUaEx533weCPGuxdoNpSAh4z21nIjtvoa00aAXo=",
)
load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()

http_archive(
    name = "liblz4",
    build_file = "@embag//lz4:BUILD",
    sha256 = "658ba6191fa44c92280d4aa2c271b0f4fbc0e34d249578dd05e50e76d0e5efcc",
    strip_prefix = "lz4-1.9.2",
    urls = ["https://github.com/lz4/lz4/archive/v1.9.2.tar.gz"],
)

http_archive(
    name = "rules_foreign_cc",
    sha256 = "476303bd0f1b04cc311fc258f1708a5f6ef82d3091e53fd1977fa20383425a6a",
    strip_prefix = "rules_foreign_cc-0.10.1",
    url = "https://github.com/bazelbuild/rules_foreign_cc/releases/download/0.10.1/rules_foreign_cc-0.10.1.tar.gz",
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
  urls = ["https://github.com/protocolbuffers/protobuf/releases/download/v28.3/protobuf-28.3.zip"],
  strip_prefix="protobuf-28.3",
  integrity = "sha256-s7TDts/nS3eurpkJ/DwTAwknF/cbwhVNfBlhrNr1/kw=",
  patches = [
    "@//third_party:protobuf_0001-use-rules-python-headers.patch",
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
  name = "com_github_grpc_grpc",
  urls = ["https://github.com/grpc/grpc/archive/refs/tags/v1.65.2.zip"],
  strip_prefix = "grpc-1.65.2",
  patches = [
    "@//third_party:grpc-0001-disable-apple-support.patch"
  ],
  patch_args=["-p1"],
  integrity = "sha256-sRdeHMvwBhaL64iCSGLCYFlt/pgVQZf24l05HuGT3W4="
)

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps", "grpc_test_only_deps")
grpc_deps()
load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
grpc_extra_deps()

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
  name = "cpptrace",
  url = "https://github.com/jeremy-rifkin/cpptrace/archive/refs/tags/v0.2.1.zip",
  strip_prefix = "cpptrace-0.2.1",
  build_file = "@//third_party:BUILD.cpptrace",
  patches = [
    "@//third_party:cpptrace-0001-config.patch",
  ],
  patch_args=["-p1"],
  sha256 = "d274b672286825aba5ba11f4a87e3c1995ac43a5c4fd57f319fb77eb574cfcfd"
)

http_archive(
  name = "assert",
  url = "https://github.com/jeremy-rifkin/libassert/archive/f81df0aae6915fdbf7d5ea2ac24f77cb2e0e7ee1.zip",
  build_file = "@//third_party:BUILD.assert",
  strip_prefix = "libassert-f81df0aae6915fdbf7d5ea2ac24f77cb2e0e7ee1",
  sha256 = "b1da53cbb265b2d943ed9063696567751c62c3e2836b6bfa04d1cf0b8eb5a971",
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

http_archive(
  name = "drake_lib_jammy",
  url = "https://github.com/RobotLocomotion/drake/releases/download/v1.34.0/drake-1.34.0-jammy.tar.gz",
  strip_prefix="drake",
  build_file="@//third_party:BUILD.drake",
  sha256="aec27c1e65d5ec587a325dc0a4a462e0a0464fcdc17ca47c64dc2cf49ccb51a3",
)

http_archive(
  name = "drake_lib_noble",
  url = "https://github.com/RobotLocomotion/drake/releases/download/v1.34.0/drake-1.34.0-noble.tar.gz",
  strip_prefix="drake",
  build_file="@//third_party:BUILD.drake",
  sha256="393eda8c6e24e83635b96f8b8a541e2ca6a6b3b58deb115bf1e96a026dd3b26e",
)

http_archive(
  name = "opencv",
  url = "https://github.com/opencv/opencv/archive/refs/tags/4.7.0.zip",
  strip_prefix="opencv-4.7.0",
  build_file="//third_party:BUILD.opencv",
  sha256 = "db6fb5e7dc76829d738fdbcdba11a810c66ca0a4752e531eaf3d793361e96de8",
)

http_archive(
  name = "opencv_contrib",
  url = "https://github.com/opencv/opencv_contrib/archive/refs/tags/4.7.0.zip",
  strip_prefix="opencv_contrib-4.7.0",
  build_file="//third_party:BUILD.opencv_contrib",
  integrity = "sha256-7wAYE+w5IVWTzp3rOuxwqFJ49XoO2IsY9vYVJlhVQ2w="
)


load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "b1e80761a8a8243d03ebca8845e9cc1ba6c82ce7c5179ce2b295cd36f7e394bf",
    urls = ["https://github.com/bazelbuild/rules_docker/releases/download/v0.25.0/rules_docker-v0.25.0.tar.gz"],
)

load(
    "@io_bazel_rules_docker//repositories:repositories.bzl",
    container_repositories = "repositories",
)
container_repositories()

load("@io_bazel_rules_docker//repositories:deps.bzl", container_deps = "deps")

container_deps()

load(
    "@io_bazel_rules_docker//container:container.bzl",
    "container_pull",
)

container_pull(
  name = "cuda_base",
  registry = "docker.io",
  repository = "nvidia/cuda",
  tag = "11.7.1-devel-ubuntu22.04",
  digest = "sha256:a9f41355320a9a029e75ce6227132de853b28870fe53dc740ad463b6eda19c83",
)

load("@io_bazel_rules_docker//python3:image.bzl", _py_image_repos = "repositories")
_py_image_repos()

http_archive(
    name = "approxcdf",
    urls = ["https://github.com/david-cortes/approxcdf/archive/ec5486c083ad1ca93834a17fd1347657a3181add.zip"],
    strip_prefix="approxcdf-ec5486c083ad1ca93834a17fd1347657a3181add",
    build_file="//third_party:BUILD.approxcdf",
    sha256 = "bef56ff1eb3e4e2ae2f91fe21a4317a4d7744e7f1930170e2dacc9619438a52d",
    patch_args = ["-p1"],
    patches = ["//third_party:approxcdf_0001-add-missing-curly.patch"]
)

http_archive(
  name = "annoy",
  urls = ["https://github.com/spotify/annoy/archive/2be37c9e015544be2cf60c431f0cccc076151a2d.zip"],
  strip_prefix = "annoy-2be37c9e015544be2cf60c431f0cccc076151a2d",
  build_file = "//third_party:BUILD.annoy",
  sha256 = "330e67b9f25173f6b15b44238f3caabebefe927e64c7ed981683924f5db682e7",
  patch_args = ["-p1"],
  patches = ["//third_party:annoy_0001-add-maybe-unused-attributes.patch"]
)

http_archive(
  name = "bs_thread_pool",
  urls = ["https://github.com/bshoshany/thread-pool/archive/refs/tags/v4.1.0.zip"],
  strip_prefix = "thread-pool-4.1.0",
  build_file = "//third_party:BUILD.thread_pool",
  integrity = "sha256-oYooSRZTByr+J/duGfpRD1gBw4EWCUgPHadAmGHg8P4=",
)

http_archive(
  name = "sqlite3",
  urls = ["https://www.sqlite.org/2024/sqlite-amalgamation-3460000.zip"],
  strip_prefix = "sqlite-amalgamation-3460000",
  build_file = "//third_party:BUILD.sqlite3",
  integrity = "sha256-cSp9CdKiJlL7BqSa9RbgUZeaOYStsGfahnYOYO1Rp/U="
)

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "symphony_lake_parser",
    urls = ["https://github.com/pizzaroll04/SymphonyLakeDataset/archive/4b38c2519270a43f76858e6c97c1f1de1735e5d0.zip"],
    strip_prefix = "SymphonyLakeDataset-4b38c2519270a43f76858e6c97c1f1de1735e5d0",
    build_file = "//third_party:BUILD.symphony_lake_parser",
    sha256 = "d203e486507c7950ae9a346406fe2c42f8b2d204e8d25ba07a47c834f4ae8ede",
)

http_archive(
    name = "gtsam",
    build_file = "@rules_gtsam//third_party:gtsam.BUILD",
    sha256 = "8b44d6b98a3b608664d1c9a7c1383a406550499d894533bb0183e6cf487e6457",
    strip_prefix = "gtsam-4.2.0",
    urls = ["https://github.com/borglab/gtsam/archive/4.2.0.tar.gz"],
    patch_args=["-p1"],
    patches = [
        "//third_party:gtsam_0001-remove-redundant-template-params.patch",
        "//third_party:gtsam_0002-remove-trivial-copy-constructor.patch",
    ],
)

http_archive(
  name = "rules_gtsam",
  urls = ["https://github.com/pizzaroll04/rules_gtsam/archive/c554d4f1dcd86cf191067094549c57d70030ba97.zip"],
  sha256 = "1f7a36b162adb0d6ff737543cdbc7d560339934aa64948224b4c8980112604ac",
  strip_prefix = "rules_gtsam-c554d4f1dcd86cf191067094549c57d70030ba97",
)
load("@rules_gtsam//bzl:init_deps.bzl", "gtsam_init_deps")
gtsam_init_deps()
load("@rules_gtsam//bzl:repositories.bzl", "gtsam_repositories")
gtsam_repositories()

http_archive(
  name = "dbow2",
  urls = ["https://github.com/dorian3d/DBoW2/archive/3924753db6145f12618e7de09b7e6b258db93c6e.zip"],
  strip_prefix = "DBoW2-3924753db6145f12618e7de09b7e6b258db93c6e",
  build_file = "//third_party:BUILD.dbow2",
  integrity = "sha256-yrI1lbG5RfEh8ubiEtUCP9ToKNbJrmMGX+AMsJVABvg=",
  patch_args = ["-p1"],
  patches = ["//third_party:dbow2_0001-prefix-include-paths.patch"]
)

http_archive(
  name = "opengv",
  urls = ["https://github.com/laurentkneip/opengv/archive/91f4b19c73450833a40e463ad3648aae80b3a7f3.zip"],
  strip_prefix="opengv-91f4b19c73450833a40e463ad3648aae80b3a7f3",
  build_file="//third_party:BUILD.opengv",
  patch_args = ["-p1"],
  patches = ["//third_party:opengv_0001-prefix-unsupported-eigen-include-paths.patch",
             "//third_party:opengv_0002-unused-parameter.patch"],
  integrity = "sha256-gIK3IvE6rGDpxvN3BA+EPFKPvZ7Zi7Ix33IIcm2RULA="
)

http_archive(
  name = "kimera_rpgo",
  urls = ["https://github.com/MIT-SPARK/Kimera-RPGO/archive/ab3fe8c30dd587f5b2ba4ca276bf92cbd593dcf5.zip"],
  strip_prefix="Kimera-RPGO-ab3fe8c30dd587f5b2ba4ca276bf92cbd593dcf5",
  build_file="//third_party:BUILD.kimera_rpgo",
  integrity = "sha256-IkZM58CdITLhDNf7bYcgPFli2NSr/yfIUnWSHAP3hkM="
)

http_archive(
  name = "kimera_vio",
  urls = ["https://github.com/ewfuentes/Kimera-VIO/archive/master.zip"],
  strip_prefix = "Kimera-VIO-master",
  build_file = "//third_party:BUILD.kimera_vio",
  integrity = "sha256-dWzux1ioSB8LfovhQFAcbmlotNo19x++7ptw9S64UGM=",
)

http_archive(
  name = "nlohmann_json",
  urls = ["https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.zip"],
  strip_prefix = "json-3.11.3",
  integrity = "sha256-BAIrBdgG61/3MCPCgLaGl9Erk+G3JnoLIqGjnsdXgGk=",
)

http_archive(
  name = "spectacular_log_snippet",
  urls = ["https://www.dropbox.com/scl/fi/6y2x4nstw7h3xx6jvbzzg/20241212_150605.zip?rlkey=rwktg7egzki7vqeq9qpfaml1l&dl=1"],
  build_file = "//third_party:BUILD.zip_file",
  integrity = "sha256-7WD3AyKE+v7fA6ohAqk1UXBU+i2n9uH/6HcovNhORhA="
)

http_archive(
  name = "libosmium",
  urls = ["https://github.com/osmcode/libosmium/archive/refs/tags/v2.20.0.zip"],
  build_file = "//third_party:BUILD.libosmium",
  strip_prefix = "libosmium-2.20.0",
  integrity = "sha256-cS7BoPRNtinhyvatqlM2kV9CPLD2ZJFC14I5i1p1x3k=",
)

http_archive(
  name = "protozero",
  urls = ["https://github.com/mapbox/protozero/archive/refs/tags/v1.7.1.zip"],
  build_file = "//third_party:BUILD.protozero",
  strip_prefix = "protozero-1.7.1",

)

http_archive(
  name = "openstreetmap_snippet",
  urls = ["https://www.dropbox.com/scl/fi/ku5dwktiul0wzx9ocelwc/us-virgin-islands-latest.osm.zip?rlkey=ggo7dnskexdqxvira6m6y32f3&dl=1"],
  build_file = "//third_party:BUILD.zip_file",
  integrity = "sha256-miPvL6co2035EGlkvbcksmTO6HwB/AknVLQK+/YDet0="
)
