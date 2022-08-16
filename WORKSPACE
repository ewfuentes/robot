load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

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
  name = "sophus",
  urls = ["https://github.com/strasdat/Sophus/archive/refs/tags/v22.04.1.zip"],
  strip_prefix= "Sophus-22.04.1",
  build_file="@//third_party:BUILD.sophus",
  patches = [
    "@//third_party:sophus-0001-cpp20-va-args-changes.patch",
    "@//third_party:sophus-0002-quote-eigen-includes.patch",
  ],
  patch_args=["-p1"],
  sha256 = "60d1d6c81426af8f330960002fb351db06e595501274310ddbe7bfc0aacda97a",
)