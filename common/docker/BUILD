
load("@io_bazel_rules_docker//container:container.bzl", "container_image")
load("@io_bazel_rules_docker//docker/package_managers:download_pkgs.bzl", "download_pkgs")
load("@io_bazel_rules_docker//docker/package_managers:install_pkgs.bzl", "install_pkgs")

download_pkgs(
   name = "python3_pkgs",
   image_tar = "@cuda_base//image",
   packages = ["python3", "python-is-python3"],
   tags = ["manual"]
)

install_pkgs(
    name = "cuda_python",
    image_tar = "@cuda_base//image",
    installables_tar = "python3_pkgs.tar",
    installation_cleanup_commands = "rm -rf /var/lib/apt/lists/*",
    output_image_name="cuda_python",
    visibility=["//visibility:public"],
    tags = ["manual"]
)


