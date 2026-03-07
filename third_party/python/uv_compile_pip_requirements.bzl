load("@rules_python//python:versions.bzl", "MINOR_MAPPING")

def uv_compile_pip_requirements(name, requirements_in, requirements_txt, python_version):
    """Drop-in replacement for compile_pip_requirements using uv.

    Creates:
      - {name}.update: bazel run target to regenerate the lockfile
      - {name}: bazel test target to check lockfile freshness
    """
    version_dotted = python_version.replace("_", ".")
    resolution_version_dotted = MINOR_MAPPING.get(version_dotted, version_dotted)

    uv_data = select({
        "@platforms//cpu:x86_64": ["@uv_x86_64//:uv"],
        "@platforms//cpu:aarch64": ["@uv_aarch64//:uv"],
    })

    native.sh_binary(
        name = name + ".update",
        srcs = ["//third_party/python:uv_pip_compile.sh"],
        args = [
            "update",
            version_dotted,
            resolution_version_dotted,
            "$(location {})".format(requirements_in),
            "$(rootpath {})".format(requirements_txt),
        ],
        data = [requirements_in, requirements_txt] + uv_data,
        tags = ["requires-network"],
    )

    native.sh_test(
        name = name,
        srcs = ["//third_party/python:uv_pip_compile.sh"],
        args = [
            "test",
            version_dotted,
            resolution_version_dotted,
            "$(location {})".format(requirements_in),
            "$(location {})".format(requirements_txt),
        ],
        data = [requirements_in, requirements_txt] + uv_data,
        tags = ["no-cache", "requires-network", "external"],
        timeout = "moderate",
    )
