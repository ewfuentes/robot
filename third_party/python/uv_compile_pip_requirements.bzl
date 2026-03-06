def uv_compile_pip_requirements(name, requirements_in, requirements_txt, python_version):
    """Drop-in replacement for compile_pip_requirements using uv.

    Creates:
      - {name}.update: bazel run target to regenerate the lockfile
      - {name}: bazel test target to check lockfile freshness
    """
    version_dotted = python_version.replace("_", ".")

    native.sh_binary(
        name = name + ".update",
        srcs = ["//third_party/python:uv_pip_compile.sh"],
        args = [
            "update",
            version_dotted,
            "$(location {})".format(requirements_in),
            "$(rootpath {})".format(requirements_txt),
        ],
        data = [requirements_in, requirements_txt],
        tags = ["requires-network"],
    )

    native.sh_test(
        name = name,
        srcs = ["//third_party/python:uv_pip_compile.sh"],
        args = [
            "test",
            version_dotted,
            "$(location {})".format(requirements_in),
            "$(location {})".format(requirements_txt),
        ],
        data = [requirements_in, requirements_txt],
        tags = ["no-cache", "requires-network", "external"],
        timeout = "eternal",
    )
