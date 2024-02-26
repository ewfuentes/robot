
# Python Notes

## Adding New Versions of Python

To add new versions of Python, add a new block in the WORKSPACE file like so:
```
python_register_toolchains(
    name = "python3_8",
    python_version = "3.8.10",
)
load("@python3_8//:defs.bzl", interpreter_3_8 = "interpreter")
load("@rules_python//python:pip.bzl", "pip_parse")
pip_parse(
  name = "pip_3_8",
  python_interpreter_target = interpreter_3_8,
  requirements_lock = "@//third_party/python:requirements.txt",
)

load("@pip_3_8//:requirements.bzl", install_deps_3_8 = "install_deps")
install_deps_3_8()
```

Note that the toolchain that is registered first will be used by default.

## Building Against Different Python Versions

If the version has been defined in `.bazelrc`, you can pass in `--config=python-X.Y`.

If it hasn't been defined in `.bazelrc`, a different Python version can be used by passing in a `--extra_toolchains` argument to the build or run command. For example, if the name passed to `python_register_toolchains` is `python_3_10`, then the argument to be passed is `--extra_toolchains @python_3_10_toolchains//:x86_64-unknown-linux-gnu`. The list of available toolchains can be seen by running: `bazel query "@python3_10_toolchains//:*"`. To test that the appropriate Python version is getting selected, you can run `bazel build --extra_toolchains=<toolchain_from_above> //common/python:pybind_example_test`.

Note that if you are building a pybind extension, you must pass in the `.*_py_cc_toolchain`. If you are running a python script, you must pass in the `.*_toolchain`. Passing in both is possible.
