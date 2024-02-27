
# Python Notes

## Adding New Versions of Python

To add new versions of Python, locate the `python_register_multi_toolchains` invocation in the WORKSPACE file and add it to the list of `python_version`. Additionally, in the following `multi_pip_parse` rule, add an entry for the `requirements.txt`.

```
DEFAULT_PYTHON_VERSION = "3.10"
python_register_multi_toolchains(
  name="python",
  python_versions = ['3.10', '3.8.10'], # add new entry here
  default_version = DEFAULT_PYTHON_VERSION
)

load("@python//:pip.bzl", "multi_pip_parse")

multi_pip_parse(
  name="pip",
  default_version = DEFAULT_PYTHON_VERSION,
  python_interpreter_target = {
    "3.8.10": "@python_3_8_10_host//:python",
    "3.10": "@python_3_10_host//:python"
  },
  requirements_lock = {
    "3.8.10": "//third_party/python:requirements_3_8_10.txt",
    "3.10": "//third_party/python:requirements_3_10.txt"
    # Add new entry here
  },
)
```

In `//third_party/python:BUILD`, load the `compile_pip_requirements` from the new version and instantiate a new compile pip requirements rule. To generate the requirements file, run `bazel run //third_party/python:requirements_<new_version>.update`. Note that you may need to create an empty file first before the rule succeeds.

## Building Against Different Python Versions

If the version has been defined in `.bazelrc`, you can pass in `--config=python-X.Y` to build with that toolchain.

If the version hasn't been defined in `.bazelrc`, you can pass in `--@rules_python//python/config_settings:python_version=<version>` to your build command to specify that version. 

If you wish to define a target that uses a different toolchain by default, you may do the following:
```
load("@python_<version>//:defs.bzl", py_binary_<version> = "py_binary")

py_binary_<version>(
  name = "target_name",
  ...
)
```
