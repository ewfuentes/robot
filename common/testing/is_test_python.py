
import os

def is_test() -> bool:
    return 'BAZEL_TEST' in os.environ and 'BUILD_WORKSPACE_DIRECTORY' not in os.environ
