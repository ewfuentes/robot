from toolchain import git_info
import base64
# Usage example:

with open("/tmp/diff.txt", 'w') as f:
    f.write(base64.b64decode(git_info.STABLE_GIT_DIFF).decode())
