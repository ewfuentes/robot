Format: 
clang-format-15 -i **/*.cc **/*.hh

Build all bazel
bazel build //...

Rebase:
git rebase <commit hash to define chunk> --onto <branch name>
This will rebase everything from current head to commit hash onto the given branch name

Rebuild compile_commands.json
bazel run :refresh_compile_commands


Run CI tests locally:
bazel test //...

Run a specific test:
bazel run //experimental/overhead_matching:kimera_spectacular_data_provider_test -- --gtest_filter=KimeraSpectacularDataProviderTest.real_data

Run with GDB 
gdb --args  bazel-bin/experimental/overhead_matching/kimera_spectacular_data_provider_test --gtest_filter=KimeraSpectacularDataProviderTest.real_data

Work on updating an external repo: 
clone repo into temp 
copy BUILD from /third_party/BUILD.third_party to /tmp/repo/BUILD
touch /tmp/repo/WORKSPACE

# run with 
bazel run //experimental/overhead_matching:kimera_vio_pipeline_test --override_repository=kimera_vio=/tmp/Kimera-VIO

when fix is done, git diff > /tmp/repo/fix.patch
then delete the integrety = sha256 from the WORKSPACE file and rebuild your target. Copy the new sha256


# For jupyter notebook
bazel run //common/python:jupyter_notebook

# When you start deveoping off a branch, then that branch gets squashed and merged into master, you need to rebase your branch off master
git rebase --onto main old_base_branch my_branch


# Updating bazel python deps
Add the new dep to requirements_3_10.in file (check pypi for version)
Run the following command to update the bazel dependencies
`bazel run //third_party/python:requirements_3_12.update`

Running ipython in bazel test:
just swap bazel run // with ./bazel-bin/ and replace : with /