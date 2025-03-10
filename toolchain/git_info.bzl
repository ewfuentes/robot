def _git_info_impl(ctx):
    out = ctx.actions.declare_file("git_info.py")
    
    ctx.actions.write(
        output = out,
        content = """
# Generated file, do not edit directly
GIT_COMMIT = "{}"
GIT_DIFF = "{}"
""".format(
            ctx.var.get("STABLE_GIT_COMMIT", "unknown"),
            ctx.var.get("STABLE_GIT_DIFF", ""),
        ),
    )
    
    return [DefaultInfo(files = depset([out]))]

git_info = rule(
    implementation = _git_info_impl,
    outputs = {"out": "git_info.py"},
)