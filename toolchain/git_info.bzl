def _git_info_impl(ctx):
    output = ctx.actions.declare_file("git_info.py")

    ctx.actions.run_shell(
        inputs = [ctx.info_file, ctx.version_file],
        outputs = [output],
        command = '''
        echo "# Autogenerated file. Do not edit." > {out}
        cat {info} | awk '{{print $1 " = \\"" $2 "\\""}}' >> {out}
        cat {version} | awk '{{print $1 " = \\"" $2 "\\""}}' >> {out}
        '''.format(
            info = ctx.info_file.path,
            version = ctx.version_file.path,
            out = output.path,
        ),
    )

    return [DefaultInfo(files = depset([output]))]

git_info = rule(
    implementation = _git_info_impl,
)


