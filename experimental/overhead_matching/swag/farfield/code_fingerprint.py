"""What code computed an artifact, as a digest of the sources that can.

An artifact's identity has to include the code that produced it, or a change
to a stage silently yields a stale product. The pipeline used the repository's
git commit for that, which is both too coarse and too weak:

- too coarse, because *any* commit anywhere -- a viewer restyle, a docstring,
  another project in the monorepo -- changes it, so every artifact in the tree
  is invalidated by work that cannot have affected it. That is what made the
  global build identity unusable in practice and forced `stage_reuse`'s
  human attestation into existence;
- too weak, because it says nothing about *which* code. Two commits differing
  only in a comment are as different as two commits that rewrote the tracker.

So fingerprint the sources that actually compute the artifact: an entry module
plus every farfield module it transitively imports. A change to
`localization/viewer.py` does not move `tracking/run_tracking`'s fingerprint,
and a change to `geometry.py` moves nearly everything's -- both correct.

The import graph is walked statically rather than read out of `sys.modules`,
for three reasons: it does not depend on what happened to be imported first,
it does not drag in test modules, and it can be computed for a stage this
process is not running.

DELIBERATE LIMITS, because a fingerprint that overstates its coverage is worse
than one whose gaps are known:

- only modules under this package are followed. numpy, torch and Bazel's own
  toolchains are outside the fingerprint; pinning those is the lockfile's job.
- data files a module reads at import (the viewer stylesheets, for instance)
  are not followed. Modules that depend on such a file should hash it into
  their own artifact config, where it is visible.
- a dynamic import inside a function body is invisible to a static walk. There
  are none in the farfield stage entry points today; `_assert_no_dynamic_farfield_import`
  keeps it that way rather than trusting it.
"""

from __future__ import annotations

import ast
import hashlib
from functools import lru_cache
from pathlib import Path

PACKAGE = "experimental.overhead_matching.swag.farfield"
SCHEMA = "farfield_code_fingerprint/v1"

# Everything under the package root lives beside this file.
_ROOT = Path(__file__).resolve().parent


class CodeFingerprintError(ValueError):
    """A module's source cannot be located or read."""


def _module_path(module: str) -> Path | None:
    """Source file for a dotted farfield module, or None if outside."""
    if module != PACKAGE and not module.startswith(PACKAGE + "."):
        return None
    relative = module[len(PACKAGE):].lstrip(".")
    base = _ROOT / Path(*relative.split(".")) if relative else _ROOT
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    raise CodeFingerprintError(
        f"{module!r} is a farfield module with no source at {base}.py")


def _imported_modules(tree: ast.AST, module: str) -> set[str]:
    """Dotted farfield modules named by one module's import statements."""
    package = module.rsplit(".", 1)[0] if "." in module else module
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # Relative import: resolve against this module's package.
                parts = package.split(".")
                base = ".".join(parts[:len(parts) - node.level + 1])
                root = f"{base}.{node.module}" if node.module else base
            else:
                root = node.module or ""
            # `from x.y import z` names either x.y.z (a module) or an
            # attribute of x.y. Offer both; _module_path rejects neither
            # silently -- a name that is not a module simply has no source.
            found.add(root)
            found.update(f"{root}.{alias.name}" for alias in node.names)
    return {name for name in found
            if name == PACKAGE or name.startswith(PACKAGE + ".")}


def _assert_no_dynamic_farfield_import(tree: ast.AST, module: str) -> None:
    """A static walk cannot see `importlib.import_module` or a nested import.

    Rather than let that be a silent hole in every fingerprint, refuse. If a
    stage ever genuinely needs a dynamic import, the fix is to name the target
    explicitly here, not to accept a fingerprint that quietly omits it.
    """
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"):
            raise CodeFingerprintError(
                f"{module!r} calls import_module, which a static import walk "
                "cannot follow; its fingerprint would silently omit the "
                "target")


def _closure(entry: str) -> dict[str, Path]:
    """Entry module plus every farfield module it transitively imports."""
    resolved: dict[str, Path] = {}
    pending = [entry]
    while pending:
        module = pending.pop()
        if module in resolved:
            continue
        try:
            path = _module_path(module)
        except CodeFingerprintError:
            if module == entry:
                raise
            # `from pkg.mod import name`: `pkg.mod.name` is an attribute, not
            # a module. `pkg.mod` itself was offered alongside it and resolves.
            continue
        if path is None:
            continue
        resolved[module] = path
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as error:
            raise CodeFingerprintError(
                f"cannot read source of {module!r}: {error}") from error
        _assert_no_dynamic_farfield_import(tree, module)
        pending.extend(_imported_modules(tree, module))
    return resolved


@lru_cache(maxsize=None)
def fingerprint(entry_module: str) -> str:
    """Digest of `entry_module` and the farfield sources it can reach.

    Stable across checkouts: keyed on each module's path relative to the
    package root and the bytes of its source, never on an absolute path or a
    timestamp.
    """
    closure = _closure(entry_module)
    digest = hashlib.sha256()
    digest.update(SCHEMA.encode("utf-8"))
    for module in sorted(closure):
        path = closure[module]
        digest.update(b"\0")
        digest.update(str(path.relative_to(_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest()
                      .encode("utf-8"))
    return digest.hexdigest()


def modules(entry_module: str) -> tuple[str, ...]:
    """The fingerprinted module set, for reporting why an id moved."""
    return tuple(sorted(_closure(entry_module)))
