"""What code made this, recorded rather than enforced.

Code is deliberately NOT part of `artifact_identity`. It was, briefly, and the
measurement that killed the idea is worth keeping: three commits touching only
viewer HTML invalidated all eight producing stages, and even excluding those,
one ordinary day's work invalidated two of eight. In a research tree code
changes constantly and data does not, so gating on code means near-permanent
invalidation of exactly the artifacts that cost money to make.

Two further reasons it was the wrong tool:

- these artifacts are not byte-reproducible anyway. Extraction, audit and
  matching are provider calls with real variance, so re-running the IDENTICAL
  code yields different landmarks. "The code changed, so this is invalid" is a
  much weaker claim than it sounds when the same code would also not reproduce.
- gating and recording are separable, and only recording is needed. Keep the
  commit and the working diff and the question "was this made by different
  code?" stays answerable whenever it is asked -- which is when a result looks
  wrong, not on every read.

So: identity gates on DATA lineage (upstream artifacts, resolved config, input
digests), and this module records code as provenance.

RECORD THE CONTENT, NOT A POINTER. A commit hash is a reference into a mutable
object store and references rot: the commits stamped into the artifacts on disk
today survive through exactly one hand-made safety branch, because a force-push
orphaned them. A `git gc` would leave 115 artifacts pointing at nothing. And a
SHA does not identify a dirty tree at all, which is the normal state while
developing. So the diff travels with the record.

The mechanism is the one `common/torch/load_and_save_models.py` already uses:
Bazel's `--workspace_status_command` (`toolchain/workspace_status.sh`) stamps
`STABLE_GIT_COMMIT` and a base64 `STABLE_GIT_DIFF` into a generated
`toolchain/git_info.py`.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

SCHEMA = "farfield_code_provenance/v1"
UNKNOWN = "unknown"

# Presentation is excluded from the "did the code change?" signal, and only
# from that signal -- these files still appear in the recorded diff, which is
# verbatim.
#
# `publication` refreshes the data-root index on every publish, which is
# wanted: the browsable tree should never lag the artifacts. The side effect is
# that `viewers.indexes` and `viewers.page` sit in every producing stage's
# import closure, so a restyle reads as "the tracker changed". It cannot: no
# edit to an HTML template alters `predictions.jsonl`. Excluding presentation
# keeps the signal about science while leaving the refresh alone.
PRESENTATION_SUFFIXES = (".html", ".css", ".js")
PRESENTATION_PATH_MARKERS = (
    "/viewers/",
    "_viewer.py",
    "_viewer_assets/",
    "/viewer_assets/",
)


class CodeProvenanceError(ValueError):
    """A recorded code-provenance block is malformed."""


def _git_info():
    try:
        from toolchain import git_info  # noqa: PLC0415
    except ImportError:
        return None
    return git_info


def _diff_text() -> str | None:
    info = _git_info()
    raw = getattr(info, "STABLE_GIT_DIFF", None) if info else None
    if not raw:
        return None
    try:
        return base64.b64decode(raw).decode("utf-8", errors="replace")
    except (ValueError, TypeError):
        return None


def is_presentation(path: str) -> bool:
    """Whether a repository path is presentation rather than computation."""
    if path.endswith(PRESENTATION_SUFFIXES):
        return True
    return any(marker in f"/{path}" for marker in PRESENTATION_PATH_MARKERS)


def split_diff(diff: str) -> list[tuple[str, str]]:
    """A unified diff as (path, hunk-text) pairs, one per file."""
    sections: list[tuple[str, str]] = []
    path, lines = None, []
    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if path is not None:
                sections.append((path, "".join(lines)))
            lines = [line]
            parts = line.split(" b/", 1)
            path = parts[1].strip() if len(parts) == 2 else UNKNOWN
        else:
            lines.append(line)
    if path is not None:
        sections.append((path, "".join(lines)))
    return sections


def computational_diff(diff: str | None) -> str:
    """The diff with presentation files removed.

    This is what the change signal is computed over. The verbatim diff is kept
    separately, so nothing is lost -- only the question "did anything that can
    affect an artifact change?" gets a useful answer.
    """
    if not diff:
        return ""
    return "".join(text for path, text in split_diff(diff)
                   if not is_presentation(path))


def record() -> dict[str, Any]:
    """The code-provenance block to store beside an artifact.

    Always returns a well-formed block. Outside `bazel run` the stamp is
    absent and the commit reads `unknown`; that is a truthful record of not
    knowing, which is the point of writing it down rather than gating on it.
    """
    info = _git_info()
    commit = getattr(info, "STABLE_GIT_COMMIT", None) if info else None
    diff = _diff_text()
    computational = computational_diff(diff)
    return {
        "schema": SCHEMA,
        "commit": commit or UNKNOWN,
        # Verbatim, so the record stands alone when the commit is unreachable.
        "diff": diff or "",
        "diff_sha256": hashlib.sha256((diff or "").encode()).hexdigest(),
        # The digest the change signal compares. Presentation-only edits leave
        # this identical, so a restyle does not read as a science change.
        "computational_diff_sha256": hashlib.sha256(
            computational.encode()).hexdigest(),
        "dirty": bool(diff),
        "computationally_dirty": bool(computational),
    }


def validate(block: Any) -> dict[str, Any]:
    if not isinstance(block, Mapping):
        raise CodeProvenanceError("code provenance must be an object")
    missing = sorted({"schema", "commit", "computational_diff_sha256"}
                     - set(block))
    if missing:
        raise CodeProvenanceError(f"code provenance is missing {missing}")
    if block["schema"] != SCHEMA:
        raise CodeProvenanceError(
            f"unsupported code-provenance schema {block['schema']!r}")
    return dict(block)


def differs(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Whether two records describe different computational code.

    Presentation edits are not a difference. An `unknown` commit on either
    side IS a difference, because not knowing is not the same as matching.
    """
    left, right = validate(left), validate(right)
    if UNKNOWN in (left["commit"], right["commit"]):
        return True
    return (left["commit"] != right["commit"]
            or left["computational_diff_sha256"]
            != right["computational_diff_sha256"])


def lineage_summary(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Whether a set of artifacts was made by one code state, and by which.

    The failure this design accepts is a stale artifact quietly entering a
    comparison -- an evaluation table with one leg from before a fix and one
    from after is a wrong conclusion nobody can see, and no gate will stop it
    now. So it is reported instead: `code_differs` is the flag a viewer or
    report shows, and it is deliberately one boolean rather than a list,
    because "these were not all built the same way" is the whole message.
    """
    blocks = [validate(item) for item in records]
    if not blocks:
        return {"n_artifacts": 0, "code_differs": False, "commits": [],
                "any_dirty": False, "any_unknown": False}
    commits = sorted({block["commit"] for block in blocks})
    digests = {block["computational_diff_sha256"] for block in blocks}
    unknown = UNKNOWN in commits
    return {
        "n_artifacts": len(blocks),
        # One flag: differing commits, differing computational diffs, or an
        # unknown among them all mean the same thing to a reader.
        "code_differs": len(commits) > 1 or len(digests) > 1 or unknown,
        "commits": commits,
        "any_dirty": any(block.get("computationally_dirty") for block in blocks),
        "any_unknown": unknown,
    }


def describe(summary: Mapping[str, Any]) -> str:
    """One line for a viewer header or a report."""
    if not summary["n_artifacts"]:
        return "no artifacts in this lineage"
    if not summary["code_differs"]:
        detail = ("with uncommitted changes" if summary["any_dirty"]
                  else "clean")
        return (f"lineage built from one code state "
                f"({summary['commits'][0][:12]}, {detail})")
    parts = [f"lineage spans {len(summary['commits'])} code state(s)"]
    if summary["any_unknown"]:
        parts.append("including one that recorded no commit")
    if summary["any_dirty"]:
        parts.append("including uncommitted changes")
    return "CODE DIFFERS: " + ", ".join(parts)
