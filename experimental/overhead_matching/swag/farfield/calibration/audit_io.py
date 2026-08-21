"""Reader for the semantic-audit artifact inside a tracking run.

The audit stage writes `<run_dir>/semantic_audit/results.jsonl` (one batch
result per line) plus `audit_meta.json` mapping each result `key` to the
track it audited. This module is the one place that mapping is decoded;
every consumer that needs "which tracks were audited, and what did the audit
say" (offset sweep, localization export) goes through `load_audits` rather
than re-parsing the JSONL.

The parse is deliberately minimal: each line carries the model's audit
payload as JSON text at `response.candidates[0].content.parts[0].text`, and
consumers only need the payload as a dict (`valid_segments` for tracklet
fusion, verdict/semantics fields downstream). No schema validation happens
here; error lines and unparseable payloads are skipped, exactly as the audit
stage's own reader skips them.
"""

import json
from pathlib import Path


def load_audits(run_dir: Path) -> dict:
    """track_id -> audit dict from the semantic-audit artifact, or {} when
    the audit stage has not run.

    Audit membership is the support gate (see tracking/tracklets.py): a
    track absent from this mapping has no canonical semantics, so callers
    must not let it reach matching or the filter.
    """
    audit_dir = Path(run_dir) / "semantic_audit"
    results = audit_dir / "results.jsonl"
    meta_path = audit_dir / "audit_meta.json"
    if not (results.exists() and meta_path.exists()):
        return {}
    meta = json.loads(meta_path.read_text())
    audits = {}
    with open(results) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            key = record.get("key")
            if record.get("error") or key not in meta:
                continue
            try:
                text = (record["response"]["candidates"][0]["content"]
                        ["parts"][0]["text"])
                audits[meta[key]["track_id"]] = json.loads(text)
            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                continue
    return audits
