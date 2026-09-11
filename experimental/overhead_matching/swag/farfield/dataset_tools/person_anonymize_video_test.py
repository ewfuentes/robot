import json
import hashlib
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.farfield.dataset_tools import person_anonymize_video


def _instance(mask, confidence=0.4):
    return {"mask": mask, "confidence": confidence, "passes": ["native"]}


def test_scan_evidence_round_trip(tmp_path: Path):
    direct = np.zeros((7, 13), dtype=bool)
    weak = np.zeros_like(direct)
    direct[1:4, 2:8] = True
    weak[5, 11] = True
    path = tmp_path / "frame_00000003.npz"
    person_anonymize_video.write_scan_evidence(
        path, frame_index=3, source_frame_index=393,
        direct_mask=direct, weak_mask=weak,
        instances=[_instance(direct)])

    loaded = person_anonymize_video.load_scan_evidence(path, (7, 13))
    assert loaded["frame_index"] == 3
    assert loaded["source_frame_index"] == 393
    assert np.array_equal(loaded["direct_mask"], direct)
    assert np.array_equal(loaded["weak_mask"], weak)
    assert not np.any(loaded["vehicle_mask"])
    assert loaded["instances"] == [{
        "confidence": 0.4,
        "passes": ["native"],
        "class_id": 0,
        "area_pixels": 18,
    }]


def test_committed_scan_indices_accepts_resumable_sparse_results(tmp_path: Path):
    frames = tmp_path / "frames"
    frames.mkdir()
    mask = np.zeros((4, 8), dtype=bool)
    for index in (0, 2):
        person_anonymize_video.write_scan_evidence(
            frames / f"frame_{index:08d}.npz", frame_index=index,
            source_frame_index=10 + index, direct_mask=mask, weak_mask=mask,
            instances=[])
    assert person_anonymize_video.committed_scan_indices(
        frames, 3, (4, 8), 10) == {0, 2}


def test_committed_scan_indices_rejects_identity_mismatch(tmp_path: Path):
    frames = tmp_path / "frames"
    frames.mkdir()
    mask = np.zeros((4, 8), dtype=bool)
    person_anonymize_video.write_scan_evidence(
        frames / "frame_00000000.npz", frame_index=0,
        source_frame_index=99, direct_mask=mask, weak_mask=mask,
        instances=[])
    try:
        person_anonymize_video.committed_scan_indices(
            frames, 1, (4, 8), 10)
    except ValueError as error:
        assert "identity mismatch" in str(error)
    else:
        raise AssertionError("mismatched source identity was accepted")


def test_locked_stage_refuses_changed_spec(tmp_path: Path):
    output = tmp_path / "result"
    with person_anonymize_video._locked_stage(output, {"setting": 1}):
        pass
    try:
        with person_anonymize_video._locked_stage(output, {"setting": 2}):
            pass
    except ValueError as error:
        assert "changed scan specification" in str(error)
    else:
        raise AssertionError("changed resume specification was accepted")


def test_scan_spec_is_stable_json():
    assert json.loads(person_anonymize_video._json_bytes({"b": 2, "a": 1})) == {
        "a": 1, "b": 2}


def test_auto_device_prefers_cuda_and_records_resolved_device(monkeypatch):
    monkeypatch.setattr(
        person_anonymize_video, "_cuda_available", lambda: True)
    assert person_anonymize_video._resolve_inference_device("auto") == "0"
    monkeypatch.setattr(
        person_anonymize_video, "_cuda_available", lambda: False)
    assert person_anonymize_video._resolve_inference_device("auto") == "cpu"
    assert person_anonymize_video._resolve_inference_device("cuda:1") == (
        "cuda:1")


def test_render_parser_accepts_manifest_bound_nvenc_backend():
    parser = person_anonymize_video.build_parser()
    default_args = parser.parse_args([
        "render", "--policy_dir", "/tmp/policy", "--output_dir",
        "/tmp/output",
    ])
    assert default_args.encoder == "software"
    args = parser.parse_args([
        "render", "--policy_dir", "/tmp/policy", "--output_dir",
        "/tmp/output", "--encoder", "nvenc",
    ])
    assert args.encoder == "nvenc"


def test_scan_implementation_binds_scanner_reader_and_detector(monkeypatch):
    observed = []

    def fake_sha256(path):
        observed.append(Path(path).name)
        return "01" * 32

    monkeypatch.setattr(
        person_anonymize_video.anonymize_video, "sha256_file", fake_sha256)
    result = person_anonymize_video._implementation_sha256()

    assert len(result) == 64
    assert set(observed) == {
        "anonymize_video.py", "person_anonymize_video.py",
        "person_segmentation_preview.py"}


def test_scan_finalization_revalidates_inputs_code_and_runtime(
        tmp_path: Path, monkeypatch):
    source = tmp_path / "source.mp4"
    weights = tmp_path / "weights.pt"
    source.write_bytes(b"source")
    weights.write_bytes(b"weights")
    implementation = person_anonymize_video._implementation_sha256()
    runtime = {"runtime": "fixed"}
    monkeypatch.setattr(
        person_anonymize_video, "_scan_library_versions", lambda: runtime)
    spec = {
        "source": {"sha256": hashlib.sha256(b"source").hexdigest()},
        "weights": {"sha256": hashlib.sha256(b"weights").hexdigest()},
        "implementation_sha256": implementation,
        "library_versions": runtime,
    }

    person_anonymize_video._verify_scan_finalization_inputs(
        spec, source, weights)
    source.write_bytes(b"changed")
    try:
        person_anonymize_video._verify_scan_finalization_inputs(
            spec, source, weights)
    except ValueError as error:
        assert "scan source video" in str(error)
    else:
        raise AssertionError("changed scan source was accepted")

    source.write_bytes(b"source")
    weights.write_bytes(b"changed")
    try:
        person_anonymize_video._verify_scan_finalization_inputs(
            spec, source, weights)
    except ValueError as error:
        assert "scan model weights" in str(error)
    else:
        raise AssertionError("changed scan model was accepted")

    weights.write_bytes(b"weights")
    spec["implementation_sha256"] = "00" * 32
    try:
        person_anonymize_video._verify_scan_finalization_inputs(
            spec, source, weights)
    except ValueError as error:
        assert "implementation changed" in str(error)
    else:
        raise AssertionError("changed scan implementation was accepted")

    spec["implementation_sha256"] = implementation
    spec["library_versions"] = {"runtime": "changed"}
    try:
        person_anonymize_video._verify_scan_finalization_inputs(
            spec, source, weights)
    except ValueError as error:
        assert "runtime changed" in str(error)
    else:
        raise AssertionError("changed scan runtime was accepted")


def test_stage_implementation_binds_expected_modules():
    policy = person_anonymize_video._stage_implementation(
        "policy-test", include_persistence=True)
    render = person_anonymize_video._stage_implementation(
        "render-test", include_persistence=False)
    assert policy["contract"] == "policy-test"
    assert set(policy["module_sha256"]) == {
        "anonymize_video", "person_anonymize_video",
        "person_mask_persistence", "person_segmentation_preview"}
    assert render["contract"] == "render-test"
    assert "person_mask_persistence" not in render["module_sha256"]
    assert all(len(value) == 64 for value in policy["module_sha256"].values())


def test_conservative_resize_never_drops_a_tiny_direct_mask():
    mask = np.zeros((8, 16), dtype=bool)
    mask[3, 7] = True
    resized = person_anonymize_video._conservative_resize_mask(mask, 8, 4)
    assert np.any(resized)


def test_boundary_weak_person_candidate_is_retained_for_review():
    direct = np.zeros((12, 20), dtype=bool)
    weak = np.zeros_like(direct)
    direct[3:9, 4:9] = True
    weak[2:8, 13:18] = True

    result = person_anonymize_video._boundary_persistence(direct, weak)

    assert np.array_equal(result.accepted_mask, direct)
    assert not np.any(result.temporal_fill_mask)
    assert len(result.review_flags) == 1
    assert result.review_flags[0].reasons == (
        "boundary_unconfirmed_candidate",)
    assert np.array_equal(result.review_flags[0].mask, weak)


def test_boundary_adjacent_person_is_retained_as_one_sided_review():
    shape = (48, 96)
    frame = np.zeros((*shape, 3), dtype=np.uint8)
    direct = np.zeros(shape, dtype=bool)
    weak = np.zeros_like(direct)
    adjacent = np.zeros_like(direct)
    adjacent[16:34, 37:51] = True

    result = person_anonymize_video._boundary_persistence(
        direct, weak, boundary_frame=frame, adjacent_frame=frame,
        adjacent_direct_mask=adjacent)

    assert not np.any(result.temporal_fill_mask)
    assert len(result.review_flags) == 1
    assert result.review_flags[0].reasons == (
        "boundary_one_sided_person_evidence",)
    assert np.array_equal(result.review_flags[0].mask, adjacent)


def test_plate_candidate_requires_geometry_and_vehicle_context():
    vehicle = np.zeros((100, 200), dtype=bool)
    vehicle[35:70, 60:150] = True
    candidate = {"box": [0.45, 0.50, 0.60, 0.57], "confidence": 0.8}
    accepted, metrics = person_anonymize_video.validate_plate_candidate(
        candidate, vehicle)
    assert accepted
    assert metrics["geometry_plausible"]
    assert metrics["vehicle_overlap"] > 0.1

    rejected, metrics = person_anonymize_video.validate_plate_candidate(
        candidate, np.zeros_like(vehicle))
    assert not rejected
    assert metrics["vehicle_overlap"] == 0


def test_plate_candidate_rejects_a_panorama_spanning_box():
    vehicle = np.ones((100, 200), dtype=bool)
    accepted, metrics = person_anonymize_video.validate_plate_candidate(
        {"box": [0.1, 0.1, 0.8, 0.6], "confidence": 0.9}, vehicle)
    assert not accepted
    assert not metrics["geometry_plausible"]


def test_plausible_plate_is_review_only_without_vehicle_context(
        tmp_path: Path):
    frames = tmp_path / "frames"
    frames.mkdir()
    empty = np.zeros((100, 200), dtype=bool)
    evidence = frames / "frame_00000000.npz"
    person_anonymize_video.write_scan_evidence(
        evidence, frame_index=0, source_frame_index=10,
        direct_mask=empty, weak_mask=empty, vehicle_mask=empty, instances=[])
    scan_manifest = {
        "preprocessing": {"resolution": [200, 100]},
        "evidence": {"directory": "frames"},
    }
    scan_rows = [{
        "sha256": person_anonymize_video.anonymize_video.sha256_file(evidence)}]
    candidate = {
        "category": "license_plate_candidate",
        "source": "yolov9_plate_raw_qa",
        "confidence": 0.93,
        "box": [0.45, 0.50, 0.53, 0.58],
    }

    applied, review, audit = person_anonymize_video._filtered_plate_policy(
        [[candidate]], tmp_path, scan_manifest, scan_rows)

    assert not applied[0]
    assert len(review[0]) == 1
    assert review[0][0]["decision"] == "rejected_missing_vehicle_context"
    assert audit[0]["candidates"] == review[0]


def test_plausible_plate_is_blurred_with_vehicle_context(tmp_path: Path):
    frames = tmp_path / "frames"
    frames.mkdir()
    empty = np.zeros((100, 200), dtype=bool)
    vehicle = np.zeros_like(empty)
    vehicle[45:70, 80:130] = True
    evidence = frames / "frame_00000000.npz"
    person_anonymize_video.write_scan_evidence(
        evidence, frame_index=0, source_frame_index=10,
        direct_mask=empty, weak_mask=empty, vehicle_mask=vehicle, instances=[])
    scan_manifest = {
        "preprocessing": {"resolution": [200, 100]},
        "evidence": {"directory": "frames"},
    }
    scan_rows = [{
        "sha256": person_anonymize_video.anonymize_video.sha256_file(evidence)}]
    candidate = {
        "category": "license_plate_candidate",
        "source": "yolov9_plate_raw_qa",
        "confidence": 0.93,
        "box": [0.45, 0.50, 0.53, 0.58],
    }

    applied, review, audit = person_anonymize_video._filtered_plate_policy(
        [[candidate]], tmp_path, scan_manifest, scan_rows)

    assert len(applied[0]) == 1
    assert applied[0][0]["source"] == "yolov9_plate_vehicle_validated"
    assert not review[0]
    assert audit[0]["candidates"][0]["decision"] == (
        "accepted_vehicle_supported")


def test_policy_evidence_round_trip(tmp_path: Path):
    temporal = np.zeros((9, 17), dtype=bool)
    raw = np.zeros_like(temporal)
    display = np.zeros_like(temporal)
    temporal[2:5, 4:11] = True
    raw[7, 15] = True
    path = tmp_path / "frame_00000002.npz"
    person_anonymize_video.write_policy_evidence(
        path, frame_index=2, source_frame_index=392,
        temporal_mask=temporal, review_raw_mask=raw,
        review_display_mask=display, metadata={"fills": ["test"]})
    loaded = person_anonymize_video.load_policy_evidence(path, (9, 17))
    assert np.array_equal(loaded["temporal_mask"], temporal)
    assert np.array_equal(loaded["review_raw_mask"], raw)
    assert np.array_equal(loaded["review_display_mask"], display)
    assert loaded["metadata"] == {"fills": ["test"]}


def test_person_blur_is_mask_shaped_and_local():
    rng = np.random.default_rng(4)
    frame = rng.integers(0, 256, size=(240, 480, 3), dtype=np.uint8)
    original = frame.copy()
    mask = np.zeros((60, 120), dtype=bool)
    mask[25:35, 50:65] = True
    regions = person_anonymize_video.person_blur_regions(mask, 480, 240)
    assert regions
    person_anonymize_video.apply_person_blur(frame, mask)
    assert np.any(frame != original)
    assert np.array_equal(frame[:20, :20], original[:20, :20])


def test_render_chunks_cover_every_frame_once():
    assert person_anonymize_video._chunk_ranges(11, 4) == [
        (0, 4), (4, 8), (8, 11)]


def test_concat_sidecar_rejects_orphan_output(tmp_path: Path):
    spec = {
        "source": {"width": 20, "height": 10},
        "output_fps": 3.0,
        "frame_count": 2,
        "render": {"review_width": 20, "review_speedup": 5.0},
    }
    spec_sha = hashlib.sha256(
        person_anonymize_video._json_bytes(spec)).hexdigest()
    ranges = [(0, 1), (1, 2)]
    for start, end in ranges:
        chunk = tmp_path / "chunks" / person_anonymize_video._chunk_name(
            start, end)
        chunk.mkdir(parents=True)
        items = {}
        for kind in ("full", "review"):
            video = chunk / f"{kind}.mp4"
            video.write_bytes(f"{kind}-{start}".encode())
            items[f"{kind}_video"] = {
                "path": video.name,
                "sha256": person_anonymize_video.anonymize_video.sha256_file(
                    video),
            }
        (chunk / "chunk_manifest.json").write_text(json.dumps({
            "start_frame": start,
            "end_frame_exclusive": end,
            "render_spec_sha256": spec_sha,
            "extracted_frame_sha256": {},
            **items,
        }))

    calls = []
    original_run = person_anonymize_video.subprocess.run
    original_validate = person_anonymize_video._validate_video_contract

    def fake_run(command, check):
        assert check
        calls.append(command)
        Path(command[-1]).write_bytes(b"verified-concat")

    def fake_validate(path, **_):
        assert path.read_bytes() == b"verified-concat"
        return {"synthetic": True}

    person_anonymize_video.subprocess.run = fake_run
    person_anonymize_video._validate_video_contract = fake_validate
    try:
        output = tmp_path / "full-output.mp4"
        first = person_anonymize_video._concat_chunks(
            tmp_path, ranges, "full", output, spec)
        assert first["output_sha256"] == (
            person_anonymize_video.anonymize_video.sha256_file(output))
        assert len(calls) == 1

        output.write_bytes(b"planted-unblurred-output")
        second = person_anonymize_video._concat_chunks(
            tmp_path, ranges, "full", output, spec)
        assert output.read_bytes() == b"verified-concat"
        assert len(calls) == 2
        assert second["output_sha256"] == first["output_sha256"]
        quarantines = list((tmp_path / "failed_concats").iterdir())
        assert len(quarantines) == 1
        assert (quarantines[0] / "full-output.mp4").read_bytes() == (
            b"planted-unblurred-output")

        person_anonymize_video._concat_chunks(
            tmp_path, ranges, "full", output, spec)
        assert len(calls) == 2
    finally:
        person_anonymize_video.subprocess.run = original_run
        person_anonymize_video._validate_video_contract = original_validate


def test_review_html_is_file_url_safe_and_native_resolution(tmp_path: Path):
    path = tmp_path / "review.html"
    person_anonymize_video._write_render_review_html(
        path, full_name="blurred.mp4", review_name="review.mp4",
        ledger_name="detections.jsonl", clip_start_s=130.0,
        output_fps=3.0, review_speedup=5.0, source_start_frame=390,
        source_width=7680, source_height=3840, frame_count=8,
        flagged_indices=[1, 6])
    html = path.read_text()
    assert "const flagged=[1,6]" in html
    assert "fetch(" not in html
    assert "width:7680px;height:3840px" in html
    assert "Math.floor(review.currentTime*fps*speed" in html
    assert "shift-click" in html
    assert "halfWindow=0.25/fps" in html

    # The rounded JSON bounds select exactly one 3 fps source-grid timestamp,
    # including frames whose times are repeating thirds.
    for target in range(8):
        center = 130.0 + target / 3.0
        start = round(max(130.0, center - 0.25 / 3.0), 9)
        end = round(min(130.0 + 8 / 3.0, center + 0.25 / 3.0), 9)
        selected = [index for index in range(8)
                    if start <= 130.0 + index / 3.0 < end]
        assert selected == [target]

    # A deterministic retry validates the existing file instead of trusting it.
    person_anonymize_video._write_render_review_html(
        path, full_name="blurred.mp4", review_name="review.mp4",
        ledger_name="detections.jsonl", clip_start_s=130.0,
        output_fps=3.0, review_speedup=5.0, source_start_frame=390,
        source_width=7680, source_height=3840, frame_count=8,
        flagged_indices=[1, 6])
