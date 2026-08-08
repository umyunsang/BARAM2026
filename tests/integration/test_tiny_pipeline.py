import json
import subprocess
import sys
from pathlib import Path


def _run_fixture(root: Path) -> dict[str, object]:
    program = """
import json
import sys
from pathlib import Path
from baram.workflows import run_tiny_fixture_pipeline

result = run_tiny_fixture_pipeline(Path(sys.argv[1]))
payload = json.loads(result.receipt_paths[0].read_text(encoding="utf-8"))
print(json.dumps({"payload": payload, "summary_sha256": result.summary_sha256}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(root)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_tiny_pipeline_reproduces_hashes_in_fresh_processes(tmp_path: Path) -> None:
    """Catches path-dependent audit, OOF, policy, final-fit, or CSV artifacts."""
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = _run_fixture(first_root)
    second = _run_fixture(second_root)
    assert first == second
    assert set(first["payload"]) == {
        "expected_candidate_sha256",
        "expected_policy_sha256",
        "manifest_sha256",
        "policy_sha256",
        "quantile_sha256",
        "spatial_feature_sha256",
        "submission_sha256",
    }
    assert not (first_root / "artifacts" / "locks" / "lockbox-2024.consumed.json").exists()
    assert not (second_root / "artifacts" / "locks" / "lockbox-2024.consumed.json").exists()
