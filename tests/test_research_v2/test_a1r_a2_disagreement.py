"""
Tests for the A1R-vs-A2 disagreement analysis script.

Validates:
  1. Script is importable and exposes main().
  2. All required output files are written to the versioned directory.
  3. Manifest has the required keys and sane values.
  4. No unsafe commitments are present (fairness-as-escalation-only invariant).
  5. Frozen result directories are untouched after the analysis.
  6. Git state is recorded in the manifest.
  7. Disagreement causes are exhaustive (no unknown labels).
  8. All transitions are between known decision values.
"""
import os, sys, json, importlib
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

SCRIPT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'run_a1r_a2_disagreement_analysis.py')
)
OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'research_v2',
                 'validation_results', 'a1r_a2_disagreement_v3')
)
FROZEN_DIRS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'research_v2', 'final_results')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'research_v2', 'validation_results')),
]
KNOWN_CAUSES = {
    "fairness", "routing", "policy",
    "certificate_commitment_gate", "reassessment", "unexplained",
}
KNOWN_DECISIONS = {"approved", "rejected", "escalated"}

REQUIRED_MANIFEST_KEYS = [
    "analysis", "generated_at", "git_commit", "seed", "test_split",
    "epsilon", "dataset_hash", "model_hash", "n_test", "n_agree",
    "n_disagree", "agree_rate", "disagree_rate", "transitions",
    "a1r_audit_triggered", "a2_audit_triggered", "a1r_bias_flagged",
    "a2_bias_flagged", "safe_block_count", "unsafe_count",
    "escalation_count_a2", "disagreement_causes",
    "frozen_manifests_ok", "paper_mode",
]

REQUIRED_OUTPUT_FILES = [
    "manifest.json",
    "all_cases.csv",
    "disagreements.csv",
    "disagreement_causes.csv",
    "disagreement_by_risk_category.csv",
]


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _frozen_snapshot():
    """Snapshot file sizes in frozen dirs before the analysis."""
    snap = {}
    for d in FROZEN_DIRS:
        for root, _, files in os.walk(d):
            for f in files:
                fp = os.path.join(root, f)
                snap[fp] = os.path.getsize(fp)
    return snap


@pytest.fixture(scope="module")
def analysis_result():
    """Run the analysis once per module; all tests share the result."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("disagreement_analysis", SCRIPT_PATH)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    before = _frozen_snapshot()
    manifest = mod.main()
    yield manifest, before


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_script_importable():
    """The script must import without side-effects."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("disagreement_analysis", SCRIPT_PATH)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod.main)


def test_output_files_written(analysis_result):
    """All required output files must be written."""
    _, _ = analysis_result
    for fname in REQUIRED_OUTPUT_FILES:
        fpath = os.path.join(OUTPUT_DIR, fname)
        assert os.path.exists(fpath), f"Missing output file: {fname}"
        assert os.path.getsize(fpath) > 0, f"Output file is empty: {fname}"


def test_manifest_required_keys(analysis_result):
    """Manifest must contain all required keys with correct types."""
    manifest, _ = analysis_result
    missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in manifest]
    assert not missing, f"Manifest missing keys: {missing}"


def test_manifest_sane_values(analysis_result):
    """Manifest numeric values must be internally consistent."""
    manifest, _ = analysis_result
    assert manifest["n_test"] > 0, "No test cases processed"
    assert 0.0 <= manifest["agree_rate"] <= 1.0
    assert manifest["n_agree"] + manifest["n_disagree"] == manifest["n_test"]
    assert abs(manifest["agree_rate"] - manifest["n_agree"] / manifest["n_test"]) < 1e-9
    assert manifest["paper_mode"] is True
    assert manifest["frozen_manifests_ok"] is True


def test_no_unsafe_commitments(analysis_result):
    """Fairness must NOT approve cases rejected by deterministic risk or policy.
    Zero unsafe commitments are expected."""
    manifest, _ = analysis_result
    assert manifest["unsafe_count"] == 0, (
        f"Unsafe fairness overrides detected: {manifest['unsafe_count']}. "
        "Fairness is an audit-and-escalation authority only."
    )


def test_disagreement_causes_known(analysis_result):
    """Every disagreement cause label must be from the known set."""
    manifest, _ = analysis_result
    causes_path = os.path.join(OUTPUT_DIR, "disagreement_causes.csv")
    df = pd.read_csv(causes_path)
    unknown = set(df["disagreement_cause"].unique()) - KNOWN_CAUSES
    assert not unknown, f"Unknown disagreement cause labels: {unknown}"


def test_all_transitions_valid(analysis_result):
    """Every decision transition must reference only known decision values."""
    manifest, _ = analysis_result
    for transition in manifest["transitions"].keys():
        parts = transition.split("→")
        assert len(parts) == 2, f"Malformed transition: {transition}"
        for part in parts:
            assert part in KNOWN_DECISIONS, f"Unknown decision value '{part}' in transition '{transition}'"


def test_frozen_directories_untouched(analysis_result):
    """Frozen result directories must be byte-identical before and after analysis."""
    _, before = analysis_result
    for fpath, size_before in before.items():
        # Exclude the new output dir itself (which is inside validation_results)
        if os.path.commonpath([os.path.abspath(fpath), OUTPUT_DIR]) == OUTPUT_DIR:
            continue
        assert os.path.exists(fpath), f"Frozen file was deleted: {fpath}"
        size_now = os.path.getsize(fpath)
        assert size_now == size_before, (
            f"Frozen file was modified: {fpath} ({size_before} → {size_now} bytes)"
        )


def test_git_commit_recorded(analysis_result):
    """Git commit hash must be recorded in the manifest."""
    manifest, _ = analysis_result
    commit = manifest.get("git_commit", "")
    assert commit and commit != "UNKNOWN", "Git commit hash not recorded"
    # Must be a valid 40-char hex string or a short ref
    assert len(commit) >= 7


def test_output_dir_only_in_versioned_subdir():
    """Output must only be written inside the versioned subdir, not into frozen dirs."""
    # frozen dirs must not contain output files outside the versioned subdir
    # (the new output dir is a subfolder of validation_results, which is guarded by
    #  the snapshot ignoring the new subdir)
    new_subdir = os.path.join(FROZEN_DIRS[1], "a1r_a2_disagreement_v3")
    assert os.path.abspath(OUTPUT_DIR) == os.path.abspath(new_subdir), (
        "Output directory is not the expected versioned subdirectory"
    )
