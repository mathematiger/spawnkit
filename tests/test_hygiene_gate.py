"""Tests for ``scripts/hygiene_gate.py``.

This file is on the gate's own exclude list (``DEFAULT_EXCLUDES``), and that is deliberate: proving
the gate fires requires feeding it a sample of every pattern class, so the samples below are exactly
the strings the gate exists to reject. Excluding the test file is what makes the test possible.

The two assistant-vendor tokens are still assembled from fragments rather than written out, so that
the samples stay readable without spelling those tokens anywhere in the repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import hygiene_gate  # noqa: E402
import pytest  # noqa: E402

VENDOR_A = "cl" + "aude"
VENDOR_B = "Anthro" + "pic"

# One sample per pattern class, paired with the pattern name the gate should report.
PATTERN_SAMPLES: list[tuple[str, str]] = [
    (f"see the {VENDOR_A} notes", VENDOR_A),
    (f"vendored by {VENDOR_B} Inc.", "anthro" + "pic"),
    ("Co-Authored-By: someone <a@b.c>", "co-authored-by"),
    ("This file was generated with a tool", "generated with"),
    ("Written with AI assistance", "written with ai"),
    ("(c) Fraunhofer Institute", "fraunhofer"),
    ("host node0.omnia.cluster is down", "omnia.cluster"),
    ("git clone https://gitlab.cc-asp.example/x.git", "gitlab.cc-asp"),
    ("data lives in /mnt/data/runs", "/mnt/data"),
    ("interpreter at /mnt/home/user/.venv/bin/python", "/mnt/home"),
    ("#SBATCH --mail-user=someone@example.org", "mail-user"),
    ("sbatch --partition=queue job.sh", "--partition="),
    ("from ap3_mcts.worker_runtime import x", "ap3_mcts"),
    ("import pandapower as pp", "pandapower"),
    ("token = ghp_" + "A" * 24, "github-token"),
    ("aws_key = AKIA" + "ABCDEFGHIJKLMNOP", "aws-access-key-id"),
    ("-----BEGIN RSA PRIVATE KEY-----", "private-key-header"),
]

CLEAN_FILES: dict[str, str] = {
    "README.md": "# spawnkit\n\nThe layer below the trainer.\n",
    "src/pkg/__init__.py": '"""A module."""\n\nVALUE = 1\n',
    "docs/guide.md": "Run the worker monitor and it stops the run on a death.\n",
}


def _write_tree(root: Path, files: dict[str, str]) -> None:
    """Materialise ``{relative path: content}`` under ``root``."""
    for relative_path, content in files.items():
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def test_clean_tree_passes(tmp_path: Path) -> None:
    _write_tree(tmp_path, CLEAN_FILES)
    assert hygiene_gate.scan_tree(tmp_path) == []


def test_clean_tree_exits_zero(tmp_path: Path) -> None:
    _write_tree(tmp_path, CLEAN_FILES)
    assert hygiene_gate.main(["--root", str(tmp_path)]) == 0


@pytest.mark.parametrize(("sample", "pattern_name"), PATTERN_SAMPLES, ids=[name for _, name in PATTERN_SAMPLES])
def test_each_pattern_class_is_caught(tmp_path: Path, sample: str, pattern_name: str) -> None:
    _write_tree(tmp_path, {"notes.md": f"harmless line\n{sample}\nanother harmless line\n"})
    hits = hygiene_gate.scan_tree(tmp_path)
    assert [hit.pattern for hit in hits] == [pattern_name]
    assert hits[0].path == "notes.md"
    assert hits[0].line == 2
    assert hits[0].render() == f"notes.md:2: {pattern_name}"


def test_literals_match_case_insensitively(tmp_path: Path) -> None:
    _write_tree(tmp_path, {"notes.md": "FRAUNHOFER\n"})
    assert [hit.pattern for hit in hygiene_gate.scan_tree(tmp_path)] == ["fraunhofer"]


def test_credential_shapes_stay_case_sensitive(tmp_path: Path) -> None:
    _write_tree(tmp_path, {"notes.md": "akia" + "abcdefghijklmnop" + "\n"})
    assert hygiene_gate.scan_tree(tmp_path) == []


def test_hit_in_a_path_is_reported_at_line_zero(tmp_path: Path) -> None:
    _write_tree(tmp_path, {"notes/pandapower_setup.md": "entirely innocent content\n"})
    hits = hygiene_gate.scan_tree(tmp_path)
    assert [(hit.path, hit.line, hit.pattern) for hit in hits] == [
        ("notes/pandapower_setup.md", 0, "pandapower"),
    ]


def test_explicit_exclude_suppresses_a_hit(tmp_path: Path) -> None:
    _write_tree(tmp_path, {"legacy/notes.md": "(c) Fraunhofer\n"})
    assert hygiene_gate.scan_tree(tmp_path) != []
    assert hygiene_gate.scan_tree(tmp_path, ["legacy/notes.md"]) == []
    assert hygiene_gate.scan_tree(tmp_path, ["legacy/"]) == []
    assert hygiene_gate.scan_tree(tmp_path, ["legacy/*.md"]) == []


def test_exclude_flag_on_the_command_line(tmp_path: Path) -> None:
    _write_tree(tmp_path, {"legacy/notes.md": "(c) Fraunhofer\n"})
    assert hygiene_gate.main(["--root", str(tmp_path)]) == 1
    assert hygiene_gate.main(["--root", str(tmp_path), "--exclude", "legacy/"]) == 0


def test_default_excludes_skip_tooling_caches(tmp_path: Path) -> None:
    _write_tree(
        tmp_path,
        {
            ".venv/lib/site-packages/x.py": "import pandapower\n",
            "__pycache__/x.py": "import pandapower\n",
            ".mypy_cache/3.12/x.json": '{"note": "/mnt/data"}',
            "pkg.egg-info/PKG-INFO": "Author: Fraunhofer\n",
        },
    )
    assert hygiene_gate.scan_tree(tmp_path) == []


def test_binary_file_does_not_crash_the_gate(tmp_path: Path) -> None:
    binary = tmp_path / "blob.bin"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"\x00\x01\x02fraunhofer\x00\xff\xfe/mnt/data")
    _write_tree(tmp_path, CLEAN_FILES)
    assert hygiene_gate.scan_tree(tmp_path) == []


def test_undecodable_text_file_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "latin1.txt").write_bytes("Fraunhöfer".encode("latin-1"))
    assert hygiene_gate.scan_tree(tmp_path) == []


def test_tooling_paths_are_exempt_from_path_scanning(tmp_path: Path) -> None:
    _write_tree(
        tmp_path,
        {
            hygiene_gate.TOOLING_DOC: "Repository conventions.\n",
            hygiene_gate.TOOLING_DIR + "skills/release/SKILL.md": "Release procedure.\n",
        },
    )
    assert hygiene_gate.scan_tree(tmp_path) == []


def test_tooling_directory_still_scans_the_rest_of_the_path(tmp_path: Path) -> None:
    _write_tree(tmp_path, {hygiene_gate.TOOLING_DIR + "skills/pandapower.md": "clean content\n"})
    assert [hit.pattern for hit in hygiene_gate.scan_tree(tmp_path)] == ["pandapower"]


def test_tooling_document_content_is_still_scanned(tmp_path: Path) -> None:
    _write_tree(tmp_path, {hygiene_gate.TOOLING_DOC: "(c) Fraunhofer\nimport pandapower\n"})
    assert sorted(hit.pattern for hit in hygiene_gate.scan_tree(tmp_path)) == ["fraunhofer", "pandapower"]


def test_tooling_document_may_name_home_paths(tmp_path: Path) -> None:
    home_path_line = "Use /mnt/home/user/project/.venv/bin/python for the clean-install gate.\n"
    _write_tree(tmp_path, {hygiene_gate.TOOLING_DOC: home_path_line})
    assert hygiene_gate.scan_tree(tmp_path) == []


def test_home_path_exemption_does_not_leak_to_other_files(tmp_path: Path) -> None:
    home_path_line = "Use /mnt/home/user/project/.venv/bin/python for the clean-install gate.\n"
    _write_tree(tmp_path, {"docs/setup.md": home_path_line})
    assert [hit.pattern for hit in hygiene_gate.scan_tree(tmp_path)] == ["/mnt/home"]


def test_every_declared_pattern_class_has_a_sample() -> None:
    covered = {name for _, name in PATTERN_SAMPLES}
    declared = {pattern.name for pattern in hygiene_gate.PATTERNS}
    assert declared == covered


def test_repository_tree_is_clean() -> None:
    assert [hit.render() for hit in hygiene_gate.scan_tree(REPO_ROOT)] == []
