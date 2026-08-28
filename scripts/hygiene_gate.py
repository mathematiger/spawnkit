"""Forbidden-string gate for the working tree.

A public repository must not carry the private context it was extracted from. This script walks the
tree and fails the build if it finds a vendor or institution name, a machine-local absolute path, a
cluster scheduler flag, one of the private package names this code used to live next to, or the
common shape of a leaked credential.

The gate is read-only on purpose. There is no ``--fix``: a hit is a judgement call about what the
public wording should be, and a script cannot make it. The same entry point runs locally and in
CI (``.github/workflows/hygiene.yml`` does nothing but call this file), so a green local run and a
green CI run mean exactly the same thing.

Two things are scanned for every file: its **content**, line by line, and its **path**, so that a
file named after a private project is caught even when its text is clean. A path hit is reported
with line number ``0``.

Deliberate exclusions
---------------------
Four entries in :data:`DEFAULT_EXCLUDES` and :data:`PATH_NAME_ALLOWLIST` exist because the file or
path legitimately contains what the gate forbids. Each is a decision, not an oversight:

``scripts/hygiene_gate.py``
    This file holds the pattern list itself. Scanning it would report every pattern against its own
    definition. The two assistant-vendor tokens are additionally assembled from fragments below, so
    that even this file does not spell them out.
``tests/test_hygiene_gate.py``
    The test suite must feed the gate a sample of every pattern class to prove it fires. Those
    samples are test data, and excluding the file is what makes the test possible at all.
The tooling document and the tooling directory (see :data:`PATH_NAME_ALLOWLIST`)
    Their *names* are fixed by an external convention and cannot be renamed. Only the path string is
    exempt; their content is scanned in full, with the single per-file carve-out below.
Home-directory paths inside the tooling document (see :data:`PER_FILE_PATTERN_EXEMPTIONS`)
    That document describes which interpreter to use on the machine it lives on, which is only
    useful as an absolute path. No other file in the tree may name one.

Usage
-----
::

    python scripts/hygiene_gate.py                      # scan the repository root
    python scripts/hygiene_gate.py --root some/subtree
    python scripts/hygiene_gate.py --exclude docs/legacy.md --exclude 'notes/*.md'

Exit codes
----------
``0``
    Nothing found.
``1``
    At least one hit. Every hit is printed as ``path:line: pattern``.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

# The two assistant-vendor tokens are assembled from fragments rather than written out, so that this
# file does not itself contain the strings it forbids. The compiled regex is identical either way.
_VENDOR_TOKENS: tuple[str, ...] = ("cl" + "aude", "anthro" + "pic")

#: Literal substrings, matched case-insensitively anywhere in a line or a path.
FORBIDDEN_LITERALS: tuple[str, ...] = (
    *_VENDOR_TOKENS,
    "co-authored-by",
    "generated with",
    "written with ai",
    "fraunhofer",
    "omnia.cluster",
    "gitlab.cc-asp",
    "/mnt/data",
    "/mnt/home",
    "mail-user",
    "--partition=",
    "ap3_mcts",
    "pandapower",
)

#: Credential shapes, as ``(name, regex)``. These stay case-**sensitive**: the casing is part of the
#: shape, and folding it would turn ``AKIA[0-9A-Z]{16}`` into a match for ordinary hex-looking text.
FORBIDDEN_CREDENTIAL_SHAPES: tuple[tuple[str, str], ...] = (
    ("github-token", r"ghp_[A-Za-z0-9]{20,}"),
    ("aws-access-key-id", r"AKIA[0-9A-Z]{16}"),
    ("private-key-header", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


@dataclass(frozen=True)
class ForbiddenPattern:
    """One forbidden string, named for reporting and compiled for matching.

    Attributes
    ----------
    name
        Human-readable identifier printed with every hit.
    regex
        Compiled expression searched against a single line or a path string.
    """

    name: str
    regex: re.Pattern[str]


def _build_patterns() -> tuple[ForbiddenPattern, ...]:
    """Compile the literal and credential-shape patterns into one ordered tuple.

    Returns
    -------
    tuple of ForbiddenPattern
        Literals first, in declaration order, then the credential shapes.
    """
    literals = (ForbiddenPattern(text, re.compile(re.escape(text), re.IGNORECASE)) for text in FORBIDDEN_LITERALS)
    shapes = (ForbiddenPattern(name, re.compile(regex)) for name, regex in FORBIDDEN_CREDENTIAL_SHAPES)
    return (*literals, *shapes)


#: Every pattern the gate enforces.
PATTERNS: tuple[ForbiddenPattern, ...] = _build_patterns()

# Names fixed by an external tooling convention. They cannot be renamed, so their path string is
# exempt from path scanning -- their content is not.
TOOLING_DOC: str = _VENDOR_TOKENS[0].upper() + ".md"
TOOLING_DIR: str = "." + _VENDOR_TOKENS[0] + "/"

#: Path prefixes exempt from *path-name* scanning. A path under one of these is scanned with the
#: prefix stripped, so ``<tooling-dir>/skills/pandapower.md`` is still caught.
PATH_NAME_ALLOWLIST: tuple[str, ...] = (TOOLING_DOC, TOOLING_DIR)

#: Per-file carve-outs: ``{relative path: pattern names not enforced in that file}``. The tooling
#: document names the interpreters to use on this machine, which is only useful as an absolute path.
PER_FILE_PATTERN_EXEMPTIONS: dict[str, frozenset[str]] = {
    TOOLING_DOC: frozenset({"/mnt/home"}),
}

#: Paths never scanned. Entries ending in ``/`` match a directory *name* anywhere in the tree (glob
#: allowed); other entries match a repository-relative path, a path prefix, or a glob.
DEFAULT_EXCLUDES: tuple[str, ...] = (
    ".git/",
    ".venv/",
    "venv/",
    "__pycache__/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".pytest_cache/",
    ".tox/",
    ".nox/",
    "build/",
    "dist/",
    ".eggs/",
    "*.egg-info/",
    "htmlcov/",
    "node_modules/",
    "scripts/hygiene_gate.py",
    "tests/test_hygiene_gate.py",
)

#: Bytes read to decide whether a file is binary.
_SNIFF_BYTES = 8192


@dataclass(frozen=True)
class Hit:
    """A single forbidden-string occurrence.

    Attributes
    ----------
    path
        Repository-relative path of the offending file.
    line
        1-based line number, or ``0`` when the hit is in the path itself.
    pattern
        Name of the pattern that fired.
    """

    path: str
    line: int
    pattern: str

    def render(self) -> str:
        """Format the hit as ``path:line: pattern``.

        Returns
        -------
        str
            One line of gate output.
        """
        return f"{self.path}:{self.line}: {self.pattern}"


def is_excluded(relative_path: str, excludes: Sequence[str]) -> bool:
    """Report whether a repository-relative path is covered by an exclude entry.

    Parameters
    ----------
    relative_path
        Path relative to the scan root, using forward slashes.
    excludes
        Exclude entries. A trailing ``/`` means "a directory of this name anywhere"; otherwise the
        entry matches an exact path, a path prefix, or a glob over the whole relative path.

    Returns
    -------
    bool
        ``True`` when the path must not be scanned.
    """
    parts = relative_path.split("/")
    for entry in excludes:
        if entry.endswith("/"):
            directory = entry[:-1]
            if any(fnmatch(part, directory) for part in parts[:-1]):
                return True
        elif relative_path == entry or relative_path.startswith(entry + "/") or fnmatch(relative_path, entry):
            return True
    return False


def looks_binary(path: Path) -> bool:
    """Report whether a file should be treated as binary and skipped.

    A NUL byte in the first :data:`_SNIFF_BYTES`, or an unreadable file, counts as binary.

    Parameters
    ----------
    path
        File to sniff.

    Returns
    -------
    bool
        ``True`` when the file must not be scanned as text.
    """
    try:
        with path.open("rb") as handle:
            return b"\x00" in handle.read(_SNIFF_BYTES)
    except OSError:
        return True


def _read_text(path: Path) -> str | None:
    """Read a file as UTF-8, returning ``None`` when it is not decodable text.

    Parameters
    ----------
    path
        File to read.

    Returns
    -------
    str or None
        The decoded text, or ``None`` for undecodable or unreadable files.
    """
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _active_patterns(relative_path: str, patterns: Sequence[ForbiddenPattern]) -> tuple[ForbiddenPattern, ...]:
    """Drop the patterns carved out for this particular file.

    Parameters
    ----------
    relative_path
        Path relative to the scan root.
    patterns
        Full pattern set.

    Returns
    -------
    tuple of ForbiddenPattern
        The patterns enforced in this file.
    """
    exempt = PER_FILE_PATTERN_EXEMPTIONS.get(relative_path, frozenset())
    if not exempt:
        return tuple(patterns)
    return tuple(pattern for pattern in patterns if pattern.name not in exempt)


def _scannable_path_fragment(relative_path: str) -> str:
    """Strip an allowlisted prefix from a path before scanning the path itself.

    Parameters
    ----------
    relative_path
        Path relative to the scan root.

    Returns
    -------
    str
        The part of the path that is subject to path scanning; empty when the whole path is exempt.
    """
    for allowed in PATH_NAME_ALLOWLIST:
        if relative_path == allowed:
            return ""
        if allowed.endswith("/") and relative_path.startswith(allowed):
            return relative_path[len(allowed) :]
    return relative_path


def scan_path_name(relative_path: str, patterns: Sequence[ForbiddenPattern]) -> Iterator[Hit]:
    """Yield hits found in the path string itself.

    Parameters
    ----------
    relative_path
        Path relative to the scan root.
    patterns
        Patterns to enforce.

    Yields
    ------
    Hit
        One hit per matching pattern, with ``line == 0``.
    """
    fragment = _scannable_path_fragment(relative_path)
    if not fragment:
        return
    for pattern in patterns:
        if pattern.regex.search(fragment):
            yield Hit(relative_path, 0, pattern.name)


def scan_text(text: str, relative_path: str, patterns: Sequence[ForbiddenPattern]) -> Iterator[Hit]:
    """Yield hits found in file content.

    Parameters
    ----------
    text
        Full file content.
    relative_path
        Path relative to the scan root, used only for reporting.
    patterns
        Patterns to enforce.

    Yields
    ------
    Hit
        One hit per matching pattern per line.
    """
    for number, line in enumerate(text.splitlines(), start=1):
        for pattern in patterns:
            if pattern.regex.search(line):
                yield Hit(relative_path, number, pattern.name)


def scan_file(path: Path, relative_path: str, patterns: Sequence[ForbiddenPattern] = PATTERNS) -> list[Hit]:
    """Scan one file's path and content.

    Parameters
    ----------
    path
        File on disk.
    relative_path
        Path relative to the scan root.
    patterns
        Patterns to enforce; defaults to :data:`PATTERNS`.

    Returns
    -------
    list of Hit
        Path hits first, then content hits. Binary files yield content hits of none.
    """
    active = _active_patterns(relative_path, patterns)
    hits = list(scan_path_name(relative_path, active))
    if looks_binary(path):
        return hits
    text = _read_text(path)
    if text is None:
        return hits
    hits.extend(scan_text(text, relative_path, active))
    return hits


def iter_files(root: Path, excludes: Sequence[str]) -> Iterator[tuple[Path, str]]:
    """Walk the tree, yielding every file that is not excluded.

    Parameters
    ----------
    root
        Directory to walk.
    excludes
        Exclude entries, as understood by :func:`is_excluded`.

    Yields
    ------
    tuple of (Path, str)
        The file on disk and its path relative to ``root``.
    """
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative_path = path.relative_to(root).as_posix()
        if is_excluded(relative_path, excludes):
            continue
        yield path, relative_path


def scan_tree(
    root: Path,
    excludes: Iterable[str] | None = None,
    patterns: Sequence[ForbiddenPattern] = PATTERNS,
) -> list[Hit]:
    """Scan a whole tree for forbidden strings.

    Parameters
    ----------
    root
        Directory to scan.
    excludes
        Extra exclude entries appended to :data:`DEFAULT_EXCLUDES`; ``None`` means defaults only.
    patterns
        Patterns to enforce; defaults to :data:`PATTERNS`.

    Returns
    -------
    list of Hit
        Every hit found, in path order.
    """
    all_excludes = (*DEFAULT_EXCLUDES, *(excludes or ()))
    hits: list[Hit] = []
    for path, relative_path in iter_files(root, all_excludes):
        hits.extend(scan_file(path, relative_path, patterns))
    return hits


def _repository_root() -> Path:
    """Locate the repository root relative to this file.

    Returns
    -------
    Path
        The directory containing ``scripts/``.
    """
    return Path(__file__).resolve().parents[1]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line.

    Parameters
    ----------
    argv
        Argument vector, or ``None`` to read ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed ``root`` and ``exclude`` values.
    """
    parser = argparse.ArgumentParser(
        prog="hygiene_gate",
        description="Fail if a forbidden string appears anywhere in the tree. Read-only; there is no --fix.",
    )
    parser.add_argument("--root", type=Path, default=_repository_root(), help="directory to scan (default: repo root)")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PATH",
        help="extra path, path prefix or glob to skip; repeatable",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the gate and report.

    Parameters
    ----------
    argv
        Argument vector, or ``None`` to read ``sys.argv``.

    Returns
    -------
    int
        ``0`` when the tree is clean, ``1`` when anything was found.
    """
    args = _parse_args(argv)
    hits = scan_tree(args.root, args.exclude)
    if not hits:
        print(f"hygiene gate: clean ({args.root})")
        return 0
    for hit in hits:
        print(hit.render())
    print(f"hygiene gate: {len(hits)} forbidden string(s) found")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
