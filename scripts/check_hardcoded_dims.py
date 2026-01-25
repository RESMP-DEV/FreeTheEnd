#!/usr/bin/env python3
"""Pre-commit hook to detect hardcoded observation/action dimensions.

This script scans Python files for suspicious hardcoded dimension values
that should instead use constants from minecraft_sim.constants.

Usage:
    python scripts/check_hardcoded_dims.py [files...]

    # Or as a pre-commit hook (see .pre-commit-config.yaml)

Exit codes:
    0 - No issues found
    1 - Suspicious patterns detected
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple


class Issue(NamedTuple):
    """A detected issue in a file."""
    file: Path
    line_num: int
    line: str
    pattern: str
    message: str


# =============================================================================
# Patterns to detect
# =============================================================================

# Each tuple: (regex_pattern, description, severity)
# Severity: "error" = must fix, "warning" = review recommended
SUSPICIOUS_PATTERNS: list[tuple[str, str, str]] = [
    # Hardcoded obs_dim=32 (should be 48)
    (
        r'\bobs_dim\s*[=:]\s*32\b',
        "Hardcoded obs_dim=32. Main observation is 48 floats. "
        "Use OBSERVATION_SIZE from constants.py, or PROGRESS_OBSERVATION_SIZE (32) "
        "if this is specifically for progress tracking.",
        "error",
    ),

    # Hardcoded shape=(32,) for observations
    (
        r'shape\s*=\s*\(\s*32\s*,?\s*\)',
        "Hardcoded shape=(32,). If this is the main observation, use (OBSERVATION_SIZE,). "
        "If this is progress tracking, use (PROGRESS_OBSERVATION_SIZE,).",
        "warning",
    ),

    # Hardcoded reshape to 32
    (
        r'\.reshape\s*\([^)]*,\s*32\s*\)',
        "Hardcoded reshape(..., 32). Use OBSERVATION_SIZE or PROGRESS_OBSERVATION_SIZE.",
        "warning",
    ),

    # np.zeros(32, ...) that might be observation-related
    (
        r'np\.zeros\s*\(\s*32\s*[,)]',
        "Hardcoded np.zeros(32). If this is an observation buffer, "
        "use OBSERVATION_SIZE or PROGRESS_OBSERVATION_SIZE.",
        "warning",
    ),

    # Hardcoded observation_size=32
    (
        r'\bobservation_size\s*[=:]\s*32\b',
        "Hardcoded observation_size=32. Main observation is 48 floats. "
        "Use OBSERVATION_SIZE from constants.py.",
        "error",
    ),

    # Note: We don't flag hardcoded 48 since that's the correct value.
    # The issue is when 32 is used instead of 48.
]

# Paths to ignore (relative to repo root)
IGNORE_PATTERNS: list[str] = [
    "tests/test_progression.py",  # Legitimately tests 32-dim progress vector
    "tests/test_fallback.py",     # Tests may check specific dimensions
    "tests/test_determinism.py",  # Tests may check specific dimensions
    "progression.py",             # Defines the 32-dim progress observation
    "constants.py",               # Defines the constants themselves
    ".venv/",                     # Virtual environment
    "build/",                     # Build artifacts
    "__pycache__/",               # Cache
    ".git/",                      # Git directory
]


# =============================================================================
# Detection logic
# =============================================================================


def should_ignore(file_path: Path) -> bool:
    """Check if a file should be ignored."""
    path_str = str(file_path)
    return any(pattern in path_str for pattern in IGNORE_PATTERNS)


def check_file(file_path: Path) -> list[Issue]:
    """Check a single file for suspicious patterns."""
    if should_ignore(file_path):
        return []

    if not file_path.suffix == ".py":
        return []

    try:
        content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    issues: list[Issue] = []
    lines = content.splitlines()

    for pattern, message, severity in SUSPICIOUS_PATTERNS:
        for i, line in enumerate(lines, start=1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # Skip lines that import or define constants
            if "OBSERVATION_SIZE" in line or "PROGRESS_OBSERVATION_SIZE" in line:
                if "import" in line or "=" in line.split("#")[0]:
                    continue

            if re.search(pattern, line):
                issues.append(Issue(
                    file=file_path,
                    line_num=i,
                    line=line.strip(),
                    pattern=pattern,
                    message=f"[{severity.upper()}] {message}",
                ))

    return issues


def check_files(files: list[Path]) -> list[Issue]:
    """Check multiple files for issues."""
    all_issues: list[Issue] = []

    for file_path in files:
        if file_path.is_file():
            all_issues.extend(check_file(file_path))
        elif file_path.is_dir():
            for py_file in file_path.rglob("*.py"):
                all_issues.extend(check_file(py_file))

    return all_issues


def format_issue(issue: Issue) -> str:
    """Format an issue for display."""
    return (
        f"{issue.file}:{issue.line_num}: {issue.message}\n"
        f"    {issue.line}"
    )


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check for hardcoded dimension values that should use constants."
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        default=[Path(".")],
        help="Files or directories to check (default: current directory)",
    )
    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="Only report errors, not warnings",
    )
    args = parser.parse_args()

    issues = check_files(args.files)

    if args.errors_only:
        issues = [i for i in issues if "[ERROR]" in i.message]

    if issues:
        print(f"Found {len(issues)} issue(s):\n")
        for issue in issues:
            print(format_issue(issue))
            print()

        # Count errors vs warnings
        errors = sum(1 for i in issues if "[ERROR]" in i.message)
        warnings = len(issues) - errors

        print(f"Summary: {errors} error(s), {warnings} warning(s)")

        if errors > 0:
            print("\nPlease use constants from minecraft_sim.constants:")
            print("    from minecraft_sim.constants import OBSERVATION_SIZE, PROGRESS_OBSERVATION_SIZE")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
