"""
End-to-end smoke demo for the GTNH_Agent project.

Usage (from repo root, after setting up a virtualenv):

    (.venv) pip install -e ".[dev]"
    (.venv) python -m scripts.demo_end_to_end

What this does:

  1. Verifies you're running from the project root and on a supported Python version.
  2. Checks that pytest is available (installed via the dev extras).
  3. Runs a curated subset of tests that exercise:
       - environment loading / config bridge
       - semantics ingestion / tech graph
       - skills registry + packs wiring
       - full agent loop smoke / lab integration path

If everything passes, you get a short success message.
If something fails, you get a clear pointer instead of a cryptic traceback.

This is meant as:
  - A "does this repo actually work?" button for reviewers.
  - A sanity check for you after cloning or changing machines.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


def _print_header() -> None:
    print("=" * 72)
    print(" GTNH_Agent · End-to-End Demo / Smoke Runner")
    print("=" * 72)
    print()


def _check_python_version() -> bool:
    required = (3, 10)
    if sys.version_info < required:
        print(
            f"[FAIL] Python {required[0]}.{required[1]}+ required; "
            f"found {sys.version_info.major}.{sys.version_info.minor}."
        )
        print("       Create a 3.10+ virtualenv and try again.")
        return False
    print(f"[OK] Python version: {sys.version_info.major}.{sys.version_info.minor}")
    return True


def _check_project_root() -> bool:
    """Heuristically ensure we're running from the repo root."""
    cwd = Path.cwd()
    pyproject = cwd / "pyproject.toml"
    src_dir = cwd / "src"
    tests_dir = cwd / "tests"

    if not pyproject.exists() or not src_dir.exists() or not tests_dir.exists():
        print("[WARN] This does not look like the project root.")
        print("       Expected to find pyproject.toml, src/, and tests/ in current dir.")
        print("       cd into the repo root and re-run:")
        print("           python -m scripts.demo_end_to_end")
        return False

    print(f"[OK] Project root: {cwd}")
    return True


def _check_pytest() -> bool:
    try:
        import pytest  # type: ignore  # noqa: F401
    except ImportError:
        print("[FAIL] pytest is not installed in this environment.")
        print("       Install dev dependencies first:")
        print('           pip install -e ".[dev]"')
        return False

    print("[OK] pytest available")
    return True


def _run_tests(test_paths: Sequence[str]) -> int:
    """Run a focused subset of tests that exercise the main integration paths."""
    import pytest

    print()
    print("Running end-to-end smoke tests via pytest...")
    print("------------------------------------------------")
    for path in test_paths:
        print(f"  - {path}")
    print("------------------------------------------------")
    print()

    # Run pytest programmatically.
    # Return code semantics:
    #   0 = all good
    #   1 = tests failed
    #   2+ = pytest/internal error
    return pytest.main(list(test_paths))


def main() -> int:
    _print_header()

    ok = True
    ok &= _check_python_version()
    ok &= _check_project_root()
    ok &= _check_pytest()

    if not ok:
        print()
        print("[ABORT] Demo prerequisites not met. Fix the issues above and re-run.")
        return 1

    # Curated subset of tests that:
    #   - prove config/env/semantics wiring
    #   - validate skills + curriculum
    #   - run lab/agent smoke paths
    test_subset = [
        "tests/test_phase012_bootstrap.py",
        "tests/test_env_loader.py",
        "tests/test_semantics_categorization.py",
        "tests/test_skill_registry.py",
        "tests/test_full_system_smoke.py",
        "tests/test_lab_integration_happy_path.py",
    ]

    # Filter to only those that actually exist, in case the test suite evolves.
    existing = [p for p in test_subset if Path(p).exists()]
    if not existing:
        print("[FAIL] None of the expected smoke-test files were found.")
        print("       Make sure you're on a recent version of the repo, ")
        print("       or update scripts/demo_end_to_end.py to match your test layout.")
        return 1

    exit_code = _run_tests(existing)

    print()
    if exit_code == 0:
        print("✅ End-to-end demo passed.")
        print("   Core configuration, semantics, skills, and lab integration are healthy.")
    else:
        print("❌ Demo failed.")
        print("   At least one smoke test failed. See pytest output above for details.")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

