#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path


def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def snapshot_repo(repo: Path) -> dict[str, list[str]]:
    """
    Capture repository files as:
        {"path/to/file.py": ["line 1", "line 2", ...]}

    This matches the T4 before/after schema.
    """
    files = run_git(repo, "ls-files", "-z").split("\0")

    snapshot = {}

    for relative_path in files:
        if not relative_path:
            continue

        path = repo / relative_path

        if not path.is_file():
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        snapshot[relative_path] = text.splitlines()

    return snapshot


def git_diff(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "diff", "--no-ext-diff"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture T4 repository snapshots."
    )

    parser.add_argument(
        "--repo",
        required=True,
        help="Path to the repository being measured.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path.",
    )

    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    output = Path(args.output).resolve()

    if not (repo / ".git").exists():
        raise SystemExit(f"Not a Git repository: {repo}")

    commit = run_git(repo, "rev-parse", "HEAD").strip()

    snapshot = snapshot_repo(repo)

    result = {
        "commit": commit,
        "snapshot": snapshot,
        "diff": git_diff(repo),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Repository: {repo}")
    print(f"Commit: {commit}")
    print(f"Files captured: {len(snapshot)}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()