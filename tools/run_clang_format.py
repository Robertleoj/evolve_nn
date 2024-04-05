#!/usr/bin/env python3

"""A tool to run clang-format on our source code.

Note that this is assuming this is run from the repo root.
"""
import subprocess
from itertools import chain
from pathlib import Path

repo_paths = ("src", "include")
ignore_folders = ("ViTPose_pytorch",)


def include(path: Path) -> bool:
    """Return True iff this path should be included.

    Each path is checked against the ignore_folders

    Args:
        path: The path to check.
    """
    for ignore_folder in ignore_folders:
        if str(path).find(ignore_folder) != -1:
            return False
    return True


def run_clang_format() -> None:
    """Run Clang format on our CPP files in the repository."""
    assert Path(".git").exists, "This command should run in repo root."

    all_paths = []
    for name in repo_paths:
        source_paths = Path(name).rglob("*.cpp")
        header_paths = Path(name).rglob("*.hpp")
        all_paths.extend([str(el) for el in chain(source_paths, header_paths)])

    all_paths = list(filter(include, all_paths))
    subprocess.run(["clang-format", "-i"] + all_paths)


if __name__ == "__main__":
    run_clang_format()
