#!/usr/bin/env python3

"""A script to build the C++ backend and install the bindings into the source tree."""

import os
import shutil
import subprocess
from pathlib import Path

from fire import Fire

BUILD_DIR = "build"

MODULE_NAME = "foundation.cpython-310-x86_64-linux-gnu.so"


def has_executable(path):
    for item in path.rglob("*"):
        if item.is_file() and os.access(item, os.X_OK):
            return True
    return False


def symlink_executables(source_dir, target_dir):
    source_dir = Path(source_dir).resolve()
    target_dir = Path(target_dir).resolve()

    shutil.rmtree(target_dir, ignore_errors=True)

    if not source_dir.is_dir():
        raise ValueError("Source directory does not exist or is not a directory")

    for item in source_dir.rglob("*"):
        if item.is_dir() and has_executable(item):
            (target_dir / item.relative_to(source_dir)).mkdir(parents=True, exist_ok=True)
        elif item.is_file() and os.access(item, os.X_OK):
            target_item = target_dir / item.relative_to(source_dir)
            target_item.parent.mkdir(parents=True, exist_ok=True)
            target_item.symlink_to(item)


def check_in_repo() -> None:
    """Check that we are executing this from repo root."""
    assert Path(".git").exists(), "This command should run in repo root."


def build() -> None:
    """(Re)build the C++ backend."""
    check_in_repo()

    build_path = Path("build")
    build_path.mkdir(exist_ok=True)

    subprocess.run(["cmake", "-B", str(build_path), "-G", "Ninja"], check=True)

    subprocess.run(["ninja", "-C", str(build_path)])

    # Make sure that target was built
    target_path = build_path / "src" / "project" / "foundation" / MODULE_NAME
    assert target_path.exists()

    # Replace or create symlink
    deploy_path = Path("src/project") / MODULE_NAME
    if deploy_path.is_symlink():
        deploy_path.unlink()

    deploy_path.symlink_to(target_path.resolve())

    cpp_script_path = build_path / "cpp_scripts"
    cpp_executables_path = Path("cpp_executables")

    symlink_executables(cpp_script_path, cpp_executables_path)

    subprocess.run(
        ["stubgen", "-p", "foundation", "-o", "./src/project", "--include-docstring"],
        env=dict(os.environ, PYTHONPATH="./src/project"),
    )

def clean() -> None:
    """Clean the build folder and remove the symlink, if any."""
    check_in_repo()
    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    shutil.rmtree("cpp_executables", ignore_errors=True)

    # Remove the symlink, if any
    deploy_path = Path(f"src/project/{MODULE_NAME}")
    if deploy_path.is_symlink():
        deploy_path.unlink()


def clean_build() -> None:
    """First clean and then build."""
    clean()
    build()


if __name__ == "__main__":
    Fire({"build": build, "clean": clean, "clean_build": clean_build})
