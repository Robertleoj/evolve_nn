#!/usr/bin/env python3

"""A script to build the C++ backend and install the bindings into the source tree."""

import os
import shutil
import subprocess
from pathlib import Path

from fire import Fire

BUILD_DIR = "build"

MODULE_NAME = "foundation.cpython-310-x86_64-linux-gnu.so"


def check_in_repo() -> None:
    """Check that we are executing this from repo root."""
    assert Path(".git").exists(), "This command should run in repo root."


def build() -> None:
    """(Re)build the C++ backend."""
    check_in_repo()

    build_path = Path("build")
    build_path.mkdir(exist_ok=True)

    subprocess.run(["cmake", "-B", str(build_path), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"], check=True)

    subprocess.run(["ninja", "-C", str(build_path)])

    # Make sure that target was built
    target_path = build_path / "src" / "project" / "foundation" / MODULE_NAME
    assert target_path.exists()

    # Replace or create symlink
    deploy_path = Path("src/project") / MODULE_NAME
    if deploy_path.is_symlink():
        deploy_path.unlink()

    deploy_path.symlink_to(target_path.resolve())

    subprocess.run(
        ["stubgen", "-p", "foundation", "-o", "./src/project", "--include-docstring"],
        env=dict(os.environ, PYTHONPATH="./src/project"),
    )


def clean() -> None:
    """Clean the build folder and remove the symlink, if any."""
    check_in_repo()
    shutil.rmtree(BUILD_DIR, ignore_errors=True)

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
