from pathlib import Path


def get_repo_root() -> Path:
    return Path(__file__).parent.parent.parent.parent

def get_results_dir() -> Path:
    return get_repo_root() / "runs"