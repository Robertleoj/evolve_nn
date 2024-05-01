from pathlib import Path
from datetime import datetime


def get_repo_root() -> Path:
    return Path(__file__).parent.parent.parent.parent


def get_results_dir() -> Path:
    return get_repo_root() / "runs"

def get_date_path() -> str:
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y%m%d_%H%M%S")
    return folder_name

