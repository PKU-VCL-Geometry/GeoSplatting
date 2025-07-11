from pathlib import Path


def get_last_modified_time(file: Path) -> float:
    assert file.exists(), f"Path {file} does not exist."
    assert file.is_file(), f"Path {file} is not a file."
    return file.stat().st_mtime
