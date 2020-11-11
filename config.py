from pathlib import Path
from typing import Dict


def init_path(path, files) -> (Path, Dict[str, Path]):
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    path_files = {file: path / file for file in files}
    return path, path_files


RAW_PATH, RAW_FILES = init_path("data/raw", ["simplified-recipes-1M.npz"])
PROCESSED_PATH, PROCESSED_FILES = init_path(
    "data/processed", ["recipes_train.txt", "recipes_val.txt"]
)
