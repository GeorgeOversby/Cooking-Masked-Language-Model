from doit.action import CmdAction

from config import PROCESSED_FILES, RAW_FILES


def run_with_output(shell_command):
    return {"actions": [CmdAction(shell_command, buffering=1)], "verbosity": 2}


def task_install_dependencies():
    return {
        **run_with_output("bash 01_install_dependencies.sh"),
        "file_dep": ["01_install_dependencies.sh"],
        "targets": [],
    }


def task_run_tests():
    return {
        **run_with_output("python -m pytest ."),
        "file_dep": [],
        "targets": [],
    }


def task_formatting():
    return {
        **run_with_output("black ."),
        "file_dep": [],
        "targets": [],
    }


def task_load_data():
    return {
        **run_with_output("bash 02_download_data.sh"),
        "file_dep": ["02_download_data.sh"],
        "targets": list(RAW_FILES.values()),
    }


def task_process_data():
    return {
        **run_with_output("python 03_process_data.py"),
        "file_dep": list(RAW_FILES.values()) + ["03_process_data.py"],
        "targets": list(PROCESSED_FILES.values()),
    }


def task_train_model():
    return {
        **run_with_output("python 04_train_model.py"),
        "file_dep": list(PROCESSED_FILES.values()) + ["04_train_model.py"],
        "targets": [],
    }
