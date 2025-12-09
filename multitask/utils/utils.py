import os.path as path

PATH = "multitask/data_store/"


def path_to_file(filename: str) -> str:
    return path.join(PATH, filename + ".csv")
