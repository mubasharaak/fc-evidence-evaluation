import json


def load_data(path: str):
    """Loads data from path."""
    with open(path, "r", encoding="utf-8") as file:
        test_dataset = json.load(file)

    return test_dataset
