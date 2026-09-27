import yaml
from pathlib import Path


def load_config(config_path):
    config_path = Path(config_path)

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config