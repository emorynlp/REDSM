import pyhocon
import sys, os


def parse_configs(config_path):
    return pyhocon.ConfigFactory.parse_file(config_path)


def get_config(config_name, config_path, create_dir=True):
    config = parse_configs(config_path)[config_name]
    if create_dir:
        os.makedirs(
            os.path.join(
                config["save_dir"], config["dataset_name"], config["pretrained"]
            ),
            exist_ok=True,
        )
    return config
