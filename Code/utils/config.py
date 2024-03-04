import os
import json
#from dotmap import DotMap

def get_config_from_json(json_file):
    with open(json_file, "r") as config_file:
        config = json.load(config_file)

    return config

def load_config(config_filepath):
    assert os.path.exists(config_filepath), "Unexistent config file."

    config = get_config_from_json(config_filepath)
    return config

def save_config(output_path, config_dict):
    with open(output_path, "w") as json_file:
        json.dump(config_dict, json_file, indent=4)
