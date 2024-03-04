import os
from utils.args import get_args
from utils.config import load_config, save_config

def make_dirs(config):
    EXPERIMENT_DIR = os.path.join("experiments", config["name"])
    ENG_MODEL_DIR = os.path.join(EXPERIMENT_DIR, "model_engagements")
    SPD_MODEL_DIR = os.path.join(EXPERIMENT_DIR, "model_media_spend")
    LOGS_DIR = os.path.join(EXPERIMENT_DIR, "logs")
    RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
    
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    os.makedirs(ENG_MODEL_DIR, exist_ok=True)
    os.makedirs(SPD_MODEL_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    save_config_path = os.path.join(EXPERIMENT_DIR, "config.json")
    save_config(save_config_path, config)

    # for idx in range(config["pre_process"]["kfolds"]):
    #     dir_path = os.path.join(MODEL_DIR, f"fold{idx}")
    #     os.makedirs(dir_path, exist_ok=True)


def main(config_filepath):
    config = load_config(config_filepath)
    make_dirs(config)

if __name__ == "__main__":
    ARGS = get_args()
    config_filepath = ARGS.config_filepath
    main(config_filepath)    



