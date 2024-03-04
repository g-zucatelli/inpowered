import os
import pickle
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
plt.ioff()

from models.bert_mlp import bert_mlp

from utils.args import get_args
from utils.config import load_config


def save_pickle(filename, file):
    filehandler = open(filename, "wb")
    pickle.dump(file, filehandler)
    filehandler.close()

def load_pickle(filename):
    file = open(filename,'rb')
    object_file = pickle.load(file)
    file.close()
    return object_file


def save_scatter_prediction(config, predictions, test_true, _flag, MAX=200):
    fig = plt.figure(figsize=(5,3))
    plt.scatter(predictions, test_true, alpha=0.4)

    plt.plot(np.array(range(MAX)), np.array(range(MAX)), '--', alpha=0.4)

    _target_output = _flag.title()
    plt.xlabel(f"Prediction {_target_output}"); plt.ylabel(f"True {_target_output}")
    plt.xlim([0, MAX]); plt.ylim([0, MAX])

    img_path = os.path.join("experiments", config["name"], f"results/scatter_{_flag}.pdf")
    plt.savefig(img_path, bbox_inches='tight')
    plt.close(fig)

def select_keys(encoded_values, _keys):
    filtered_encoded_values = {_key: encoded_values[_key] for _key in _keys} 
    encoded_indexes = [list(encoded_values.keys()).index(_key) for _key in _keys]
    return filtered_encoded_values, encoded_indexes


def test_target_model(config, encoded_values_indexes, _flag):
    encoded_values, encoded_indexes = encoded_values_indexes

    model_path = os.path.join("experiments", config["name"], f"model_{_flag}/model.keras")

    if not config["train"]["use_bert"]:
        model = tf.keras.models.load_model(model_path)
    else:
        model = bert_mlp(config, encoded_values)
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
        
        model.load_weights(model_path.replace(".keras", ".hdf5"))
    

    data_path = config["pre_process"]["processed_data_path"]
    pkl_data = load_pickle(data_path)

    test_num, test_cat, test_bert, test_true = pkl_data[1]
    
    test_cat = test_cat[:, encoded_indexes]
    test_true = test_true[_flag]

    test_cat_in = [test_cat[:, idx] for idx in range(test_cat.shape[1])]
    test_data = [test_num]
    test_data.extend(test_cat_in)

    if config["train"]["use_bert"]:
        test_data.append(test_bert)

    print("Evaluating on test data")
    results = model.evaluate(test_data, test_true)
    predictions = model.predict(test_data)

    print(f"\n{config['name']} \n{_flag.upper()}: Test Loss:", results, '\n')
    save_scatter_prediction(config, predictions, test_true, _flag)
    


def main(config_filepath):
    config = load_config(config_filepath)
    desired_models = config["pre_process"]["target_output"]

    for _flag in desired_models:
        encoded_values_path = os.path.join("experiments", config["name"], "encoded_values.pkl")
        encoded_values = load_pickle(encoded_values_path)
        
        if _flag == "engagements":
            _keys = config["pre_process"]["engage_cat_input"]
        elif _flag == "media_spend":
            _keys = config["pre_process"]["spend_cat_input"]
        else:
            _keys = list(encoded_values.keys())
        
        encoded_values_indexes = select_keys(encoded_values, _keys)

        test_target_model(config, encoded_values_indexes, _flag)




if __name__ == "__main__":
    ARGS = get_args()
    config_filepath = ARGS.config_filepath
    main(config_filepath)  
