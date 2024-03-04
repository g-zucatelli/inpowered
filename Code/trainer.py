import os
import pickle
import numpy as np
import tensorflow as tf

from models.mlp import emb_mlp
from models.bert_mlp import bert_mlp

from utils.args import get_args
from utils.config import load_config

NUMERICAL_KEYS = ['bid', 'budget', 'engagements','page_views', 'clicks', 
                  'active_days', 'media_spend', 'media_cpc','cpe']

CATEGORICAL_KEYS = ['group', 'item', 'channel', 'date', 'headline', 'storySummary',
                    'IABCategory', 'targetGeo', 'targetInterest', 'targetAge', 'targetOs', 
                    'targetDevices','targetGender', 'targetLanguages', 'CATEGORY_1']


def save_pickle(filename, file):
    filehandler = open(filename, "wb")
    pickle.dump(file, filehandler)
    filehandler.close()

def load_pickle(filename):
    file = open(filename,'rb')
    object_file = pickle.load(file)
    file.close()
    return object_file

def log_loss(config, history, _flag):
    filepath = os.path.join("experiments", config["name"], "exp_loss.log")

    _idx = np.where(history.history['val_loss'] == np.min(history.history['val_loss']))[0][0]
    _loss = history.history['loss'][_idx]
    _val_loss = history.history['val_loss'][_idx]
    
    _str = f"{_flag.upper()} {config['name']}\nTrain Loss: {_loss} | Valid Loss: {_val_loss}\n"

    with open(filepath, "a") as fhandle:
        fhandle.write(_str)

def get_train_params(config, model_weights_path):
    _learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=config["train"]["learning_rate"],
                                                                decay_steps=config["train"]["decay_steps"],
                                                                decay_rate=config["train"]["decay_rate"])

    _board_path = os.path.join("experiments", config['name'], "logs")
    _board = tf.keras.callbacks.TensorBoard(log_dir=_board_path)

    _best_model = tf.keras.callbacks.ModelCheckpoint(model_weights_path, save_best_only=True, 
                                                     monitor='val_loss', 
                                                     mode='min')
    
    _callbacks = [_board, _best_model]
    return _learning_rate, _callbacks


def select_keys(encoded_values, _keys):
    filtered_encoded_values = {_key: encoded_values[_key] for _key in _keys} 
    encoded_indexes = [list(encoded_values.keys()).index(_key) for _key in _keys]
    return filtered_encoded_values, encoded_indexes


def train_target_model(config, encoded_values_indexes, _flag): # "engagements or media_spend"
    encoded_values, encoded_indexes = encoded_values_indexes

    if config["train"]["use_bert"]:
        model = bert_mlp(config, encoded_values)
    else:
        model = emb_mlp(config, encoded_values)
    
    model.summary()

    model_weights_path = os.path.join("experiments", config["name"], f"model_{_flag}/model.hdf5")
    _learning_rate, _callbacks = get_train_params(config, model_weights_path)

    _opt  = tf.keras.optimizers.Adam(learning_rate=_learning_rate)
    _loss = tf.keras.losses.MeanSquaredError()

    model.compile(optimizer=_opt, loss=_loss)
    
    # Data
    data_path = config["pre_process"]["processed_data_path"]
    pkl_data = load_pickle(data_path)

    train_num, train_cat, train_bert, train_true = pkl_data[0]
    test_num, test_cat, test_bert, test_true = pkl_data[1]

    train_cat = train_cat[:, encoded_indexes]
    test_cat = test_cat[:, encoded_indexes]

    train_true = train_true[_flag]
    test_true = test_true[_flag]

    train_cat_in = [train_cat[:, idx] for idx in range(train_cat.shape[1])]
    train_data = [train_num]
    train_data.extend(train_cat_in)

    if config["train"]["use_bert"]:
        train_data.append(train_bert)

    if config["train"]["use_test_as_valid"]:
        test_cat_in = [test_cat[:, idx] for idx in range(test_cat.shape[1])]
        test_data = [test_num]
        test_data.extend(test_cat_in)

        if config["train"]["use_bert"]:
            test_data.append(test_bert)

        _valid_data = (test_data, test_true)
    else:
        _valid_data = None
        
    history = model.fit(
            x=train_data,
            y=train_true,
            batch_size=config["train"]["batch_size"],
            epochs=config["train"]["epochs"],
            verbose="auto",
            callbacks=_callbacks,
            validation_split=config["train"]["valid_split"],
            validation_data=_valid_data,
            shuffle=True,
            initial_epoch=0
        )

    model.load_weights(model_weights_path)
    model_path = model_weights_path.replace(".hdf5",".keras")
    model.save(model_path)

    history_path = model_weights_path.replace("model.hdf5", "train_history.pkl")
    save_pickle(history_path, history)
    log_loss(config, history, _flag)


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

        train_target_model(config, encoded_values_indexes, _flag)



if __name__ == "__main__":
    ARGS = get_args()
    config_filepath = ARGS.config_filepath
    main(config_filepath)  


