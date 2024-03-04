import os
import pickle
import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler


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


def train_test_split_idxs(config, num_samples):
    np.random.seed(config["seed"])
    list_idxs = list(range(num_samples))
    np.random.shuffle(list_idxs)
    
    split = int(num_samples*config["pre_process"]["train_split"])

    return list_idxs[:split], list_idxs[split:], 


def save_encoded_trainable_values(config, df, _cat_keys):
    dict_sets = {}
    for _key in _cat_keys:
        dict_sets[_key] = list(set(df[_key]))
    
    path_out = os.path.join("experiments", config["name"], "encoded_values.pkl")
    save_pickle(path_out, dict_sets)


def date_to_day(date):
    dt = datetime.datetime.strptime(date, '%Y-%m-%d')
    return str(datetime.datetime.weekday(dt))

def remove_target_outliers(df, std_limit=3):
    for _target in ["engagements", "media_spend"]:
        _mean, _std = np.mean(df[_target]), np.std(df[_target])
        df = df.loc[df[_target] < _mean + std_limit*_std]
    return df

def process_num_df(df, _num_keys):
    # standardized values
    df_filt = df[_num_keys].copy()
    standard_scaler = StandardScaler()
    num_scaled = standard_scaler.fit_transform(df_filt)

    num_stats = [standard_scaler.mean_, standard_scaler.var_]

    return num_scaled, num_stats

def create_bert_text(df):
    print("\nUsing Headlines and Story Summaries as BERT text.\n")
    df_filt = df.copy()
    df_filt["bert_text"] = df_filt["headline"]+" "+df_filt["storySummary"]
    return np.array(df_filt["bert_text"])

def process_cat_df(config, df, _cat_keys):
    df_filt = df[_cat_keys].copy()

    # num2str
    for _key in ["group", "item"]:
        if _key in _cat_keys:
            df_filt[_key] = df_filt[_key].apply(lambda x: str(x))

    # process date
    if "date" in _cat_keys:
        df_filt['date'] = df_filt['date'].apply(lambda x: date_to_day(x))

    # NaNs to UNK
    df_filt = df_filt.fillna(value="UNK")

    # Add UNK line for Encoder
    unk_data = ["UNK" for _ in range(len(_cat_keys))]
    df_unk_line = pd.DataFrame([unk_data], columns=df_filt.columns.to_list())
    df_filt = pd.concat([df_filt, df_unk_line])
    
    # Save encoded values for train/test
    save_encoded_trainable_values(config, df_filt, _cat_keys)

    # Encoding
    ordinal_encoder = OrdinalEncoder()
    cat_encoded = ordinal_encoder.fit_transform(df_filt)

    # Remove UNK from last row
    cat_encoded = cat_encoded[:-1,:].copy()

    return cat_encoded


def main(config_filepath):
    config = load_config(config_filepath)
    
    _num_keys = config["pre_process"]["num_input"]
    _cat_keys = config["pre_process"]["cat_input"]
    _out_keys = config["pre_process"]["target_output"]

    if not _num_keys:
        _num_keys = NUMERICAL_KEYS
    
    if not _cat_keys:
        _cat_keys = CATEGORICAL_KEYS
    
    df_path = config["datapath"]
    df = pd.read_csv(df_path)

    if config["pre_process"]["remove_outliers"]:
        _len_init = len(df)
        df = remove_target_outliers(df, config["pre_process"]["std_limit"])
        _len_final = len(df)
        _diff = _len_init-_len_final
        print(f"\n Total of {_diff} outliers filtered ({100*_diff/_len_init}%).\n")

    num_scaled, num_stats = process_num_df(df, _num_keys)
    cat_encoded = process_cat_df(config, df, _cat_keys)
    
    bert_text   = create_bert_text(df)

    true_values = {_key: np.array(list(df[_key])) for _key in _out_keys}

    num_samples = len(cat_encoded)
    train_idxs, test_idxs = train_test_split_idxs(config, num_samples)

    train_num = num_scaled[train_idxs, :].copy()
    train_cat = cat_encoded[train_idxs, :].copy()
    train_bert = bert_text[train_idxs].copy()

    train_true = {_key: true_values[_key][train_idxs].copy() for _key in _out_keys}
    

    test_num = num_scaled[test_idxs, :].copy()
    test_cat = cat_encoded[test_idxs, :].copy()
    test_bert = bert_text[test_idxs].copy()
    
    test_true = {_key: true_values[_key][test_idxs].copy() for _key in _out_keys}
    

    _train_data = (train_num, train_cat, train_bert, train_true)
    _test_data  = (test_num, test_cat, test_bert, test_true)

    output_path = config["pre_process"]["processed_data_path"]
    save_pickle(output_path, (_train_data, _test_data))

    output_stats_path = os.path.join("experiments", config["name"], "num_stats.pkl")
    save_pickle(output_stats_path, num_stats)


if __name__ == "__main__":
    ARGS = get_args()
    config_filepath = ARGS.config_filepath
    main(config_filepath)  

