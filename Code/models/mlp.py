import tensorflow as tf

def emb_mlp(config, encoded_values):
    # Numeric Inputs
    _num_input = tf.keras.layers.Input(shape=(2,)) 

    # Create Embeddings
    num_encoded_values = len(encoded_values)
    list_keys = list(encoded_values.keys())
    _cat_emb_dim = config["train"]["cat_emb_dim"]
    
    _cat_inputs = [tf.keras.layers.Input(shape=(1,)) for _ in range(num_encoded_values)]
    _cat_embedings = [tf.keras.layers.Embedding(input_dim=len(encoded_values[list_keys[idx]]), output_dim=_cat_emb_dim)(_cat_inputs[idx]) 
              for idx in range(num_encoded_values)]
    
    # Layers
    _hid_layer = tf.keras.layers.Concatenate()(_cat_embedings)
    _hid_layer = tf.keras.layers.Flatten()(_hid_layer)
    
    
    # Unify features
    _hid_layer = tf.keras.layers.Concatenate()([_num_input, _hid_layer])
    
    # MLP
    _dense_dims = config["train"]["dense_dims"]
    _drop_out_rate = config["train"]["_drop_out_rate"]

    for _dim in _dense_dims:
        _hid_layer = tf.keras.layers.Dense(_dim, activation="relu",
                                       kernel_regularizer=tf.keras.regularizers.L2(0.01))(_hid_layer)
        # _hid_layer = tf.keras.layers.BatchNormalization()(_hid_layer)
        _hid_layer = tf.keras.layers.Dropout(_drop_out_rate)(_hid_layer)
    
    outputs = tf.keras.layers.Dense(1, activation="linear")(_hid_layer)
    model = tf.keras.Model(inputs=[_num_input, _cat_inputs], outputs=outputs)
    return model