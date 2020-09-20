def create_lstm_model(fingerprint_input, model_settings, model_size_info, is_training):
    """Builds a model with a lstm layer (with output projection layer and
       peep-hole connections)
    Based on model described in https://arxiv.org/abs/1705.02411
    model_size_info: [projection size, memory cells in LSTM]
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input, [-1, input_time_size, input_frequency_size])

    num_classes = model_settings['label_count']
    projection_units = model_size_info[0]
    LSTM_units = model_size_info[1]
    with tf.name_scope('LSTM-Layer'):
    with tf.variable_scope("lstm"):
        lstmcell = tf.contrib.rnn.LSTMCell(LSTM_units, use_peepholes=True, num_proj=projection_units)
        _, last = tf.nn.dynamic_rnn(cell=lstmcell, inputs=fingerprint_4d, dtype=tf.float32)
        flow = last[-1]

    with tf.name_scope('Output-Layer'):
        W_o = tf.get_variable('W_o', shape=[projection_units, num_classes], initializer=tf.contrib.layers.xavier_initializer())
        b_o = tf.get_variable('b_o', shape=[num_classes])
        logits = tf.matmul(flow, W_o) + b_o

    if is_training:
        return logits, dropout_prob
    else:
        return logits