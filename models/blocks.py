import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper

def _feedforward_block(inputs, hidden_dims, num_units, scope, dropout_keep_prob=1.0, reuse = False, initializer = None):
    """
        :param inputs: tensor with shape (batch_size, hidden_size)
        :param scope: scope name
        :return: output: tensor with shape (batch_size, num_units)
    """
    with tf.variable_scope(scope, reuse=reuse):
        if initializer is None:
            initializer = tf.random_normal_initializer(0.0, 0.1)

        with tf.variable_scope('feed_foward_layer1'):
            inputs = tf.nn.dropout(inputs, dropout_keep_prob)
            outputs = tf.layers.dense(inputs, hidden_dims, tf.nn.relu, kernel_initializer = initializer)
        with tf.variable_scope('feed_foward_layer2'):
            outputs = tf.nn.dropout(outputs, dropout_keep_prob)
            results = tf.layers.dense(outputs, num_units, tf.nn.tanh, kernel_initializer = initializer)
        return results


def _bilstm_block(inputs, num_units, scope, dropout_keep_prob=1.0, seq_len=None, reuse=False):
    """
       :param inputs: tensor with shape (batch_size, hidden_size)
       :param scope: scope name
       :output:  tuple (outputs, output_states) where outputs is a tuple (output_fw, output_bw)
    """
    #print(seq_len)
    with tf.variable_scope(scope, reuse=reuse):
        # reuse lstm cell for fw and bw cell
        lstm_cell = LSTMCell(num_units=num_units)
        lstmDrop = lambda: DropoutWrapper(lstm_cell, output_keep_prob=dropout_keep_prob)
        lstm_cell_fw, lstm_cell_bw = lstmDrop(), lstmDrop()
        output = tf.nn.bidirectional_dynamic_rnn(cell_fw = lstm_cell_fw,
                                                 cell_bw = lstm_cell_bw,
                                                 inputs = inputs,
                                                 sequence_length = seq_len,
                                                 dtype = tf.float32)
        return output

