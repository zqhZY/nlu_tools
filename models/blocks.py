import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper
import numpy as np

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

def _build_char_cnn(ids_input, options):
	'''
	options contains key 'char_cnn': {
	'n_characters': 60,
	# includes the start / end characters
	'max_characters_per_token': 17,
	'filters': [
		[1, 32],
		[2, 32],
		[3, 64],
		[4, 128],
		[5, 256],
		[6, 512],
		[7, 512]
	],
	'activation': 'tanh',
	# for the character embedding
	'embedding': {'dim': 16}
	# for highway layers
	# if omitted, then no highway layers
	'n_highway': 2,
	}
	'''
	projection_dim = options.projection_dim
	cnn_options = options.char_cnn
	filters = cnn_options.filters
	n_filters = sum(f[1] for f in filters)
	max_chars = cnn_options.max_token_len
	char_embed_dim = cnn_options.dim
	n_chars = cnn_options.n_characters
	if cnn_options.activation == 'tanh':
		activation = tf.nn.tanh
	elif cnn_options.activation == 'relu':
		activation = tf.nn.relu

	# the character embeddings
	with tf.device("/cpu:0"):
		#embedding_weights = tf.get_variable(
		#		"char_embed", [n_chars, char_embed_dim],
		#		dtype=tf.float32,
		#		initializer=tf.random_uniform_initializer(-1.0, 1.0)
		#)
            with tf.variable_scope("chars"):
                if "embedding_char____" not in options:
                    print("warning: random initialize char embedding...")
                    embedding_weights = tf.get_variable(name="_char_cnn_embedding", \
                                                  dtype=tf.float32, \
                                                  shape=[n_chars, char_embed_dim],
                                                  initializer=tf.random_uniform_initializer(-1.0, 1.0))
                else:
                    embedding_weights = tf.get_variable( initializer=options.embedding_char,
                                                  name="_char_cnn_embedding",
                                                  dtype=tf.float32,
                                                  trainable=options.train_embedding)
		# shape (batch_size, unroll_steps, max_chars, embed_dim)
            char_embedding = tf.nn.embedding_lookup(embedding_weights, ids_input)

	# the convolutions
	def make_convolutions(inp):
		with tf.variable_scope('CNN') as scope:
			convolutions = []
			for i, (width, num) in enumerate(filters):
				if cnn_options.activation == 'relu':
					# He initialization for ReLU activation
					# with char embeddings init between -1 and 1
					#w_init = tf.random_normal_initializer(
					#    mean=0.0,
					#    stddev=np.sqrt(2.0 / (width * char_embed_dim))
					#)

					# Kim et al 2015, +/- 0.05
					w_init = tf.random_uniform_initializer(
						minval=-0.05, maxval=0.05)
				elif cnn_options.activation == 'tanh':
					# glorot init
					w_init = tf.random_normal_initializer(
						mean=0.0,
						stddev=np.sqrt(1.0 / (width * char_embed_dim))
					)
				w = tf.get_variable(
					"W_cnn_%s" % i,
					[1, width, char_embed_dim, num],
					initializer=w_init,
					dtype=tf.float32)
				b = tf.get_variable(
					"b_cnn_%s" % i, [num], dtype=tf.float32,
					initializer=tf.constant_initializer(0.0))

				conv = tf.nn.conv2d(
						inp, w,
						strides=[1, 1, 1, 1],
						padding="VALID") + b
				# now max pool
				conv = tf.nn.max_pool(
						conv, [1, 1, max_chars-width+1, 1],
						[1, 1, 1, 1], 'VALID')

				# activation
				conv = activation(conv)
				conv = tf.squeeze(conv, squeeze_dims=[2])

				convolutions.append(conv)

		return tf.concat(convolutions, 2)

	embedding = make_convolutions(char_embedding)

	# for highway and projection layers
	n_highway = cnn_options.get('n_highway')
	use_highway = n_highway is not None and n_highway > 0
	use_proj = n_filters != projection_dim

	if use_highway or use_proj:
		#   reshape from (batch_size, n_tokens, dim) to (-1, dim)
		batch_size_n_tokens = tf.shape(embedding)[0:2]
		embedding = tf.reshape(embedding, [-1, n_filters])

	# set up weights for projection
	if use_proj:
		assert n_filters > projection_dim
		with tf.variable_scope('CNN_proj') as scope:
				W_proj_cnn = tf.get_variable(
					"W_proj", [n_filters, projection_dim],
					initializer=tf.random_normal_initializer(
						mean=0.0, stddev=np.sqrt(1.0 / n_filters)),
					dtype=tf.float32)
				b_proj_cnn = tf.get_variable(
					"b_proj", [projection_dim],
					initializer=tf.constant_initializer(0.0),
					dtype=tf.float32)

	# apply highways layers
	def high(x, ww_carry, bb_carry, ww_tr, bb_tr):
		carry_gate = tf.nn.sigmoid(tf.matmul(x, ww_carry) + bb_carry)
		transform_gate = tf.nn.relu(tf.matmul(x, ww_tr) + bb_tr)
		return carry_gate * transform_gate + (1.0 - carry_gate) * x

	if use_highway:
		highway_dim = n_filters

		for i in range(n_highway):
			with tf.variable_scope('CNN_high_%s' % i) as scope:
				W_carry = tf.get_variable(
					'W_carry', [highway_dim, highway_dim],
					# glorit init
					initializer=tf.random_normal_initializer(
						mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
					dtype=tf.float32)
				b_carry = tf.get_variable(
					'b_carry', [highway_dim],
					initializer=tf.constant_initializer(-2.0),
					dtype=tf.float32)
				W_transform = tf.get_variable(
					'W_transform', [highway_dim, highway_dim],
					initializer=tf.random_normal_initializer(
						mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
					dtype=tf.float32)
				b_transform = tf.get_variable(
					'b_transform', [highway_dim],
					initializer=tf.constant_initializer(0.0),
					dtype=tf.float32)

			embedding = high(embedding, W_carry, b_carry,
							 W_transform, b_transform)

	# finally project down if needed
	if use_proj:
		embedding = tf.matmul(embedding, W_proj_cnn) + b_proj_cnn

	# reshape back to (batch_size, tokens, dim)
	if use_highway or use_proj:
		shp = tf.concat([batch_size_n_tokens, [projection_dim]], axis=0)
		embedding = tf.reshape(embedding, shp)

	# at last assign attributes for remainder of the model
	return embedding
