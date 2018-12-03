from base.base_model import BaseModel
import tensorflow as tf


class CNN(BaseModel):
    def __init__(self, config):
        super(CNN, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        # input 
        self.x = tf.placeholder(tf.int32, name="X", shape=[None, self.config.max_sent_length])
        self.x_char_cnn = tf.placeholder(tf.int32, name="X_char_cnn", shape=[None, self.config.max_sent_length, self.config.char_cnn.max_token_len])
        self.x_pinyin = tf.placeholder(tf.int32, name="X_pinyin", shape=[None, self.config.max_sent_length, self.config.char_cnn.max_token_len])
        self.y = tf.placeholder(tf.float32, shape=[None, self.config.n_classes])

        # build self.embedding
        self._embedding = self.add_word_embedding()
        print("embedd shape", self._embedding.shape)
        self.embed = tf.nn.embedding_lookup(self._embedding, self.x)
        print("word embedd shape", self.embed.shape)
        #self.embed1 = tf.nn.dropout(self.embed1, self.config.dropout)
        self.embed = tf.expand_dims(self.embed, -1)
        # add conv+maxpooling
        pools = []
        print("word embedd shape", self.embed.shape)
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.name_scope("conv-maxpool-{}".format(filter_size)):
                # conv layer
                filter_shape = [filter_size, self.config.word_dim, 1, self.config.num_filters]
                print("filter size: ", filter_shape)
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
                conv = tf.nn.conv2d(self.embed, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                # add nonlinear
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # add maxpooling
                pooled = tf.nn.max_pool(h, ksize=[1, self.config.max_sent_length-filter_size+1, 1, 1], strides=[1, 1, 1, 1], padding="VALID", name="pool")
                print("pooled size", pooled.shape)
                pools.append(pooled)

        num_filters_all = self.config.num_filters * len(self.config.filter_sizes)
        self.h_pool = tf.concat(pools, 3)
        print("h_pool size:", self.h_pool.shape)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_all])
        print("dense feature size:", self.h_pool_flat.shape)

        # add dropout
        with tf.name_scope("dropout"):
            self.f_drop = tf.nn.dropout(self.h_pool_flat, self.config.dropout)

        # network architecture
        with tf.name_scope("output"):
            d0 = tf.layers.dense(self.f_drop, 64, activation=tf.nn.relu, name="dense0")
            d0 = tf.nn.dropout(d0, self.config.dropout)
            d1 = tf.layers.dense(d0, 32, activation=tf.nn.relu, name="dense1")
            d1 = tf.nn.dropout(d1, self.config.dropout)
            d2 = tf.layers.dense(d1, self.config.n_classes, name="dense2")

        d2 = tf.squeeze(d2)
        print(d2)
        #global_step = tf.Variable(0, trainable=False)
        #learning_rate = tf.train.exponential_decay(self.config.learning_rate, self.global_step_tensor, decay_steps=10, decay_rate=0.9)
        #boundaries = [3205, 6410]
        #values = [0.01, 0.001, 0.0001]
        #learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        
        with tf.name_scope("loss"):
            if self.config.loss == "mse":
                self.cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.y, predictions=d2))
            else:
                self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d2))
                correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cost, global_step=self.global_step_tensor)


    def add_word_embedding(self):
        """
        Defines self.word_embeddings
        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words"):
            if "embedding_word" not in self.config:
                print("warning: random initialize word embedding...")
                _word_embedding = tf.get_variable(name="_word_embedding", \
                                                  dtype=tf.float32, \
                                                  shape=[self.config.nwords, self.config.word_dim])
            else:
                _word_embedding = tf.get_variable( initializer=self.config.embedding_word,
                                                  name="_word_embedding",
                                                  dtype=tf.float32,
                                                  trainable=self.config.train_embedding)
        print(_word_embedding.shape)
        return _word_embedding

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

