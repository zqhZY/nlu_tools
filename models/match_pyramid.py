from base.base_model import BaseModel
import tensorflow as tf


class MatchPyramid(BaseModel):
    def __init__(self, config):
        super(MatchPyramid, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        # input 
        self.x1 = tf.placeholder(tf.int32, name="X1", shape=[None, self.config.max_sent_length])
        self.x2 = tf.placeholder(tf.int32, name="X2", shape=[None, self.config.max_sent_length])
        self.y = tf.placeholder(tf.float32, shape=[None])

        # build self.embedding
        self._embedding = self.add_word_embedding()
        print("embedd shape", self._embedding.shape)
        self.embed1 = tf.nn.embedding_lookup(self._embedding, self.x1)
        print("word embedd shape", self.embed1.shape)
        #self.embed1 = tf.nn.dropout(self.embed1, self.config.dropout)
        #self.embed1 = tf.expand_dims(self.embed1, -1)

        self.embed2 = tf.nn.embedding_lookup(self._embedding, self.x2)
        #self.embed2 = tf.nn.dropout(self.embed2, self.config.dropout)
        #self.embed2 = tf.expand_dims(self.embed2, -1)
        self.cross = tf.matmul(self.embed1, self.embed2, transpose_b=True, name="sim_matrix")
        self.cross_with_channel = tf.expand_dims(self.cross, -1)

        # add conv+maxpooling
        filter_shape = [self.config.filter_size[0], self.config.filter_size[1], 1, self.config.num_filters]
        print(filter_shape)
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
        conv = tf.nn.conv2d(self.cross_with_channel, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
        conv_out = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        pooled = tf.nn.max_pool(conv_out, ksize=[1, self.config.pool_size[0], self.config.pool_size[1], 1], strides=[1, 1, 1, 1], padding="VALID", name="pool")

        num_filters_all = self.config.num_filters * self.config.max_sent_length * self.config.max_sent_length
        self.pool_flatted = tf.reshape(self.pooled, [-1, num_filters_all])
        print("sem_feature size:", self.pool_flatted.shape)

        # add dropout
        with tf.name_scope("dropout"):
            self.f_drop = tf.nn.dropout(self.pool_flatted, self.config.dropout)

        # network architecture
        with tf.name_scope("output"):
            d0 = tf.layers.dense(self.f_drop, 512, activation=tf.nn.relu, name="dense0")
            d1 = tf.layers.dense(d0, 128, activation=tf.nn.relu, name="dense1")
            d2 = tf.layers.dense(d1, 1, name="dense2")

        d2 = tf.squeeze(d2)
        print(d2)
        
        with tf.name_scope("loss"):
            self.cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.y, predictions=d2))
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cost,
                                                                                         global_step=self.global_step_tensor)
            # computer Pearson 
            #correct_prediction = tf.equal(d2, self.y)
            #self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def add_word_embedding(self):
        """
        Defines self.word_embeddings
        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words"):
            if "embedding" not in self.config:
                print("warning: random initialize word embedding...")
                _word_embedding = tf.get_variable(name="_word_embedding", \
                                                  dtype=tf.float32, \
                                                  shape=[self.config.nwords, self.config.word_dim])
            else:
                _word_embedding = tf.get_variable( initializer=self.config.embedding,
                                                  name="_word_embedding",
                                                  dtype=tf.float32,
                                                  trainable=self.config.train_embedding)
        print(_word_embedding.shape)
        return _word_embedding

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

