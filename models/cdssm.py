from base.base_model import BaseModel
import tensorflow as tf


class CDSSM(BaseModel):
    def __init__(self, config):
        super(CDSSM, self).__init__(config)
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
        self.embed1 = tf.expand_dims(self.embed1, -1)

        self.embed2 = tf.nn.embedding_lookup(self._embedding, self.x2)
        #self.embed2 = tf.nn.dropout(self.embed2, self.config.dropout)
        self.embed2 = tf.expand_dims(self.embed2, -1)

       
        # add conv+maxpooling
        pools1 = []
        pools2 = []
        print("word embedd shape", self.embed1.shape)
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.name_scope("conv-maxpool-{}".format(filter_size)):
                # conv layer
                filter_shape = [filter_size, self.config.word_dim, 1, self.config.num_filters]
                print(filter_shape)
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
                conv1 = tf.nn.conv2d(self.embed1, W, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
                conv2 = tf.nn.conv2d(self.embed2, W, strides=[1, 1, 1, 1], padding="VALID", name="conv2")
                # add nonlinear
                h1 = tf.nn.relu(tf.nn.bias_add(conv1, b), name="relu1")
                h2 = tf.nn.relu(tf.nn.bias_add(conv2, b), name="relu2")

                # add maxpooling
                pooled1 = tf.nn.max_pool(h1, ksize=[1, self.config.max_sent_length-filter_size+1, 1, 1], strides=[1, 1, 1, 1], padding="VALID", name="pool1")
                pooled2 = tf.nn.max_pool(h2, ksize=[1, self.config.max_sent_length-filter_size+1, 1, 1], strides=[1, 1, 1, 1], padding="VALID", name="pool2")
                print("pooled size", pooled1.shape)
                pools1.append(pooled1)
                pools2.append(pooled2)

        num_filters_all = self.config.num_filters * len(self.config.filter_sizes)
        self.h_pool1 = tf.concat(pools1, 3)
        print("h_pool1 size:", self.h_pool1.shape)
        self.h_pool1_flat = tf.reshape(self.h_pool1, [-1, num_filters_all])
        self.h_pool2 = tf.concat(pools2, 3)
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, num_filters_all])
        self.dot = tf.multiply(self.h_pool1_flat, self.h_pool2_flat, name="elem_dot") 
        self.sub = tf.abs(tf.subtract(self.h_pool1_flat, self.h_pool2_flat, name="elem_sub"), name="elem_abs") 
        self.sem_feature = tf.concat([self.dot, self.sub], 1)
        print("sem_feature size:", self.sem_feature.shape)

        # add dropout
        with tf.name_scope("dropout"):
            self.f_drop = tf.nn.dropout(self.sem_feature, self.config.dropout)

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

