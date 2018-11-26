from base.base_model import BaseModel
from models.blocks import _feedforward_block, _bilstm_block 
from utils.utils import print_shape
import tensorflow as tf


class LSTM(BaseModel):
    def __init__(self, config):
        super(LSTM, self).__init__(config)
        self.build_model()
        self.init_saver()

    def init_placeholder(self):
        """
           x1_mask: actual length of x1
           x2_mask: actual length of x2
        """
        self.is_training = tf.placeholder(tf.bool)
        # input 
        self.x = tf.placeholder(tf.int32, name="X", shape=[None, self.config.max_sent_length])
        self.y = tf.placeholder(tf.float32, shape=[None, self.config.n_classes])
        self.x_mask = tf.placeholder(tf.int32, shape=[None], name="x_actual_len")
        print(self.x_mask)

    def input_encoding_block(self, scope):
        """
           encoding block
        """
        # build self.embedding
        self._embedding = self.add_word_embedding()
        print("embedd shape", self._embedding.shape)
        self.embed = tf.nn.embedding_lookup(self._embedding, self.x)
        #self.embed1 = tf.nn.dropout(self.embed1, self.config.dropout)
        #self.embed1 = tf.expand_dims(self.embed1, -1)
        print("word embedd1 shape", self.embed.shape)

        with tf.variable_scope(scope):
            # a_bar = BiLSTM(a, i) (1)
            # b_bar = BiLSTM(b, i) (2)
            if self.config.using_actual_len:
                outputs, final_states = _bilstm_block(self.embed, self.config.hidden_size, 'bilstm', seq_len=self.x_mask)
            else:
                outputs, final_states = _bilstm_block(self.embed, self.config.hidden_size, 'bilstm',self.config.dropout)

            print(final_states[0][0])
            bar = tf.concat((final_states[0][0], final_states[1][0]), axis=1)
            print_shape('bar', bar)
            return bar 

    def build_model(self):

        # init placehoders
        self.init_placeholder()

        # build encoding block
        bar = self.input_encoding_block("encoding_block")
        d2 = _feedforward_block(bar, self.config.dense_size, self.config.n_classes, 'feed_forward', self.config.dropout)

        if self.config.lr_decay:
            self.lr = tf.train.exponential_decay(learning_rate=self.config.learning_rate, global_step=self.global_step_tensor, decay_steps=self.config.decay_step, decay_rate=0.9, staircase=False)
        else:
            self.lr = self.config.learning_rate

        with tf.name_scope("loss"):
            if self.config.loss == "mse":
                self.cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.y, predictions=d2))
            else:
                self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d2))
                correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.cost,
                                                                      global_step=self.global_step_tensor)


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

