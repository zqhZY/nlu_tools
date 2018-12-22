from base.base_model import BaseModel
from models.blocks import _feedforward_block, _bilstm_block 
from utils.utils import print_shape
import tensorflow as tf


class ESIM(BaseModel):
    def __init__(self, config):
        super(ESIM, self).__init__(config)
        self.build_model()
        self.init_saver()

    def init_placeholder(self):
        """
           x1_mask: actual length of x1
           x2_mask: actual length of x2
        """
        self.is_training = tf.placeholder(tf.bool)
        # input 
        self.x1 = tf.placeholder(tf.int32, name="X1", shape=[None, self.config.max_sent_length])
        self.x2 = tf.placeholder(tf.int32, name="X2", shape=[None, self.config.max_sent_length])
        self.y = tf.placeholder(tf.float32, shape=[None, 2])
        self.x1_mask = tf.placeholder(tf.int32, shape=[None], name="x1_actual_len")
        self.x2_mask = tf.placeholder(tf.int32, shape=[None], name="x2_actual_len")
        #self.x1_mask = tf.placeholder(tf.float32, shape=[None], name="x1_actual_len")
        #self.x2_mask = tf.placeholder(tf.float32, shape=[None], name="x2_actual_len")
        print(self.x1_mask)

    def input_encoding_block(self, scope):
        """
           encoding block
        """
        # build self.embedding
        self._embedding = self.add_word_embedding()
        print("embedd shape", self._embedding.shape)
        self.embed1 = tf.nn.embedding_lookup(self._embedding, self.x1)
        #self.embed1 = tf.nn.dropout(self.embed1, self.config.dropout)
        #self.embed1 = tf.expand_dims(self.embed1, -1)
        print("word embedd1 shape", self.embed1.shape)

        self.embed2 = tf.nn.embedding_lookup(self._embedding, self.x2)
        #self.embed2 = tf.nn.dropout(self.embed2, self.config.dropout)
        #self.embed2 = tf.expand_dims(self.embed2, -1)
        print("word embedd2 shape", self.embed2.shape)

        with tf.variable_scope(scope):
            # a_bar = BiLSTM(a, i) (1)
            # b_bar = BiLSTM(b, i) (2)
            if self.config.using_actual_len:
                outputs_x1, final_states_x1 = _bilstm_block(self.embed1, self.config.hidden_size, 'bilstm', seq_len=self.x1_mask)
                outputs_x2, final_states_x2 = _bilstm_block(self.embed2, self.config.hidden_size, 'bilstm', seq_len=self.x2_mask, reuse=True)
            else:
                outputs_x1, final_states_x1 = _bilstm_block(self.embed1, self.config.hidden_size, 'bilstm',self.config.dropout)
                outputs_x2, final_states_x2 = _bilstm_block(self.embed2, self.config.hidden_size, 'bilstm', self.config.dropout, reuse=True)

            a_bar = tf.concat(outputs_x1, axis=2)
            b_bar = tf.concat(outputs_x2, axis=2)
            print_shape('a_bar', a_bar)
            print_shape('b_bar', b_bar)
            return a_bar, b_bar

    def local_infer_block(self, x1_bar, x2_bar, scope):
        """
           attention block
           x1_bar: shape (batch, seq_len, 2*hidden_size)
           x2_bar: shape (batch, seq_len, 2*hidden_size)
        """
        attention_matrix = tf.matmul(x1_bar, x2_bar, transpose_b=True, name="attention_matrix")
        print_shape("attention_matrix shape:", attention_matrix)

        attention_soft_x1 = tf.nn.softmax(attention_matrix)
        attention_soft_x2 = tf.nn.softmax(tf.transpose(attention_matrix))
        attention_soft_x2 = tf.transpose(attention_soft_x2)
        print_shape("soft attention x1", attention_soft_x1)
        print_shape("soft attention x2", attention_soft_x2)

        x1_atten = tf.matmul(attention_soft_x1, x2_bar)
        x2_atten = tf.matmul(attention_soft_x2, x1_bar)
        print_shape("x1 atten", x1_atten)
        print_shape("x2 atten", x2_atten)
        
        x1_diff = tf.subtract(x1_bar, x1_atten)
        x1_mul = tf.multiply(x1_bar, x1_atten)
        print_shape("x1 diff", x1_diff)
        print_shape("x1 mul", x1_mul)
        
        x2_diff = tf.subtract(x2_bar, x2_atten)
        x2_mul = tf.multiply(x2_bar, x2_atten)
        print_shape("x2 diff", x2_diff)
        print_shape("x2 mul", x2_mul)

        # [batch, seq_len, 4*hidden_size]
        # [x1, x1_atten, x1-x1_atten, x1*x1_atten]
        m_a = tf.concat([x1_bar, x1_atten, x1_diff, x1_mul], axis = 2)
        m_b = tf.concat([x2_bar, x2_atten, x2_diff, x2_mul], axis = 2)
        print_shape("m_a", m_a)
        print_shape("m_b", m_b)

        return m_a, m_b

    def composition_block(self, m_a, m_b, hidden_size, scope):
        """
        :param m_a: concat of [a_bar, a_hat, a_diff, a_mul], tensor with shape (batch_size, seq_length, 4 * 2 * hidden_size)
        :param m_b: concat of [b_bar, b_hat, b_diff, b_mul], tensor with shape (batch_size, seq_length, 4 * 2 * hidden_size)
        :param hiddenSize: biLSTM cell's hidden states size
        :param scope: scope name
        outputV_a, outputV_b: hidden states of biLSTM, tuple (forward LSTM cell, backward LSTM cell)
        v_a, v_b: concate of biLSTM hidden states, tensor with shape (batch_size, seq_length, 2 * hidden_size)
        v_a_avg, v_b_avg: timestep (axis = seq_length) average of v_a, v_b, tensor with shape (batch_size, 2 * hidden_size)
        v_a_max, v_b_max: timestep (axis = seq_length) max value of v_a, v_b, tensor with shape (batch_size, 2 * hidden_size)
        v: concat of [v_a_avg, v_b_avg, v_a_max, v_b_max], tensor with shape (batch_size, 4 * 2 * hidden_size)
        :return: y_hat: output of feed forward layer, tensor with shape (batch_size, n_classes)
        """
        with tf.variable_scope(scope):
            outputV_a, finalStateV_a = _bilstm_block(m_a, hidden_size, 'biLSTM', self.config.dropout)
            outputV_b, finalStateV_b = _bilstm_block(m_b, hidden_size, 'biLSTM', self.config.dropout, reuse=True)
            v_a = tf.concat(outputV_a, axis = 2)
            v_b = tf.concat(outputV_b, axis = 2)

            print_shape('v_a', v_a)
            print_shape('v_b', v_b)

            # v_{a,avg} = \sum_{i=1}^l_a \frac{v_a,i}{l_a}, v_{a,max} = \max_{i=1} ^ l_a v_{a,i} (18)
            # v_{b,avg} = \sum_{j=1}^l_b \frac{v_b,j}{l_b}, v_{b,max} = \max_{j=1} ^ l_b v_{b,j} (19)
            v_a_avg = tf.reduce_mean(v_a, axis = 1)
            v_b_avg = tf.reduce_mean(v_b, axis = 1)
            v_a_max = tf.reduce_max(v_a, axis = 1)
            v_b_max = tf.reduce_max(v_b, axis = 1)
            print_shape('v_a_avg', v_a_avg)
            print_shape('v_a_max', v_a_max)

            # v = [v_{a,avg}; v_{a,max}; v_{b,avg}; v_{b_max}] (20)
            v = tf.concat([v_a_avg, v_a_max, v_b_avg, v_b_max], axis = 1)
            print_shape('v', v)
            y_hat = _feedforward_block(v, self.config.dense_size, self.config.n_classes, 'feed_forward', self.config.dropout)
            return y_hat


    def build_model(self):

        # init placehoders
        self.init_placeholder()

        # build encoding block
        x1_bar, x2_bar = self.input_encoding_block("encoding_block")
        
        # build local inference block
        # [batch, seq_len, 4*hidden_size]
        m_x1, m_x2 = self.local_infer_block(x1_bar, x2_bar, "local_infer_block")

        d2 = self.composition_block(m_x1, m_x2, self.config.hidden_size, "composition_block")

        #d2 = tf.squeeze(d2)
        #print(d2)
        
        with tf.name_scope("loss"):
            if self.config.loss == "mse":
                self.cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.y, predictions=d2))
            else:
                self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d2))
                correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cost,
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

