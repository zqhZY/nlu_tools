from tensorflow.contrib import learn
import re
import numpy as np
import jieba
from pypinyin import lazy_pinyin

_pad = 0
_unk = 0

class SemDataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        self.ids_train, self.tx1s_train_raw, self.tx2s_train_raw, self.labels_train, self.tx1_len, self.tx2_len = self.load_data(self.config.train_data, True)
        self.ids_dev, self.tx1s_dev_raw, self.tx2s_dev_raw, self.labels_dev, self.tx1_len_dev, self.tx2_len_dev = self.load_data(self.config.dev_data, True)

        # build voc dict
        if self.config.load_voc:
            self.word_processor = learn.preprocessing.VocabularyProcessor.restore(self.config.word_token_conf.voc_path)
        else:
            #self.char_processor = self.prepare_processor(self.config.char_token_conf, self.char_seg, export_vector=False)
            self.word_processor = self.prepare_processor(self.config.word_token_conf, None, export_vector=False)
            #self.pinyin_processor = self.prepare_processor(self.config.pinyin_token_conf, self.pinyin_seg, export_vector=False)
            #self.char2ids = self.char_processor.vocabulary_._mapping
            #self.pinyin2ids = self.pinyin_processor.vocabulary_._mapping

		# transform data to word ids
        print("vocab size is {}".format(len(self.word_processor.vocabulary_)))
        self.tx1s_train = np.array(list(self.word_processor.transform(self.tx1s_train_raw)))
        self.tx2s_train = np.array(list(self.word_processor.transform(self.tx2s_train_raw)))
        self.tx1s_dev = np.array(list(self.word_processor.transform(self.tx1s_dev_raw)))
        self.tx2s_dev = np.array(list(self.word_processor.transform(self.tx2s_dev_raw)))

        print("train data num is {}".format(len(self.tx1s_train)))
        print("dev data num is {}".format(len(self.tx1s_dev)))
        self.num_batches_per_epoch_train = int((len(self.tx1s_train)-1)/self.config.batch_size) + 1
        self.num_batches_per_epoch_dev = int((len(self.tx1s_dev)-1)/self.config.batch_size) + 1

        # build data iter
        self.train_data_iter = self.batch_iter(list(zip(self.ids_train, self.tx1s_train, self.tx2s_train, self.labels_train, self.tx1_len, self.tx2_len)), \
                                               self.config.batch_size, \
                                               self.config.num_epochs, \
                                               self.config.shuffle_data)

    def get_dev_iter(self):
        return self.batch_iter(list(zip(self.ids_dev, self.tx1s_dev, self.tx2s_dev, self.labels_dev, self.tx1_len_dev, self.tx2_len_dev)), \
                               self.config.batch_size, \
                               1, \
                               False)


    def pred_process(self, text):
        """process for predict pipeline"""
        word_ids = np.array(list(self.word_processor.transform([text])))
        char_ids = np.array(list(self.char_transform([text])))
        pinyin_ids = np.array(list(self.char_transform([text], is_pinyin=True)))
        return word_ids, char_ids, pinyin_ids

    def data_preproces(self, text):
        pattern = "[,.'\"]"
        pattern2 = "(it's)|(It's)"
    
        text = re.sub(pattern, "", text)
        text = re.sub(pattern2, "it is", text)
        text = text.lower()
        return text

    def jieba_seg(self, docs):
        for doc in docs:
            yield list(jieba.cut(doc))

    def char_seg(self, docs):
        for doc in docs:
            yield [c for c in doc]

    def pinyin_seg(self, docs):
        for doc in docs:
            doc_pinyin = [lazy_pinyin(word)[0] for word in doc]
            yield doc_pinyin

    def word2char_ids(self, word, is_pinyin=False):
        if is_pinyin:
            char2ids = self.pinyin2ids
        else:
            char2ids = self.char2ids
        
        code = np.zeros([self.config.max_token_length], dtype=np.int32)
        code[:] = _pad # pad num
        for i, char in enumerate(word[:self.config.max_token_length]):
            if char in char2ids:
                code[i] = char2ids[char]
            else:
                code[i] = _unk # pad num
        return code

    def encode_chars(self, sentence, is_pinyin=False, split=True):
        if split:
            char_ids = [self.word2char_ids(word, is_pinyin) for word in sentence.split()]
        else:
            char_ids = [self.word2char_ids(word, is_pinyin) for word in sentence]
        return char_ids

    def char_transform(self, docs, is_pinyin=False):
        n_sentences = len(docs)
        x_char_ids = np.zeros(
            (n_sentences, self.config.max_sent_length, self.config.max_token_length),
            dtype=np.int64
        )
        for k, doc in enumerate(docs):
            sent = list(jieba.cut(doc))
            length = len(sent)
            c_ids = self.encode_chars(sent, split=False, is_pinyin=is_pinyin)
            x_char_ids[k, :length, :] = c_ids[:self.config.max_sent_length]
        return x_char_ids

    def build_data(self):
        """build data """
        self.char_processor = self.prepare_processor(self.config.char_token_conf, self.char_seg)
        self.word_processor = self.prepare_processor(self.config.word_token_conf, self.jieba_seg)
        #self.build_tokenize(self.config.max_sent_length, self.config.pinyin_dict_path, self.pinyin_seg)

    def prepare_processor(self, config, tokenizer, export_vector=True):
        """build offline data"""
        # save word processor
        print("fit vocab_processor")
        print(config)
        vocab_processor = learn.preprocessing.VocabularyProcessor(config.max_sent_length, min_frequency=config.min_frequency, tokenizer_fn=tokenizer)
        vocab_processor.fit(self.tx1s_train_raw + self.tx2s_train_raw + self.tx1s_dev_raw + self.tx2s_dev_raw)
        
        # save word dict
        print("save word dict")
        word_dict = vocab_processor.vocabulary_._mapping
        self.save_dict(word_dict, config.dict_path)
        # export trimmed glove vector
        if export_vector:
            print("export trimmed embedding")
            self.export_trimmed_glove_vectors(word_dict, config.embedding_path, config.trimmed_embedding_name, config.word_dim)
        return vocab_processor

    def export_trimmed_glove_vectors(self, vocab, glove_filename, trimmed_filename, dim):
        """Saves glove vectors in numpy array
        Args:
            vocab: dictionary vocab[word] = index
            glove_filename: a path to a glove file
            trimmed_filename: a path where to store a matrix in npy
            dim: (int) dimension of embeddings
        """
        embeddings = np.zeros([len(vocab), dim])
        print(embeddings.shape)
        with open(glove_filename) as f:
            for line in f:
                line = line.strip().split(' ')
                word = line[0]
                embedding = [float(x) for x in line[1:]]
                if word in vocab:
                    word_idx = vocab[word]
                    embeddings[word_idx] = np.asarray(embedding, dtype=np.float32)

        np.savez_compressed(trimmed_filename, embeddings=embeddings)

    def get_trimmed_glove_vectors(self, embedding_file):
        """
        Args:
            filename: path to the npz file
        Returns:
            matrix of embeddings (np array)
        """
        try:
            with np.load(embedding_file) as data:
                return np.float32(data["embeddings"])
        except IOError:
            raise MyIOError(embedding_file)

    def get_word_dict(self):
        return self.vocab_processor.vocabulary_._mapping

    def save_dict(self, word_dict, save_file):
        with open(save_file, "w") as f:
            for k, v in word_dict.items():
                f.write(str(v) + "\t" + k + "\n")

    def load_data(self, file_name, with_label=True):
        
        ids = []
        tx1s = []
        tx2s = []
        labels = []
        tx1_lens = []
        tx2_lens = []
        with open(file_name) as f:
            for line in f:
                tokens = line.strip().split("\t")
                ids.append([tokens[0]])
                tx1s.append(self.data_preproces(tokens[1]))
                tx2s.append(self.data_preproces(tokens[2]))
                t1_len = len(tokens[1].split())
                t2_len = len(tokens[2].split())
                if t1_len > self.config.max_sent_length:
                    t1_len = self.config.max_sent_length
                if t2_len > self.config.max_sent_length:
                    t2_len = self.config.max_sent_length
                tx1_lens.append(t1_len)
                tx2_lens.append(t2_len)
                labels.append([float(tokens[3])])
        return ids, tx1s, tx2s, np.array(labels), np.array(tx1_lens), np.array(tx2_lens)
        
    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            print("current epoch is {}".format(epoch))
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]


class MSRPGenerator(SemDataGenerator):
    def __init__(self, config):
        super(MSRPGenerator, self).__init__(config)

    def load_data(self, file_name, with_label=True):
        
        ids = []
        tx1s = []
        tx2s = []
        labels = []
        head = True
        tx1_lens = []
        tx2_lens = []
        with open(file_name) as f:
            for line in f:
                if head:
                    head = False
                    continue
                tokens = line.strip().split("\t")
                ids.append([tokens[1], tokens[2]])
                tx1s.append(self.data_preproces(tokens[3]))
                tx2s.append(self.data_preproces(tokens[4]))
                t1_len = len(tokens[1].split())
                t2_len = len(tokens[2].split())
                if t1_len > self.config.max_sent_length:
                    t1_len = self.config.max_sent_length
                if t2_len > self.config.max_sent_length:
                    t2_len = self.config.max_sent_length
                tx1_lens.append(t1_len)
                tx2_lens.append(t2_len)
                if int(tokens[0]) == 0:
                    labels.append([0, 1])
                else:
                    labels.append([1, 0])

        return ids, tx1s, tx2s, np.array(labels), np.array(tx1_lens), np.array(tx2_lens)

class ATECGenerator(SemDataGenerator):
    def __init__(self, config):
        super(ATECGenerator, self).__init__(config)

    def load_data(self, file_name, with_label=True):
        
        ids = []
        tx1s = []
        tx2s = []
        labels = []
        tx1_lens = []
        tx2_lens = []
        with open(file_name) as f:
            for line in f:
                tokens = line.strip().split("\t")
                ids.append([tokens[1], tokens[2]])
                tx1s.append(self.data_preproces(tokens[3]))
                tx2s.append(self.data_preproces(tokens[4]))
                t1_len = len(tokens[1].split())
                t2_len = len(tokens[2].split())
                if t1_len > self.config.max_sent_length:
                    t1_len = self.config.max_sent_length
                if t2_len > self.config.max_sent_length:
                    t2_len = self.config.max_sent_length
                tx1_lens.append(t1_len)
                tx2_lens.append(t2_len)
                if int(tokens[0]) == 0:
                    labels.append([0, 1])
                else:
                    labels.append([1, 0])

        return ids, tx1s, tx2s, np.array(labels), np.array(tx1_lens, dtype=np.int32), np.array(tx2_lens, dtype=np.int32)

