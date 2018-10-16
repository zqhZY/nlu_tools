from tensorflow.contrib import learn
import numpy as np
import jieba


class SemDataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        self.ids_train, self.tx1s_train, self.tx2s_train, self.labels_train = self.load_data(self.config.train_data, True)
        self.ids_dev, self.tx1s_dev, self.tx2s_dev, self.labels_dev = self.load_data(self.config.dev_data, True)

        # tokenize data
        if self.config.tokenized is not True:
            self.tx1s_train = [" ".join(jieba.lcut(x)) for x in self.tx1s_train]
            self.tx2s_train = [" ".join(jieba.lcut(x)) for x in self.tx2s_train]
            self.tx1s_dev = [" ".join(jieba.lcut(x)) for x in self.tx1s_dev]
            self.tx2s_dev = [" ".join(jieba.lcut(x)) for x in self.tx2s_dev]

        # build voc dict
        if self.config.load_voc:
            self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(self.config.voc_path)
        else:
            self.vocab_processor = learn.preprocessing.VocabularyProcessor(self.config.max_sent_length)
            self.vocab_processor.fit(self.tx1s_train + self.tx2s_train)
            self.vocab_processor.save(self.config.voc_path)
            # save word dict
            self.save_word_dict(self.vocab_processor.vocabulary_._mapping)
    
        # transform data to word ids
        print("vocab size is {}".format(len(self.vocab_processor.vocabulary_)))
        self.tx1s_train = np.array(list(self.vocab_processor.transform(self.tx1s_train)))
        self.tx2s_train = np.array(list(self.vocab_processor.transform(self.tx2s_train)))
        self.tx1s_dev = np.array(list(self.vocab_processor.transform(self.tx1s_dev)))
        self.tx2s_dev = np.array(list(self.vocab_processor.transform(self.tx2s_dev)))

        print("train data num is {}".format(len(self.tx1s_train)))
        print("dev data num is {}".format(len(self.tx1s_dev)))
        self.num_batches_per_epoch_train = int((len(self.tx1s_train)-1)/self.config.batch_size) + 1
        self.num_batches_per_epoch_dev = int((len(self.tx1s_dev)-1)/self.config.batch_size) + 1

        # build data iter
        self.train_data_iter = self.batch_iter(list(zip(self.ids_train, self.tx1s_train, self.tx2s_train, self.labels_train)), \
                                               self.config.batch_size, \
                                               self.config.num_epochs, \
                                               self.config.shuffle_data)
        self.dev_data_iter = self.batch_iter(list(zip(self.ids_dev, self.tx1s_dev, self.tx2s_dev, self.labels_dev)), \
                                             self.config.batch_size, \
                                             self.config.num_epochs, \
                                             False)
        #self.input = np.ones((500, 784))
        #self.y = np.ones((500, 10))

    def save_word_dict(self, word_dict):
        with open(self.config.word_dict_path, "w") as f:
            for k, v in word_dict.items():
                f.write(str(v) + "\t" + k + "\n")

    def load_data(self, file_name, with_label=True):
        
        ids = []
        tx1s = []
        tx2s = []
        labels = []
        with open(file_name) as f:
            for line in f:
                tokens = line.strip().split("\t")
                ids.append([tokens[0]])
                tx1s.append(tokens[1])
                tx2s.append(tokens[2])
                labels.append([float(tokens[3])])
        return ids, tx1s, tx2s, np.array(labels)
        
    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]

    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
        for epoch in range(num_epochs+1):
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

