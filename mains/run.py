import tensorflow as tf
import sys
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score


sys.path.extend(['..'])

from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import * 


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except Exception as e:
        print("missing or invalid arguments %s" % e)
        exit(0)

    # create your data generator
    #data = SemDataGenerator(config)
    #data = MSRPGenerator(config)
    data = import_object(config.task_config.data_generator, config)

    if args.step == "build_data":
        print("build data....")
        data.build_data()
    elif args.step == "train":
        # create the experiments dirs
        create_dirs([config.summary_dir, config.checkpoint_dir])
        # create tensorflow session
        sess = tf.Session()
    
        # load word2vec
        config.embedding_char = data.get_trimmed_glove_vectors(config.char_token_conf.trimmed_embedding) 
        config.embedding_word = data.get_trimmed_glove_vectors(config.word_token_conf.trimmed_embedding) 
        #config.embedding_pinyin = data.get_trimmed_glove_vectors(self.config.pinyin_token_conf.trimmed_embedding) 

        model = import_object(config.task_config.name, config)
        # create tensorboard logger
        logger = Logger(sess, config)
        # create trainer and pass all the previous components to it
        trainer = import_object(config.task_config.trainer, sess, model, data, config, logger)
        #load model if exists
        #model.load(sess)
        # here you train your model
        trainer.train()
    elif args.step == "pred":
        # create tensorflow session
        sess = tf.Session()
        # load word2vec
        config.embedding_char = data.get_trimmed_glove_vectors(config.char_token_conf.trimmed_embedding) 
        config.embedding_word = data.get_trimmed_glove_vectors(config.word_token_conf.trimmed_embedding) 
        # init and load model to pred
        model = import_object(config.task_config.name, config)
        model.load(sess)

        test_ids, test_raw, true_labels, _ = data.test_data
        pred_labels = []
        for i, text_raw in enumerate(test_raw):
            x, x_char, x_pinyin = data.pred_process(text_raw)
            feed_dict = {model.x: x, model.x_char_cnn: x_char, model.x_pinyin: x_pinyin, model.is_training: False}
            pred = sess.run([model.d2], feed_dict=feed_dict)
            pred_labels.append(data.id2label[np.argmax(pred[0])])
        
        # print acc, recall and f1
        acc = accuracy_score(true_labels, pred_labels)
        recall = recall_score(true_labels, pred_labels, average='macro')
        f1 = f1_score(true_labels, pred_labels, average='micro')
        print("acc is {}, recall is {}, f1 is {}".format(acc, recall, f1))
    elif args.step == "tune":

        import itertools
        tune_num = 0
        param_names = config["parameter_tune"].keys()
        print(param_names)
        cand_params = [config["parameter_tune"][pname] for pname in param_names]
        print(cand_params)
        for params in itertools.product(*cand_params):
            print(params)
            for i, param_name in enumerate(param_names):
                config[param_name] = params[i]
            #print(config)
            data = ATECGenerator(config)
            create_dirs([config.summary_dir, config.checkpoint_dir])
            sess = tf.Session()
            config.embedding = data.get_trimmed_glove_vectors() 
            model = ESIM(config)
            logger = Logger(sess, config)
            trainer = SentSemTrainer(sess, model, data, config, logger)
            trainer.train()
            tf.reset_default_graph()
    else:
        print("no support step!!")

if __name__ == '__main__':
    main()
