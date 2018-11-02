import tensorflow as tf
import sys

sys.path.extend(['..'])

from data_loader.sent_sem_loader import SemDataGenerator, MSRPGenerator, ATECGenerator
from models.cdssm import CDSSM
from trainers.sent_sem_trainer import SentSemTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args


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
    data = ATECGenerator(config)

    if args.step == "build_data":
        print("build data....")
        data.build_data()
    elif args.step == "train":
        # create the experiments dirs
        create_dirs([config.summary_dir, config.checkpoint_dir])
        # create tensorflow session
        sess = tf.Session()
    
        # load word2vec
        config.embedding = data.get_trimmed_glove_vectors() 

        model = CDSSM(config)
        # create tensorboard logger
        logger = Logger(sess, config)
        # create trainer and pass all the previous components to it
        trainer = SentSemTrainer(sess, model, data, config, logger)
        #load model if exists
        #model.load(sess)
        # here you train your model
        trainer.train()
    else:
        print("no support step!!")

if __name__ == '__main__':
    main()
