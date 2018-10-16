import tensorflow as tf
import sys

sys.path.extend(['..'])

from data_loader.sent_sem_loader import SemDataGenerator 
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

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    data = SemDataGenerator(config)
    
    #for batch in data.train_data_iter:
    #    ids, x1, x2, y = zip(*batch)
    #    print(ids, x1, x2, y)
    #    break
    # create an instance of the model you want
    model = CDSSM(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = SentSemTrainer(sess, model, data, config, logger)
    #load model if exists
    #model.load(sess)
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
