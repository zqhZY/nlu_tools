from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class SentSemTrainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(SentSemTrainer, self).__init__(sess, model, config,logger, data_loader=data)

    def train_epoch(self):
        loop = tqdm(range(self.data_loader.num_batches_per_epoch_train))
        print(self.data_loader.num_batches_per_epoch_train)
        losses = []
        accs = []
        for _ in loop:
            loss = self.train_step()
            losses.append(loss)
            #accs.append(acc)
        #for batch in self.data_loader.train_data_iter:
        #    _, x1, x2, y = zip(*batch)
        #    y = np.squeeze(y)
        #    loss = self.train_step(x1, x2, y)
        #    losses.append(loss)
        loss = np.mean(losses)
        #acc = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            #'acc': acc,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        batch = next(self.data_loader.train_data_iter)
        _, x1, x2, y = zip(*batch)
        y = np.squeeze(y)
        feed_dict = {self.model.x1: x1, self.model.x2: x2, self.model.y: y, self.model.is_training: True}
        _, loss = self. sess.run([self.model.train_step, self.model.cost],
                                     feed_dict=feed_dict)
        return loss
