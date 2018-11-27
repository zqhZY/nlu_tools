from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class ClassifyTrainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(ClassifyTrainer, self).__init__(sess, model, config,logger, data_loader=data)
        self.max_acc = 0

    def train(self):
        """
        This is the main loop of training override.
        Looping on the epochs
        :return:
        """
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs, 1):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)
            self.eval(cur_epoch)

    def train_epoch(self):
        loop = tqdm(range(self.data_loader.num_batches_per_epoch_train))
        print(self.data_loader.num_batches_per_epoch_train)
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'train/loss': loss,
            'train/acc': acc,
        }
        print("Train- loss:{:.4f}   acc:{:.4f}".format(loss, acc))
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        batch = next(self.data_loader.train_data_iter)
        _, x, x_char, x_pinyin, y, x_mask= zip(*batch)
        y = np.squeeze(y)
        if self.config.using_actual_len:
            feed_dict = {self.model.x: x, self.model.x_char: x_char, self.model.y: y, self.model.is_training: True, self.model.x_mask: x_mask}
        else:
            feed_dict = {self.model.x: x, self.model.x_char: x_char, self.model.y: y, self.model.is_training: True}
        _, loss, acc = self. sess.run([self.model.train_step, self.model.cost, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc

    def eval(self, cur_epoch):
        # load eval data iter
        eval_data_iter = self.data_loader.get_dev_iter()
        losses = []
        accs = []
        for batch in eval_data_iter:
           _, x, x_char, x_pinyin, y, x_mask = zip(*batch)
           y = np.squeeze(y)
           if self.config.using_actual_len:
               feed_dict = {self.model.x: x, self.model.x_char: x_char, self.model.y: y, self.model.is_training: True, self.model.x_mask: x_mask}
           else:
               feed_dict = {self.model.x: x, self.model.x_char: x_char, self.model.y: y, self.model.is_training: True}
           #feed_dict = {self.model.x1: x1, self.model.x2: x2, self.model.y: y, self.model.is_training: False}
           if self.config.lr_decay:
               loss, acc, lr = self.sess.run([self.model.cost, self.model.accuracy, self.model.lr], feed_dict=feed_dict)
               print("lr now is {}".format(lr))
           else:
               loss, acc = self.sess.run([self.model.cost, self.model.accuracy], feed_dict=feed_dict)
           losses.append(loss)
           accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)
        summaries_dict = {
            'eval/loss': loss,
            'eval/acc': acc,
        }
        cur_it = self.model.global_step_tensor.eval(self.sess)
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        print("Val-{}  loss:{:.4f}   acc:{:.4f}".format(cur_epoch, loss, acc))
        if acc > self.max_acc:
            self.max_acc = acc
            print("epoch-{} new acc value loss:{:.4f}   acc:{:.4f}".format(cur_epoch, loss, self.max_acc))
        
