import os
import time
from utils import set_seed, Logger


class ATBase(object):

    def __init__(self, model, eps=8.0 / 255, log_dir='./log/', name='', device=None, seed=0):
        set_seed(seed)
        self.model = model
        self.eps = eps
        output_dir = os.path.join(log_dir, name)
        output_dir = os.path.join(output_dir,
                                  time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '-seed-' + str(seed))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.output_dir = output_dir
        log_file = os.path.join(output_dir, 'output.log')
        self.logger = Logger(log_file)

        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = device

    def train(self, opt, scheduler, train_loader, test_loader, epoch):
        raise NotImplementedError
