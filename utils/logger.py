import logging
import datetime


class Logger:
    def __init__(self, name, format='%(message)s'):
        self.name = name
        self.format = format
        self.level = logging.INFO
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)

        # Logger configuration
        self.formatter = logging.Formatter(self.format)
        self.file_handler = logging.FileHandler('log/' + name + '.log')
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(self.formatter)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.stream_handler)
        self.logger.addHandler(self.file_handler)

        self.date_format = '%Y-%m-%d %H:%M'

    def start_info(self, optim=None, scheduler=None, comment=''):
        today = datetime.datetime.today()
        msg = '\n...........\n'
        msg += '[' + today.strftime(self.date_format) + '] '
        msg += comment + '\n'
        msg += 'Start training:\n'
        if optim:
            msg += 'Optimizer: ' + str(optim.__class__).split('.')[-1][:-2] + '\n'
            msg += str(optim.state_dict()['param_groups'][0]) + '\n'
        if scheduler:
            msg += 'Scheduler: ' + str(scheduler.__class__).split('.')[-1][:-2] + '\n'
            msg += str(scheduler.state_dict()) + '\n'
        self.logger.info(msg)

    def epoch_info(self, epoch, train_loss):
        msg = 'Epoch: ' + str(epoch) + ' Train loss: ' + str(train_loss)
        self.logger.info(msg)
