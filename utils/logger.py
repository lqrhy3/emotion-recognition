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

    def start_info(self, hyperparameters=None, optim=None, scheduler=None, transforms=None, comment=''):
        today = datetime.datetime.today()
        msg = '\n................................\n'
        msg += '[' + today.strftime(self.date_format) + '] '
        msg += comment + '\n'
        msg += 'Train info:\n'
        msg += '___\n'

        if hyperparameters:
            msg += 'Hyperparameters:\n'
            for key in hyperparameters:
                msg += key + ': ' + str(hyperparameters[key]) + '\n'
            msg += '___\n'
        if optim:
            msg += 'Optimizer: ' + str(optim.__class__.__name__) + '\n'
            msg += str(optim.state_dict()['param_groups'][0]) + '\n'
            msg += '___\n'
        if scheduler:
            msg += 'Scheduler: ' + str(scheduler.__class__.__name__) + '\n'
            msg += str(scheduler.state_dict()) + '\n'
            msg += '___\n'

        if transforms:
            msg += 'Train transformations:\n'
            msg += transforms.__str__()[transforms.__str__().find('[')+1:transforms.__str__().find(']')].strip() + '\n'
            msg += '___'
        self.logger.info(msg)

    def epoch_info(self, epoch, train_loss):
        msg = 'Epoch: ' + str(epoch) + '\n'  # + '  Total loss: ' + str(train_loss['Total loss']) + '\n'
        for key in train_loss:
            msg += '\t' + key + ': ' + str(train_loss[key]) + '\n'
        self.logger.info(msg)
