import logging
import datetime
import os


class Logger:
    def __init__(self, name, task='', session_id='', format='%(message)s'):
        self.name = name
        self.format = format
        self.level = logging.INFO
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        self.path_to_log = os.path.join('log', task, session_id, name + '.log')

        # Logger configuration
        self.formatter = logging.Formatter(self.format)
        self.file_handler = logging.FileHandler(self.path_to_log)
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(self.formatter)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.stream_handler)
        self.logger.addHandler(self.file_handler)

        self.date_format = '%Y-%m-%d %H:%M'

    def info(self, msg):
        self.logger.info(msg)

    def start_info(self, hyperparameters=None, optim=None, scheduler=None, transforms=None, comment=''):
        """Logging external information about training.
        Saving all hyperparameters, optimizer info, scheduler info, transfromations used,
        start time and general comment.
        """
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
            if transforms.__class__.__module__.split('.')[0] == 'torchvision':
                transforms = transforms.transforms
            for transform in transforms:
                msg += transform.__str__() + '\n'
            msg += '___'
        self.info(msg)

    def epoch_info(self, epoch, train_loss, val_loss, val_metrics):
        """Logging per epoch information.
        Saving epoch number, all loss components and total loss during training phase
        and validation loss, validation metrics during validation phase.
        """
        msg = 'Epoch: ' + str(epoch) + '\n'
        msg += '\tTrain loss:\n'
        msg += str(train_loss) + '\n'
        msg += '\tValidation loss:\n'
        msg += str(val_loss) + '\n'
        msg += '\tValidation metrics:\n'
        msg += str(val_metrics) + '\n'
        self.info(msg)
