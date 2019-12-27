import logging
import datetime


class Logger:
    def __init__(self, name, session_id='', format='%(message)s'):
        self.name = name
        self.format = format
        self.level = logging.INFO
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)

        # Logger configuration
        self.formatter = logging.Formatter(self.format)
        self.file_handler = logging.FileHandler('log/' + session_id + '/' + name + '.log')
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(self.formatter)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.stream_handler)
        self.logger.addHandler(self.file_handler)

        self.date_format = '%Y-%m-%d %H:%M'

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
            msg += transforms.__str__()[transforms.__str__().find('[')+1:transforms.__str__().find(']')].strip() + '\n'
            msg += '___'
        self.logger.info(msg)

    def epoch_info(self, epoch, loss, val_metrics, phase):
        """Logging per epoch information.
        Saving epoch number, all loss components and total loss during training phase
        and validation loss, validation metrics during validation phase.
        """
        if phase == 'train':
            msg = 'Epoch: ' + str(epoch) + '\n'
            for key in loss:
                msg += '\t' + key + ': ' + str(loss[key]) + '\n'
        elif phase == 'val':
            msg = 'Validation loss: '
            msg += str(loss['Total loss']) + '\n'
            msg += 'Validation IoU' + str(val_metrics) + '\n'
        self.logger.info(msg)
