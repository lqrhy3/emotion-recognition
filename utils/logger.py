import logging


class Logger:
    def __init__(self, name, format='%(message)s'):
        self.name = name
        self.format = format
        self.level = logging.INFO
        self.logger = logging.getLogger(name)

        # Logger configuration
        self.formatter = logging.Formatter(self.format)
        self.file_handler = logging.FileHandler('log/' + name + '.log')
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(self.formatter)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.stream_handler)
        self.logger.addHandler(self.file_handler)

    def info(self, epoch, train_loss, train_metric=None):
        msg = 'Epoch: ' + str(epoch) + ' Loss: ' + str(train_loss)
        self.logger.info(msg)
