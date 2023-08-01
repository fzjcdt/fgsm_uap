
import random
import logging


class Logger(object):
    """
    Helper class for logging.
    Arguments:
        path (str): Path to log file.
    """

    def __init__(self, path: str):
        # random string to avoid overwriting logs
        logger_name = path + ''.join(random.choice('0123456789ABCDEFG') for i in range(16))
        self.logger = logging.getLogger(logger_name)
        self.path = path
        self.setup_file_logger()

    def setup_file_logger(self):
        hdlr = logging.FileHandler(self.path, 'a')
        self.logger.addHandler(hdlr)
        self.logger.setLevel(logging.INFO)

    def log(self, message: str):
        self.logger.info(message)

    def new_line(self):
        self.logger.info('')