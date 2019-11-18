from mobo.log import Logger
from threading import Thread
import os


def test_logger_serial():
    logger = Logger()
    logger.log("test")
    assert os.path.exists(logger.path)
    os.remove(logger.path)
