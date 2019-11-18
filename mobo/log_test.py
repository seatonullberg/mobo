from mobo.log import Logger
import os


def test_logger_serial():
    logger = Logger()
    logger.log("test")
    assert os.path.exists(logger.path)
    os.remove(logger.path)
