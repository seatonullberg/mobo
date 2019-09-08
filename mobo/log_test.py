from mobo.log import Logger
from threading import Thread
import os


def test_logger_serial():
    logger = Logger()
    logger.log("test")
    assert os.path.exists(logger.path)
    os.remove(logger.path)


def test_logger_parallel():
    logger = Logger()

    def inner_log(lgr, m):
        lgr.log(m)

    t0 = Thread(target=inner_log, args=(logger, "test0"))
    t1 = Thread(target=inner_log, args=(logger, "test1"))
    t0.start()
    t1.start()
    t0.join()
    t1.join()
    assert os.path.exists(logger.path)
    os.remove(logger.path)
