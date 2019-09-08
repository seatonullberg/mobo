from datetime import datetime
from filelock import FileLock
import os


class Logger(object):
    """Implementation of a centralized log file.

    Args:
        path (optional) (str): Path to the log file.
    """
    def __init__(self, path="mobo.log"):
        assert type(path) is str
        self._path = path

    @property
    def path(self):
        return self._path

    def log(self, msg):
        """Writes a message to the log file.
        
        Args:
            msg (str): The message to write.
        """
        assert type(msg) is str
        lock = FileLock(self.path + ".lock")
        with lock:
            now = datetime.now()
            msg = "{}\n{}\n\n".format(now, msg)
            with open(self.path, "a") as f:
                f.write(msg)
        os.remove(self.path + ".lock")
