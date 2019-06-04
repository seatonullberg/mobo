

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
