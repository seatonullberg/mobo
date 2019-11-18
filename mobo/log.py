from datetime import datetime


class Logger(object):
    """A centralized log file.

    Args:
        path: Path to the log file.
    """
    def __init__(self, path: str = "mobo.log") -> None:
        self.path = path

    def log(self, msg: str) -> None:
        """Writes a message to the log file.
        
        Args:
            msg: The message to write.
        """
        now = datetime.now()
        msg = "{}\n{}\n\n".format(now, msg)
        with open(self.path, "a") as f:
            f.write(msg)
