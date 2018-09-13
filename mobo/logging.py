"""
LogFile with locking to support multiprocessing
"""

import fcntl
import errno


class LogFile(object):

    def __init__(self, filename='mobo.log'):

        self.filename = filename
        self.msg_format = "{}\n"

    def write(self, s):

        with open(self.filename, 'a') as f:

            # attempt to get lock
            while True:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except IOError as e:
                    if e.errno != errno.EAGAIN:
                        raise
                    else:
                        continue

            # write message
            f.write(self.msg_format.format(s))

            # release the lock
            fcntl.flock(f, fcntl.LOCK_UN)
