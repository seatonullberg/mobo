from mobo.engines import Task

import numpy as np


class MergeFilesTask(Task):

    def __init__(self, kwargs, target=None):
        if target is None:
            target = self.merge_files
        super().__init__(target=target,
                         kwargs=kwargs)

    def merge_files(self):
        pass


class ReadFileTask(Task):

    def __init__(self, kwargs, target=None):
        if target is None:
            target = self.read
        super().__init__(target=target,
                         kwargs=kwargs)

    def read(self, filename):
        """
        :param filename: str of filename to open and convert to array
        """
        assert type(filename) == str
        extension = filename.split('.')[-1]
        if extension == 'npy':
            arr = np.load(filename)
        else:
            raise NotImplementedError("unsupported file type .{}".format(extension))

        self.set_persistent(key='data_from_file',
                            value=arr)


class WriteFileTask(Task):

    def __init__(self, kwargs, target=None):
        if target is None:
            target = self.write
        super().__init__(target=target,
                         kwargs=kwargs)

    def write(self, filename, data):
        """
        :param filename: str of filename to write data to
        :param data: any object to write to file
        """
        assert type(filename) == str
        extension = filename.split('.')[-1]
        if extension == 'npy':
            np.save(file=filename,
                    arr=data)
        else:
            raise NotImplementedError("unsupported file type .{}".format(extension))
        # saves nothing to database
