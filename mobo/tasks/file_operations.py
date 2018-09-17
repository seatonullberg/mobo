from mobo.engines import Task


class MergeFilesTask(Task):

    def __init__(self, index, data_key=None, args=None):
        super().__init__(parallel=False,
                         index=index,
                         target=self.merge_files,
                         data_key=data_key,
                         args=args)

    def merge_files(self):
        pass
