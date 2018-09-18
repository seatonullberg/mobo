from mobo.engines import Task


class MergeFilesTask(Task):

    def __init__(self, index, kwargs):
        super().__init__(parallel=False,
                         index=index,
                         target=self.merge_files,
                         kwargs=kwargs)

    def merge_files(self):
        pass
