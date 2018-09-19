from mobo.engines import Task


class MergeFilesTask(Task):

    def __init__(self, index, kwargs, parallel=False, target=None):
        if target is None:
            target = self.merge_files
        super().__init__(parallel=parallel,
                         index=index,
                         target=target,
                         kwargs=kwargs)

    def merge_files(self):
        pass
