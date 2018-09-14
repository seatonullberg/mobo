"""
A collection of iterative data analysis pipelines
"""
import yaml


# A wrapper to conveniently modularize analysis tasks
class Task(object):

    def __init__(self, parallel, index, target):
        assert type(parallel) is bool
        self._parallel = parallel

        assert type(index) is int
        self._index = index

        assert type(target) is function
        self._target = target

    @property
    def parallel(self):
        return self._parallel

    @property
    def index(self):
        return self._index

    @property
    def target(self):
        return self._target


# Manages a collection of Task objects
class TaskEngine(object):

    def __init__(self, evaluator):
        assert type(evaluator) == function
        self.evaluator = evaluator

        self._task_lock = False
        self._task_list = []

    @property
    def configuration(self):
        try:
            configuration = yaml.load(open("configuration.yaml"))
        except FileNotFoundError:
            # should be custom error
            raise

        return configuration

    def add_task(self, task):
        # modify the workflow with a new task object
        assert type(task) is Task   # should be custom error

        # make sure the object can accept new tasks
        if self._task_lock:
            raise ValueError    # this should be a custom error

        # make sure the index does not conflict with that of another task
        for t in self._task_list:
            if task.index == t.index:
                raise ValueError    # should be a custom error

        self._task_list.append(task)

    def start(self, kwargs):
        '''
        :param kwargs: dict of args to pass to the first task in the list
        :return: kwargs from the last task in the list
        '''
        # lock the state of the object
        self._task_lock = True

        # order the tasks
        ordered_list = sorted(self._task_list,
                              key=lambda x: x.index,
                              reverse=False)

        # iterate through the tasks
        # pass results on as kwargs
        for t in ordered_list:
            kwargs = t.target(**kwargs)

        return kwargs


# Iterate over a collection of task objects
class IterativeTaskEngine(TaskEngine):

    def __init__(self, evaluator):
        self.evaluator = evaluator
        super().__init__(self.evaluator)
