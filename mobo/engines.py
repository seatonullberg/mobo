"""
A collection of iterative data analysis pipelines
"""


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
class CalculationFramework(object):

    def __init__(self):
        self._task_lock = False
        self._task_list = []

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

    def start(self):
        # lock the state of the object
        self._task_lock = True

        # order the tasks
        ordered_list = sorted(self._task_list,
                              key=lambda x: x.index,
                              reverse=False)

        # iterate through the tasks
        # TODO
        for t in ordered_list:
            t.target()


# Controls the operations between each iteration
# Manages a CalculationFramework or a series of CalculationFrameworks
class IterativeFramework(object):

    def __init__(self):
        self._frame_lock = False
        self._frame_list = []

    def add_framework(self, framework):
        # modify the iteration queue with a new CalculationFramework
        assert type(framework) is CalculationFramework

        # make sure framework is accepting new frameworks
        if self._frame_lock:
            raise ValueError    # should be custom error

        # make sure the indices do not conflict
        for f in self._frame_list:
            if framework.index == f.index:
                raise ValueError    # should be custom error

        self._frame_list.append(framework)

    def start(self):
        # lock the state
        self._frame_lock = True

        # order the frames
        ordered_list = sorted(self._frame_list,
                              key=lambda x: x.index,
                              reverse=False)

        # iterate through frames
        # TODO
        for f in ordered_list:
            f.start()
