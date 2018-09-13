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
        pass

    def add_task(self, task):
        # modify the workflow with a new task object
        assert type(task) is Task


# Controls the operations between each iteration
# Manages a CalculationFramework or a series of CalculationFrameworks
class IterativeFramework(object):

    def __init__(self):
        pass

    def add_framework(self, framework):
        # modify the iteration queue with a new CalculationFramework
        assert type(framework) is CalculationFramework





