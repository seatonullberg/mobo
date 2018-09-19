"""
A collection of iterative data analysis pipelines
"""
from mobo.logging import LogFile

import yaml
import time
from queue import Queue
from threading import Thread


# convenience object to fake a two way queue
class DoubleQueue(object):

    def __init__(self):
        self.client_queue = Queue()
        self.server_queue = Queue()


# convenience object to distinguish strings from persistent keys
class moboKey(object):

    def __init__(self, key):
        self.key = key


# A wrapper to conveniently modularize analysis tasks
class Task(object):

    def __init__(self, parallel, target, kwargs):
        """
        :param parallel: (bool) set parallel processing preference
        :param target: (function) the function to call as a task
        :param kwargs: (dict) used to store arguments for target (can contain Key objects to pull from DB)
        """
        assert type(parallel) == bool
        self._parallel = parallel

        assert callable(target)
        self._target = target

        # process kwargs
        assert type(kwargs) == dict
        self.kwargs = kwargs

        # use to retrieve info from engine
        self.queue = DoubleQueue()

    def start(self):
        kwargs = {}
        for k, v in self.kwargs.items():
            if isinstance(v, moboKey):
                value = self.get_persistent(key=v.key)  # return entire object
                kwargs[k] = value
            else:
                kwargs[k] = v
        self.target(**kwargs)

    def get_persistent(self, key):
        assert type(key) == str
        self.queue.client_queue.put(key)
        while self.queue.server_queue.empty():
            continue
        return self.queue.server_queue.get()

    def set_persistent(self, key, value):
        assert type(key) == str
        self.queue.client_queue.put((key, value))
        # the rest is handled in the engine

    @property
    def parallel(self):
        return self._parallel

    @property
    def target(self):
        return self._target

    @property
    def configuration(self):
        try:
            configuration = yaml.load(open("configuration.yaml"))
        except FileNotFoundError:
            # should be custom error
            raise

        return configuration


# convenience class to signal behavior in Pipeline
class Fork(object):

    def __init__(self, iterators, task):
        self.iterators = iterators
        self.task = task


# convenience class to signal behavior in Pipeline
class Join(object):

    def __init__(self):
        pass


# compile Tasks, Forks, and Joins into a useful task list
class Pipeline(object):

    def __init__(self):
        self.locked = False
        self.queue = DoubleQueue()
        self._last_fork_size = None
        self._components = []  # ordered list of tasks

    @property
    def compile(self):
        # return the final pipeline as a list of Tasks
        self.locked = True
        return self._components

    def add_component(self, component):
        assert isinstance(component, Task) or isinstance(component, Fork) or isinstance(component, Join)
        if self.locked:
            raise RuntimeError("The pipeline has already been compiled")
        else:
            self._components.append(component)

    def get_persistent(self, key):
        assert type(key) == str
        self.queue.client_queue.put(key)
        while self.queue.server_queue.empty():
            continue
        return self.queue.server_queue.get()

    def make_fork(self, task, iterators=None):
        # create all_args ==> [{k0: iter0_0, k1: iter1_0}, {k0: iter0_1, k1: iter1_1},...]
        # number of tasks to produce is len(all_args) OR self._last_fork_size if iterators is None
        if iterators is not None:
            i = 0
            all_args = []
            end = 0
            while len(all_args) < end or end == 0:
                d = {}
                for key, value in iterators.items():
                    if isinstance(value, moboKey):
                        value = self.get_persistent(value.key)

                    end = len(value)
                    value = value[i]
                    d[key] = value

                    if len(d) >= len(iterators):
                        i += 1
                        all_args.append(d)
                        d = {}

        # determine number of tasks to make
        if iterators is None:
            new_tasks = self._last_fork_size
        else:
            new_tasks = len(all_args)

        # add len(all_args) tasks to self._components
        i = 0
        tasks = []
        while i < new_tasks:
            # modify the kwargs of the forked task
            if iterators is not None:
                task.kwargs.update(all_args[i])
            tasks.append(task)
            i += 1
        self._last_fork_size = new_tasks    # update the latest fork size

        return tasks


# Manages a collection of Task objects
class TaskEngine(object):

    def __init__(self):
        self._task_list = []
        self._queue_list = []
        self.database = {}  # used for data persistence
        self.is_complete = False
        self._forking = False

        self.pipeline = Pipeline()
        self._queue_list.append(self.pipeline.queue)

        self.mpi_comm = None
        self.mpi_rank = None
        self.mpi_size = None
        self.mpi_procname = None

        self.logger = LogFile()

        # monitor tasks for requested information
        thread = Thread(target=self._manage_tasks)
        thread.start()

    @property
    def configuration(self):
        try:
            configuration = yaml.load(open("configuration.yaml"))
        except FileNotFoundError:
            # should be custom error
            raise

        return configuration

    def add_component(self, component):
        """
        :param component: a Task/Fork/Join object
        """
        # modify the workflow with a new object
        assert isinstance(component, Task) or isinstance(component, Fork) or isinstance(component, Join)
        self.pipeline.add_component(component)

    def start(self):
        # log event
        self.logger.write("Starting the TaskEngine")

        # mpi setup
        if self.configuration['mpi']['use_mpi']:
            self._setup_mpi()

        # iterate through the components
        ordered_component_list = self.pipeline.compile
        for c in ordered_component_list:
            if isinstance(c, Fork):
                self._forking = True
                time.sleep(0.1)     # TODO: why does this prevent the program from crashing
                tasks = self.pipeline.make_fork(task=c.task, iterators=c.iterators)   # fork with modified kwargs
                for t in tasks:
                    self._start(t)
            elif isinstance(c, Join):
                self._forking = False
            elif isinstance(c, Task):
                if self._forking:
                    tasks = self.pipeline.make_fork(task=c)     # fork without modified kwargs
                    for t in tasks:
                        self._start(t)
                else:
                    self._start(c)  # do not fork

        self.is_complete = True
        self.logger.write("TaskEngine complete")

    def _start(self, task):
        self.logger.write("Starting Task: {}".format(type(task).__name__))
        # add to queue list
        self._queue_list.append(task.queue)
        # check for parallelization
        if task.parallel:
            if self.configuration['mpi']['use_mpi']:
                # mpi will automatically do it in parallel
                task.start()
            else:
                # do some other parallel method
                raise NotImplementedError("mpi is currently the only supported parallel processing library")
        else:
            if self.configuration['mpi']['use_mpi']:
                # use just one mpi rank
                self.use_single_rank(task=task)
                self.mpi_comm.WORLD_BARRIER()
            else:
                # standard single core processing
                task.start()
        self.logger.write("Completed Task: {}".format(type(task).__name__))

    def use_single_rank(self, task):
        """
        :param task: The Task object to be run on a single MPI rank
        """
        # always run solo tasks on rank 0
        if self.mpi_rank == 0:
            task.start()

    def _setup_mpi(self):
        # log event
        self.logger.write("Setting up MPI environment")

        from mpi4py import MPI
        self.mpi_comm = MPI.COMM_WORLD
        self.mpi_rank = self.mpi_comm.Get_rank()
        self.mpi_size = self.mpi_comm.Get_size()
        self.mpi_procname = MPI.Get_processor_name()

    def _manage_tasks(self):
        # log event
        self.logger.write("Starting the task manager")

        while not self.is_complete:
            for q in self._queue_list:
                if not q.client_queue.empty():
                    data = q.client_queue.get()
                    if type(data) == str:   # the task is requesting a value with a key
                        response = self.database[data]
                        q.server_queue.put(response)
                        self.logger.write("task manager returned: {}".format(data))
                    elif type(data) == tuple:   # the task is adding data to the database dict
                        key = data[0]
                        value = data[1]
                        self.database[key] = value
                        self.logger.write("task manager stored: {}".format(key))


# Iterate over a collection of task objects
class IterativeTaskEngine(TaskEngine):

    def __init__(self, n_iterations):
        """
        :param n_iterations: (int) number of iterations to complete
        """
        self.n_iterations = n_iterations
        super().__init__()

    def start(self):
        for n in self.n_iterations:
            super().start()
