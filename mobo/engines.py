"""
A collection of iterative data analysis pipelines
"""
from mobo.logging import LogFile

import yaml
from queue import Queue
from threading import Thread


# convenience object to fake a two way queue
class DoubleQueue(object):

    def __init__(self):
        self.client_queue = Queue()
        self.server_queue = Queue()


# convenience object to distinguish strings from persistent keys
class moboKey(object):

    def __init__(self, key, index=None, size=None):
        self.key = key
        self.size = size    # used for forking; indicates the size of an iterable
        self.index = index  # used for forking; gets a particular index of an iterable


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
                if v.index is None:
                    value = self.get_persistent(key=v.key)  # return entire object
                else:
                    value = self.get_persistent(key=v.key)
                    print(value)
                    value = value[v.index]     # return only one index of an iterable
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
        self._forking = False
        self._last_fork_size = None
        self._components = []  # ordered list of tasks

    @property
    def compile(self):
        # return the final pipeline as a list of Tasks
        self.locked = True
        return self._components

    def add_component(self, component):
        # make sure the pipeline is open to additions
        if self.locked:
            raise RuntimeError("The pipeline has already been compiled")

        # process the component
        if isinstance(component, Task):
            if self._forking:
                self._make_fork(task=component)     # fork the Task without kwargs modification
            else:
                self._components.append(component)     # add the Task directly
        elif isinstance(component, Fork):
            self._forking = True
            self._make_fork(task=component.task,
                            iterators=component.iterators)     # fork the Task and modify kwargs
        elif isinstance(component, Join):
            self._forking = False

    def _make_fork(self, task, iterators=None):
        # create all_args ==> [{k0: iter0_0, k1: iter1_0}, {k0: iter0_1, k1: iter1_1},...]
        # number of tasks to produce is len(all_args) OR self._last_fork_size if iterators is None
        if iterators is not None:
            i = 0
            all_args = []
            end = 0
            while len(all_args) < end or end == 0:
                d = {}
                for key in iterators:
                    if isinstance(iterators[key], moboKey):
                        v = iterators[key]
                        v.index = i
                        end = v.size
                    else:
                        v = iterators[key][i]
                        end = len(v)
                    d[key] = v

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
        while i < new_tasks:
            # modify the kwargs of the forked task
            if iterators is not None:
                task.kwargs.update(all_args[i])
            self._components.append(task)
            i += 1
        self._last_fork_size = new_tasks    # update the latest fork size


# Manages a collection of Task objects
class TaskEngine(object):

    def __init__(self):
        self.pipeline = Pipeline()
        self._task_list = []
        self._queue_list = []
        self.database = {}  # used for data persistence
        self.is_complete = False

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

        # start monitoring the tasks with threads and queues
        ordered_task_list = self.pipeline.compile
        for t in ordered_task_list:
            self._queue_list.append(t.queue)

        # iterate through the tasks
        for t in ordered_task_list:
            self.logger.write("Starting Task: {}".format(type(t).__name__))
            # check for parallelization
            if t.parallel:
                if self.configuration['mpi']['use_mpi']:
                    # mpi will automatically do it in parallel
                    t.start()
                else:
                    # do some other parallel method
                    raise NotImplementedError("mpi is currently the only supported parallel processing library")
            else:
                if self.configuration['mpi']['use_mpi']:
                    # use just one mpi rank
                    self.use_single_rank(task=t)
                    self.mpi_comm.WORLD_BARRIER()
                else:
                    # standard single core processing
                    t.start()
            self.logger.write("Completed Task: {}".format(type(t).__name__))
        self.is_complete = True

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
