"""
A collection of iterative data analysis pipelines
"""
from mobo.logging import LogFile

import yaml
from queue import Queue
from threading import Thread


# convenient object to fake a two way queue
class DoubleQueue(object):

    def __init__(self):
        self.client_queue = Queue()
        self.server_queue = Queue()


# A wrapper to conveniently modularize analysis tasks
class Task(object):

    def __init__(self, parallel, index, target, data_key=None, args=None):
        """
        :param parallel: (bool) set parallel processing preference
        :param index: (int) position in which the task will run relative to others in the same TaskEngine
        :param target: (function) the function to call as a task
        :param data_key (str) key used to retrieve data from persistent DB in TaskEngine
        """
        assert type(parallel) == bool
        self._parallel = parallel

        assert type(index) == int
        self._index = index

        assert callable(target)
        self._target = target

        # process optional args
        self.data_key = data_key
        self.args = args

        # use to retrieve info from engine
        self.queue = DoubleQueue()

    def start(self):
        if self.data_key is None:
            args = self.args
        else:
            args = self.get_persistent(self.data_key)
        self.target(args)

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
    def index(self):
        return self._index

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


# Manages a collection of Task objects
class TaskEngine(object):

    def __init__(self, evaluator):
        '''
        :param evaluator: (function) an evaluation function used to get errors
        '''
        assert callable(evaluator)
        self.evaluator = evaluator

        self._task_lock = False
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

    def init_from_list(self, evaluator, task_list):
        """
        initialize the object with an ordered list of Task objects
        :param evaluator: the evaluation function to be passed through
        :param task_list: ordered list of Task objects
        """
        self.__init__(evaluator)
        for t in task_list:
            self.add_task(t)

    def add_task(self, task):
        # modify the workflow with a new task object
        assert isinstance(task, Task)  # should be custom error

        # make sure the object can accept new tasks
        if self._task_lock:
            raise ValueError    # this should be a custom error

        # make sure the index does not conflict with that of another task
        for t in self._task_list:
            if task.index == t.index:
                raise ValueError    # should be a custom error

        # add to queue list
        self._queue_list.append(task.queue)
        # add to task list
        self._task_list.append(task)
        # log event
        self.logger.write("Added {} to TaskEngine".format(type(task).__name__))

    def start(self):
        # log event
        self.logger.write("Starting the TaskEngine")

        # lock the state of the object
        self._task_lock = True

        # order the tasks
        ordered_list = sorted(self._task_list,
                              key=lambda x: x.index,
                              reverse=False)

        # mpi setup
        if self.configuration['mpi']['use_mpi']:
            self._setup_mpi()

        # iterate through the tasks
        # pass results forward as kwargs
        for t in ordered_list:
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

    def __init__(self, evaluator, n_iterations):
        """
        :param evaluator: (function) an evaluation function used to get errors
        """
        self.evaluator = evaluator
        self.n_iterations = n_iterations
        super().__init__(self.evaluator)

    def start(self):
        for n in self.n_iterations:
            super().start()
