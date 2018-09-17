"""
A collection of iterative data analysis pipelines
"""
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

    def __init__(self, parallel, index, target, name=None):
        '''
        :param parallel: (bool) set parallel processing preference
        :param index: (int) position in which the task will run relative to others in the same TaskEngine
        :param target: (function) the function to call as a task
        :param name: (str) a user provided name for the task or the name of the target function
        '''
        assert type(parallel) is bool
        self._parallel = parallel

        assert type(index) is int
        self._index = index

        assert type(target) is function
        self._target = target

        if name is None:
            self._name = target.__name__
        elif type(name) is str:
            self._name = name
        else:
            raise TypeError("name must be a string")

        # use to retrieve info from engine
        self.queue = DoubleQueue()

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
    def name(self):
        return self._name

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
        assert type(evaluator) == function
        self.evaluator = evaluator

        self._task_lock = False
        self._task_list = []
        self._queue_list = []
        self.database = {}  # used for data persistence

        self.mpi_comm = None
        self.mpi_rank = None
        self.mpi_size = None
        self.mpi_procname = None

    @property
    def configuration(self):
        try:
            configuration = yaml.load(open("configuration.yaml"))
        except FileNotFoundError:
            # should be custom error
            raise

        return configuration

    def init_from_list(self, evaluator, task_list):
        '''
        initialize the object with an ordered list of Task objects
        :param evaluator: the evaluation function to be passed through
        :param task_list: ordered list of Task objects
        '''
        self.__init__(evaluator)
        for t in task_list:
            self.add_task(t)

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

        # add to queue list
        self._queue_list.append(task.queue)

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

        # mpi setup
        if self.configuration['mpi']['use_mpi']:
            self._setup_mpi()

        # monitor tasks for requested information
        thread = Thread(target=self._manage_tasks)
        thread.start()

        # iterate through the tasks
        # pass results forward as kwargs
        for t in ordered_list:
            # check for parallelization
            if t.parallel:
                if self.configuration['mpi']['use_mpi']:
                    # mpi will automatically do it in parallel
                    kwargs = t.target(**kwargs)
                else:
                    # do some other parallel method
                    raise NotImplementedError("mpi is currently the only supported parallel processing library")
            else:
                if self.configuration['mpi']['use_mpi']:
                    # use just one mpi rank
                    kwargs = self.use_single_rank(task=t, kwargs=kwargs)
                    self.mpi_comm.WORLD_BARRIER()
                else:
                    # standard single core processing
                    kwargs = t.target(**kwargs)

        return kwargs

    def use_single_rank(self, task, kwargs):
        '''
        :param task: The Task object to be run on a single MPI rank
        :param kwargs: the kwargs from the prior task
        :return kwargs: the kwargs passed by the target
        '''
        # always run solo tasks on rank 0
        if self.mpi_rank == 0:
            kwargs = task.target(**kwargs)
            return kwargs

    def _setup_mpi(self):
        from mpi4py import MPI
        self.mpi_comm = MPI.COMM_WORLD
        self.mpi_rank = self.mpi_comm.Get_rank()
        self.mpi_size = self.mpi_comm.Get_size()
        self.mpi_procname = MPI.Get_processor_name()

    def _manage_tasks(self):
        while True:
            for q in self._queue_list:
                if not q.client_queue.empty():
                    data = q.client_queue.get()
                    if type(data) == str:   # the task is requesting a value with a key
                        response = self.database[data]
                        q.server_queue.put(response)
                    elif type(data) == tuple:   # the task is adding data to the database dict
                        key = data[0]
                        value = data[1]
                        self.database[key] = value


# Iterate over a collection of task objects
class IterativeTaskEngine(TaskEngine):

    def __init__(self, evaluator, n_iterations):
        '''
        :param evaluator: (function) an evaluation function used to get errors
        '''
        self.evaluator = evaluator
        self.n_iterations = n_iterations
        super().__init__(self.evaluator)

    def start(self, kwargs):
        for n in self.n_iterations:
            kwargs = super().start(**kwargs)

        return kwargs
