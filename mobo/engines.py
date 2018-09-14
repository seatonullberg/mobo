"""
A collection of iterative data analysis pipelines
"""
import yaml


# A wrapper to conveniently modularize analysis tasks
class Task(object):

    def __init__(self, parallel, index, target):
        '''
        :param parallel: (bool) set parallel processing preference
        :param index: (int) position in which the task will run relative to others in the same TaskEngine
        :param target: (function) the function to call as a task
        '''
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
        '''
        :param evaluator: (function) an evaluation function used to get errors
        '''
        assert type(evaluator) == function
        self.evaluator = evaluator

        self._task_lock = False
        self._task_list = []

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

        # mpi setup
        if self.configuration['mpi']['use_mpi']:
            self._setup_mpi()

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
