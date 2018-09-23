"""
A collection of iterative data analysis pipelines
"""
from mobo.logging import LogFile

import yaml
from queue import Queue
from threading import Thread


# Convenience object to create two-way queues
class DoubleQueue(object):

    def __init__(self):
        self.client_queue = Queue()
        self.server_queue = Queue()


# Convenience object to distinguish normal strings from persistent keys
class moboKey(object):

    def __init__(self, key):
        self.key = key


# Convenience class to signal behavior in Pipeline
class Fork(object):

    def __init__(self, iterators, task):
        self.iterators = iterators
        self.task = task


# Convenience class to signal behavior in Pipeline
class Join(object):

    def __init__(self, iterators, task):
        self.iterators = iterators
        self.task = task


# Provide a convenient wrapper to lock access after "compilation"
class Pipeline(object):

    def __init__(self):
        self.locked = False
        self._components = []   # ordered list of Task/Fork/Join objects

    def compile_components(self):
        # return the final pipeline as a list
        self.locked = True
        return self._components

    def add_component(self, component):
        assert isinstance(component, Task) or isinstance(component, Fork) or isinstance(component, Join)
        if self.locked:
            raise RuntimeError("The pipeline has already been compiled")    # make custom
        else:
            self._components.append(component)


# A wrapper to conveniently modularize analysis tasks
class Task(object):

    def __init__(self, target, kwargs):
        """
        :param target: (function) the function to call as a task
        :param kwargs: (dict) used to store arguments for target (can contain moboKey objects to pull from DB)
        """
        assert callable(target)
        self._target = target

        # process kwargs
        assert type(kwargs) == dict
        self.kwargs = kwargs

        # use to retrieve data from persistent database in engine
        self.queue = DoubleQueue()

        # use for forking
        self.local_database = None

        # use for forking
        self.is_forking = False

    def start(self):
        kwargs = {}
        for k, v in self.kwargs.items():
            # get the moboKey values at runtime
            if isinstance(v, moboKey):
                value = self.get_persistent(key=v.key)
                kwargs[k] = value
            else:
                kwargs[k] = v
        self.target(**kwargs)

    def get_persistent(self, key):
        """
        :param key: (str) used to get data from persistent database in engine
        """
        assert type(key) == str

        if self.is_forking:
            return self.local_database[key]
        else:
            self.queue.client_queue.put(key)
            while self.queue.server_queue.empty():
                continue
            return self.queue.server_queue.get()

    def set_persistent(self, key, value):
        """
        :param key: (str) the key that value will be stored as in the persistent database
        :param value: any object
        """
        assert type(key) == str

        if self.is_forking:
            self.local_database[key] = value
        else:
            self.queue.client_queue.put((key, value))
            # wait for the setting to be done to prevent KeyErrors later
            while not self.queue.client_queue.empty():
                continue

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
class TaskEngine(Pipeline):

    def __init__(self):
        super().__init__()
        self._task_list = []
        self._queue_list = []
        self.database = {}  # used for data persistence across tasks
        self.local_databases = None   # used for forking
        self.is_complete = False
        self._forking = False
        self._last_fork_size = None

        self.mpi_comm = None
        self.mpi_rank = None
        self.mpi_size = None
        self.mpi_procname = None

        self.logger = LogFile()

        # provide Tasks access to  self.database through a perpetual monitor thread
        thread = Thread(target=self._manage_database)
        thread.start()

    @property
    def configuration(self):
        try:
            configuration = yaml.load(open("configuration.yaml"))
        except FileNotFoundError:
            # should be custom error
            raise

        return configuration

    def start(self):
        # log event
        self.logger.write("Starting the TaskEngine")

        # mpi setup
        if self.configuration['mpi']['use_mpi']:
            self._setup_mpi()

        # iterate through the components
        ordered_component_list = self.compile_components()
        for c in ordered_component_list:
            if isinstance(c, Fork):
                tasks = self._fork(task=c.task, iterators=c.iterators)     # fork and modify kwargs
                threads = []
                for i, t in enumerate(tasks):
                    thread = Thread(target=self._start, args=(t, i))
                    thread.start()
                    threads.append(thread)
                for thread in threads:
                    thread.join()
            elif isinstance(c, Join):
                task = self._join(task=c.task, iterators=c.iterators)   # join and modify kwargs if iterators provided
                self._start(task=task)
            elif isinstance(c, Task):
                if self._forking:
                    tasks = self._fork(task=c)     # fork without modification of kwargs
                    threads = []
                    for i, t in enumerate(tasks):
                        thread = Thread(target=self._start, args=(t, i))
                        thread.start()
                        threads.append(thread)
                    for thread in threads:
                        thread.join()
                else:
                    self._start(c)  # do not fork

        self.is_complete = True
        self.logger.write("TaskEngine complete")

    def _start(self, task, fork_id=None):
        """
        :param task: Task object
        """
        self.logger.write("Starting Task: {}".format(type(task).__name__))
        # add to queue list
        self._queue_list.append(task.queue)

        if self.configuration['mpi']['use_mpi']:
            # assign task to a rank
            self._use_single_rank(task=task, fork_id=fork_id)
            self.mpi_comm.WORLD_BARRIER()
        else:
            # standard single core processing
            task.start()
            if fork_id is not None:
                self.local_databases[fork_id] = task.local_database
        self.logger.write("Completed Task: {}".format(type(task).__name__))

    def _use_single_rank(self, task, fork_id=None):
        """
        :param task: Task object to be run on rank {fork_id}
        :param fork_id: int indicating the rank to use for the task
        """
        if fork_id is None:
            fork_id = 0
        if self.mpi_rank == fork_id:
            task.start()
            self.local_databases[fork_id] = task.local_database

    def _setup_mpi(self):
        # log event
        self.logger.write("Setting up MPI environment")

        from mpi4py import MPI
        self.mpi_comm = MPI.COMM_WORLD
        self.mpi_rank = self.mpi_comm.Get_rank()
        self.mpi_size = self.mpi_comm.Get_size()
        self.mpi_procname = MPI.Get_processor_name()

    def _manage_database(self):
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

    def _fork(self, task, iterators=None):
        """
        :param task: Task object
        :param iterators: (dict) contains string keys and iterator values that will be
                                 split amongst len(iterator) tasks
        """
        # create all_args ==> [{k0: iter0_0, k1: iter1_0}, {k0: iter0_1, k1: iter1_1},...]
        # number of tasks to produce is len(all_args) OR self._last_fork_size if iterators is None
        self._forking = True
        all_args = []
        if iterators is not None:
            i = 0
            end = -1
            while len(all_args) < end or end == -1:
                d = {}
                _full_iterators = []
                for key, value in iterators.items():
                    if isinstance(value, moboKey):
                        value = self.database[value.key]
                    _full_iterators.append(value)
                    end = len(value)
                    value = value[i]
                    d[key] = value

                    if len(d) >= len(iterators):
                        i += 1
                        all_args.append(d)
                        d = {}

                # verify that all iterators are the same size
                # this assertion is not optimally positioned
                assert all(len(i) == len(_full_iterators[0]) for i in _full_iterators)

        # determine number of tasks to make
        if iterators is None:
            n_tasks = self._last_fork_size
        else:
            n_tasks = len(all_args)

        # add len(all_args) tasks to self._components
        if self.local_databases is None:
            self.local_databases = {i: self.database for i in range(n_tasks)}

        # TODO: fix how the tasks kwargs are assigned
        # right now the last value is repeated for every member of tasks
        # create the new tasks
        i = 0
        tasks = []
        while i < n_tasks:
            # make new instance of Task
            _task = task.__class__(kwargs=task.kwargs)
            self._queue_list.append(_task.queue)
            # modify the kwargs of the forked task
            if iterators is not None:
                _task.kwargs.update(all_args[i])
            # indicate forked status
            _task.is_forking = True
            # set proper local database
            _task.local_database = self.local_databases[i]
            # store initialized task
            tasks.append(_task)
            i += 1

        self._last_fork_size = n_tasks    # update the latest fork size
        return tasks

    def _join(self, task, iterators=None):
        """
        :param task: Task object
        :param iterators: (dict) contains string keys and iterator values that will be
                                 joined from prior tasks in fork
        """
        self._forking = False
        joined_dict = {}
        if iterators is not None:
            for key, value in iterators.items():
                # k is the key in the local database
                # key is the key in the kwargs
                if isinstance(value, moboKey):
                    k = value.key
                else:
                    k = key
                joined_dict[key] = []
                for i, d in self.local_databases.items():
                    joined_dict[key].append(d[k])

        # modify the kwargs of the task
        task.kwargs.update(joined_dict)
        return task


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
