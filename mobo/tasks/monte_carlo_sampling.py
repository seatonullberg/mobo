from mobo.engines import Task

from scipy.stats import gaussian_kde


class KDEMonteCarloTask(Task):

    def __init__(self, kwargs, parallel=False, target=None):
        if target is None:
            target = self.sample
        super().__init__(parallel=parallel,
                         target=target,
                         kwargs=kwargs)

    def sample(self, data, bandwidth, n_samples):
        """
        :param data: an array-like object or list of array-like objects
        :param bandwidth: list of ints or int to estimate KDE bandwidth
        :param n_samples: int to indicate how many samples to pull from data
        """
        if type(data) == list and type(bandwidth) == list:
            samples = []
            for b, d in zip(bandwidth, data):
                for n in range(n_samples):
                    s = self._sample(d, b)
                    samples.append(s)
        elif type(data) != list and type(bandwidth) == list:
            raise TypeError("incompatible types for data and bandwidth")
        elif type(data) == list and type(bandwidth) != list:
            print(self.local_database.keys())
            raise TypeError("incompatible types for data and bandwidth")
        else:
            # both data and bandwidth are individuals
            samples = []
            for n in range(n_samples):
                s = self._sample(data, bandwidth)
                samples.append(s)
        self.set_persistent(key='kde_samples',
                            value=samples)

    # TODO: this is broken
    def _sample(self, data, bandwidth):
        kde = gaussian_kde(data, bandwidth)
        sample = kde.resample()
        return sample
