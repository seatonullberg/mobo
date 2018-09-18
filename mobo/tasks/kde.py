from mobo.engines import Task
from mobo.kde import silverman_h, chiu_h


class KDEBandwidthTask(Task):

    def __init__(self, index, kwargs):
        super().__init__(parallel=False,
                         index=index,
                         target=self.calculate_bandwidth,
                         kwargs=kwargs)

    def calculate_bandwidth(self, data):
        """
        :param data: an array-like object or list of array-like objects
        """
        bandwidth_type = self.configuration['kde']['bandwidth_type']

        if bandwidth_type == 'chiu':
            if type(data) == list:
                bandwidth = [chiu_h(d) for d in data]
            else:
                bandwidth = chiu_h(data)
        elif bandwidth_type == 'silverman':
            if type(data) == list:
                bandwidth = [silverman_h(d) for d in data]
            else:
                bandwidth = silverman_h(data)
        else:
            raise ValueError("bandwidth_type must be chiu or silverman")    # custom error

        self.set_persistent(key='kde_bandwidth',
                            value=bandwidth)
