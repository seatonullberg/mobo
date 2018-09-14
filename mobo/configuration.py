"""
Configuration object for user defined tweaks to the implementation
"""
import yaml


class BaseConfiguration(dict):

    def __init__(self):
        super().__init__()

    def to_yaml(self):
        yaml.dump(self, open('configuration.yaml', 'w'))


class Configuration(object):

    def __init__(self, custom_config=None):
        if type(custom_config) is None or type(custom_config) == dict:
            pass
        else:
            raise TypeError("custom_config must be dict or None")

        self.custom_config = custom_config

    @property
    def default(self):
        # build a default
        d = BaseConfiguration()
        d = default_options(d)
        # write the yaml file
        d.to_yaml()
        return d

    @property
    def custom(self):
        # load default
        d = self.default
        # add tweaks
        for key, value in self.custom_config:
            d[key] = value
        # write the yaml file
        d.to_yaml()
        return d


"""
===============================
Very ugly default configuration 
===============================
"""


def default_options(d):
    # clustering techniques
    d['clustering'] = {}
    d['clustering']['dbscan'] = {}
    d['clustering']['dbscan']['args'] = {}
    # manifold learning techniques
    d['manifold_learning'] = {}
    d['manifold_learning']['tsne'] = {}
    d['manifold_learning']['tsne']['args'] = {}
    # logging statements
    d['logging'] = {}
    d['logging']['do_logging'] = True
    d['logging']['filename'] = 'mobo.log'
    # mpi options
    d['mpi'] = {}
    d['mpi']['use_mpi'] = True
    return d
