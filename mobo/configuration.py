"""
Configuration object for user defined tweaks to the implementation
"""


# this object is what will be accepted by the internals as a configuration
class BaseConfiguration(dict):

    def __init__(self):
        super().__init__()


class Configuration(object):

    def __init__(self, custom_config=None):
        if type(custom_config) is None or type(custom_config) == dict:
            pass
        else:
            raise TypeError("custom_config must be dict or None")

        self.custom_config = custom_config

    def _build_configuration(self, custom_configuration):
        pass

    @property
    def custom(self):
        d = BaseConfiguration()
        return d

    @property
    def default(self):
        d = BaseConfiguration()
        return d
