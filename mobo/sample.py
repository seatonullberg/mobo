

class BaseSampler(object):
    """Representation of a general distribution sampler."""

    def __init__(self):
        pass

    def draw(self):
        err = ("{} does not implement the required `draw` method."
               .format(self.__class__.__name__))
        raise NotImplementedError(err)

