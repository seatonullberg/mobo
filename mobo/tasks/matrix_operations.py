from mobo.engines import Task

import numpy as np
import pandas as pd


class GroupByColumnValue(Task):

    def __init__(self, kwargs, parallel=False, target=None):
        if target is None:
            target = self.subselect
        super().__init__(parallel=parallel,
                         target=target,
                         kwargs=kwargs)

    def subselect(self, data, col_id, drop_selector):
        """
        :param data: an array-like object
        :param col_id: int if array string if DataFrame
                       - indicates which column to group by
        :param drop_selector: bool indicating whether or not to include the
                              grouping column in the result
        """
        if isinstance(data, pd.DataFrame):
            assert type(col_id) == str
            col_values = set(data[col_id])
            subselections = []
            for v in col_values:
                sub = data.loc[data[col_id] == v]
                if drop_selector:
                    sub = sub.drop(col_id, 1)
                subselections.append(sub)
        elif isinstance(data, np.ndarray):
            assert type(col_id) == int
            col_values = set(data[:, col_id])
            print(col_values)
            subselections = []
            for v in col_values:
                sub = data[np.where(data[:, col_id] == v)]
                if drop_selector:
                    sub = np.delete(sub, col_id, 1)
                subselections.append(sub)
        else:
            raise TypeError("data must be numpy.ndarray or pandas.DataFrame")
            # make it a custom error

        self.set_persistent(key='grouped_subselections',
                            value=subselections)


class ConcatenatePairTask(Task):

    def __init__(self, kwargs, parallel=False, target=None):
        if target is None:
            target = self.concatenate
        super().__init__(parallel=parallel,
                         target=target,
                         kwargs=kwargs)

    def concatenate(self, a1, a2, axis):
        """
        :param a1: array-like object
        :param a2: array-like object
        :param axis: axis along which to join arrays
        * type(a1) == type(a2)
        """
        if type(a1) == type(a2):
            if type(a1) == np.ndarray:
                result = self._process_numpy(a1, a2, axis)
            elif type(a1) == pd.DataFrame:
                result = self._process_pandas(a1, a2, axis)
            else:
                raise TypeError("a1 and a2 must be np.ndarray or pd.DataFrame")
        else:
            raise TypeError("a1 and a2 must have same type")
            # make custom error

        self.set_persistent(key='concatenated_data',
                            value=result)

    def _process_numpy(self, a1, a2, axis):
        return np.concatenate((a1, a2), axis)

    def _process_pandas(self, a1, a2, axis):
        return pd.concat([a1, a2], axis)


class ConcatenateListTask(Task):

    def __init__(self, kwargs, parallel=False, target=None):
        if target is None:
            target = self.concatenate
        super().__init__(parallel=parallel,
                         target=target,
                         kwargs=kwargs)

    def concatenate(self, data_list, axis):
        if all(isinstance(d, pd.DataFrame) for d in data_list):
            result = self._process_pandas(data_list, axis)
        elif all(isinstance(d, np.ndarray) for d in data_list):
            result = self._process_numpy(data_list, axis)
        else:
            raise TypeError("data_list members must all be same array-like type")

        self.set_persistent(key='concatenated_data',
                            value=result)

    def _process_numpy(self, data_list, axis):
        return np.concatenate(data_list, axis)

    def _process_pandas(self, data_list, axis):
        return pd.concat(data_list, axis)
