from mobo.configuration import GlobalConfiguration
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from typing import Dict, List


class Optimizer(object):
    """Optimization pipeline initialized from a configuration.
    
    Args:
        configuration: Configuration object to initialize from.
    """
    def __init__(self, configuration: GlobalConfiguration) -> None:
        self.configuration = configuration

    def __call__(self) -> None:
        # initialize the dataframe
        path = self.configuration.initial_data_path
        if path is None:
            index = range(self.configuration.n_samples)
            df = pd.DataFrame(columns=self.df_column_names, index=index)
            init_dist = self._generate_initial_parameter_distributions()
            df[self.parameter_names] = init_dist
        else:
            df = pd.read_csv(path)
        # loop over each iteration
        for i in range(self.configuration.n_iterations):
            df = self._evaluate(df, i) # generate errors
            df = self._filter(df, i)   # remove poor parameterizations
            df = df.reset_index()
            df = self._project(df, i)  # project parameters onto 2D
            df = self._cluster(df, i)  # cluster projected parameters
            df = self._export(df, i)   # write iteration data to file
            df = self._sample(df, i)   # generate new parameter distribution

    @property
    def parameter_names(self) -> List[str]:
        return [p.name for p in self.configuration.parameters]

    @property
    def qoi_names(self) -> List[str]:
        return [q.name for q in self.configuration.qois]

    @property
    def error_names(self) -> List[str]:
        return ["{}_error".format(qn) for qn in self.qoi_names]

    @property
    def projection_names(self) -> List[str]:
        return ["projection_0", "projection_1"]

    @property
    def cluster_names(self) -> List[str]:
        return ["cluster_id"]

    @property
    def df_column_names(self) -> List[str]:
        return list(
            self.parameter_names + self.qoi_names + self.error_names +
            self.projection_names + self.cluster_names
        )

    def _evaluate(self, df: pd.DataFrame, iteration: int) -> pd.DataFrame:
        """Evaluate each qoi for each parameterization."""
        err_calc = self.configuration.local_configurations[iteration].error_calculator
        qoi_targets = np.array([
            qoi.target for qoi in self.configuration.qois
        ])
        all_qoi_values: List[np.ndarray] = []
        all_error_values: List[np.ndarray] = []
        for _, parameterization in df[self.parameter_names].iterrows():
            qoi_values = np.array([
                qoi.evaluator(parameterization.to_dict()) 
                for qoi in self.configuration.qois
            ])
            all_qoi_values.append(qoi_values)
            error_values = err_calc(actual=qoi_values, target=qoi_targets)
            all_error_values.append(error_values)
        qoi_arr = np.array(all_qoi_values)
        error_arr = np.array(all_error_values)
        df.update(
            {qn: qoi_arr[:, i] for i, qn in enumerate(self.qoi_names)}
        )
        df.update(
            {en: error_arr[:, i] for i, en in enumerate(self.error_names)}
        )
        return df

    def _filter(self, df: pd.DataFrame, iteration: int) -> pd.DataFrame:
        """Filter out poor parameterizations."""
        filters = self.configuration.local_configurations[iteration].filters
        for f in filters:
            mask = f(df[self.error_names].to_numpy())
            df = df[mask]
        return df

    def _project(self, df: pd.DataFrame, iteration: int) -> pd.DataFrame:
        """Project parameter space down to a 2D space."""
        proj = self.configuration.local_configurations[iteration].projector
        proj_arr = proj(df[self.parameter_names].to_numpy())
        df.update(
            {pn: proj_arr[:, i] for i, pn in enumerate(self.projection_names)}
        )
        return df

    def _cluster(self, df: pd.DataFrame, iteration: int) -> pd.DataFrame:
        """Assign cluster ids to projected parameters."""
        clust = self.configuration.local_configurations[iteration].clusterer
        cluster_ids = clust(df[self.projection_names].to_numpy())
        # this is an exception because cluster_names is really just on name
        df.update(
            {self.cluster_names[0]: cluster_ids}
        )
        return df

    def _export(self, df: pd.DataFrame, iteration: int) -> pd.DataFrame:
        """Export the results of an iteration as a csv file."""
        filename = "mobo_iteration_{}.csv".format(iteration)
        df.to_csv(filename)
        return df

    def _sample(self, df: pd.DataFrame, iteration: int) -> pd.DataFrame:
        """Resample the filtered distribution."""
        cluster_ids = set(df[self.cluster_names].to_numpy().flatten())
        n_samples = self.configuration.local_configurations[iteration].n_samples
        n_samples_per_cluster = n_samples // len(cluster_ids) # TODO
        samples = []
        for cluster_id in cluster_ids:
            data = df[df[self.cluster_names[0]] == cluster_id] # another issue caused by single element list
            data_arr = data[self.parameter_names].to_numpy(float)
            # i will never understand the double transpose
            kde = gaussian_kde(data_arr.T)
            samples.append(kde.resample(n_samples_per_cluster).T)
        new_samples_arr = np.vstack(samples) # concat new samples
        old_samples_arr = df[self.parameter_names].to_numpy()
        samples_arr = np.vstack((old_samples_arr, new_samples_arr)) # concat old samples
        index = range(len(samples_arr))
        df = pd.DataFrame(columns=self.df_column_names, index=index) # overwrite df
        df.update(
            {pn: samples_arr[:, i] for i, pn in enumerate(self.parameter_names)}
        )
        return df

    def _generate_initial_parameter_distributions(self) -> np.ndarray:
        """Generate a uniform distribution over the parameter bounds."""
        lows = np.array([p.lower_bound for p in self.configuration.parameters])
        highs = np.array([p.upper_bound for p in self.configuration.parameters])
        size = (self.configuration.n_samples, lows.shape[0])
        return np.random.uniform(low=lows, high=highs, size=size)
