from mobo.configuration import GlobalConfiguration
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from typing import List


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
            df = pd.DataFrame(columns=self.df_column_names)
            init_dist = self._generate_initial_parameter_distributions()
            df[self.parameter_names] = init_dist
        else:
            df = pd.read_csv(path)
        # loop over each iteration
        for i in range(self.configuration.n_iterations):
            self._evaluate(df, i) # generate errors
            self._filter(df, i)   # remove poor parameterizations
            self._project(df, i)  # project parameters onto 2D
            self._cluster(df, i)  # cluster projected parameters
            self._export(df, i)   # write iteration data to file
            self._sample(df, i)   # generate new parameter distribution

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

    def _evaluate(self, df: pd.DataFrame, iteration: int) -> None:
        """Evaluate each qoi for each parameterization."""
        err_calc = self.configuration.local_configurations[iteration].error_calculator
        errors = []
        for p in df[self.parameter_names]:
            qoi_values = np.array([
                q.evaluator(p.to_dict()) for q in self.configuration.qois
            ])
            qoi_targets = np.array([
                q.target for q in self.configuration.qois
            ])
            errors.append(err_calc(actual=qoi_values, target=qoi_targets))
        df[self.error_names] = errors

    def _filter(self, df: pd.DataFrame, iteration: int) -> None:
        """Filter out poor parameterizations."""
        filters = self.configuration.local_configurations[iteration].filters
        for f in filters:
            mask = f(df[self.error_names])
            df = df[mask]

    def _project(self, df: pd.DataFrame, iteration: int) -> None:
        """Project parameter space down to a 2D space."""
        proj = self.configuration.local_configurations[iteration].projector
        projection = proj(df[self.parameter_names])
        df[self.projection_names] = projection

    def _cluster(self, df: pd.DataFrame, iteration: int) -> None:
        """Assign cluster ids to projected parameters."""
        clust = self.configuration.local_configurations[iteration].clusterer
        cluster_ids = clust(df[self.projection_names])
        df[self.cluster_names] = cluster_ids

    def _export(self, df: pd.DataFrame, iteration: int) -> None:
        """Export the results of an iteration as a csv file."""
        filename = "mobo_iteration_{}.csv".format(iteration)
        df.to_csv(filename)

    def _sample(self, df: pd.DataFrame, iteration: int) -> None:
        """Resample the filtered distribution."""
        cluster_ids = set(df[self.cluster_names])
        n_samples = self.configuration.local_configurations[iteration].n_samples
        n_samples_per_cluster = n_samples // len(cluster_ids) # TODO
        samples = []
        for cluster_id in cluster_ids:
            data = df[self.parameter_names].where(df[self.cluster_names] == cluster_id)
            kde = gaussian_kde(data)
            samples.append(kde.resample(n_samples_per_cluster).T)
        samples = np.vstack(samples) # concat new samples
        old_samples = df[self.parameter_names].to_numpy()
        samples = np.vstack((samples, old_samples)) # concat old samples
        df = pd.DataFrame(columns=self.df_column_names) # overwrite df
        df[self.parameter_names] = samples

    def _generate_initial_parameter_distributions(self) -> np.ndarray:
        """Generate a uniform distribution over the parameter bounds."""
        lows = np.array([p.lower_bound for p in self.configuration.parameters])
        highs = np.array([p.upper_bound for p in self.configuration.parameters])
        return np.random.uniform(low=lows, high=highs, 
                                 size=self.configuration.n_samples)
