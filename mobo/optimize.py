from datetime import datetime
from mobo.configuration import GlobalConfiguration
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from typing import List, Optional


class Optimizer(object):
    """Optimization pipeline initialized from a configuration.

    Args:
        configuration: Configuration object to initialize from.
    """
    def __init__(self, configuration: GlobalConfiguration) -> None:
        self.configuration = configuration

    def __call__(self) -> None:
        self._log("Beginning the optimization process...")
        # initialize the dataframe
        path = self.configuration.initial_data_path
        if path is None:
            self._log("Generating initial parameter distributions...")
            index = range(self.configuration.n_samples)
            df = pd.DataFrame(columns=self.df_column_headers, index=index)
            init_dist = self._generate_initial_parameter_distributions()
            df[self.parameter_headers] = init_dist
        else:
            self._log("Reading initial parameter distributions from file...")
            df = pd.read_csv(path)
        # loop over each iteration
        last_df: Optional[pd.DataFrame] = None
        for i in range(len(self.configuration.local_configurations)):
            iteration_start = datetime.now()
            self._log("\nBeginning iteration {}...".format(i))
            # do not resample the first iteration
            if i == 0:
                self._log("Skipped resampling step for the first iteration.")
            else:
                df = self._sample(last_df, i) # returns a new df
            # evaluate the parameterizations
            df = self._evaluate(df, i)
            # filter out poor parameterization
            df = pd.concat([df, last_df]) # rejoin the old dataframe
            df = self._filter(df, i)
            # reset index for proper joining
            df.reset_index(inplace=True, drop=True)
            # project the filtered parameters onto a 2D space
            df = self._project(df, i)
            # cluster the projected parameter space
            df = self._cluster(df, i)
            # write the iteration data to file
            df = self._export(df, i)
            # store for sampling
            last_df = df
            iteration_end = datetime.now()
            time_delta = round(
                (iteration_end - iteration_start).total_seconds(), 3
            )
            self._log(
                "Completed iteration {} in {} seconds.\n".format(i, time_delta)
            )

    @property
    def parameter_headers(self) -> List[str]:
        return [p.name for p in self.configuration.parameters]

    @property
    def qoi_headers(self) -> List[str]:
        return [q.name for q in self.configuration.qois]

    @property
    def error_headers(self) -> List[str]:
        return ["{}_error".format(qn) for qn in self.qoi_headers]

    @property
    def projection_headers(self) -> List[str]:
        return ["projection_0", "projection_1"]

    @property
    def cluster_header(self) -> str:
        return "cluster_id"

    @property
    def df_column_headers(self) -> List[str]:
        return list(
            self.parameter_headers + self.qoi_headers + self.error_headers +
            self.projection_headers + [self.cluster_header]
        )

    def _sample(self, df: pd.DataFrame, iteration: int) -> pd.DataFrame:
        """Resample the filtered distribution."""
        self._log("Resampling parameter space via KDE...")
        cluster_ids = set(df[self.cluster_header].to_numpy())
        n_samples = self.configuration.local_configurations[iteration].n_samples
        n_samples_per_cluster = n_samples // len(cluster_ids)
        self._log(
            "\tDrawing {} samples per cluster...".format(n_samples_per_cluster)
        )
        samples = []
        for cluster_id in cluster_ids:
            self._log("\tCluster {}:".format(cluster_id))
            data = df[df[self.cluster_header] == cluster_id]
            self._log("\t\tsamples: {}".format(len(data)))
            data_arr = data[self.parameter_headers].to_numpy(float)
            # linalg error when num samples is less than num parameters
            try:
                kde = gaussian_kde(data_arr.T)
                bandwidth = kde.factor
            except (np.linalg.LinAlgError, ValueError, RuntimeWarning):
                msg = (
                    "\n\t[WARNING] The number of samples in cluster {} is too small. \n"
                    "\tSamples will not be drawn from this cluster.\n"
                ).format(cluster_id)
                self._log(msg)
                continue
            else:
                self._log("\t\tbandwidth: {:.6}".format(bandwidth))
                samples.append(kde.resample(n_samples_per_cluster).T)
        samples_arr = np.vstack(samples)
        index = range(len(samples_arr))
        new_df = pd.DataFrame(columns=self.df_column_headers, index=index)
        new_df[self.parameter_headers] = samples_arr
        return new_df

    def _evaluate(self, df: pd.DataFrame, iteration: int) -> pd.DataFrame:
        """Evaluate each qoi for each parameterization."""
        self._log("Evaluating parameterizations...")
        err_calc = self.configuration.local_configurations[iteration].error_calculator
        qoi_targets = np.array([
            qoi.target for qoi in self.configuration.qois
        ])
        all_qoi_values: List[np.ndarray] = []
        all_error_values: List[np.ndarray] = []
        for _, parameterization in df[self.parameter_headers].iterrows():
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
            {qh: qoi_arr[:, i] for i, qh in enumerate(self.qoi_headers)}
        )
        df.update(
            {eh: error_arr[:, i] for i, eh in enumerate(self.error_headers)}
        )
        return df

    def _filter(self, df: pd.DataFrame, iteration: int) -> pd.DataFrame:
        """Filter out poor parameterizations."""
        self._log("Filtering parameterizations...")
        self._log("\tSamples before filtration: {}".format(len(df)))
        filters = self.configuration.local_configurations[iteration].filters
        for f in filters:
            mask = f(df[self.error_headers].to_numpy())
            df = df[mask]
        self._log("\tSamples after filtration: {}".format(len(df)))
        return df

    def _project(self, df: pd.DataFrame, iteration: int) -> pd.DataFrame:
        """Project parameter space down to a 2D space."""
        self._log("Projecting parameter space down to 2D...")
        proj = self.configuration.local_configurations[iteration].projector
        proj_arr = proj(df[self.parameter_headers].to_numpy())
        df.update(
            {pn: proj_arr[:, i] for i, pn in enumerate(self.projection_headers)}
        )
        return df

    def _cluster(self, df: pd.DataFrame, iteration: int) -> pd.DataFrame:
        """Assign cluster ids to projected parameters."""
        self._log("Clustering projected parameter space...")
        clust = self.configuration.local_configurations[iteration].clusterer
        cluster_ids = clust(df[self.projection_headers].to_numpy())
        df.update(
            {self.cluster_header: cluster_ids}
        )
        return df

    def _export(self, df: pd.DataFrame, iteration: int) -> pd.DataFrame:
        """Export the results of an iteration as a csv file."""
        filename = "mobo_iteration_{}.csv".format(iteration)
        df.to_csv(filename)
        self._log("Exported iteration data to {}.".format(filename))
        return df

    def _generate_initial_parameter_distributions(self) -> np.ndarray:
        """Generate a uniform distribution over the parameter bounds."""
        lows = np.array([p.lower_bound for p in self.configuration.parameters])
        highs = np.array([p.upper_bound for p in self.configuration.parameters])
        size = (self.configuration.n_samples, lows.shape[0])
        return np.random.uniform(low=lows, high=highs, size=size)

    def _log(self, msg: str) -> None:
        """Log a message to file or stdout."""
        logger = self.configuration.logger
        if logger is None:
            print(msg)
        else:
            logger.log(msg)
