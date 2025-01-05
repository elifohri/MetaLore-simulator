from typing import Dict

import pandas as pd


class Monitor:
    def __init__(
        self, scalar_metrics: Dict, kpi_metrics: Dict, ue_metrics: Dict, bs_metrics: Dict, ss_metrics: Dict, **kwargs
    ):

        self.scalar_metrics: Dict = scalar_metrics
        self.kpi_metrics: Dict = kpi_metrics
        self.ue_metrics: Dict = ue_metrics
        self.bs_metrics: Dict = bs_metrics
        self.ss_metrics: Dict = ss_metrics

        self.scalar_results: Dict = None
        self.kpi_results: Dict = None
        self.ue_results: Dict = None
        self.bs_results: Dict = None
        self.ss_results: Dict = None

    def reset(self):
        """Reset tracked results for all metrics."""

        self.scalar_results = {name: [] for name in self.scalar_metrics}
        self.kpi_results = {name: [] for name in self.kpi_metrics}
        self.ue_results = {name: [] for name in self.ue_metrics}
        self.bs_results = {name: [] for name in self.bs_metrics}
        self.ss_results = {name: [] for name in self.ss_metrics}

    def update(self, simulation):
        """Evaluate and update metrics given the simulation state."""

        # evaluate scalar, ue, bs, ss metrics by passing the simulation state
        scalar_updates = {
            name: metric(simulation) for name, metric in self.scalar_metrics.items()
        }
        kpi_updates = {
            name: metric(simulation) for name, metric in self.kpi_metrics.items()
        }
        ue_updates = {
            name: metric(simulation) for name, metric in self.ue_metrics.items()
        }
        bs_updates = {
            name: metric(simulation) for name, metric in self.bs_metrics.items()
        }
        ss_updates = {
            name: metric(simulation) for name , metric in self.ss_metrics.items()
        }

        # update results by appending the metrics' return values
        self.scalar_results = {
            name: self.scalar_results[name] + [scalar_updates[name]]
            for name in self.scalar_metrics
        }
        self.kpi_results = {
            name: self.kpi_results[name] + [kpi_updates[name]]
            for name in self.kpi_metrics
        }
        self.ue_results = {
            name: self.ue_results[name] + [ue_updates[name]] for name in self.ue_metrics
        }
        self.bs_results = {
            name: self.bs_results[name] + [bs_updates[name]] for name in self.bs_metrics
        }
        self.ss_results = {
            name: self.ss_results[name] + [ss_updates[name]] for name in self.ss_metrics
        }

    def load_results(self):
        """Outputs results of tracked metrics as data frames."""

        # Load scalar results with index (metric; time)
        scalar_results = pd.DataFrame(self.scalar_results)
        scalar_results.index.names = ["Time Step"]

        # Load kpi results with index (metric; time)
        kpi_results = pd.DataFrame(self.kpi_results)
        kpi_results.index.names = ["Time Step"]

        # Load UE results with index (metric, UE ID; time)
        ue_results = {
            (metric, ue_id): [values.get(ue_id) for values in entries]
            for metric, entries in self.ue_results.items()
            for ue_id in set().union(*entries)
        }
        ue_results = pd.DataFrame(ue_results).transpose()

        # Convert index to MultiIndex
        ue_results.index = pd.MultiIndex.from_tuples(ue_results.index, names=["Metric", "UE ID"])

        # Align time axis along rows
        ue_results = ue_results.stack()
        ue_results.index.names = ["Metric", "UE ID", "Time Step"]
        ue_results = ue_results.reorder_levels(["Time Step", "UE ID", "Metric"])
        ue_results = ue_results.unstack()

        # Load BS results with index (metric, BS ID; time)
        bs_results = {
            (metric, bs_id): [values.get(bs_id) for values in entries]
            for metric, entries in self.bs_results.items()
            for bs_id in set().union(*entries)
        }
        bs_results = pd.DataFrame(bs_results).transpose()

        # Convert index to MultiIndex
        bs_results.index = pd.MultiIndex.from_tuples(bs_results.index, names=["Metric", "BS ID"])

        # Align time axis along rows
        bs_results = bs_results.stack()
        bs_results.index.names = ["Metric", "BS ID", "Time Step"]
        bs_results = bs_results.reorder_levels(["Time Step", "BS ID", "Metric"])
        bs_results = bs_results.unstack()

        # Load Sensor (SS) results with index (metric, Sensor ID; time)
        ss_results = {
            (metric, sensor_id): [values.get(sensor_id) for values in entries]
            for metric, entries in self.ss_results.items()
            for sensor_id in set().union(*entries)
        }
        ss_results = pd.DataFrame(ss_results).transpose()

        # Convert index to MultiIndex
        ss_results.index = pd.MultiIndex.from_tuples(ss_results.index, names=["Metric", "Sensor ID"])

        # Align time axis along rows
        ss_results = ss_results.stack()
        ss_results.index.names = ["Metric", "Sensor ID", "Time Step"]
        ss_results = ss_results.reorder_levels(["Time Step", "Sensor ID", "Metric"])
        ss_results = ss_results.unstack()

        return scalar_results, kpi_results, ue_results, bs_results, ss_results

    def info(self):
        """Outputs the latest results as a dictionary."""

        # Return empty infos if there are no scalar results.
        if any(len(results) == 0 for results in self.scalar_results.values()):
            return {}
        
        # Return empty infos if there are no important results.
        if any(len(results) == 0 for results in self.kpi_results.values()):
            return {}

        scalar_info = {name: values[-1] for name, values in self.scalar_results.items()}
        kpi_info = {name: values[-1] for name, values in self.kpi_results.items()}
        ue_info = {name: values[-1] for name, values in self.ue_results.items()}
        bs_info = {name: values[-1] for name, values in self.bs_results.items()}
        ss_info = {name: values[-1] for name, values in self.ss_results.items()}

        return {**scalar_info, **kpi_info, **ue_info, **bs_info, **ss_info}