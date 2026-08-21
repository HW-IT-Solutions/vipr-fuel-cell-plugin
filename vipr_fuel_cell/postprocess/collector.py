"""VIPR DataCollector integration for PEMFC posterior results."""

from __future__ import annotations

import numpy as np

from vipr_fuel_cell.postprocess.plot_scripts.posterior_snapshot import (
    make_plot as make_posterior_snapshot_plot,
)
from vipr_fuel_cell.predict.posterior_result import (
    PEMFCPosteriorResult,
    PosteriorSnapshot,
)


def _label_with_unit(label: str, unit: str) -> str:
    return f"{label} [{unit}]" if unit else label


def _format_axis_value(label: str, value: float, unit: str) -> str:
    suffix = f" {unit}" if unit else ""
    return f"{label} {float(value):g}{suffix}"


def _posterior_snapshot_plot_data(
    prediction: PEMFCPosteriorResult,
    snapshot: PosteriorSnapshot,
) -> dict[str, np.ndarray]:
    """Return portable arrays consumed by the standalone snapshot plotter."""
    names = prediction.parameter_names
    return {
        "bin_edges": np.asarray(
            [snapshot.histograms[name].bin_edges for name in names], dtype=float
        ),
        "densities": np.asarray(
            [snapshot.histograms[name].density for name in names], dtype=float
        ),
        "posterior_means": np.asarray(
            [
                prediction.statistics[name].mean[snapshot.valid_time_step_index]
                for name in names
            ],
            dtype=float,
        ),
        "parameter_labels": np.asarray(
            [prediction.parameters[name].label for name in names]
        ),
        "parameter_units": np.asarray(
            [prediction.parameters[name].unit for name in names]
        ),
        "valid_time_step_index": np.asarray([snapshot.valid_time_step_index]),
        "time_value": np.asarray([snapshot.time]),
        "time_label": np.asarray([prediction.time_label]),
        "time_unit": np.asarray([prediction.time_unit]),
    }


class PEMFCDataCollector:
    """Create one result item with trajectories and posterior snapshots."""

    def __init__(self, app):
        self.app = app
        app.hook.register(
            "INFERENCE_POSTPROCESS_PRE_PRE_FILTER_HOOK",
            self.collect_posterior_results,
        )

    def collect_posterior_results(self, app, data=None, result=None):
        prediction = result if result is not None else data
        if not isinstance(prediction, dict):
            return
        if prediction.get("prediction_type") != "pemfc_cinn_posterior":
            return

        try:
            self._collect(app, prediction)
        except Exception as exc:  # collector errors must not discard inference
            app.log.exception(f"PEMFC DataCollector failed: {exc}")

    def _collect(self, app, prediction: dict) -> None:
        prediction = PEMFCPosteriorResult.model_validate(prediction)
        collector = app.datacollector.create_item_collector(0)
        time = prediction.time
        statistics = prediction.statistics

        table = collector.table(
            "pemfc_parameter_summary",
            "Reconstructed PEMFC operating parameters",
            key_column="parameter",
        )
        for name in prediction.parameter_names:
            descriptor = prediction.parameters[name]
            label = descriptor.label
            parameter_id = descriptor.id
            parameter_statistics = statistics[name]
            values = np.asarray(parameter_statistics.mean, dtype=float)
            posterior_std = np.asarray(parameter_statistics.std, dtype=float)
            table.add_row(
                parameter=label,
                parameter_id=parameter_id,
                unit=descriptor.unit,
                mean=float(np.mean(values)),
                minimum=float(np.min(values)),
                maximum=float(np.max(values)),
                mean_posterior_std=float(np.mean(posterior_std)),
            )

            diagram = (
                collector.diagram(f"pemfc_{parameter_id}", label)
                .set_data("time", time)
                .set_data("posterior_mean", parameter_statistics.mean)
                .add_series(
                    "time", "posterior_mean", label="Posterior mean", kind="line"
                )
            )
            diagram.set_metadata(
                x_label=_label_with_unit(
                    prediction.time_label, prediction.time_unit
                ),
                y_label=_label_with_unit(label, descriptor.unit),
            )

        if prediction.snapshots:
            histogram_table = collector.table(
                "pemfc_posterior_snapshot_histograms",
                "Empirical posterior histogram data",
            )
            for snapshot in prediction.snapshots:
                plot_data = _posterior_snapshot_plot_data(prediction, snapshot)
                figure = make_posterior_snapshot_plot(**plot_data)
                image_id = (
                    "pemfc_posterior_distributions_"
                    f"index_{snapshot.valid_time_step_index}"
                )
                try:
                    (
                        collector.image(
                            image_id,
                            "PEMFC posterior distributions at "
                            + _format_axis_value(
                                prediction.time_label,
                                snapshot.time,
                                prediction.time_unit,
                            ),
                        )
                        .set_plot_script(
                            make_posterior_snapshot_plot,
                            data_format="npz",
                        )
                        .set_plot_data(**plot_data)
                        .set_from_matplotlib(figure, format="svg")
                        .set_metadata(
                            valid_time_step_index=snapshot.valid_time_step_index,
                            time=snapshot.time,
                            time_label=prediction.time_label,
                            time_unit=prediction.time_unit,
                            distribution="empirical_histogram",
                        )
                    )
                finally:
                    import matplotlib.pyplot as plt

                    plt.close(figure)

                for name in prediction.parameter_names:
                    descriptor = prediction.parameters[name]
                    histogram = snapshot.histograms[name]
                    mean = prediction.statistics[name].mean[
                        snapshot.valid_time_step_index
                    ]
                    for bin_index, density in enumerate(histogram.density):
                        histogram_table.add_row(
                            valid_time_step_index=snapshot.valid_time_step_index,
                            time=snapshot.time,
                            time_unit=prediction.time_unit,
                            parameter=descriptor.label,
                            parameter_id=descriptor.id,
                            unit=descriptor.unit,
                            bin_left=histogram.bin_edges[bin_index],
                            bin_right=histogram.bin_edges[bin_index + 1],
                            density=density,
                            count=histogram.counts[bin_index],
                            posterior_mean=mean,
                        )

        app.datacollector.data.batch_metadata.update(
            {
                "domain": "pemfc",
                "model_type": "conditional_invertible_neural_network",
                **prediction.metadata.model_dump(mode="python"),
            }
        )
        app.log.info(
            f"Stored PEMFC posterior table, {len(prediction.parameter_names)} "
            f"mean diagrams and {len(prediction.snapshots)} posterior snapshot image(s)"
        )
