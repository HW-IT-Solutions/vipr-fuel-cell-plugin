"""VIPR DataCollector integration for PEMFC posterior results."""

from __future__ import annotations

import numpy as np

from vipr_fuel_cell.contracts import PEMFCPosteriorResult, quantile_key


class PEMFCDataCollector:
    """Create one result item containing parameter summaries and trajectories."""

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
        references = prediction.reference_values
        configured_quantiles = prediction.metadata.quantiles
        lower_key = (
            quantile_key(min(configured_quantiles)) if configured_quantiles else None
        )
        upper_key = (
            quantile_key(max(configured_quantiles)) if configured_quantiles else None
        )

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
                reference=references.get(name),
            )

            diagram = (
                collector.diagram(f"pemfc_{parameter_id}", label)
                .set_data("time", time)
                .set_data("posterior_mean", parameter_statistics.mean)
                .add_series(
                    "time", "posterior_mean", label="Posterior mean", kind="line"
                )
            )
            if lower_key is not None:
                diagram = diagram.set_data(
                    "posterior_lower", parameter_statistics.quantiles[lower_key]
                ).add_series(
                    "time", "posterior_lower", label=f"q={lower_key}", kind="line"
                )
            if upper_key is not None and upper_key != lower_key:
                diagram = (
                    diagram.set_data("posterior_upper", parameter_statistics.quantiles[upper_key])
                    .add_series(
                        "time", "posterior_upper", label=f"q={upper_key}", kind="line"
                    )
                )
            if name in references:
                diagram = diagram.set_data(
                    "reference", [references[name]] * len(time)
                ).add_series("time", "reference", label="Reference", kind="line")
            diagram.set_metadata(
                x_label=f"{prediction.time_label} [{prediction.time_unit}]",
                y_label=f"{label} [{descriptor.unit}]",
                lower_quantile=lower_key,
                upper_quantile=upper_key,
            )

        app.datacollector.data.batch_metadata.update(
            {
                "domain": "pemfc",
                "model_type": "conditional_invertible_neural_network",
                **prediction.metadata.model_dump(mode="python"),
            }
        )
        app.log.info(
            f"Stored PEMFC posterior table and {len(prediction.parameter_names)} diagrams"
        )
