"""VIPR plugin for PEMFC operating-parameter inversion."""

from vipr_fuel_cell.load_data.dataset_loader import PEMFCDatasetLoader
from vipr_fuel_cell.load_model.cinn_loader import PEMFCCINNModelLoader
from vipr_fuel_cell.predict.posterior_predictor import PEMFCPosteriorPredictor
from vipr_fuel_cell.preprocess.condition_preprocessor import PEMFCConditionPreprocessor
from vipr_fuel_cell.postprocess.collector import PEMFCDataCollector

__all__ = [
    "PEMFCDatasetLoader",
    "PEMFCCINNModelLoader",
    "PEMFCPosteriorPredictor",
    "PEMFCConditionPreprocessor",
    "PEMFCDataCollector",
]


def load(app):
    """Register the fuel-cell inference components with VIPR."""
    app.log.info("Loading VIPR fuel-cell plugin")
    app.handler.register(PEMFCDatasetLoader)
    app.handler.register(PEMFCCINNModelLoader)
    app.handler.register(PEMFCPosteriorPredictor)

    # The preprocessing filter is discovered through its decorator when the
    # module is imported above. The collector is an explicit runtime hook.
    collector = PEMFCDataCollector(app)
    app.extend("pemfc_dc", collector)
    app.log.info("VIPR fuel-cell plugin loaded")
