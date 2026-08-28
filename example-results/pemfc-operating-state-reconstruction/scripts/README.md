# VIPR Plot Scripts

Standalone Python scripts to reproduce and customise VIPR result plots.
No VIPR installation required.

## Usage

```bash
pip install -r requirements.txt
```

### PEMFC posterior distributions at Simulation step 41

```bash
python plot_pemfc_posterior_distributions_index_40.py
# or save to file:
python plot_pemfc_posterior_distributions_index_40.py -o my_plot.svg
```

Data file: `pemfc_posterior_distributions_index_40_data.npz`

## Diagram scripts (auto-generated)

### Cell current density

```bash
python plot_pemfc_cell_current_density.py
# or save to file:
python plot_pemfc_cell_current_density.py -o my_plot.svg
```

Data file: `../diagrams/pemfc_cell_current_density_Posterior_mean.csv (+ more per series)`

### Anode inlet relative humidity

```bash
python plot_pemfc_anode_inlet_relative_humidity.py
# or save to file:
python plot_pemfc_anode_inlet_relative_humidity.py -o my_plot.svg
```

Data file: `../diagrams/pemfc_anode_inlet_relative_humidity_Posterior_mean.csv (+ more per series)`

### Cathode inlet relative humidity

```bash
python plot_pemfc_cathode_inlet_relative_humidity.py
# or save to file:
python plot_pemfc_cathode_inlet_relative_humidity.py -o my_plot.svg
```

Data file: `../diagrams/pemfc_cathode_inlet_relative_humidity_Posterior_mean.csv (+ more per series)`

### Anode inlet temperature

```bash
python plot_pemfc_anode_inlet_temperature.py
# or save to file:
python plot_pemfc_anode_inlet_temperature.py -o my_plot.svg
```

Data file: `../diagrams/pemfc_anode_inlet_temperature_Posterior_mean.csv (+ more per series)`

### Cathode inlet temperature

```bash
python plot_pemfc_cathode_inlet_temperature.py
# or save to file:
python plot_pemfc_cathode_inlet_temperature.py -o my_plot.svg
```

Data file: `../diagrams/pemfc_cathode_inlet_temperature_Posterior_mean.csv (+ more per series)`

### Anode inlet pressure

```bash
python plot_pemfc_anode_inlet_pressure.py
# or save to file:
python plot_pemfc_anode_inlet_pressure.py -o my_plot.svg
```

Data file: `../diagrams/pemfc_anode_inlet_pressure_Posterior_mean.csv (+ more per series)`

### Cathode inlet pressure

```bash
python plot_pemfc_cathode_inlet_pressure.py
# or save to file:
python plot_pemfc_cathode_inlet_pressure.py -o my_plot.svg
```

Data file: `../diagrams/pemfc_cathode_inlet_pressure_Posterior_mean.csv (+ more per series)`

### Hydrogen stoichiometry

```bash
python plot_pemfc_hydrogen_stoichiometry.py
# or save to file:
python plot_pemfc_hydrogen_stoichiometry.py -o my_plot.svg
```

Data file: `../diagrams/pemfc_hydrogen_stoichiometry_Posterior_mean.csv (+ more per series)`

### Oxygen stoichiometry

```bash
python plot_pemfc_oxygen_stoichiometry.py
# or save to file:
python plot_pemfc_oxygen_stoichiometry.py -o my_plot.svg
```

Data file: `../diagrams/pemfc_oxygen_stoichiometry_Posterior_mean.csv (+ more per series)`
