"""Names, English labels and units used by the published PEMFC cINN."""

PARAMETER_NAMES = [
    "J_Cell_A_cm2",
    "RH_In_An_Ist",
    "RH_In_Cath_Ist",
    "T_In_An_Ist",
    "T_In_Cath_Ist",
    "p_In_An_Ist",
    "p_In_Cath_Ist",
    "Stoech_In_H2_An_Ist",
    "Stoech_In_O2_Cath_Ist",
]

PARAMETER_UNITS = {
    "J_Cell_A_cm2": "A/cm²",
    "RH_In_An_Ist": "1",
    "RH_In_Cath_Ist": "1",
    "T_In_An_Ist": "K",
    "T_In_Cath_Ist": "K",
    "p_In_An_Ist": "atm",
    "p_In_Cath_Ist": "atm",
    "Stoech_In_H2_An_Ist": "1",
    "Stoech_In_O2_Cath_Ist": "1",
}

PARAMETER_LABELS = {
    "J_Cell_A_cm2": "Cell current density",
    "RH_In_An_Ist": "Anode inlet relative humidity",
    "RH_In_Cath_Ist": "Cathode inlet relative humidity",
    "T_In_An_Ist": "Anode inlet temperature",
    "T_In_Cath_Ist": "Cathode inlet temperature",
    "p_In_An_Ist": "Anode inlet pressure",
    "p_In_Cath_Ist": "Cathode inlet pressure",
    "Stoech_In_H2_An_Ist": "Hydrogen stoichiometry",
    "Stoech_In_O2_Cath_Ist": "Oxygen stoichiometry",
}

PARAMETER_IDS = {
    "J_Cell_A_cm2": "cell_current_density",
    "RH_In_An_Ist": "anode_inlet_relative_humidity",
    "RH_In_Cath_Ist": "cathode_inlet_relative_humidity",
    "T_In_An_Ist": "anode_inlet_temperature",
    "T_In_Cath_Ist": "cathode_inlet_temperature",
    "p_In_An_Ist": "anode_inlet_pressure",
    "p_In_Cath_Ist": "cathode_inlet_pressure",
    "Stoech_In_H2_An_Ist": "hydrogen_stoichiometry",
    "Stoech_In_O2_Cath_Ist": "oxygen_stoichiometry",
}
