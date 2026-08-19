"""Names and units used by the published PEMFC cINN models."""

FULL_CONDITION_NAMES = [
    "U_cell_V",
    "E_Con",
    "E_Ohm",
    "E_Cross",
    "E_act",
    "Imax_An_fit",
    "Imax_Kath_fit",
    "N_flux_H2O_th_An",
    "N_flux_H2O_th_Kath",
    "N_flux_H2_th_An",
    "N_flux_O2_th_Kath",
]

REDUCED_CONDITION_NAMES = [
    "U_cell_V",
    "E_Con",
    "E_Ohm",
    "E_act",
]

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

REFERENCE_NAME_MAPPING = {
    "Current.DensitySimulation": "J_Cell_A_cm2",
    "RelativeHumidity.InletAnodeINI": "RH_In_An_Ist",
    "RelativeHumidity.InletCathodeINI": "RH_In_Cath_Ist",
    "Temperature.InletAnodeINI": "T_In_An_Ist",
    "Temperature.InletCathodeINI": "T_In_Cath_Ist",
    "Pressure.InletAnodeINI": "p_In_An_Ist",
    "Pressure.InletCathodeINI": "p_In_Cath_Ist",
    "Stoechiometry.Hydrogen": "Stoech_In_H2_An_Ist",
    "Stoechiometry.Oxygen": "Stoech_In_O2_Cath_Ist",
}
