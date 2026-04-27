from __future__ import annotations

ELEMENTS = [
    "Ni",
    "Co",
    "Cr",
    "Al",
    "Ti",
    "Ta",
    "W",
    "Mo",
    "Re",
    "Nb",
    "Hf",
    "C",
    "B",
]

PROCESS_FEATURES = [
    "solution_temp_c",
    "solution_time_h",
    "aging_temp_c",
    "aging_time_h",
    "cooling_rate_c_s",
]

TEST_FEATURES = [
    "test_temp_c",
    "stress_mpa",
]

REGRESSION_TARGETS = [
    "yield_strength_mpa",
    "tensile_strength_mpa",
    "creep_log_life_h",
    "oxidation_mass_gain_mg_cm2",
]

CLASSIFICATION_TARGETS = [
    "creep_life_class",
]

TARGET_COLUMNS = REGRESSION_TARGETS + CLASSIFICATION_TARGETS

ATOMIC_WEIGHT = {
    "Ni": 58.69,
    "Co": 58.93,
    "Cr": 52.00,
    "Al": 26.98,
    "Ti": 47.87,
    "Ta": 180.95,
    "W": 183.84,
    "Mo": 95.95,
    "Re": 186.21,
    "Nb": 92.91,
    "Hf": 178.49,
    "C": 12.01,
    "B": 10.81,
}

ATOMIC_RADIUS_PM = {
    "Ni": 124,
    "Co": 125,
    "Cr": 128,
    "Al": 143,
    "Ti": 147,
    "Ta": 146,
    "W": 139,
    "Mo": 139,
    "Re": 137,
    "Nb": 146,
    "Hf": 159,
    "C": 70,
    "B": 85,
}

ELECTRONEGATIVITY = {
    "Ni": 1.91,
    "Co": 1.88,
    "Cr": 1.66,
    "Al": 1.61,
    "Ti": 1.54,
    "Ta": 1.50,
    "W": 2.36,
    "Mo": 2.16,
    "Re": 1.90,
    "Nb": 1.60,
    "Hf": 1.30,
    "C": 2.55,
    "B": 2.04,
}

DENSITY_G_CM3 = {
    "Ni": 8.90,
    "Co": 8.90,
    "Cr": 7.19,
    "Al": 2.70,
    "Ti": 4.51,
    "Ta": 16.65,
    "W": 19.25,
    "Mo": 10.28,
    "Re": 21.02,
    "Nb": 8.57,
    "Hf": 13.31,
    "C": 2.26,
    "B": 2.34,
}

APPROX_COST_INDEX = {
    "Ni": 1.0,
    "Co": 3.5,
    "Cr": 0.8,
    "Al": 0.4,
    "Ti": 1.5,
    "Ta": 8.0,
    "W": 6.0,
    "Mo": 4.0,
    "Re": 30.0,
    "Nb": 4.5,
    "Hf": 10.0,
    "C": 0.2,
    "B": 0.3,
}

GAMMA_PRIME_FORMERS = ["Al", "Ti", "Ta", "Nb"]
REFRACTORY_ELEMENTS = ["W", "Mo", "Ta", "Re", "Nb"]
OXIDATION_RESISTANCE_ELEMENTS = ["Cr", "Al"]