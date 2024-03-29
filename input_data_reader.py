import pandas as pd
import numpy as np
import os


all_oxide_params_SG1 = {
    "MnO": {"Tb": 1399, "Tm": 1529},
    "TiO2": {"Tb": 1395, "Tm": 1525},
    "SiO2": {"Tb": 1576, "Tm": 1706},
    "Al2O3": {"Tb": 1848, "Tm": 1978},
}

all_oxide_params_ShHa15 = {
    "TiO": {"Tb": 1329, "Tm": 1459},
    "Cr2O3": {"Tb": 1367, "Tm": 1497},
    "MnO": {"Tb": 1369, "Tm": 1499},
    "Ti2O3": {"Tb": 1470, "Tm": 1600},
    "TiO2": {"Tb": 1475, "Tm": 1605},
    "Ti3O5": {"Tb": 1485, "Tm": 1615},
    "MnSiO3": {"Tb": 1513, "Tm": 1643},
    "Mn2Si3O4": {"Tb": 1530, "Tm": 1660},
    "SiO2": {"Tb": 1587, "Tm": 1718},
    "Al2TiO5": {"Tb": 1633, "Tm": 1764},
    "Al2SiO5": {"Tb": 1665, "Tm": 1795},
    "Al5Si2O13": {"Tb": 1681, "Tm": 1811},
    "MnAl2O4": {"Tb": 1700, "Tm": 1830},
    "Al2O3": {"Tb": 1701, "Tm": 1831},
}


def read_data(directory_path=os.path.join("Data", "ShHa"), ref_num=5):
    data_path_array = np.random.choice(os.listdir(directory_path), size=ref_num, replace=False)
    references = []
    for path in data_path_array:
        data = pd.read_csv(os.path.join(directory_path, path))
        references.append((np.array(data["dO"], dtype="float64"), np.array(data["Temperature"], dtype="float64")))
    return references, data_path_array


def get_oxide_params(material_type, oxide_names):
    oxide_params = {}
    if material_type.lower() == "shha15":
        all_oxide_params = all_oxide_params_ShHa15
    elif material_type.lower() == "sg1":
        all_oxide_params = all_oxide_params_SG1
    else:
        raise AttributeError("Only SG1 and ShHa15 material_type is supported!")
    for oxide_name in oxide_names:
        if oxide_name in all_oxide_params:
            oxide_params[oxide_name] = all_oxide_params[oxide_name]
        else:
            raise AttributeError(f"{material_type} does not have any {oxide_name}")
    return oxide_params
