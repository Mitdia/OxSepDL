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
    "TiO": {"Tb": 1223, "Tm": 1353},
    "Cr2O3": {"Tb": 1367, "Tm": 1497},
    "MnO": {"Tb": 1369, "Tm": 1499},
    "Ti2O3": {"Tb": 1376, "Tm": 1507},
    "Ti3O5": {"Tb": 1396, "Tm": 1526},
    "TiO2": {"Tb": 1399, "Tm": 1529},
    "MnSiO3": {"Tb": 1513, "Tm": 1643},
    "Mn2SiO4": {"Tb": 1530, "Tm": 1660},
    "SiO2": {"Tb": 1587, "Tm": 1718},
    "Al2TiO5": {"Tb": 1598, "Tm": 1728},
    "Al2SiO5": {"Tb": 1665, "Tm": 1795},
    "Al6Si2O13": {"Tb": 1681, "Tm": 1811},
    "MnAl2O4": {"Tb": 1700, "Tm": 1830},
    "Al2O3": {"Tb": 1701, "Tm": 1831},
    "MgAl2O4": {"Tb": 1783, "Tm": 1913},
    "MgSiO3": {"Tb": 1804, "Tm": 1934},
    "CaAl4O7": {"Tb": 1822, "Tm": 1952},
    "CaAl2O4": {"Tb": 1872, "Tm": 2002},
    "Mg2SiO4": {"Tb": 1923, "Tm": 2054},
    "Ca2SiO4": {"Tb": 2093, "Tm": 2223},
    "MgO": {"Tb": 2107, "Tm": 2237},
    "CaO": {"Tb": 2165, "Tm": 2295},
}


all_oxide_params_ShHaNew = {
    "Cr2O3": {"Tb": 1138, "Tm": 1268},
    "TiO": {"Tb": 1350, "Tm": 1480},
    "MnO": {"Tb": 1357, "Tm": 1487},
    "Ti2O3": {"Tb": 1489, "Tm": 1619},
    "TiO2": {"Tb": 1490, "Tm": 1620},
    "Ti3O5": {"Tb": 1502, "Tm": 1633},
    "MnSiO3": {"Tb": 1508, "Tm": 1639},
    "Mn2SiO4": {"Tb": 1523, "Tm": 1653},
    "SiO2": {"Tb": 1587, "Tm": 1718},
    "Al2TiO5": {"Tb": 1697, "Tm": 1827},
    "Al2SiO5": {"Tb": 1727, "Tm": 1857},
    "Al6Si2O13": {"Tb": 1753, "Tm": 1884},
    "MnAl2O4": {"Tb": 1772, "Tm": 1902},
    "Al2O3": {"Tb": 1799, "Tm": 1930},
    "MgSiO3": {"Tb": 1804, "Tm": 1934},
    "MgAl2O4": {"Tb": 1852, "Tm": 1982},
    "CaAl4O7": {"Tb": 1915, "Tm": 2045},
    "Mg2SiO4": {"Tb": 1923, "Tm": 2054},
    "CaAl2O4": {"Tb": 1956, "Tm": 2086},
    "Ca2SiO4": {"Tb": 2093, "Tm": 2223},
    "MgO": {"Tb": 2107, "Tm": 2237},
    "CaO": {"Tb": 2165, "Tm": 2295},
}


def read_data(directory_path=os.path.join("Data", "ShHaAligned"), files_names=None, ref_num=5):
    if files_names is None:
        files_names = np.random.choice(os.listdir(directory_path), size=ref_num, replace=False)
    references = []
    for path in files_names:
        data = pd.read_csv(os.path.join(directory_path, path))
        references.append((np.array(data["dO"], dtype="float64"), np.array(data["Temperature"], dtype="float64")))
    return references, files_names


def get_oxide_params(material_type, oxide_names):
    oxide_params = {}
    if material_type.lower() == "shha15":
        all_oxide_params = all_oxide_params_ShHa15
    elif material_type.lower() == "sg1":
        all_oxide_params = all_oxide_params_SG1
    elif material_type.lower() == "shhanew":
        all_oxide_params = all_oxide_params_ShHaNew
    else:
        raise AttributeError("Only SG1, ShHa15 and ShHaNew material_type is supported!")
    for oxide_name in oxide_names:
        if oxide_name in all_oxide_params:
            oxide_params[oxide_name] = all_oxide_params[oxide_name]
        else:
            raise AttributeError(f"{material_type} does not have any {oxide_name}")
    return oxide_params
