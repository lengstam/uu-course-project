# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:32:52 2023

@author: Linus Engstam
"""

import pandas as pd
import numpy as np

def data_saving(
        year: float = 2021,
) -> pd.DataFrame:
    """ Returns a dataframe with a column for each hourly variable that is to be saved"""
    if year == 2020:
        return pd.DataFrame({'Biogas (CH4) [mol/h]': np.zeros(8784), 'Biogas (CO2) [mol/h]': np.zeros(8784), 'H2 demand [mol/h]': np.zeros(8784), 'AEF emissions [gCO$_2$/kWh]': np.zeros(8784), 'AEF emissions [gCO$_2$/kWh]': np.zeros(8784), 'Elspot [€/MWh]': np.zeros(8784), 'Elz dispatch [kWh/h]': np.zeros(8784), \
                             'System dispatch [kWh/h]': np.zeros(8784), 'Meth el [kWh/h]': np.zeros(8784), 'Biogas comp [kWh/h]': np.zeros(8784), 'Standby': np.zeros(8784), 'Elz cold start': np.zeros(8784), 'Grid use [kWh/h]': np.zeros(8784), 'Wind use [kWh/h]': np.zeros(8784), 'Wind gen [kWh/h]': np.zeros(8784), \
                                 'PV use [kWh/h]': np.zeros(8784), 'PV gen [kWh/h]': np.zeros(8784), 'Battery state [%]': np.zeros(8784), 'Stack efficiency [%]': np.zeros(8784), 'System efficiency [%]': np.zeros(8784), \
                                     'H2 production [kg/h]': np.zeros(8784), 'H2 to storage [mol/h]': np.zeros(8784), 'H2 from storage [mol/h]': np.zeros(8784), 'H2 storage [%]': np.zeros(8784), 'H2 to meth [mol/h]': np.zeros(8784), 'Meth CH4 in [mol/h]': np.zeros(8784), 'Meth CO2 in [mol/h]': np.zeros(8784), \
                                         'Elz heat [kWh/h]': np.zeros(8784), 'Meth heat [kWh/h]': np.zeros(8784), 'WWTP heat demand [kWh/h]': np.zeros(8784), 'O2 out [mol/h]': np.zeros(8784), 'O2 WWTP [mol/h]': np.zeros(8784), 'H2O cons [mol/h]': np.zeros(8784), 'Meth in temp [C]': np.zeros(8784), \
                                         'CH4 out [mol/h]': np.zeros(8784), 'H2 out [mol/h]': np.zeros(8784), 'CO2 out [mol/h]': np.zeros(8784), 'H2O(g) out [mol/h]': np.zeros(8784), 'H2O(l) out [mol/h]': np.zeros(8784), 'CH4 flared [mol/h]': np.zeros(8784)})
    else:
        return pd.DataFrame({'Biogas (CH4) [mol/h]': np.zeros(8760), 'Biogas (CO2) [mol/h]': np.zeros(8760), 'H2 demand [mol/h]': np.zeros(8760), 'AEF emissions [gCO$_2$/kWh]': np.zeros(8760), 'AEF emissions [gCO$_2$/kWh]': np.zeros(8760), 'Elspot [€/MWh]': np.zeros(8760), 'Elz dispatch [kWh/h]': np.zeros(8760), \
                             'System dispatch [kWh/h]': np.zeros(8760), 'Meth el [kWh/h]': np.zeros(8760), 'Biogas comp [kWh/h]': np.zeros(8760), 'Standby': np.zeros(8760), 'Elz cold start': np.zeros(8760), 'Grid use [kWh/h]': np.zeros(8760), 'Wind use [kWh/h]': np.zeros(8760), 'Wind gen [kWh/h]': np.zeros(8760), \
                                 'PV use [kWh/h]': np.zeros(8760), 'PV gen [kWh/h]': np.zeros(8760), 'Battery state [%]': np.zeros(8760), 'Stack efficiency [%]': np.zeros(8760), 'System efficiency [%]': np.zeros(8760), \
                                     'H2 production [kg/h]': np.zeros(8760), 'H2 to storage [mol/h]': np.zeros(8760), 'H2 from storage [mol/h]': np.zeros(8760), 'H2 storage [%]': np.zeros(8760), 'H2 to meth [mol/h]': np.zeros(8760), 'Meth CH4 in [mol/h]': np.zeros(8760), 'Meth CO2 in [mol/h]': np.zeros(8760), \
                                         'Elz heat [kWh/h]': np.zeros(8760), 'Meth heat [kWh/h]': np.zeros(8760), 'WWTP heat demand [kWh/h]': np.zeros(8760), 'O2 out [mol/h]': np.zeros(8760), 'O2 WWTP [mol/h]': np.zeros(8760), 'H2O cons [mol/h]': np.zeros(8760), 'Meth in temp [C]': np.zeros(8760), \
                                         'CH4 out [mol/h]': np.zeros(8760), 'H2 out [mol/h]': np.zeros(8760), 'CO2 out [mol/h]': np.zeros(8760), 'H2O(g) out [mol/h]': np.zeros(8760), 'H2O(l) out [mol/h]': np.zeros(8760), 'CH4 flared [mol/h]': np.zeros(8760)})
        
        
