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
        return pd.DataFrame({'Biogas (CH4) [mol/h]': np.zeros(8784), 'Biogas (CO2) [mol/h]': np.zeros(8784), 'H2 demand [mol/h]': np.zeros(8784), 'Emissions [gCO$_2$/kWh]': np.zeros(8784), 'Elspot [€/MWh]': np.zeros(8784), 'Elz dispatch [kWh/h]': np.zeros(8784), 'System dispatch [kWh/h]': np.zeros(8784), 'Standby el. [kWh/h]': np.zeros(8784), \
                             'Elz standby': np.zeros(8784), 'Elz cold start': np.zeros(8784), 'Elz grid [kWh/h]': np.zeros(8784), 'Elz wind [kWh/h]': np.zeros(8784), 'Wind gen [kWh/h]': np.zeros(8784), \
                                 'Elz PV [kWh/h]': np.zeros(8784), 'PV gen [kWh/h]': np.zeros(8784), 'Battery state [%]': np.zeros(8784), 'Battery discharging [kWh/h]': np.zeros(8784), 'H2 production [mol/h]': np.zeros(8784), 'Stack efficiency [%]': np.zeros(8784), 'System efficiency [%]': np.zeros(8784), \
                                 'Unmet demand [kgH2/h]': np.zeros(8784), 'H2 used [kg/h]': np.zeros(8784), 'H2 to meth [mol/h]': np.zeros(8784), 'H2 to storage [mol/h]': np.zeros(8784), 'H2 from storage [mol/h]': np.zeros(8784), \
                             'H2 storage [%]': np.zeros(8784), 'H2 overproduction [mol/h]': np.zeros(8784), 'Elz heat [kWh/h]': np.zeros(8784), 'H2 comp [kWh/h]': np.zeros(8784), 'H2 temp [C]': np.zeros(8784), \
                                 'O2 out [mol/h]': np.zeros(8784), 'H2O cons [mol/h]': np.zeros(8784), 'Biogas comp [kWh/h]': np.zeros(8784), 'Biogas temp [C]': np.zeros(8784), \
                                     'Meth CH4 in [mol/h]': np.zeros(8784), 'Meth H2 in [mol/h]': np.zeros(8784), 'Meth CO2 in [mol/h]': np.zeros(8784), 'Meth in temp [C]': np.zeros(8784), 'Preheating [kWh/h]': np.zeros(8784), \
                                         'Meth CH4 out [mol/h]': np.zeros(8784), 'Meth H2 out [mol/h]': np.zeros(8784), 'Meth CO2 out [mol/h]': np.zeros(8784), 'Microbial CO2 cons [mol/h]': np.zeros(8784), \
                                             'Meth H2O(g) out [mol/h]': np.zeros(8784), 'Meth H2O(l) out [mol/h]': np.zeros(8784), 'Meth el [kWh/h]': np.zeros(8784), 'Meth heat [kWh/h]': np.zeros(8784), \
                                             'Cond CH4 out [mol/h]': np.zeros(8784), 'Cond H2 out [mol/h]': np.zeros(8784), 'Cond CO2 out [mol/h]': np.zeros(8784), 'Cond H2O(l) out [mol/h]': np.zeros(8784), \
                                                 'Cond heat [kWh/h]': np.zeros(8784), 'Cond el [kWh/h]': np.zeros(8784), 'Cond temp out [C]': np.zeros(8784),  \
                                                 'H2O recirc [mol/h]': np.zeros(8784), 'CH4 out [mol/h]': np.zeros(8784), 'Recirc CH4 [mol/h]': np.zeros(8784), \
                                                     'Recirc H2 [mol/h]': np.zeros(8784), 'Recirc CO2 [mol/h]': np.zeros(8784), \
                                                     'CH4 loss [mol/h]': np.zeros(8784), 'H2 loss [mol/h]': np.zeros(8784), 'CO2 loss [mol/h]': np.zeros(8784), \
                                                         'Recirc temp [C]': np.zeros(8784), 'Recirc pres [bar]': np.zeros(8784), 'O2 WWTP [mol/h]': np.zeros(8784), 'O3 WWTP [mol/h]': np.zeros(8784), 'Flare fraction [-]': np.zeros(8784)})
    else:
        return pd.DataFrame({'Biogas (CH4) [mol/h]': np.zeros(8760), 'Biogas (CO2) [mol/h]': np.zeros(8760), 'H2 demand [mol/h]': np.zeros(8760), 'Emissions [gCO$_2$/kWh]': np.zeros(8760), 'Elspot [€/MWh]': np.zeros(8760), 'Elz dispatch [kWh/h]': np.zeros(8760), 'System dispatch [kWh/h]': np.zeros(8760), 'Standby el. [kWh/h]': np.zeros(8760), \
                             'Elz standby': np.zeros(8760), 'Elz cold start': np.zeros(8760), 'Elz grid [kWh/h]': np.zeros(8760), 'Elz wind [kWh/h]': np.zeros(8760), 'Wind gen [kWh/h]': np.zeros(8760), \
                                 'Elz PV [kWh/h]': np.zeros(8760), 'PV gen [kWh/h]': np.zeros(8760), 'Battery state [%]': np.zeros(8760), 'Battery discharging [kWh/h]': np.zeros(8760), 'H2 production [mol/h]': np.zeros(8760), 'Stack efficiency [%]': np.zeros(8760), 'System efficiency [%]': np.zeros(8760), \
                                 'Unmet demand [kgH2/h]': np.zeros(8760), 'H2 used [kg/h]': np.zeros(8760), 'H2 to meth [mol/h]': np.zeros(8760), 'H2 to storage [mol/h]': np.zeros(8760), 'H2 from storage [mol/h]': np.zeros(8760), \
                             'H2 storage [%]': np.zeros(8760), 'H2 overproduction [mol/h]': np.zeros(8760), 'Elz heat [kWh/h]': np.zeros(8760), 'H2 comp [kWh/h]': np.zeros(8760), 'H2 temp [C]': np.zeros(8760), \
                                 'O2 out [mol/h]': np.zeros(8760), 'H2O cons [mol/h]': np.zeros(8760), 'Biogas comp [kWh/h]': np.zeros(8760), 'Biogas temp [C]': np.zeros(8760), \
                                     'Meth CH4 in [mol/h]': np.zeros(8760), 'Meth H2 in [mol/h]': np.zeros(8760), 'Meth CO2 in [mol/h]': np.zeros(8760), 'Meth in temp [C]': np.zeros(8760), 'Preheating [kWh/h]': np.zeros(8760), \
                                         'Meth CH4 out [mol/h]': np.zeros(8760), 'Meth H2 out [mol/h]': np.zeros(8760), 'Meth CO2 out [mol/h]': np.zeros(8760), 'Microbial CO2 cons [mol/h]': np.zeros(8760), \
                                             'Meth H2O(g) out [mol/h]': np.zeros(8760), 'Meth H2O(l) out [mol/h]': np.zeros(8760), 'Meth el [kWh/h]': np.zeros(8760), 'Meth heat [kWh/h]': np.zeros(8760), \
                                             'Cond CH4 out [mol/h]': np.zeros(8760), 'Cond H2 out [mol/h]': np.zeros(8760), 'Cond CO2 out [mol/h]': np.zeros(8760), 'Cond H2O(l) out [mol/h]': np.zeros(8760), \
                                                 'Cond heat [kWh/h]': np.zeros(8760), 'Cond el [kWh/h]': np.zeros(8760), 'Cond temp out [C]': np.zeros(8760),  \
                                                 'H2O recirc [mol/h]': np.zeros(8760), 'CH4 out [mol/h]': np.zeros(8760), 'Recirc CH4 [mol/h]': np.zeros(8760), \
                                                     'Recirc H2 [mol/h]': np.zeros(8760), 'Recirc CO2 [mol/h]': np.zeros(8760), \
                                                     'CH4 loss [mol/h]': np.zeros(8760), 'H2 loss [mol/h]': np.zeros(8760), 'CO2 loss [mol/h]': np.zeros(8760), \
                                                         'Recirc temp [C]': np.zeros(8760), 'Recirc pres [bar]': np.zeros(8760), 'O2 WWTP [mol/h]': np.zeros(8760), 'O3 WWTP [mol/h]': np.zeros(8760), 'Flare fraction [-]': np.zeros(8760)})
        
        
