# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:29:00 2023

@author: Linus Engstam
"""

import math
import pandas as pd
import numpy as np


def compressor(
        flow, #mol/h
        temp_in: int = 80, #C
        p_in: int = 30, #bar
        p_out: int = 100, #bar
        n_isen: float = 0.7,
        n_motor: float = 0.95,
        year: float = 2021,
        N: float = 1,
        z = 1,
        k = 1.41
) -> pd.DataFrame:
    """
    Returns hourly compressor electricity consumption.
    
    Parameters
    ----------
    flow : int or float
        Biogas flow rate through the compressor [mol/h].
    temp_in : str
        Rated inlet temperature of the gas [C].
    p_in : int or float
        Gas inlet pressure [bar].
    p_out : str
        Gas outlet pressure [bar].
    n_isen : int or float
        Isentropic efficiency [fraction].
    n_motor : str
        Motor efficiency [fraction].
    year : int or float
        Simulation year.
    N : float
        Number of compressor stages.
    z : float
        Compressibility factor.
    k : float
        Ratio of specific heats.

    Returns
    -------
    power : array (8760/8784x1)
        Compressor electricity consumption [kW].

    Notes
    -----
    Assuming no part load efficiency variation and not temperature change.
    
    """    
    R = 8.314 # Ideal gas constant
    power = (N*(k/(k-1))*(z/n_isen)*(temp_in + 273.15)*flow*R*(((p_out/p_in)**((k-1)/(N*k)))-1)) / (n_motor*3600*1000) #[kWh] dividing to get mol/s and kW
        
    return power


def electrolyzer(
        dispatch,
        prod,
        aux,
        heat_time,
        startups,
        h2o_cons: int = 10,
        temp: int = 80,
        h2o_temp: int = 15,
        year: float = 2021,
) -> pd.DataFrame:
    """
    Returns hydrogen production [mol/h] to methanation/storage, heat production [kWh], 
    H2 compressor energy [kWh], H2 temperature [C] and O2 flow [mol/h] for one hour of operation
    
    Parameters
    ----------
    dispatch : array (8760/8784x1)
        Electricity input to the electrolyzer [kW].
    prod : array (8760/8784x1)
        Hydrogen production from the electrolyzer [kgH2/h].
    aux : float
        Auxiliary electricity consumption [kW].
    heat_time : float
        The time during which no uable heat is produced after a cold start [min].
    startups : array (8760/8784x1)
        Cold starts.
    h2o_cons : float
        Water consumption of the electrolyzer [lH2O/kgH2].
    temp : int or float
        Operating temperature [C].
    h2o_temp : float
        Input water temperature [C].
    year : float
        Simulation year.

    Returns
    -------
    h2_flow : array (8760/8784x1)
        Hydrogen production from the electrolyzer [mol/h].
    net_heat : array (8760/8784x1)
        Net heat generation of the electrolyzer [kW].
    T_out : float
        Outlet temperature [C].
    o2_flow : array (8760/8784x1)
        Oxygen production from the electrolyzer [mol/h].
    startups : array (8760/8784x1)
        Cold starts.
    h2o_cons : array (8760/8784x1)
        Water consumption of the electrolyzer [mol/h].
    stack_efficiency : array (8760/8784x1)
        Calculated stack efficiency [fraction].
    system_efficiency : array (8760/8784x1)
        Calculated system efficiency [fraction].
    heat : float
        Total (non-net) heat generation of the electrolyzer [kW].

    Notes
    -----
    None
    
    """
    # Calculate flows in mol/h
    h2_flow = prod * 1000 / 2.02 # H2 produced [mol/h] 
    o2_flow = h2_flow / 2 # O2 produced [mol/h] 
    h2o_cons = h2_flow * (h2o_cons*997/(1000*18.02/2.02)) # Water consumed [mol/h]
    
    # Efficiency calculation
    dispatch = np.round(dispatch,5)
    stack_efficiency = np.divide(prod * 39.4, dispatch - aux, out=np.zeros_like(dispatch)+1, where=(dispatch-aux)!=0)
    sys_efficiency = np.divide(prod * 39.4, dispatch, out=np.zeros_like(dispatch), where=dispatch!=0)
    
    # Thermal model
    h2o_heating = h2o_cons * 75.3 * (temp - h2o_temp) / (3600*1000) # #Input water heating [kWh] 75.3 is the specific heat capacity of water in J/(K*mol)
    heat = np.maximum((dispatch-aux) * (1-stack_efficiency),0) # Heat production [kWh/h]
    net_heat = heat - h2o_heating # Removing input water heating
    net_heat = heat - (net_heat*startups*heat_time/60) # Removing cold start heat loss
    
    if isinstance(dispatch,float): 
        T_out = temp # Assuming no temperature change
    else:
        if year == 2020:
            T_out = np.zeros(8784) + temp 
        else:
            T_out = np.zeros(8760) + temp   
    
    return h2_flow, net_heat, T_out, o2_flow, h2o_cons, stack_efficiency, sys_efficiency, heat


def mixer(
        h2, #[mol/h]
        co2, #[mol/h]
        ch4, #[mol/h]
        h2_temp: float = 80, #C
        bg_temp: float = 40, #C
) -> pd.DataFrame:
    """
    Returns gas flow and temperature after mixing of biogas, hydrogen and methane.
    
    Parameters
    ----------
    h2 : array (8760/8784x1)
        Hydrogen flow [mol/h].
    co2 : array (8760/8784x1)
        Carbon dioxide flow [mol/h].
    ch4 : array (8760/8784x1)
        Methane flow [mol/h].
    h2_temp : float
        Initial hydrogen temperature [C].
    bg_temp : float
        Initial biogas temperature [C].
        
    Returns
    -------
    total_flow : array (8760/8784x3)
        Array containing the total flow in the order [Hydrogen, Carbon dioxide, Methane] [mol/h].
    T_mix : array (8760/8784x1)
        Array containing the temperature of the gas mixture [C].
        
    Notes
    -----
    None
    
    """
    total_flow = np.array([h2, co2, ch4], dtype=float) #H2, CO2, CH4 [mol/h]
    
    #Temperature change
    #Specific heat capacities [J/mol*K] (Strumpler and Brosig)
    cp_h2 = 28.82
    cp_co2 = 37.11
    cp_ch4 = 35.31
    
    #Mixing flows. If no hydrogen flow, the temperature does not change
    if isinstance(h2,float):
        if total_flow[0] > 0:
            T_mix = ((cp_h2*h2*h2_temp)+(cp_co2*co2*bg_temp)+(cp_ch4*ch4*bg_temp)) / ((cp_h2*h2)+(cp_co2*co2)+(cp_ch4*ch4))
        else:
            T_mix = bg_temp
    else:
        T_mix = np.divide(((cp_h2*h2*h2_temp)+(cp_co2*co2*bg_temp)+(cp_ch4*ch4*bg_temp)), ((cp_h2*h2)+(cp_co2*co2)+(cp_ch4*ch4)), out=np.zeros_like(h2)+bg_temp, where=((cp_h2*h2)+(cp_co2*co2)+(cp_ch4*ch4))!=0)
    
    return total_flow, T_mix


# def preheater(
#         flow, #H2, CO2, CH4
#         T_in, #C
#         T_out, #C
# ) -> pd.DataFrame:
#     """ Returns energy required for inlet gas to reach methanation temperature """
    
#     #Specific heat capacities [J/mol*K] (Strumpler and Brosig)
#     cp_h2 = 28.82
#     cp_co2 = 37.11
#     cp_ch4 = 35.31
    
#     heat_req = np.maximum(0,((cp_h2*flow[0]*(T_out-T_in)) + (cp_co2*flow[1]*(T_out-T_in)) + (cp_ch4*flow[2]*(T_out-T_in))) / (3600*1000)) #[kWh/h]
    
#     return heat_req


def methanation(
        meth_flow, #H2, CO2, CH4
        T, #C
        T_in,
        ch4_nm3_mol: float = 0.02243, #[Nm3/mol]
        microb_cons: float = 0.06,
        el_cons: float = 0.5, #[kWh/Nm3 CH4 produced]  In Schlautmann et al. 2020 for example.
        n: float = 0.998
) -> pd.DataFrame:
    """ 
    Returns post-methanation flow composition, electricity demand and waste heat.
    
    Parameters
    ----------
    meth_flow : array (8760/8784x3)
        Array containing the hourly inlet flow in the order [Hydrogen, Carbon dioxide, Methane] [mol/h].
    T : float
        Methanation operating temperature.
    T_in : array (8760/8784x1)
        Array containing the temperature of the gas mixture [C].
    ch4_nm3_mol : float
        Mol to volume conversion [Nm3/mol].
    microb_cons : float
        Microbial carbon dioxide consumption [fraction].
    el_cons : float
        Electricity consumption [kWh/Nm3CH4out].
    n : float
        Carbon dioxide conversion rate [fraction].
        
    Returns
    -------
    output_flow : array (8760/8784x4)
        Array containing the hourly inlet flow in the order [Hydrogen, Carbon dioxide, Methane, Water] [mol/h].
    el : array (8760/8784x1)
        Array containing the hourly electricity consumption [kW].
    heat : array (8760/8784x1)
        Array containing the hourly heat generation [kW].
    h2o_cond : array (8760/8784x1)
        Array containing the hourly flow of condensed water [mol/h].
    microbial_cons : array (8760/8784x1)
        Array containing the hourly microbial CO2 consumption [mol/h].
        
    Notes
    -----
    Assuming no specific part-load characterisitcs.
    """
    microbial_cons = meth_flow[1] * microb_cons # Microbial CO2 consumption
    co2_use = meth_flow[1] * (1-microb_cons)
    
    # Methanation process
    co2_conv = np.minimum(co2_use, meth_flow[0]/4) * n
    h2_conv = co2_conv * 4
    ch4_prod = co2_conv
    h2o_prod = co2_conv * 2
    cond_frac = 1 - (0.2504/10) # Saturated steam pressure at 65 degrees
    h2o_cond = h2o_prod * cond_frac
    h2o_gas = (1-cond_frac) * h2o_prod
    output_flow = np.array([(meth_flow[0]-h2_conv), (meth_flow[1]-co2_conv-microbial_cons), (meth_flow[2]+ch4_prod), h2o_gas])
    
    # Electricity consumption
    el = (output_flow[2]-meth_flow[2]) * ch4_nm3_mol * el_cons
        
    # Heat generation
    hf_h2 = 0
    hf_co2 = -393522
    hf_ch4 = -74873
    hf_h2o = -241827
    hoc = -40800 # [J/mol]
    
    #Outputs minus inputs (CH4 + H20 + Condensation) - (CO2 and H2)
    dHr = (((ch4_prod * hf_ch4) + (h2o_prod * hf_h2o) + (h2o_cond * hoc)) - ((co2_conv * hf_co2) + (h2_conv * hf_h2))) / (3600*1000) #[kWh/h]
    
    #If inlet temperature is higher than operating temperature the gas is assumed to be cooled to the operating and relase more heat from the process
    #Specific heat of input gas [J/(mol*K)]
    cp_h2 = 28.82
    cp_co2 = 37.11
    cp_ch4 = 35.31
    dHin = ((meth_flow[0] * cp_h2 * (T_in - T)) + (meth_flow[1] * cp_co2 * (T_in - T)) + (meth_flow[2] * cp_ch4 * (T_in - T))) / (3600*1000) #[kWh/h]
    
    heat = -dHr + dHin
        
    return output_flow, el, heat, h2o_cond, microbial_cons

#Should the condenser also cool to a specified temperature? What temp is suitable? 4 C in Kirchbacher et al. 2018.
# def condenser(
#         flow, #H2, CO2, CH4, H2O
#         T_in,
#         n: float = 1,
#         year: float = 2021,
# ) -> pd.DataFrame:
#     """ Returns output flow, waste heat and electricity demand """    
#     h2o_removed = flow[3]     #Assuming 100 % H2O removal
#     output_flow = np.array([flow[0], flow[1], flow[2]]) #[H2, CO2, CH4]
    
#     # el = 0 # Assuming no energy consumed
    
#     #Heat of condensation
#     hoc = 40800 #[J/mol] source?
#     heat = hoc * h2o_removed / (1000*3600) #[kWh/h]
#     # Implement temperature reduction heat!
#     # Usable heat
#     heat = heat * n
    
#     if isinstance(h2o_removed,float):
#         T_out = 4 #Kirschbacher et al. 2018
#     else:
#         if year == 2020:
#             T_out = np.zeros(8784) + 4
#         else:
#             T_out = np.zeros(8760) + 4
    
#     return output_flow, h2o_removed, T_out, heat#, el


# def membrane(
#         mem_inlet_flow, #H2, CO2, CH4
#         T_in: int = 65,
#         p_in: int = 7,
#         year: float = 2021,
# ) -> pd.DataFrame:
#     """ Returns purified flow composition, outlet temperature and pressure
#     as well as recirulation stream composition (and gas losses) """
#     #Assuming 100 % purification currently
    
#     outlet_flow = mem_inlet_flow[2] #[mol/h] CH4
#     if isinstance(outlet_flow,float):
#         loss = np.array([mem_inlet_flow[0], mem_inlet_flow[1], 0], dtype=float) #[mol/h] H2, CO2, CH4
#     else:
#         if year == 2020:
#             loss = np.array([mem_inlet_flow[0], mem_inlet_flow[1], np.zeros(8784)], dtype=float) #[mol/h] H2, CO2, CH4 
#         else:
#             loss = np.array([mem_inlet_flow[0], mem_inlet_flow[1], np.zeros(8760)], dtype=float) #[mol/h] H2, CO2, CH4 
#     # recirc = np.array([(mem_inlet_flow[0]-loss[0]), (mem_inlet_flow[1]-loss[1]), (mem_inlet_flow[2]-outlet_flow-loss[2])]) #[mol/h] H2, CO2, CH4
    
#     if isinstance(outlet_flow,float):
#         T_out = T_in
#         p_out = p_in
#     else:
#         if year == 2020:
#             T_out = np.zeros(8784) + T_in
#             p_out = np.zeros(8784) + p_in
#         else:
#             T_out = np.zeros(8760) + T_in
#             p_out = np.zeros(8760) + p_in
    
#     return outlet_flow, loss, T_out, p_out#, recirc



    
    
    