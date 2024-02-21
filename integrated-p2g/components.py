# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:29:00 2023

@author: Linus Engstam
"""

import math
import pandas as pd
import numpy as np

#Assume intercooling and utilize the waste heat?
#What about automatic cooling during decompression when exiting storage? Could be assumed to be heated by the ambient?
def compressor(
        flow, #mol/h
        temp_in: int = 80, #C
        p_in: int = 30, #bar
        p_out: int = 100, #bar
        n_isen: float = 0.7,
        n_motor: float = 0.95,
        year: float = 2021,
) -> pd.DataFrame:
    """ Returns compressor size """
    """ Assuming no part load efficiency variation
    (for biogas compressor this will likely have little impact)"""
    
    N = 1#round(math.log10(p_out/p_in)/math.log10(3.1)) #Khan et al. ch. 5.2
    z = 1 #assumption, should be a little higher depending in p and T
    k = 1.41
    R = 8.314
    T_in = temp_in + 273.15
    power = (N*(k/(k-1))*(z/n_isen)*T_in*flow*R*(((p_out/p_in)**((k-1)/(N*k)))-1)) / (n_motor*3600*1000) #[kWh] dividing to get mol/s and kW

    #Temperature increase (assume ideal gas, pV=nRT, https://phys.libretexts.org/Bookshelves/University_Physics/Book%3A_University_Physics_(OpenStax)/Book%3A_University_Physics_II_-_Thermodynamics_Electricity_and_Magnetism_(OpenStax)/03%3A_The_First_Law_of_Thermodynamics/3.07%3A_Adiabatic_Processes_for_an_Ideal_Gas#:~:text=When%20an%20ideal%20gas%20is,work%20and%20its%20temperature%20drops.)
    #T_out = (p_out/p_in) * ((p_in/p_out)**(1/k)) * T_in
    #Does it make sense to assume such an increase? Probably not as very large...
    #T_out = T_in - 273.15
    if isinstance(flow,float):
        T_out = T_in - 273.15
    else:
        if year == 2020:
            T_out = np.zeros(8784) + T_in - 273.15
        else:
            T_out = np.zeros(8760) + T_in - 273.15
        
    return power, T_out

#In a more detailed electrolyzer: from a cold start, some "waste" heat could be used to heat up the electrolyzer to operating temperature.
#Need to define how long this will take and define thermal parameters based on that.
#Storage etc. should not be handled within the electrolyzer component!
def electrolyzer_simple(
        dispatch,
        params,
        #demand,
        temp: int = 80,
        p: int = 30,
        #h2st_p: int = 100,
        size: int = 1000, #kWh
        #n_isen: float = 0.7,
        #n_motor: float = 0.95,
        h2o_temp: int = 15,
) -> pd.DataFrame:
    """ Returns hydrogen production [mol/h] to methanation/storage, heat production [kWh], 
    H2 compressor energy [kWh], H2 temperature [C] and O2 flow [mol/h] for one hour of operation """
    #Yet to consider startup and water use/recirculation (water does only need to be in economic part I think)
    
    #Definitions
    h2_lhv_mol = 0.06726599999999999
    
    #Determine efficiency at current hour
    load = dispatch / size
    part_load_n = np.interp(load, params.iloc[:,0], params.iloc[:,1])
    
    #Calculate gas production (how to handle mismatch between MILP and this more detailed model?)
    h2_flow = dispatch * part_load_n / h2_lhv_mol #[mol/h] H2 produced
    #h2st_in = max(0,(h2_flow-demand)) #[mol/h] H2 flow into storage
    #h2st_out = max(0,(demand-h2_flow)) #[mol/h] H2 flow out from storage
    o2_flow = h2_flow / 2 #[mol/h]
    h2o_cons = h2_flow #[mol/h]
    #h2_meth = h2_flow - h2st_in + h2st_out #[mol/h] H2 to methanation
    
    #Thermal model
    #Input water heating
    h2o_heating = h2o_cons * 75.3 * (temp - h2o_temp) / (3600*1000) #[kWh] 75.3 is the specific heat capacity of water in J/(K*mol)
    #Heat production
    heat = dispatch * (1-((39.4/33.3)*part_load_n)) #[kWh/h]
    net_heat = heat - h2o_heating
    if isinstance(dispatch,float):
        T_out = temp
    else:
        T_out = np.zeros(8760) + temp
    
    # #H2 compression (only what goes into the storage)
    # comp_power, T_out = compressor(flow=h2st_in, temp_in=temp, p_in=p, p_out=h2st_p, n_isen=n_isen, n_motor=n_motor)
    
    #return h2_flow, h2st_in, h2st_out, heat, comp_power, T_out, o2_flow, h2o_cons, h2_meth
    return h2_flow, net_heat, T_out, o2_flow, h2o_cons

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
    """ Returns hydrogen production [mol/h] to methanation/storage, heat production [kWh], 
    H2 compressor energy [kWh], H2 temperature [C] and O2 flow [mol/h] for one hour of operation """
   
    #Calculate gas production in mol/h
    h2_flow = prod * 1000 / 2.02 #[mol/h] H2 produced
    o2_flow = h2_flow / 2 #[mol/h]
    h2o_cons = h2_flow * (h2o_cons*997/(1000*18.02/2.02)) #[mol/h] 10 l/kg H2 from several sources, e.g. Nel Hydrogen
    
    #Efficiency calculation
    # stack_efficiency = prod * 39.4 / dispatch #HHV
    # sys_efficiency = prod * 39.4 / (dispatch + aux) #HHV
    dispatch = np.round(dispatch,5)
    
    stack_efficiency = np.divide(prod * 39.4, dispatch - aux, out=np.zeros_like(dispatch)+1, where=(dispatch-aux)!=0)
    sys_efficiency = np.divide(prod * 39.4, dispatch, out=np.zeros_like(dispatch), where=dispatch!=0)
    
    #Thermal model
    #Input water heating
    h2o_heating = h2o_cons * 75.3 * (temp - h2o_temp) / (3600*1000) #[kWh] 75.3 is the specific heat capacity of water in J/(K*mol)
    #Water evaporation
    # hoc = -40800 #[J/mol] source?
    # h2o_evap = h2o_cons * -hoc / (3600*1000)
    #Output hydrogen, oxygen and steam heat loss (including this reduces heat generation by ~2 %)
    # cp_h2 = 28.82
    # cp_o2 = 29.39 #https://webbook.nist.gov/cgi/cbook.cgi?ID=C7782447&Mask=1&Type=JANAFG&Table=on
    # cp_steam = 33.59
    # T_amb = h2o_temp
    # h2_heat_loss = h2_flow * cp_h2 * (temp-T_amb) / (3600*1000) #[kWh/h]
    # o2_heat_loss = o2_flow * cp_o2 * (temp-T_amb) / (3600*1000) #[kWh/h]
    # p_anode = 30
    # p_cathode = 30
    # p_sat_80 = 0.4772 #saturation pressure at 80 C
    # steam_anode = (p_sat_80/(p_anode-p_sat_80)) * o2_flow
    # steam_cathode = (p_sat_80/(p_cathode-p_sat_80)) * h2_flow
    # steam_heat_loss = (steam_anode+steam_cathode) * cp_steam * (temp-T_amb) / (3600*1000) #[kWh/h]
    #Heat production
    heat = np.maximum((dispatch-aux) * (1-stack_efficiency),0) #[kWh/h]
    net_heat = heat - h2o_heating# - h2_heat_loss - o2_heat_loss - steam_heat_loss
    #Cold start heat loss
    net_heat = net_heat - (net_heat*startups*heat_time/60) #how much of the hour is spent at too low temperature during cold starts?
    
    if isinstance(dispatch,float):
        T_out = temp
    else:
        if year == 2020:
            T_out = np.zeros(8784) + temp 
        else:
            T_out = np.zeros(8760) + temp   
    
    #return h2_flow, h2st_in, h2st_out, heat, comp_power, T_out, o2_flow, h2o_cons, h2_meth
    return h2_flow, net_heat, T_out, o2_flow, h2o_cons, stack_efficiency, sys_efficiency, heat

#should define recirculation here as well if included?
def mixer(
        h2, #[mol/h]
        co2, #[mol/h]
        ch4, #[mol/h]
        h2_temp: int = 80, #C
        bg_temp: int = 40, #C
) -> pd.DataFrame:
    """ Returns gas flow after mixing of biofas, hydrogen (and recirculated flow) """
    
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
    
    # if total_flow[0] > 0:
    #     T_mix = ((cp_h2*h2*h2_temp)+(cp_co2*co2*bg_temp)+(cp_ch4*ch4*bg_temp)) / ((cp_h2*h2)+(cp_co2*co2)+(cp_ch4*ch4))
    # else:
    #     T_mix = bg_temp
    
    return total_flow, T_mix


def preheater(
        flow, #H2, CO2, CH4
        T_in, #C
        T_out, #C
) -> pd.DataFrame:
    """ Returns energy required for inlet gas to reach methanation temperature """
    
    #Specific heat capacities [J/mol*K] (Strumpler and Brosig)
    cp_h2 = 28.82
    cp_co2 = 37.11
    cp_ch4 = 35.31
    
    heat_req = np.maximum(0,((cp_h2*flow[0]*(T_out-T_in)) + (cp_co2*flow[1]*(T_out-T_in)) + (cp_ch4*flow[2]*(T_out-T_in))) / (3600*1000)) #[kWh/h]
    
    return heat_req


# def methanation_simple(
#         flow, #H2, CO2, CH4
#         rated_flow, #[mol/h]
#         params, #Load range, efficiency, heat, el
#         T, #C
#         microb_cons: float = 0.06,
#         meth_type: str = "bio",
# ) -> pd.DataFrame:
#     """ Returns post-methanation flow composition, electricity demand and waste heat 
#     CURRENTLY NOT CONSIDERING STARTUP AND RAMPING CONSTRAINTS """
    
#     #Part load behavior (should this be defined on hydrogen share of input flow?)
#     load = flow.sum() / rated_flow
#     n = np.interp(load, params.iloc[:,0], params.iloc[:,1]) #CO2 conversion?
#     #heat = np.interp(load, params.iloc[:,0], params.iloc[:,2]) * load * 1000 #[kWh/h] #Using method at the bottom instead
#     el = np.interp(load, params.iloc[:,0], params.iloc[:,3]) * load * 1000 #[kWh/h]
    
#     #Microbial CO2 consumption
#     flow[1] = flow[1] * (1-microb_cons)
    
#     #Methanation process
#     co2_conv = min(flow[1], flow[0]/4) * n
#     h2_conv = co2_conv * n * 4
#     ch4_prod = co2_conv
#     h2o_prod = co2_conv * 2
#     if meth_type == "bio": #gas contra liquid water output
#         h2o_cond = h2o_prod * 0.966 #(Goffart de Roeck et al.)
#         h2o_gas = (1-0.966) * h2o_prod
#     elif meth_type == "cat":
#         h2o_cond = 0
#         h2o_gas = h2o_prod

    
#     #Microbes consume some CO2 as well? This would increase the H2 demand of the process.

#     output_flow = np.array([(flow[0]-h2_conv), (flow[1]-co2_conv), (flow[2]+ch4_prod), h2o_gas]) #H2, CO2, CH4, H2O(g)
    
#     #Heat generation (Find values at operating temperature, below are for 25 C?)
#     #Heat of formation [J/mol] (Strumpler and Brosig)
#     #Could simply use heat of reaction (-165kJ/mol) and heat of condenstation?
#     #See Thema et al. (2019) for example of mass and heat flow equations. Could include electricity losses for heat
#     hf_h2 = 0
#     hf_co2 = -393522
#     hf_ch4 = -74873
#     hf_h2o = -241827
#     hoc = -40800 #[J/mol] source?
    
#     #Outputs minus inputs (CH4 + H20 + Condensation) - (CO2 and H2)
#     dH = (((ch4_prod * hf_ch4) + (h2o_prod * hf_h2o) + (h2o_cond * hoc)) - ((co2_conv * hf_co2) + (h2_conv * hf_h2))) / (3600*1000) #[kWh/h]
#     heat = -dH
    
#     return output_flow, el, heat, h2o_cond

def methanation(
        meth_flow, #H2, CO2, CH4
        rated_flow, #[mol CO2 reacted/h]
        T, #C
        T_in,
        ch4_nm3_mol: float = 0.02243, #[Nm3/mol]
        microb_cons: float = 0.06,
        el_cons: float = 0.5, #[kWh/Nm3 CH4 produced]  In Schlautmann et al. 2020 for example.
        meth_type: str = "bio",
        hot_start_time: float = 15.0, #hot start-up time in minutes
        cold_start_time: float = 60.0, #cold start-up time in minutes
        prev_mode: str = "on", #on/hot/cold standby from previous timestep
        n: float = 0.998
) -> pd.DataFrame:
    """ Returns post-methanation flow composition, electricity demand and waste heat 
    CURRENTLY NOT CONSIDERING STARTUP AND RAMPING CONSTRAINTS """
    #Conversion rate
    
    #Microbial CO2 consumption
    microbial_cons = meth_flow[1] * microb_cons
    # flow[1] = flow[1] * (1-microb_cons)
    co2_use = meth_flow[1] * (1-microb_cons)
    
    #Methanation process
    co2_conv = np.minimum(co2_use, meth_flow[0]/4) * n
    h2_conv = co2_conv * 4
    ch4_prod = co2_conv
    h2o_prod = co2_conv * 2
    cond_frac = 1 - (0.2504/10) #saturated steam pressure at 65 degrees   (0.966 #(Goffart de Roeck et al.))  
    h2o_cond = h2o_prod * cond_frac
    h2o_gas = (1-cond_frac) * h2o_prod

    output_flow = np.array([(meth_flow[0]-h2_conv), (meth_flow[1]-co2_conv-microbial_cons), (meth_flow[2]+ch4_prod), h2o_gas]) #H2, CO2, CH4, H2O(g)
    
    #Electricity consumption
    #Potential values: 0.5 kWh/Nm3 CH4 produced from Thema et al. 2019 for CSTR. Lower (~0.35 kWh/Nm3 CH4 converted) in De Roeck et al. 2022 and higher in Michailos et al. 2021 (0.16 kWh/Nm3 input gas --> ~0.67 per CH4 converted?)
    #Conservatively 2.5 % of electrolyzer input in Calbry-Muzyka. 0.5 kWh/Nm3 in Schlautmann as well.
    #Only accounting for converted CO2:
    el = (output_flow[2]-meth_flow[2]) * ch4_nm3_mol * el_cons #[kWh]
        
    #Heat generation (Find values at operating temperature, below are for 25 C?)
    #Heat of formation [J/mol] (Strumpler and Brosig)
    #Could simply use heat of reaction (-165kJ/mol) and heat of condenstation?
    #See Thema et al. (2019) for example of mass and heat flow equations. Could include electricity losses for heat
    hf_h2 = 0
    hf_co2 = -393522
    hf_ch4 = -74873
    hf_h2o = -241827
    hoc = -40800 #[J/mol] source?
    #could get from NIST: https://webbook.nist.gov/cgi/cbook.cgi?Name=water&Units=SI&cTG=on
    
    #Outputs minus inputs (CH4 + H20 + Condensation) - (CO2 and H2)
    dHr = (((ch4_prod * hf_ch4) + (h2o_prod * hf_h2o) + (h2o_cond * hoc)) - ((co2_conv * hf_co2) + (h2_conv * hf_h2))) / (3600*1000) #[kWh/h]
    
    #If inlet temperature is higher than operating temperature the gas is assumed to be cooled to the operating and relase more heat from the process
    #Specific heat of input gas [J/(mol*K)] (Strumpler and Brosig)
    cp_h2 = 28.82
    cp_co2 = 37.11
    cp_ch4 = 35.31
    dHin = ((meth_flow[0] * cp_h2 * (T_in - T)) + (meth_flow[1] * cp_co2 * (T_in - T)) + (meth_flow[2] * cp_ch4 * (T_in - T))) / (3600*1000) #[kWh/h]
    
    heat = -dHr + dHin
        
    return output_flow, el, heat, h2o_cond, microbial_cons

#Should the condenser also cool to a specified temperature? What temp is suitable? 4 C in Kirchbacher et al. 2018.
def condenser(
        flow, #H2, CO2, CH4, H2O
        T_in,
        n: float = 1,
        year: float = 2021,
) -> pd.DataFrame:
    """ Returns output flow, waste heat and electricity demand """
    #Assuming 100 % H2O removal
    
    h2o_removed = flow[3]
    output_flow = np.array([flow[0], flow[1], flow[2]]) #[H2, CO2, CH4]
    
    el = 0
    if isinstance(h2o_removed,float):
        T_out = 0
    else:
        T_out = np.zeros(8760)
    
    #Heat of condensation
    hoc = 40800 #[J/mol] source?
    heat = hoc * h2o_removed / (1000*3600) #[kWh/h]
    #Also do temperature reduction heat!
    #Usable heat
    heat = heat * n
    
    if isinstance(h2o_removed,float):
        T_out = 4 #Kirschbacher et al. 2018
    else:
        if year == 2020:
            T_out = np.zeros(8784) + 4
        else:
            T_out = np.zeros(8760) + 4
    
    return output_flow, heat, el, h2o_removed, T_out


def membrane(
        mem_inlet_flow, #H2, CO2, CH4
        T_in: int = 65,
        p_in: int = 7,
        year: float = 2021,
) -> pd.DataFrame:
    """ Returns purified flow composition, outlet temperature and pressure
    as well as recirulation stream composition (and gas losses) """
    #Assuming 100 % purification currently
    
    outlet_flow = mem_inlet_flow[2] #[mol/h] CH4
    if isinstance(outlet_flow,float):
        loss = np.array([mem_inlet_flow[0], mem_inlet_flow[1], 0], dtype=float) #[mol/h] H2, CO2, CH4
    else:
        if year == 2020:
            loss = np.array([mem_inlet_flow[0], mem_inlet_flow[1], np.zeros(8784)], dtype=float) #[mol/h] H2, CO2, CH4 
        else:
            loss = np.array([mem_inlet_flow[0], mem_inlet_flow[1], np.zeros(8760)], dtype=float) #[mol/h] H2, CO2, CH4 
    recirc = np.array([(mem_inlet_flow[0]-loss[0]), (mem_inlet_flow[1]-loss[1]), (mem_inlet_flow[2]-outlet_flow-loss[2])]) #[mol/h] H2, CO2, CH4
    
    if isinstance(outlet_flow,float):
        T_out = T_in
        p_out = p_in
    else:
        if year == 2020:
            T_out = np.zeros(8784) + T_in
            p_out = np.zeros(8784) + p_in
        else:
            T_out = np.zeros(8760) + T_in
            p_out = np.zeros(8760) + p_in
    
    return outlet_flow, recirc, loss, T_out, p_out



    
    
    