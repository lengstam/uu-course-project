# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 16:54:39 2023

@author: Linus Engstam
"""
import pandas as pd
import numpy as np
import math

def biogas_plant(
        data: str = "real",
        size: float = 1,
        comp: float = 1,
        year: float = 2021,
) -> pd.DataFrame:
    """ Returns a dataframe with hourly biogas plant outlet flow composition"""
    #Flow rates
    ch4_lhv_mol = 0.22295600000000002
    if data == "set":
        #input_comp = np.array([biogas_comp,]*8760)
        ch4_rate = np.zeros(8760) + (size*1000) / ch4_lhv_mol #mol/h
        co2_rate = ch4_rate * (comp[1]/comp[0]) #mol/h
        heat_demand = size * 1000 * 0.05 #assumption for now
        if year == 2020:
            ch4_rate = np.zeros(8784) + (size*1000) / ch4_lhv_mol #mol/h
            co2_rate = ch4_rate * (comp[1]/comp[0]) #mol/h
    elif data == "real":
        bg_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\Biogas flow.xlsx'
        bg_data = pd.read_excel(bg_read)
        ch4_rate = bg_data.iloc[:,0] #Nm3/h
        co2_rate = bg_data.iloc[:,1] #Nm3/h
        nm3_to_mol = 0.022414 #Nm3 to mol at 0 C and 1 atm for ideal gas
        ch4_rate = ch4_rate / nm3_to_mol
        co2_rate = co2_rate / nm3_to_mol
        ch4_rate.replace(np.nan,0)
        co2_rate.replace(np.nan,0)
        if year == 2020:
            ch4_rate = pd.concat([ch4_rate,ch4_rate.iloc[-24:]])
            # ch4_rate = pd.concat([ch4_rate,ch4_rate.iloc[-26]])
            # ch4_rate = ch4_rate.append(ch4_rate.iloc[-24:-1])
            # ch4_rate = ch4_rate.append(ch4_rate.iloc[-26])
            # ch4_rate = list(ch4_rate)
            co2_rate = pd.concat([co2_rate,co2_rate.iloc[-24:]])
            # co2_rate = pd.concat([co2_rate,co2_rate.iloc[-26]])
            # co2_rate = co2_rate.append(co2_rate.iloc[-24:-1])
            # co2_rate = co2_rate.append(co2_rate.iloc[-26])
            # co2_rate = list(co2_rate)
            ch4_rate = ch4_rate.reset_index(drop=True)
            co2_rate = co2_rate.reset_index(drop=True)
        
        #Should this be determined here? Only if variable I guess.
        # heat_demand = np.zeros(8760) #kWh/h
    
    return pd.DataFrame({'CH4': ch4_rate, 'CO2': co2_rate})#, heat_demand


def byprod_loads(
        o2_scale: float = 1, #scaling factor for o2 flow
        heat_scale: float = 1, #scaling factor for heat flow
        year: float = 2021,
) -> pd.DataFrame:
    """ Returns an hourly oxygen demand for the WWTP"""
    #Read and process data
    o2_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\O2 flow.xlsx'
    o2_data = pd.read_excel(o2_read).iloc[:,0] * o2_scale
    heat_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\Heat demand.xlsx'
    total_heat = pd.read_excel(heat_read).iloc[:,0] * heat_scale
    digester_heat = pd.read_excel(heat_read).iloc[:,1] * heat_scale
    aux_heat = pd.read_excel(heat_read).iloc[:,2] * heat_scale
    if year == 2020:
        o2_data = pd.concat([o2_data,o2_data.iloc[-24:]])
        # o2_data = o2_data.append(o2_data.iloc[-24:])
        # o2_data = o2_data.append(o2_data.iloc[-26])
        total_heat = pd.concat([total_heat,total_heat.iloc[-24:]])
        # total_heat = total_heat.append(total_heat.iloc[-24:])
        # total_heat = total_heat.append(total_heat.iloc[-26])
        digester_heat = pd.concat([digester_heat,digester_heat.iloc[-24:]])
        # digester_heat = digester_heat.append(digester_heat.iloc[-24:])
        # digester_heat = digester_heat.append(digester_heat.iloc[-26])
        aux_heat = pd.concat([aux_heat,aux_heat.iloc[-24:]])
        # aux_heat = aux_heat.append(aux_heat.iloc[-24:])
        # aux_heat = aux_heat.append(aux_heat.iloc[-26])
        o2_data = o2_data.reset_index(drop=True)
        total_heat = total_heat.reset_index(drop=True)
        aux_heat = aux_heat.reset_index(drop=True)
        digester_heat = digester_heat.reset_index(drop=True)
    
    return o2_data, total_heat, digester_heat, aux_heat

def electrolyzer_simple(
        size: int = 1, #MW
        n: float = 0.6, #LHV
        min_load: float = 0.1, #minimum load
        startup_time: int = 5, #minutes (cold start)
        cooldown_time: int = 6, #hours
        temp: int = 80, #C
        pressure: int = 30, #bar
) -> pd.DataFrame:
    """ Returns part load efficiency and heat production of the electrolyzer"""
    
    load_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] #make these larger for increased accuracy
    efficiency = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6] #LHV
    
    heat_gen = size * 1000 * (1-((39.4/33.3)*n)) #HHV efficiency
    waste_heat = [heat_gen, heat_gen, heat_gen, heat_gen, heat_gen, heat_gen, heat_gen, heat_gen, heat_gen, heat_gen]
    
    elz_part_load = pd.DataFrame({'Load range': load_range, 'Efficiency': efficiency, 'Waste heat': waste_heat})
    
    return elz_part_load


def electrolyzer(
        stack_efficiency, #at rated power
        system_efficiency, #at rated power
        pwl_points, #number of linearized segments
        elz_size, #[MW]
        # stack_rep, #[years/hours?]
        degr_mode: str = "annual", #average/annual (meaning degradation value per FLHs or year)
        degr_year: float = 5.0, #which year of operation to consider for degradation
        elz_degr: float = 0.0, #per FLHs or [year]?
) -> pd.DataFrame:
    """ Returns piece-wise linearization parameters for part load efficiency"""
    
    #The following code was taken from Ginsberg et al. baseline efficiency
    x = np.linspace(0, 6, num=60001)
    #x = np.round(x, 1) #rounding needs to be 1
    
    Fit_1 = 1.44926681  #C
    Fit_2 = 2.71725684 #A
    Fit_3 = 0.06970714 #K
    Y = lambda X: (Fit_1 + Fit_2 * (1 - math.exp(-Fit_3 * X)))
    Y_vector = np.vectorize(Y)
    y_fit_baseline = Y_vector(x)  
    #converting to efficiency
    u_th = 1.481
    eff_curve = u_th / y_fit_baseline
    #find current density with the closest efficiency to what we are aiming for and corresponding efficiency
    rated_eff = min(eff_curve, key=lambda x:abs(x-stack_efficiency))
    rated_current_index = np.where(eff_curve == rated_eff)[0][0]
    # rated_current_index = intervalIndex(rated_current_index,pwl_points) #want evenly sized intervals
    # rated_current = x[rated_current_index] #[A/cm2]
    rated_eff = eff_curve[rated_current_index]
    #auxiliary consumption
    # aux_cons = (rated_eff - system_efficiency) * elz_size * 1000 #[kW]
    aux_cons = elz_size*1000 - (elz_size*1000*system_efficiency/stack_efficiency) #[kW]
    #create efficiency curves
    # interval_length = rated_current_index / pwl_points
    stack_range = []
    stack_efficiency_curve = []
    system_efficiency_curve = []
    h2_prod = []
    # heat_prod = []
    system_range = []
    for i in range(pwl_points+1):
        # system_range.append(interval_length*(i)/rated_current_index)
        system_range.append(i/(pwl_points))
        # stack_range.append(interval_length*(i)/rated_current_index)
        stack_range.append(((system_range[i]*elz_size*1000)-aux_cons)/((elz_size*1000)-aux_cons))
        stack_range[0] = 0.0
        # stack_eff = min(eff_curve, key=lambda x:abs(x-stack_efficiency))
        # current_index = np.where(eff_curve == stack_eff)[0][0]
        stack_efficiency_curve.append(eff_curve[round(stack_range[i]*rated_current_index)])
        # stack_efficiency_curve.append(eff_curve[int(interval_length*(i))])
        # h2_prod.append(stack_range[i]*((elz_size*1000)-aux_cons)*stack_efficiency_curve[i]/39.4)
        h2_prod.append(stack_range[i]*((elz_size*1000)-aux_cons)*stack_efficiency_curve[i]/39.4)
        if i == 0:
            system_efficiency_curve.append(0)
        else:
            system_efficiency_curve.append((h2_prod[i]*39.4)/(system_range[i]*(elz_size*1000)))
        # heat_prod.append((1-stack_efficiency_curve[i]))

    
    #degradation
    if elz_degr > 0:
        #only determining stack efficiency after half its lifetime (rounding up) to account for an "average" year
        #Non-linear degradation
        # degradation_factor = math.pow(1-elz_degr,degr_year)
        #Assuming linear degradation (in %-points)
        degradation_factor = elz_degr*degr_year/100
        stack_efficiency_curve_degr = np.array(stack_efficiency_curve) - degradation_factor
        elz_size_degr = ((((elz_size*1000)-aux_cons) * (stack_efficiency_curve[-1]/stack_efficiency_curve_degr[-1])) + aux_cons) / 1000
        h2_prod1 = np.array(stack_range)*((elz_size*1000)-aux_cons)*stack_efficiency_curve/39.4
        # h2_prod = np.array(h2_prod)
        # system_efficiency_curve = (h2_prod*39.4)/(np.array(system_range)*(elz_size*1000)),
        # system_efficiency_curve = np.divide((h2_prod1*39.4), (np.array(system_range)*(elz_size*1000)), out=np.zeros_like(system_range), where=(np.array(system_range)*(elz_size*1000))!=0)
        system_efficiency_curve = np.divide((h2_prod1*39.4), (np.array(system_range)*(elz_size_degr*1000)), out=np.zeros_like(system_range), where=(np.array(system_range)*(elz_size*1000))!=0)
        system_efficiency_curve[0] = 0
        # heat_prod = 1 - stack_efficiency_curve
    
    #plotting
    # plt.plot(np.array(stack_range)*100,np.array(system_efficiency_curve)*100, label='System efficiency')
    # plt.plot(np.array(stack_range)*100,np.array(stack_efficiency_curve)*100, label='Stack efficiency')
    # plt.ylabel('Efficiency [%]')
    # plt.xlabel('Load range [%]')
    # plt.legend()
    # plt.plot(stack_range,h2_prod)
    
    #piece-wise linearization (y=k*x+m form)
    k_values = []
    m_values = []
    for i in range(pwl_points):
        k_values.append((h2_prod[i+1]-h2_prod[i])/(system_range[i+1]-system_range[i]))
        if i == 0:
            m_values.append((h2_prod[i]))
        else:
            m_values.append(h2_prod[i] - (system_range[i]*k_values[i]))
    
    
    # def intervalIndex(a, b):
    #     c = int(a / b)
    #     a1 = b * c
    #     if (a * b) > 0:
    #         a2 = b * (c + 1)
    #     else:
    #         a2 = b * (c - 1)
     
    #     if abs(a - a1) < abs(a - a2):
    #         return a1
    #     return a2
    
    return k_values, m_values, aux_cons, system_efficiency_curve[-1], stack_efficiency_curve_degr[-1]


def methanation(
        size: int = 1, #MW
        n: float = 0.99,
        meth_type: str = "bio",
        min_load: float = 0, #minimum load
        startup_time: int = 15, #minutes (cold start)
        cooldown_time: int = 6, #hours
        temp: int = 65, #C
        pressure: int = 7, #bar
) -> pd.DataFrame:
    """ Returns part load efficiency and heat production of the methanation reactor"""
    """ What to do here exactly? """

    load_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    efficiency = [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99] #CO2 conversion?
    
    heat_gen = size * 0.2
    waste_heat = [heat_gen*load_range[0], heat_gen*load_range[1], heat_gen*load_range[2], heat_gen*load_range[3], heat_gen*load_range[4], \
                  heat_gen*load_range[5], heat_gen*load_range[6], heat_gen*load_range[7], heat_gen*load_range[8], heat_gen*load_range[9], heat_gen*load_range[10]]
    
    el = size * 0.1
    el_demand = [el*load_range[0], el*load_range[1], el*load_range[2], el*load_range[3], el*load_range[4], \
                 el*load_range[5], el*load_range[6], el*load_range[7], el*load_range[8], el*load_range[9], el*load_range[10]]
    
    meth_part_load = pd.DataFrame({'Load range': load_range, 'Efficiency': efficiency, 'Waste heat': waste_heat, 'Electricity demand': el_demand})
    
    return meth_part_load


def renewables(
        wind_size: float = 0.5, #MW
        pv_size: float = 0.5, #MW
        pv_degr: float = 0.0, #[%/y]
        lifetime: float = 20.0, #[y]
        year: float = 2021,
) -> pd.DataFrame:
    """ Returns hourly wind and PV generation """
    #Not considered cut-in and cut-out speeds for wind.
    
    wind_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\wind (Uppsala).xlsx'
    wind_gen = np.array(pd.read_excel(wind_read) * (wind_size/3))[0:8760,0] #removing last day
    #wind_read = pd.read_excel('wind (Uppsala).xlsx')
    pv_read = wind_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\solar (Uppsala).xlsx'
    pv_gen = np.array(pd.read_excel(pv_read) * (pv_size/3))[0:8760,0] #removing last day
    pv_gen = pv_gen * (1-(round(lifetime/2)*pv_degr/100))
    #pv_read = pd.read_excel('solar (Uppsala).xlsx')
    #wind_gen = wind_read.iloc[:,0] * (wind_size/3)
    #pv_gen = pv_read.iloc[:,0] * (pv_size/3)
    if year == 2020:
        wind_gen = np.array(pd.read_excel(wind_read) * (wind_size/3))[0:,0]
        pv_gen = np.array(pd.read_excel(pv_read) * (pv_size/3))[0:,0]
    
    return pd.DataFrame({'Wind generation': wind_gen, 'PV generation': pv_gen})


def compressor(
        flow, #mol/s
        temp_in: int = 80, #C
        p_in: int = 30, #bar
        p_out: int = 100, #bar
        n_isen: float = 0.7,
        n_motor: float = 0.95,
) -> pd.DataFrame:
    """ Returns compressor size """
    N = 1#math.log10(p_out/p_in)/math.log10(3.1) #Khan et al. ch. 5.2
    z = 1 #assumption, should be a little higher depending in p and T
    k = 1.41
    R = 8.314
    T = temp_in + 273.15
    comp_size = (N*(k/(k-1))*(z/n_isen)*T*flow*R*(((p_out/p_in)**((k-1)/(N*k)))-1)) / (n_motor*1000) #kW

    return comp_size





