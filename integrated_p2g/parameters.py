# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 16:54:39 2023

@author: Linus Engstam
"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import components as comps

"""
Functions providing annual hourly datasets.
"""

def biogas_plant(
        data: str = "real",
        size: float = 1,
        comp: float = 1,
        year: int = 2021):
    """
    Returns an array containing the hourly biogas plant outlet flow in mol/h.
    
    Parameters
    ----------
    data: str {'real', 'set'}
        The input 'data' decides whether actual plant data should be imported ('real') or a constant flow be assumed ('set').
    size: float [mol/h], optional
        If 'data' is 'set', then a biogas plant size should be defined. This is done based on the hourly biogas flow in mol/h.
    comp: list, optional
        Furthermore, the biogas composition should be defined if 'data' is 'set'. This is done using a list of composition (fractions) in the following order: [CH4, CO2, etc.]
    year: int
        The simulation year is defined to determine the size of the output depending on the number of hours in a selected year.
        
    Returns
    -------
    biogas_flow: array [mol/h]
        A NumPy array containing hourly values of CH4 and CO2 flows is returned in the order [CH4, CO2].
    """
    if data == "set":
        if year == 2020:
            ch4_rate = np.zeros(8784) + (size*comp[0]) # Methane flow rate [mol/h]
            co2_rate = ch4_rate * (comp[1]/comp[0]) # Carbon dioxide flow rate [mol/h]
        else:
            ch4_rate = np.zeros(8760) + (size*comp[0]) # Methane flow rate [mol/h]
            co2_rate = ch4_rate * (comp[1]/comp[0]) # Carbon dioxide flow rate [mol/h]
    elif data == "real":
        bg_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\Biogas flow.xlsx' # Reading data
        bg_data = pd.read_excel(bg_read)
        ch4_rate = bg_data.iloc[:,0] # Methane flow rate [Nm3/h]
        co2_rate = bg_data.iloc[:,1] # Carbon dioxide flow rate [Nm3/h]
        nm3_to_mol = 0.022414 # Conversion factor from Nm3 to mol at 0 C and 1 atm for ideal gas
        ch4_rate = ch4_rate / nm3_to_mol # Methane flow rate [mol/h]
        co2_rate = co2_rate / nm3_to_mol # Carbon dioxide flow rate [mol/h]
        ch4_rate.replace(np.nan,0) # Assuming zero flow when no data is present
        co2_rate.replace(np.nan,0) # Assuming zero flow when no data is present
        if year == 2020:
            ch4_rate = pd.concat([ch4_rate,ch4_rate.iloc[-24:]])
            co2_rate = pd.concat([co2_rate,co2_rate.iloc[-24:]])
            ch4_rate = ch4_rate.reset_index(drop=True)
            co2_rate = co2_rate.reset_index(drop=True)
    
    biogas_flow = np.array([ch4_rate,co2_rate]).transpose()
    
    return biogas_flow


def byprod_loads(
        o2_scale: float = 1,
        heat_scale: float = 1,
        year: int = 2021):
    """
    Returns an array containing actual hourly heat and oxygen demands for the WWTP.
    
    Parameters
    ----------
    o2_scale: float
        Scaling factor used for varying the size (but not shape) of the oxygen demand.
    heat_scale: float
        Scaling factor used for varying the size (but not shape) of the heat demand.
    year: int
        The simulation year is defined to determine the size of the output depending on the number of hours in a selected year.
    
    Returns
    -------
    oxygen: array [mol/h]
        A full year of hourly oxygen demand values at the WWTP.
    total_heat: array [kWh/h]
        A full year of hourly overall heat demand values at the WWTP.
    digester_heat: array [kWh/h]
        A full year of hourly digester heat demand values at the WWTP.
    aux_heat: array [kWh/h]
        A full year of hourly auxiliary heat demand values at the WWTP.
    
    """
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
    
    oxygen = np.array(o2_data)
    total_heat = np.array(total_heat)
    digester_heat = np.array(digester_heat)
    aux_heat = np.array(aux_heat)
    
    return oxygen, total_heat, digester_heat, aux_heat


def renewables(
        wind_size: float = 0.5,
        pv_size: float = 0.5,
        pv_degr: float = 0.0,
        lifetime: int = 20,
        year: float = 2021):
    """
    Returns an array containing hourly wind and PV generation values.
    
    Parameters
    ----------
    wind_size: float [MW]
        Sets the rated wind capacity and scales the time series based on this.
    pv_size: float [MW]
        Sets the rated PV capacity and scales the time series based on this.
    pv_degr: float [%/yr]
        Annual degradation of the PV panel.
    lifetime: int [years]
        Lifetime of the PV system, used to estimated the average production considering panel degradation.
    year: int
        The simulation year is defined to determine the size of the output depending on the number of hours in a selected year.
    
    Returns
    -------
    wind_gen: array [kWh/h]
        Annual wind generation with an hourly resolution.
    pv_gen: array [kWh/h]
        Annual PV generation with an hourly resolution.
    """
    
    wind_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\wind (Uppsala).xlsx'
    wind_gen = np.array(pd.read_excel(wind_read) * (wind_size/3000))[0:8760,0] #removing last day
    pv_read = wind_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\solar (Uppsala).xlsx'
    pv_gen = np.array(pd.read_excel(pv_read) * (pv_size/3000))[0:8760,0] #removing last day
    pv_gen = pv_gen * (1-(round(lifetime/2)*pv_degr/100))
    if year == 2020:
        wind_gen = np.array(pd.read_excel(wind_read) * (wind_size/3))[0:,0]
        pv_gen = np.array(pd.read_excel(pv_read) * (pv_size/3))[0:,0]
    
    return wind_gen, pv_gen


"""
Classes containing technical parameters for each component as well as a single class for economic data.
"""

class Electrolyzer():
    """
    
    COULD MOVE ECONOMIC CALCULATION TO INSIDE HERE AS WELL?
    """
    # Class variables (constant)
    n_sys = 0.75 #HHV system efficiency at full load (part-load is estimated)
    n_stack = n_sys + 0.05 #HHV stack efficiency at full load
    start_time = 5 #minutes (cold start)
    start_cost = start_time/60 #0 #startup cost [fraction of rated power]
    standby_cost = 0.02 #standby cost [fraction of rated power]
    heatup_time = 60 #[minutes] the time during which no usable waste heat is released (look into in more detail)
    temp = 80 #C
    h2o_temp = 15 #[C] inlet water temperature
    pres = 30 #bar
    degr = 1 #[% per year/1000 FLHs depending on stack replacement time below] from Ginsberg, strengthened by van der Roest and within ranges of Buttler
    stack_rep = 10 #hours if > 1000, years if < 100. If year, define if it is replaced after ten years or at the start of year ten? After ten years now.
    degr_year = round(stack_rep/2) #which year of operation to consider for efficiency degradation (assuming 0 as year 1)
    water_cons = 10 #[l/kgH2]
    
    capex = 1500 #[€/kW] at 5 MW
    capex_ref = 5000 #[kW]
    opex = 4 #% of CAPEX
    scaling = 0.75 #scaling factor for CAPEX
    water_cost = 0.5 #€/m3 (including purification) Se förstudie för värde
    stack_cost = 0.5 #fraction of CAPEX
    
    def __init__(self, size):
        self.size = size * 1000 # [kW]
        self.h2_max = self.size * self.n_sys / 39.4
    
    # def sizing(self, size):
    #     self.size = size * 1000 # [kW]
    #     self.h2_max = self.size * self.n_sys / 39.4
    
    def efficiency(self, plot):
        """
        Returns piece-wise linearization parameters for part load efficiency. Must be called after defSize()
        The code was adapted from Ginsberg et al (LINK). Using baseline efficiency.
        
        Parameters
        ----------
        plot: str {'plot', ''}
            If 'plot', the linearized and non-linearized efficiency curves will be plotted.
        
        Returns
        -------
        k_values:
            
        m_values:
            
        aux_cons:
            
        sys_eff:
            
        stack_eff:

        """
        x = np.linspace(0, 6, num=60001)        
        Fit_1 = 1.44926681  # C
        Fit_2 = 2.71725684 # A
        Fit_3 = 0.06970714 # K
        Y = lambda X: (Fit_1 + Fit_2 * (1 - math.exp(-Fit_3 * X)))
        Y_vector = np.vectorize(Y)
        y_fit_baseline = Y_vector(x)  
        u_th = 1.481
        eff_curve = u_th / y_fit_baseline # Converting overvoltage to efficiency
        pwl_points = 10 # Ten point linearization
        
        # Find current density with the closest efficiency to what we are aiming for and corresponding efficiency
        rated_eff = min(eff_curve, key=lambda x:abs(x-self.n_stack))
        rated_current_index = np.where(eff_curve == rated_eff)[0][0]
        rated_eff = eff_curve[rated_current_index]
        # Auxiliary consumption
        self.aux_cons = self.size - (self.size*self.n_sys/self.n_stack) #[kW]
        # Create efficiency curves
        stack_range = []
        stack_efficiency_curve = []
        system_efficiency_curve = []
        h2_prod = []
        system_range = []
        # system_range = (np.arange(10)+1)*0.1
        # stack_range = ((system_range*self.size)-self.aux_cons)/(self.size-self.aux_cons)
        # stack_range[0] = 0.0
        # # stack_efficiency_curve = (eff_curve[np.round(stack_range*rated_current_index)])
        # stack_efficiency_curve = eff_curve[0:round(rated_current_index/pwl_points)*pwl_points:round(rated_current_index/pwl_points)]
        # h2_prod = (stack_range*(self.size-self.aux_cons)*stack_efficiency_curve/39.4)
        # system_efficiency_curve = np.divide(h2_prod*39.4, system_range*self.size, out=np.zeros_like(system_range), where=system_range!=0)
                                            
        for i in range(pwl_points+1):
            system_range.append(i/(pwl_points))
            stack_range.append(((system_range[i]*self.size)-self.aux_cons)/(self.size-self.aux_cons))
            stack_range[0] = 0.0
            stack_efficiency_curve.append(eff_curve[round(stack_range[i]*rated_current_index)])
            h2_prod.append(stack_range[i]*(self.size-self.aux_cons)*stack_efficiency_curve[i]/39.4)
            if i == 0:
                system_efficiency_curve.append(0)
            else:
                system_efficiency_curve.append((h2_prod[i]*39.4)/(system_range[i]*self.size))

        # Degradation. Only determining stack efficiency after half its lifetime (rounding up) to account for an "average" year. Assuming linear degradation (in %-points)
        degradation_factor = self.degr*self.degr_year/100
        stack_efficiency_curve_degr = np.array(stack_efficiency_curve) - degradation_factor
        elz_size_degr = (((self.size-self.aux_cons) * (stack_efficiency_curve[-1]/stack_efficiency_curve_degr[-1])) + self.aux_cons) / 1000
        h2_prod1 = np.array(stack_range)*(self.size-self.aux_cons)*stack_efficiency_curve/39.4
        system_efficiency_curve = np.divide((h2_prod1*39.4), (np.array(system_range)*(elz_size_degr*1000)), out=np.zeros_like(system_range), where=(np.array(system_range)*self.size)!=0)
        system_efficiency_curve[0] = 0
        
        # Plotting
        if plot == "plot" or plot == "Plot":
            plt.plot(np.array(system_range)*100,np.array(system_efficiency_curve)*100, label='System efficiency')
            plt.plot(np.array(system_range)*100,np.array(stack_efficiency_curve)*100, label='Stack efficiency')
            plt.ylabel('Efficiency [%]')
            plt.xlabel('Load range [%]')
            plt.legend()
            # plt.plot(stack_range,h2_prod)
        
        # Piece-wise linearization (y=k*x+m form)
        k_values = []
        m_values = []
        for i in range(pwl_points):
            k_values.append((h2_prod[i+1]-h2_prod[i])/(system_range[i+1]-system_range[i]))
            if i == 0:
                m_values.append((h2_prod[i]))
            else:
                m_values.append(h2_prod[i] - (system_range[i]*k_values[i]))
                
        self.k_values = k_values
        self.m_values = m_values
        self.n_sys_degr = system_efficiency_curve[-1]
        self.n_stack_degr = stack_efficiency_curve_degr[-1]
        
        self.size_degr = self.size * self.n_sys / self.n_sys_degr
        self.min_load = self.aux_cons / (self.size_degr)
        heat_gen = (self.size_degr - self.aux_cons) * (1-self.n_stack_degr)
        heat_h2o = ((self.h2_max* 1000 / 2.02) * (self.water_cons*997/(1000*18.02/2.02))) * 75.3 * (self.temp - self.h2o_temp) / (3600*1000)
        self.heat_max = heat_gen - heat_h2o
        
        
class Methanation():
    """
    
    """
    # Class parameters
    temp = 65 #C
    pres = 10 #bar
    start = 0 #minutes (cold start) NEED TO IMPLEMENT THIS ALSO OUTSIDE THE MILP
    min_load = 0 #minimum load
    n = 0.99 #CO2 conversion efficiency
    microb_cons = 0#0.06 #fraction of extra CO2 required for microbial growth
    standby_energy = 0#0.01 #fraction of rated electricity input required for stand-by
    el_cons = 0.5 #[kWh/Nm3 CH4 produced] In Schlautmann et al. 2020 for example.
    
    capex = 900 #[€/kWCH4] assumed at 5 MW
    capex_ref = 5000 #[kW]
    opex = 8 #% of CAPEX
    scaling = 0.65 #scaling factor for CAPEX
    
    ch4_hhv_vol = 11.05 #kWh/Nm3
    ch4_hhv_kg = 15.44 #kWh/kg
    ch4_hhv_mol = ch4_hhv_kg / (1000/16.04) #kWh/mol
    nm3_mol = ch4_hhv_mol / ch4_hhv_vol #Nm3/mol
    
    def __init__(self, size, co2_min):
        self.size = size * 1000
        self.size_mol = self.size / self.ch4_hhv_mol
        self.size_vector = np.zeros(24,) + self.size_mol
        self.flow_max = self.size_mol / co2_min
        self.flow_min = self.size_mol * self.min_load
        __, self.el_max, self.heat_max, __, __ = comps.methanation(meth_flow=[self.size_mol*4/(1-self.microb_cons),self.size_mol/(1-self.microb_cons),0], rated_flow=self.size_mol/(1-self.microb_cons), T=self.temp, T_in=self.temp, el_cons=self.el_cons)
        self.spec_heat = self.heat_max / (4*self.size_mol*2.02/1000)
        self.spec_el = self.el_max / (4*self.size_mol*2.02/1000) #[kWh/kgH2]
        
        
class Storage():
    """
    
    """
    #Class variables
    capex = 500 #€/kgH2 (is this CAPEX true for low-pressure storage?)
    opex = 1.5 #% of CAPEX
    
    def __init__(self):
        return
    

class Compressor():
    """
    
    """
    # Class variables
    n_isen = 0.75
    n_motor = 0.95
    
    capex_ref = 30000
    opex = 5
    scaling = 0.48
    
    N = 1 # Number of compressor stages 
    z = 1 # Assumption, should be a little higher depending in p and T
    k = 1.41 # Ratio of specific heat
    R = 8.314 # Gas constant
    
    def __init__(self, flow, p_out, p_in, temp_in):
        self.size = (self.N*(self.k/(self.k-1))*(self.z/self.n_isen)*(temp_in+273.15)*flow*self.R*(((p_out/p_in)**((self.k-1)/(self.N*self.k)))-1)) / (self.n_motor*1000)
        self.spec_el = self.size / (flow*3600)
        
    # def econ(self):
    #     self.capex = self.capex_ref*(self.size**self.scaling)
    #     bg_comp_opex = self.opex * 0.01 * bg_comp_capex
        
        
# def electrolyzer(
#         stack_efficiency, #at rated power
#         system_efficiency, #at rated power
#         pwl_points, #number of linearized segments
#         elz_size, #[MW]
#         # stack_rep, #[years/hours?]
#         degr_mode: str = "annual", #average/annual (meaning degradation value per FLHs or year)
#         degr_year: float = 5.0, #which year of operation to consider for degradation
#         elz_degr: float = 0.0, #per FLHs or [year]?
# ) -> pd.DataFrame:
#     """ Returns piece-wise linearization parameters for part load efficiency"""
    
#     #The following code was taken from Ginsberg et al. baseline efficiency
#     x = np.linspace(0, 6, num=60001)
#     #x = np.round(x, 1) #rounding needs to be 1
    
#     Fit_1 = 1.44926681  #C
#     Fit_2 = 2.71725684 #A
#     Fit_3 = 0.06970714 #K
#     Y = lambda X: (Fit_1 + Fit_2 * (1 - math.exp(-Fit_3 * X)))
#     Y_vector = np.vectorize(Y)
#     y_fit_baseline = Y_vector(x)  
#     #converting to efficiency
#     u_th = 1.481
#     eff_curve = u_th / y_fit_baseline
#     #find current density with the closest efficiency to what we are aiming for and corresponding efficiency
#     rated_eff = min(eff_curve, key=lambda x:abs(x-stack_efficiency))
#     rated_current_index = np.where(eff_curve == rated_eff)[0][0]
#     # rated_current_index = intervalIndex(rated_current_index,pwl_points) #want evenly sized intervals
#     # rated_current = x[rated_current_index] #[A/cm2]
#     rated_eff = eff_curve[rated_current_index]
#     #auxiliary consumption
#     # aux_cons = (rated_eff - system_efficiency) * elz_size * 1000 #[kW]
#     aux_cons = elz_size*1000 - (elz_size*1000*system_efficiency/stack_efficiency) #[kW]
#     #create efficiency curves
#     # interval_length = rated_current_index / pwl_points
#     stack_range = []
#     stack_efficiency_curve = []
#     system_efficiency_curve = []
#     h2_prod = []
#     # heat_prod = []
#     system_range = []
#     for i in range(pwl_points+1):
#         # system_range.append(interval_length*(i)/rated_current_index)
#         system_range.append(i/(pwl_points))
#         # stack_range.append(interval_length*(i)/rated_current_index)
#         stack_range.append(((system_range[i]*elz_size*1000)-aux_cons)/((elz_size*1000)-aux_cons))
#         stack_range[0] = 0.0
#         # stack_eff = min(eff_curve, key=lambda x:abs(x-stack_efficiency))
#         # current_index = np.where(eff_curve == stack_eff)[0][0]
#         stack_efficiency_curve.append(eff_curve[round(stack_range[i]*rated_current_index)])
#         # stack_efficiency_curve.append(eff_curve[int(interval_length*(i))])
#         # h2_prod.append(stack_range[i]*((elz_size*1000)-aux_cons)*stack_efficiency_curve[i]/39.4)
#         h2_prod.append(stack_range[i]*((elz_size*1000)-aux_cons)*stack_efficiency_curve[i]/39.4)
#         if i == 0:
#             system_efficiency_curve.append(0)
#         else:
#             system_efficiency_curve.append((h2_prod[i]*39.4)/(system_range[i]*(elz_size*1000)))
#         # heat_prod.append((1-stack_efficiency_curve[i]))

    
#     #degradation
#     if elz_degr > 0:
#         #only determining stack efficiency after half its lifetime (rounding up) to account for an "average" year
#         #Non-linear degradation
#         # degradation_factor = math.pow(1-elz_degr,degr_year)
#         #Assuming linear degradation (in %-points)
#         degradation_factor = elz_degr*degr_year/100
#         stack_efficiency_curve_degr = np.array(stack_efficiency_curve) - degradation_factor
#         elz_size_degr = ((((elz_size*1000)-aux_cons) * (stack_efficiency_curve[-1]/stack_efficiency_curve_degr[-1])) + aux_cons) / 1000
#         h2_prod1 = np.array(stack_range)*((elz_size*1000)-aux_cons)*stack_efficiency_curve/39.4
#         # h2_prod = np.array(h2_prod)
#         # system_efficiency_curve = (h2_prod*39.4)/(np.array(system_range)*(elz_size*1000)),
#         # system_efficiency_curve = np.divide((h2_prod1*39.4), (np.array(system_range)*(elz_size*1000)), out=np.zeros_like(system_range), where=(np.array(system_range)*(elz_size*1000))!=0)
#         system_efficiency_curve = np.divide((h2_prod1*39.4), (np.array(system_range)*(elz_size_degr*1000)), out=np.zeros_like(system_range), where=(np.array(system_range)*(elz_size*1000))!=0)
#         system_efficiency_curve[0] = 0
#         # heat_prod = 1 - stack_efficiency_curve
    
#     #plotting
#     # plt.plot(np.array(stack_range)*100,np.array(system_efficiency_curve)*100, label='System efficiency')
#     # plt.plot(np.array(stack_range)*100,np.array(stack_efficiency_curve)*100, label='Stack efficiency')
#     # plt.ylabel('Efficiency [%]')
#     # plt.xlabel('Load range [%]')
#     # plt.legend()
#     # plt.plot(stack_range,h2_prod)
    
#     #piece-wise linearization (y=k*x+m form)
#     k_values = []
#     m_values = []
#     for i in range(pwl_points):
#         k_values.append((h2_prod[i+1]-h2_prod[i])/(system_range[i+1]-system_range[i]))
#         if i == 0:
#             m_values.append((h2_prod[i]))
#         else:
#             m_values.append(h2_prod[i] - (system_range[i]*k_values[i]))
    
    
#     # def intervalIndex(a, b):
#     #     c = int(a / b)
#     #     a1 = b * c
#     #     if (a * b) > 0:
#     #         a2 = b * (c + 1)
#     #     else:
#     #         a2 = b * (c - 1)
     
#     #     if abs(a - a1) < abs(a - a2):
#     #         return a1
#     #     return a2
    
#     return k_values, m_values, aux_cons, system_efficiency_curve[-1], stack_efficiency_curve_degr[-1]


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


def compressor(
        flow, #mol/s
        temp_in: int = 80, #C
        p_in: int = 30, #bar
        p_out: int = 100, #bar
        n_isen: float = 0.7,
        n_motor: float = 0.95,):
    """
    Returns compressor size and electricity consumption.
    
    Parameters
    ----------
    flow: float [mol/s]
        Rated flow rate of gas through the compressor.
    temp_in: float [C]
        Rated inlet temperature of the gas.
    p_in: float [bar]
        Gas inlet pressure.
    p_out: float [bar]
        Gas outlet pressure.
    n_isen: float
        Isentropic efficiency.
    n_motor: float
        Motor efficiency
    
    Returns
    -------
    comp_size: float [kW]
        Rated compressor size based on flow rate.
    comp_spec_el: float [kWh/mol]
        Specific electricity consumption of the compressor.
    """

    return comp_size, comp_spec_el





