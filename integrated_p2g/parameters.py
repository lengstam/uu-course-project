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
Classes containing technical parameters for each component as well as a single class for economic data.
"""

class Electrolyzer():
    """
    Contains all electrolyzer related values.
    
    Parameters
    ----------
    size : float
        Defines the size of the current instance of Electrolyzer [MW].
    
    Attributes
    ----------
    n_sys : float
        Electrolyzer system efficiency, including auxiliary consumption. HHV basis [fraction].
    n_stack : float
        Electrolyzer stack efficiency [fraction]. HHV basis.
    start_time : float
        Cold start-up time [min].
    start_cost : float
        Cold start-up cost [fraction of rated power].
    standby_cost : float
        Standby energy consumption [fraction of rated power].
    heatup_time : float
        Time during which no usable waste heat is released after a cold start [min].
    temp : float
        Operating temperature [C].
    h2o_temp : float
        Input water temperature [C].
    pres : float
        Operating pressure [bar].
    degr : float
        Stack degradation rate [%/year].
    stack_rep : float
        Stack replacement time [years].
    degr_year : float
        Year during which average degradation has been reached [years].
    water_cons : float
        Water consumption [liter/kgH2].
    capex : float
        Capital expenditure at reference size [€/kW].
    capex_ref : float
        Reference size for CAPEX [MW].
    opex : float
        Operating expenditure [% of CAPEX].
    scaling : float
        CAPEX scaling factor.
    water_cost : float
        Water cost [€/m3].
    stack_cost : float
        Stack replacement cost [% of CAPEX].
        
    Methods
    -------
    efficiency(plot)
        Calculates the post-degradation characteristics of the class instance and
        provides linearization parameters for the efficiency curve.
        
    Notes
    -----
    None
    """
    # Class attributes 
    n_sys = 0.75
    n_stack = n_sys + 0.05
    start_time = 5
    start_cost = start_time/60
    standby_cost = 0.02
    heatup_time = 60
    temp = 80
    h2o_temp = 15
    pres = 30
    degr = 1
    stack_rep = 10
    degr_year = round(stack_rep/2)
    water_cons = 10 #[l/kgH2]
    capex = 1500 #[€/kW] at 5 MW
    capex_ref = 5000 #[kW]
    opex = 4 #% of CAPEX
    scaling = 0.75 #scaling factor for CAPEX
    water_cost = 0.5 #€/m3 (including purification) Se förstudie för värde
    stack_cost = 0.5 #fraction of CAPEX
    
    def __init__(self, size):
        """
        Attributes
        ----------
        size : float
            The size of the current instance of Electrolyzer [kW].
        h2_max : float
            Hydrogen production at rated capacity [kg/h].
        standby_el : float
            Electricity consumption of electrolyzer during standby [kW].
        """
        self.size = size * 1000 # [kW]
        self.h2_max = self.size * self.n_sys / 39.4
        self.standby_el = self.size * self.standby_cost
    
    def efficiency(self, plot):
        """
        Returns piece-wise linearization parameters for part load efficiency. Must be called after defSize()
        The code was adapted from Ginsberg et al (LINK). Using baseline efficiency.
        
        Parameters
        ----------
        plot: str {'plot', ''}
            If 'plot', the linearized efficiency curves will be plotted.
        
        Attributes
        ---------------
        aux_cons : float
            Auxiliary electricity consumption [kW].
        k_values : float
            Linear term of linearization parameters.
        m_values : float
            Constant term of linearization parameters.
        n_sys_degr : float
            System efficiency after degradation [fraction].
        n_stack_degr : float
            Stack efficiency after degradation [fraction].
        size_degr : float
            Actual electrolyzer input capacity after degradation. [kW]
        min_load : float
            Minimum electrolyzer load [fraction].
        heat_max : float
            Maximum electrolyzer heat generation [kW].
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
        if plot == 'plot' or plot == 'Plot':
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
    Contains all methanation related values.
    
    Parameters
    ----------
    size : float
        Defines the size of the current instance of Electrolyzer [MW].
    co2_min : float
        Minimum CO2 fraction of the biogas flow [fraction].
    
    Attributes
    ----------
    temp : float
        Operating temperature [C].
    pres : float
        Operating pressure [bar].
    start : float (Not implemented)
        Cold start-up time [min].
    min_load : float
        Minimum load [fraction of rated power].
    n : float
        CO2 conversion efficiency [fraction].
    microb_cons : float
        Microbial CO2 consumption [fraction].
    standby_energy : float
        Standby energy consumption [fraction of rated electricity consumption].
    el_cons : float
        Standby energy consumption [kWh/Nm3CH4 converted].
    ch4_hhv_vol : float
        Methane HHV energy content [kWh/m3].
    ch4_hhv_kg : float
        Methane HHV energy content [kWh/kg].
    ch4_hhv_mol : float
        Methane HHV energy content [kWh/mol].
    nm3_mol : float
        Mol to volume conversion [Nm3/mol].
    capex : float
        Capital expenditure at reference size [€/kW].
    capex_ref : float
        Reference size for CAPEX [MW].
    opex : float
        Operating expenditure [% of CAPEX].
    scaling : float
        CAPEX scaling factor.
        
    Methods
    -------
    None
        
    Notes
    -----
    None
    """
    # Class parameters
    temp = 65
    pres = 10
    # start = 0
    min_load = 0
    n = 0.99
    microb_cons = 0
    standby_energy = 0
    el_cons = 0.5
    ch4_hhv_vol = 11.05
    ch4_hhv_kg = 15.44
    ch4_hhv_mol = ch4_hhv_kg / (1000/16.04)
    nm3_mol = ch4_hhv_mol / ch4_hhv_vol
    capex = 900
    capex_ref = 5000
    opex = 8
    scaling = 0.65
    
    def __init__(self, size, co2_min):
        """
        Attributes
        ----------
        size : float
            The rated capacity of the current instance of Methanation [kWCH4out].
        size_mol : float
            The rated capacity of the current instance of Methanation [mol/h].
        size_vector : float
            Array of rated capacity.
        flow_min : float
            Minimum CO2 flow rate through reactor [mol/h].
        heat_max : float
            Maximum heat generation from reactor [kW].
        spec_heat : float
            Heat generation per unit gas [kWh/kgH2].
        spec_el : float
            Electricity consumption per unit gas [kWh/kgH2].
        """
        self.size = size * 1000
        self.size_mol = self.size / self.ch4_hhv_mol
        self.size_vector = np.zeros(24,) + self.size_mol
        self.flow_max = self.size_mol / co2_min
        self.flow_min = self.size_mol * self.min_load
        __, el_max, self.heat_max, __, __ = comps.methanation(meth_flow=[self.size_mol*4/(1-self.microb_cons),self.size_mol/(1-self.microb_cons),0], T=self.temp, T_in=self.temp, el_cons=self.el_cons)
        self.spec_heat = self.heat_max / (4*self.size_mol*2.02/1000)
        self.spec_el = el_max / (4*self.size_mol*2.02/1000) #[kWh/kgH2]
        
        
class Storage():
    """
    Contains all storage related values.
    
    Parameters
    ----------
    size : float
        Defines the size of the current instance of storage [kgH2/MWh/kgO2/MWh].
    
    Attributes
    ----------
    None
        
    Methods
    -------
    None
    
    Notes
    -----
    Oxygen and heat storages not implemented.
    """      
    def __init__(self, storage_type, size):
        """        
        Attributes
        ----------
        size : float
            Defines the size of the current instance of storage [kgH2/MWh/kgO2/MWh].
        capex : float
            Capital expenditure [€/sizeunit].
        opex : float
            Operational expenditure [% of CAPEX].
        eff : float
            Storage round efficiency [fraction].
        """
        self.size = size
        
        if storage_type == 'H2' or storage_type == 'h2' or storage_type == 'Hydrogen' or storage_type == 'hydrogen':
            self.capex = 500
            self.opex = 1.5
            self.eff = 1
        elif storage_type == 'Battery' or storage_type == 'battery' or storage_type == 'Bat' or storage_type == 'bat':
            self.capex = 300
            self.opex = 2
            self.eff = 1
        elif storage_type == 'O2' or storage_type == 'o2' or storage_type == 'Oxygen' or storage_type == 'oxygen':
            self.capex = 0
            self.opex = 0
            self.eff = 1
        elif storage_type == 'Heat' or storage_type == 'heat' or storage_type == 'Thermal' or storage_type == 'thermal':
            self.capex = 0
            self.opex = 0
            self.eff = 1
    

class Compressor():
    """
    Contains all compressor related values.
    
    Parameters
    ----------
    flow: float [mol/s]
        Rated flow rate of gas through the compressor.
    p_in: float [bar]
        Gas inlet pressure.
    p_out: float [bar]
        Gas outlet pressure.
    temp_in: float [C]
        Rated inlet temperature of the gas.
    
    Class attributes
    ---------------
    n_isen: float
        Isentropic efficiency.
    n_motor: float
        Motor efficiency
    N: float
        Number of compressor stages.
    z: float
        Compressibility factor.
    k: float
        Ratio of specific heats.
    R: float
        Ideal gas constant.
    capex_ref: float
        Capital expenditure at 1 kW [€/kW].
    opex: float
        Operational expenditure [% of CAPEX]
    scaling: float
        CAPEX scaling factor.
    
    Returns
    -------
    comp_size: float [kW]
        Rated compressor size based on flow rate.
    comp_spec_el: float [kWh/mol]
        Specific electricity consumption of the compressor.
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


class Renewables():
    """
    
    """
    # Class variables
    wind_efs = 15 #[kgCO2/MWh]
    pv_efs = 70 #[kgCO2/MWh]
    pv_degr = 0.5 #[%/y]
    
    wind_lcoe = 40 # [€/MWh] assumed PPA price
    pv_lcoe = 45 # [€/MWh]
    
    def __init__(self, wind_size, pv_size, year, lifetime):
        self.wind_size = wind_size
        self.pv_size = pv_size
        
        wind_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\wind (Uppsala).xlsx' # Reading Excel data
        pv_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\solar (Uppsala).xlsx'
        if year == 2020:
            self.wind_gen = np.array(pd.read_excel(wind_read) * (wind_size/3000))[0:8784,0] # Saving data
            self.pv_gen = np.array(pd.read_excel(pv_read) * (pv_size/3000))[0:8784,0] 
        else:
            self.wind_gen = np.array(pd.read_excel(wind_read) * (wind_size/3000))[0:8760,0] # Removing last day
            self.pv_gen = np.array(pd.read_excel(pv_read) * (pv_size/3000))[0:8760,0]
        self.pv_gen *= (1-(round(lifetime/2)*self.pv_degr/100)) # PV degradation
    

class Biogas():
    """
    
    """
    # Class variables
    pres = 1 #bar
    temp = 50 #C
    ef = 50 #[gCO2/kWh]
    lcoe = 65 #[€/MWh raw biogas]

    def __init__(self, data, year):
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
        
        self.flow = np.array([ch4_rate,co2_rate]).transpose()
        self.min_co2 = np.min(np.divide(self.flow[:,1], self.flow[:,1]+self.flow[:,0], out=np.zeros_like(self.flow[:,0])+1, where=self.flow[:,1]+self.flow[:,0]!=0)) #[mol/h] maximum theoretical flow rate to methanation

class Heat():
    """
    
    """
    # Class variables
    usable = 0.8 #heat exchanger efficiency (van der Roest et al. 2023)
    ems = 112 #[gCO2/kWh]
    ems_marginal = 30 #[gCO2/kWh]
    scale = 1
    
    capex = 260 #[€/kWth]
    opex = 2
    capex_ref = 400 #[kWth]
    scaling = 0.3
    piping_capex = 230 #[€/m]
    dh_price = np.array([35,22,35,53]) # [spring, summer, autumn, winter]

    def __init__(self, year):
        #Read and process data
        heat_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\Heat demand.xlsx'
        total_heat = pd.read_excel(heat_read).iloc[:,0] * self.scale
        digester_heat = pd.read_excel(heat_read).iloc[:,1] * self.scale
        aux_heat = pd.read_excel(heat_read).iloc[:,2] * self.scale
        if year == 2020:
            total_heat = pd.concat([total_heat,total_heat.iloc[-24:]])
            digester_heat = pd.concat([digester_heat,digester_heat.iloc[-24:]])
            aux_heat = pd.concat([aux_heat,aux_heat.iloc[-24:]])
        
        self.demand_tot = np.array(total_heat)
        self.demand_bg = np.array(digester_heat)
        self.demand_aux = np.array(aux_heat)
        return
        

class Oxygen():
    """
    
    """
    # Class variables
    replacement = 100/21 #how much can we reduce the aeration flow by?
    sote_increase = 1 #reduced o2 demand due to higher driving force, need to implement properly for costs if used as we're still replacing the same amount of air!
    aerator_air = 1/17 #[kWh/kgO2]
    aerator_o2 = aerator_air/(replacement) #[kWh/kgO2]
    aerator_savings = (aerator_air - aerator_o2) * sote_increase #[kWh/kgO2] energy savings per kg pure O2
    scale = 1 # Scale of demand
    
    piping_capex = 540 #[€/m]
    aerator_capex = 70 #[€/kW electrolyzer]
    aerator_ref = 1250 #[MWel]
    opex = 2 
    aerator_scaling = 0.6
    
    def __init__(self, year):
        #Read and process data
        o2_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\O2 flow.xlsx'
        o2_data = pd.read_excel(o2_read).iloc[:,0] * self.scale
        if year == 2020:
            o2_data = pd.concat([o2_data,o2_data.iloc[-24:]])
            
        self.demand = np.array(o2_data)


class TechnoEconomics():
    """
    
    
    LHV not fully implemented.
    
    
    """
    # Class variables
    lifetime = 20 #years
    discount = 8 #[%]
    co2_cost = 0 #[€/tonCO2]
    install_cost = 20 #[% of total CAPEX]
    piping_dist = 1000 #[m]
    
    def __init__(self, hv):
        if hv == "HHV" or hv == "hhv":
            ch4_vol = 11.05 #kWh/Nm3
            ch4_kg = 15.44 #kWh/kg
            self.ch4_mol = ch4_kg / (1000/16.04) #kWh/mol
            self.h2_kg = 39.4 #kWh/kg
            self.nm3_mol = self.ch4_mol / ch4_vol #Nm3/mol
        elif hv == "LHV" or hv == "lhv":
            ch4_vol = 9.94 #kWh/Nm3
            ch4_kg = 13.9 #kWh/kg
            self.ch4_mol = ch4_kg / (1000/16.04) #kWh/mol
            self.h2_kg = 33.3 #kWh/kg
            self.nm3_mol = self.ch4_mol / ch4_vol #Nm3/mol
        
        
class Grid():
    """
    
    
    Returns numpy arrays containing annual hourly electricity grid prices, and average and marginal hourly emission factors.
    
    
    """
    # Class variables
    fee = 10 #€/MWh (only PV on-site?)
    
    def __init__(self, year, zone):
        spot_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\Spot prices\elspot prices ' + str(year) + '.xlsx'
        spot_price = pd.read_excel(spot_read) + self.fee
        self.spot_price = np.array(spot_price[zone].tolist()) # Grid prices
        
        efs_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\EFs\efs_' + str(zone) + '_' + str(year) + '.xlsx'
        efs = pd.read_excel(efs_read)
        self.aefs = np.array(efs.iloc[:,0])
        self.mefs = np.array(efs.iloc[:,1]) 
    
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


def res_gen(
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
    wind_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\wind (Uppsala).xlsx' # Reading Excel data
    pv_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\solar (Uppsala).xlsx'
    if year == 2020:
        wind_gen = np.array(pd.read_excel(wind_read) * (wind_size/3000))[0:8784,0] # Saving data
        pv_gen = np.array(pd.read_excel(pv_read) * (pv_size/3000))[0:8784,0] 
    else:
        wind_gen = np.array(pd.read_excel(wind_read) * (wind_size/3000))[0:8760,0] # Removing last day
        pv_gen = np.array(pd.read_excel(pv_read) * (pv_size/3000))[0:8760,0]
    pv_gen = pv_gen * (1-(round(lifetime/2)*pv_degr/100)) # PV degradation
    
    return wind_gen, pv_gen


def grid(
        year: int = 2021,
        zone: str = "SE3",
        grid_fee: float = 10.0):
    """
    Returns numpy arrays containing annual hourly electricity grid prices, and average and marginal hourly emission factors.
    
    """
    spot_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\Spot prices\elspot prices ' + str(year) + '.xlsx'
    spot_price = pd.read_excel(spot_read) + grid_fee
    spot_price = np.array(spot_price[zone].tolist()) # Grid prices
    
    efs_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\EFs\efs_' + str(zone) + '_' + str(year) + '.xlsx'
    efs = pd.read_excel(efs_read)
    aefs = np.array(efs.iloc[:,0])
    mefs = np.array(efs.iloc[:,1])  
    
    return spot_price, aefs, mefs
    
    
    
    
    
    
