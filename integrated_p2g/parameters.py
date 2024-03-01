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
        
        if storage_type == 'H2' or storage_type == 'h2' or storage_type == 'Hydrogen' or storage_type == 'hydrogen':
            self.capex = 500
            self.opex = 1.5
            self.eff = 1
            self.size = size
        elif storage_type == 'Battery' or storage_type == 'battery' or storage_type == 'Bat' or storage_type == 'bat':
            self.capex = 300
            self.opex = 2
            self.eff = 1
            self.size = size * 1000 # [kWh]
        elif storage_type == 'O2' or storage_type == 'o2' or storage_type == 'Oxygen' or storage_type == 'oxygen':
            self.capex = 0
            self.opex = 0
            self.eff = 1
            self.size = size
        elif storage_type == 'Heat' or storage_type == 'heat' or storage_type == 'Thermal' or storage_type == 'thermal':
            self.capex = 0
            self.opex = 0
            self.eff = 1
            self.size = size * 1000 # [kWh]
    

class Compressor():
    """
    Contains all compressor related values.
    
    Parameters
    ----------
    flow: float 
        Rated flow rate of gas through the compressor [mol/s].
    p_in: float 
        Gas inlet pressure [bar].
    p_out: float 
        Gas outlet pressure [bar].
    temp_in: float 
        Rated inlet temperature of the gas [C].
    
    Class attributes
    ---------------
    n_isen: float
        Isentropic efficiency [fraction].
    n_motor: float
        Motor efficiency [fraction].
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
        Operational expenditure [% of CAPEX].
    scaling: float
        CAPEX scaling factor.
    
    Methods
    -------
    None
    
    Notes
    -----
    None
    
    """
    # Class variables
    n_isen = 0.75
    n_motor = 0.95
    N = 1 
    z = 1
    k = 1.41
    R = 8.314
    capex_ref = 30000
    opex = 5
    scaling = 0.48
    
    def __init__(self, flow, p_out, p_in, temp_in):
        """        
        Attributes
        ----------
        comp_size: float 
            Rated compressor size based on flow rate [kW].
        comp_spec_el: float 
            Specific electricity consumption of the compressor [kWh/mol].
            
        """
        self.size = (self.N*(self.k/(self.k-1))*(self.z/self.n_isen)*(temp_in+273.15)*flow*self.R*(((p_out/p_in)**((self.k-1)/(self.N*self.k)))-1)) / (self.n_motor*1000)
        self.spec_el = self.size / (flow*3600)


class Renewables():
    """
    Contains all wind and solar PV related values, as well as hourly generation profiles.
    
    Parameters
    ----------
    wind_size: float
        Rated capacity of wind power [kW].
    pv_size: float
        Rated capacity of solar power [kW].
    year: float
        Simulation year for determining generation array length.
    lifetime: float
        System lifetime for determining solar PV degradation [years].
    
    Class attributes
    ---------------
    wind_efs: float
        Carbon intensity of wind power [kgCO2/MWh].
    pv_efs: float
        Carbon intensity of wind power [kgCO2/MWh].
    pv_degr: float
        Annual PV panel degradation [%/year].
    wind_lcoe: float
        Production cost for wind power [€/MWh].
    pv_lcoe: float
        Production cost for solar PV [€/MWh].
    
    Methods
    -------
    None
    
    Notes
    -----
    None
    
    """
    wind_efs = 15
    pv_efs = 70
    pv_degr = 0.5
    wind_lcoe = 40
    pv_lcoe = 45
    
    def __init__(self, wind_size, pv_size, year, lifetime):
        """        
        Attributes
        ----------
        wind_gen: array (8760/8784x1)
            Hourly values of wind generation [kWh/h].
        pv_gen: array (8760/8784x1)
            Hourly values of PV generation [kWh/h].
            
        """
        self.wind_size = wind_size
        self.pv_size = pv_size
        
        wind_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\wind (Uppsala).xlsx' # Reading Excel data
        pv_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\solar (Uppsala).xlsx'
        if year == 2020:
            try:
                self.wind_gen = np.array(pd.read_excel(wind_read) * (wind_size/3000))[0:8784,0] # Saving data
            except FileNotFoundError:
                print("Error: Wind generation directory not found or file does not exist. Please define a generation profile in the Renewables class.")
            try:
                self.pv_gen = np.array(pd.read_excel(pv_read) * (pv_size/3000))[0:8784,0]
            except FileNotFoundError:
                print("Error: PV generation directory not found or file does not exist. Please define a generation profile in the Renewables class.")
        else:
            try:
                self.wind_gen = np.array(pd.read_excel(wind_read) * (wind_size/3000))[0:8760,0] # Removing last day
            except FileNotFoundError:
                print("Error: Wind generation directory not found or file does not exist. Please define a generation profile in the Renewables class.")
            try:
                self.pv_gen = np.array(pd.read_excel(pv_read) * (pv_size/3000))[0:8760,0]
            except FileNotFoundError:
                print("Error: PV generation directory not found or file does not exist. Please define a generation profile in the Renewables class.")
        
        self.pv_gen *= (1-(round(lifetime/2)*self.pv_degr/100)) # PV degradation
    

class Biogas():
    """
    Contains all biogas related values, as well as an hourly production profile.
    
    Parameters
    ----------
    data: str {'real', 'set'}
        Determines production dataset, either from 'real' data or using a 'set' constant profile.
    year: float
        Simulation year for determining generation array length.
    
    Class attributes
    ---------------
    pres: float
        Outlet pressure from anaerobic digestion [bar].
    temp: float
        Outlet temperature from anaerobic digestion [C].
    ef: float
        Carbon intensity of biogas [kgCO2/MWh].
    lcoe: float
        Production cost for biogas [€/MWh].
    
    Methods
    -------
    None
    
    Notes
    -----
    None
    
    """
    pres = 1
    temp = 50
    ef = 50
    lcoe = 65

    def __init__(self, data, year):
        """        
        Attributes
        ----------
        flow: array (8760/8784x2)
            Hourly methane and CO2 production [mol/h].
        min_co2: float
            Minimum CO2 fraction in the biogas flow [fraction].
            
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
            try:
                bg_data = pd.read_excel(bg_read)
            except FileNotFoundError:
                print("Error: Biogas production directory not found or file does not exist. Set data variable to 'set' to use an arrbitrary demand and modify it within the Biogas class.")
        
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
    Contains all heat related values, as well as hourly heat demand profiles.
    
    Parameters
    ----------
    data: str {'real', 'set'}
        Determines production dataset, either from 'real' data or using a 'set' constant profile.
    year: float
        Simulation year for determining generation array length.
    
    Class attributes
    ---------------
    usable: float
        Fraction of produced heat that can be utilized [fraction].
    ems: float
        Carbon intensity of replaced heat [kgCO2/MWh].
    ems_marginal: float
        Marginal carbon intensity of replaced heat [kgCO2/MWh].
    scale: float
        Scaling factor for real heat demand profile.
    set_demand_tot: float
        Value of set total heat demand at the WWTP [kW] (Edit this if no real demand profile is available).
    set_demand_aux: float
        Value of set auxiliary heat demand at the WWTP [kW] (Edit this if no real demand profile is available).
    capex : float
        Capital expenditure of heat integration system at reference size [€/kW].
    capex_ref : float
        Reference size for CAPEX [kWth].
    opex : float
        Operating expenditure [% of CAPEX].
    scaling : float
        CAPEX scaling factor.
    piping_capex: float
        Capital expenditure of heat piping [€/m].
    dh_price: array (4x1)
        District heating seasonal prices in the form [spring, summer, autumn, winter] [€/MWh].
    
    Methods
    -------
    None
    
    Notes
    -----
    None
    
    """
    usable = 0.8
    ems = 112
    ems_marginal = 30
    scale = 1
    set_demand_tot = 500
    set_demand_aux = 100
    capex = 260
    capex_ref = 400
    opex = 2
    scaling = 0.3
    piping_capex = 230
    dh_price = np.array([35,22,35,53])

    def __init__(self, data, year):
        """        
        Attributes
        ----------
        demand_tot: array (8760/8784x1)
            Hourly total heat demand [kW].
        demand_bg: array (8760/8784x1)
            Hourly sludge digestion heat demand [kW].
        demand_aux: array (8760/8784x1)
            Hourly auxiliary heat demand [kW].
            
        """
        if data == 'real' or data == 'Real':
            heat_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\Heat demand.xlsx'
            try:
                total_heat = pd.read_excel(heat_read).iloc[:,0] * self.scale
                digester_heat = pd.read_excel(heat_read).iloc[:,1] * self.scale
                aux_heat = pd.read_excel(heat_read).iloc[:,2] * self.scale
            except FileNotFoundError:
                print("Error: Heat demand directory not found or file does not exist. Set data variable to 'set' to use an arrbitrary demand and modify it within the Heat class.")
            if year == 2020:
                total_heat = pd.concat([total_heat,total_heat.iloc[-24:]])
                digester_heat = pd.concat([digester_heat,digester_heat.iloc[-24:]])
                aux_heat = pd.concat([aux_heat,aux_heat.iloc[-24:]])
            self.demand_tot = np.array(total_heat)
            self.demand_bg = np.array(digester_heat)
            self.demand_aux = np.array(aux_heat)
        elif data == 'set' or data == 'Set':
            if year == 2020:
                self.demand_tot = np.zeros(8784,) + self.set_demand_tot
                self.demand_aux = np.zeros(8784,) + self.set_demand_aux
                self.demand_bg = np.zeros(8784,) + self.set_demand_tot - self.demand_aux
            else:
                self.demand_tot = np.zeros(8760,) + self.set_demand_tot
                self.demand_aux = np.zeros(8760,) + self.set_demand_aux
                self.demand_bg = np.zeros(8760,) + self.set_demand_tot - self.demand_aux
        

class Oxygen():
    """
    Contains all heat related values, as well as hourly heat demand profiles.
    
    Parameters
    ----------
    data: str {'real', 'set'}
        Determines production dataset, either from 'real' data or using a 'set' constant profile.
    year: float
        Simulation year for determining generation array length.
    
    Class attributes
    ---------------
    replacement: float
        Reduction in flow rate for the aeration process.
    sote_increase: float
        Increase in oxygen transfer rate from pure oxygen use. 10 % increase is written 1.1.
    aerator_air: float
        Electricity consumption of conventional aeration [kWh/kgO2].
    aerator_o2: float
        Electricity consumption of pure oxygen aeration[kWh/kgO2].
    aerator_savings: float
        Electricity savings from pure oxygen use [kWh/kgO2].
    scale: float
        Scaling factor for real oxygen demand profile.
    set_demand: float
        Value of set oxygen demand at the WWTP [mol/h] (Edit this if no real demand profile is available).
    aerator_capex : float
        Capital expenditure of the pure oxygen aeration system at reference size [€/kW].
    aerator_ref : float
        Reference size for CAPEX [kW of electrolyzer capacity].
    opex : float
        Operating expenditure [% of CAPEX].
    aerator_scaling : float
        CAPEX scaling factor.
    piping_capex: float
        Capital expenditure of oxygen piping [€/m].
    
    Methods
    -------
    None
    
    Notes
    -----
    None
    
    """
    # Class variables
    replacement = 100/21 
    sote_increase = 1
    aerator_air = 1/17
    aerator_o2 = aerator_air/(replacement)
    aerator_savings = (aerator_air - aerator_o2) * sote_increase
    scale = 1
    set_demand = 100000
    aerator_capex = 70
    aerator_ref = 1250
    opex = 2 
    aerator_scaling = 0.6
    piping_capex = 540
    
    def __init__(self, data, year):
        """        
        Attributes
        ----------
        demand: array (8760/8784x1)
            Hourly oxygen demand [mol/h].
            
        """
        if data == 'real' or data == 'Real':
            o2_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\O2 flow.xlsx'
            try:
                o2_data = pd.read_excel(o2_read).iloc[:,0] * self.scale
            except FileNotFoundError:
                print("Error: Oxygen demand directory not found or file does not exist. Set data variable to 'set' to use an arrbitrary demand and modify it within the Oxygen class.")
            if year == 2020:
                o2_data = pd.concat([o2_data,o2_data.iloc[-24:]])
            self.demand = np.array(o2_data)
        elif data == 'set' or data == 'Set':
            if year == 2020:
                self.demand = np.zeros(8784,) + self.set_demand


class TechnoEconomics():
    """
    Contains all values related to techno-economic assessment, as well as KPI determination methods.
    
    Parameters
    ----------
    hv: str {'hhv', 'lhv'}
        Type of heating value used for techno-economic calculations.
    lifetime: float
        System lifetime [years].
    discount: float
        Discount rate [%].
    
    Class attributes
    ---------------
    co2_cost: float
        Cost of carbon dioxide [€/tCO2].
    install_cost: float
        Installation cost [% of total CAPEX].
    piping_dist: float
        Piping distance for heat and oxygen integration costs [m].
    
    Methods
    -------
    lcox(opex, capex, stack, prod, stack_reps, rep_years)
        Returns the levelized cost of X, where X depends on the input data.
    npv(self, opex, income, capex, stack, stack_reps, rep_years)
        Returns the net present value.
    emissions(self, aefs, mefs, wind_efs, pv_efs, grid, wind, pv, prod, heat_use, heat_aef, heat_mef, o2_use, bg_ef, flared)
        Returns net specific emissions using boh average and marginal emission factors.
    
    Notes
    -----
    LHV not fully implemented.
    
    """
    co2_cost = 0
    install_cost = 20
    piping_dist = 1000
    
    def __init__(self, hv, lifetime, discount):
        """        
        Attributes
        ----------
        ch4_mol: float
            HHV for methane on a molar basis [kWh/mol].
        h2_kg: float
            HHV for hydrogen on a mass basis [kWh/kg].
        nm3_mol: float
            Amount of normal cubic meters per mol of a gas [Nm3/mol].
            
        """
        self.lifetime = lifetime
        self.discount = discount
        
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
    
    def lcox(self, opex, capex, stack, prod, stack_reps, rep_years):
        """
        Returns the levelized cost of X.
        
        Parameters
        ----------
        opex: float
            Total OPEX [€].
        capex: float
            Total CAPEX [€].
        stack: float
            Stack replacement cost [€].
        prod: str
            Amount of produced gas [MWh]
        stack_reps: float
            Number of stack replacements.
        rep_years: array
            The years when stack replacements take place.
        
        Returns
        ---------------
        lcox : float
            Levelized cost of X [€/MWh].
            
        """
        #Define total parameters
        full_opex = 0
        full_ch4 = 0
        full_stack = 0
        if stack_reps > 0: #Define stack replacement parameters
            stack_cost = stack / stack_reps
        else:
            stack_cost = 0
        for y in range(self.lifetime): #Total OPEX and CH4 production
            full_opex = full_opex + (opex / pow(1 + (self.discount/100),y))
            full_ch4 = full_ch4 + (prod / pow(1 + (self.discount/100),y))
        for i in range(stack_reps): #Discounting stack replacements
            full_stack = full_stack + (stack_cost / pow(1 + (self.discount/100),rep_years[i]))

        lcox = (capex + full_opex + full_stack) / full_ch4
        return lcox
    
    def npv(self, opex, income, capex, stack, stack_reps, rep_years):
        """
        Returns the net present value.
        
        Parameters
        ----------
        opex: float
            Total OPEX [€].
        income : float
            Total annual income [€].
        capex: float
            Total CAPEX [€].
        stack: float
            Stack replacement cost [€].
        stack_reps: float
            Number of stack replacements.
        rep_years: array
            The years when stack replacements take place.
        
        Returns
        ---------------
        npv : float
            Net present value [€].
            
        """
        annual_flow = 0
        total_stack = 0
        if stack_reps > 0:
            stack_cost = stack / stack_reps
        else:
            stack_cost = 0
        for y in range(self.lifetime):
            annual_flow = annual_flow + ((income-opex) / pow(1 + (self.discount/100),y))
            
        for i in range(stack_reps):
            total_stack = total_stack + (stack_cost / pow(1 + (self.discount/100),rep_years[i]))
            
        npv = annual_flow - capex - total_stack
        return npv
    
    def efficiencies(self, p2g_prod, tot_prod, heat_use, o2_use, el_cons, tot_heat, usable_heat):
        """
        Returns multiple efficiency values.

        Parameters
        ----------
        p2g_prod : float
            Methane production from the P2G system [MWh].
        tot_prod : float
            Total methane production [MWh].
        heat_use : float
            Total heat use [kWh].
        o2_use : array
            Hourly oxygen use [mol/h].
        el_cons : float
            Total electricity consumption of the P2G system [kWh].
        tot_heat : float
            Total heat production [kWh]
        usable_heat : float
            Usable heat fraction.

        Returns
        -------
        aef_net : float
            Net specific emissions based on average emission factors [kgCO2/MWhCH4].
        tot_prod : float
            Net specific emissions based on marginal emission factors [kgCO2/MWhCH4].
        aef_avg : float
            Average AEF used [kgCO2/MWhel].
        o2_use : float
            Average MEF used [kgCO2/MWhel].

        """
        n_gas = 100 * (p2g_prod * 1000) / el_cons # Efficiency of P2G system without by-products [%]
        n_heat = 100 * ((p2g_prod * 1000) + heat_use) / el_cons # Including heat [%]
        n_o2 = 100 * ((p2g_prod * 1000) + o2_use) / el_cons # Including oxygen [%]
        n_tot = 100 * ((p2g_prod * 1000) + heat_use + o2_use) / el_cons # Including heat and oxygen [%]
        n_biomethane = 100 * ((tot_prod * 1000) + heat_use + o2_use) / (el_cons + ((tot_prod-p2g_prod)*1000)) # Upgrading efficiency, including biomethane [%]
        n_max = 100 * ((p2g_prod * 1000) + (tot_heat*usable_heat) + o2_use) / el_cons # Theoretical with all usable heat and oxygen [%]
        n_theory = 100 * ((p2g_prod * 1000) + tot_heat + o2_use) / el_cons # Theoretical with all heat and oxygen [%]
        
        return n_gas, n_heat, n_o2, n_tot, n_biomethane, n_max, n_theory
    
    def emissions(self, aefs, mefs, wind_efs, pv_efs, grid, wind, pv, prod, heat_use, heat_aef, heat_mef, o2_use, bg_ef, flared):
        
        aef_ems = ((grid * aefs / 1000).sum() + (wind * wind_efs / 1000).sum() + (pv * pv_efs / 1000).sum()) / prod # Average emissions from electricity consumption without curtailment [kgCO2/MWhCH4]
        mef_ems = ((grid * mefs / 1000).sum() + (wind * wind_efs / 1000).sum() + (pv * pv_efs / 1000).sum()) / prod # Marginal emissions from electricity consumption without curtailment [kgCO2/MWhCH4]
        aef_avg = (aef_ems*prod) / ((grid / 1000).sum()) # Average AEF emissions from electricity consumption [kgCO2/MWhCH4]
        mef_avg = (mef_ems*prod) / ((grid / 1000).sum()) # Average MEF emissions from electricity consumption [kgCO2/MWhCH4]
        aef_ems_red_heat = (heat_use * heat_aef / 1000) / prod # Average mission reductions from heat use [kgCO2/MWhCH4]
        mef_ems_red_heat = (heat_use * heat_mef/ 1000) / prod # Marginal mission reductions from heat use [kgCO2/MWhCH4]
        # aef_ems_red_heat = ((heat_prod.sum() * heat_frac_use) * heat.ems / 1000) / ch4_p2g # [kgCO2/MWhCH4] (For assuming a fixed heat use fraction)
        # mef_ems_red_heat = ((heat_prod.sum() * heat_frac_use) * heat.ems_marginal / 1000) / ch4_p2g # [kgCO2/MWhCH4] (For assuming a fixed heat use fraction)
        aef_red_o2 = (o2_use * aefs).sum() / (1000*prod) # Average mission reductions from oxygen use [kgCO2/MWhCH4]
        mef_red_o2 = (o2_use * mefs).sum() / (1000*prod) # Marginal mission reductions from oxygen use [kgCO2/MWhCH4]
        bgloss_ems_increase = (bg_ef * flared * self.ch4_mol) / (1000*prod) # Emission increase from biogas losses [kgCO2/MWhCH4]
        aef_net = aef_ems - aef_red_o2 - aef_ems_red_heat + bgloss_ems_increase # Net system climate impact [kgCO2/MWhCH4]
        mef_net = mef_ems - mef_red_o2 - mef_ems_red_heat + bgloss_ems_increase # Net system climate impact [kgCO2/MWhCH4]
        
        return aef_net, mef_net, aef_avg, mef_avg
        
class Grid():
    """  
    Returns numpy arrays containing annual hourly electricity grid prices, and average and marginal hourly emission factors.
    
    Parameters
    ----------
    year: float
        Simulation year for determining generation array length.
    zone: str {'SE1', 'SE2', 'SE3', 'SE4'}
        Determines which Swedish electricity bidding zone prices and emissions to use.
    
    Class attributes
    ---------------
    fee: float
        Grid fee which is added to the spot price [€/MWh].
    
    Methods
    -------
    None
    
    Notes
    -----
    None
    
    """
    fee = 10 #€/MWh
    
    def __init__(self, year, zone):
        """
        Attributes
        ----------
        spot_price : array (8760/8784x1)
            Hourly spot prices in the specified bidding zone and year [€/MWh].
        aefs : array (8760/8784x1)
            Hourly average emission factors in the specified bidding zone and year [kgCO2/MWh].
        mefs : array (8760/8784x1)
            Hourly marginal emission factors in the specified bidding zone and year [kgCO2/MWh].

        """
        spot_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\Spot prices\elspot prices ' + str(year) + '.xlsx'
        spot_price = pd.read_excel(spot_read) + self.fee
        self.spot_price = np.array(spot_price[zone].tolist()) # Grid prices
        
        efs_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\EFs\efs_' + str(zone) + '_' + str(year) + '.xlsx'
        efs = pd.read_excel(efs_read)
        self.aefs = np.array(efs.iloc[:,0])
        self.mefs = np.array(efs.iloc[:,1]) 
    
