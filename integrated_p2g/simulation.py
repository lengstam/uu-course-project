# -*- coding: utf-8 -*-
"""
Created on Thu Nov 2 11:41:17 2023

@author: Linus Engstam
"""

import numpy as np
import pandas as pd
import math
import parameters as params
import components as comps
import other
import dispatch
import plotting
import matplotlib.pyplot as plt
from matplotlib import rc
from tabulate import tabulate
import matplotlib.patches as mpatches
import matplotlib.lines as lines
import plotly.graph_objects as go
import urllib, json
import seaborn as sns
import time

"""
This is a model of an integrated power-to-gas (P2G) system, as described in <link to study when published>.

This is the script from which simulations of the integrated P2G model can be conducted.
The system consists of a PEM electrolyzer, compressed hydrogen storage and a methanation reactor.
Capabilities for including a battery for storage of excess renweable generation has also been implemented.
Electricity can be sourced from hte grid and directly coupled wind and solar energy.

To run a simulation, define electrolyzer [MWel], storage [kgH2], methanation [MWCH4out] and battery [MWh] capacities in a list format.
Include renewable generation by defining wind and PV oversizing factors (ratio to electrolyzer capacity, e.g. 2 for 10 MW with a 5 MW electrolyzer).
To run an optimization, simply define multiple capacities for the components you wish to optimize in the list, separated by a comma (e.g. [1,2,3])
Furthermore, define the simulation year (2018-2021) and the bidding zone (SE1 - SE4).
Detailed process data can be saved be setting 'simulation_detals' to 'yes', and plots can be produced be setting 'plots' to 'yes'.

Further changes to the input parameters can be made within the classes containing each component within the 'parameters.py' module.

"""


"""
Comments on current version (1.0a):

H2 storage not discharging at the end of each error, perhaps the reason behind issues?

"""

# start = time.time()

""" Optimization parameters """
elz_size_vector = [8.5]#[6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12] # [MW]
meth_scale_vector = [5]#[3.5,4,4.5,5] # [MWCH4]
h2st_size_vector = [300]#[0,100,200,300,400,500,600,700,800,900,1000] # [kg]
wind_size_vector = [1.25] # [ratio to elz]
pv_size_vector = [1.25] # [ratio to elz]
bat_size_vector = [0] # [MWh]

""" Simulation parameters """
year = 2021 # 2018-2021 available
bidding_zone = 'SE3' # ['SE1', 'SE2', 'SE3', 'SE4']
simulation_details = 'no' # ['Yes', 'No'] A "Process" dataframe with all process stages during all hours is created.
plots = 'no' # ['Yes', 'No']

""" Creating component classes for variable storage in components """
tec = params.TechnoEconomics(hv='hhv', lifetime=20, discount=8) # Techno-economic parameters
biogas = params.Biogas(data='real', year=year) # Biogas production
storage = params.Storage(storage_type='H2', size=0) # Hydrogen storage. Setting arbitrary sizes to be adjusted later within the optimization loop.
bat = params.Storage(storage_type='Battery', size=0) # Battery storage. Not included in study.
res = params.Renewables(wind_size=3000, pv_size=3000, year=year, lifetime=tec.lifetime) # Renewables (wind, PV). Assuming arbitrary sizes to be adjusted later within the optimization loop.
grid = params.Grid(year=year, zone=bidding_zone) # Electricity grid parameters
o2 = params.Oxygen(year=year, data='real') # Oxygen utilization system
heat = params.Heat(year=year, data='real') # Heat utilization system

""" Define run-type """
if len(elz_size_vector) == 1 and len(meth_scale_vector) == 1 and len(h2st_size_vector) == 1 and len(wind_size_vector) == 1 and len(pv_size_vector) == 1 and len(bat_size_vector) == 1:
    run_type = 'single'
    cost_breakdown = pd.DataFrame({'Costs': ['Electrolyser','Stack','Water','Storage','Meth','Comp','Heat','O2','Installation','Flaring','Grid','PV','Wind','Curtailment','O2 income','Heat income','Total']}).set_index('Costs')
    
    if simulation_details == 'Yes' or simulation_details == 'yes':
        process = other.data_saving(year=year) # Initiate process data saving
        process['Elspot [€/MWh]'] = grid.spot_price
        process['Biogas (CH4) [mol/h]'] = biogas.flow[:,0]
        process['Biogas (CO2) [mol/h]'] = biogas.flow[:,1]
        process['O2 WWTP [mol/h]'] = o2.demand
        process['WWTP heat demand [kWh/h]'] = heat.demand_tot
        
else:
    run_type = 'optimization'
    sims = len(elz_size_vector) * len(h2st_size_vector) * len(wind_size_vector) * len(pv_size_vector) * len(meth_scale_vector) * len(bat_size_vector)
    count = 0 # Defining counting variables
    # Defining results dataframe
    results = pd.DataFrame({'KPIs': ['LCOP2G (curt)', 'LCOP2G', 'MSP', 'MSP (no curt)', 'LCOE', 'Gas eff.', 'Heat eff.', 'Tot eff.', 'AEF net', 'MEF net', 'Starts', 'Standby', 'FLHs', 'Loss [%]', 'O2 util.', 'O2 dem.', 'Heat util', 'Heat dem.', 'RES [%]', 'LCOP2G BY diff.', 'LCOP2G BY rel.', 'MSP BY diff.', 'MSP BY rel.', 'NPV O2', 'NPV HEAT']}).set_index('KPIs')

""" Defining additional simulation parameters """
if year == 2020: # Number of hours during the specified year
    hrs = 8784
else:
    hrs = 8760

""" Run optimization """
for e in range(len(elz_size_vector)):
    for m in range(len(meth_scale_vector)):
        for s in range(len(h2st_size_vector)):
            for w in range(len(wind_size_vector)):
                for p in range(len(pv_size_vector)):
                    for b in range(len(bat_size_vector)):
                        
                        """ Re-define optimized components """
                        pem = params.Electrolyzer(elz_size_vector[e]) # Create electrolyzer
                        pem.efficiency('No plot') # Create electrolyzer efficiency curve
                        meth = params.Methanation(meth_scale_vector[m], biogas.min_co2) # Create methanation reactor
                        storage.size = h2st_size_vector[s] # Define storage size for this run
                        bat.size = bat_size_vector[b]
                        wind_size = wind_size_vector[w] * pem.size
                        pv_size = pv_size_vector[p] * pem.size
                        res.wind_gen *= wind_size / res.wind_size # Update wind generation using new capacity
                        res.wind_size = wind_size # Update wind size
                        res.pv_gen *= pv_size / res.pv_size # Update wind generation using new capacity
                        res.pv_size = pv_size # Update wind size
                        bg_comp = params.Compressor(meth.flow_max/3600, meth.pres, biogas.pres, biogas.temp) # Create biogas compressor
                        
                        """ Initiate process simulation """
                        # Initial values for dynamic variables
                        h2_storage = 0
                        elz_on = 0 # Start in off mode
                        elz_standby = 1 # Assuming no cold start from initial start
                        elz_off = 0
                        prev_mode = 1 # Assuming fast start for initial electrolyzer
                        bat_storage = 0
                        
                        # Creating variables for data saving
                        electrolyzer = np.zeros(len(grid.spot_price))
                        wind_use = np.zeros(len(grid.spot_price))
                        pv_use = np.zeros(len(grid.spot_price))
                        grid_use = np.zeros(len(grid.spot_price))
                        h2_used = np.zeros(len(grid.spot_price))
                        h2_storage_list = np.zeros(len(grid.spot_price))
                        h2_production = np.zeros(len(grid.spot_price))
                        electrolyzer_on = np.zeros(len(grid.spot_price))
                        electrolyzer_off = np.zeros(len(grid.spot_price))
                        electrolyzer_standby = np.zeros(len(grid.spot_price))
                        battery_state = np.zeros(len(grid.spot_price))
                        sys_op = np.zeros(len(grid.spot_price))
                        
                        # Define hourly hydrogen demand
                        h2_demand = ((biogas.flow[:,1] * 4) * (1-meth.microb_cons)) # 4:1 ratio of H2 and CO2 [mol/h] minus recycled CO2 and H2 and microbial consumption
                        h2_demand = np.minimum(h2_demand,np.zeros(hrs,)+(meth.size_mol*4)) # Also limited by methanation reactor size
                        h2_demand = np.where(h2_demand<(meth.flow_min*4),0,h2_demand)
                        h2_demand_kg = h2_demand * 2.02 / 1000
                        
                        """ Daily electrolyzer dispatch on day-ahead market """
                        for d in range(int(hrs/24)): # Simulating on a daily basis
                            # Specific hours during day "d"
                            i1 = d*24
                            i2 = i1 + 24
                            
                            # Check last hour of previous day
                            if d != 0:
                                if elz_on == 1 or elz_standby == 1:
                                    prev_mode = 1
                                else:
                                    prev_mode = 0
                            
                            # Daily dispatch optimization
                            elz_dispatch = dispatch.p2g_wwtp3(h2_dem=h2_demand_kg[i1:i2], heat_demand=heat.demand_tot[i1:i2], heat_value=heat.dh_price, usable_heat=heat.usable, meth_spec_heat=meth.spec_heat, o2_demand=o2.demand[i1:i2]*32/1000, o2_power=o2.aerator_savings, k_values=pem.k_values, m_values=pem.m_values, grid=grid.spot_price[i1:i2], wind=res.wind_gen, pv=res.pv_gen, elz_max=pem.size_degr, elz_min=pem.min_load*pem.size_degr, elz_eff=pem.n_sys, aux_cons=pem.aux_cons, meth_max=meth.size_mol*2.02*4/1000, meth_min=meth.min_load*meth.size_mol*4*2.02/1000, h2st_max=storage.size, h2st_prev=h2_storage, prev_mode=prev_mode, startup_cost=pem.start_cost, standby_cost=pem.standby_cost, bat_cap=bat.size, bat_eff=bat.eff, bat_prev=bat_storage, meth_el_factor=meth.spec_el, h2o_cons=pem.water_cons, temp=pem.temp, h2o_temp=pem.h2o_temp, biogas=biogas.flow[i1:i2], comp_el_factor=bg_comp.spec_el, elz_startup_time=pem.start_time/60)# wind_cost=wind_lcoe, pv_cost=pv_lcoe)
                                   
                            # Save daily data
                            electrolyzer[i1:i2] = elz_dispatch.iloc[:,0]
                            wind_use[i1:i2] = elz_dispatch.iloc[:,2]
                            pv_use[i1:i2] = elz_dispatch.iloc[:,3]
                            grid_use[i1:i2] = elz_dispatch.iloc[:,1]
                            h2_storage_list[i1:i2] = elz_dispatch.iloc[:,5]
                            h2_production[i1:i2] = elz_dispatch.iloc[:,7]
                            h2_used[i1:i2] = elz_dispatch.iloc[:,8]
                            electrolyzer_on[i1:i2] = elz_dispatch.iloc[:,11]
                            electrolyzer_standby[i1:i2] = elz_dispatch.iloc[:,12]
                            electrolyzer_off[i1:i2] = elz_dispatch.iloc[:,13]
                            elz_on = electrolyzer_on[-1]
                            elz_off = electrolyzer_off[-1]
                            elz_standby = electrolyzer_standby[-1]
                            battery_state[i1:i2] = elz_dispatch.iloc[:,19]
                            h2_storage = h2_storage_list[-1]
                            bat_storage = battery_state[-1]
                            sys_op[i1:i2] = elz_dispatch.iloc[:,24]
                        
                        h2_storage_list_prev = np.roll(h2_storage_list, 1) # Creating previous hour storage array
                        h2_storage_list_prev[0] = 0
                        
                        """ Hourly operation based on dispatch """
                        # Number of cold starts
                        starts = np.zeros(len(electrolyzer))
                        starts[0] = 0
                        for i in range(len(starts)):
                            if i > 0:
                                if electrolyzer[i-1] == 0 and electrolyzer[i] > 0 and electrolyzer_standby[i-1] != 1:
                                    starts[i] = 1
                                else:
                                    starts[i] = 0                            
                        
                        # Hydrogen production
                        h2_flow, elz_heat, T_h2_out, o2_flow, h2o_cons, stack_eff, sys_eff, elz_heat_nonnet = comps.electrolyzer(dispatch=electrolyzer, prod=h2_production, aux=pem.aux_cons, temp=pem.temp, h2o_temp=pem.h2o_temp, heat_time=pem.heatup_time, startups=starts, h2o_cons=pem.water_cons, year=year)
                        h2st_in = np.maximum(0,h2_production-h2_used) * 1000 / 2.02
                        h2st_out = np.maximum(0,h2_used-h2_production) * 1000 / 2.02
                        h2_meth = h2_used * 1000 / 2.02 # Converting to mol/h
                        
                        # Defining biogas flow into the P2G system
                        co2_flow = h2_meth / ((1-meth.microb_cons)*4) # CO2 required to match hydrogen production
                        p2g_frac = np.divide(co2_flow, biogas.flow[:,1].T, out=np.zeros_like((co2_flow)), where=biogas.flow[:,1]!=0) # Fraction of biogas used in P2G
                        biogas_in = biogas.flow.T * p2g_frac # Biogas input to the P2g system
                        flared_gas = biogas.flow[:,0].T * abs(np.around((1-p2g_frac),6)) # Amount of non-upgraded, and thus flared, biomethane
                        
                        # Biogas compression (flow rate in; compressor power)
                        bg_comp_power = comps.compressor(flow=biogas_in.sum(axis=0), temp_in=biogas.temp, p_in=biogas.pres, p_out=meth.pres, n_isen=bg_comp.n_isen, n_motor=bg_comp.n_motor, year=year, N=bg_comp.N, z=bg_comp.z, k=bg_comp.k) #[kWh]

                        # Gas mixing (Biogas, hydrogen, temp in; total flow and temp out)
                        inlet_flow, T_inlet = comps.mixer(h2=h2_meth, co2=biogas_in[1], ch4=biogas_in[0], h2_temp=pem.temp, bg_temp=biogas.temp)

                        # Methanation (molar flows, temp. in; molar flows, electricity consumption, excess heat, condensed water and microbial CO2 consumption out)
                        meth_outlet_flow, meth_power, meth_heat, h2o_cond1, microbial_co2 = comps.methanation(meth_flow=inlet_flow, T=meth.temp, T_in=T_inlet, el_cons=meth.el_cons, n=meth.n, microb_cons=meth.microb_cons)

                        if simulation_details == 'Yes' or simulation_details == 'yes': # Storing detailed results
                            process['H2 demand [mol/h]'] = h2_demand
                            process['Elz dispatch [kWh/h]'] = electrolyzer
                            process['System dispatch [kWh/h]'] = sys_op
                            process['Standby'] = electrolyzer_standby
                            process['Elz cold start'] = starts
                            process['Wind use [kWh/h]'] = wind_use
                            process['Wind gen [kWh/h]'] = res.wind_gen
                            process['PV use [kWh/h]'] = pv_use
                            process['PV gen [kWh/h]'] = res.pv_gen
                            process['Grid use [kWh/h]'] = grid_use
                            process['H2 production [kg/h]'] = h2_production
                            process['H2 to meth [mol/h]'] = h2_meth
                            process['H2 to storage [mol/h]'] = h2st_in
                            process['H2 from storage [mol/h]'] = h2st_out
                            if storage.size > 0:
                                process['H2 storage [%]'] = (h2_storage_list/(storage.size))*100
                            process['Elz heat [kWh/h]'] = elz_heat
                            process['O2 out [mol/h]'] = o2_flow
                            process['H2O cons [mol/h]'] = h2o_cons
                            process['Biogas comp [kWh/h]'] = bg_comp_power
                            process['Meth CH4 in [mol/h]'] = inlet_flow[2]
                            process['Meth CO2 in [mol/h]'] = inlet_flow[1]
                            process['Meth in temp [C]'] = T_inlet
                            process['CH4 out [mol/h]'] = meth_outlet_flow[2]
                            process['H2 out [mol/h]'] = meth_outlet_flow[0]
                            process['CO2 out [mol/h]'] = meth_outlet_flow[1]
                            process['H2O(g) out [mol/h]'] = meth_outlet_flow[3]
                            process['H2O(l) out [mol/h]'] = h2o_cond1
                            process['Meth el [kWh/h]'] = meth_power
                            process['Meth heat [kWh/h]'] = meth_heat
                            process['CH4 flared [mol/h]'] = flared_gas
                            process['Stack efficiency [%]'] = stack_eff * 100
                            process['System efficiency [%]'] = sys_eff * 100
                            if bat.size > 0:
                                process['Battery state [%]'] = np.array(battery_state) * 100 / bat.size

                        """ Technical analysis """
                        # Gas production and flaring
                        ch4_p2g = (meth_outlet_flow[2] - inlet_flow[2]).sum() * tec.ch4_mol / 1000 # Annual CH4 production from P2G [MWh]
                        ch4_total = meth_outlet_flow[2].sum() * tec.ch4_mol / 1000 #  Total annual CH4 production [MWh]
                        flare_frac = (flared_gas.sum()/biogas.flow[:,0].sum()) * 100 # Fraction of biogas that was flared [%]
                          
                        # Electrolyzer details
                        elz_flh = round(electrolyzer.sum() / (pem.size_degr)) # Full load hours of the electrolyzer
                        stack_reps = math.floor((tec.lifetime-1) / pem.stack_rep) # Number of stack replacements during project lifetime (minus 1 since only 2 replacements are required every ten years for a 30 year lifetime for example)
                        if stack_reps == 1: # Define stack replacement years
                            rep_years = np.array([pem.stack_rep])
                        elif stack_reps == 2:
                            rep_years = np.array([pem.stack_rep, pem.stack_rep*2])
                        elif stack_reps == 3:
                            rep_years = np.array([pem.stack_rep, pem.stack_rep*2, pem.stack_rep*3])
                        elif stack_reps == 0:
                            rep_years = np.array([0])
                        
                        # Electricity
                        if bat.size > 0: # If battery is included
                            bat_in = np.zeros(len(electrolyzer))
                            bat_in_wind = np.zeros(len(electrolyzer))
                            excess_wind = np.zeros(len(electrolyzer))
                            bat_in_pv = np.zeros(len(electrolyzer))
                            excess_pv = np.zeros(len(electrolyzer))
                            bat_in_grid = np.zeros(len(electrolyzer))
                            bat_loss = np.zeros(len(electrolyzer))
                            for h in range(len(electrolyzer)):
                                if h == 0:
                                    bat_in[i] = battery_state[h]
                                    bat_in_pv[i] = np.round(np.minimum((res.pv_gen[h] - pv_use[h]), bat_in[h]),6)
                                    excess_pv[i] = np.round(np.maximum((res.pv_gen[h] - pv_use[h]) - bat_in_pv[h],0),6)
                                    bat_in_wind[i] = np.round(np.minimum((res.wind_gen[h] - wind_use[h]), bat_in[h] - bat_in_pv[h]),6)
                                    excess_wind[i] = np.round(np.maximum((res.wind_gen[h] - wind_use[h]) - bat_in_wind[h],0),6)
                                    bat_in_grid[i] = np.round(bat_in[h] - bat_in_pv[h] - bat_in_wind[h],6)
                                    bat_loss[i] = bat_in[h] * (1-bat.eff)
        
                                else:
                                    bat_in[i] = np.round((np.maximum((battery_state[h] - battery_state[h-1]),0)),6)
                                    bat_in_pv[i] = np.round(np.minimum((res.pv_gen[h] - pv_use[h]), bat_in[h]),6)
                                    excess_pv[i] = np.round(np.maximum((res.pv_gen[h] - pv_use[h]) - bat_in_pv[h],0),6)
                                    bat_in_wind[i] = np.round(np.minimum((res.wind_gen[h] - wind_use[h]), bat_in[h] - bat_in_pv[h]),6)
                                    excess_wind[i] = np.round(np.maximum((res.wind_gen[h] - wind_use[h]) - bat_in_wind[h],0),6)
                                    bat_in_grid[i] = np.round(bat_in[h] - bat_in_pv[h] - bat_in_wind[h],6)
                                    bat_loss[i] = bat_in[h] * (1-bat.eff)
                            
                            res_frac = (wind_use + pv_use + bat_in_wind + bat_in_pv).sum() * 100  / sys_op.sum() # Renewable electricity fraction [%]
                            if wind_size > 0:
                                wind_frac = (wind_use + bat_in_wind).sum() / (wind_use + pv_use + bat_in_wind + bat_in_pv).sum() # Wind fraction [%]
                            if pv_size > 0:
                                pv_frac = (pv_use + bat_in_pv).sum() / (wind_use + pv_use + bat_in_wind + bat_in_pv).sum() # PV fraction [%]

                        else: # If no battery
                            
                            excess_wind = np.maximum(res.wind_gen - wind_use, np.zeros(hrs,)) # Excess wind [MWh]
                            excess_pv = np.maximum(res.pv_gen - pv_use, np.zeros(hrs,)) # Excess PV [MWh]
                            res_frac = (wind_use + pv_use).sum() / sys_op.sum() # Renewable electricity fraction [%]
                            if wind_size > 0:
                                wind_frac = wind_use.sum() / (wind_use + pv_use).sum() # Wind fraction [%]
                            if pv_size > 0:
                                pv_frac = pv_use.sum() / (wind_use + pv_use).sum() # PV fraction [%]

                        curtailment = (excess_wind.sum() + excess_pv.sum()) / 1000 # Total curtailment [MWh]
                        tot_res = (res.wind_gen.sum() + res.pv_gen.sum()) / 1000 # Total RES generation [MWh]
                        if tot_res > 0:
                            curt_frac = curtailment * 100 / tot_res # Curtailed fraction of RES generation [%]
                        else:
                            curt_frac = 0
                            
                        # Storage details
                        if storage.size > 0: 
                            h2st_cycles = np.round(h2st_in.sum() * 2.02 / (storage.size * 1000)) # Number of full storage cycles
                            h2st_90 = (((h2_storage_list*100/storage.size) > 90) * 1).sum() / len(h2_storage_list) # Percent of time above 90 % full
                        else:
                            h2st_cycles = 0
                            h2st_90 = 0
                        
                        # Battery details
                        if bat.size > 0: 
                            bat_cycles = np.round(bat_in.sum() / bat.size) # Number of full storage cycles
                            bat_90 = (((battery_state*100/bat.size) > 90) * 1).sum() / len(battery_state) # Percent of time above 90 % full
                        else:
                            bat_cycles = 0
                            bat_90 = 0
                        
                        # Heat details
                        heat_prod = heat.usable * (elz_heat + meth_heat) # Low-grade heat produced [kWh/h] (excluding condenser heat)
                        heat_prod_out = np.clip(heat_prod, a_min=0, a_max=None) # Exclude potential internal heat consumption by electrolyzer
                        # total_heat_demand = (2*(heat.demand_tot - heat.demand_aux)) + heat.demand_aux # Use for testing thermophilic digestion in the WWTP
                        heat_wwtp = np.minimum(heat_prod_out, heat.demand_tot)
                        heat_elz_use = -np.clip(heat_prod, a_min=None, a_max=0) # All internal heat consumption by electrolyzer 
                        heat_use_frac = heat_wwtp.sum() * 100 / heat_prod_out.sum() # How much of the output heat is utilized [%]
                        # heat_use_frac = ((heat_wwtp.sum()+3037000) / heat_prod_out.sum()) * 100 # Assuming hygienization of co-digestion substrate
                        if heat.demand_tot.sum() > 0:
                            heat_wwtp_use_frac = (heat_wwtp.sum() / heat.demand_tot.sum()) * 100
                        else:
                            heat_wwtp_use_frac = 0

                        # Oxygen
                        o2_wwtp = np.minimum(o2_flow, o2.demand)
                        o2_loss = np.maximum(o2_flow - o2_wwtp,0).sum()
                        o2_use_frac = (o2_wwtp.sum() / o2_flow.sum()) * 100
                        o2_wwtp_use_frac = (o2_wwtp.sum() / o2.demand.sum()) * 100
                        o2_energy_savings = o2.aerator_savings * o2_wwtp * 32 / 1000 #[kWh]
                        # o2_energy_frac = 100 * o2_energy_savings / electrolyzer.sum() # % of electrolyzer energy input saved

                        """ Economic analysis """
                        # Electricity costs
                        if bat.size > 0:
                            wind_cost = ((wind_use+bat_in_wind) * (res.wind_lcoe + grid.fee)).sum() / 1000 # Wind via PPA
                            pv_cost = ((pv_use+bat_in_pv) * (res.pv_lcoe)).sum() / 1000 # Local PV
                        else:
                            wind_cost = (wind_use * (res.wind_lcoe + grid.fee)).sum() / 1000 # Wind via PPA
                            pv_cost = (pv_use * res.pv_lcoe).sum() / 1000 # Local PV
                        curt_cost = ((excess_wind.sum() * res.wind_lcoe) + (excess_pv.sum() * res.pv_lcoe)) / 1000 # Including curtailed generation at the same cost
                        grid_cost = (grid_use * grid.spot_price).sum() / 1000 # Grid fee already included
                        startup_costs = (pem.size_degr * pem.start_cost * grid.spot_price * starts / 1000).sum()
                        el_cost = wind_cost + pv_cost + grid_cost + startup_costs
                        el_cost_curt = (el_cost + curt_cost) 
                        # Averages
                        # avg_grid_price = grid_cost * 1000 / process['Grid use [kWh/h]'].sum() # [€/MWh]
                        # avg_tot_price = el_cost_curt * 1000 / process['System dispatch [kWh/h]'].sum() # [€/MWh]
                        # avg_el_ch4 = el_cost_curt / ch4_p2g # [€/MWh]
                        
                        # Storage costs
                        # O2 storage [Not implemented]
                        # o2st_CAPEX = o2st_cap * o2st_capex
                        # o2st_OPEX = o2st_cap * (o2st_opex/100)
                        # Heat storage [Not implemented]
                        # heatst_CAPEX = heatst_cap * heatst_capex
                        # heatst_OPEX = heatst_cap * (heatst_opex/100)
                        # Battery
                        bat_CAPEX = bat.size * bat.capex # Battery CAPEX
                        bat_OPEX = bat_CAPEX * (bat.opex/100) # Battery OPEX
                        # Hydrogen storage
                        h2st_CAPEX = storage.size * storage.capex # H2 storage CAPEX
                        h2st_OPEX = storage.opex * 0.01 * h2st_CAPEX # H2 storage OPEX

                        # Hydrogen costs
                        elz_CAPEX = pem.capex * pem.capex_ref * ((pem.size/pem.capex_ref)**pem.scaling) # Electrolyzer CAPEX with scaling
                        elz_OPEX = pem.opex * 0.01 * elz_CAPEX # Electrolyzer fixed OPEX
                        h2o_opex = pem.water_cost * h2o_cons.sum() * 18.02 / (1000*997) # Water [€/m3 * mol * g/mol / (1000*kg/m3)]
                        stack_COST = pem.stack_cost * elz_CAPEX # Total cost of stack replacements
                        H2_CAPEX = elz_CAPEX + h2st_CAPEX # Total hydrogen CAPEX
                        H2_OPEX = elz_OPEX + h2o_opex + h2st_OPEX # Total hydrogen OPEX
                        H2_STACK = stack_COST # Stack replacement costs

                        # Biogas costs
                        biogas_loss_cost = biogas.lcoe * flared_gas.sum() * tec.ch4_mol / 1000 # Cost of flared biogas (for LCOP2G)
                        biogas_cost_tot = biogas.lcoe * biogas.flow[:,0].sum() * tec.ch4_mol / 1000 # Cost of all produced biogas (for LCOM)
                        
                        # Methanation costs
                        meth_CAPEX = meth.capex * meth.capex_ref * ((meth.size/meth.capex_ref)**meth.scaling) # Methanation CAPEX with scaling per MWCH4 out
                        meth_OPEX = meth.opex * 0.01 * meth_CAPEX # Methanarion fixed OPEX
                        bg_comp_capex = 30000 * (bg_comp.size**0.48) # Biogas compressor CAPEX
                        bg_comp_opex = bg_comp.opex * 0.01 * bg_comp_capex # Biogas compressor OPEX
                        METH_CAPEX = meth_CAPEX + bg_comp_capex # Total methanation CAPEX
                        METH_OPEX = meth_OPEX + bg_comp_opex # Total methanation OPEX

                        # By-product integration costs
                        heat_size = (pem.heat_max + meth.heat_max) * heat.usable # Size of heat system [kW]
                        heat_system_CAPEX = heat.capex * heat.capex_ref * ((heat_size/heat.capex_ref)**heat.scaling) # Heat equipment CAPEX
                        heat_piping_CAPEX = heat.piping_capex * tec.piping_dist # Heat piping CAPEX
                        heat_integration_CAPEX = heat_system_CAPEX + heat_piping_CAPEX # Total heat CAPEX
                        heat_integration_OPEX = heat_integration_CAPEX * (heat.opex/100) # Heat OPEX
                        o2_aerator_CAPEX = o2.aerator_capex * o2.aerator_ref * ((pem.size/o2.aerator_ref)**o2.aerator_scaling) # Oxygen aerator CAPEX
                        o2_piping_CAPEX = o2.piping_capex * tec.piping_dist # Oxygen piping CAPEX
                        o2_integration_CAPEX = o2_piping_CAPEX + o2_aerator_CAPEX # Total oxygen CAPEX
                        o2_integration_OPEX = o2_integration_CAPEX * (o2.opex/100) # Oxygen OPEX
                        BY_CAPEX = heat_integration_CAPEX + o2_integration_CAPEX # Total by-product CAPEX
                        BY_OPEX = heat_integration_OPEX + o2_integration_OPEX # Total by-product OPEX
                        # rel_heat_capex = heat_integration_CAPEX * 100 / (elz_CAPEX + meth_CAPEX) # Heat CAPEX share of PEM and methanation
                        # rel_o2_capex = o2_integration_CAPEX * 100 / (elz_CAPEX + meth_CAPEX) # Oxygen CAPEX share of PEM and methanation
                        
                        co2_opex = tec.co2_cost * co2_flow.sum() * 44.01 / (1000*1000) # CO2 cost (assumed zero)
                        
                        # Overall costs
                        CAPEX = H2_CAPEX + METH_CAPEX + BY_CAPEX + bat_CAPEX # Total component CAPEX
                        OPEX = H2_OPEX + METH_OPEX + BY_OPEX + el_cost + bat_OPEX + co2_opex # Total OPEX
                        OPEX_curt = H2_OPEX + METH_OPEX + BY_OPEX + el_cost_curt + bat_OPEX + co2_opex
                        CAPEX *= (1+(tec.install_cost/100)) # Total CAPEX including installation
                        OPEX_tot = OPEX + biogas_loss_cost # Including flared biogas costs
                        OPEX_tot_curt = OPEX_curt + biogas_loss_cost # Including flared biogas and curtailment costs
                        OPEX_msp = OPEX_curt + biogas_cost_tot # Including biogas plant costs
                        OPEX_msp_nocurt = OPEX + biogas_cost_tot # Including biogas plant and curtailment costs
                        
                        # By-product income
                        o2_income = (o2_wwtp * 32 * o2.aerator_savings * grid.spot_price / (1000*1000)).sum() # Oxygen
                        # o2_income = ((o2_wwtp * 32 * aerator_savings * grid.spot_price / (1000*1000))*0.85).sum() + ((o2_wwtp * 32 * 0.7 * grid.spot_price / (1000*1000))*0.15).sum() # Income with a fraction of ozone production
                        heat_income = (((heat_wwtp[0:1415]-heat_elz_use[0:1415]).sum() + (heat_wwtp[8016:8759]-heat_elz_use[8016:8759]).sum())*heat.dh_price[3]/1000) + (((heat_wwtp[1416:3623]-heat_elz_use[1416:3623]).sum() + (heat_wwtp[5832:8015]-heat_elz_use[5832:8015]).sum())*heat.dh_price[0]/1000) + (((heat_wwtp[3624:5831]-heat_elz_use[3624:5831]).sum())*heat.dh_price[1]/1000) #Heat income with variable DH price
                        # heat_income = heat_income + (np.average([dh_winter,dh_summer,dh_spr_aut,dh_spr_aut]) * 3037) # Assuming replacing hygienization as well
                        # heat_income = np.average([dh_winter,dh_summer,dh_spr_aut,dh_spr_aut]) * heat_prod.sum() * heat_frac_use / 1000 # Assuming a fix utilization factor
                        by_income = o2_income + heat_income # Total by-product income

                        """ KPI calculations """

                        # Economic KPIs
                        # Levelized costs
                        lcop2g = tec.lcox(opex=OPEX_tot-by_income, capex=CAPEX, stack=H2_STACK, prod=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) # Levelized cost of power-to-gas [€/MWh of CH4]
                        lcop2g_curt = tec.lcox(opex=OPEX_tot_curt-by_income, capex=CAPEX, stack=H2_STACK, prod=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) # Levelized cost of power-to-gas including curtailment cost [€/MWh of CH4]
                        lcom = tec.lcox(opex=OPEX_msp-by_income, capex=CAPEX, stack=H2_STACK, prod=ch4_total, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        lcom_no_curt = tec.lcox(opex=OPEX_msp_nocurt-by_income, capex=CAPEX, stack=H2_STACK, prod=ch4_total, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        # lcop2g_noo2 = tec.lcox(opex=OPEX_tot-heat_income-o2_integration_OPEX, capex=CAPEX-o2_integration_CAPEX, stack=H2_STACK, prod=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        lcop2g_noo2_curt = tec.lcox(opex=OPEX_tot_curt-heat_income-o2_integration_OPEX, capex=CAPEX-o2_integration_CAPEX, stack=H2_STACK, prod=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        # lcom_noo2_curt = tec.lcox(opex=OPEX_msp-heat_income-o2_integration_OPEX, capex=CAPEX-o2_integration_CAPEX, stack=H2_STACK, prod=ch4_total, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        # lcop2g_noheat = tec.lcox(opex=OPEX_tot-o2_income-heat_integration_OPEX, capex=CAPEX-heat_integration_CAPEX, stack=H2_STACK, prod=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        lcop2g_noheat_curt = tec.lcox(opex=OPEX_tot_curt-o2_income-heat_integration_OPEX, capex=CAPEX-heat_integration_CAPEX, stack=H2_STACK, prod=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        # lcom_noheat_curt = tec.lcox(opex=OPEX_msp-o2_income-heat_integration_OPEX, capex=CAPEX-heat_integration_CAPEX, stack=H2_STACK, prod=ch4_total, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        # lcop2g_nobys = tec.lcox(opex=OPEX_tot-o2_integration_OPEX-heat_integration_OPEX, capex=CAPEX-o2_integration_CAPEX-heat_integration_CAPEX, stack=H2_STACK, prod=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        lcop2g_nobys_curt = tec.lcox(opex=OPEX_tot_curt-o2_integration_OPEX-heat_integration_OPEX, capex=CAPEX-o2_integration_CAPEX-heat_integration_CAPEX, stack=H2_STACK, prod=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        # lcom_nobys_curt = tec.lcox(opex=OPEX_msp-o2_integration_OPEX-heat_integration_OPEX, capex=CAPEX-o2_integration_CAPEX-heat_integration_CAPEX, stack=H2_STACK, prod=ch4_total, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        # lcop2g_diff = lcop2g_nobys - lcop2g
                        # lcop2g_diff_curt = lcop2g_nobys_curt - lcop2g_curt
                        # lcom_diff_curt = msp_nobys_curt - lcom
                        # lcop2g_diff_rel = lcop2g_diff / lcop2g
                        # lcop2g_diff_rel_curt = lcop2g_diff_curt / lcop2g_curt
                        # lcom_diff_rel_curt = msp_diff_curt / lcom
                        # lcop2g_diff_rel_o2 = (lcop2g-lcop2g_noheat) / lcop2g
                        # lcop2g_diff_rel_o2_curt = (lcop2g_curt-lcop2g_noheat_curt) / lcop2g_curt
                        # lcom_diff_rel_o2_curt = (lcom-msp_noheat_curt) / lcom
                        
                        # Net present values
                        # income_gas = by_income + (gas_price * ch4_p2g) #[€] Income including gas sales
                        # npv = tec.npv(opex=OPEX_tot, income=income_gas, capex=CAPEX, stack=H2_STACK, stack_reps=stack_reps, rep_years=rep_years) #[€]
                        npv_o2 = tec.npv(opex=o2_integration_OPEX, income=o2_income, capex=o2_integration_CAPEX, stack=0, stack_reps=0, rep_years=rep_years) #[€]
                        npv_heat = tec.npv(opex=heat_integration_OPEX, income=heat_income, capex=heat_integration_CAPEX, stack=0, stack_reps=0, rep_years=rep_years) #[€]
                        
                        # Comparison to amine scrubbing
                        # amine_flow_rate = 1200 #[Nm3/hr]
                        # amine_scrubber_CAPEX = 2300 * amine_flow_rate #[€]
                        # amine_scrubber_el_cost = sum((process['CH4 out [mol/h]'] - process['Meth CH4 in [mol/h]']) * tec.nm3_mol * 0.1 * grid.spot_price / 1000) #[€]
                        # amine_scrubber_heat_cost = sum((process['CH4 out [mol/h]'] - process['Meth CH4 in [mol/h]']) * tec.nm3_mol * 0.6 * np.mean(heat.dh_price) / 1000) #[€]
                        # amine_scrubber_opex_fix = amine_scrubber_CAPEX * 0.05
                        # amine_scrubber_OPEX = amine_scrubber_el_cost + amine_scrubber_heat_cost + amine_scrubber_opex_fix
                        # npv_rep = tec.npv(opex=OPEX_tot-amine_scrubber_OPEX, income=INCOME_GAS, capex=CAPEX-amine_scrubber_CAPEX, stack=H2_STACK, stack_reps=stack_reps, rep_years=rep_years) #[€]
                        # lcoe_amine = kpis.lcoe(opex=amine_scrubber_OPEX, capex=amine_scrubber_CAPEX, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_total-ch4_p2g, stack_reps=0, rep_years=0) #[€/MWh of CH4]
                        # lcom_amine = kpis.lcoe(opex=amine_scrubber_OPEX+biogas_cost_tot, capex=amine_scrubber_CAPEX, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_total-ch4_p2g, stack_reps=0, rep_years=0) #[€/MWh of CH4]
                        
                        # Efficiencies
                        tot_energy_cons = (electrolyzer + bg_comp_power + meth_power + (electrolyzer_standby*pem.standby_el)).sum() # System energy consumption [kWh]
                        n_gas, n_heat, n_o2, n_tot, n_biomethane, n_max, n_theory = tec.efficiencies(p2g_prod=ch4_p2g, tot_prod=ch4_total, heat_use=heat_wwtp.sum(), o2_use=o2_energy_savings.sum(), el_cons=tot_energy_cons, tot_heat=(elz_heat + meth_heat).sum(), usable_heat=heat.usable)
                        # Emissions
                        aef_net, mef_net, __, __ = tec.emissions(aefs=grid.aefs, mefs=grid.mefs, wind_efs=res.wind_efs, pv_efs=res.pv_efs, grid=grid_use, wind=wind_use, pv=pv_use, prod=ch4_p2g, heat_use=heat_wwtp.sum(), heat_aef=heat.ems, heat_mef=heat.ems_marginal, o2_use=o2_energy_savings, bg_ef=biogas.ef, flared=flared_gas.sum())
                           
                        if run_type == "optimization":
                            results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(pem.size/1000,meth.size/1000,storage.size,wind_size/1000,pv_size/1000,bat.size)] = [lcop2g_curt, lcop2g, lcom, lcom_no_curt, lcoe, n_gas, n_tot, n_heat_o2, aef_net, mef_net, sum(starts), sum(electrolyzer_standby), elz_flh, flare_frac, o2_use_frac, o2_wwtp_use_frac, heat_use_frac, heat_wwtp_use_frac, res_frac, lcop2g_diff_curt, lcop2g_diff_rel_curt, msp_diff_curt, msp_diff_rel_curt, npv_o2, npv_heat] #Saving optimization results
                            count = count + 1 # Counting simulations
                            print('{}/{} simulations performed'.format(count,sims))
                            
                            # Cost breakdown
                            # total = kpis.lcoe(opex=OPEX_tot_curt, capex=CAPEX, stack=H2_STACK, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                            # elz_lcoe = kpis.lcoe(opex=elz_OPEX, capex=elz_CAPEX, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                            # stack_rep_lcoe = kpis.lcoe(opex=0, capex=0, stack=stack_COST, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) * 100 / total #[€/MWh of CH4]
                            # water_lcoe = kpis.lcoe(opex=h2o_opex, capex=0, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                            # h2st_lcoe = kpis.lcoe(opex=h2st_OPEX, capex=h2st_CAPEX, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                            # meth_lcoe = kpis.lcoe(opex=meth_OPEX, capex=meth_CAPEX, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                            # comp_lcoe = kpis.lcoe(opex=bg_comp_opex, capex=bg_comp_capex, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                            # heat_lcoe = kpis.lcoe(opex=heat_integration_OPEX, capex=heat_integration_CAPEX, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                            # o2_lcoe = kpis.lcoe(opex=o2_integration_OPEX, capex=o2_integration_CAPEX, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                            # grid_lcoe = kpis.lcoe(opex=grid_cost, capex=0, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                            # pv_lcoe1 = kpis.lcoe(opex=pv_cost, capex=0, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                            # wind_lcoe1 = kpis.lcoe(opex=wind_cost, capex=0, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                            # bg_loss_lcoe = kpis.lcoe(opex=biogas_loss_cost, capex=0, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total
                            # install_lcoe = kpis.lcoe(opex=0, capex=INSTALL, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total
                            # curt_lcoe1 = kpis.lcoe(opex=curt_cost, capex=0, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                            # o2_income_lcoe = kpis.lcoe(opex=-o2_income, capex=0, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                            # heat_income_lcoe = kpis.lcoe(opex=-heat_income, capex=0, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                            # cost_breakdown['{} MW'.format(elz_size)] = [elz_lcoe,stack_rep_lcoe,water_lcoe,h2st_lcoe,meth_lcoe,comp_lcoe,heat_lcoe,o2_lcoe,install_lcoe,bg_loss_lcoe,grid_lcoe,pv_lcoe1,wind_lcoe1,curt_lcoe1,o2_income_lcoe,heat_income_lcoe,lcop2g_curt] # Table

""" Printing and plotting simulation results """
if run_type == "single":
    # Print tables with main KPIs
    table_kpi = [['LCOP2G', 'LCOM', 'Gas eff.', 'O2 eff.', 'AEF net', 'MEF net', 'Loss %', 'RES [%]'], \
                  [lcop2g_curt, lcom, n_gas, n_tot, aef_net, mef_net, flare_frac, res_frac]]
    print(tabulate(table_kpi, headers='firstrow', tablefmt='fancy_grid'))

    table_by = [['O2 util. [%]', 'O2 dem. [%]', 'Heat util. [%]', '% Heat dem. [%]'], \
          [o2_use_frac, o2_wwtp_use_frac, heat_use_frac, heat_wwtp_use_frac]]
    print(tabulate(table_by, headers='firstrow', tablefmt='fancy_grid')) 
    
    # Plotting
    if plots == 'Yes' or plots == 'yes':        
        plotting.dispatch(electrolyzer=electrolyzer, elz_size_degr=elz_size_degr, h2_flow=h2_flow, elz_h2_max=elz_h2_max, h2_storage=(h2_storage_list/(storage.size))*100, spot_price=grid.spot_price, biogas_flow=biogas.flow, \
                     h2_demand=h2_demand, h2_used=h2_used, heat_demand_tot=heat_demand_tot, meth_heat=meth_heat, usable_heat=heat.usable, elz_heat=elz_heat, o2_wwtp=o2_wwtp, o2_flow=o2_flow)
        
        # plotting.byprods() # Not implemented yet
        
        plotting.sankey(grid_use=grid_use, pv_use=pv_use, wind_use=wind_use, bat_in_pv=bat_in_pv, bat_in_wind=bat_in_wind, electrolyzer=electrolyzer, biogas_in=biogas_in, ch4_mol=tec.ch4_mol, bg_comp_power=bg_comp_power, h2_used=h2_used, h2_kg=tec.h2_kg, \
                   flared=flared_gas, meth_el=meth_power, elz_heat_nonnet=elz_heat_nonnet, meth_heat=meth_heat, ch4_total=ch4_total, heat_wwtp=heat_wwtp, o2_energy_savings=o2_energy_savings, aerator_savings=o2.aerator_savings, o2_loss=o2_loss)
    
    
    # End time
    # end = time.time()
    # total_time = end - start
    # print('Time: ' + str(total_time))