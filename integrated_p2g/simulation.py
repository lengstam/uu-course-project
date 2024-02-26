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
import kpis
import dispatch
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
Integrated power-to-gas model

"""


"""
Comments on current version (1.0a):

REDO SIMULATIONS
    MAY HAVE DOUBLE-COUNTED CURTATAILED PV ABOVE 2X CAPACITY (PURE PV), TRY AND REDO
    HAVE USED THE SAME 24 HOURS FOR COMPRESSION CONSUMPTION
    HAVE USED A TOO HIGH VALUE FOR SPOT PRICE FROM GRID IN DISPATCH
"""

# start = time.time()

""" Optimization parameters """
elz_size_vector = [8.5]#[6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12] # [MW]
meth_scale_vector = [5]#[3.5,4,4.5,5] # [MWCH4]
h2st_size_vector = [400]#[0,100,200,300,400,500,600,700,800,900,1000] # [kg]
wind_size_vector = [1.25] # [ratio to elz]
pv_size_vector = [1.25] # [ratio to elz]
bat_size_vector = [0] # [MWh]

""" Simulation parameters """
year = 2021 # 2018-2021 available
bidding_zone = 'SE3' # ['SE1', 'SE2', 'SE3', 'SE4']
simulation_details = 'Yes' # ['Yes', 'No'] A "Process" dataframe with all process stages during all hours is created.

""" Creating component classes for variable storage in un-optimized components """
tec = params.TechnoEconomics(hv='hhv') # Techno-economic parameters
storage = params.Storage(h2_size=0, bat_size=0, o2_size=0, heat_size=0) # Storages (hydrogen, battery, oxygen, heat). Setting arbitrary sizes to be adjusted later within the optimization loop.
biogas = params.Biogas(data='real', year=year) # Biogas production
res = params.Renewables(wind_size=3000, pv_size=3000, year=year, lifetime=tec.lifetime) # Renewables (wind, PV). Assuming arbitrary sizes to be adjusted later within the optimization loop.
grid = params.Grid(year=year, zone=bidding_zone) # Electricity grid parameters
o2 = params.Oxygen(year=year) # Oxygen utilization system
heat = params.Heat(year=year) # Heat utilization system

""" Define run-type """
if len(elz_size_vector) == 1 and len(meth_scale_vector) == 1 and len(h2st_size_vector) == 1 and len(wind_size_vector) == 1 and len(pv_size_vector) == 1 and len(bat_size_vector) == 1:
    run_type = 'single'
    cost_breakdown = pd.DataFrame({'Costs': ['Electrolyser','Stack','Water','Storage','Meth','Comp','Heat','O2','Installation','Flaring','Grid','PV','Wind','Curtailment','O2 income','Heat income','Total']}).set_index('Costs')
    
    if simulation_details == 'Yes' or simulation_details == 'yes':
        process = other.data_saving(year=year) # Initiate process data saving
        #Save simulation independent data
        process['Elspot [€/MWh]'] = grid.spot_price
        process['Biogas (CH4) [mol/h]'] = biogas.flow[:,0]
        process['Biogas (CO2) [mol/h]'] = biogas.flow[:,1]
        process['O2 WWTP [mol/h]'] = o2.demand
        process['WWTP heat demand [kWh/h]'] = heat.demand_tot
        
else:
    run_type = 'optimization'
    #Counting etc.
    sims = len(elz_size_vector) * len(h2st_size_vector) * len(wind_size_vector) * len(pv_size_vector) * len(meth_scale_vector) * len(bat_size_vector)
    count = 0
    results = pd.DataFrame({'KPIs': ['LCOP2G (curt)', 'LCOP2G', 'MSP', 'MSP (no curt)', 'LCOE', 'Gas eff.', 'Heat eff.', 'Tot eff.', 'AEF net', 'MEF net', 'Starts', 'Standby', 'FLHs', 'Loss [%]', 'O2 util.', 'O2 dem.', 'Heat util', 'Heat dem.', 'RES [%]', 'LCOP2G BY diff.', 'LCOP2G BY rel.', 'MSP BY diff.', 'MSP BY rel.', 'NPV O2', 'NPV HEAT']}).set_index('KPIs')

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
                        storage.h2_size = h2st_size_vector[s] # Define storage size for this run
                        storage.bat_size = bat_size_vector[b]
                        wind_size = wind_size_vector[w] * pem.size
                        pv_size = pv_size_vector[p] * pem.size
                        res.wind_gen *= wind_size / res.wind_size # Update wind generation using new capacity
                        res.wind_size = wind_size # Update wind size
                        res.pv_gen *= pv_size / res.pv_size # Update wind generation using new capacity
                        res.pv_size = pv_size # Update wind size
                        bg_comp = params.Compressor(meth.flow_max/3600, meth.pres, biogas.pres, biogas.temp) # Create biogas compressor
                        
                        """ Process simulation """
                            
                         
                        #Initial values for dynamic variables
                        # T_elz = 20 #assuming ambient temperature
                        # T_meth = 0
                        h2_storage = 0
                        elz_on = 0 #start in off mode
                        elz_standby = 1 #assuming no cold start from initial start
                        elz_off = 0
                        prev_mode = 1 #assuming fast start for initial electrolyzer
                        bat_storage = 0
                        # meth_on = 0 #start in off mode
                        # prev_mode_meth = 1 #assuming no methanation start from inital start
                        
                        #Electrolyzer dispatch
                        electrolyzer = []
                        H2_demand = []
                        wind_use = []
                        pv_use = []
                        grid_use = []
                        h2_used = []
                        unmet_demand = []
                        h2_storage_list = []
                        h2_production = []
                        electrolyzer_on = []
                        electrolyzer_start = []
                        electrolyzer_off = []
                        electrolyzer_standby = []
                        o_prod = []
                        o_use = []
                        h_prod = []
                        h_use = []
                        h_inc = []
                        battery_state = []
                        battery_dis = []
                        grid_inc = []
                        o2_inc = []
                        sys_op = []
                        sb_el = []
                        m_el = []
                        c_el = []
                        e_h_dem = []
                        h2_prod_start = []
                        h2_prod_real = []
                        
                        # Daily electrolyzer dispatch on day-ahead market
                        if year == 2020:
                            hrs = 8784
                        else:
                            hrs = 8760
                        for d in range(int(hrs/24)): #daily basis
                            # Annual hours during specific day
                            i1 = d*24
                            i2 = i1 + 24
                            
                            h2_demand_hr = ((biogas.flow[i1:i2,1] * 4) * (1-meth.microb_cons))# - (4*gas_recirc[1]) - gas_recirc[0] #4:1 ratio of H2 and CO2 [mol/h] minus recycled CO2 and H2 and microbial consumption
                            h2_demand_hr = np.minimum(h2_demand_hr,meth.size_vector*4) #Also limited by methanation reactor size
                            h2_demand_hr = np.where(h2_demand_hr<(meth.flow_min*4),0,h2_demand_hr)
                            h2_demand_hr = np.transpose(h2_demand_hr)
                            h2_demand_hr_kg = h2_demand_hr * 2.02 / 1000 #mol to kg
                            
                            #Also, consider if/when we should fill up the storage ahead of the next day?

                            #Check last hour of previous day
                            if d != 0:
                                if elz_on == 1 or elz_standby == 1:
                                    prev_mode = 1
                                else:
                                    prev_mode = 0
                            
                            # Daily dispatch optimization
                            elz_dispatch = dispatch.p2g_wwtp3(h2_demand=h2_demand_hr_kg, heat_demand=heat.demand_tot[i1:i2], heat_value=heat.dh_price, usable_heat=heat.usable, meth_spec_heat=meth.spec_heat, o2_demand=o2.demand[i1:i2]*32/1000, o2_power=o2.aerator_savings, k_values=pem.k_values, m_values=pem.m_values, grid=grid.spot_price[i1:i2], wind=res.wind_gen, pv=res.pv_gen, elz_max=pem.size_degr, elz_min=pem.min_load*pem.size_degr, elz_eff=pem.n_sys, aux_cons=pem.aux_cons, meth_max=meth.size_mol*2.02*4/1000, meth_min=meth.min_load*meth.size_mol*4*2.02/1000, h2st_max=storage.h2_size, h2st_prev=h2_storage, prev_mode=prev_mode, startup_cost=pem.start_cost, standby_cost=pem.standby_cost, bat_cap=storage.bat_size, bat_eff=storage.bat_eff, bat_prev=bat_storage, meth_el_factor=meth.spec_el, h2o_cons=pem.water_cons, temp=pem.temp, h2o_temp=pem.h2o_temp, biogas=biogas.flow[i1:i2], comp_el_factor=bg_comp.spec_el, elz_startup_time=pem.start_time/60)# wind_cost=wind_lcoe, pv_cost=pv_lcoe)
                                   
                            #Save daily data
                            H2_demand.extend(h2_demand_hr)
                            electrolyzer.extend(elz_dispatch.iloc[:,0])
                            wind_use.extend(elz_dispatch.iloc[:,2])
                            pv_use.extend(elz_dispatch.iloc[:,3])
                            grid_use.extend(elz_dispatch.iloc[:,1])
                            h2_storage_list.extend(elz_dispatch.iloc[:,5])
                            h2_production.extend(elz_dispatch.iloc[:,7])
                            h2_used.extend(elz_dispatch.iloc[:,8])
                            unmet_demand.extend(np.round(elz_dispatch.iloc[:,9],4))
                            electrolyzer_on.extend(elz_dispatch.iloc[:,11])
                            electrolyzer_standby.extend(elz_dispatch.iloc[:,12])
                            electrolyzer_off.extend(elz_dispatch.iloc[:,13])
                            electrolyzer_start.extend(elz_dispatch.iloc[:,14])
                            elz_on = electrolyzer_on[-1]
                            elz_off = electrolyzer_off[-1]
                            elz_standby = electrolyzer_standby[-1]
                            o_prod.extend(elz_dispatch.iloc[:,15])
                            o_use.extend(elz_dispatch.iloc[:,16])
                            h_prod.extend(elz_dispatch.iloc[:,17])
                            h_use.extend(elz_dispatch.iloc[:,18])
                            battery_state.extend(elz_dispatch.iloc[:,19])
                            battery_dis.extend(elz_dispatch.iloc[:,20])
                            h_inc.extend(elz_dispatch.iloc[:,21])
                            grid_inc.extend(elz_dispatch.iloc[:,22])
                            o2_inc.extend(elz_dispatch.iloc[:,23])
                            # if h2_used[-1] > 0:
                            #     meth_on = 1
                            # else:
                            #     meth_on = 0
                            h2_storage = h2_storage_list[-1]
                            bat_storage = battery_state[-1]
                            sys_op.extend(elz_dispatch.iloc[:,24])
                            sb_el.extend(elz_dispatch.iloc[:,25])
                            m_el.extend(elz_dispatch.iloc[:,26])
                            c_el.extend(elz_dispatch.iloc[:,27])
                            e_h_dem.extend(elz_dispatch.iloc[:,28])
                            h2_prod_start.extend(elz_dispatch.iloc[:,29])
                            h2_prod_real.extend(elz_dispatch.iloc[:,30])
                        
                        #Converting lists to arrays
                        H2_demand = np.asarray(H2_demand)
                        if year == 2020:
                            H2_demand = H2_demand.reshape(8784,)
                        else:
                            H2_demand = H2_demand.reshape(8760,)
                        electrolyzer = np.asarray(electrolyzer)
                        h2_storage_list = np.asarray(h2_storage_list)
                        h2_storage_list_prev = np.roll(h2_storage_list, 1)
                        h2_storage_list_prev[0] = 0
                        h2_production = np.asarray(h2_production)
                        h2_used = np.asarray(h2_used)
                        electrolyzer_start = np.asarray(electrolyzer_start)
                            
                        #HOURLY OPERATION
                        #Hydrogen production
                        h2_flow, elz_heat, T_h2_out, o2_flow, h2o_cons, stack_eff, sys_eff, elz_heat_nonnet = comps.electrolyzer(dispatch=electrolyzer, prod=h2_production, aux=pem.aux_cons, temp=pem.temp, h2o_temp=pem.h2o_temp, heat_time=pem.heatup_time, startups=electrolyzer_start, h2o_cons=pem.water_cons, year=year)
                        h2st_in = np.maximum(0,(h2_production)-h2_used) * 1000 / 2.02
                        h2st_out = np.minimum(np.maximum(0,h2_used-h2_production),(h2_storage_list_prev*1000/2.02)) * 1000 / 2.02
                        h2_meth = h2_used * 1000 / 2.02
                        
                        
                        #Flow definitions
                        
                        co2_flow = h2_meth / ((1-meth.microb_cons)*4)
                        p2g_frac = np.divide(co2_flow, biogas.flow[:,1].T, out=np.zeros_like((co2_flow)), where=biogas.flow[:,1]!=0)
                        biogas_in = biogas.flow.T * p2g_frac
                        flared_gas = biogas.flow.T * abs(np.around((1-p2g_frac),6))
                        if year == 2020:
                            flared_gas = [np.zeros(8784,), flared_gas[1], flared_gas[0]] #[H2, CO2, CH4]
                        else:
                            flared_gas = [np.zeros(8760,), flared_gas[1], flared_gas[0]] #[H2, CO2, CH4]
                        
                        # Biogas compression (flow rate in; compressor power and temp. out(?))
                        bg_comp_power, T_bg_comp = comps.compressor(flow=biogas_in.sum(axis=0), temp_in=biogas.temp, p_in=biogas.pres, p_out=meth.pres, n_isen=bg_comp.n_isen, n_motor=bg_comp.n_motor, year=year) #[kWh]

                        # Gas mixing (Biogas, hydrogen, temp in; temp out)
                        inlet_flow, T_inlet = comps.mixer(h2=h2_meth, co2=biogas_in[1], ch4=biogas_in[0], h2_temp=pem.temp, bg_temp=biogas.temp)

                        #Methanation (molar flows, temp. in; molar flows, excess heat, electricity consumption out)
                        meth_outlet_flow, meth_power, meth_heat, h2o_cond1, microbial_co2 = comps.methanation(meth_flow=inlet_flow, rated_flow=meth.flow_max, T=meth.temp, T_in=T_inlet, el_cons=meth.el_cons, n=meth.n, microb_cons=meth.microb_cons)
                        meth_outlet_flow = np.asarray(meth_outlet_flow)
                        h2o_cond1 = np.asarray(h2o_cond1)

                        # Condenser (H2O in, energy out) [Not fully implemented]
                        # cond_outlet_flow, h2o_cond2, T_cond_out, cond_heat, cond_power = comps.condenser(flow=meth_outlet_flow, T_in=meth.temp, year=year)
                        
                        # Gas cleaning (molar flows and temp in; pure stream and recirculation out, energy consumption) [Not fully implemented]
                        # ch4_out, gas_loss, T_out, p_out = comps.membrane(mem_inlet_flow=cond_outlet_flow, T_in=T_cond_out, p_in=meth.pres, year=year)
                        
                        if simulation_details == 'Yes' or simulation_details == 'yes': # Storing detailed results
                            process['H2 demand [mol/h]'] = H2_demand
                            process['Elz dispatch [kWh/h]'] = electrolyzer
                            process['System dispatch [kWh/h]'] = sys_op
                            process['Standby el. [kWh/h]'] = sb_el
                            process['Wind use [kWh/h]'] = wind_use
                            process['Wind gen [kWh/h]'] = res.wind_gen
                            process['PV use [kWh/h]'] = pv_use
                            process['PV gen [kWh/h]'] = res.pv_gen
                            process['Grid use [kWh/h]'] = grid_use
                            process['Unmet demand [kgH2/h]'] = unmet_demand
                            process['H2 production [kg/h]'] = h2_production
                            process['H2 to meth [mol/h]'] = h2_meth
                            process['H2 to storage [mol/h]'] = h2st_in
                            process['H2 from storage [mol/h]'] = h2st_out
                            if storage.h2_size > 0:
                                process['H2 storage [%]'] = (h2_storage_list/(storage.h2_size))*100
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
                            process['CH4 flared [mol/h]'] = flared_gas[2]
                            process['Stack efficiency [%]'] = stack_eff * 100
                            process['System efficiency [%]'] = sys_eff * 100
                            if storage.bat_size > 0:
                                process['Battery state [%]'] = np.array(battery_state) * 100 / storage.bat_size
                                process['Battery discharging [kWh/h]'] = np.array(battery_dis)


                        """ TECHNICAL ANALYSIS """

                        # Gas production
                        ch4_p2g = (meth_outlet_flow[2] - inlet_flow[2]).sum() * tec.ch4_mol / 1000 #[MWh] Annual CH4 production increase from P2G
                        ch4_total = meth_outlet_flow[2].sum() * tec.ch4_mol / 1000
                        # Gas loss
                        flare_frac = (flared_gas[2].sum()/biogas.flow[:,0].sum()) * 100 #[%]
                                                
                        # Stack replacement
                        elz_flh = round(electrolyzer.sum() / (pem.size_degr)) # Full load hours of the electrolyzer
                        if pem.stack_rep > 1000: #counting hours
                            stack_reps = math.floor((elz_flh*tec.lifetime) / pem.stack_rep) #number of stack replacements during project lifetime
                        else:
                            stack_reps = math.floor((tec.lifetime-1) / pem.stack_rep) #number of stack replacements during project lifetime, minus 1 since only 2 replacements are required every ten years for a 30 year lifetime for example

                        #ELECTRICITY USE
                        
                        # Number of cold starts
                        starts = np.zeros(len(electrolyzer))
                        starts[0] = 0
                        for i in range(len(starts)):
                            if i > 0:
                                if electrolyzer[i-1] == 0 and electrolyzer[i] > 0 and electrolyzer_standby[i-1] != 1:
                                    starts[i] = 1
                                else:
                                    starts[i] = 0
                        if simulation_details == 'Yes' or simulation_details == 'yes':
                            process['Elz cold start'] = starts
                        
                        # Excess wind/PV. First using any excess PV (cheaper), then any excess wind if residual load remains
                        if storage.bat_size > 0: # If battery is included
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
                                    bat_loss[i] = bat_in[h] * (1-storage.bat_eff)
        
                                else:
                                    bat_in[i] = np.round((np.maximum((battery_state[h] - battery_state[h-1]),0)),6)
                                    bat_in_pv[i] = np.round(np.minimum((res.pv_gen[h] - pv_use[h]), bat_in[h]),6)
                                    excess_pv[i] = np.round(np.maximum((res.pv_gen[h] - pv_use[h]) - bat_in_pv[h],0),6)
                                    bat_in_wind[i] = np.round(np.minimum((res.wind_gen[h] - wind_use[h]), bat_in[h] - bat_in_pv[h]),6)
                                    excess_wind[i] = np.round(np.maximum((res.wind_gen[h] - wind_use[h]) - bat_in_wind[h],0),6)
                                    bat_in_grid[i] = np.round(bat_in[h] - bat_in_pv[h] - bat_in_wind[h],6)
                                    bat_loss[i] = bat_in[h] * (1-storage.bat_eff)
                            
                            # Renewable electricity fraction
                            res_frac = (wind_use + pv_use + bat_in_wind + bat_in_pv).sum() / sys_op.sum()
                            if wind_size > 0:
                                wind_frac = (wind_use + bat_in_wind).sum() / (wind_use + pv_use + bat_in_wind + bat_in_pv).sum()
                            if pv_size > 0:
                                pv_frac = (pv_use + bat_in_pv).sum() / (wind_use + pv_use + bat_in_wind + bat_in_pv).sum()

                        else:
                            if year == 2020:
                                excess_wind = np.maximum(res.wind_gen - wind_use, np.zeros(8784,))
                                excess_pv = np.maximum(res.pv_gen - pv_use, np.zeros(8784,))
                            else:
                                excess_wind = np.maximum(res.wind_gen - wind_use, np.zeros(8760,))
                                excess_pv = np.maximum(res.pv_gen - pv_use, np.zeros(8760,))
                            
                            # Renewable electricity fraction
                            res_frac = (wind_use + pv_use).sum() / sys_op.sum()
                            if wind_size > 0:
                                wind_frac = wind_use.sum() / (wind_use + pv_use).sum()
                            if pv_size > 0:
                                pv_frac = pv_use.sum() / (wind_use + pv_use).sum()

                        
                        residual_load = sys_op - electrolyzer #np.array(bg_comp_power + meth_power + (electrolyzer_standby * pem.standby_cost * pem.size_degr))
                        curtailment = (excess_wind.sum() + excess_pv.sum()) / 1000 #[MWh]
                        tot_res = (res.wind_gen.sum() + res.pv_gen.sum()) / 1000 #[MWh]
                        if tot_res > 0:
                            curt_frac = curtailment * 100 / tot_res #[%]
                        else:
                            curt_frac = 0
                            
                        if storage.h2_size > 0: # Storage details
                            h2st_cycles = np.round(h2st_in.sum() * 2.02 / (storage.h2_size * 1000)) # Number of full storage cycles
                            h2st_90 = (((h2_storage_list*100/storage.h2_size) > 90) * 1).sum() / len(h2_storage_list) # Percent of time above 90 % full
                        else:
                            h2st_cycles = 0
                            h2st_90 = 0
                        
                        if storage.bat_size > 0: # Battery details
                            bat_cycles = np.round(bat_in.sum() / storage.bat_size) # Number of full storage cycles
                            bat_90 = (((battery_state*100/storage.bat_size) > 90) * 1).sum() / len(battery_state) # Percent of time above 90 % full
                        else:
                            bat_cycles = 0
                            bat_90 = 0
                        
                        # HEAT
                        heat_prod = heat.usable * (elz_heat + meth_heat) # Low-grade heat produced [kWh/h] (excluding condenser heat)
                        heat_prod_out = np.clip(heat_prod, a_min=0, a_max=None) # Exclude potential interal heat consumption by electrolyzer
                        # total_heat_demand = (2*(heat.demand_tot - heat.demand_aux)) + heat.demand_aux # Use for testing thermophilic digestion in the WWTP
                        # for i in range(len(process['O2 out [mol/h]'])):
                            # heat_wwtp.append(min(heat_prod_out[i], heat.demand_tot[i])) #using "heat_prod_out" to avoid negative values at low electrolyzer loads and no methanation
                        heat_wwtp = np.min(heat_prod_out, heat.demand_tot)
                        heat_elz_use = -np.clip(heat_prod, a_min=0, a_max=None) # All interal heat consumption by electrolyzer
                        heat_use_frac = heat_wwtp.sum() * 100 / heat_prod_out.sum() # How much of the output heat is utilized
                        # heat_use_frac_net = (heat_wwtp.sum() / heat_prod.sum()) * 100 #how much of the heat is utilized, including heat consumption by stack
                        # heat_use_frac = ((heat_wwtp.sum()+3037000) / heat_prod_out.sum()) * 100 # Assuming hygienization of co-digestion substrate
                        if heat.demand_tot.sum() > 0:
                            heat_wwtp_use_frac = (heat_wwtp.sum() / heat.demand_tot.sum()) * 100
                        else:
                            heat_wwtp_use_frac = 0

                        # OXYGEN
                        # o2_wwtp = []
                        # for i in range(len(o2_flow)):
                        #     o2_wwtp.append(min(o2_flow[i], o2.demand[i]))
                        o2_wwtp = np.minimum(o2_flow, o2.demand)
                        o2_loss = np.maximum(o2_flow - o2_wwtp,0).sum()
                        o2_use_frac = (o2_wwtp.sum() / o2_flow.sum()) * 100
                        o2_wwtp_use_frac = (o2_wwtp.sum() / o2.demand.sum()) * 100
                        o2_energy_savings = o2.aerator_savings * o2_wwtp.sum() * 32 / 1000 #[kWh]
                        # o2_energy_frac = 100 * o2_energy_savings / electrolyzer.sum() # % of electrolyzer energy input saved

                        """ ECONOMIC ANALYSIS """
                        #Should also move to separate script?

                        #ELECTRICITY
                        if storage.bat_size > 0:
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
                        #Averages
                        # avg_grid_price = grid_cost * 1000 / process['Grid use [kWh/h]'].sum() # [€/MWh]
                        # avg_tot_price = el_cost_curt * 1000 / process['System dispatch [kWh/h]'].sum() # [€/MWh]
                        # avg_el_ch4 = el_cost_curt / ch4_p2g # [€/MWh]
                        
                        # STORAGES
                        # O2 storage [Not implemented]
                        # o2st_CAPEX = o2st_cap * o2st_capex
                        # o2st_OPEX = o2st_cap * (o2st_opex/100)
                        # Heat storage [Not implemented]
                        # heatst_CAPEX = heatst_cap * heatst_capex
                        # heatst_OPEX = heatst_cap * (heatst_opex/100)
                        # Battery
                        bat_CAPEX = storage.bat_size * storage.bat_capex
                        bat_OPEX = storage.bat_size * (storage.bat_opex/100)

                        # Hydrogen costs
                        elz_CAPEX = pem.capex * pem.capex_ref * ((pem.size/pem.capex_ref)**pem.scaling) # Electrolyzer CAPEX with scaling
                        elz_OPEX = pem.opex * 0.01 * elz_CAPEX # Electrolyzer fixed OPEX
                        h2o_opex = pem.water_cost * h2o_cons.sum() * 18.02 / (1000*997) # Water [€/m3 * mol * g/mol / (1000*kg/m3)]
                        stack_COST = pem.stack_cost * elz_CAPEX # Total cost of stack replacements
                        h2st_CAPEX = storage.h2_size * storage.h2st_capex # H2 storage CAPEX
                        h2st_OPEX = storage.h2st_opex * 0.01 * h2st_CAPEX # H2 storage OPEX
                        H2_CAPEX = elz_CAPEX + h2st_CAPEX # Total hydrogen CAPEX
                        H2_OPEX = elz_OPEX + h2o_opex + h2st_OPEX # Total hydrogen OPEX
                        H2_STACK = stack_COST # Stack replacement costs

                        # Biogas costs
                        biogas_loss_cost = biogas.lcoe * flared_gas[2].sum() * tec.ch4_mol / 1000
                        biogas_cost_tot = biogas.lcoe * biogas.flow[:,0].sum() * tec.ch4_mol / 1000
                        
                        # Methanation costs
                        meth_CAPEX = meth.capex * meth.capex_ref * ((meth.size/meth.capex_ref)**meth.scaling) # Methanation CAPEX with scaling per MWCH4 out
                        meth_OPEX = meth.opex * 0.01 * meth_CAPEX # Methanarion fixed OPEX
                        bg_comp_capex = 30000*(bg_comp.size**0.48) # Biogas compressor CAPEX
                        bg_comp_opex = bg_comp.opex * 0.01 * bg_comp_capex # Biogas compressor OPEX
                        METH_CAPEX = meth_CAPEX + bg_comp_capex # Total methanation CAPEX
                        METH_OPEX = meth_OPEX + bg_comp_opex # Total methanation OPEX

                        # By-product integration costs
                        heat_size = (pem.heat_max + meth.heat_max) * heat.usable # [kW] Size of heat system
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
                        
                        co2_opex = tec.co2_cost * sum(co2_in) * 44.01 / (1000*1000) # CO2 cost (assumed zero)
                        
                        # Overall costs
                        CAPEX = H2_CAPEX + METH_CAPEX + BY_CAPEX + bat_CAPEX
                        OPEX = H2_OPEX + METH_OPEX + BY_OPEX + el_cost + bat_OPEX + co2_opex
                        OPEX_curt = H2_OPEX + METH_OPEX + BY_OPEX + el_cost_curt + bat_OPEX + co2_opex
                        # Installation cost
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
                        BY_INCOME = o2_income + heat_income # Total by-product income

                        """ KPI calculations """

                        #ECONOMIC KPIs
                        #LCOE (discounting stack replacement)
                        if pem.stack_rep > 1000: #hours
                            if stack_reps == 1:
                                rep_years = np.array([(math.ceil(pem.stack_rep/elz_flh))])
                            elif stack_reps == 2:
                                rep_years = np.array([(math.ceil(pem.stack_rep/elz_flh)), (math.ceil(2*pem.stack_rep/elz_flh))])
                            elif pem.stack_reps == 3:
                                rep_years = np.array([(math.ceil(pem.stack_rep/elz_flh)), (math.ceil(2*pem.stack_rep/elz_flh)), (math.ceil(3*pem.stack_rep/elz_flh))])
                        else:
                            if stack_reps == 1:
                                rep_years = np.array([pem.stack_rep])
                            elif stack_reps == 2:
                                rep_years = np.array([pem.stack_rep, pem.stack_rep*2])
                            elif stack_reps == 3:
                                rep_years = np.array([pem.stack_rep, pem.stack_rep*2, pem.stack_rep*3])
                            elif stack_reps == 0:
                                rep_years = np.array([0])
                        
                        lcoe = kpis.lcoe(opex=OPEX-BY_INCOME, capex=CAPEX, stack=H2_STACK, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]

                        #Net present value (discounting stack replacement)
                        # INCOME_GAS = BY_INCOME + (gas_price * ch4_p2g) #[€] Income including gas sales
                        # npv = kpis.npv(opex=OPEX_tot, income=INCOME_GAS, capex=CAPEX, stack=H2_STACK, dr=tec.discount, lt=tec.lifetime, stack_reps=stack_reps, rep_years=rep_years) #[€]

                        #LCOP2G (including lost biogas LCOE)
                        lcop2g = kpis.lcoe(opex=OPEX_tot-BY_INCOME, capex=CAPEX, stack=H2_STACK, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        lcop2g_curt = kpis.lcoe(opex=OPEX_tot_curt-BY_INCOME, capex=CAPEX, stack=H2_STACK, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]

                        #Minimum selling price
                        msp = kpis.lcoe(opex=OPEX_msp-BY_INCOME, capex=CAPEX, stack=H2_STACK, dr=tec.discount, lt=tec.lifetime, ch4=ch4_total, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        msp_no_curt = kpis.lcoe(opex=OPEX_msp_nocurt-BY_INCOME, capex=CAPEX, stack=H2_STACK, dr=tec.discount, lt=tec.lifetime, ch4=ch4_total, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        #Comparison to amine scrubbing (Check Vo et al., Ardolino et al., Energiforsk 2016, Angelidaki et al. 2018)
                        #Is it really reasonable to assume the maximum value?
                        amine_flow_rate = 1200 #[Nm3/hr]
                        amine_scrubber_CAPEX = 2500 * amine_flow_rate * tec.nm3_mol #[€]
                        amine_scrubber_el_cost = sum((process['CH4 out [mol/h]'] - process['Meth CH4 in [mol/h]']) * tec.nm3_mol * 0.14 * grid.spot_price / 1000) #[€]
                        amine_scrubber_heat_cost = sum((process['CH4 out [mol/h]'] - process['Meth CH4 in [mol/h]']) * tec.nm3_mol * 0.55 * np.mean(heat.dh_price) / 1000) #[€]
                        amine_scrubber_opex_fix = amine_scrubber_CAPEX * 0.04
                        amine_scrubber_OPEX = amine_scrubber_el_cost + amine_scrubber_heat_cost + amine_scrubber_opex_fix
                        # npv_rep = kpis.npv(opex=OPEX_tot-amine_scrubber_OPEX, income=INCOME_GAS, capex=CAPEX-amine_scrubber_CAPEX, stack=H2_STACK, dr=tec.discount, lt=tec.lifetime, stack_reps=stack_reps, rep_years=rep_years) #[€]
                        lcoe_amine = kpis.lcoe(opex=amine_scrubber_OPEX, capex=amine_scrubber_CAPEX, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_total-ch4_p2g, stack_reps=0, rep_years=0) #[€/MWh of CH4]
                        msp_amine = kpis.lcoe(opex=amine_scrubber_OPEX+biogas_cost_tot, capex=amine_scrubber_CAPEX, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_total-ch4_p2g, stack_reps=0, rep_years=0) #[€/MWh of CH4]


                        #By-product analysis
                        npv_o2 = kpis.npv(opex=o2_integration_OPEX, income=o2_income, capex=o2_integration_CAPEX, stack=0, dr=tec.discount, lt=tec.lifetime, stack_reps=0, rep_years=rep_years) #[€]
                        npv_heat = kpis.npv(opex=heat_integration_OPEX, income=heat_income, capex=heat_integration_CAPEX, stack=0, dr=tec.discount, lt=tec.lifetime, stack_reps=0, rep_years=rep_years) #[€]
                        # lcop2g_noo2 = kpis.lcoe(opex=OPEX_tot-heat_income-o2_integration_OPEX, capex=CAPEX-o2_integration_CAPEX, stack=H2_STACK, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        lcop2g_noo2_curt = kpis.lcoe(opex=OPEX_tot_curt-heat_income-o2_integration_OPEX, capex=CAPEX-o2_integration_CAPEX, stack=H2_STACK, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        msp_noo2_curt = kpis.lcoe(opex=OPEX_msp-heat_income-o2_integration_OPEX, capex=CAPEX-o2_integration_CAPEX, stack=H2_STACK, dr=tec.discount, lt=tec.lifetime, ch4=ch4_total, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        # lcop2g_noheat = kpis.lcoe(opex=OPEX_tot-o2_income-heat_integration_OPEX, capex=CAPEX-heat_integration_CAPEX, stack=H2_STACK, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        lcop2g_noheat_curt = kpis.lcoe(opex=OPEX_tot_curt-o2_income-heat_integration_OPEX, capex=CAPEX-heat_integration_CAPEX, stack=H2_STACK, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        msp_noheat_curt = kpis.lcoe(opex=OPEX_msp-o2_income-heat_integration_OPEX, capex=CAPEX-heat_integration_CAPEX, stack=H2_STACK, dr=tec.discount, lt=tec.lifetime, ch4=ch4_total, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        # lcop2g_nobys = kpis.lcoe(opex=OPEX_tot-o2_integration_OPEX-heat_integration_OPEX, capex=CAPEX-o2_integration_CAPEX-heat_integration_CAPEX, stack=H2_STACK, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        lcop2g_nobys_curt = kpis.lcoe(opex=OPEX_tot_curt-o2_integration_OPEX-heat_integration_OPEX, capex=CAPEX-o2_integration_CAPEX-heat_integration_CAPEX, stack=H2_STACK, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        msp_nobys_curt = kpis.lcoe(opex=OPEX_msp-o2_integration_OPEX-heat_integration_OPEX, capex=CAPEX-o2_integration_CAPEX-heat_integration_CAPEX, stack=H2_STACK, dr=tec.discount, lt=tec.lifetime, ch4=ch4_total, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        # lcop2g_diff = lcop2g_nobys - lcop2g
                        lcop2g_diff_curt = lcop2g_nobys_curt - lcop2g_curt
                        msp_diff_curt = msp_nobys_curt - msp
                        # lcop2g_diff_rel = lcop2g_diff / lcop2g
                        lcop2g_diff_rel_curt = lcop2g_diff_curt / lcop2g_curt
                        msp_diff_rel_curt = msp_diff_curt / msp
                        # lcop2g_diff_rel_o2 = (lcop2g-lcop2g_noheat) / lcop2g
                        lcop2g_diff_rel_o2_curt = (lcop2g_curt-lcop2g_noheat_curt) / lcop2g_curt
                        msp_diff_rel_o2_curt = (msp-msp_noheat_curt) / msp

                        #Comparison to fossil use? I.e. transport etc. Include ETS price

                        #TECHNICAL KPIs
                        #System efficiency
                        tot_energy_cons = (process['Elz dispatch [kWh/h]'] + + process['Biogas comp [kWh/h]'] + process['Meth el [kWh/h]'] + process['Standby el. [kWh/h]']).sum()
                        #Methane/Electricity
                        n_gas = (ch4_p2g * 1000) / tot_energy_cons

                        #Including by-products
                        n_tot = ((ch4_p2g * 1000) + sum(heat_wwtp)) / tot_energy_cons
                        # n_tot = ((ch4_p2g * 1000) + (heat_prod.sum() * heat_frac_use)) / tot_energy_cons
                        n_tot_o2 = ((ch4_p2g * 1000) + sum(heat_wwtp) + o2_energy_savings) / tot_energy_cons
                        # n_tot_o2 = ((ch4_p2g * 1000) + (heat_prod.sum() * heat_frac_use) + o2_energy_savings) / tot_energy_cons
                        n_biomethane = ((ch4_total * 1000) + sum(heat_wwtp) + o2_energy_savings) / (tot_energy_cons + ((ch4_total-ch4_p2g)*1000))
                        
                        #Theoretical total (using all produced heat)
                        n_theory = ((ch4_p2g * 1000) + (process['Elz heat [kWh/h]'] + process['Meth heat [kWh/h]']).sum()) / tot_energy_cons
                        n_theory_o2 = ((ch4_p2g * 1000) + (process['Elz heat [kWh/h]'] + process['Meth heat [kWh/h]']).sum() + o2_energy_savings) / tot_energy_cons
                        n_max_o2 = ((ch4_p2g * 1000) + ((process['Elz heat [kWh/h]'] + process['Meth heat [kWh/h]']).sum()*heat.usable) + o2_energy_savings) / tot_energy_cons

                        #Upgrading efficiency (including biogas)
                        n_upgrade = ((ch4_total * 1000) + sum(heat_wwtp) + o2_energy_savings) / (tot_energy_cons + (biogas_in[0,:].sum()*tec.ch4_mol))
                        
                        #ENVIRONMENTAL KPIs
                        aef_ems = ((process['Grid use [kWh/h]'] * grid.aefs / 1000).sum() + (process['Wind use [kWh/h]'] * res.wind_efs / 1000).sum() + (process['PV use [kWh/h]'] * res.pv_efs / 1000).sum()) / ch4_p2g #[kgCO2/MWhCH4]
                        mef_ems = ((process['Grid use [kWh/h]'] * grid.mefs / 1000).sum() + (process['Wind use [kWh/h]'] * res.wind_efs / 1000).sum() + (process['PV use [kWh/h]'] * res.pv_efs / 1000).sum()) / ch4_p2g #[kgCO2/MWhCH4]
                        aef_avg = (aef_ems*ch4_p2g) / ((process['Grid use [kWh/h]'] / 1000).sum())
                        mef_avg = (mef_ems*ch4_p2g) / ((process['Grid use [kWh/h]'] / 1000).sum())

                        
                        #Emission reductions from by-product use
                        aef_ems_red_heat = (sum(heat_wwtp) * heat.ems / 1000) / ch4_p2g #[kgCO2/MWhCH4]
                        mef_ems_red_heat = (sum(heat_wwtp) * heat.ems_marginal / 1000) / ch4_p2g #[kgCO2/MWhCH4]
                        # aef_ems_red_heat = ((heat_prod.sum() * heat_frac_use) * heat.ems / 1000) / ch4_p2g #[kgCO2/MWhCH4]
                        # mef_ems_red_heat = ((heat_prod.sum() * heat_frac_use) * heat.ems_marginal / 1000) / ch4_p2g #[kgCO2/MWhCH4]
                        aef_red_o2 = ((o2.aerator_savings * o2_wwtp * 32 / 1000) * grid.aefs).sum() / (1000*ch4_p2g) #[kgCO2/MWhCH4]
                        mef_red_o2 = ((o2.aerator_savings * o2_wwtp * 32 / 1000) * grid.mefs).sum() / (1000*ch4_p2g) #[kgCO2/MWhCH4]
                        
                        #Emission increase from biogas losses
                        bgloss_ems_increase = (biogas.ef * process['CH4 flared [mol/h]'].sum() * tec.ch4_mol) / (1000*ch4_p2g) #[kgCO2/MWhCH4]
                        
                        #Net system climate impact
                        aef_net = aef_ems - aef_red_o2 - aef_ems_red_heat + bgloss_ems_increase
                        mef_net = mef_ems - mef_red_o2 - mef_ems_red_heat + bgloss_ems_increase
                        
                        if run_type == "simulation":
                            # Cost breakdown table
                            # Data
                            total = kpis.lcoe(opex=OPEX_tot_curt, capex=CAPEX, stack=H2_STACK, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                            elz_lcoe = kpis.lcoe(opex=elz_OPEX, capex=elz_CAPEX, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                            stack_rep_lcoe = kpis.lcoe(opex=0, capex=0, stack=stack_COST, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) * 100 / total #[€/MWh of CH4]
                            water_lcoe = kpis.lcoe(opex=h2o_opex, capex=0, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                            h2st_lcoe = kpis.lcoe(opex=h2st_OPEX, capex=h2st_CAPEX, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                            meth_lcoe = kpis.lcoe(opex=meth_OPEX, capex=meth_CAPEX, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                            comp_lcoe = kpis.lcoe(opex=bg_comp_opex, capex=bg_comp_capex, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                            heat_lcoe = kpis.lcoe(opex=heat_integration_OPEX, capex=heat_integration_CAPEX, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                            o2_lcoe = kpis.lcoe(opex=o2_integration_OPEX, capex=o2_integration_CAPEX, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                            grid_lcoe = kpis.lcoe(opex=grid_cost, capex=0, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                            pv_lcoe1 = kpis.lcoe(opex=pv_cost, capex=0, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                            wind_lcoe1 = kpis.lcoe(opex=wind_cost, capex=0, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                            bg_loss_lcoe = kpis.lcoe(opex=biogas_loss_cost, capex=0, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total
                            install_lcoe = kpis.lcoe(opex=0, capex=INSTALL, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total
                            curt_lcoe1 = kpis.lcoe(opex=curt_cost, capex=0, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                            o2_income_lcoe = kpis.lcoe(opex=-o2_income, capex=0, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                            heat_income_lcoe = kpis.lcoe(opex=-heat_income, capex=0, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
    
                            # Table
                            cost_breakdown['{} MW'.format(elz_size)] = [elz_lcoe,stack_rep_lcoe,water_lcoe,h2st_lcoe,meth_lcoe,comp_lcoe,heat_lcoe,o2_lcoe,install_lcoe,bg_loss_lcoe,grid_lcoe,pv_lcoe1,wind_lcoe1,curt_lcoe1,o2_income_lcoe,heat_income_lcoe,lcop2g_curt]

                        if run_type == "optimization":
                            #Saving optimization results
                            # result_series = pd.Series([lcoe, npv, msp, n_gas, n_tot, n_tot_o2, aef_net, mef_net, flare_frac, o2_use_frac, o2_wwtp_use_frac, heat_use_frac, heat_wwtp_use_frac, res_frac])
                            # results = pd.concat([results,result_series], axis=1)
                            # results.columns = ['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size,meth_scale,h2st_size,wind_size,pv_size,bat_size)]
                            results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(pem.size/1000,meth.size/1000,storage.h2_size,wind_size/1000,pv_size/1000,storage.bat_size)] = [lcop2g_curt, lcop2g, msp, msp_no_curt, lcoe, n_gas, n_tot, n_tot_o2, aef_net, mef_net, sum(starts), sum(electrolyzer_standby), elz_flh, flare_frac, o2_use_frac, o2_wwtp_use_frac, heat_use_frac, heat_wwtp_use_frac, res_frac, lcop2g_diff_curt, lcop2g_diff_rel_curt, msp_diff_curt, msp_diff_rel_curt, npv_o2, npv_heat]
                            count = count + 1
                            print('{}/{} simulations performed'.format(count,sims))
    


# Save KPIS in a dictionary instead? And print that?
if run_type == "single":
    #PRINTING
    table_kpi = [['LCOP2G', 'MSP', 'Gas eff.', 'O2 eff.', 'AEF net', 'MEF net', 'Loss %', 'RES [%]'], \
                  [lcop2g_curt, msp, n_gas, n_tot_o2, aef_net, mef_net, flare_frac, res_frac]]
    print(tabulate(table_kpi, headers='firstrow', tablefmt='fancy_grid'))

    table_by = [['O2 util. [%]', 'O2 dem. [%]', 'Heat util. [%]', '% Heat dem. [%]'], \
          [o2_use_frac, o2_wwtp_use_frac, heat_use_frac, heat_wwtp_use_frac]]
    print(tabulate(table_by, headers='firstrow', tablefmt='fancy_grid'))

    # end = time.time()

    # #Subtract Start Time from The End Time
    # total_time = end - start
    # print("\n"+ str(total_time))    

    # #Costs and energy consumption
    # # Data
    # r = [0,0.25]#,2,3,4]
    # #costs
    # total = kpis.lcoe(opex=OPEX_tot, capex=CAPEX, stack=H2_STACK, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
    # elz_lcoe = kpis.lcoe(opex=elz_OPEX, capex=elz_CAPEX, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
    # stack_rep_lcoe = kpis.lcoe(opex=0, capex=0, stack=stack_COST, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) * 100 / total #[€/MWh of CH4]
    # water_lcoe = kpis.lcoe(opex=h2o_opex, capex=0, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
    # h2st_lcoe = kpis.lcoe(opex=h2st_OPEX, capex=h2st_CAPEX, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
    # meth_lcoe = kpis.lcoe(opex=meth_OPEX, capex=meth_CAPEX, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
    # comp_lcoe = kpis.lcoe(opex=bg_comp_opex, capex=bg_comp_capex, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
    # heat_lcoe = kpis.lcoe(opex=heat_integration_OPEX, capex=heat_integration_CAPEX, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
    # o2_lcoe = kpis.lcoe(opex=o2_integration_OPEX, capex=o2_integration_CAPEX, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
    # bat_lcoe = kpis.lcoe(opex=bat_OPEX, capex=bat_CAPEX, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
    # grid_lcoe = kpis.lcoe(opex=grid_cost, capex=0, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
    # pv_lcoe = kpis.lcoe(opex=pv_cost, capex=0, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
    # wind_lcoe = kpis.lcoe(opex=wind_cost, capex=0, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
    # bg_loss_lcoe = kpis.lcoe(opex=biogas_loss_cost, capex=0, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total
    # install_lcoe = kpis.lcoe(opex=0, capex=INSTALL, stack=0, dr=tec.discount, lt=tec.lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total
    # #energy
    # elz_energy = sum(process['Elz dispatch [kWh/h]'] + process['Standby el. [kWh/h]']) *100 / tot_energy_cons
    # meth_energy = sum(process['Biogas comp [kWh/h]']) * 100 / tot_energy_cons
    # comp_energy = sum(process['Meth el [kWh/h]']) * 100 / tot_energy_cons
    
    # cost_data = pd.DataFrame({'Electrolyser': [elz_energy,elz_lcoe], 'Stack replacement': [0,stack_rep_lcoe], 'Water': [0,water_lcoe], 'H$_2$ storage': [0,h2st_lcoe], \
    #                           'Methanation': [meth_energy,meth_lcoe], 'Compressor': [comp_energy,comp_lcoe], 'Heat integration': [0,heat_lcoe], \
    #                                   'Oxygen integration': [0,o2_lcoe], 'Battery': [0,bat_lcoe], 'Electricity (grid)': [0,grid_lcoe], \
    #                                       'Electricity (PV)': [0,pv_lcoe],  'Electricity (wind)': [0,wind_lcoe], 'Biogas loss': [0,bg_loss_lcoe], \
    #                                           'Installation': [0,install_lcoe]})
    # cost_data2 = pd.DataFrame({'Electrolyser': [elz_lcoe], 'Stack replacement': [stack_rep_lcoe], 'Water': [water_lcoe], 'H$_2$ storage': [h2st_lcoe], \
    #                           'Methanation': [meth_lcoe], 'Compressor': [comp_lcoe], 'Heat integration': [heat_lcoe], \
    #                                   'Oxygen integration': [o2_lcoe], 'Battery': [bat_lcoe], 'Electricity (grid)': [grid_lcoe], \
    #                                       'Electricity (PV)': [pv_lcoe],  'Electricity (wind)': [wind_lcoe], 'Biogas loss': [bg_loss_lcoe], \
    #                                           'Installation': [install_lcoe]})
     
    # # plot
    # barWidth = 0.2
    # names = ('Electricity', 'Cost')#,'C','D','E')
    # # Create bars
    # plt.barh(r, cost_data['Electrolyser'], color='steelblue', edgecolor='white', height=barWidth, label="Electrolyser", lw=0)
    # plt.barh(r, cost_data['Stack replacement'], left=cost_data['Electrolyser'], color='lightsteelblue', edgecolor='white', height=barWidth, label="Stack replacement", lw=0)
    # plt.barh(r, cost_data['Water'], left=[i+j for i,j in zip(cost_data['Electrolyser'], cost_data['Stack replacement'])], color='royalblue', edgecolor='white', height=barWidth, label="Water", lw=0)
    # plt.barh(r, cost_data['H$_2$ storage'], left=[i+j+k for i,j,k in zip(cost_data['Electrolyser'], cost_data['Stack replacement'], cost_data['Water'])], color='slategrey', edgecolor='white', height=barWidth, label="H$_2$ storage", lw=0)
    # plt.barh(r, cost_data['Methanation'], left=[i+j+k+l for i,j,k,l in zip(cost_data['Electrolyser'], cost_data['Stack replacement'], cost_data['Water'], cost_data['H$_2$ storage'])], color='mediumseagreen', edgecolor='white', height=barWidth, label="Methanation", lw=0)
    # plt.barh(r, cost_data['Compressor'], left=[i+j+k+l+m for i,j,k,l,m in zip(cost_data['Electrolyser'], cost_data['Stack replacement'], cost_data['Water'], cost_data['H$_2$ storage'], cost_data['Methanation'])], color='springgreen', edgecolor='white', height=barWidth, label="Compressor", lw=0)
    # plt.barh(r, cost_data['Heat integration'], left=[i+j+k+l+m+n for i,j,k,l,m,n in zip(cost_data['Electrolyser'], cost_data['Stack replacement'], cost_data['Water'], cost_data['H$_2$ storage'], cost_data['Methanation'], cost_data['Compressor'])], color='lightcoral', edgecolor='white', height=barWidth, label="Heat integration", lw=0)
    # plt.barh(r, cost_data['Oxygen integration'], left=[i+j+k+l+m+n+o for i,j,k,l,m,n,o in zip(cost_data['Electrolyser'], cost_data['Stack replacement'], cost_data['Water'], cost_data['H$_2$ storage'], cost_data['Methanation'], cost_data['Compressor'], cost_data['Heat integration'])], color='mediumpurple', edgecolor='white', height=barWidth, label="O$_2$ integration", lw=0)
    # plt.barh(r, cost_data['Battery'], left=[i+j+k+l+m+n+o+p for i,j,k,l,m,n,o,p in zip(cost_data['Electrolyser'], cost_data['Stack replacement'], cost_data['Water'], cost_data['H$_2$ storage'], cost_data['Methanation'], cost_data['Compressor'], cost_data['Heat integration'], cost_data['Oxygen integration'])], color='goldenrod', edgecolor='white', height=barWidth, label="Battery", lw=0)
    # plt.barh(r, cost_data['Electricity (grid)'], left=[i+j+k+l+m+n+o+p+q for i,j,k,l,m,n,o,p,q in zip(cost_data['Electrolyser'], cost_data['Stack replacement'], cost_data['Water'], cost_data['H$_2$ storage'], cost_data['Methanation'], cost_data['Compressor'], cost_data['Heat integration'], cost_data['Oxygen integration'], cost_data['Battery'])], color='gold', edgecolor='white', height=barWidth, label="Electricity (grid)", lw=0)
    # plt.barh(r, cost_data['Electricity (PV)'], left=[i+j+k+l+m+n+o+p+q+r for i,j,k,l,m,n,o,p,q,r in zip(cost_data['Electrolyser'], cost_data['Stack replacement'], cost_data['Water'], cost_data['H$_2$ storage'], cost_data['Methanation'], cost_data['Compressor'], cost_data['Heat integration'], cost_data['Oxygen integration'], cost_data['Battery'], cost_data['Electricity (grid)'])], color='gold', hatch='//', edgecolor='k', height=barWidth, label="Electricity (PV)", lw=0)
    # plt.barh(r, cost_data['Electricity (wind)'], left=[i+j+k+l+m+n+o+p+q+r+s for i,j,k,l,m,n,o,p,q,r,s in zip(cost_data['Electrolyser'], cost_data['Stack replacement'], cost_data['Water'], cost_data['H$_2$ storage'], cost_data['Methanation'], cost_data['Compressor'], cost_data['Heat integration'], cost_data['Oxygen integration'], cost_data['Battery'], cost_data['Electricity (grid)'], cost_data['Electricity (PV)'])], color='gold', hatch='..', edgecolor='k', height=barWidth, label="Electricity (wind)", lw=0)
    # plt.barh(r, cost_data['Biogas loss'], left=[i+j+k+l+m+n+o+p+q+r+s+t for i,j,k,l,m,n,o,p,q,r,s,t in zip(cost_data['Electrolyser'], cost_data['Stack replacement'], cost_data['Water'], cost_data['H$_2$ storage'], cost_data['Methanation'], cost_data['Compressor'], cost_data['Heat integration'], cost_data['Oxygen integration'], cost_data['Battery'], cost_data['Electricity (grid)'], cost_data['Electricity (PV)'], cost_data['Electricity (wind)'])], color='sandybrown', edgecolor='white', height=barWidth, label="Biogas loss", lw=0)
    # plt.barh(r, cost_data['Installation'], left=[i+j+k+l+m+n+o+p+q+r+s+t+u for i,j,k,l,m,n,o,p,q,r,s,t,u in zip(cost_data['Electrolyser'], cost_data['Stack replacement'], cost_data['Water'], cost_data['H$_2$ storage'], cost_data['Methanation'], cost_data['Compressor'], cost_data['Heat integration'], cost_data['Oxygen integration'], cost_data['Battery'], cost_data['Electricity (grid)'], cost_data['Electricity (PV)'], cost_data['Electricity (wind)'], cost_data['Biogas loss'])], color='darkgrey', edgecolor='white', height=barWidth, label="Installation", lw=0)
    # # Custom x axis
    # plt.yticks(r, names)
    # # plt.ylabel("System configuration")
    # plt.xlim(0,100)
    # # Add a legend
    # plt.legend(loc='upper left', bbox_to_anchor=(1,1.03), ncol=1)
     
    # # Show graphic
    # plt.show()
    
    #BY-PRODUCTS
    #Income
    # if heat_income < 0:
    #     heat_income = 0
    # plt.pie([heat_income, o2_income])
    # plt.legend(['Internal heat', 'O2 sales'])
    # plt.show()
    
    #Production and demand
    # x = range(0,8760)
    # #Oxygen
    # plt.plot(x, process['O2 out [mol/h]'], label='O2 prod. [mol/h]')
    # plt.plot(x, process['O2 WWTP [mol/h]'], label='O2 demand [mol/h]')
    # plt.legend()
    # plt.show()
    
    # #Heat
    # plt.plot(x, heat_prod_out, label='Heat prod. [kW]')
    # plt.plot(x, heat.demand_tot, label='Total heat demand [kW]')
    # plt.plot(x, heat.demand_bg, label='Digestion heat demand [kW]')
    # plt.legend()
    # plt.show()


    #Dispatch 
    #For poster
    # x1 = 2301+48#500 #starting hour
    # x2 = x1+24#600 #one week later
    # d1 = x1 - x1%24 + 24 #start of the first new day
    # elzload_plot = (h2_flow[x1:x2]*2.02/1000)*100/elz_h2_max
    # h2st_plot = np.array(process['H2 storage [%]'][x1:x2])
    # ep_plot = grid.spot_price[x1:x2]
    # h2dem_plot = (H2_demand[x1:x2]*2.02/1000)*100 / elz_h2_max
    # o2prod_plot = np.array(process['O2 out [mol/h]'][x1:x2] / 1000)
    # x = range(0,x2-x1)
    
    # fig, ax1 = plt.subplots()
    # l1 = ax1.plot(x,elzload_plot, color='steelblue', label='Electrolyser')
    # l2 = ax1.fill_between(x,h2st_plot, color='lightskyblue', label='Hydrogen storage')
    # ax3 = ax1.twinx()
    # l3 = ax3.plot(x,ep_plot, color='darkorange', label='Electricity price')
    # l4 = ax1.plot(x,h2dem_plot, color='seagreen', label='H$_2$ demand')
    # ax1.set_xlabel('Hour')
    # ax1.set_ylabel('Load [%]', color='k')
    # ax3.set_ylabel('El. price [€/MWh]', color='darkorange')
    # # lns = l1+l2+l3+l4
    # # labs = [l.get_label() for l in lns]
    # # ax1.legend(lns, labs, loc='upper left')
    # #Indicating days
    # # for i in range(math.floor((x2-x1)/24)):
    # #     if d1 == x1:
    # #         if i % 2 == 1:
    # #             ax1.axvspan(d1-x1+(24*i),d1-x1+((24*(i+1))-1), facecolor='0.2', alpha=0.2, zorder=-1)
    # #     elif d1 != x1:
    # #         if i % 2 == 0:
    # #             ax1.axvspan(d1-x1+(24*i),d1-x1+((24*(i+1))-1), facecolor='0.2', alpha=0.2, zorder=-1)
    # # ax3.set_ylim(0,20000)
    # ax1.set_ylim(0,110)
    # ax3.set_ylim(0,110)
    # ax1.set_xlim(0,x2-x1-1)
    # plt.show()
    
    # #DEMAND AND SUPPLY MISMATCH AT LOWER LOADS, LIKELY DUE TO NOT CONSIDERING PART-LOAD EFFICIENCY
    # #colors
    # teal_dark = sns.dark_palette("teal",5)
    # teal_light = sns.light_palette("teal",5)
    # orange_dark = sns.dark_palette("orange",5)
    
    # x1 = 2208#500 #starting hour
    # x2 = x1+(24*7)#600 #one week later
    # d1 = x1 - x1%24 + 24 #start of the first new day
    # elzload_plot = electrolyzer[x1:x2]*100/(elz_size_degr*1000)
    # h2prod_plot = (h2_flow[x1:x2]*2.02/1000)*100/elz_h2_max
    # elz_plot = electrolyzer[x1:x2]
    # h2st_plot = np.array(process['H2 storage [%]'][x1-1:x2-1])
    # ep_plot = grid.spot_price[x1:x2]
    # bg_plot = biogas.flow[x1:x2,1]
    # h2dem_plot = (H2_demand[x1:x2]*2.02/1000)*100 / elz_h2_max
    # h2use_plot = h2_used[x1:x2]*100/elz_h2_max
    # htdem_plot = np.array(np.maximum(heat_demand_tot[x1:x2]-(process['Meth heat [kWh/h]'][x1:x2]*usable_heat),0))
    # htprod_plot = usable_heat*elz_heat[x1:x2]
    # o2dem_plot = np.array(process['O2 WWTP [mol/h]'][x1:x2] * 32 / 1000)
    # o2prod_plot = np.array(process['O2 out [mol/h]'][x1:x2] * 32 / 1000)
    # x = range(0,x2-x1)
    
    # fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
    # l1 = ax1.plot(x,h2prod_plot, color=teal_dark[4], label='H$_2$ production')
    # l2 = ax1.fill_between(x,h2st_plot, color=teal_light[1], label='Hydrogen storage')
    # ax3 = ax1.twinx()
    # l3 = ax3.plot(x,ep_plot, color=orange_dark[4], label='Electricity price')
    # l4 = ax1.plot(x,h2dem_plot, color=teal_dark[2], label='H$_2$ demand', ls='--')
    # ax2.set_xlabel('Hour')
    # ax1.set_ylabel('Load [%]', color='k')
    # ax3.set_ylabel('El. price [€/MWh]', color=orange_dark[4])
    # # lns = l1+l2+l3+l4
    # # labs = [l.get_label() for l in lns]
    # # ax1.legend(lns, labs, loc='upper left')
    # #Indicating days
    # for i in range(math.floor((x2-x1)/24)):
    #     if d1 == x1:
    #         if i % 2 == 1:
    #             ax1.axvspan(d1-x1+(24*i),d1-x1+((24*(i+1))), facecolor='0.2', alpha=0.2, zorder=-1)
    #     elif d1 != x1:
    #         if i % 2 == 0:
    #             ax1.axvspan(d1-x1+(24*i),d1-x1+((24*(i+1))), facecolor='0.2', alpha=0.2, zorder=-1)
    # # ax3.set_ylim(0,20000)
    # ax1.set_ylim(0,120)
    # ax3.set_ylim(0,250)
    # ax1.set_xlim(0,x2-x1-1)
    # plt.text(0.00,1.03,'(a)', fontsize=10, transform=ax1.transAxes)
    # plt.text(0.00,-0.17,'(b)', fontsize=10, transform=ax1.transAxes)
    # # plt.show()
    
    # #Axis colors
    # ax3.spines['right'].set_color(orange_dark[4])
    # ax3.xaxis.label.set_color(orange_dark[4])
    # ax3.tick_params(axis='y', colors=orange_dark[4])
    
    # #By-products
    # # fig, ax1 = plt.subplots()
    # # l1 = ax1.plot(x,elz_plot, color='blue', label='Electrolyser')
    # l2 = ax2.plot(x,htdem_plot, color='indianred', ls='--', label='Heat demand')
    # l3 = ax2.plot(x,htprod_plot, color='indianred', label='Heat production')
    # # ax1.plot(x,o2_demand[0:50], color='purple')
    # # ax2 = ax1.twinx()
    # # ax2.spines.right.set_position(("axes", 1.2))
    # # l4 = ax2.plot(x,grid.spot_price[x1:x2], color='darkorange', label='Electricity price')
    # ax4 = ax2.twinx()
    # l5 = ax4.plot(x,o2dem_plot, color='mediumpurple', ls='--', label='O2 demand')
    # l6 = ax4.plot(x,o2prod_plot, color='mediumpurple', label='O2 production')
    # ax2.set_xlabel('Hour')
    # ax2.set_ylabel('Heat [kW]', color='indianred')
    # # ax1.set_ylabel('Oxygen demand [MW]', color='purple')
    # # ax2.set_ylabel('Electricity price [€/MWh]', color='darkorange')
    # ax4.set_ylabel('Oxygen [kg/h]', color='mediumpurple')
    # lns = l2+l3+l5+l6
    # labs = [l.get_label() for l in lns]
    # # ax1.legend(lns, labs, loc='upper left')
    # #Indicating days
    # for i in range(math.floor((x2-x1)/24)):
    #     if d1 == x1:
    #         if i % 2 == 1:
    #             ax2.axvspan(d1-x1+(24*i),d1-x1+((24*(i+1))), facecolor='0.2', alpha=0.2, zorder=-1)
    #     elif d1 != x1:
    #         if i % 2 == 0:
    #             ax2.axvspan(d1-x1+(24*i),d1-x1+((24*(i+1))), facecolor='0.2', alpha=0.2, zorder=-1)
    # ax4.set_ylim(0,6000)
    # ax2.set_ylim(0,2000)
    # # ax2.set_ylim(0,250)
    # ax2.set_xlim(0,x2-x1-1)
    
    # #Axis colors
    # ax4.spines['right'].set_color('mediumpurple')
    # ax4.yaxis.label.set_color('mediumpurple')
    # ax4.tick_params(axis='y', colors='mediumpurple')
    # ax2.spines['left'].set_color('indianred')
    # ax2.yaxis.label.set_color('indianred')
    # ax2.tick_params(axis='y', colors='indianred')
    
    # #Legend 1
    # h2_prod_patch = lines.Line2D([0], [0], color=teal_dark[4], lw=3, label='H$_2$ production')
    # h2_dem_patch = lines.Line2D([0], [0], color=teal_dark[2], lw=3, ls='--', label='H$_2$ demand')
    # h2st_patch = mpatches.Patch(facecolor=teal_light[1], edgecolor=teal_light[2], label='H$_2$ storage', linewidth=0)
    # el_patch = lines.Line2D([0], [0], color=orange_dark[4], lw=3, label='Electricity price')

    # legend = ax1.legend(loc='center',
    #     handles=[h2_prod_patch,h2_dem_patch,el_patch,h2st_patch],
    #     numpoints=1,
    #     frameon=False,
    #     bbox_to_anchor=(0.1, 0.98, 0.8, 0.15), 
    #     bbox_transform=ax3.transAxes,
    #     mode='expand', 
    #     ncol=4, 
    #     borderaxespad=-.46,
    #     prop={'size': 9,},
    #     handletextpad=0.5,
    #     handlelength=2.3)
    
    # #Legend 2
    # o2_prod_patch = lines.Line2D([0], [0], color='mediumpurple', lw=3, label='O$_2$ production')
    # o2_dem_patch = lines.Line2D([0], [0], color='mediumpurple', lw=3, ls='--', label='O$_2$ demand')
    # heat_prod_patch = lines.Line2D([0], [0], color='indianred', lw=3, label='Heat production')
    # heat_dem_patch = lines.Line2D([0], [0], color='indianred', lw=3, ls='--', label='Heat demand')

    # legend = ax2.legend(loc='center',
    #     handles=[o2_prod_patch,o2_dem_patch,heat_prod_patch,heat_dem_patch],
    #     numpoints=1,
    #     frameon=False,
    #     bbox_to_anchor=(0.07, 0.98, 0.87, -2.25), 
    #     bbox_transform=ax3.transAxes,
    #     mode='expand', 
    #     ncol=4, 
    #     borderaxespad=-.46,
    #     prop={'size': 9,},
    #     handletextpad=0.5,
    #     handlelength=2.3)
    
    # plt.show()
    
    
    # 
    #Emission bar chart
    # p2g_bar = [220.6, 48.6, 1511.4, 51.4]
    # p2g_heat_bar = [-4.1, -21.9, -6.9, 7.1]
    # p2g_o2_bar = [1.5, -0.3, -10.1, 0.6]
    # p2g_total_bar = [218.0, 26.4, 1494.1, 59.1]
    
    # #Relative values
    # p2g_bar = [100,100,100] #[LCOP2G, emissions, efficiency]
    # p2g_heat_bar = [-2,-44.6,13.5]
    # p2g_o2_bar = [1,-0.6,1.2]
    # p2g_total_bar = [99,54.8,114.7]
    
    # data = np.array([p2g_bar, p2g_heat_bar, p2g_o2_bar])

    # data_shape = np.shape(data)

    # # Take negative and positive data apart and cumulate
    # def get_cumulated_array(data, **kwargs):
    #     cum = data.clip(**kwargs)
    #     cum = np.cumsum(cum, axis=0)
    #     d = np.zeros(np.shape(data))
    #     d[1:] = cum[:-1]
    #     return d  

    # cumulated_data = get_cumulated_array(data, min=0)
    # cumulated_data_neg = get_cumulated_array(data, max=0)

    # # Re-merge negative and positive data.
    # row_mask = (data<0)
    # cumulated_data[row_mask] = cumulated_data_neg[row_mask]
    # data_stack = cumulated_data

    # cols = ["gray", "indianred", "slateblue"]
    
    # fig = plt.figure()
    # ax = plt.subplot(111)

    # for i in np.arange(0, data_shape[0]):
    #     ax.bar(np.arange(data_shape[1]), data[i], bottom=data_stack[i], color=cols[i], width=0.6)
        
    # ax.plot([0,1,2], p2g_total_bar, marker='o', lw=0, color='k', markersize=7)
    
    # plt.xticks([0,1,2], ['LCOP2G', 'CO$_2$ emissions', 'Efficiency'], fontweight='bold')
    # plt.axhline(y=0,linewidth=0.5, color='k')
    # p2g_patch = mpatches.Patch(color='gray', label='P2G system', linewidth=0)
    # heat_patch = mpatches.Patch(color='indianred', label='Heat', alpha=0.5, linewidth=0)
    # o2_patch = mpatches.Patch(color='slateblue', label='Oxygen', alpha=0.5, linewidth=0)
    # #grid_patch = mpatches.Patch(color='darkorange', label='Grid purchases', linewidth=0)
    # net_patch = lines.Line2D([0], [0], color='k', lw=0, marker='o', label='Net value')
    # legend = fig.legend(loc='upper right', handles=[p2g_patch, heat_patch, o2_patch, net_patch], bbox_to_anchor=(-0.1, 0.78, 1, 0.1))
    # plt.show()
    
    # bars = np.add(p2g_bar, p2g_heat_bar).tolist()
 
    # # The position of the bars on the x-axis
    # r = [0,1,2]
 
    # # Names of group and bar width
    # names = ['MSP','Emissions','Efficiency']
    # barWidth = 1
 
    # # Create brown bars
    # plt.bar(r, p2g_bar, color='#7f6d5f', edgecolor='white', width=barWidth)
    # # Create green bars (middle), on top of the first ones
    # plt.bar(r, p2g_heat_bar, bottom=p2g_bar, color='#557f2d', edgecolor='white', width=barWidth)
    # # Create green bars (top)
    # plt.bar(r, p2g_o2_bar, bottom=bars, color='#2d7f5e', edgecolor='white', width=barWidth)
 
    # # Custom X axis
    # plt.xticks(r, names, fontweight='bold')
    # plt.xlabel("group")
 
    # # Show graphic
    # plt.show()
    #Sankey diagram (include efficiencies somehow?) (on annual basis or steady state?)
    #Flows
  #   gr_el = process['Elz grid [kWh/h]'].sum()/1000
  #   pv_el = (process['Elz PV [kWh/h]']+bat_in_pv).sum()/1000
  #   wi_el = (process['Elz wind [kWh/h]']+bat_in_wind).sum()/1000
  #   el_bat = sum(bat_in_wind+bat_in_pv)/1000 + 1000
  #   bat_el = el_bat*bat_eff
  #   bat_ls= el_bat-bat_el
  #   el_h2 = process['Elz dispatch [kWh/h]'].sum()/1000
  #   bg_cm = (biogas_in[0,:].sum()*tec.ch4_mol/1000)
  #   el_cm = process['Biogas comp [kWh/h]'].sum()/1000
  #   h2_ch4 = process['H2 used [kg/h]'].sum()*tec.h2_kg/1000
  #   bg_ls = process['CH4 flared [mol/h]'].sum()*tec.ch4_mol/1000
  #   cm_ch4 = bg_cm+el_cm
  #   el_ch4 = process['Meth el [kWh/h]'].sum()/1000
  #   h2_ht = elz_heat_nonnet.sum()/1000
  #   ch4_ht = process['Meth heat [kWh/h]'].sum()/1000
  #   ch4_ch4 = ch4_total
  #   ht_ww = heat_wwtp.sum()/1000
  #   ht_ls = h2_ht+ch4_ht-ht_ww
  #   o2_ww = o2_energy_savings/1000
  #   o2_ls = o2.aerator_savings * o2_loss * 32 / (1000*1000)
  #   h2_ls = el_h2-h2_ch4-h2_ht
  #   ch4_ls = h2_ch4+cm_ch4+el_ch4-ch4_ch4-ch4_ht
    
    
  #   import plotly.io as pio
  #   pio.renderers.default='browser'
  #   opacity = 0.4
  #   fig = go.Figure(data=[go.Sankey(
  #   node = dict(
  #     pad = 15,
  #     thickness = 10,
  #     line = dict(color = "black", width = 0.5),
  #     label = ["Grid", "PV", "Wind", "Biogas", "Electricity", "Battery", "Electrolysis", "Compression", "Oxygen", "Methanation", "Heat", "Losses", "WWTP", "Unused heat", "Methane"],
  #     color = "gray"
  #   ),
  #   link = dict(
  #     source = [0, 1, 2, 4, 5, 5, 4, 3, 4, 6, 7, 4, 6, 9, 9, 10, 10, 8, 8, 3, 6, 9], # indices correspond to labels.
  #     target = [4, 4, 4, 5, 4, 11, 6, 7, 7, 9, 9, 9, 10, 10, 14, 12, 11, 12, 11, 11, 11, 11],
  #     value = [gr_el,pv_el,wi_el,el_bat,bat_el,bat_ls,el_h2,bg_cm,el_cm,h2_ch4,cm_ch4,el_ch4,h2_ht,ch4_ht,ch4_ch4,ht_ww,ht_ls,o2_ww,o2_ls,bg_ls,h2_ls,ch4_ls],
  #     # label =  [],
  #     color =  ['gold','gold','gold','gold','gold','gold','gold','seagreen','gold','steelblue','seagreen','gold','indianred','indianred','coral','indianred','indianred','purple','purple','seagreen','steelblue','coral']
  # ))])

  #   fig.update_layout(title_text="P2G energy flows", font_size=10)
  #   fig.show()
    
    
    #Electricity from different sources and amount to electrolysis
    #H2 (and CO2/CH4) to CH4 and loss(?), heat to WWTP and loss, O2 (as aeration energy) to WWTP
    # print(res_frac)
    # if wind_size > 0:
    #     print(wind_frac)
    # if pv_size > 0:
    #     print(pv_frac)

# elif run_type == "optimization":
    #Include color as a third variable: https://stackoverflow.com/questions/8202605/how-to-color-scatter-markers-as-a-function-of-a-third-variable
    #For example for increasing renewable share
    #Could indicate PF only with edge colors and line and not full color to allow for both?
    
    # #Determine lowest LCOP2G configuration
    # min_msp = min(results.iloc[2,:])
    # index2 = results.columns[results.eq(min_msp).any()]

    # #(Could do a bubble plot to get a third KPI as dot size, but perhaps unnecessarily complex?)
    # #Pareto front (MSP vs. AEF)
    # fig, ax1 = plt.subplots()
    # #Scatter
    # ax1.plot(results.iloc[2,:], results.iloc[6,:], ls='none', marker='o') #LCOE vs. AEF
    # #Pareto
    # sorted_list = sorted([[results.iloc[2,i], results.iloc[6,i]] for i in range(len(results.iloc[2,:]))], reverse=False)
    # pareto_front = [sorted_list[0]]
    # for pair in sorted_list[1:]:
    #     if pair[1] < pareto_front[-1][1]: #could be an issue not to have "<="
    #                 pareto_front.append(pair)
    # pf_lcoe = [pair[0] for pair in pareto_front]
    # pf_aef = [pair[1] for pair in pareto_front]
    # ax1.plot(pf_lcoe, pf_aef, ls='-', marker='o', color='r')

    # ax1.set_ylabel('Spec. ems (AEF) [kg$_{CO_2}$/MWh$_{CH_4}$]')
    # ax1.set_xlabel('MSP [€/MWh]')
    
    # #Pareto front (MSP vs. MEF)
    # fig, ax1 = plt.subplots()
    # #Scatter
    # ax1.plot(results.iloc[2,:], results.iloc[7,:], ls='none', marker='o') #LCOE vs. AEF
    # #Pareto
    # sorted_list = sorted([[results.iloc[2,i], results.iloc[7,i]] for i in range(len(results.iloc[2,:]))], reverse=False)
    # pareto_front = [sorted_list[0]]
    # for pair in sorted_list[1:]:
    #     if pair[1] < pareto_front[-1][1]: #could be an issue not to have "<="
    #                 pareto_front.append(pair)
    # pf_lcoe = [pair[0] for pair in pareto_front]
    # pf_aef = [pair[1] for pair in pareto_front]
    # ax1.plot(pf_lcoe, pf_aef, ls='-', marker='o', color='r')

    # ax1.set_ylabel('Spec. ems (MEF) [kg$_{CO_2}$/MWh$_{CH_4}$]')
    # ax1.set_xlabel('MSP [€/MWh]')

    # #Determine lowest LCOP2G configuration
    # min_lcop2g = min(results.iloc[0,:])
    # index_min = results.columns[results.eq(min_lcop2g).any()]   
    # meth_use = 5 #the methanation fraction value used in elz vs. h2st plot
    # h2st_use = 400 #the hydrogen storage value used in elz vs. meth plot
    
    # #2D COLOR PLOTS
    # #defining vectors
    # # elz_size_vector = [6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12] #MW
    # # meth_scale_vector = [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2] #ratio to elz
    # # h2st_size_vector = [0,1,2,3,4,5] #hours
    # #LCOP2G
    # #Electrolyzer vs. H2 storage
    # dx = elz_size_vector[1] - elz_size_vector[0]
    # dy = h2st_size_vector[1] - h2st_size_vector[0]
    # y, x = np.mgrid[slice(h2st_size_vector[0], h2st_size_vector[-1] + dy, dy),
    #                 slice(elz_size_vector[0], elz_size_vector[-1] + dx, dx)]
    
    # meth_index = meth_scale_vector.index(meth_use)
    
    # fig, (ax1, ax2) = plt.subplots(1,2)
    # Z = np.zeros([len(h2st_size_vector),len(elz_size_vector)])
    # for x in range(len(h2st_size_vector)):
    #     for y in range(len(elz_size_vector)):
    #         Z[x,y] = results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size_vector[y],meth_scale_vector[meth_index],h2st_size_vector[x],wind_size_vector[0]*elz_size_vector[y],pv_size_vector[0]*elz_size_vector[y],bat_size_vector[0])][0]
    #         lcop2g_test = results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size_vector[y],meth_scale_vector[meth_index],h2st_size_vector[x],wind_size_vector[0]*elz_size_vector[y],pv_size_vector[0]*elz_size_vector[y],bat_size_vector[0])][0]
    #         if lcop2g_test == min_lcop2g:
    #             elz_mini = elz_size_vector[y]
    #             h2st_mini = h2st_size_vector[x]
                
    # levels = np.linspace(190,250,50)
    # X = elz_size_vector
    # Y = h2st_size_vector
    # elz_h2st = ax1.contourf(X, Y, Z, zorder=1, levels=levels)
    # fig.colorbar(elz_h2st, ticks=[190,200,210,220,230,240,250])
    # ax1.plot(elz_mini,h2st_mini, color='k', marker='o', zorder=2)
    # ax1.set_xlabel('Electrolyser [MW$_{el}$]')
    # ax1.set_ylabel('Hydrogen storage [hours]')
    # plt.text(0.01,1.02,'(a) Fixed methanation ratio of {}.'.format(meth_use), fontsize=10, transform=ax1.transAxes)
    
    # #Electrolyzer vs. methanation reactor
    # dx = elz_size_vector[1] - elz_size_vector[0]
    # dy = meth_scale_vector[1] - meth_scale_vector[0]
    # y, x = np.mgrid[slice(meth_scale_vector[0], meth_scale_vector[-1] + dy, dy),
    #                 slice(elz_size_vector[0], elz_size_vector[-1] + dx, dx)]
    
    # h2st_index = h2st_size_vector.index(h2st_use)
    # # fig, ax = plt.subplots()
    # Z = np.zeros([len(meth_scale_vector),len(elz_size_vector)])
    # for x in range(len(meth_scale_vector)):
    #     for y in range(len(elz_size_vector)):
    #         Z[x,y] = results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size_vector[y],meth_scale_vector[x],h2st_size_vector[h2st_index],wind_size_vector[0]*elz_size_vector[y],pv_size_vector[0]*elz_size_vector[y],bat_size_vector[0])][0]
    #         lcop2g_test = results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size_vector[y],meth_scale_vector[x],h2st_size_vector[h2st_index],wind_size_vector[0]*elz_size_vector[y],pv_size_vector[0]*elz_size_vector[y],bat_size_vector[0])][0]
    #         if lcop2g_test == min_lcop2g:
    #             elz_min = elz_size_vector[y]
    #             meth_min = meth_scale_vector[x]
                
    # levels = np.linspace(190,250,50)
    # X = elz_size_vector
    # Y = meth_scale_vector
    # elz_meth = ax2.contourf(X, Y, Z, zorder=1, levels=levels)
    # fig.colorbar(elz_meth, ticks=[190,200,210,220,230,240,250])
    # ax2.plot(elz_min,meth_min, color='k', marker='o', zorder=2)
    # ax2.set_xlabel('Electrolyser [MW$_{el}$]')
    # ax2.set_ylabel('Methanation ratio [-]')
    # # ax2.set_title('LCOP2G')
    # plt.text(0.01,1.02,'(b) Fixed hydrogen storage capacity of {} hours.'.format(h2st_use), fontsize=10, transform=ax2.transAxes)
    
    # fig.suptitle('LCOP2G [€/MWh$_{CH_4}$]', fontsize=15)
    
    # #GAS LOSSES
    # meth_use = 5 #the methanation fraction value used in elz vs. h2st plot
    # h2st_use = 300 #the hydrogen storage value used in elz vs. meth plot
    # #Electrolyzer vs. H2 storage
    # dx = elz_size_vector[1] - elz_size_vector[0]
    # dy = h2st_size_vector[1] - h2st_size_vector[0]
    # y, x = np.mgrid[slice(h2st_size_vector[0], h2st_size_vector[-1] + dy, dy),
    #                 slice(elz_size_vector[0], elz_size_vector[-1] + dx, dx)]
    
    # meth_index = meth_scale_vector.index(meth_use)
    
    # fig, (ax1, ax2) = plt.subplots(1,2)
    # Z = np.zeros([len(h2st_size_vector),len(elz_size_vector)])
    # for x in range(len(h2st_size_vector)):
    #     for y in range(len(elz_size_vector)):
    #         Z[x,y] = results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size_vector[y],meth_scale_vector[meth_index],h2st_size_vector[x],wind_size_vector[0]*elz_size_vector[y],pv_size_vector[0]*elz_size_vector[y],bat_size_vector[0])][12]
    #         lcop2g_test = results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size_vector[y],meth_scale_vector[meth_index],h2st_size_vector[x],wind_size_vector[0]*elz_size_vector[y],pv_size_vector[0]*elz_size_vector[y],bat_size_vector[0])][12]
    #         if lcop2g_test == min_lcop2g:
    #             elz_mini = elz_size_vector[y]
    #             h2st_mini = h2st_size_vector[x]
                
    # levels = np.linspace(0,18,50)
    # X = elz_size_vector
    # Y = h2st_size_vector
    # elz_h2st = ax1.contourf(X, Y, Z, zorder=1, levels=levels)
    # fig.colorbar(elz_h2st, ticks=[0,2,4,6,8,10,12,14,16,18])#,20,30])
    # # ax1.plot(elz_mini,h2st_mini, color='k', marker='o', zorder=2)
    # ax1.set_xlabel('Electrolyser [MW$_{el}$]')
    # ax1.set_ylabel('Hydrogen storage [hours]')
    # plt.text(0.01,1.02,'(a) Fixed methanation ratio of {}.'.format(meth_use), fontsize=10, transform=ax1.transAxes)
    
    # #Electrolyzer vs. methanation reactor
    # dx = elz_size_vector[1] - elz_size_vector[0]
    # dy = meth_scale_vector[1] - meth_scale_vector[0]
    # y, x = np.mgrid[slice(meth_scale_vector[0], meth_scale_vector[-1] + dy, dy),
    #                 slice(elz_size_vector[0], elz_size_vector[-1] + dx, dx)]
    
    # h2st_index = h2st_size_vector.index(h2st_use)
    # Z = np.zeros([len(meth_scale_vector),len(elz_size_vector)])
    # for x in range(len(meth_scale_vector)):
    #     for y in range(len(elz_size_vector)):
    #         Z[x,y] = results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size_vector[y],meth_scale_vector[x],h2st_size_vector[h2st_index],wind_size_vector[0]*elz_size_vector[y],pv_size_vector[0]*elz_size_vector[y],bat_size_vector[0])][12]
    #         lcop2g_test = results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size_vector[y],meth_scale_vector[x],h2st_size_vector[h2st_index],wind_size_vector[0]*elz_size_vector[y],pv_size_vector[0]*elz_size_vector[y],bat_size_vector[0])][12]
    #         if lcop2g_test == min_lcop2g:
    #             elz_min = elz_size_vector[y]
    #             meth_min = meth_scale_vector[x]
                
    # levels = np.linspace(0,18,50)
    # X = elz_size_vector
    # Y = meth_scale_vector
    # elz_meth = ax2.contourf(X, Y, Z, zorder=1, levels=levels)
    # fig.colorbar(elz_meth, ticks=[0,2,4,6,8,10,12,14,16,18])
    # # ax2.plot(elz_min,meth_min, color='k', marker='o', zorder=2)
    # ax2.set_xlabel('Electrolyser [MW$_{el}$]')
    # ax2.set_ylabel('Methanation ratio [-]')
    # plt.text(0.01,1.02,'(b) Fixed hydrogen storage capacity of {} hours.'.format(h2st_use), fontsize=10, transform=ax2.transAxes)
    
    # fig.suptitle('Gas loss [%]', fontsize=15)
    
    # #OTHERS (16/17 for by-product NPV, 3-5 for efficiency)
    # meth_use = 5 #the methanation fraction value used in elz vs. h2st plot
    # h2st_use = 300 #the hydrogen storage value used in elz vs. meth plot
    # #Electrolyzer vs. H2 storage
    # dx = elz_size_vector[1] - elz_size_vector[0]
    # dy = h2st_size_vector[1] - h2st_size_vector[0]
    # y, x = np.mgrid[slice(h2st_size_vector[0], h2st_size_vector[-1] + dy, dy),
    #                 slice(elz_size_vector[0], elz_size_vector[-1] + dx, dx)]
    
    # meth_index = meth_scale_vector.index(meth_use)
    
    # fig, (ax1, ax2) = plt.subplots(1,2)
    # Z = np.zeros([len(h2st_size_vector),len(elz_size_vector)])
    # for x in range(len(h2st_size_vector)):
    #     for y in range(len(elz_size_vector)):
    #         Z[x,y] = results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size_vector[y],meth_scale_vector[meth_index],h2st_size_vector[x],wind_size_vector[0]*elz_size_vector[y],pv_size_vector[0]*elz_size_vector[y],bat_size_vector[0])][6]
    #         lcop2g_test = results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size_vector[y],meth_scale_vector[meth_index],h2st_size_vector[x],wind_size_vector[0]*elz_size_vector[y],pv_size_vector[0]*elz_size_vector[y],bat_size_vector[0])][6]
    #         if lcop2g_test == min_lcop2g:
    #             elz_mini = elz_size_vector[y]
    #             h2st_mini = h2st_size_vector[x]
                
    # # levels = np.linspace(0,5,50)
    # X = elz_size_vector
    # Y = h2st_size_vector
    # elz_h2st = ax1.contourf(X, Y, Z, zorder=1)#, levels=levels)
    # fig.colorbar(elz_h2st)#, ticks=[0,2,4,6,8,10,12])#,20,30])
    # # ax1.plot(elz_mini,h2st_mini, color='k', marker='o', zorder=2)
    # ax1.set_xlabel('Electrolyser [MW$_{el}$]')
    # ax1.set_ylabel('Hydrogen storage [hours]')
    # plt.text(0.01,1.02,'(a) Fixed methanation ratio of {}.'.format(meth_use), fontsize=10, transform=ax1.transAxes)
    
    # #Electrolyzer vs. methanation reactor
    # dx = elz_size_vector[1] - elz_size_vector[0]
    # dy = meth_scale_vector[1] - meth_scale_vector[0]
    # y, x = np.mgrid[slice(meth_scale_vector[0], meth_scale_vector[-1] + dy, dy),
    #                 slice(elz_size_vector[0], elz_size_vector[-1] + dx, dx)]
    
    # h2st_index = h2st_size_vector.index(h2st_use)
    # Z = np.zeros([len(meth_scale_vector),len(elz_size_vector)])
    # for x in range(len(meth_scale_vector)):
    #     for y in range(len(elz_size_vector)):
    #         Z[x,y] = results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size_vector[y],meth_scale_vector[x],h2st_size_vector[h2st_index],wind_size_vector[0]*elz_size_vector[y],pv_size_vector[0]*elz_size_vector[y],bat_size_vector[0])][6]
    #         lcop2g_test = results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size_vector[y],meth_scale_vector[x],h2st_size_vector[h2st_index],wind_size_vector[0]*elz_size_vector[y],pv_size_vector[0]*elz_size_vector[y],bat_size_vector[0])][6]
    #         if lcop2g_test == min_lcop2g:
    #             elz_min = elz_size_vector[y]
    #             meth_min = meth_scale_vector[x]
                
    # # levels = np.linspace(0,2,50)
    # X = elz_size_vector
    # Y = meth_scale_vector
    # elz_meth = ax2.contourf(X, Y, Z, zorder=1)#, levels=levels)
    # fig.colorbar(elz_meth)#, ticks=[0,10,20,30,40,50])
    # # ax2.plot(elz_min,meth_min, color='k', marker='o', zorder=2)
    # ax2.set_xlabel('Electrolyser [MW$_{el}$]')
    # ax2.set_ylabel('Methanation ratio [-]')
    # plt.text(0.01,1.02,'(b) Fixed hydrogen storage capacity of {} hours.'.format(h2st_use), fontsize=10, transform=ax2.transAxes)
    
    # fig.suptitle('System efficiency [%]', fontsize=15)
    
    
    
    
    
    
    
    
    