# -*- coding: utf-8 -*-
"""
Created on Thu Nov 2 11:41:17 2023

@author: Linus Engstam
"""

import numpy as np
import pandas as pd
import math
import P2G.parameters as params
import P2G.components as comps
import P2G.other as other
import P2G.kpis as kpis
import P2G.dispatch as dispatch
import matplotlib.pyplot as plt
from matplotlib import rc
from tabulate import tabulate
import matplotlib.patches as mpatches
import matplotlib.lines as lines
import plotly.graph_objects as go
import urllib, json
import seaborn as sns


"""
Comments on current version


"""
    

"""MAIN P2G MODEL"""

#MAY HAVE DOUBLE-COUNTED CURTATAILED PV ABOVE 2X CAPACITY (PURE PV), TRY AND REDO

""" Optimization parameters """
# res_scale = 
elz_size_vector = [8.5]#[6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12] #MW
meth_scale_vector = [5]#[3.5,4,4.5,5] #MWout #[0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2] #ratio to elz
h2st_size_vector = [400]#[0,100,200,300,400,500,600,700,800,900,1000] #kg #[0,1,2,3,4,5] #hours
wind_size_vector = [1.25] #ratio
pv_size_vector = [1.25] #ratio
bat_size_vector = [0] #hours

#OPTIMIZING FOR CURTAILMENT COST NOW

""" Simulation parameters """

#CHECK EFFICIENCY CURVE
#COULD SORT DATA USING CLASS OR DICTIONARY LIKE STRUCT IN MATLAB?


year = 2021 #not adapted to 8784 hours?
bidding_zone = "SE3"
by_use = "WWTP"
ef_type = "aef" #aef/mef/fix
scenario = "grid" #grid/res
alpha = 0
pwl_points = 10 #number of points for linearization of efficiencies
excess_res_frac = 0.5 #the fraction of installed local RES capacity above which there was assumed to be an excess of renewable electricity

# Technical parameters
#Biogas plant
biogas_pres = 1 #bar
biogas_temp = 50 #C

# Electrolyzer
# elz_size = 8 #MW
elz_n_system = 0.75 #HHV system efficiency at full load (part-load is estimated)
# elz_aux = 0.05 #auxiliary component consumption as share of rated power (van der Roest, check other sources). Assuming constant.
elz_n_stack = elz_n_system + 0.05 #HHV stack efficiency at full load
# elz_min = 0 #minimum load (now defined further down based on auxiliary consumption)
elz_start = 5 #minutes (cold start)
elz_start_cost = elz_start/60 #0 #startup cost [fraction of rated power]
elz_standby_cost = 0.02 #standby cost [fraction of rated power]
# elz_cooldown = 6 #hours
elz_heatup_time = 60 #[minutes] the time during which no usable waste heat is released (look into in more detail)
elz_temp = 80 #C
elz_h2o_temp = 15 #[C] inlet water temperature
elz_pres = 30 #bar
elz_degr = 1 #[% per year/1000 FLHs depending on stack replacement time below] from Ginsberg, strengthened by van der Roest and within ranges of Buttler
stack_rep = 10 #hours if > 1000, years if < 100. If year, define if it is replaced after ten years or at the start of year ten? After ten years now.
degr_year = round(stack_rep/2) #which year of operation to consider for efficiency degradation (assuming 0 as year 1)
elz_water_cons = 10 #[l/kgH2]
h2_hv = 39.4 #kWh/kgH2

#H2 storage
# h2st_size = 4 #hours
# h2st_n = 1 #Hydrogen storage round-trip efficiency (not implemented yet)

#Methanation (should it be sized?)
# meth_size = elz_size * 0.85 #[MW] could just define scale compared to "expected" from electrolyzer here, and define real size in "parameters"
# meth_scale = 1 #scale copared to electrolyzer pairing
meth_temp = 65 #C
meth_pres = 10 #bar
meth_start = 0 #minutes (cold start) NEED TO IMPLEMENT THIS ALSO OUTSIDE THE MILP
meth_min = 0 #minimum load
# meth_cooldown = 6 #what to define here?
meth_n = 0.99 #CO2 conversion efficiency
microb_cons = 0#0.06 #fraction of extra CO2 required for microbial growth
meth_standby_energy = 0#0.01 #fraction of rated electricity input required for stand-by
meth_el_cons = 0.5 #[kWh/Nm3 CH4 produced] In Schlautmann et al. 2020 for example.

#Membrane (should be simple, just an output purity or similar?)

#Heat storage (Not implemented)
# heatst_size = 0 #unit? hours?

#Oxygen storage (Not implemented)
# o2st_size = 0 #hours

#Battery storage (Not implemented)
bat_size = 0 #hours of full electrolyzer electricity input capacity
bat_eff = 0.95 #round trip efficiency of battery

#Electricity
# wind_size = 0 #MW
# pv_size = 0 #MW
wind_efs = 15 #[kgCO2/MWh]
pv_efs = 70 #[kgCO2/MWh]
pv_degr = 0.5 #[%/y]
#Grid emissions for electrolyzer dispatch optimization
# if ef_type == "aef":
#     efs_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\EFs\efs_' + str(bidding_zone) + '_' + str(year) + '.xlsx'
#     efs = pd.read_excel(efs_read)
#     efs = efs.iloc[:,0]
# elif ef_type == "mef":
#     efs_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\EFs\efs_' + str(bidding_zone) + '_' + str(year) + '.xlsx'
#     efs = pd.read_excel(efs_read)
#     efs = efs.iloc[:,1]
    
#Other
usable_heat = 0.8 #heat exchanger efficiency (van der Roest et al. 2023)
comp_n = 0.75 #compressor isentropic efficiency (use 0.75 % instead?)
n_motor = 0.95 #electrical motor efficiency
# cond_n = 0.8 #condenser efficiency (how much heat can be utilized) But probably too low temperature?

#By-products
#Do at parameter stage. What needs to be defined here?
#What about emission reduction from heat use? Both internal and DH?
dh_emissions = 112 #[gCO2/kWh]
dh_emissions_marginal = 30 #[gCO2/kWh]
# pellet_emissions = 100 #[gCO2/kWh] find more accurate value!
#Aerators
o2_replacement_factor = 100/21 #how much can we reduce the aeration flow by?
sote_increase = 1 #reduced o2 demand due to higher driving force, need to implement properly for costs if used as we're still replacing the same amount of air!
aerator_air = 1/17 #[kWh/kgO2]
aerator_o2 = aerator_air/(o2_replacement_factor) #[kWh/kgO2]
#Donald & Love
# aerator_air = 1.38 / 0.21 #[kWh/kgO2]
# aerator_o2 = 1.38
aerator_savings = (aerator_air - aerator_o2) * sote_increase #[kWh/kgO2] energy savings per kg pure O2
# aerator_savings_mol = aerator_savings / (1000 / 32) #[kWh/mol] energy savings per mol pure O2
#Heat
heat_scale = 1
# digester_heat = 300 #[kW] (could assume 10 % of biogas energy content as in Michailos et al.)
# heat_frac_use = 1

#Gas parameters (which are used in the end?)
ch4_hhv_vol = 11.05 #kWh/Nm3
# ch4_lhv_vol = 9.94 #kWh/Nm3
ch4_hhv_kg = 15.44 #kWh/kg
# ch4_lhv_kg = 13.9 #kWh/kg
ch4_hhv_mol = ch4_hhv_kg / (1000/16.04) #kWh/mol
# ch4_lhv_mol = ch4_lhv_kg / (1000/16.04) #kWh/mol
nm3_mol = ch4_hhv_mol / ch4_hhv_vol #Nm3/mol
# h2_hhv_vol = 3.53 #kWh/Nm3
# h2_lhv_vol = 3 #kWh/Nm3
h2_hhv_kg = 39.4 #kWh/kg
# h2_lhv_kg = 33.3 #kWh/kg
h2_hhv_mol = h2_hhv_kg / (1000/2.02) #kWh/mol
# h2_lhv_mol = h2_lhv_kg / (1000/2.02) #kWh/mol

# Economical parameters
#General
lifetime = 20 #years
discount = 8 #[%]
co2_cost = 0 #[€/tonCO2]

#Biogas plant (should it be included?)
# biogas_capex = 1000 #[€/kW CH4]
# biogas_opex = 10 #[% of CAPEX]
biogas_lcoe = 65 #[€/MWh raw biogas]
biogas_ef = 50 #[gCO2/kWh]

#Electrolyzer
elz_capex = 1500 #[€/kW] at 5 MW
elz_capex_ref = 5 #[MW]
elz_opex = 4 #% of CAPEX
elz_scaling = 0.75 #scaling factor for CAPEX
water_cost = 0.5 #€/m3 (including purification) Se förstudie för värde
stack_cost = 0.5 #fraction of CAPEX

#H2 storage
h2st_capex = 500 #€/kgH2 (is this CAPEX true for low-pressure storage?)
h2st_opex = 1.5 #% of CAPEX
# h2st_pres = 30 #bar (if it requires high pressure we want to avoid storage when not needed?)

#Methanation
meth_capex = 900 #[€/kWCH4] assumed at 5 MW
meth_capex_ref = 5 #[MW]
meth_opex = 8 #% of CAPEX
meth_scaling = 0.65 #scaling factor for CAPEX

#Installation cost
install_cost = 20 #[% of total CAPEX]

#Membrane (included in methanation?)
# mem_capex = 0
# mem_opex = 0 #% of CAPEX

#Heat storage
# heatst_capex = 0
# heatst_opex = 0 #% of CAPEX

#Oxygen storage
# o2st_capex = 0
# o2st_opex = 0 #% of CAPEX

#Battery storage
bat_capex = 300 #[€/kWh]
bat_opex = 2 #% of CAPEX

#Electricity
# wind_capex = 1000 #€/kW
# wind_opex = 3 #% of capex
wind_lcoe = 40 #[€/MWh] assumed PPA price
# pv_capex = 1000 #€/kW
# pv_opex = 1 #% of CAPEX
pv_lcoe = 45 #€/MWh
grid_fee = 10 #€/MWh (only PV on-site?)
# res_cost = 40
# res_cost_milp = 0

#Other components
# bg_clean_capex = 0 #(desulfurification and drying)
# comp_capex = 0
comp_opex = 5 #% of CAPEX (Khan et al.)
# hex_capex = 0
# hex_opex = 0 #% of CAPEX
# cond_capex = 0 #can probably assume this already exists?
# cond_opex = 0

#By-products
#Internal heat from pellets: 4.3 kWh heat/kg, efficiency of 90 %, 200 €/ton --> 4.3*0.9*1000 / 200 = 19.35 € ~ 20 €/MWh
# pellet_cost = 100 #[€/MWh]
#DH pricing (converted SEK to EUR)
# dh_price = 50 #[€/MWh]
# dh_fix = 89 #[€/kW and year] only applies if we completely replace DH
dh_winter = 53 #[€/MWh] minus for annual usage discount
dh_spr_aut = 35 #[€/MWh]
dh_summer = 22 #[€/MWh]

# o2_price = 80 #[€/ton]
# gas_price = 180 #[€/MWh] LHV or HHV?
#Ozone?
#CAPEX/OPEX
heat_integration_capex = 260 #[€/kWth]
heat_integration_ref = 400 #[kWth]
heat_piping_capex = 230 #[€/m]
# o2_integration_capex = 70+77 #[€/kW electrolyzer] 70 for piping etc, 77 for pure O2 aeration tech.
# o2_integration_ref = 0
o2_piping_capex = 540 #[€/m]
o2_aerator_capex = 70 #[€/kW electrolyzer]
o2_aerator_ref = 1.25 #[MWel]
# o2_piping_ref = 4.8 #[MWel]
piping_dist = 1000 #[m]
heat_integration_opex = 2
o2_integration_opex = 2 
heat_integration_scaling = 0.3
o2_integration_scaling = 0.6


""" DEFINING OPTIMIZATION """

#Run type
if len(elz_size_vector) == 1 and len(meth_scale_vector) == 1 and len(h2st_size_vector) == 1 and len(wind_size_vector) == 1 and len(pv_size_vector) == 1 and len(bat_size_vector) == 1:
    run_type = "single"
else:
    run_type = "optimization"
    #Counting etc.
    sims = len(elz_size_vector) * len(h2st_size_vector) * len(wind_size_vector) * len(pv_size_vector) * len(meth_scale_vector) * len(bat_size_vector)
    count = 0

#Create results dataframe
results = pd.DataFrame({'KPIs': ['LCOP2G (curt)', 'LCOP2G', 'MSP', 'MSP (no curt)', 'LCOE', 'Gas eff.', 'Heat eff.', 'Tot eff.', 'AEF net', 'MEF net', 'Starts', 'Standby', 'FLHs', 'Loss [%]', 'O2 util.', 'O2 dem.', 'Heat util', 'Heat dem.', 'RES [%]', 'LCOP2G BY diff.', 'LCOP2G BY rel.', 'MSP BY diff.', 'MSP BY rel.', 'NPV O2', 'NPV HEAT']}).set_index('KPIs')
cost_breakdown = pd.DataFrame({'Costs': ['Electrolyser','Stack','Water','Storage','Meth','Comp','Heat','O2','Installation','Flaring','Grid','PV','Wind','Curtailment','O2 income','Heat income','Total']}).set_index('Costs')

#Run simulation for all investigated sizes
for e in range(len(elz_size_vector)):
    for m in range(len(meth_scale_vector)):
        for s in range(len(h2st_size_vector)):
            for w in range(len(wind_size_vector)):
                for p in range(len(pv_size_vector)):
                    for b in range(len(bat_size_vector)):
                        
                        """ DEFINING COMPONENTS """
                        elz_size = elz_size_vector[e]
                        meth_scale = meth_scale_vector[m]
                        h2st_size = h2st_size_vector[s]
                        wind_size = wind_size_vector[w] * elz_size
                        pv_size = pv_size_vector[p] * elz_size
                        bat_size = bat_size_vector[b]
                        
                        #Biogas plant
                        biogas_flow = params.biogas_plant(year=year)

                        #Electrolyzer
                        #what is needed? Part load efficiency, heat production, other parameters for detailed operation model?
                        k_values, m_values, elz_auxiliary, elz_sys_eff_degr, elz_stack_eff_degr = params.electrolyzer(elz_size=elz_size, system_efficiency=elz_n_system, stack_efficiency=elz_n_stack, pwl_points=pwl_points, elz_degr=elz_degr, degr_year=degr_year)
                        elz_h2_max = elz_size * 1000 * elz_n_system / 39.4
                        # elz_h2_max = elz_size * 1000 * elz_sys_eff_degr / 39.4
                        elz_size_degr = elz_size * elz_n_system / elz_sys_eff_degr
                        elz_min = elz_auxiliary / (elz_size_degr*1000)
                        elz_heat_gen = ((elz_size_degr*1000) - elz_auxiliary) * (1-elz_stack_eff_degr)
                        elz_heat_h2o = ((elz_h2_max* 1000 / 2.02) * (elz_water_cons*997/(1000*18.02/2.02))) * 75.3 * (elz_temp - elz_h2o_temp) / (3600*1000)
                        elz_heat_max = elz_heat_gen - elz_heat_h2o

                        #Methanation (MOVE THIS TO OTHER SCRIPT WHEN DONE)
                        #Is cost of methanation based on methane output capacity or electrolyzer electricity input?
                        # meth_max_size_mw = max(biogas_flow.iloc[:,1]) * ch4_hhv_mol / 1000 #[MW methane output] 1 mol CO2 in = 1 mol CH4 out, HHV or LHV? Should this be rounded up?
                        # meth_size_mol = meth_scale * (elz_size * 1000 * elz_n_system / (4*h2_hhv_mol)) #Using non-degraded efficiency
                        # meth_size = meth_size_mol * ch4_hhv_mol / 1000 #[MWth out]
                        meth_size = meth_scale
                        meth_size_mol = meth_size * 1000 / ch4_hhv_mol
                        # meth_size_co2 = meth_size * 1000 / ch4_hhv_mol #Maximum CO2 flow rate in mol
                        meth_size_vector = np.zeros(24,) + meth_size_mol
                        meth_standby_cons = meth_standby_energy * meth_el_cons * meth_size_mol * nm3_mol #[kWh]
                        meth_loss_factor = meth_start / 60 #the amount of H2 lost due to cold methanation start-up
                        # meth_params = params.methanation(size=meth_size, n=meth_n, meth_type=meth_type, min_load=meth_min, startup_time=meth_start, cooldown_time=meth_cooldown, temp=meth_temp, pressure=meth_pres)
                        # meth_el_cons_tot = meth_el_cons * nm3_mol * meth_size_mol #methanation electricity consumption
                        meth_flow_max = meth_size_mol / min(biogas_flow.iloc[:,1]/(biogas_flow.iloc[:,1]+biogas_flow.iloc[:,0])) #[mol/h] maximum theoretical flow rate to methanation
                        meth_flow_min = meth_size_mol * meth_min
                        __, meth_el_max, meth_heat_max, __, __ = comps.methanation(meth_flow=[meth_size_mol*4/(1-microb_cons),meth_size_mol/(1-microb_cons),0], rated_flow=meth_size_mol/(1-microb_cons), T=meth_temp, T_in=meth_temp, el_cons=meth_el_cons)
                        meth_spec_heat = meth_heat_max / (4*meth_size_mol*2.02/1000)
                        meth_spec_el = meth_el_max / (4*meth_size_mol*2.02/1000) #[kWh/kgH2]

                        #Storages (using non-degraded efficiency)
                        h2st_cap = h2st_size #h2st_size * elz_size * 1000 * elz_n_system / h2_hhv_kg #[kg]
                        # o2st_cap = (o2st_size * elz_size * 1000 * elz_n_system / (2*h2_hhv_mol)) * (32/1000) #[kg] half the mol of H2, molar mass of O2 is 32/1000 kgO2/mol
                        # heatst_cap = heatst_size * elz_size * (1-elz_n_system) * usable_heat #[MWh]
                        bat_cap = bat_size * elz_size * 1000 #[kWh]

                        #Membrane

                        #H2 Compressor (temperature increase during operation calculations?) (diaphragm as of now, best choice?)
                        # h2_flow_max = elz_size * 1000 * elz_n_system / (h2_hhv_mol) #mol/h
                        # h2_comp_size = params.compressor(flow=h2_flow_max/3600, temp_in=elz_temp, p_in=elz_pres, p_out=h2st_pres, n_isen=comp_n, n_motor=n_motor)

                        #Biogas compressor (using formulas above)
                        #Capacity
                        bg_comp_size = params.compressor(flow=meth_flow_max/3600, temp_in=biogas_temp, p_in=biogas_pres, p_out=meth_pres, n_isen=comp_n, n_motor=n_motor)
                        comp_spec_el = bg_comp_size / meth_flow_max #[kWh/mol compressed gas]
                        
                        #Electricity
                        # p2g_el_cons = (elz_size_degr*1000) + meth_el_cons_tot + bg_comp_size
                        res_gen = params.renewables(wind_size=wind_size, pv_size=pv_size, pv_degr=pv_degr, lifetime=lifetime, year=year)
                        if scenario == "res":
                            res_tot = res_gen.sum(axis=1)
                            wind_excess = np.where(res_gen.iloc[:,0] >= wind_size*1000*excess_res_frac, 1, 0)
                            pv_excess = np.where(res_gen.iloc[:,1] >= pv_size*1000*excess_res_frac, 1, 0)
                            res_excess = np.where(((wind_excess == 1) | (pv_excess == 1)), 1, 0)
                        
                        spot_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\elspot prices ' + str(year) + '.xlsx'
                        spot_price = pd.read_excel(spot_read) + grid_fee
                        spot_price = np.array(spot_price[bidding_zone].tolist())
                        # spot_price = np.zeros(8760,) + 67.85
                        
                        #Heat exchangers (now assuming an overall cost for heat integration)
                        # elz_heat_max = elz_params.iloc[len(elz_params)-1,2]
                        # meth_heat_max = meth_params.iloc[len(meth_params)-1,2]

                        #Condenser (now assuming an overall cost for heat integration)
                        # meth_h20_max = biogas_flow.iloc[0,1] * 2 #4H2 + CO2 --> CH4 + 2H2O
                        # h2o_heat_of_cond = 40.8/3600 #kWh/mol
                        # cond_heat_max = meth_h20_max * h2o_heat_of_cond #what about temperature decrease?

                        #By-products
                        o2_demand, heat_demand_tot, heat_demand_bg, heat_demand_aux = params.byprod_loads(o2_scale=1/sote_increase, heat_scale=heat_scale, year=year)
                        # heat_demand_tot = heat_demand_tot*4
                        
                        """ PROCESS SIMULATION """
                        #Initiate process data
                        process = other.data_saving(year=year)

                        #Save independent data
                        # process['Emissions [gCO$_2$/kWh]'] = efs
                        process['Elspot [€/MWh]'] = spot_price
                         
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

                        #Converting to numpy arrays
                        biogas_flow_arr = np.array(biogas_flow)
                        
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
                        
                        #Daily electrolyzer dispatch on day-ahead market
                        if year == 2020:
                            hrs = 8784
                        else:
                            hrs = 8760
                        for d in range(int(hrs/24)): #daily basis
                            #hour values
                            i1 = d*24
                            i2 = i1 + 24
                            
                            h2_demand_hr = ((np.array([biogas_flow.iloc[i1:i2,1]]) * 4) * (1-microb_cons))# - (4*gas_recirc[1]) - gas_recirc[0] #4:1 ratio of H2 and CO2 [mol/h] minus recycled CO2 and H2 and microbial consumption
                            h2_demand_hr = np.minimum(h2_demand_hr,meth_size_vector*4) #Also limited by methanation reactor size
                            h2_demand_hr = np.where(h2_demand_hr<(meth_flow_min*4),0,h2_demand_hr)
                            h2_demand_hr = np.transpose(h2_demand_hr)
                            h2_demand_hr_kg = h2_demand_hr * 2.02 / 1000 #mol to kg
                            
                            #Also, consider if/when we should fill up the storage ahead of the next day?
                            
                            #By-product demands
                            heat_demand_hr = np.array(heat_demand_tot[i1:i2])
                            o2_demand_hr = np.array(o2_demand[i1:i2])
                        
                            
                            #Check last hour of previous day
                            if d != 0:
                                if elz_on == 1 or elz_standby == 1:
                                    prev_mode = 1
                                else:
                                    prev_mode = 0
                            #Multi och single objective optimization
                            # if alpha == 0:
                                # if scenario == "grid":
                            elz_dispatch = dispatch.p2g_wwtp3(h2_demand=h2_demand_hr_kg, heat_demand=heat_demand_hr, heat_value=[dh_spr_aut,dh_summer,dh_spr_aut,dh_winter], usable_heat=usable_heat, meth_spec_heat=meth_spec_heat, o2_demand=o2_demand_hr*32/1000, o2_power=aerator_savings, k_values=k_values, m_values=m_values, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size_degr*1000, elz_min=elz_min*elz_size_degr*1000, elz_eff=elz_n_system, aux_cons=elz_auxiliary, meth_max=meth_size_mol*2.02*4/1000, meth_min=meth_min*meth_size_mol*4*2.02/1000, h2st_max=h2st_cap, h2st_prev=h2_storage, prev_mode=prev_mode, startup_cost=elz_start_cost, standby_cost=elz_standby_cost, bat_cap=bat_cap, bat_eff=bat_eff, bat_prev=bat_storage, meth_el_factor=meth_spec_el, h2o_cons=elz_water_cons, temp=elz_temp, h2o_temp=elz_h2o_temp, biogas=biogas_flow_arr, comp_el_factor=comp_spec_el, elz_startup_time=elz_start/60)# wind_cost=wind_lcoe, pv_cost=pv_lcoe)
                                # elif scenario == "res":
                                    # elz_dispatch = dispatch.p2g_wwtp2_res(h2_demand=h2_demand_hr_kg, heat_demand=heat_demand_hr, heat_value=[dh_spr_aut,dh_summer,dh_spr_aut,dh_winter], usable_heat=usable_heat, meth_spec_heat=meth_spec_heat, o2_demand=o2_demand_hr*32/1000, o2_power=aerator_savings, k_values=k_values, m_values=m_values, grid=spot_price[i1:i2], res=res_excess[i1:i2], elz_max=elz_size_degr*1000, elz_min=elz_min*elz_size_degr*1000, elz_eff=elz_n_system, aux_cons=elz_auxiliary, meth_max=meth_size_mol*2.02*4/1000, meth_min=meth_min*meth_size_mol*4*2.02/1000, h2st_max=h2st_cap, h2st_prev=h2_storage, prev_mode=prev_mode, startup_cost=elz_start_cost, standby_cost=elz_standby_cost, bat_cap=bat_cap, bat_eff=bat_eff, bat_prev=bat_storage, meth_el_factor=meth_spec_el, h2o_cons=elz_water_cons, temp=elz_temp, h2o_temp=elz_h2o_temp, biogas=biogas_flow_arr, comp_el_factor=comp_spec_el, elz_startup_time=elz_start/60, res_cost=res_cost_milp)# wind_cost=wind_lcoe, pv_cost=pv_lcoe)
                            # elif alpha == 1:
                                # elz_dispatch = 1#dispatch.ems_daily_pl(demand=h2_demand_kg, grid=efs[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size_degr*1000, elz_min=elz_min*elz_size_degr*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, wind_ef=0, pv_ef=0)#wind_ef=wind_efs, pv_ef=pv_efs)
                            # elif alpha > 0 and alpha < 1:
                                #To determine the nadir and utopian values for cost and emissions, for normalization of multi-objective function
                                # elz_dispatch1, cost_utp, ems_nad = 1#dispatch.multi_daily_pl(demand=h2_demand_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size_degr*1000, elz_min=elz_min*elz_size_degr*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, alpha=0, efs=efs[i1:i2], wind_ef=0, pv_ef=0, wind_cost=0, pv_cost=0)#wind_ef=wind_efs, pv_ef=pv_efs, wind_cost=wind_lcoe+grid_fee, pv_cost=pv_lcoe+grid_fee)
                                # elz_dispatch2, ems_utp, cost_nad = 1#dispatch.multi_daily_pl(demand=h2_demand_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size_degr*1000, elz_min=elz_min*elz_size_degr*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, alpha=1, efs=efs[i1:i2], wind_ef=0, pv_ef=0, wind_cost=0, pv_cost=0)#wind_ef=wind_efs, pv_ef=pv_efs, efs=efs[i1:i2], wind_cost=wind_lcoe+grid_fee, pv_cost=pv_lcoe+grid_fee)                
                                # C_NAD.append(cost_nad)
                                # C_UTP.append(cost_utp)
                                # E_NAD.append(ems_nad)
                                # E_UTP.append(ems_utp)
                                #Using nadir and utopian values to perform multi-objective optimization of electrolyzer dispatch
                                # if abs(cost_utp-cost_nad) < 1 or abs(ems_utp-ems_nad) < 1:
                                    # test.append(1)
                                    # elz_dispatch = elz_dispatch1
                                # else:
                                    # elz_dispatch, __, __ = 1#dispatch.multi_daily_pl(demand=h2_demand_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size_degr*1000, elz_min=elz_min*elz_size_degr*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, alpha=alpha, cost_norm=abs(cost_nad-cost_utp), ems_norm=abs(ems_nad-ems_utp), ems_utp=ems_utp, cost_utp=cost_utp, efs=efs[i1:i2], wind_ef=0, pv_ef=0, wind_cost=0, pv_cost=0)#wind_ef=wind_efs, pv_ef=pv_efs, efs=efs[i1:i2], wind_cost=wind_lcoe+grid_fee, pv_cost=pv_lcoe+grid_fee)
                                    # test.append(0)
                                   
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
                        h2_flow, elz_heat, T_h2_out, o2_flow, h2o_cons, stack_eff, sys_eff, elz_heat_nonnet = comps.electrolyzer(dispatch=electrolyzer, prod=h2_production, aux=elz_auxiliary, temp=elz_temp, h2o_temp=elz_h2o_temp, heat_time=elz_heatup_time, startups=electrolyzer_start, h2o_cons=elz_water_cons, year=year)
                        h2st_in = np.maximum(0,(h2_production)-h2_used) * 1000 / 2.02
                        h2st_out = np.minimum(np.maximum(0,h2_used-h2_production),(h2_storage_list_prev*1000/2.02)) * 1000 / 2.02
                        h2_meth = h2_used * 1000 / 2.02
                        
                        
                        #Flow definitions
                        
                        co2_flow = h2_meth / ((1-microb_cons)*4)
                        # max_co2_flow_p2g = np.minimum(co2_flow, elz_h2_max*1000/2.02/4)
                        # used_co2 = h2_meth
                        p2g_frac = np.divide(co2_flow, biogas_flow_arr[:,1].T, out=np.zeros_like((co2_flow)), where=biogas_flow_arr[:,1]!=0)
                        biogas_in = biogas_flow_arr.T * p2g_frac
                        bg_flow = biogas_in.sum(axis=0)
                        flared_gas = biogas_flow_arr.T * abs(np.around((1-p2g_frac),6))
                        if year == 2020:
                            flared_gas = [np.zeros(8784,), flared_gas[1], flared_gas[0]] #[H2, CO2, CH4]
                        else:
                            flared_gas = [np.zeros(8760,), flared_gas[1], flared_gas[0]] #[H2, CO2, CH4]
                        
                        #Biogas compression (flow rate in; compressor power and temp. out(?))
                        bg_comp_power, T_bg_comp = comps.compressor(flow=bg_flow, temp_in=biogas_temp, p_in=biogas_pres, p_out=meth_pres, n_isen=comp_n, n_motor=n_motor, year=year) #[kWh]

                        #Gas mixing (Biogas, hydrogen, temp in; temp out?)
                        co2_in = biogas_in[1]#biogas_flow.iloc[hour,1]# + gas_recirc[1]
                        h2_in = h2_meth# + gas_recirc[0]
                        ch4_in = biogas_in[0]#biogas_flow.iloc[hour,0]# + gas_recirc[2]
                        inlet_flow, T_inlet = comps.mixer(h2=h2_in, co2=co2_in, ch4=ch4_in, h2_temp=elz_temp, bg_temp=biogas_temp)

                        #Pre-heating (temp in and out in, molar flow in (specific heat); energy (heat) consumption out) (is this needed for biological?)
                        # pre_heating = comps.preheater(flow=inlet_flow, T_in=T_inlet, T_out=meth_temp)

                        # #Methanation startup flaring
                        # startup_flare = np.asarray([(np.array(meth_startup) * inlet_flow[0,:] * (1-meth_loss_factor)), (np.array(meth_startup) * inlet_flow[1,:] * (1-meth_loss_factor)), (np.array(meth_startup) * inlet_flow[2,:] * (1-meth_loss_factor))])
                        # flared_gas = np.add(flared_gas, startup_flare)
                        # inlet_flow = np.add(inlet_flow,-startup_flare)

                        #Methanation (molar flows, temp. in; molar flows, excess heat, electricity consumption out)
                        meth_outlet_flow, meth_power, meth_heat, h2o_cond1, microbial_co2 = comps.methanation(meth_flow=inlet_flow, rated_flow=meth_flow_max, T=meth_temp, T_in=T_inlet, el_cons=meth_el_cons, n=meth_n, microb_cons=microb_cons)
                        meth_outlet_flow = np.asarray(meth_outlet_flow)
                        h2o_cond1 = np.asarray(h2o_cond1)
                        
                        # meth_flow_max = meth_size_mol
                        # meth_outlet_flow = []
                        # meth_power = []
                        # meth_heat = []
                        # h2o_cond1 = []
                        # microbial_co2 = []
                        # for h in range(len(h2_flow)):
                        #     meth_outlet_flow_h, meth_power_h, meth_heat_h, h2o_cond1_h, microbial_co2_h = comps.methanation(meth_flow=inlet_flow.T[h], rated_flow=meth_flow_max, T=meth_temp, T_in=T_inlet[h], el_cons=meth_el_cons)
                        #     meth_outlet_flow.append(meth_outlet_flow_h)
                        #     meth_power.append(meth_power_h)
                        #     meth_heat.append(meth_heat_h)
                        #     h2o_cond1.append(h2o_cond1_h)
                        #     microbial_co2.append(microbial_co2_h)
                        # meth_outlet_flow = np.asarray(meth_outlet_flow)
                        #     #Do something with the dynamic variables here?
                        # h2o_cond1 = np.asarray(h2o_cond1)

                        #Condenser (H2O in, energy out)
                        cond_outlet_flow, cond_heat, cond_power, h2o_cond2, T_cond_out = comps.condenser(flow=meth_outlet_flow, T_in=meth_temp, year=year)

                        #Gas cleaning (molar flows and temp in; pure stream and recirculation out, energy consumption?)
                        ch4_out, recirc_flow, gas_loss, T_out, p_out = comps.membrane(mem_inlet_flow=cond_outlet_flow, T_in=T_cond_out, p_in=meth_pres, year=year)
                        #now there is no recirculation, only losses
                        #Losses from non-used biogas
                        total_gas_loss = gas_loss + flared_gas #[H2, CO2, CH4]
                        #PERHAPS WE NEED TO CONSIDER FLARING? Or just losses and assume flared could be easier?

                        #Other variables
                        #Recirculated water
                        h2o_recirc = 0 #h2o_cond1 + h2o_cond2
                        
                        #Rated total system power
                        # sys_nominal = math.ceil(((elz_size_degr*1000) + max(meth_power) + max(bg_comp_power)) / 100) * 100
                        sys_nominal = round(((elz_size_degr*1000) + max(meth_power) + max(bg_comp_power)),0)
                        
                        #TEMPORARY, REMOVE!
                        starts = []
                        for i in range(len(electrolyzer)):
                            if i > 0:
                                if electrolyzer[i-1] == 0 and electrolyzer[i] > 0 and electrolyzer_standby[i-1] != 1:
                                    starts.append(1)
                                else:
                                    starts.append(0)
                            else:
                                starts.append(0)
                        
                        
                        #Storing results
                        process['Biogas (CH4) [mol/h]'] = list(biogas_flow.iloc[:,0])
                        process['Biogas (CO2) [mol/h]'] = list(biogas_flow.iloc[:,1])
                        process['H2 demand [mol/h]'] = list(H2_demand)
                        process['Elz dispatch [kWh/h]'] = list(electrolyzer)
                        process['System dispatch [kWh/h]'] = list(sys_op)
                        process['Standby el. [kWh/h]'] = list(sb_el)
                        process['Elz wind [kWh/h]'] = list(wind_use)
                        process['Wind gen [kWh/h]'] = list(res_gen.iloc[:,0])
                        process['Elz PV [kWh/h]'] = list(pv_use)
                        process['PV gen [kWh/h]'] = list(res_gen.iloc[:,1])
                        process['Elz grid [kWh/h]'] = list(grid_use)
                        process['H2 used [kg/h]'] = list(h2_used)
                        process['Unmet demand [kgH2/h]'] = list(unmet_demand)
                        process['O2 WWTP [mol/h]'] = list(o2_demand)
                        # if by_use == "WWTP+O3":
                            # process['O3 WWTP [mol/h]'] = list(hourly_o3)
                        process['H2 production [mol/h]'] = h2_flow
                        process['H2 to meth [mol/h]'] = h2_in
                        process['H2 to storage [mol/h]'] = h2st_in
                        process['H2 from storage [mol/h]'] = h2st_out
                        if h2st_cap > 0:
                            process['H2 storage [%]'] = (h2_storage_list/(h2st_cap))*100 #[%]
                        else:
                            process['H2 storage [%]'] = 0
                        process['Elz heat [kWh/h]'] = elz_heat#h_prod#
                        process['H2 temp [C]'] = elz_temp#h_use#
                        process['O2 out [mol/h]'] = o2_flow
                        process['H2O cons [mol/h]'] = h2o_cons
                        process['Biogas comp [kWh/h]'] = bg_comp_power
                        process['Biogas temp [C]'] = T_bg_comp
                        process['Meth CH4 in [mol/h]'] = inlet_flow[2]
                        process['Meth H2 in [mol/h]'] = inlet_flow[0]
                        process['Meth CO2 in [mol/h]'] = inlet_flow[1]
                        process['Meth in temp [C]'] = T_inlet
                        # process['Preheating [kWh/h]'] = pre_heating
                        process['Meth CH4 out [mol/h]'] = meth_outlet_flow[2]
                        process['Meth H2 out [mol/h]'] = meth_outlet_flow[0]
                        process['Meth CO2 out [mol/h]'] = meth_outlet_flow[1]
                        process['Meth H2O(g) out [mol/h]'] = meth_outlet_flow[3]
                        process['Meth H2O(l) out [mol/h]'] = h2o_cond1
                        process['Meth el [kWh/h]'] = meth_power
                        process['Meth heat [kWh/h]'] = meth_heat
                        process['Cond CH4 out [mol/h]'] = cond_outlet_flow[2]
                        process['Cond H2 out [mol/h]'] = cond_outlet_flow[0]
                        process['Cond CO2 out [mol/h]'] = cond_outlet_flow[1]
                        process['Cond H2O(l) out [mol/h]'] = h2o_cond2
                        process['Cond heat [kWh/h]'] = cond_heat
                        process['Cond el [kWh/h]'] = cond_power
                        process['H2O recirc [mol/h]'] = h2o_recirc
                        process['Cond temp out [C]'] = T_cond_out
                        process['CH4 out [mol/h]'] = ch4_out
                        process['Recirc CH4 [mol/h]'] = recirc_flow[2]
                        process['Recirc H2 [mol/h]'] = recirc_flow[0]
                        process['Recirc CO2 [mol/h]'] = recirc_flow[1]
                        process['CH4 loss [mol/h]'] = gas_loss[2]
                        process['H2 loss [mol/h]'] = gas_loss[0]
                        process['CO2 loss [mol/h]'] = gas_loss[1]
                        process['Recirc temp [C]'] = T_out
                        process['Recirc pres [bar]'] = p_out
                        process['Microbial CO2 cons [mol/h]'] = microbial_co2
                        process['CH4 flared [mol/h]'] = flared_gas[2]
                        process['Stack efficiency [%]'] = stack_eff * 100
                        process['System efficiency [%]'] = sys_eff * 100
                        process['Elz standby'] = electrolyzer_standby 
                        if scenario == "grid":
                            process['Battery state [%]'] = np.divide(np.array(battery_state) * 100, bat_cap, out=np.zeros_like(battery_state), where=bat_cap!=0)
                            process['Battery discharging [kWh/h]'] = np.array(battery_dis)


                        """ TECHNICAL ANALYSIS """

                        #Gas production
                        ch4_p2g = (process['CH4 out [mol/h]'] - process['Meth CH4 in [mol/h]']).sum() * ch4_hhv_mol / 1000 #[MWh] Annual CH4 production increase from P2G
                        ch4_total = process['CH4 out [mol/h]'].sum() * ch4_hhv_mol / 1000
                        #Gas loss
                        ch4_loss = process['CH4 loss [mol/h]'].sum() * ch4_hhv_mol
                        loss_frac = (process['CH4 loss [mol/h]'].sum() / biogas_flow.iloc[:,0].sum()) * 100 #[%] 
                        flare_frac = (process['CH4 flared [mol/h]'].sum()/biogas_flow.iloc[:,0].sum()) * 100 #[%]
                        total_loss_frac = loss_frac + flare_frac
                        
                        # lifetime = 18
                        
                        #Operation
                        #Stack replacement
                        elz_flh = round(process['Elz dispatch [kWh/h]'].sum() / (elz_size_degr*1000)) #full load hours of the electrolyzer
                        if stack_rep > 1000: #counting hours
                            stack_reps = math.floor((elz_flh*lifetime) / stack_rep) #number of stack replacements during project lifetime
                        else:
                            stack_reps = math.floor((lifetime-1) / stack_rep) #number of stack replacements during project lifetime, minus 1 since only 2 replacements are required every ten years for a 30 year lifetime for example
                        #COULD MAKE SURE NO LATE REPLACEMENTS BY ROUNDING DOWN TO PREVIOUS NUMBER IF ONLY AT FOR EXAMPLE 2.1

                        #ELECTRICITY USE
                        
                        #number of cold starts
                        starts = []
                        for i in range(len(process['Elz dispatch [kWh/h]'])):
                            if i > 0:
                                if process['Elz dispatch [kWh/h]'][i-1] == 0 and process['Elz dispatch [kWh/h]'][i] > 0 and process['Elz standby'][i-1] != 1:
                                    starts.append(1)
                                else:
                                    starts.append(0)
                            else:
                                starts.append(0)
                        process['Elz cold start'] = starts

                        # #methanation stand-by(off)
                        # meth_standby = []
                        # for i in range(len(process['Elz dispatch [kWh/h]'])):
                        #     if process['Meth el [kWh/h]'][i] > 0:
                        #         meth_standby.append(0)
                        #     else:
                        #         meth_standby.append(1)
                        
                        if scenario == "grid":
                            #Battery operation
                            bat_in = []
                            bat_in_wind = []
                            excess_wind = []
                            bat_in_pv = []
                            excess_pv = []
                            bat_in_grid = []
                            bat_loss = []
                            for h in range(len(process['Battery state [%]'])):
                                if h == 0:
                                    bat_in.append((process['Battery state [%]'][h]) * bat_cap / 100)
                                    bat_in_pv.append(round(min((res_gen.iloc[h,1] - process['Elz PV [kWh/h]'][h]), bat_in[h]),6))#(bat_cap-(process['Battery state [%]'][h]*bat_cap/100))),6))
                                    excess_pv.append(round(max((res_gen.iloc[h,1] - process['Elz PV [kWh/h]'][h]) - bat_in_pv[h],0),6))#(bat_cap-(process['Battery state [%]'][h]*bat_cap/100)),0),6))
                                    bat_in_wind.append(round(min((res_gen.iloc[h,0] - process['Elz wind [kWh/h]'][h]), bat_in[h] - bat_in_pv[h]),6))#((bat_cap-(process['Battery state [%]'][h]*bat_cap/100))-bat_in_pv[h])),6))
                                    excess_wind.append(round(max((res_gen.iloc[h,0] - process['Elz wind [kWh/h]'][h]) - bat_in_wind[h],0),6))#((bat_cap-(process['Battery state [%]'][h]*bat_cap/100))-bat_in_pv[h]),0),6))
                                    bat_in_grid.append(round(bat_in[h] - bat_in_pv[h] - bat_in_wind[h],6))
                                    bat_loss.append(bat_in[h] * (1-bat_eff))
    
                                else:
                                    bat_in.append(round((max((process['Battery state [%]'][h] - process['Battery state [%]'][h-1]),0) * bat_cap / 100),6))
                                    bat_in_pv.append(round(min((res_gen.iloc[h,1] - process['Elz PV [kWh/h]'][h]), bat_in[h]),6))#(bat_cap-(process['Battery state [%]'][h]*bat_cap/100))),6))
                                    excess_pv.append(round(max((res_gen.iloc[h,1] - process['Elz PV [kWh/h]'][h]) - bat_in_pv[h],0),6))#(bat_cap-(process['Battery state [%]'][h]*bat_cap/100)),0),6))
                                    bat_in_wind.append(round(min((res_gen.iloc[h,0] - process['Elz wind [kWh/h]'][h]), bat_in[h] - bat_in_pv[h]),6))#((bat_cap-(process['Battery state [%]'][h]*bat_cap/100))-bat_in_pv[h])),6))
                                    excess_wind.append(round(max((res_gen.iloc[h,0] - process['Elz wind [kWh/h]'][h]) - bat_in_wind[h],0),6))#((bat_cap-(process['Battery state [%]'][h]*bat_cap/100))-bat_in_pv[h]),0),6))
                                    bat_in_grid.append(round(bat_in[h] - bat_in_pv[h] - bat_in_wind[h],6))
                                    bat_loss.append(bat_in[h] * (1-bat_eff))
    
                            
                            #Excess wind/PV (how to get 100 % RES here? Either include storage here, or include auxiliaries in MILP)
                            if year == 2020:
                                excess_wind = np.maximum(np.array(res_gen.iloc[:,0] - process['Elz wind [kWh/h]'] - bat_in_wind), np.zeros(8784,))
                                excess_pv = np.maximum(np.array(res_gen.iloc[:,1] - process['Elz PV [kWh/h]'] - bat_in_pv), np.zeros(8784,))
                            else:
                                excess_wind = np.maximum(np.array(res_gen.iloc[:,0] - process['Elz wind [kWh/h]'] - bat_in_wind), np.zeros(8760,))
                                excess_pv = np.maximum(np.array(res_gen.iloc[:,1] - process['Elz PV [kWh/h]'] - bat_in_pv), np.zeros(8760,))
                            # non_exported_pv = np.maximum(np.array(excess_pv) - sys_nominal,0) #Used in local RES case
                            #Using first any excess PV (cheaper), then any excess wind if residual load remains
                            residual_load = np.array(process['Biogas comp [kWh/h]'] + process['Meth el [kWh/h]'] + (process['Elz standby'] * elz_standby_cost * elz_size_degr))# + (np.array(meth_standby) * meth_standby_cons))
                            # used_excess_pv = np.minimum(residual_load, excess_pv)
                            # used_excess_wind = np.minimum(residual_load - used_excess_pv, excess_wind)
                            # excess_res_load_frac = sum(used_excess_pv+used_excess_wind) / sum(residual_load) #how much of the residual load could we cover with excess wind/pv
                            curtailment = (excess_wind.sum() + excess_pv.sum()) / 1000 #[MWh]
                            tot_res = (res_gen.iloc[:,0].sum() + res_gen.iloc[:,1].sum()) / 1000 #[MWh]
                            if tot_res > 0:
                                curt_frac = curtailment * 100 / tot_res #[%]
                            else:
                                curt_frac = 0
                            #Renewable electricity fraction
                            res_frac = sum(process['Elz wind [kWh/h]'] + process['Elz PV [kWh/h]'] + bat_in_wind + bat_in_pv) / sum(process['System dispatch [kWh/h]'])
                            if wind_size > 0:
                                # unused_wind_frac = sum((excess_wind-used_excess_wind)) / sum(res_gen.iloc[:,0])
                                # unused_res_frac = sum((excess_pv-used_excess_pv) + (excess_wind-used_excess_wind)) / sum(res_gen.iloc[:,0] + res_gen.iloc[:,1])
                                wind_frac = sum(process['Elz wind [kWh/h]'] + bat_in_wind) / sum(process['Elz wind [kWh/h]'] + process['Elz PV [kWh/h]'] + bat_in_wind + bat_in_pv)
                            if pv_size > 0:
                                # unused_pv_frac = sum((excess_pv-used_excess_pv)) / sum(res_gen.iloc[:,1])
                                pv_frac = sum(process['Elz PV [kWh/h]'] + bat_in_pv) / sum(process['Elz wind [kWh/h]'] + process['Elz PV [kWh/h]'] + bat_in_wind + bat_in_pv)
                        elif scenario == "res":
                            res_frac = 1
                        #STORAGE
                        if h2st_cap > 0:
                            #Number of full storage cycles
                            h2st_cycles = round(process['H2 to storage [mol/h]'].sum() * 2.02 / (h2st_cap * 1000))
                            #Percent of time above 90 % full
                            h2st_90 = ((process['H2 storage [%]'] > 90) * 1).sum() / len(process['H2 storage [%]'])
                        else:
                            h2st_cycles = 0
                            h2st_90 = 0
                        
                        if bat_cap > 0:
                            #Number of full storage cycles
                            bat_cycles = round(sum(bat_in) / bat_cap)
                            #Percent of time above 90 % full
                            bat_90 = ((process['Battery state [%]'] > 90) * 1).sum() / len(process['Battery state [%]'])
                        else:
                            bat_cycles = 0
                            bat_90 = 0
                        
                        #HEAT
                        heat_wwtp = []
                        heat_prod = usable_heat * np.array(process['Elz heat [kWh/h]'] + process['Meth heat [kWh/h]']) #low-grade heat produced [kWh/h] (excluding condenser heat for now)
                        heat_prod_out = heat_prod.clip(min=0) #exclude all interal heat consumption by electrolyzer
                        total_heat_demand = heat_demand_tot# + process['Preheating [kWh/h]'] #currently excluding preheating since it is negligible (and included in methanation?)
                        # total_heat_demand = (2*(heat_demand_tot - heat_demand_aux)) + heat_demand_aux
                        if by_use == "WWTP":
                            for i in range(len(process['O2 out [mol/h]'])):
                                heat_wwtp.append(min(heat_prod_out[i], total_heat_demand[i])) #using "heat_prod_out" to avoid negative values at low electrolyzer loads and no methanation
                            heat_wwtp = np.array(heat_wwtp)
                            heat_elz_use = heat_prod.clip(max=0) #all interal heat consumption by electrolyzer
                            heat_elz_use = np.array(-heat_elz_use)
                            heat_loss = (sum(np.maximum(heat_prod - total_heat_demand,0)) / heat_prod_out.sum()) * 100
                            heat_use_frac = (heat_wwtp.sum() / heat_prod_out.sum()) * 100 #how much of the output heat is utilized
                            heat_use_frac_net = (heat_wwtp.sum() / heat_prod.sum()) * 100 #how much of the heat is utilized, including heat consumption by stack
                            #assuming hygienization:
                            # heat_use_frac = ((heat_wwtp.sum()+3037000) / heat_prod_out.sum()) * 100
                            if total_heat_demand.sum() > 0:
                                heat_wwtp_use_frac = (heat_wwtp.sum() / total_heat_demand.sum()) * 100
                            else:
                                heat_wwtp_use_frac = 0

                        #OXYGEN
                        o2_wwtp = []
                        o3_wwtp = []
                        if by_use == "WWTP":
                            for i in range(len(process['O2 out [mol/h]'])):
                                o2_wwtp.append(min(process['O2 out [mol/h]'][i], process['O2 WWTP [mol/h]'][i]))
                            o2_wwtp = np.array(o2_wwtp)
                            o2_loss = sum(np.maximum(process['O2 out [mol/h]'] - o2_wwtp,0))
                            o2_loss_frac = (sum(np.maximum(process['O2 out [mol/h]'] - o2_wwtp,0)) / process['O2 out [mol/h]'].sum()) * 100
                            o2_use_frac = (o2_wwtp.sum() / process['O2 out [mol/h]'].sum()) * 100
                            o2_wwtp_use_frac = (o2_wwtp.sum() / process['O2 WWTP [mol/h]'].sum()) * 100
                            o2_energy_savings = aerator_savings * sum(o2_wwtp) * 32 / 1000 #[kWh]
                            o2_energy_frac = 100 * o2_energy_savings / sum(process['Elz dispatch [kWh/h]']) #% of electrolyzer energy input saved
                            
                        #OZONE (could make a better economic case if it's already going to be done)
                        # o3_demand_annual = 1800000 #[kgO3/y]
                        # o2_o3_ratio = 10 #[10 kgO2/kgO3]
                        # o2_o3_demand = o3_demand_annual * o2_o3_ratio
                        # o3_savings = 5 #[kWh/kgO3]
                        # total_o3_savings = o3_savings * o3_demand_annual
                        # o3_savings_cost = total_o3_savings * np.mean(spot_price) / 1000

                        """ ECONOMIC ANALYSIS """
                        #Include DH here if excess heat, and some heat source if deficit
                        #Should also move to separate script?
                        #What to do with auxiliary electricity use (h2 and bg compressors etc)? Now included in OPEX, but emissions etc?

                        #ELECTRICITY
                        #Renewables (using LCOE)
                        # if scenario == "grid":
                        wind_cost = ((process['Elz wind [kWh/h]']+bat_in_wind) * (wind_lcoe + grid_fee)).sum() / 1000
                        # pv_cost = ((process['Elz PV [kWh/h]']+bat_in_pv) * (pv_lcoe)).sum() / 1000 #assuming on-site generation, i.e. no grid fee
                        pv_cost = ((process['Elz PV [kWh/h]']+bat_in_pv) * (pv_lcoe)).sum() / 1000 #For local RES case
                        # elif scenario == "res":
                            # res_cost = (process['Elz wind [kWh/h]'] * res_cost).sum() / 1000
                        curt_cost = ((excess_wind.sum() * wind_lcoe) + (excess_pv.sum() * pv_lcoe)) / 1000 #[€]
                        #Grid
                        grid_cost = (process['Elz grid [kWh/h]'] * spot_price).sum() / 1000 #[€] grid fee already included
                        #Auxiliary costs (cover a fraction of each using excess renewables)
                        #Start-up costs
                        startup_costs = (elz_size_degr * elz_start_cost * spot_price * starts).sum()
                        #Standby costs (spot or fixed price?)
                        # standby_costs = (process['Elz standby'] * elz_standby_cost * elz_size_degr * spot_price).sum() 
                        #Methanation
                        # meth_el = (process['Meth el [kWh/h]'] * spot_price).sum() * excess_res_load_frac / 1000
                        # meth_standby_el = (np.array(meth_standby) * meth_standby_cons * spot_price).sum() * excess_res_load_frac #CHECK AND FIX THIS
                        #Compressor
                        # bg_comp_el = (process['Biogas comp [kWh/h]'] * spot_price).sum() * excess_res_load_frac / 1000
                        #Overall electricity costs
                        # meth_el_cost = meth_el# + meth_standby_el
                        # elz_el_cost = grid_cost + startup_costs + standby_costs
                        # if scenario == "grid":
                        el_cost = wind_cost + pv_cost + grid_cost + startup_costs# + meth_standby_el
                        el_cost_curt = (el_cost + curt_cost)
                        # el_cost_curt = process['System dispatch [kWh/h]'].sum() * 10 / 1000
                        # elif scenario == "res":
                            # el_cost = res_cost + grid_cost + startup_costs# + meth_standby_el
                        # tot_el = sum(process['Elz dispatch [kWh/h]'] + (process['Elz standby'] * elz_standby_cost * elz_size_degr) + process['Meth el [kWh/h]'] + process['Biogas comp [kWh/h]'] + (np.array(meth_standby) * meth_standby_cons))
                        # el_cost1 = ((0.5*tot_el*(wind_lcoe + grid_fee)) / 1000) + ((0.5*tot_el*pv_lcoe) / 1000)
                        #Averages
                        avg_grid_price = grid_cost * 1000 / process['Elz grid [kWh/h]'].sum() #[€/MWh]
                        avg_tot_price = el_cost_curt * 1000 / process['System dispatch [kWh/h]'].sum() #[€/MWh]
                        avg_el_ch4 = el_cost_curt / ch4_p2g #[€/MWh]
                        #STORAGES
                        #O2 storage
                        # o2st_CAPEX = o2st_cap * o2st_capex
                        # o2st_OPEX = o2st_cap * (o2st_opex/100)
                        #Heat storage
                        # heatst_CAPEX = heatst_cap * heatst_capex
                        # heatst_OPEX = heatst_cap * (heatst_opex/100)
                        #Battery
                        bat_CAPEX = bat_cap * bat_capex
                        bat_OPEX = bat_cap * (bat_opex/100)

                        #HYDROGEN
                        #Electrolyzer
                        # elz_capex = 900*0.5
                        # elz_CAPEX = elz_capex * elz_size * 1000
                        elz_CAPEX = elz_capex * elz_capex_ref * ((elz_size/elz_capex_ref)**elz_scaling) * 1000 #CAPEX with scaling
                        elz_OPEX = (elz_opex*0.01*elz_CAPEX)
                        #Water
                        h2o_opex = water_cost * (process['H2O cons [mol/h]']-process['H2O recirc [mol/h]']).sum() * 18.02 / (1000*997) # €/m3 * mol * g/mol / (1000*kg/m3)
                        # h2o_opex = water_cost * (process['H2O cons [mol/h]']).sum() * 18.02 / (1000*997) # €/m3 * mol * g/mol / (1000*kg/m3)
                        #what about when recirculated doesn't match electrolyzer operation?
                        #Stack replacement
                        stack_COST = stack_cost * elz_CAPEX #total cost of stack replacements
                        #Storage
                        h2st_CAPEX = h2st_cap * h2st_capex
                        h2st_OPEX = h2st_opex * 0.01 * h2st_CAPEX
                        #Compressor (USD!!!)
                        # h2_csomp_capex = (63684.6 * (h2_comp_size**0.4603) * 1.3) * 0.75 #(Khan et al. ch. 5.2 with CAD to USD conversion)
                        # h2_comp_opex = comp_opex * 0.01 * h2_comp_capex
                        # h2_comp_el = (process['H2 comp [kWh/h]'] * spot_price).sum() / 1000
                        #Total hydrogen costs
                        H2_CAPEX = elz_CAPEX + h2st_CAPEX# + h2_comp_capex
                        H2_OPEX = elz_OPEX + h2o_opex + h2st_OPEX# + h2_comp_opex + h2_comp_el
                        H2_STACK = stack_COST #IS THIS CAPEX, OPEX OR OTHER?

                        #BIOGAS
                        # biogas_CAPEX = biogas_capex * max(biogas_flow.iloc[:,0]) * ch4_hhv_mol #[€]
                        # biogas_OPEX = biogas_CAPEX * biogas_opex / 100 #[€]
                        biogas_loss_cost = biogas_lcoe * (process['CH4 flared [mol/h]'].sum()) * ch4_hhv_mol / 1000
                        biogas_cost_tot = biogas_lcoe * (process['Biogas (CH4) [mol/h]'].sum()) * ch4_hhv_mol / 1000
                        
                        #METHANATION
                        #Reactor
                        # meth_capex = 600*0.9
                        # meth_CAPEX = meth_capex * meth_size * 1000 #(is the cost based on electricity input or gas output?)
                        meth_CAPEX = meth_capex * meth_capex_ref * ((meth_size/meth_capex_ref)**meth_scaling) * 1000 #CAPEX with scaling per MWth out
                        meth_opex_fix = meth_opex * 0.01 * meth_CAPEX #fixed opex
                        meth_OPEX = meth_opex_fix# + meth_el + meth_standby (electricity included above)
                        #Compressor (USD!!! DO CAD TO EUR INSTEAD)
                        # bg_comp_capex = (63684.6 * (bg_comp_size**0.4603) * 1.3) * 0.75 #(Khan et al. ch. 5.2 with CAD to USD conversion)
                        bg_comp_capex = 30000*(bg_comp_size**0.48) #(Kirchbacher et al., 2019)
                        # bg_comp_capex = (15000/1.1) * ((bg_comp_size/10)**0.9) #
                        # bg_comp_capex2 = (6310000*1.4/1.1) * ((bg_comp_size/10000)**0.67) #*1.4 to account for inflation
                        bg_comp_opex = comp_opex * 0.01 * bg_comp_capex
                        #Total methanation costs
                        METH_CAPEX = meth_CAPEX + bg_comp_capex
                        METH_OPEX = meth_OPEX + bg_comp_opex# + bg_comp_el (compressor electricity included above)

                        #GAS LOSSES
                        # loss_cost = 0 #process['CH4 loss [mol/h]'].sum() * ch4_hhv_mol * gas_price #LHV or HHV?

                        #BY-PRODUCT SYSTEM COSTS (include scaling effects with a SF of 0.6?)
                        heat_size = (elz_heat_max + meth_heat_max) * usable_heat
                        heat_system_CAPEX = heat_integration_capex * heat_integration_ref * ((heat_size/heat_integration_ref)**heat_integration_scaling) #(elz_size+meth_size) * 1000 * heat_integration_capex #(elz_CAPEX+meth_CAPEX) * heat_integration_capex/100
                        heat_piping_CAPEX = heat_piping_capex * piping_dist
                        heat_integration_CAPEX = heat_system_CAPEX + heat_piping_CAPEX
                        heat_integration_OPEX = heat_integration_CAPEX * (heat_integration_opex/100)
                        o2_piping_CAPEX = o2_piping_capex * piping_dist
                        o2_aerator_CAPEX = o2_aerator_capex * o2_aerator_ref * 1000 * ((elz_size/o2_aerator_ref)**o2_integration_scaling)
                        o2_integration_CAPEX = o2_piping_CAPEX + o2_aerator_CAPEX #elz_size * 1000 * o2_integration_capex
                        o2_integration_OPEX = o2_integration_CAPEX * (o2_integration_opex/100)
                        BY_CAPEX = heat_integration_CAPEX + o2_integration_CAPEX
                        BY_OPEX = heat_integration_OPEX + o2_integration_OPEX
                        rel_heat_capex = heat_integration_CAPEX * 100 / (elz_CAPEX + meth_CAPEX)
                        rel_o2_capex = o2_integration_CAPEX * 100 / (elz_CAPEX + meth_CAPEX)
                        
                        #CO2 cost
                        co2_opex = co2_cost * sum(co2_in) * 44.01 / (1000*1000)
                        
                        #OVERALL COSTS
                        CAPEX = H2_CAPEX + METH_CAPEX + BY_CAPEX + bat_CAPEX# + o2st_CAPEX + heatst_CAPEX
                        OPEX = H2_OPEX + METH_OPEX + BY_OPEX + el_cost + bat_OPEX + co2_opex# + o2st_OPEX + heatst_OPEX + loss_cost
                        OPEX_curt = H2_OPEX + METH_OPEX + BY_OPEX + el_cost_curt + bat_OPEX + co2_opex# o2st_OPEX + heatst_OPEX + loss_cost
                        #Installation cost
                        INSTALL = CAPEX * (0+(install_cost/100))
                        CAPEX = CAPEX + INSTALL

                        #Including biogas plant costs
                        # CAPEX_tot = CAPEX + biogas_CAPEX
                        # OPEX_tot = OPEX + biogas_OPEX
                        OPEX_tot = OPEX + biogas_loss_cost
                        OPEX_tot_curt = OPEX_curt + biogas_loss_cost
                        OPEX_msp = OPEX_curt + biogas_cost_tot
                        OPEX_msp_nocurt = OPEX + biogas_cost_tot
                        #BY-PRODUCT INCOME
                        #Oxygen
                        o2_income = sum(o2_wwtp * 32 * aerator_savings * spot_price / (1000*1000)) #[€]
                        # o2_income = (sum(o2_wwtp * 32 * aerator_savings * spot_price / (1000*1000))*0.85) + (sum(o2_wwtp * 32 * 0.7 * spot_price / (1000*1000))*0.15)
                        #Heat
                        # heat_income = (heat_wwtp.sum() - heat_elz_use.sum()) * dh_price / 1000 #[€]
                        #Variable DH price
                        heat_income = (((heat_wwtp[0:1415]-heat_elz_use[0:1415]).sum() + (heat_wwtp[8016:8759]-heat_elz_use[8016:8759]).sum())*dh_winter/1000) + (((heat_wwtp[1416:3623]-heat_elz_use[1416:3623]).sum() + (heat_wwtp[5832:8015]-heat_elz_use[5832:8015]).sum())*dh_spr_aut/1000) + (((heat_wwtp[3624:5831]-heat_elz_use[3624:5831]).sum())*dh_summer/1000) 
                        #assuming replacing hygienization as well:
                        # heat_income = heat_income + (np.average([dh_winter,dh_summer,dh_spr_aut,dh_spr_aut]) * 3037) #[€]
                        #assuming fix utilization factor:
                        # heat_income = np.average([dh_winter,dh_summer,dh_spr_aut,dh_spr_aut]) * heat_prod.sum() * heat_frac_use / 1000
                        #Total
                        BY_INCOME = o2_income + heat_income

                        """ KPI calculations """

                        #ECONOMIC KPIs
                        #LCOE (discounting stack replacement)
                        if stack_rep > 1000: #hours
                            if stack_reps == 1:
                                rep_years = np.array([(math.ceil(stack_rep/elz_flh))])
                            elif stack_reps == 2:
                                rep_years = np.array([(math.ceil(stack_rep/elz_flh)), (math.ceil(2*stack_rep/elz_flh))])
                            elif stack_reps == 3:
                                rep_years = np.array([(math.ceil(stack_rep/elz_flh)), (math.ceil(2*stack_rep/elz_flh)), (math.ceil(3*stack_rep/elz_flh))])
                        else:
                            if stack_reps == 1:
                                rep_years = np.array([stack_rep])
                            elif stack_reps == 2:
                                rep_years = np.array([stack_rep, stack_rep*2])
                            elif stack_reps == 3:
                                rep_years = np.array([stack_rep, stack_rep*2, stack_rep*3])
                            elif stack_reps == 0:
                                rep_years = np.array([0])
                        
                        lcoe = kpis.lcoe(opex=OPEX-BY_INCOME, capex=CAPEX, stack=H2_STACK, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]

                        #Net present value (discounting stack replacement)
                        # INCOME_GAS = BY_INCOME + (gas_price * ch4_p2g) #[€] Income including gas sales
                        # npv = kpis.npv(opex=OPEX_tot, income=INCOME_GAS, capex=CAPEX, stack=H2_STACK, dr=discount, lt=lifetime, stack_reps=stack_reps, rep_years=rep_years) #[€]

                        #LCOP2G (including lost biogas LCOE)
                        lcop2g = kpis.lcoe(opex=OPEX_tot-BY_INCOME, capex=CAPEX, stack=H2_STACK, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        lcop2g_curt = kpis.lcoe(opex=OPEX_tot_curt-BY_INCOME, capex=CAPEX, stack=H2_STACK, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]

                        #Minimum selling price
                        msp = kpis.lcoe(opex=OPEX_msp-BY_INCOME, capex=CAPEX, stack=H2_STACK, dr=discount, lt=lifetime, ch4=ch4_total, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        msp_no_curt = kpis.lcoe(opex=OPEX_msp_nocurt-BY_INCOME, capex=CAPEX, stack=H2_STACK, dr=discount, lt=lifetime, ch4=ch4_total, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        #Comparison to amine scrubbing (Check Vo et al., Ardolino et al., Energiforsk 2016, Angelidaki et al. 2018)
                        #Is it really reasonable to assume the maximum value?
                        amine_flow_rate = 1200 #[Nm3/hr]
                        amine_scrubber_CAPEX = 2500 * amine_flow_rate * nm3_mol #[€]
                        amine_scrubber_el_cost = sum((process['CH4 out [mol/h]'] - process['Meth CH4 in [mol/h]']) * nm3_mol * 0.14 * spot_price / 1000) #[€]
                        amine_scrubber_heat_cost = sum((process['CH4 out [mol/h]'] - process['Meth CH4 in [mol/h]']) * nm3_mol * 0.55 * dh_spr_aut / 1000) #[€]
                        amine_scrubber_opex_fix = amine_scrubber_CAPEX * 0.04
                        amine_scrubber_OPEX = amine_scrubber_el_cost + amine_scrubber_heat_cost + amine_scrubber_opex_fix
                        # npv_rep = kpis.npv(opex=OPEX_tot-amine_scrubber_OPEX, income=INCOME_GAS, capex=CAPEX-amine_scrubber_CAPEX, stack=H2_STACK, dr=discount, lt=lifetime, stack_reps=stack_reps, rep_years=rep_years) #[€]
                        lcoe_amine = kpis.lcoe(opex=amine_scrubber_OPEX, capex=amine_scrubber_CAPEX, stack=0, dr=discount, lt=lifetime, ch4=ch4_total-ch4_p2g, stack_reps=0, rep_years=0) #[€/MWh of CH4]
                        msp_amine = kpis.lcoe(opex=amine_scrubber_OPEX+biogas_cost_tot, capex=amine_scrubber_CAPEX, stack=0, dr=discount, lt=lifetime, ch4=ch4_total-ch4_p2g, stack_reps=0, rep_years=0) #[€/MWh of CH4]


                        #By-product analysis
                        npv_o2 = kpis.npv(opex=o2_integration_OPEX, income=o2_income, capex=o2_integration_CAPEX, stack=0, dr=discount, lt=lifetime, stack_reps=0, rep_years=rep_years) #[€]
                        npv_heat = kpis.npv(opex=heat_integration_OPEX, income=heat_income, capex=heat_integration_CAPEX, stack=0, dr=discount, lt=lifetime, stack_reps=0, rep_years=rep_years) #[€]
                        # lcop2g_noo2 = kpis.lcoe(opex=OPEX_tot-heat_income-o2_integration_OPEX, capex=CAPEX-o2_integration_CAPEX, stack=H2_STACK, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        lcop2g_noo2_curt = kpis.lcoe(opex=OPEX_tot_curt-heat_income-o2_integration_OPEX, capex=CAPEX-o2_integration_CAPEX, stack=H2_STACK, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        msp_noo2_curt = kpis.lcoe(opex=OPEX_msp-heat_income-o2_integration_OPEX, capex=CAPEX-o2_integration_CAPEX, stack=H2_STACK, dr=discount, lt=lifetime, ch4=ch4_total, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        # lcop2g_noheat = kpis.lcoe(opex=OPEX_tot-o2_income-heat_integration_OPEX, capex=CAPEX-heat_integration_CAPEX, stack=H2_STACK, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        lcop2g_noheat_curt = kpis.lcoe(opex=OPEX_tot_curt-o2_income-heat_integration_OPEX, capex=CAPEX-heat_integration_CAPEX, stack=H2_STACK, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        msp_noheat_curt = kpis.lcoe(opex=OPEX_msp-o2_income-heat_integration_OPEX, capex=CAPEX-heat_integration_CAPEX, stack=H2_STACK, dr=discount, lt=lifetime, ch4=ch4_total, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        # lcop2g_nobys = kpis.lcoe(opex=OPEX_tot-o2_integration_OPEX-heat_integration_OPEX, capex=CAPEX-o2_integration_CAPEX-heat_integration_CAPEX, stack=H2_STACK, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        lcop2g_nobys_curt = kpis.lcoe(opex=OPEX_tot_curt-o2_integration_OPEX-heat_integration_OPEX, capex=CAPEX-o2_integration_CAPEX-heat_integration_CAPEX, stack=H2_STACK, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        msp_nobys_curt = kpis.lcoe(opex=OPEX_msp-o2_integration_OPEX-heat_integration_OPEX, capex=CAPEX-o2_integration_CAPEX-heat_integration_CAPEX, stack=H2_STACK, dr=discount, lt=lifetime, ch4=ch4_total, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
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
                        tot_energy_cons = (process['Elz dispatch [kWh/h]'] + process['H2 comp [kWh/h]'] + process['Biogas comp [kWh/h]'] + process['Meth el [kWh/h]'] + process['Cond el [kWh/h]'] + process['Standby el. [kWh/h]']).sum()
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
                        n_max_o2 = ((ch4_p2g * 1000) + ((process['Elz heat [kWh/h]'] + process['Meth heat [kWh/h]']).sum()*usable_heat) + o2_energy_savings) / tot_energy_cons

                        #Upgrading efficiency (including biogas)
                        n_upgrade = ((ch4_total * 1000) + sum(heat_wwtp) + o2_energy_savings) / (tot_energy_cons + (biogas_in[0,:].sum()*ch4_hhv_mol))
                        
                        #ENVIRONMENTAL KPIs
                        #Emissions (INCLUDE OTHER ELECTRICITY CONSUMPTION!)
                        efs_kpi = kpis.efs(bz=bidding_zone, yr=year) #[AEFs, MEFs]
                        aef_ems = ((process['Elz grid [kWh/h]'] * efs_kpi.iloc[:,0] / 1000).sum() + (process['Elz wind [kWh/h]'] * wind_efs / 1000).sum() + (process['Elz PV [kWh/h]'] * pv_efs / 1000).sum()) / ch4_p2g #[kgCO2/MWhCH4]
                        mef_ems = ((process['Elz grid [kWh/h]'] * efs_kpi.iloc[:,1] / 1000).sum() + (process['Elz wind [kWh/h]'] * wind_efs / 1000).sum() + (process['Elz PV [kWh/h]'] * pv_efs / 1000).sum()) / ch4_p2g #[kgCO2/MWhCH4]
                        aef_avg = (aef_ems*ch4_p2g) / ((process['Elz grid [kWh/h]'] / 1000).sum())
                        mef_avg = (mef_ems*ch4_p2g) / ((process['Elz grid [kWh/h]'] / 1000).sum())

                        # aef_ems1 = (((0.5*tot_el*wind_efs + grid_fee) / 1000) + ((0.5*tot_el*pv_efs) / 1000)) / ch4_p2g
                        
                        #Emission reductions from by-product use
                        aef_ems_red_heat = (sum(heat_wwtp) * dh_emissions / 1000) / ch4_p2g #[kgCO2/MWhCH4]
                        mef_ems_red_heat = (sum(heat_wwtp) * dh_emissions_marginal / 1000) / ch4_p2g #[kgCO2/MWhCH4]
                        # aef_ems_red_heat = ((heat_prod.sum() * heat_frac_use) * dh_emissions / 1000) / ch4_p2g #[kgCO2/MWhCH4]
                        # mef_ems_red_heat = ((heat_prod.sum() * heat_frac_use) * dh_emissions_marginal / 1000) / ch4_p2g #[kgCO2/MWhCH4]
                        aef_red_o2 = ((aerator_savings * o2_wwtp * 32 / 1000) * efs_kpi.iloc[:,0]).sum() / (1000*ch4_p2g) #[kgCO2/MWhCH4]
                        mef_red_o2 = ((aerator_savings * o2_wwtp * 32 / 1000) * efs_kpi.iloc[:,1]).sum() / (1000*ch4_p2g) #[kgCO2/MWhCH4]
                        
                        #Emission increase from biogas losses
                        bgloss_ems_increase = (biogas_ef * process['CH4 flared [mol/h]'].sum() * ch4_hhv_mol) / (1000*ch4_p2g) #[kgCO2/MWhCH4]
                        
                        #Net system climate impact
                        aef_net = aef_ems - aef_red_o2 - aef_ems_red_heat + bgloss_ems_increase
                        mef_net = mef_ems - mef_red_o2 - mef_ems_red_heat + bgloss_ems_increase
                        
                        
                        #Cost breakdown table
                        #data
                        total = kpis.lcoe(opex=OPEX_tot_curt, capex=CAPEX, stack=H2_STACK, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
                        elz_lcoe = kpis.lcoe(opex=elz_OPEX, capex=elz_CAPEX, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                        stack_rep_lcoe = kpis.lcoe(opex=0, capex=0, stack=stack_COST, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) * 100 / total #[€/MWh of CH4]
                        water_lcoe = kpis.lcoe(opex=h2o_opex, capex=0, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                        h2st_lcoe = kpis.lcoe(opex=h2st_OPEX, capex=h2st_CAPEX, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                        meth_lcoe = kpis.lcoe(opex=meth_OPEX, capex=meth_CAPEX, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                        comp_lcoe = kpis.lcoe(opex=bg_comp_opex, capex=bg_comp_capex, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                        heat_lcoe = kpis.lcoe(opex=heat_integration_OPEX, capex=heat_integration_CAPEX, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                        o2_lcoe = kpis.lcoe(opex=o2_integration_OPEX, capex=o2_integration_CAPEX, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                        grid_lcoe = kpis.lcoe(opex=grid_cost, capex=0, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                        pv_lcoe1 = kpis.lcoe(opex=pv_cost, capex=0, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                        wind_lcoe1 = kpis.lcoe(opex=wind_cost, capex=0, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                        bg_loss_lcoe = kpis.lcoe(opex=biogas_loss_cost, capex=0, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total
                        install_lcoe = kpis.lcoe(opex=0, capex=INSTALL, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total
                        curt_lcoe1 = kpis.lcoe(opex=curt_cost, capex=0, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                        o2_income_lcoe = kpis.lcoe(opex=-o2_income, capex=0, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
                        heat_income_lcoe = kpis.lcoe(opex=-heat_income, capex=0, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]

                        #table
                        cost_breakdown['{} MW'.format(elz_size)] = [elz_lcoe,stack_rep_lcoe,water_lcoe,h2st_lcoe,meth_lcoe,comp_lcoe,heat_lcoe,o2_lcoe,install_lcoe,bg_loss_lcoe,grid_lcoe,pv_lcoe1,wind_lcoe1,curt_lcoe1,o2_income_lcoe,heat_income_lcoe,lcop2g_curt]

                        if run_type == "optimization":
                            #Saving optimization results
                            # result_series = pd.Series([lcoe, npv, msp, n_gas, n_tot, n_tot_o2, aef_net, mef_net, total_loss_frac, o2_use_frac, o2_wwtp_use_frac, heat_use_frac, heat_wwtp_use_frac, res_frac])
                            # results = pd.concat([results,result_series], axis=1)
                            # results.columns = ['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size,meth_scale,h2st_size,wind_size,pv_size,bat_size)]
                            results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size,meth_scale,h2st_size,wind_size,pv_size,bat_size)] = [lcop2g_curt, lcop2g, msp, msp_no_curt, lcoe, n_gas, n_tot, n_tot_o2, aef_net, mef_net, sum(starts), sum(electrolyzer_standby), elz_flh, total_loss_frac, o2_use_frac, o2_wwtp_use_frac, heat_use_frac, heat_wwtp_use_frac, res_frac, lcop2g_diff_curt, lcop2g_diff_rel_curt, msp_diff_curt, msp_diff_rel_curt, npv_o2, npv_heat]
                            count = count + 1
                            print('{}/{} simulations performed'.format(count,sims))
    



# if run_type == "single":
#     #PRINTING
#     table_kpi = [['LCOP2G', 'MSP', 'Gas eff.', 'O2 eff.', 'AEF net', 'MEF net', 'Loss %', 'RES [%]'], \
#                  [lcop2g_curt, msp, n_gas, n_tot_o2, aef_net, mef_net, total_loss_frac, res_frac]]
#     print(tabulate(table_kpi, headers='firstrow', tablefmt='fancy_grid'))

#     table_by = [['O2 util. [%]', 'O2 dem. [%]', 'Heat util. [%]', '% Heat dem. [%]'], \
#           [o2_use_frac, o2_wwtp_use_frac, heat_use_frac, heat_wwtp_use_frac]]
#     print(tabulate(table_by, headers='firstrow', tablefmt='fancy_grid'))


    # #Costs and energy consumption
    # # Data
    # r = [0,0.25]#,2,3,4]
    # #costs
    # total = kpis.lcoe(opex=OPEX_tot, capex=CAPEX, stack=H2_STACK, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]
    # elz_lcoe = kpis.lcoe(opex=elz_OPEX, capex=elz_CAPEX, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
    # stack_rep_lcoe = kpis.lcoe(opex=0, capex=0, stack=stack_COST, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) * 100 / total #[€/MWh of CH4]
    # water_lcoe = kpis.lcoe(opex=h2o_opex, capex=0, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
    # h2st_lcoe = kpis.lcoe(opex=h2st_OPEX, capex=h2st_CAPEX, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
    # meth_lcoe = kpis.lcoe(opex=meth_OPEX, capex=meth_CAPEX, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
    # comp_lcoe = kpis.lcoe(opex=bg_comp_opex, capex=bg_comp_capex, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
    # heat_lcoe = kpis.lcoe(opex=heat_integration_OPEX, capex=heat_integration_CAPEX, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
    # o2_lcoe = kpis.lcoe(opex=o2_integration_OPEX, capex=o2_integration_CAPEX, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
    # bat_lcoe = kpis.lcoe(opex=bat_OPEX, capex=bat_CAPEX, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
    # grid_lcoe = kpis.lcoe(opex=grid_cost, capex=0, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
    # pv_lcoe = kpis.lcoe(opex=pv_cost, capex=0, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
    # wind_lcoe = kpis.lcoe(opex=wind_cost, capex=0, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total #[€/MWh of CH4]
    # bg_loss_lcoe = kpis.lcoe(opex=biogas_loss_cost, capex=0, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total
    # install_lcoe = kpis.lcoe(opex=0, capex=INSTALL, stack=0, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=0, rep_years=0) * 100 / total
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
    # plt.plot(x, total_heat_demand, label='Total heat demand [kW]')
    # plt.plot(x, heat_demand_bg, label='Digestion heat demand [kW]')
    # plt.legend()
    # plt.show()


    #Dispatch 
    #For poster
    # x1 = 2301+48#500 #starting hour
    # x2 = x1+24#600 #one week later
    # d1 = x1 - x1%24 + 24 #start of the first new day
    # elzload_plot = (h2_flow[x1:x2]*2.02/1000)*100/elz_h2_max
    # h2st_plot = np.array(process['H2 storage [%]'][x1:x2])
    # ep_plot = spot_price[x1:x2]
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
    # ep_plot = spot_price[x1:x2]
    # bg_plot = biogas_flow_arr[x1:x2,1]
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
    # # l4 = ax2.plot(x,spot_price[x1:x2], color='darkorange', label='Electricity price')
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
  #   bg_cm = (biogas_in[0,:].sum()*ch4_hhv_mol/1000)
  #   el_cm = process['Biogas comp [kWh/h]'].sum()/1000
  #   h2_ch4 = process['H2 used [kg/h]'].sum()*39.4/1000
  #   bg_ls = process['CH4 flared [mol/h]'].sum()*ch4_hhv_mol/1000
  #   cm_ch4 = bg_cm+el_cm
  #   el_ch4 = process['Meth el [kWh/h]'].sum()/1000
  #   h2_ht = elz_heat_nonnet.sum()/1000
  #   ch4_ht = process['Meth heat [kWh/h]'].sum()/1000
  #   ch4_ch4 = ch4_total
  #   ht_ww = heat_wwtp.sum()/1000
  #   ht_ls = h2_ht+ch4_ht-ht_ww
  #   o2_ww = o2_energy_savings/1000
  #   o2_ls = aerator_savings * o2_loss * 32 / (1000*1000)
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
    
    
    
    
    
    
    
    
    