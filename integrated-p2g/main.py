# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:19:20 2023

@author: Linus Engstam
"""
import numpy as np
import pandas as pd
import math
import P2G.parameters as params
import P2G.components as comps
import P2G.byproducts as byprods
import P2G.other as other
import P2G.kpis as kpis
import P2G.dispatch as dispatch
import matplotlib.pyplot as plt
from tabulate import tabulate


def P2G_sim_old(
        elz_size,
        meth_size,
        h2st_size,
        wind_size,
        pv_size,
        alpha: float = 0, #determine single or multi objective dispatch optimization, and its focus. 0=cost, 1=emissions
        ef_type: str = "aef",
        run_type: str = "single",
        co2_use: str = "all",
) -> pd.DataFrame:
    """ Returns KPIs for defined component sizes """
    
    """ DEFINING PARAMETERS"""
    #COULD SORT DATA USING CLASS OR DICTIONARY LIKE STRUCT IN MATLAB?
    
    #OVERALL
        #Right now, dispatch is not solved if the electrolyzer is to small to fulfill the demand. How to solve this?
            #Storing between days could be a good method to reduce the electrolyzer size requirements?
        #Currently including grid fee in renewables cost
        #WWTP O2 demand depends a lot on "o2_factor", i.e. how much O2 is not possible to use during the process
        #Have implemented the possibility of not fulfilling the demand
            #Approach: assigned a cost to missed demand, instead of a minimum production demand
        #Note: with no storage or renewable capacity, alpha does not impact the results
        #Implemented minimum selling price which also includes the biogas plant costs, but this is essentially the same thing as the LCOE?
            #Could instead have e.g. an "LCOEtot" and "LCOEp2g" with and without the biogas plant?
        #In "daily" version any missed demand is moved to the following day. Could cause issues for small systems?
        #Have implemented electrolyzer part load and both electrolyzer and methanation startup (standby for elz as well).
    
    #QUESIONS/IDEAS:
        #Questions: should the RES have a cost in the dispatch? Now they don't. Including costs is more interesting for dispatch optimization, but does it make sense?
        #Should we model the heat transfer to the ambient and include ambient temperature etc? Likely too complicated.
            #Assume no such heat transfer!
        #Gas cleaning costs? See Calbry-Muzyka for an idea, perhaps the lowest cost (if H2S is consumed in biological process, otherwise it needs to be removed anyway...)
        #Assigned a cost to lost/flared gas based on an assumed gas price. Does this make sense for LCOE?
            #Maybe not for LCOE, should have another economic parameter as well?
            #E.g the NPV including the biogas plant etc? Since P2G has impact outside its own system borders?
        #Across day storage ideas:
            #Maybe we should end each day with some H2 in the storage to avoid potential peak values at the start of the day?
            #But this could also lead to peak values late in the day being used to charge it up?
            #Now: zero for "all" and usually full for "flex".
        #Multi-objective could be done through CO2 emission costs?
        #HEX efficiency in Bilbao? Produces usable waste heat within ranges mention in Mantei et al. Need to assume well isolated to neglect losses to atmosphere?
            #What about with input water heating included?
        #If we're recycling the water, should it not be at a higher temperature then? I.e. 65 degree water mixed with incoming?
        #METHANE DEMAND: only for P2G system or overall?
        #Could add an electrolyzer efficiency degradation per year in TEA as per van der Roest et al. 2023 simply by increasing the energy consumption?
                #Should relate to stack replacements
        #Should standby and startup costs come form spot prices or fixed?
        #Can we use waste heat to provide energy for stand-by modes?
        
    #ISSUES:
        #Lower efficiency in the "flex" case. Not sure why yet...
        #Storage not working properly, at least in "flex" case, could be a cause to issue above.
            #It seems to not carry value from previous day, and also discharge but fill storage at the same time at the end of the days?
        
    #TO DO:
        #Cooling of hydrogen (and biogas) compressor. Releases more heat, but also means lower temperature when H2 comes from storage!
        #Implement the MILP with part-load efficiency!
            #Adds about 3 seconds for 365 runs or about 20 % of MILP time
            #PWL adds about 27 seconds or ~200 % (~44 seconds in total for MILP only)
            #on/off/standby removes an 7 seconds (~ 37 seconds in total for MILP only)
            #methanation on/off adds 3 seconds (~ 40 seconds in total for MILP only)
            #Need to add the HHV efficiency as well to calculate waste heat generation
            #Need to determine the part-load efficiency within Python based on system efficiency at full load and auxiliary consumption?
        #Consider determining MOO normalization parameters using a ten point sample before the simulation of something like that?
        #Add a MILP also considering the by-product demand and savings/emission reduction
        #Implement emission reductions from by-products in TEA as well
        #Implement more accurate water consumptionn and remove water recycling? Usually ~10kg/kgH2
        #Assume no heat production during cold start as it should heat the stack/too low temp? In Crespi et al. it takes ~10 min for 60 kW, Buttler also mentions that range.
        #More accuratelt determine elz efficiency and new PWL parameters
        
    #Parameters for test run
    elz_size = 10#[1.5,2.0,2.5] #[MW]
    meth_size = 9 #[MW CH4 out]
    h2st_size = 2 #[hours] Should it be defined in hours or kg/MWh?
    wind_size = 0#[0,1,2] #[MW]
    pv_size = 0#[0,1,2] #[MW]
    alpha = 0
    ef_type = "aef"
    run_type = "single" #single/optimization
    co2_use = "daily" #"all"/"flex"/"daily"
    co2_use_fraction = 0.75
    
    # Simulation parameters
    #timestep = 1 #hours
    horizon = "fast day" #"day"/"year"
    year = 2019 #not adapted to 8784 hours...
    bidding_zone = "SE4"
    hv_mode = "HHV" #"LHV" or "HHV"
    by_use = "WWTP"

    # Technical parameters
    #Biogas plant
    biogas_data = "real" #"set"/"real"
    biogas_pres = 1 #bar
    biogas_temp = 40
    #Only used when data = "set":
    biogas_size = 1 #MW (should be Nm3/h?)
    biogas_comp = [0.6, 0.4, 0.0] #[CH4, CO2, CO?, H2O?], not sure if CO is needed? Volume basis?
    
    #Electrolyzer
    #elz_size = 2 #MW
    elz_model = "part-load" #"part-load"/"simple"
    elz_n = 0.7 #HHV system efficiency at full load (part-load is estimated)
    elz_aux = 0.05 #auxiliary component consumption as share of rated power (van der Roest, check other sources). Assuming constant. Could use rectifer consumption, but difficult with the rest...
    elz_min = 0.1 #minimum load
    elz_start = 5 #minutes (cold start)
    elz_start_cost = 0.1 #startup cost [fraction of rated power]
    elz_standby_cost = 0.02 #standby cost [fraction of rated power]
    elz_cooldown = 6 #hours
    elz_heatup_time = elz_start #[minutes] the time during which no usable waste heat is released (look into in more detail)
    elz_temp = 65 #C
    elz_h2o_temp = 15 #[C] inlet water temperature
    elz_pres = 10 #bar
    elz_degr = 0.01 #[% per year/1000 FLHs depending on stack replacement time below] from Ginsberg, strengthened by van der Roest and within ranges of Buttler
    stack_rep = 10 #hours if > 1000, years if < 100. If year, define if it is replaced after ten years or at the start of year ten? After ten years now.
    if hv_mode == "LHV":
        h2_hv = 33.3 #kWh/kgH2
    else:
        h2_hv = 39.4 #kWh/kgH2
    #something related to thermal model (time to heat up etc?)

    #H2 storage
    #h2st_size = 12 #hours
    h2st_n = 1 #Hydrogen storage round-trip efficiency (not implemented yet)

    #Methanation
    meth_type = "bio" #"bio"/"cat"
    #With fixed demand, methanation is sized based on maximum biogas flow.
    # meth_size = 1.3 #MW

    if meth_type == "bio":
        meth_temp = 65 #C
        meth_pres = 7 #bar
        meth_start = 15 #minutes (cold start)
        meth_min = 0.1 #minimum load
        meth_cooldown = 6 #what to define here?
        meth_n = 0.99 #CO2 conversion efficiency
        microb_cons = 0.06 #fraction of extra CO2 required for microbial growth
        meth_standby_cons = 0.1 #fraction of rated electricity input required for stand-by
        meth_el_cons = 0.5 #[kWh/Nm3 CH4 produced] In Schlautmann et al. 2020 for example. 
    elif meth_type == "cat":
        meth_temp = 350 #C
        meth_pres = 10 #bar
        meth_start = 180 #minutes (cold start)
        meth_min = 0.4 #minimum load
        meth_cooldown = 6 #what to define here?
        meth_n = 0.99
        microb_cons = 0

    #Membrane (should be simple, just an output purity or similar?)


    #CO2 storage (Assuming conventional upgrading)
    co2st_size = 0 #[hrs]

    #Heat storage (Not implemented)
    heatst_size = 0 #unit?

    #Oxygen storage (Not implemented)
    o2st_size = 0 #hours
    
    #Battery storage (Not implemented)
    #The battery can increase the amount of renewables used
    bat_size = 0 #hours of full electrolyzer electricity input capacity

    #Electricity
    #wind_size = 1.5 #MW
    #pv_size = 1.5 #MW
    wind_efs = 15.6 #[kgCO2/MWh]
    pv_efs = 30 #[kgCO2/MWh]
    pv_degr = 0 #
    #Grid emissions for electrolyzer dispatch optimization
    if ef_type == "aef":
        efs_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\EFs\efs_' + str(bidding_zone) + '_' + str(year) + '.xlsx'
        efs = pd.read_excel(efs_read)
        efs = efs.iloc[:,0]
    elif ef_type == "mef":
        efs_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\EFs\efs_' + str(bidding_zone) + '_' + str(year) + '.xlsx'
        efs = pd.read_excel(efs_read)
        efs = efs.iloc[:,1]

    #Other
    hex_n = 0.8 #heat exchanger efficiency (van der Roest et al. 2023)
    comp_n = 0.6 #compressor isentropic efficiency
    n_motor = 0.95 #electrical motor efficiency
    cond_n = 1 #condenser efficiency (how much heat can be utilized)

    #By-products
    if by_use == "None":
        other_heat = 0
        o2_demand = 0
        digester_heat = 0
    elif by_use == "Internal":
        other_heat = 0
        o2_demand = "Sell"
        digester_heat = 39 #[kWh/h] #assuming 70 kWh/ton substrate, 4830 tons of substrate per year, constant. (10 % of biogas energy content in Michailos et al.)
    elif by_use == "Market+":
        other_heat = 0
        o2_demand = "Sell"
        digester_heat = 39
    elif by_use == "WWTP" or by_use == "WWTP+O3":
        #Aerators
        #Rusmanis
        aerator_air = 1 / 2.25 #[kWh/kgO2]
        aerator_o2 = 1 / 12.5 #[kWh/kgO2]
        #Donald & Love
        # aerator_air = 1.38 / 0.21 #[kWh/kgO2]
        # aerator_o2 = 1.38
        
        #Income per kg pure O2:
        aerator_income = aerator_air - aerator_o2
        
        wwtp_data = "set"
        #Rusmanis
        o2_demand = 0.0641 #[kgO2/(d*PE)] per day and person equivalent
        #Donald & Love
        o2_demand = np.array([40, 25, 13, 13, 13, 13, 75, 130, 193, 170, 125, 105, 95, 85, 90, 90, 100, 120, 145, 152, 115, 100, 80, 55]) / 26000 #[kgO2/(h*PE)]
        o2_factor = 5.8 #the increase required by use in conventional aeration systems
        #D&L flow
        wwtp_inflow = np.array([120, 80, 60, 60, 60, 60, 165, 285, 330, 285, 265, 225, 225, 202, 202, 202, 225, 270, 310, 330, 310, 270, 225, 160]) * 1000 / 26000 #[L/(h*PE)]
        
        PEs = 191000 #person equivalents Kungsängsverket (2021)
        daily_o2 = o2_demand * o2_factor * PEs * 1000 / 32 #[mol O2/h]
        hourly_o2 = []
        for i in range(int(8760/24)):
            hourly_o2.extend(daily_o2)
        
        if by_use == "WWTP+O3":
            #Ozone demand
            ozonation_savings = 7 #[kWh/kgO3]
            o3_spec = 6 #[mg/L]
            o3_demand = wwtp_inflow * o3_spec / 1000 #[gO3/(h*PE)]
            daily_o3 = o3_demand * PEs / 48 #[mol O3/h]
            hourly_o3 = []
            for i in range(int(8760/24)):
                hourly_o3.extend(daily_o3)
        
        #Heat
        other_heat = 0
        digester_heat = 39
        
    #Gas parameters
    ch4_hhv_vol = 11.05 #kWh/Nm3
    ch4_lhv_vol = 9.94 #kWh/Nm3
    ch4_hhv_kg = 15.44 #kWh/kg
    ch4_lhv_kg = 13.9 #kWh/kg
    ch4_hhv_mol = ch4_hhv_kg / (1000/16.04) #kWh/mol
    ch4_lhv_mol = ch4_lhv_kg / (1000/16.04) #kWh/mol
    nm3_mol = ch4_lhv_mol / ch4_lhv_vol #Nm3/mol
    h2_hhv_vol = 3.53 #kWh/Nm3
    h2_lhv_vol = 3 #kWh/Nm3
    h2_hhv_kg = 39.4 #kWh/kg
    h2_lhv_kg = 33.3 #kWh/kg
    h2_hhv_mol = h2_hhv_kg / (1000/2.02) #kWh/mol
    h2_lhv_mol = h2_lhv_kg / (1000/2.02) #kWh/mol

    # Economical parameters
    #General
    lifetime = 25 #years
    discount = 8 #[%]

    #Biogas plant
    biogas_capex = 1000 #[€/kW CH4]
    biogas_opex = 10 #[% of CAPEX]

    #Electrolyzer
    elz_capex = 1000 #€/kW
    elz_opex = 3 #% of CAPEX
    water_cost = 5 #€/m3 (including purification) Se förstudie för värde
    stack_cost = 0.4 #fraction of CAPEX

    #H2 storage
    h2st_capex = 500 #€/kgH2
    h2st_opex = 1 #% of CAPEX
    h2st_pres = 100 #bar (if it requires high pressure we want to avoid storage when not needed?)

    #Methanation
    if meth_type == "bio":
        meth_capex = 600 #€/kW (kW elz or CH4?) Another potential value in Michailos Table A1.
        meth_opex = 1 #% of CAPEX
    elif meth_type == "cat":
        meth_capex = 600 #€/kW (kW elz or CH4?)
        meth_opex = 1 #% of CAPEX

    #Membrane
    mem_capex = 0
    mem_opex = 0 #% of CAPEX

    #Heat storage
    heatst_capex = 0
    heatst_opex = 0 #% of CAPEX

    #Oxygen storage
    o2st_capex = 0
    o2st_opex = 0 #% of CAPEX
    
    #CO2 storage
    co2st_capex = 0
    co2st_opex = 0 #€ of CAPEX
    
    #Battery storage
    bat_capex = 0
    bat_opex = 0 #% of CAPEX

    #Electricity
    wind_capex = 1000 #€/kW
    wind_opex = 3 #% of capex
    wind_lcoe = 32 #€/MWh
    pv_capex = 1000 #€/kW
    pv_opex = 1 #% of CAPEX
    pv_lcoe = 40 #€/MWh
    grid_fee = 10 #€/MWh (only PV on-site?)

    #Other components
    bg_clean_capex = 0 #(desulfurification and drying)
    comp_capex = 0
    comp_opex = 4 #% of CAPEX (Khan et al.)
    hex_capex = 0
    hex_opex = 0 #% of CAPEX
    cond_capex = 0
    cond_opex = 0

    #By-products
    #Internal heat from pellets: 4.3 kWh heat/kg, efficiency of 90 %, 200 €/ton --> 4.3*0.9*1000 / 200 = 19.35 € ~ 20 €/MWh
    internal_heat_cost = 100 #[€/MWh]
    dh_price = 50 #[€/MWh]
    o2_price = 80 #[€/ton]
    gas_price = 170 #[€/MWh] LHV or HHV?
    h2_value = gas_price * 0.7 #Assuming 30 % losses from H2 to CH4, need to use accurate value if implementing this
    #What about emission reduction from heat use? Both internal and DH?


    """ DEFINING COMPONENTS """

    #Biogas plant
    biogas_flow, biogas_heat = params.biogas_plant(data=biogas_data, size=biogas_size, comp=biogas_comp)
    
    #Electrolyzer
    #what is needed? Part load efficiency, heat production, other parameters for detailed operation model?
    if elz_model == "part-load":
        elz_params = params.electrolyzer(size=elz_size, n=elz_n, min_load=elz_min, startup_time=elz_start, cooldown_time=elz_cooldown, temp=elz_temp, pressure=elz_pres)
        elz_auxiliary = elz_size * 1000 * elz_aux #[kW]
    else:
        elz_params = params.electrolyzer_simple(size=elz_size, n=elz_n, min_load=elz_min, startup_time=elz_start, cooldown_time=elz_cooldown, temp=elz_temp, pressure=elz_pres)
    #Methanation
    #Is cost of methanation based on methane output capacity or electrolyzer electricity input?
    meth_size_mw = max(biogas_flow.iloc[:,1]) * ch4_hhv_mol / 1000 #[MW methane output] 1 mol CO2 in = 1 mol CH4 out, HHV or LHV? Should this be rounded up?
    meth_size_co2 = meth_size * 1000 / ch4_hhv_mol #Maximum CO2 flow rate in mol
    meth_size_vector = np.zeros(24,) + meth_size_co2
    meth_params = params.methanation(size=meth_size, n=meth_n, meth_type=meth_type, min_load=meth_min, startup_time=meth_start, cooldown_time=meth_cooldown, temp=meth_temp, pressure=meth_pres)
    
    #Daily H2 demand
    if co2_use == "daily":
        annual_h2_demand = 4 * biogas_flow.iloc[:,1].sum() * co2_use_fraction * (1-microb_cons)
        daily_h2_demand = (annual_h2_demand / 365) * 2.02 / 1000 #[kg/day]
    
    #Storages
    h2st_cap = h2st_size * elz_size * 1000 * elz_n / h2_lhv_kg #[kg]
    #bgst_cap = bgst_size * max(biogas_flow.iloc[:,1]) * nm3_mol #[Nm3]
    bat_cap = bat_size * elz_size * 1000 #[kWh]
    
    #Membrane
    #what is needed?


    #Heat storage
    #what is needed?


    #Oxygen storage
    #what is needed?
    o2st_cap = (o2st_size * elz_size * 1000 * elz_n / (2*h2_lhv_mol)) * (32/1000) #half the mol of H2, molar mass of O2 is 32/1000 kgO2/mol

    #Electricity
    res_gen = params.renewables(wind_size, pv_size)
    spot_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\elspot prices ' + str(year) + '.xlsx'
    spot_price = pd.read_excel(spot_read) + grid_fee
    spot_price = np.array(spot_price[bidding_zone].tolist())

    #H2 Compressor (temperature increase during operation calculations?) (diaphragm as of now, best choice?)
    h2_flow_max = elz_size * 1000 * elz_n / (h2_lhv_mol) #mol/h
    h2_comp_size = params.compressor(flow=h2_flow_max/3600, temp_in=elz_temp, p_in=elz_pres, p_out=h2st_pres, n_isen=comp_n, n_motor=n_motor)

    #Biogas compressor (using formulas above)
    #Capacity
    bg_flow_max = biogas_flow.sum(axis=1).max() #[mol/h] maximum flow rate
    bg_comp_size = params.compressor(flow=bg_flow_max/3600, temp_in=biogas_temp, p_in=biogas_pres, p_out=meth_pres, n_isen=comp_n, n_motor=n_motor)

    #Heat exchangers
    #should these be including HEX efficiency?
    elz_heat_max = elz_params.iloc[len(elz_params)-1,2]
    meth_heat_max = meth_params.iloc[len(meth_params)-1,2]

    #Condenser
    meth_h20_max = biogas_flow.iloc[0,1] * 2 #4H2 + CO2 --> CH4 + 2H2O
    h2o_heat_of_cond = 40.8/3600 #kWh/mol
    cond_heat_max = meth_h20_max * h2o_heat_of_cond #what about temperature decrease?

    """ PROCESS SIMULATION """
    #Initiate process data
    process = other.data_saving()
    
    #Save independent data
    process['Emissions [gCO$_2$/kWh]'] = efs
    process['Elspot [€/MWh]'] = spot_price
    
    #Initial values for dynamic data
    gas_recirc = np.array([0,0,0])
    T_elz = 20 #assuming ambient temperature
    T_meth = 0
    elz_on = 0 #start in off mode
    meth_on = 0 #start in off mode
    h2_storage = 0
    
    test = []
    # C_NAD = []
    # C_UTP = []
    # E_NAD = []
    # E_UTP = []
    if horizon == "day":
        #Need to save dynamic values from day to day
        for d in range(int(8760/24)): #daily basis
            
            #hour values
            i1 = d*24
            i2 = i1 + 24
            
            #How much H2 do we need? (should be updated using recirulation flow on an hourly basis?)
            #This can refined using recirculation and microbial CO2 consumption (leading to less H2 demand for methanation).
            h2_demand = ((np.array([biogas_flow.iloc[i1:i2,1]]) * 4) * (1-microb_cons))# - (4*gas_recirc[1]) - gas_recirc[0] #4:1 ratio of H2 and CO2 [mol/h] minus recycled CO2 and H2 and microbial consumption
            h2_demand = np.minimum(h2_demand,meth_size_vector*4) #Limited by methanation reactor size
            h2_demand = np.transpose(h2_demand)
            h2_demand_kg = h2_demand * 2.02 / 1000 #mol to kg
            #Also, consider if/when we should fill up the storage ahead of the next day?
            #How to operate electrolyzer to fulfill the demand (if it must be fulfilled)?
            #Need to add an extra cost to storing hydrogen due to compressor (and storage?) efficiency losses!
            
            #If there is a CO2 storage, we can operate the methanation more flexibly. Change how H2 demand is handled
            if co2_use == "flex":
                #Multi och single objective optimization
                if alpha == 0:
                    elz_dispatch = dispatch.grid_res_econ_flex(demand=h2_demand_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, gas_price=h2_value, wind_cost=0, pv_cost=0)# wind_cost=wind_lcoe, pv_cost=pv_lcoe)
                elif alpha == 1:
                    elz_dispatch = dispatch.grid_res_ems_flex(demand=h2_demand_kg, grid=efs[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, wind_ef=wind_efs, pv_ef=pv_efs)
                elif alpha > 0 and alpha < 1:
                    #To determine the nadir and utopian values for cost and emissions, for normalization of multi-objective function
                    elz_dispatch1, cost_utp, ems_nad = dispatch.grid_res_multi_flex(demand=h2_demand_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, alpha=0, efs=efs[i1:i2], wind_ef=0, pv_ef=0, wind_cost=0, pv_cost=0)#wind_cost=wind_lcoe+grid_fee, pv_cost=pv_lcoe+grid_fee, wind_ef=wind_efs, pv_ef=pv_efs,)
                    elz_dispatch2, ems_utp, cost_nad = dispatch.grid_res_multi_flex(demand=h2_demand_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, alpha=1, efs=efs[i1:i2], wind_ef=0, pv_ef=0, wind_cost=0, pv_cost=0)#wind_cost=wind_lcoe+grid_fee, pv_cost=pv_lcoe+grid_fee, wind_ef=wind_efs, pv_ef=pv_efs)                
                    #Using nadir and utopian values to perform multi-objective optimization of electrolyzer dispatch
                    if abs(cost_utp-cost_nad) < 1 or abs(ems_utp-ems_nad) < 1:
                        test.append(1)
                        elz_dispatch = elz_dispatch1
                        continue
                    else:
                        elz_dispatch, __, __ = dispatch.grid_res_multi_flex(demand=h2_demand_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, alpha=alpha, efs=efs[i1:i2], cost_norm=abs(cost_nad-cost_utp), ems_norm=(ems_nad-ems_utp), wind_ef=0, pv_ef=0, wind_cost=0, pv_cost=0)#wind_ef=wind_efs, pv_ef=pv_efs, wind_cost=wind_lcoe+grid_fee, pv_cost=pv_lcoe+grid_fee)
                        test.append(0)
                #elz_prod = h2_demand * (2.02/1000) * 33.3 / 0.6 #kWhel/h
                #elz_dispatch = np.zeros((24,1)) + elz_prod #Temporary grid-only operation fully matching demand
            elif co2_use == "all":
                #Multi och single objective optimization
                if alpha == 0:
                    elz_dispatch = dispatch.grid_res_econ_demand(demand=h2_demand_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, wind_cost=0, pv_cost=0)#wind_cost=wind_lcoe, pv_cost=pv_lcoe)
                elif alpha == 1:
                    elz_dispatch = dispatch.grid_res_ems_demand(demand=h2_demand_kg, grid=efs[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, wind_ef=0, pv_ef=0)#wind_ef=wind_efs, pv_ef=pv_efs)
                elif alpha > 0 and alpha < 1:
                    #To determine the nadir and utopian values for cost and emissions, for normalization of multi-objective function
                    elz_dispatch1, cost_utp, ems_nad = dispatch.grid_res_multi_demand(demand=h2_demand_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, alpha=0, efs=efs[i1:i2], wind_ef=0, pv_ef=0, wind_cost=0, pv_cost=0)#wind_cost=wind_lcoe+grid_fee, pv_cost=pv_lcoe+grid_fee, wind_ef=wind_efs, pv_ef=pv_efs)
                    elz_dispatch2, ems_utp, cost_nad = dispatch.grid_res_multi_demand(demand=h2_demand_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, alpha=1, efs=efs[i1:i2], wind_ef=0, pv_ef=0, wind_cost=0, pv_cost=0)#wind_cost=wind_lcoe+grid_fee, pv_cost=pv_lcoe+grid_fee, wind_ef=wind_efs, pv_ef=pv_efs)                
                    # C_NAD.append(cost_nad)
                    # C_UTP.append(cost_utp)
                    # E_NAD.append(ems_nad)
                    # E_UTP.append(ems_utp)
                    #Using nadir and utopian values to perform multi-objective optimization of electrolyzer dispatch
                    if abs(cost_utp-cost_nad) < 1 or abs(ems_utp-ems_nad) < 1:
                        test.append(1)
                        elz_dispatch = elz_dispatch1
                    else:
                        elz_dispatch, __, __ = dispatch.grid_res_multi_demand(demand=h2_demand_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, alpha=alpha, efs=efs[i1:i2], cost_norm=abs(cost_nad-cost_utp), ems_norm=(ems_nad-ems_utp), wind_ef=0, pv_ef=0, wind_cost=0, pv_cost=0)#wind_ef=wind_efs, pv_ef=pv_efs, wind_cost=wind_lcoe+grid_fee, pv_cost=pv_lcoe+grid_fee)
                        test.append(0)
                #elz_prod = h2_demand * (2.02/1000) * 33.3 / 0.6 #kWhel/h
                #elz_dispatch = np.zeros((24,1)) + elz_prod #Temporary grid-only operation fully matching demand
            
            #Variable storage
            process['H2 demand [mol/h]'][i1:i2] = list(h2_demand)
            process['Elz dispatch [kWh/h]'][i1:i2] = list(elz_dispatch.iloc[:,0])
            process['Elz wind [kWh/h]'][i1:i2] = list(elz_dispatch.iloc[:,2])
            process['Elz PV [kWh/h]'][i1:i2] = list(elz_dispatch.iloc[:,3])
            process['Elz grid [kWh/h]'][i1:i2] = list(elz_dispatch.iloc[:,1])
            if co2_use == "flex":
                process['H2 used [kg/h]'][i1:i2] = list(elz_dispatch.iloc[:,7])
                #process['CO2 dispatch [mol/h]'][i1:i2] = list(elz_dispatch.iloc[:,8])
            elif co2_use == "all":
                process['Unmet demand [kgH2/h]'][i1:i2] = list(elz_dispatch.iloc[:,7])

            if by_use == "WWTP" or by_use == "WWTP+O3":
                process['O2 WWTP [mol/h]'][i1:i2] = list(daily_o2)
                if by_use == "WWTP+O3":
                    process['O3 WWTP [mol/h]'][i1:i2] = list(daily_o3)
            #Daily operation
            for h in range(24):
                #define hour
                hour = (d*24) + h
                #Recirculation? Should methanation come earlier? Dispatch already determined before this loop...
                    #Could it be assumed to even out during an hour? As long as we put in enough H2 for all CO2 (as we do from the beginning)?
                    
                #Hydrogen production (electrolyzer electricity consumption in; flow rate to methanation/storage out, compressor power out, heat out) (what about temp out of storage?)
                #h2_flow, h2st_in, h2st_out, elz_heat, h2_comp_power, T_h2_out, o2_flow, h2o_cons, h2_meth = comps.electrolyzer_simple(dispatch=elz_dispatch.iloc[h,0], params=elz_params, demand=h2_demand[h], temp=elz_temp, p=elz_pres, h2st_p=h2st_pres, size=elz_size*1000, n_isen=comp_n, n_motor=n_motor)
                h2_flow, elz_heat, T_h2_out, o2_flow, h2o_cons = comps.electrolyzer_simple(dispatch=elz_dispatch.iloc[h,0], params=elz_params, temp=elz_temp, p=elz_pres, size=elz_size*1000)
                h2st_in = max(0,(h2_flow-h2_demand[h])) #[mol/h] H2 flow into storage
                h2st_out = min(max(0,(h2_demand[h]-h2_flow)),h2_storage) #[mol/h] H2 flow out from storage, first either zero or positive flow, then either positive flow or all stored gas
                h2_meth = h2_flow - h2st_in + h2st_out #[mol/h] H2 to methanation
                #H2 compression (only what goes into the storage)
                h2_comp_power, T_h2_comp = comps.compressor(flow=h2st_in, temp_in=T_h2_out, p_in=elz_pres, p_out=h2st_pres, n_isen=comp_n, n_motor=n_motor)
                
                #Gas separation: if full demand not fulfilled in "all" case and XXX in "flex" case
                if co2_use == "flex":
                    if h2_meth == 0:
                        p2g_frac = 0
                    else:
                        p2g_frac = process['H2 used [kg/h]'][hour] * 1000 / (2.02 * process['H2 demand [mol/h]'][hour]) #Fraction of biogas flow used in P2G system
                    biogas_in = biogas_flow.iloc[hour,:] * p2g_frac #Biogas flow corresponding to H2 use, i.e. P2G flow
                    bg_flow = biogas_in.sum()
                elif co2_use == "all":
                    #separate excess raw gas for flaring
                    if h2_meth == 0:
                        flare_fraction = 0
                    else:
                        flare_fraction = round(float(1-(h2_meth/(h2_demand[h])[0])),5) #fraction of H2 that was unmet demand
                    bg_flared = biogas_flow.iloc[hour,:] * flare_fraction
                    bg_flow = biogas_flow.sum(axis=1)[hour] * (1-flare_fraction)
                    biogas_in = biogas_flow.iloc[hour,:] * (1-flare_fraction)
                    
                #Biogas compression (flow rate in; compressor power and temp. out(?))
                bg_comp_power, T_bg_comp = comps.compressor(flow=bg_flow, temp_in=biogas_temp, p_in=biogas_pres, p_out=meth_pres, n_isen=comp_n, n_motor=n_motor) #[kWh]
                
                #Gas mixing (Biogas, hydrogen, RECIRCULATION(?) flows, temp. and pressure in; pressure and temp. out?)
                if co2_use == "flex":
                    co2_in = biogas_in[1]# + gas_recirc[1]
                    h2_in = process['H2 used [kg/h]'][hour] * 1000 / 2.02# + gas_recirc[0]
                    ch4_in = biogas_in[0]# + gas_recirc[1]
                elif co2_use == "all":
                    co2_in = biogas_in[1]#biogas_flow.iloc[hour,1]# + gas_recirc[1]
                    h2_in = h2_meth# + gas_recirc[0]
                    ch4_in = biogas_in[0]#biogas_flow.iloc[hour,0]# + gas_recirc[2]
                    
                inlet_flow, T_inlet = comps.mixer(h2=h2_in, co2=co2_in, ch4=ch4_in, h2_temp=T_h2_out, bg_temp=biogas_temp)
                #Pre-heating (temp in and out in, molar flow in (specific heat); energy (heat) consumption out) (is this needed for biological?)
                #Is a preheater necessary or should the gas be heated upon entry to the reactor
                pre_heating = comps.preheater(flow=inlet_flow, T_in=T_inlet, T_out=meth_temp)
                #Methanation (molar flows, temp. in; molar flows, excess heat, electricity consumption out)
                #meth_flow_max = bg_flow_max + h2_flow_max #HOW TO DEFINE THIS? Probably based on CO2/H2 and not total biogas to handle varying CH4/CO2 ratios in the biogas
                meth_flow_max = meth_size_co2
                meth_outlet_flow, meth_power, meth_heat, h2o_cond1, microbial_co2 = comps.methanation(flow=inlet_flow, rated_flow=meth_flow_max, params=meth_params, T=meth_temp, T_in=T_inlet, meth_type=meth_type)
                #Condenser (H2O in, energy out) WATER IS ALREADY CONDENSED IN THE BIOMETHANATION REACTOR, NOT ALL HERE (Goeffart de Roeck et al.)!
                cond_outlet_flow, cond_heat, cond_power, h2o_cond2, T_cond_out = comps.condenser(flow=meth_outlet_flow, n=cond_n, T_in=meth_temp)
                #Gas cleaning (molar flows and temp. (?) in; pure stream and recirculation out, energy consumption?)
                if co2_use == "flex":
                    ch4_out, recirc_flow, gas_loss, T_out, p_out = comps.membrane(inlet_flow=cond_outlet_flow, T_in=T_cond_out, p_in=meth_pres)
                    #difference between recirc_flow and gas_loss?
                elif co2_use == "all":
                    #flaring of any excess raw biogas
                    #if (cond_outlet_flow[2] / cond_outlet_flow.sum()) > 0.98: #if the CH4 content is high enough we do not flare
                    ch4_out = cond_outlet_flow[2]
                    recirc_flow = np.array([0,0,0])
                    #Do we have additional gas losses in other places in the system?
                    h2_loss = 0 + cond_outlet_flow[0] #Flared gas plus any non-CH4 gases remaining in the output gas are considered lost
                    co2_loss = bg_flared[1] + cond_outlet_flow[1]
                    ch4_loss = bg_flared[0]
                    gas_loss = np.array([h2_loss, co2_loss, ch4_loss])
                    T_out = T_cond_out
                    p_out = meth_pres
                        
                    
                #NOW WE CAN STILL CLEAN ANY CH4 CONCENTRATION, MEANING THAT WHEN WE CAN STILL UPGRADE RAW BIOGAS ESSENTIALLY.
                #Biomethanation does usually not require cleaning, but when we don't produce enough some gas should be flared (if the flows are separated before methanation, no separation unit is required?)
                
                #Heat utilization
                #heat_balance = byprods.heat(elz=elz_heat, meth=meth_heat, cond=cond_heat, digester=biogas_heat, pre_heating=pre_heating, other=other_heat) #(-) heat deficit, (+) heat excess
                #Oxygen utilization
                #oxygen_use = byprods.oxygen(flow=o2_flow, demand=o2_demand)
                #Water?
                #Other auxiliary loads (Goeffart de Roeck et al.)?
                
                
                #For next timestep
                #Should maybe redefine this based on how I deal with methanation start-up
                h2_overproduction = max(h2_storage + h2st_in - h2st_out - (h2st_cap*1000/2.02),0) #CURRENTLY NOT RECIRCULATED!
                h2_storage = h2_storage + h2st_in - h2st_out - h2_overproduction
                #Recirculated water
                h2o_recirc = h2o_cond1 + h2o_cond2
                #Recirculated gases
                gas_recirc = gas_loss #H2, CO2, CH4
                #States
                if h2_flow > 0:
                    elz_on = 1
                else:
                    elz_on = 0
                if meth_power > 0:
                    meth_on = 1
                else:
                    meth_on = 0
                #Put on cold standby ifa full day without operation is planned? Should be done outside this loop in that case.
                
                #Previous timestep temperatures, not yet implemented
                T_elz = T_h2_out
                T_meth = 0
        
        
            
                #Variable storage
                process['H2 production [mol/h]'][hour] = h2_flow
                process['H2 to meth [mol/h]'][hour] = h2_in
                process['H2 to storage [mol/h]'][hour] = h2st_in
                process['H2 from storage [mol/h]'][hour] = h2st_out
                if h2st_cap > 0:
                    process['H2 storage [%]'][hour] = (h2_storage/(h2st_cap*1000/2.02))*100 #[%]
                else:
                    process['H2 storage [%]'][hour] = 0
                process['H2 overproduction [mol/h]'][hour] = h2_overproduction
                process['Elz heat [kWh/h]'][hour] = elz_heat
                process['H2 comp [kWh/h]'][hour] = h2_comp_power
                process['H2 temp [C]'][hour] = T_h2_out
                process['O2 out [mol/h]'][hour] = o2_flow
                process['H2O cons [mol/h]'][hour] = h2o_cons
                process['Biogas comp [kWh/h]'][hour] = bg_comp_power
                process['Biogas temp [C]'][hour] = T_bg_comp
                process['Meth CH4 in [mol/h]'][hour] = inlet_flow[2]
                process['Meth H2 in [mol/h]'][hour] = inlet_flow[0]
                process['Meth CO2 in [mol/h]'][hour] = inlet_flow[1]
                process['Meth in temp [C]'][hour] = T_inlet
                process['Preheating [kWh/h]'][hour] = pre_heating
                process['Meth CH4 out [mol/h]'][hour] = meth_outlet_flow[2]
                process['Meth H2 out [mol/h]'][hour] = meth_outlet_flow[0]
                process['Meth CO2 out [mol/h]'][hour] = meth_outlet_flow[1]
                process['Meth H2O(g) out [mol/h]'][hour] = meth_outlet_flow[3]
                process['Meth H2O(l) out [mol/h]'][hour] = h2o_cond1
                process['Meth el [kWh/h]'][hour] = meth_power
                process['Meth heat [kWh/h]'][hour] = meth_heat
                process['Cond CH4 out [mol/h]'][hour] = cond_outlet_flow[2]
                process['Cond H2 out [mol/h]'][hour] = cond_outlet_flow[0]
                process['Cond CO2 out [mol/h]'][hour] = cond_outlet_flow[1]
                process['Cond H2O(l) out [mol/h]'][hour] = h2o_cond2
                process['Cond heat [kWh/h]'][hour] = cond_heat
                process['Cond el [kWh/h]'][hour] = cond_power
                process['H2O recirc [mol/h]'][hour] = h2o_recirc
                process['Cond temp out [C]'][hour] = T_cond_out
                process['CH4 out [mol/h]'][hour] = ch4_out
                process['Recirc CH4 [mol/h]'][hour] = recirc_flow[2]
                process['Recirc H2 [mol/h]'][hour] = recirc_flow[0]
                process['Recirc CO2 [mol/h]'][hour] = recirc_flow[1]
                process['CH4 loss [mol/h]'][hour] = gas_loss[2]
                process['H2 loss [mol/h]'][hour] = gas_loss[0]
                process['CO2 loss [mol/h]'][hour] = gas_loss[1]
                process['Recirc temp [C]'][hour] = T_out
                process['Recirc pres [bar]'][hour] = p_out
                process['Microbial CO2 cons [mol/h]'][hour] = microbial_co2
                if co2_use == "all":
                    process['Flare fraction [-]'][hour] = flare_fraction
    
    if horizon == "fast day":
        #Test to do vectorized calculations for the non-dispatch values
        #Could at the very least move some non-dynamic calculations to this method
        #This could essentially be everything except for methanation and perhaps electrolyzer
            #Unless we have recirculation, but this could be a reason to avoid it
        
        #Dynamic variables
        h2_storage = 0
        elz_on = 0
        elz_standby = 1 #assuming no cold start from initial start
        elz_off = 0
        prev_mode_meth = 1 #assuming no methanation start from inital start
        
        #Converting to numpy arrays
        biogas_flow_arr = np.array(biogas_flow)
        
        #Daily electrolyzer dispatch on day-ahead market      
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
        for d in range(int(8760/24)): #daily basis
            
            #hour values
            i1 = d*24
            i2 = i1 + 24
            
            if co2_use == "all" or co2_use == "flex":
                #How much H2 do we need? (should be updated using recirulation flow on an hourly basis?)
                #This can refined using recirculation and microbial CO2 consumption (leading to less H2 demand for methanation).
                h2_demand = ((np.array([biogas_flow.iloc[i1:i2,1]]) * 4) * (1-microb_cons))# - (4*gas_recirc[1]) - gas_recirc[0] #4:1 ratio of H2 and CO2 [mol/h] minus recycled CO2 and H2 and microbial consumption
                h2_demand = np.minimum(h2_demand,meth_size_vector*4) #Limited by methanation reactor size
                h2_demand = np.transpose(h2_demand)
                h2_demand_kg = h2_demand * 2.02 / 1000 #mol to kg
            elif co2_use == "daily":
                if d == 0:
                    h2_demand = daily_h2_demand * 1000 / 2.02
                else:
                    h2_demand = (daily_h2_demand+unmet_demand[-1]) * 1000 / 2.02 #add any unmet demand from previous day
                h2_demand_kg = daily_h2_demand 
                h2_demand_hr = ((np.array([biogas_flow.iloc[i1:i2,1]]) * 4) * (1-microb_cons))# - (4*gas_recirc[1]) - gas_recirc[0] #4:1 ratio of H2 and CO2 [mol/h] minus recycled CO2 and H2 and microbial consumption
                h2_demand_hr = np.minimum(h2_demand_hr,meth_size_vector*4) #Limited by methanation reactor size
                h2_demand_hr = np.transpose(h2_demand_hr)
                h2_demand_hr_kg = h2_demand_hr * 2.02 / 1000 #mol to kg
                
            #Also, consider if/when we should fill up the storage ahead of the next day?
            #How to operate electrolyzer to fulfill the demand (if it must be fulfilled)?
            #Need to add an extra cost to storing hydrogen due to compressor (and storage?) efficiency losses!
            
            #If there is a CO2 storage, we can operate the methanation more flexibly. Change how H2 demand is handled
            if co2_use == "flex":
                #Multi och single objective optimization
                if alpha == 0:
                    elz_dispatch = dispatch.grid_res_econ_flex(demand=h2_demand_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, gas_price=h2_value, wind_cost=0, pv_cost=0)# wind_cost=wind_lcoe, pv_cost=pv_lcoe)
                elif alpha == 1: #NOT IMPLEMENTED
                    elz_dispatch = dispatch.grid_res_ems_flex(demand=h2_demand_kg, grid=efs[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, wind_ef=0, pv_ef=0)# wind_ef=wind_efs, pv_ef=pv_efs)
                elif alpha > 0 and alpha < 1: #NOT IMPLEMENTED
                    #To determine the nadir and utopian values for cost and emissions, for normalization of multi-objective function
                    elz_dispatch1, cost_utp, ems_nad = dispatch.grid_res_multi_flex(demand=h2_demand_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, alpha=0, efs=efs[i1:i2], wind_ef=0, pv_ef=0, wind_cost=0, pv_cost=0)# efs=efs[i1:i2], wind_cost=wind_lcoe+grid_fee, pv_cost=pv_lcoe+grid_fee, wind_ef=wind_efs, pv_ef=pv_efs)
                    elz_dispatch2, ems_utp, cost_nad = dispatch.grid_res_multi_flex(demand=h2_demand_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, alpha=1, efs=efs[i1:i2], wind_ef=0, pv_ef=0, wind_cost=0, pv_cost=0)#efs=efs[i1:i2], wind_cost=wind_lcoe+grid_fee, pv_cost=pv_lcoe+grid_fee, wind_ef=wind_efs, pv_ef=pv_efs)                
                    #Using nadir and utopian values to perform multi-objective optimization of electrolyzer dispatch
                    if abs(cost_utp-cost_nad) < 1 or abs(ems_utp-ems_nad) < 1:
                        test.append(1)
                        elz_dispatch = elz_dispatch1
                    else:
                        elz_dispatch, __, __ = dispatch.grid_res_multi_flex(demand=h2_demand_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, alpha=alpha, cost_norm=abs(cost_nad-cost_utp), ems_norm=abs(ems_nad-ems_utp), efs=efs[i1:i2], wind_ef=0, pv_ef=0, wind_cost=0, pv_cost=0)#wind_ef=wind_efs, pv_ef=pv_efs, wind_cost=wind_lcoe+grid_fee, pv_cost=pv_lcoe+grid_fee)
                        test.append(0)
                #elz_prod = h2_demand * (2.02/1000) * 33.3 / 0.6 #kWhel/h
                #elz_dispatch = np.zeros((24,1)) + elz_prod #Temporary grid-only operation fully matching demand
            elif co2_use == "all":
                #Multi och single objective optimization
                if alpha == 0:
                    elz_dispatch = dispatch.grid_res_econ_demand(demand=h2_demand_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, wind_cost=0, pv_cost=0)# wind_cost=wind_lcoe, pv_cost=pv_lcoe)
                elif alpha == 1:
                    elz_dispatch = dispatch.grid_res_ems_demand(demand=h2_demand_kg, grid=efs[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, wind_ef=0, pv_ef=0)#wind_ef=wind_efs, pv_ef=pv_efs)
                elif alpha > 0 and alpha < 1:
                    #To determine the nadir and utopian values for cost and emissions, for normalization of multi-objective function
                    elz_dispatch1, cost_utp, ems_nad = dispatch.grid_res_multi_demand(demand=h2_demand_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, alpha=0, efs=efs[i1:i2], wind_ef=0, pv_ef=0, wind_cost=0, pv_cost=0)#wind_ef=wind_efs, pv_ef=pv_efs, wind_cost=wind_lcoe+grid_fee, pv_cost=pv_lcoe+grid_fee)
                    elz_dispatch2, ems_utp, cost_nad = dispatch.grid_res_multi_demand(demand=h2_demand_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, alpha=1, efs=efs[i1:i2], wind_ef=0, pv_ef=0, wind_cost=0, pv_cost=0)#wind_ef=wind_efs, pv_ef=pv_efs, efs=efs[i1:i2], wind_cost=wind_lcoe+grid_fee, pv_cost=pv_lcoe+grid_fee)                
                    # C_NAD.append(cost_nad)
                    # C_UTP.append(cost_utp)
                    # E_NAD.append(ems_nad)
                    # E_UTP.append(ems_utp)
                    #Using nadir and utopian values to perform multi-objective optimization of electrolyzer dispatch
                    if abs(cost_utp-cost_nad) < 1 or abs(ems_utp-ems_nad) < 1:
                        test.append(1)
                        elz_dispatch = elz_dispatch1
                    else:
                        elz_dispatch, __, __ = dispatch.grid_res_multi_demand(demand=h2_demand_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, alpha=alpha, cost_norm=abs(cost_nad-cost_utp), ems_norm=abs(ems_nad-ems_utp), ems_utp=ems_utp, cost_utp=cost_utp, efs=efs[i1:i2], wind_ef=0, pv_ef=0, wind_cost=0, pv_cost=0)#wind_ef=wind_efs, pv_ef=pv_efs, efs=efs[i1:i2], wind_cost=wind_lcoe+grid_fee, pv_cost=pv_lcoe+grid_fee)
                        test.append(0)

            elif co2_use == "daily":
                if elz_model == "simple": #NOT IMPLEMENTED
                    #Multi och single objective optimization
                    if alpha == 0:
                        elz_dispatch = dispatch.grid_res_econ_daily(demand=h2_demand_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, wind_cost=0, pv_cost=0)# wind_cost=wind_lcoe, pv_cost=pv_lcoe)
                    elif alpha == 1:
                        elz_dispatch = dispatch.grid_res_ems_daily(demand=h2_demand_kg, grid=efs[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, wind_ef=0, pv_ef=0)#wind_ef=wind_efs, pv_ef=pv_efs)
                    elif alpha > 0 and alpha < 1:
                        #To determine the nadir and utopian values for cost and emissions, for normalization of multi-objective function
                        elz_dispatch1, cost_utp, ems_nad = dispatch.grid_res_multi_daily(demand=h2_demand_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, alpha=0, efs=efs[i1:i2], wind_ef=0, pv_ef=0, wind_cost=0, pv_cost=0)#wind_ef=wind_efs, pv_ef=pv_efs, wind_cost=wind_lcoe+grid_fee, pv_cost=pv_lcoe+grid_fee)
                        elz_dispatch2, ems_utp, cost_nad = dispatch.grid_res_multi_daily(demand=h2_demand_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, alpha=1, efs=efs[i1:i2], wind_ef=0, pv_ef=0, wind_cost=0, pv_cost=0)#wind_ef=wind_efs, pv_ef=pv_efs, efs=efs[i1:i2], wind_cost=wind_lcoe+grid_fee, pv_cost=pv_lcoe+grid_fee)                
                        # C_NAD.append(cost_nad)
                        # C_UTP.append(cost_utp)
                        # E_NAD.append(ems_nad)
                        # E_UTP.append(ems_utp)
                        #Using nadir and utopian values to perform multi-objective optimization of electrolyzer dispatch
                        if abs(cost_utp-cost_nad) < 1 or abs(ems_utp-ems_nad) < 1:
                            test.append(1)
                            elz_dispatch = elz_dispatch1
                        else:
                            elz_dispatch, __, __ = dispatch.grid_res_multi_daily(demand=h2_demand_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, alpha=alpha, cost_norm=abs(cost_nad-cost_utp), ems_norm=abs(ems_nad-ems_utp), ems_utp=ems_utp, cost_utp=cost_utp, efs=efs[i1:i2], wind_ef=0, pv_ef=0, wind_cost=0, pv_cost=0)#wind_ef=wind_efs, pv_ef=pv_efs, efs=efs[i1:i2], wind_cost=wind_lcoe+grid_fee, pv_cost=pv_lcoe+grid_fee)
                            test.append(0)
                elif elz_model == "part-load":
                    #Check last hour of previous day
                    if elz_on == 1 or elz_standby == 1:
                        prev_mode = 1
                    else:
                        prev_mode = 0
                    if meth_on == 1:
                        prev_mode_meth = 1
                    else:
                        prev_mode_meth = 0
                    #Multi och single objective optimization
                    if alpha == 0:
                        elz_dispatch = dispatch.econ_daily_pl2(demand=h2_demand_kg, hr_demand=h2_demand_hr_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, elz_eff=elz_n, meth_max=meth_size_co2*2.02*4/1000, meth_min=meth_min*meth_size_co2*4*2.02/1000, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, wind_cost=0, pv_cost=0, prev_mode=prev_mode, prev_mode_meth=prev_mode_meth, startup_cost=elz_start_cost, standby_cost=elz_standby_cost)# wind_cost=wind_lcoe, pv_cost=pv_lcoe)
                    elif alpha == 1:
                        elz_dispatch = dispatch.ems_daily_pl(demand=h2_demand_kg, grid=efs[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, wind_ef=0, pv_ef=0)#wind_ef=wind_efs, pv_ef=pv_efs)
                    elif alpha > 0 and alpha < 1:
                        #To determine the nadir and utopian values for cost and emissions, for normalization of multi-objective function
                        elz_dispatch1, cost_utp, ems_nad = dispatch.multi_daily_pl(demand=h2_demand_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, alpha=0, efs=efs[i1:i2], wind_ef=0, pv_ef=0, wind_cost=0, pv_cost=0)#wind_ef=wind_efs, pv_ef=pv_efs, wind_cost=wind_lcoe+grid_fee, pv_cost=pv_lcoe+grid_fee)
                        elz_dispatch2, ems_utp, cost_nad = dispatch.multi_daily_pl(demand=h2_demand_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, alpha=1, efs=efs[i1:i2], wind_ef=0, pv_ef=0, wind_cost=0, pv_cost=0)#wind_ef=wind_efs, pv_ef=pv_efs, efs=efs[i1:i2], wind_cost=wind_lcoe+grid_fee, pv_cost=pv_lcoe+grid_fee)                
                        # C_NAD.append(cost_nad)
                        # C_UTP.append(cost_utp)
                        # E_NAD.append(ems_nad)
                        # E_UTP.append(ems_utp)
                        #Using nadir and utopian values to perform multi-objective optimization of electrolyzer dispatch
                        if abs(cost_utp-cost_nad) < 1 or abs(ems_utp-ems_nad) < 1:
                            test.append(1)
                            elz_dispatch = elz_dispatch1
                        else:
                            elz_dispatch, __, __ = dispatch.multi_daily_pl(demand=h2_demand_kg, grid=spot_price[i1:i2], wind=res_gen.iloc[i1:i2,0], pv=res_gen.iloc[i1:i2,1], elz_max=elz_size*1000, elz_min=elz_min*elz_size*1000, params=elz_params, h2st_max=h2st_cap, h2st_prev=h2_storage, h2_hv=h2_hv, alpha=alpha, cost_norm=abs(cost_nad-cost_utp), ems_norm=abs(ems_nad-ems_utp), ems_utp=ems_utp, cost_utp=cost_utp, efs=efs[i1:i2], wind_ef=0, pv_ef=0, wind_cost=0, pv_cost=0)#wind_ef=wind_efs, pv_ef=pv_efs, efs=efs[i1:i2], wind_cost=wind_lcoe+grid_fee, pv_cost=pv_lcoe+grid_fee)
                            test.append(0)
                
            H2_demand.extend(np.zeros(24) + h2_demand)
            electrolyzer.extend(elz_dispatch.iloc[:,0])
            wind_use.extend(elz_dispatch.iloc[:,2])
            pv_use.extend(elz_dispatch.iloc[:,3])
            grid_use.extend(elz_dispatch.iloc[:,1])
            h2_storage_list.extend(elz_dispatch.iloc[:,5])
            if co2_use == "flex":
                if elz_model == "simple":
                    h2_used.extend(elz_dispatch.iloc[:,7])
            elif co2_use == "all":
                if elz_model == "simple":
                    unmet_demand.extend(elz_dispatch.iloc[:,7])
            elif co2_use == "daily":
                if elz_model == "simple":
                    break
                elif elz_model == "part-load":
                    h2_production.extend(elz_dispatch.iloc[:,7])
                    h2_used.extend(elz_dispatch.iloc[:,8])
                    unmet_demand.extend(np.round(elz_dispatch.iloc[:,9],5))
                    electrolyzer_on.extend(elz_dispatch.iloc[:,11])
                    electrolyzer_standby.extend(elz_dispatch.iloc[:,12])
                    electrolyzer_off.extend(elz_dispatch.iloc[:,13])
                    electrolyzer_start.extend(elz_dispatch.iloc[:,14])
                    elz_on = electrolyzer_on[-1]
                    elz_off = electrolyzer_off[-1]
                    elz_standby = electrolyzer_standby[-1]
                    if h2_used[-1] > 0:
                        meth_on = 1
                    else:
                        meth_on = 0

            h2_storage = h2_storage_list[-1]
                    
        #Hourly operation
        #Converting lists to arrays
        H2_demand = np.asarray(H2_demand)
        H2_demand = H2_demand.reshape(8760,)
        electrolyzer = np.asarray(electrolyzer)
        h2_storage_list = np.asarray(h2_storage_list)
        h2_storage_list_prev = np.roll(h2_storage_list, 1)
        h2_storage_list_prev[0] = 0
        if co2_use == "flex":
            h2_used = np.asarray(h2_used)
        if elz_model == "part-load":
            h2_production = np.asarray(h2_production)
            h2_used = np.asarray(h2_used)
            electrolyzer_start = np.asarray(electrolyzer_start)
       
        #Hydrogen production
        if elz_model == "simple":
            h2_flow, elz_heat, T_h2_out, o2_flow, h2o_cons = comps.electrolyzer_simple(dispatch=electrolyzer, params=elz_params, temp=elz_temp, p=elz_pres, size=elz_size*1000, h20_temp=elz_h2o_temp)
        elif elz_model == "part-load":
            h2_flow, elz_heat, T_h2_out, o2_flow, h2o_cons, stack_eff, sys_eff = comps.electrolyzer(dispatch=electrolyzer, prod=h2_production, aux=elz_aux*elz_size*1000, temp=elz_temp, h2o_temp=elz_h2o_temp, heat_time=elz_heatup_time, startups=electrolyzer_start)
        
        if co2_use == "all" or co2_use == "flex":
            h2st_in = np.maximum(0,(h2_flow-H2_demand)) #[mol/h] H2 flow into storage
            h2st_out = np.minimum(np.maximum(0,(H2_demand-h2_flow)),(h2_storage_list_prev*1000/2.02)) #[mol/h] H2 flow out from storage, first either zero or positive flow, then either positive flow or all stored gas
            h2_meth = h2_flow - h2st_in + h2st_out #[mol/h] H2 to methanation
            #H2 compression (only what goes into the storage). DO we really need this, or can we assume low pressure storage from electrolyzer ouptut pressure only?
        elif co2_use == "daily":
            h2st_in = np.maximum(0,(h2_production)-h2_used) * 1000 / 2.02
            h2st_out = np.minimum(np.maximum(0,h2_used-h2_production),(h2_storage_list_prev*1000/2.02)) * 1000 / 2.02
            h2_meth = h2_used * 1000 / 2.02
        
        h2_comp_power, T_h2_comp = comps.compressor(flow=h2st_in, temp_in=T_h2_out, p_in=elz_pres, p_out=h2st_pres, n_isen=comp_n, n_motor=n_motor)

        #Gas separation: if full demand not fulfilled in "all" case and XXX in "flex" case
        if co2_use == "flex":
            #Fraction of biogas flow used in P2G system
            p2g_frac = np.divide((h2_used*1000), (2.02*H2_demand), out=np.zeros_like((h2_used*1000)), where=(2.02*H2_demand)!=0)
            biogas_in = biogas_flow_arr.T * p2g_frac #Biogas flow corresponding to H2 use, i.e. P2G flow
            bg_flow = biogas_in.sum(axis=0)
        elif co2_use == "all":
            #separate excess raw gas for flaring
            flare_fraction = np.around(1 - np.divide(h2_meth, H2_demand, out=np.zeros_like(h2_meth), where=H2_demand!=0),5)
            bg_flared = biogas_flow_arr.T * flare_fraction
            bg_flow = biogas_flow_arr.sum(axis=1).T * (1-flare_fraction)
            biogas_in = biogas_flow_arr.T * (1-flare_fraction)
        elif co2_use == "daily":
            co2_flow = h2_used * 1000 / (2.02*(1-microb_cons)*4)
            p2g_frac = np.divide(co2_flow, biogas_flow_arr[:,1], out=np.zeros_like((electrolyzer)), where=biogas_flow_arr[:,1]!=0)
            biogas_in = biogas_flow_arr.T * p2g_frac
            bg_flow = biogas_in.sum(axis=0)
                
        #Biogas compression (flow rate in; compressor power and temp. out(?))
        bg_comp_power, T_bg_comp = comps.compressor(flow=bg_flow, temp_in=biogas_temp, p_in=biogas_pres, p_out=meth_pres, n_isen=comp_n, n_motor=n_motor) #[kWh]
        
        #Gas mixing (Biogas, hydrogen, RECIRCULATION(?) flows, temp. and pressure in; pressure and temp. out?)
        if co2_use == "flex":
            co2_in = biogas_in[1]# + gas_recirc[1]
            h2_in = h2_used * 1000 / 2.02# + gas_recirc[0]
            ch4_in = biogas_in[0]# + gas_recirc[1]
        elif co2_use == "all" or co2_use == "daily":
            co2_in = biogas_in[1]#biogas_flow.iloc[hour,1]# + gas_recirc[1]
            h2_in = h2_meth# + gas_recirc[0]
            ch4_in = biogas_in[0]#biogas_flow.iloc[hour,0]# + gas_recirc[2]
            
        inlet_flow, T_inlet = comps.mixer(h2=h2_in, co2=co2_in, ch4=ch4_in, h2_temp=T_h2_out, bg_temp=biogas_temp)
        
        #Pre-heating (temp in and out in, molar flow in (specific heat); energy (heat) consumption out) (is this needed for biological?)
        pre_heating = comps.preheater(flow=inlet_flow, T_in=T_inlet, T_out=meth_temp)
        
        #Methanation (molar flows, temp. in; molar flows, excess heat, electricity consumption out)
        #meth_flow_max = bg_flow_max + h2_flow_max #HOW TO DEFINE THIS? Probably based on CO2/H2 and not total biogas to handle varying CH4/CO2 ratios in the biogas
        meth_flow_max = meth_size_co2
        meth_outlet_flow = []
        meth_power = []
        meth_heat = []
        h2o_cond1 = []
        microbial_co2 = []
        for h in range(len(h2_flow)):
            meth_outlet_flow_h, meth_power_h, meth_heat_h, h2o_cond1_h, microbial_co2_h = comps.methanation(flow=inlet_flow.T[h], rated_flow=meth_flow_max, params=meth_params, T=meth_temp, T_in=T_inlet[h], meth_type=meth_type, el_cons=meth_el_cons)
            meth_outlet_flow.append(meth_outlet_flow_h)
            meth_power.append(meth_power_h)
            meth_heat.append(meth_heat_h)
            h2o_cond1.append(h2o_cond1_h)
            microbial_co2.append(microbial_co2_h)
            
            #Do something with the dynamic variables here?
            
        meth_outlet_flow = np.asarray(meth_outlet_flow)
        h2o_cond1 = np.asarray(h2o_cond1)
        
        #OVERRODUCING H2?
        
        #Condenser (H2O in, energy out) WATER IS ALREADY CONDENSED IN THE BIOMETHANATION REACTOR, NOT ALL HERE (Goeffart de Roeck et al.)!
        cond_outlet_flow, cond_heat, cond_power, h2o_cond2, T_cond_out = comps.condenser(flow=meth_outlet_flow.T, n=cond_n, T_in=meth_temp)
        
        #Gas cleaning (molar flows and temp. (?) in; pure stream and recirculation out, energy consumption?)
        if co2_use == "flex" or co2_use == "daily":
            ch4_out, recirc_flow, gas_loss, T_out, p_out = comps.membrane(inlet_flow=cond_outlet_flow, T_in=T_cond_out, p_in=meth_pres)
            #now there is no recirculation, only losses
        elif co2_use == "all":
            #flaring of any excess raw biogas
            #if (cond_outlet_flow[2] / cond_outlet_flow.sum()) > 0.98: #if the CH4 content is high enough we do not flare
            ch4_out = cond_outlet_flow[2]
            recirc_flow = np.zeros_like(cond_outlet_flow)
            #Do we have additional gas losses in other places in the system?
            h2_loss = 0 + cond_outlet_flow[0] #Flared gas plus any non-CH4 gases remaining in the output gas are considered lost
            co2_loss = bg_flared[1] + cond_outlet_flow[1]
            ch4_loss = bg_flared[0]
            gas_loss = np.array([h2_loss, co2_loss, ch4_loss])
            T_out = np.zeros_like(ch4_loss) + T_cond_out
            p_out = np.zeros_like(ch4_loss) + meth_pres
            
        
        #Other variables
        h2_overproduction = np.maximum(h2_storage_list + h2st_in - h2st_out - (h2st_cap*1000/2.02),0) #CURRENTLY NOT RECIRCULATED!
        #h2_storage = h2_storage + h2st_in - h2st_out - h2_overproduction
        h2_storage_list = h2_storage_list - h2_overproduction
        #Recirculated water
        h2o_recirc = np.zeros(8760,) #h2o_cond1 + h2o_cond2 (assuming no recirculation)
        #Recirculated gases
        #gas_recirc = gas_loss #H2, CO2, CH4
        
        #Storing results
        process['Biogas (CH4) [mol/h]'] = list(biogas_flow.iloc[:,0])
        process['Biogas (CO2) [mol/h]'] = list(biogas_flow.iloc[:,1])
        process['H2 demand [mol/h]'] = list(H2_demand)
        process['Elz dispatch [kWh/h]'] = list(electrolyzer)
        process['Elz wind [kWh/h]'] = list(wind_use)
        process['Elz PV [kWh/h]'] = list(pv_use)
        process['Elz grid [kWh/h]'] = list(grid_use)
        if co2_use == "flex" or co2_use == "daily":
            process['H2 used [kg/h]'] = list(h2_used)
            #process['CO2 dispatch [mol/h]'][i1:i2] = list(elz_dispatch.iloc[:,8])
        if co2_use == "all" or co2_use == "daily":
            process['Unmet demand [kgH2/h]'] = list(unmet_demand)

        if by_use == "WWTP" or by_use == "WWTP+O3":
            process['O2 WWTP [mol/h]'] = list(hourly_o2)
            if by_use == "WWTP+O3":
                process['O3 WWTP [mol/h]'] = list(hourly_o3)
                
        process['H2 production [mol/h]'] = h2_flow
        process['H2 to meth [mol/h]'] = h2_in
        process['H2 to storage [mol/h]'] = h2st_in
        process['H2 from storage [mol/h]'] = h2st_out
        if h2st_cap > 0:
            process['H2 storage [%]'] = (h2_storage_list/(h2st_cap))*100 #[%]
        else:
            process['H2 storage [%]'] = 0
        process['H2 overproduction [mol/h]'] = h2_overproduction
        process['Elz heat [kWh/h]'] = elz_heat
        process['H2 comp [kWh/h]'] = h2_comp_power
        process['H2 temp [C]'] = T_h2_out
        process['O2 out [mol/h]'] = o2_flow
        process['H2O cons [mol/h]'] = h2o_cons
        process['Biogas comp [kWh/h]'] = bg_comp_power
        process['Biogas temp [C]'] = T_bg_comp
        process['Meth CH4 in [mol/h]'] = inlet_flow[2]
        process['Meth H2 in [mol/h]'] = inlet_flow[0]
        process['Meth CO2 in [mol/h]'] = inlet_flow[1]
        process['Meth in temp [C]'] = T_inlet
        process['Preheating [kWh/h]'] = pre_heating
        process['Meth CH4 out [mol/h]'] = meth_outlet_flow.T[2]
        process['Meth H2 out [mol/h]'] = meth_outlet_flow.T[0]
        process['Meth CO2 out [mol/h]'] = meth_outlet_flow.T[1]
        process['Meth H2O(g) out [mol/h]'] = meth_outlet_flow.T[3]
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
        if co2_use == "all":
            process['Flare fraction [-]'] = flare_fraction
        if elz_model == "part-load":
            process['Stack efficiency [%]'] = stack_eff * 100
            process['System efficiency [%]'] = sys_eff * 100
            process['Elz standby'] = electrolyzer_standby    
            
            

    
    """ TECHNICAL ANALYSIS """

    #Gas production
    if hv_mode == "LHV":
        ch4_p2g = (process['CH4 out [mol/h]'] - process['Meth CH4 in [mol/h]']).sum() * ch4_lhv_mol / 1000 #[MWh] Annual CH4 production increase from P2G
        ch4_total = process['CH4 out [mol/h]'].sum() * ch4_lhv_mol / 1000
        #Gas loss
        ch4_loss = process['CH4 loss [mol/h]'].sum() * ch4_lhv_mol
        loss_frac = (process['CH4 loss [mol/h]'].sum() / biogas_flow.iloc[:,0].sum()) * 100 #[%]
    elif hv_mode == "HHV":
        ch4_p2g = (process['CH4 out [mol/h]'] - process['Meth CH4 in [mol/h]']).sum() * ch4_hhv_mol / 1000 #[MWh] Annual CH4 production increase from P2G
        ch4_total = process['CH4 out [mol/h]'].sum() * ch4_hhv_mol / 1000
        #Gas loss
        ch4_loss = process['CH4 loss [mol/h]'].sum() * ch4_hhv_mol
        loss_frac = (process['CH4 loss [mol/h]'].sum() / biogas_flow.iloc[:,0].sum()) * 100 #[%]    
    
    #Operation
    #Stack replacement
    elz_flh = round(process['Elz dispatch [kWh/h]'].sum() / (elz_size*1000)) #full load hours of the electrolyzer
    if stack_rep > 1000: #counting hours
        stack_reps = math.floor((elz_flh*lifetime) / stack_rep) #number of stack replacements during project lifetime
    else:
        stack_reps = math.floor(lifetime / stack_rep) #number of stack replacements during project lifetime
    #COULD MAKE SURE NO LATE REPLACEMENTS BY ROUNDING DOWN TO PREVIOUS NUMBER IF ONLY AT FOR EXAMPLE 2.1
    
    #ELECTRICITY USE
    
    #number of starts
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
    
    #methanation stand-by(off)
    meth_standby = []
    for i in range(len(process['Elz dispatch [kWh/h]'])):
        if process['Meth el [kWh/h]'][i] > 0:
            meth_standby.append(0)
        else:
            meth_standby.append(1)

    #STORAGE
    #Number of full storage cycles
    if h2st_cap > 0:
        h2st_cycles = process['H2 to storage [mol/h]'].sum() * 2.02 / (h2st_cap * 1000)
    else:
        h2st_cycles = 0
    #Could be interesting to see how often it is above e.g. 90 %?

    #HEAT
    #Defining the available waste heat (what to do with high-grade (högtempererad) heat?)
    if meth_type == "bio":
        lg_heat = hex_n * np.array(process['Elz heat [kWh/h]'] + process['Meth heat [kWh/h]'] + process['Cond heat [kWh/h]']) #low-grade heat produced [kWh/h]
        #hg_heat = hex_n * np.zeros(8760,) #high-grade heat produced [kWh/h]
    elif meth_type == "cat":
        lg_heat = hex_n * process['Elz heat [kWh/h]'] + process['Cond heat [kWh/h]'] #low-grade heat produced [kWh/h]
        #hg_heat = hex_n * process['Meth heat [kWh/h]'] #high-grade heat produced [kWh/h]

    #Internal heat uses
    lg_heat_demand = np.zeros(8760) + digester_heat
    #hg_heat_demand = np.zeros(8760)

    
    #OXYGEN
    o2_wwtp = []
    o3_wwtp = []
    if by_use == "WWTP":
        for i in range(len(process['O2 out [mol/h]'])):
            o2_wwtp.append(min(process['O2 out [mol/h]'][i], process['O2 WWTP [mol/h]'][i]))
        o2_wwtp = np.array(o2_wwtp)
        o2_waste = np.minimum(process['O2 out [mol/h]'] - o2_wwtp,0)
    elif by_use == "WWTP+O3": #SHOULD O2 OR O3 BE PRIORITIZED? DEPENDS ON VALUE
        for i in range(len(process['O2 out [mol/h]'])):
            o2_wwtp.append(min(process['O2 out [mol/h]'][i], process['O2 WWTP [mol/h]'][i]))
            o3_wwtp.append(min(max(process['O2 out [mol/h]'][i] - o2_wwtp[i],0), process['O3 WWTP [mol/h]'][i] * 12.45)) #O2 demand per O3 from Peyrelasse et al. 2021
        o2_wwtp = np.array(o2_wwtp)
        o3_wwtp = np.array(o3_wwtp)
        o2_waste = np.minimum(process['O2 out [mol/h]'] - o2_wwtp - o3_wwtp,0)
    

    """ ECONOMIC ANALYSIS """
    #Include DH here if excess heat, and some heat source if deficit
    #Should also move to separate script?
    #What to do with auxiliary electricity use (h2 and bg compressors etc)? Now included in OPEX, but emissions etc?
    
    #Electrolyzer degradation
    # if elz_degr > 0: #calculations must be done over stack lifetime, and then averaged
    #     for y in range(math.ceil(stack_rep/elz_flh)): #before first stack replacement, rounding up
    #         #Degradation factor (ONLY STACK THAT DEGRADES, CURRENTLY CHANGING THE SYSTEM EFFICIENCY)
    #         if y == 0:
    #             degr_factor = 1 #based on initial efficiency
    #         else:
    #             degr_factor = degr_factor * math.pow(1-elz_degr,elz_flh/1000) #based on previous year
    #         #Stack replacement (CONSIDER WHERE THIS SHOULD TAKE PLACE)
    #         if y == math.ceil(stack_rep/elz_flh) or y == math.ceil(stack_rep*2/elz_flh) or y == math.ceil(stack_rep*3/elz_flh): #assuming stack replacement takes place at the end of the supposed year
    #             degr_factor = 1 #resetting degradation after stack replacement
    #         #ELECTRICITY
    #         wind_cost = (((process['Elz wind [kWh/h]']*(process['System efficiency [%]']/100)/(process['Stack efficiency [%]']*degr_factor/100))+elz_auxiliary) * (wind_lcoe + grid_fee)).sum() / 1000
    #         pv_cost = (((process['Elz PV [kWh/h]']*(process['System efficiency [%]']/100)/(process['Stack efficiency [%]']*degr_factor/100))+elz_auxiliary) * (pv_lcoe)).sum() / 1000 #[€] assuming no grid fee for local PV
    #         grid_cost = (((process['Elz grid [kWh/h]']*(process['System efficiency [%]']/100)/(process['Stack efficiency [%]']*degr_factor/100))+elz_auxiliary) * spot_price).sum() / 1000 #[€] grid fee already included
            
    #         #(MAYBE NEED TO INCLUDE START-UP COSTS IF THEY ARE DEFINED BASED ON THE EFFICIENCY)
            
    #         #HEAT
    #         avoided_lg_heat = np.minimum(lg_heat_demand, lg_heat)
    #         lg_heat_rem = lg_heat - avoided_lg_heat
    #         avoided_heat_income = (avoided_lg_heat).sum() * internal_heat_cost / 1000 #[€]
            
    #         #EFFICIENCY
            
            
    #         #Total sum (to be divided with lifetime for average values)
            
                
            

    # else:
    #     #ELECTRICITY
    #     wind_cost = (process['Elz wind [kWh/h]'] * (wind_lcoe + grid_fee)).sum() / 1000
    #     pv_cost = (process['Elz PV [kWh/h]'] * (pv_lcoe + grid_fee)).sum() / 1000
    #     grid_cost = (process['Elz grid [kWh/h]'] * spot_price).sum() / 1000 #[€] grid fee already included
        
    #     #BY-PRODUCTS
    #     avoided_lg_heat = np.minimum(lg_heat_demand, lg_heat)
    #     lg_heat_rem = lg_heat - avoided_lg_heat
    #     avoided_heat_income = (avoided_lg_heat).sum() * internal_heat_cost / 1000 #[€]

    #ELECTRICITY
    #Renewables (could use LCOE)
    # wind_farm_capex = wind_size * wind_capex
    # wind_farm_opex = wind_farm_capex * wind_opex / 100
    # pv_farm_capex = pv_size * pv_capex
    # pv_farm_opex = pv_farm_capex * pv_opex / 100
    wind_cost = (process['Elz wind [kWh/h]'] * (wind_lcoe + grid_fee)).sum() / 1000
    pv_cost = (process['Elz PV [kWh/h]'] * (pv_lcoe + grid_fee)).sum() / 1000
    #Grid
    grid_cost = (process['Elz grid [kWh/h]'] * spot_price).sum() / 1000 #[€] grid fee already included
    #Auxiliary electrolyzer costs
    #STARTUP AND STANDBY COSTS
    #Start-up costs
    startup_costs = (elz_size * elz_start_cost * spot_price * starts).sum()
    #Standby costs
    standby_costs = (process['Elz standby'] * elz_standby_cost * elz_size * spot_price).sum()
    #Overall
    el_cost = wind_cost + pv_cost + grid_cost + startup_costs + standby_costs
    
    #Battery
    bat_CAPEX = bat_cap * bat_capex
    bat_OPEX = bat_cap * (bat_opex/100)
    
    #HEAT
    #Heating costs
    #Methanation preheating (SHOULD BE INCLUDED AS INTERNAL HEAT DEMAND)
    preheat_OPEX = internal_heat_cost * process['Preheating [kWh/h]'].sum() / 1000
    #Heat use
    internal_heat = other_heat 

    #HYDROGEN
    #Electrolyzer
    elz_CAPEX = elz_capex * elz_size * 1000
    elz_OPEX = (elz_opex*0.01*elz_CAPEX)
    #Water
    h2o_opex = water_cost * (process['H2O cons [mol/h]']-process['H2O recirc [mol/h]']).sum() * 18.02 / (1000*997) # €/m3 * mol * g/mol / (1000*kg/m3)
    h2o_opex = water_cost * (process['H2O cons [mol/h]']).sum() * 18.02 / (1000*997) # €/m3 * mol * g/mol / (1000*kg/m3)
    #what about when recirculated doesn't match electrolyzer operation?
    #Stack replacement
    stack_COST = stack_reps * stack_cost * elz_CAPEX #total cost of stack replacements
    #Storage
    h2st_CAPEX = h2st_cap * h2st_capex
    h2st_OPEX = h2st_opex * 0.01 * h2st_CAPEX
    #Compressor (USD!!!)
    h2_comp_capex = (63684.6 * (h2_comp_size**0.4603) * 1.3) * 0.75 #(Khan et al. ch. 5.2 with CAD to USD conversion)
    h2_comp_opex = comp_opex * 0.01 * h2_comp_capex
    h2_comp_el = (process['H2 comp [kWh/h]'] * spot_price).sum() / 1000
    #Total hydrogen costs
    H2_CAPEX = elz_CAPEX + h2st_CAPEX + h2_comp_capex
    H2_OPEX = elz_OPEX + h2o_opex + h2st_OPEX + h2_comp_opex + h2_comp_el
    H2_STACK = stack_COST

    #BIOGAS
    biogas_CAPEX = biogas_capex * max(biogas_flow.iloc[:,0]) * ch4_hhv_mol #[€]
    biogas_OPEX = biogas_CAPEX * biogas_opex / 100 #[€]

    #METHANATION
    #Reactor
    meth_CAPEX = meth_capex * meth_size * 1000
    meth_opex_fix = meth_opex * 0.01 * meth_CAPEX #fixed opex
    meth_el = (process['Meth el [kWh/h]'] * spot_price).sum() / 1000
    meth_standby = (np.array(meth_standby) * meth_standby_cons * nm3_mol * spot_price).sum()
    meth_OPEX = meth_opex_fix + meth_el + meth_standby
    #Compressor (USD!!!)
    bg_comp_capex = (63684.6 * (bg_comp_size**0.4603) * 1.3) * 0.75 #(Khan et al. ch. 5.2 with CAD to USD conversion)
    bg_comp_opex = comp_opex * 0.01 * bg_comp_capex
    bg_comp_el = (process['Biogas comp [kWh/h]'] * spot_price).sum() / 1000
    #Total methanation costs
    METH_CAPEX = meth_CAPEX + bg_comp_capex
    METH_OPEX = meth_OPEX + bg_comp_opex + bg_comp_el

    #GAS LOSSES
    loss_cost = 0 #process['CH4 loss [mol/h]'].sum() * ch4_hhv_mol * gas_price #LHV or HHV?

    #HEAT EXCHANGERS AND CONDENSERS (could assume an aggregated cost for these as a percentage of the other costs)
    elz_hex_capex = elz_heat_max * hex_capex
    meth_hex_capex = meth_heat_max * hex_capex
    meth_cond_capex = cond_heat_max * cond_capex
    elz_hex_opex = (elz_hex_capex * 0.01 * hex_opex) + 0 #do they consume electricity?
    meth_hex_opex = (meth_hex_capex * 0.01 * hex_opex) + 0 #do they consume electricity?
    meth_cond_opex = (cond_capex * 0.01 * cond_opex) + 0 #do they consume electricity?
    hex_CAPEX = elz_hex_capex + meth_hex_capex + meth_cond_capex
    hex_OPEX = elz_hex_opex + meth_hex_opex + meth_cond_opex
                      
    #OVERALL COSTS
    CAPEX = H2_CAPEX + METH_CAPEX + hex_CAPEX + bat_CAPEX
    OPEX = H2_OPEX + METH_OPEX + hex_OPEX + el_cost + bat_OPEX + loss_cost
    #Including biogas plant costs
    CAPEX_tot = CAPEX + biogas_CAPEX
    OPEX_tot = OPEX + biogas_OPEX
    
    #INCOME (By-products)

    #Avoided heat production (should preheating be considered? Very small amount)
    #could use high-grade heat for low-grade applications?
    #Heat exchanger efficiency?
    avoided_lg_heat = np.minimum(lg_heat_demand, lg_heat)
    lg_heat_rem = lg_heat - avoided_lg_heat
    #avoided_hg_heat = np.minimum(hg_heat_demand, hg_heat)
    #hg_heat_rem = hg_heat - avoided_hg_heat

    avoided_heat_income = (avoided_lg_heat).sum() * internal_heat_cost / 1000 #[€]

    #District heating (what can not be used internally is sold)
    #dh_sales = (lg_heat_rem).sum() * dh_price / (1000) #[€]

    #Oxygen
    if by_use == "Market+":
        o2_sales = process['O2 out [mol/h]'].sum() * 32 * o2_price / (1000*1000) #[€] converting mol to ton O2
    elif by_use == "WWTP":
        o2_sales = (o2_wwtp * 32 * aerator_income * spot_price / (1000*1000)).sum() #[€]
    elif by_use == "WWTP+O3":
        o2_sales = (o2_wwtp * 32 * aerator_income * spot_price / (1000*1000)).sum() #[€]
        o3_sales = (o3_wwtp * 48 * ozonation_savings * spot_price / (1000*1000)).sum() #[€]
    
    
    #Total income
    if by_use != "WWTP+O3":
        INCOME_BY = avoided_heat_income + o2_sales
    else:
        INCOME_BY = avoided_heat_income + o2_sales + o3_sales

    """ KPI calculations """

    #ECONOMIC KPIs
    #LCOE (discounting stack replacement)
    #Does the row below make sense? Am I assuming there are always two replacements?
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
    lcoe = kpis.lcoe(opex=OPEX - INCOME_BY, capex=CAPEX, stack=H2_STACK, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]

    #Net present value (discounting stack replacement)
    INCOME_GAS = INCOME_BY + (gas_price * ch4_p2g) #[€] Income including gas sales
    npv = kpis.npv(opex=OPEX, income=INCOME_GAS, capex=CAPEX, stack=H2_STACK, dr=discount, lt=lifetime, ch4=ch4_p2g, stack_reps=stack_reps, rep_years=rep_years) #[€]

    #Minimum selling price (including biogas plant costs)
    msp = kpis.msp(opex=OPEX_tot-INCOME_BY, capex=CAPEX_tot, stack=H2_STACK, dr=discount, lt=lifetime, ch4=ch4_total, stack_reps=stack_reps, rep_years=rep_years) #[€/MWh of CH4]

    #Comparison to fossil use? I.e. transport etc. Include ETS price


    #TECHNICAL KPIs
    #System efficiency
    #Methane/Electrivity
    n_gas = (ch4_p2g * 1000) / (process['Elz dispatch [kWh/h]'] + process['H2 comp [kWh/h]'] + process['Biogas comp [kWh/h]'] + process['Preheating [kWh/h]'] + process['Meth el [kWh/h]'] + process['Cond el [kWh/h]']).sum()

    #Including heat
    n_tot = ((ch4_p2g * 1000) + (process['Elz heat [kWh/h]'] + process['Meth heat [kWh/h]'] + process['Cond heat [kWh/h]']).sum()) / (process['Elz dispatch [kWh/h]'] + process['H2 comp [kWh/h]'] + process['Biogas comp [kWh/h]'] + process['Preheating [kWh/h]'] + process['Meth el [kWh/h]'] + process['Cond el [kWh/h]']).sum()

    #WHAT ABOUT GAS LOSSES?

    #Including avoided use (does this make sense for internal oxygen use?)


    #ENVIRONMENTAL KPIs
    #Emissions
    efs_kpi = kpis.efs(bz=bidding_zone, yr=year) #[AEFs, MEFs]
    aef_ems = ((process['Elz grid [kWh/h]'] * efs_kpi.iloc[:,0] / 1000).sum() + (process['Elz wind [kWh/h]'] * wind_efs / 1000).sum() + (process['Elz PV [kWh/h]'] * pv_efs / 1000).sum()) / ch4_p2g #[kgCO2/MWhCH4]
    mef_ems = ((process['Elz grid [kWh/h]'] * efs_kpi.iloc[:,1] / 1000).sum() + (process['Elz wind [kWh/h]'] * wind_efs / 1000).sum() + (process['Elz PV [kWh/h]'] * pv_efs / 1000).sum()) / ch4_p2g #[kgCO2/MWhCH4]

    #Water use


    #Printing
    table = [['LCOE', 'NPV', 'MSP', 'Gas eff.', 'Total eff.', 'Spec. ems. (AEF)', 'Spec. ems. (MEF)', 'Loss %'], \
             [lcoe, npv, msp, n_gas, n_tot, aef_ems, mef_ems, loss_frac]]
    if run_type == "single":
        print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    """ Plotting """
    
    if run_type == "single":
        #Costs
        plt.pie([H2_CAPEX, H2_OPEX*20, H2_STACK, el_cost*20, METH_CAPEX, METH_OPEX*20, hex_CAPEX, hex_OPEX*20])
        plt.legend(['H2 capex', 'H2 opex', 'H2 stack', 'Electricity', 'Meth capex', 'Meth opex', 'Others capex', 'Others opex'])
        plt.show()
        
        #BY-PRODUCTS
        #Income
        if by_use == "WWTP+O3":
            # plt.pie([avoided_heat_income, o2_sales, o3_sales])
            plt.legend(['Internal heat', 'O2 sales', 'O3 sales'])
        else:
            # plt.pie([avoided_heat_income, o2_sales])
            plt.legend(['Internal heat', 'O2 sales'])
        plt.show()
        #Utilization
        #of produced O2
        if by_use == "WWTP+O3":
            # plt.pie([o2_wwtp.sum(), o3_wwtp.sum(), o2_waste.sum()])
            # plt.legend(['Aeration', 'Ozonation', 'Unused'])
            o2_utilization = (o2_wwtp + o3_wwtp).sum() / (o2_wwtp + o3_wwtp + o2_waste).sum()
        else:
            # plt.pie([o2_wwtp.sum(), o2_waste.sum()])
            # plt.legend(['Aeration', 'Unused'])
            o2_utilization = (o2_wwtp).sum() / (o2_wwtp + o2_waste).sum()
        #of produced heat
        # plt.pie([avoided_lg_heat.sum(), lg_heat_rem.sum()])
        # plt.legend(['Internal use', 'Unused'])
        heat_utilization = avoided_lg_heat.sum() / (avoided_lg_heat + lg_heat_rem).sum()
        #of total O2 demand
        aeration_demand_fulf = o2_wwtp.sum() / process['O2 WWTP [mol/h]'].sum()
        if by_use == "WWTP+O3":
            ozonation_demand_fulf = o3_wwtp.sum() / (process['O3 WWTP [mol/h]'] * 12.45).sum()
            total_o2_demand_fulf = (o2_wwtp + o3_wwtp).sum() / (process['O2 WWTP [mol/h]'] + (process['O3 WWTP [mol/h]']*12.45)).sum()
        #of total heat demand
        # plt.pie([avoided_lg_heat.sum(),(lg_heat_demand-avoided_lg_heat).sum])
        # plt.legend(['Fulfilled demand', 'Total demand'])
        heat_demand_fulf = lg_heat_demand.sum() / (avoided_lg_heat + lg_heat_rem).sum()
        #Oxygen example
        # fig, ax = plt.subplots()
        # ax.plot(process.iloc[0:23,13], label='O2 production')
        # ax.plot(process.iloc[0:23,46], label='O2 demand')
        # ax.set_ylabel('Oxygen [mol/h]')
        # plt.legend()
        

    
    return table, sum(test), process #lcoe, aef
