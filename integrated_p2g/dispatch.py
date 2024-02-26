# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 09:47:57 2023

@author: Linus Engstam
"""

import pandas as pd
import numpy as np
import pulp as plp


def grid_res_econ_demand(
    demand,
    grid,
    wind,
    pv,
    elz_max,  # kW
    elz_min,  # kW
    params,
    h2st_max,  # kg?
    h2st_prev,  # storage from previous day
    wind_cost: float = 0,
    pv_cost: float = 0,
    h2_hv: float = 33.3,  # LHV or HHV
) -> pd.DataFrame:
    """Returns electrolyzer dispatch based on both renewable and grid electricity
    optimized to minimize the cost of electricity for a fixed H2 demand"""
    # What should the cost of renewable electricity be? Free? #Leandro's concept about increasing the cost due to curtailment could be relevant?
    # No electrolyzer startup restrictions currently
    # Part load efficiency?

    # Define input data
    #load_range = params.iloc[:,0]
    #elz_n = params.iloc[:,1]
    demand = demand.flatten().tolist()

    # Determine whether we can fulfill the demand with only the electrolyzer or not
    elz_max_list = np.zeros(24) + (elz_max*0.6/h2_hv)
    h2_demand = np.array([[min(a, b) for a, b in zip(demand, elz_max_list)]])
    rest_demand = list(demand - h2_demand[0])
    h2_demand = list(h2_demand[0])

    wind = list(wind)
    pv = list(pv)
    grid = list(grid)
    h2st_prev = h2st_prev * 2.02 / 1000  # mol to kg

    # Define variables
    elz = []
    grid_el = []
    wind_el = []
    pv_el = []
    h2st = []  # unit for this? [kg] now
    #n = []
    h2_miss = []

    for i in range(len(demand)):
        # Electrolyzer operation
        elz.append(plp.LpVariable("elz_{}".format(i), 0, elz_max))
        # Purchased grid electricity
        grid_el.append(plp.LpVariable("grid_el_{}".format(i), 0, elz_max))
        # Wind-based operation
        wind_el.append(plp.LpVariable("wind_el_{}".format(i), 0, elz_max))
        # PV-based operation
        pv_el.append(plp.LpVariable("pv_el_{}".format(i), 0, elz_max))
        # Hydrogen storage
        h2st.append(plp.LpVariable("h2st_{}".format(i), 0, h2st_max))
        # Efficiency
        # n.append(plp.LpVariable("n_{}".format(i),0,1))
        # Unmet demand
        h2_miss.append(plp.LpVariable("h2_miss_{}".format(i), 0, None))

    # Problem definition
    prob = plp.LpProblem("Electricity_cost_minimization", plp.LpMinimize)

    # Constraints
    for i in range(len(demand)):
        prob += wind_el[i] <= wind[i]
        prob += pv_el[i] <= pv[i]
        prob += elz[i] == wind_el[i] + pv_el[i] + grid_el[i]

        if i == 0:  # using previous day storage value for first hour
            # prob += elz[i] >= (h2_demand[i] - h2st_prev) * h2_hv / 0.6 #EFFICIENCY #Converting kg to kWh
           # Excess H2 to storage (should be an additional efficiency loss due to compressor)
            # prob += h2st[i] == h2st_prev + (elz[i] * 0.6 / h2_hv) - h2_demand[i] #How to include compressor losses? Another minus factor converted to kgH2?

            # Unmet demand, what is not produced by the electrolyzer and not supplied through the storage
            prob += h2_miss[i] == demand[i] - \
                (elz[i]*0.6/h2_hv) - (h2st_prev - h2st[i])

            # Cannot "charge" H2 storage if we cannot fulfill demand
            if rest_demand[i] > 0:
                prob += h2st[i] <= h2st_prev

        else:
            # prob += elz[i] >= (h2_demand[i] - h2st[i-1]) * h2_hv / 0.6 #EFFICIENCY #Converting kg to kWh
            # Excess H2 to storage (should be an additional efficiency loss due to compressor)
            # prob += h2st[i] == h2st[i-1] + (elz[i] * 0.6 / h2_hv) - h2_demand[i] #How to include compressor losses? Another minus factor converted to kgH2?

            # Old attempt:
            # Making sure that storage capacity can be used to fulfill the demand
            # if rest_demand[i] <= h2st[i-1]: #if electrolyzer and storage can fulfill overall demand together, we use that
            #     prob += elz[i] >= (demand[i] - h2st[i-1]) * h2_hv / 0.6 #EFFICIENCY #Converting kg to kWh
            #     #Excess H2 to storage (should be an additional efficiency loss due to compressor)
            #     prob += h2st[i] == h2st[i-1] + (elz[i] * 0.6 / h2_hv) - demand[i] #How to include compressor losses? Another minus factor converted to kgH2?
            # else: #if electrolyzer and storage cannot fulfill the demand, we use as much of the storage as possible (since there is no economic difference as to when we use storage since it does not impact electrolyzer dispatch which is already at max)
            # #     prob += elz[i] >= 0 #EFFICIENCY #Converting kg to kWh
            # #     #Excess H2 to storage (should be an additional efficiency loss due to compressor)
            # #     prob += h2st[i] == 0
            #     prob += elz[i] == elz_max #EFFICIENCY #Converting kg to kWh
            #     #Excess H2 to storage (should be an additional efficiency loss due to compressor). In this case, the storage is emptied
            #     prob += h2st[i] == 0 #How to include compressor losses? Another minus factor converted to kgH2?

            # Unmet demand, what is not produced by the electrolyzer and not supplied through the storage
            prob += h2_miss[i] == demand[i] - \
                (elz[i]*0.6/h2_hv) - (h2st[i-1] - h2st[i])

            # Cannot "charge" H2 storage if we cannot fulfill demand
            if rest_demand[i] > 0:
                prob += h2st[i] <= h2st[i-1]

    # objective (minimize the electricity cost)
    prob += plp.lpSum([grid_el[i] * grid[i] for i in range(len(demand))]) + \
        plp.lpSum([wind_el[i] * wind_cost for i in range(len(demand))]) + \
        plp.lpSum([pv_el[i] * pv_cost for i in range(len(demand))]) + \
        plp.lpSum([h2_miss[i] * 10000000 for i in range(len(demand))])

    # check solution
    solver = plp.GUROBI_CMD()  # plp.getSolver('GUROBI_CMD')
    #solver = plp.PULP_CBC_CMD()
    status = prob.solve(solver)

    e_op = []
    grid_op = []
    wind_op = []
    pv_op = []
    storage = []
    #eff = []
    h2_missed = []

    # saving variable solutions
    for i in range(len(demand)):
        e_op.append(elz[i].varValue)
        grid_op.append(grid_el[i].varValue)
        wind_op.append(wind_el[i].varValue)
        pv_op.append(pv_el[i].varValue)
        storage.append(h2st[i].varValue)
        # eff.append(n[i].varValue)
        #h2_prod.append(e_op[i] * eff[i] / h2_hv)
        h2_missed.append(h2_miss[i].varValue)

    op = pd.DataFrame(
        {'Electrolyzer operation': e_op,
         'Grid purchases': grid_op,
         'Wind power': wind_op,
         'PV power': pv_op,
         'Electricity price': grid,
         'Storage': storage,
         'Demand': demand,
         # 'Efficiency': eff,
         'Unmet demand': h2_missed,
         'Status': status
         })
    return op


def grid_res_ems_demand(
    demand,
    grid,
    wind,
    pv,
    elz_max,
    elz_min,
    params,
    h2st_max,  # kg?
    h2st_prev,  # storage from previous day
    wind_ef: float = 15.6,
    pv_ef: float = 30,
    h2_hv: float = 33.3,  # LHV or HHV
) -> pd.DataFrame:
    """Returns electrolyzer dispatch based on both renewable and grid electricity
    optimized to minimize the cost of electricity for a fixed H2 demand"""
    # What should the cost of renewable electricity be? Free? #Leandro's concept about increasing the cost due to curtailment could be relevant?
    # No electrolyzer startup restrictions currently
    # Part load efficiency?

    # Define input data
    load_range = params.iloc[:, 0]
    elz_n = params.iloc[:, 1]
    demand = demand.flatten().tolist()

    # Determine whether we can fulfill the demand with only the electrolyzer or not
    elz_max_list = np.zeros(24) + (elz_max*0.6/h2_hv)
    h2_demand = np.array([[min(a, b) for a, b in zip(demand, elz_max_list)]])
    rest_demand = list(demand - h2_demand[0])
    h2_demand = list(h2_demand[0])

    wind = list(wind)
    pv = list(pv)
    grid = list(grid)
    h2st_prev = h2st_prev * 2.02 / 1000  # mol to kg

    # Define variables
    elz = []
    grid_el = []
    wind_el = []
    pv_el = []
    h2st = []  # unit for this? [kg] now
    n = []
    h2_miss = []

    for i in range(len(demand)):
        # Electrolyzer operation
        elz.append(plp.LpVariable("elz_{}".format(i), 0, elz_max))
        # Purchased grid electricity
        grid_el.append(plp.LpVariable("grid_el_{}".format(i), 0, elz_max))
        # Wind-based operation
        wind_el.append(plp.LpVariable("wind_el_{}".format(i), 0, elz_max))
        # PV-based operation
        pv_el.append(plp.LpVariable("pv_el_{}".format(i), 0, elz_max))
        # Hydrogen storage
        h2st.append(plp.LpVariable("h2st_{}".format(i), 0, h2st_max))
        # Efficiency
        # Hydrogen storage
        n.append(plp.LpVariable("n_{}".format(i), 0, 1))
        # Unmet demand
        h2_miss.append(plp.LpVariable("h2_miss_{}".format(i), 0, None))

    # Problem definition
    prob = plp.LpProblem("Electricity_cost_minimization", plp.LpMinimize)

    # Constraints
    for i in range(len(demand)):
        prob += wind_el[i] <= wind[i]
        prob += pv_el[i] <= pv[i]
        prob += elz[i] == wind_el[i] + pv_el[i] + grid_el[i]

        if i == 0:  # using previous day storage value for first hour
            # #Making sure that storage capacity can be used to fulfill the demand
            # prob += elz[i] >= (demand[i] - h2st_prev) * h2_hv / 0.6 #EFFICIENCY, Converting kg to kWh
            # #Excess H2 to storage (should be an additional efficiency loss due to compressor)
            # prob += h2st[i] == h2st_prev + (elz[i] * 0.6 / h2_hv) - demand[i] #How to include compressor losses?

            # Unmet demand, what is not produced by the electrolyzer and not supplied through the storage
            prob += h2_miss[i] == demand[i] - \
                (elz[i]*0.6/h2_hv) - (h2st_prev - h2st[i])

            # Cannot "charge" H2 storage if we cannot fulfill demand
            if rest_demand[i] > 0:
                prob += h2st[i] <= h2st_prev

        else:
            # #Making sure that storage capacity can be used to fulfill the demand
            # prob += elz[i] >= (demand[i] - h2st[i-1]) * h2_hv / 0.6 #EFFICIENCY #Converting kg to kWh
            # #Excess H2 to storage (should be an additional efficiency loss due to compressor)
            # prob += h2st[i] == h2st[i-1] + (elz[i] * 0.6 / h2_hv) - demand[i] #How to include compressor losses? Another minus factor converted to kgH2?

            # Unmet demand, what is not produced by the electrolyzer and not supplied through the storage
            # Unmet demand, what is not produced by the electrolyzer and not supplied through the storage
            prob += h2_miss[i] == demand[i] - \
                (elz[i]*0.6/h2_hv) - (h2st[i-1] - h2st[i])

            # Cannot "charge" H2 storage if we cannot fulfill demand
            if rest_demand[i] > 0:
                prob += h2st[i] <= h2st[i-1]

    # objective (minimize the electricity cost)
    prob += plp.lpSum([grid_el[i] * grid[i] for i in range(len(demand))]) + \
        plp.lpSum([wind_el[i] * wind_ef for i in range(len(demand))]) + \
        plp.lpSum([pv_el[i] * pv_ef for i in range(len(demand))]) + \
        plp.lpSum([h2_miss[i] * 10000000 for i in range(len(demand))])

    # check solution
    solver = plp.GUROBI_CMD()  # plp.getSolver('GUROBI_CMD')
    status = prob.solve(solver)

    e_op = []
    grid_op = []
    wind_op = []
    pv_op = []
    storage = []
    eff = []
    h2_prod = []
    h2_missed = []

    # saving variable solutions
    for i in range(len(demand)):
        e_op.append(elz[i].varValue)
        grid_op.append(grid_el[i].varValue)
        wind_op.append(wind_el[i].varValue)
        pv_op.append(pv_el[i].varValue)
        storage.append(h2st[i].varValue)
        eff.append(n[i].varValue)
        #h2_prod.append(e_op[i] * eff[i] / h2_hv)
        h2_missed.append(h2_miss[i].varValue)

    op = pd.DataFrame(
        {'Electrolyzer operation': e_op,
         'Grid purchases': grid_op,
         'Wind power': wind_op,
         'PV power': pv_op,
         'Electricity price': grid,
         'Storage': storage,
         'Demand': demand,
         # 'Efficiency': eff,
         'Unmet demand': h2_missed,
         'Status': status
         })
    return op


# Should solve for alpha=0 and alpha=1 to get pareto normalization factors (Fleschutz for the name, Grodzevich for method) and use them to normalize the objective function
# parameters and while solving for a specific alpha between 0 and 1
def grid_res_multi_demand(
    demand,
    grid,
    wind,
    pv,
    efs,  # either AEFs or MEFs
    elz_max,
    elz_min,
    params,
    h2st_max,  # kg?
    h2st_prev,  # storage from previous day
    wind_cost: float = 0,
    pv_cost: float = 0,
    h2_hv: float = 33.3,  # LHV or HHV
    alpha: float = 0.5,  # 1 = emissions, 0 = cost
    wind_ef: float = 15.6,
    pv_ef: float = 30,
    cost_utp: float = 0,
    ems_utp: float = 0,
    cost_norm: float = 1,  # pareto normalization factor for cost objective
    ems_norm: float = 1,  # pareto normalization factor for emissions objective
) -> pd.DataFrame:
    """Returns electrolyzer dispatch based on both renewable and grid electricity
    optimized to minimize the cost of electricity for a fixed H2 demand"""
    # What should the cost of renewable electricity be? Free? #Leandro's concept about increasing the cost due to curtailment could be relevant?
    # No electrolyzer startup restrictions currently
    # Part load efficiency?

    # Define input data
    load_range = params.iloc[:, 0]
    elz_n = params.iloc[:, 1]
    demand = demand.flatten().tolist()

    # Determine whether we can fulfill the demand with only the electrolyzer or not
    elz_max_list = np.zeros(24) + (elz_max*0.6/h2_hv)
    h2_demand = np.array([[min(a, b) for a, b in zip(demand, elz_max_list)]])
    rest_demand = list(demand - h2_demand[0])
    h2_demand = list(h2_demand[0])

    wind = list(wind)
    pv = list(pv)
    grid = list(grid)
    efs = list(efs)
    h2st_prev = h2st_prev * 2.02 / 1000  # mol to kg

    # Define variables
    elz = []
    grid_el = []
    wind_el = []
    pv_el = []
    h2st = []  # unit for this? [kg] now
    n = []
    cost = []
    emissions = []
    h2_miss = []

    for i in range(len(demand)):
        # Electrolyzer operation
        elz.append(plp.LpVariable("elz_{}".format(i), 0, elz_max))
        # Purchased grid electricity
        grid_el.append(plp.LpVariable("grid_el_{}".format(i), 0, elz_max))
        # Wind-based operation
        wind_el.append(plp.LpVariable("wind_el_{}".format(i), 0, elz_max))
        # PV-based operation
        pv_el.append(plp.LpVariable("pv_el_{}".format(i), 0, elz_max))
        # Hydrogen storage
        h2st.append(plp.LpVariable("h2st_{}".format(i), 0, h2st_max))
        # Efficiency
        # Hydrogen storage
        n.append(plp.LpVariable("n_{}".format(i), 0, 1))
        #Cost and emissions
        cost.append(plp.LpVariable("cost_{}".format(i), 0, None))
        emissions.append(plp.LpVariable("emissions_{}".format(i), 0, None))
        # Unmet demand
        h2_miss.append(plp.LpVariable("h2_miss_{}".format(i), 0, None))

    # Problem definition
    prob = plp.LpProblem("Electricity_cost_minimization", plp.LpMinimize)

    # Constraints
    for i in range(len(demand)):
        prob += wind_el[i] <= wind[i]
        prob += pv_el[i] <= pv[i]
        prob += elz[i] == wind_el[i] + pv_el[i] + grid_el[i]

        if i == 0:  # using previous day storage value for first hour
            # #Making sure that storage capacity can be used to fulfill the demand
            # prob += elz[i] >= (demand[i] - h2st_prev) * h2_hv / 0.6 #EFFICIENCY, Converting kg to kWh
            # #Excess H2 to storage (should be an additional efficiency loss due to compressor)
            # prob += h2st[i] == h2st_prev + (elz[i] * 0.6 / h2_hv) - demand[i] #How to include compressor losses?

            # Unmet demand, what is not produced by the electrolyzer and not supplied through the storage
            prob += h2_miss[i] == demand[i] - \
                (elz[i]*0.6/h2_hv) - (h2st_prev - h2st[i])

            # Cannot "charge" H2 storage if we cannot fulfill demand
            if rest_demand[i] > 0:
                prob += h2st[i] <= h2st_prev

        else:
            # #Making sure that storage capacity can be used to fulfill the demand
            # prob += elz[i] >= (demand[i] - h2st[i-1]) * h2_hv / 0.6 #EFFICIENCY #Converting kg to kWh
            # #Excess H2 to storage (should be an additional efficiency loss due to compressor)
            # prob += h2st[i] == h2st[i-1] + (elz[i] * 0.6 / h2_hv) - demand[i] #How to include compressor losses? Another minus factor converted to kgH2?

            # Unmet demand, what is not produced by the electrolyzer and not supplied through the storage
            # Unmet demand, what is not produced by the electrolyzer and not supplied through the storage
            prob += h2_miss[i] == demand[i] - \
                (elz[i]*0.6/h2_hv) - (h2st[i-1] - h2st[i])

            # Cannot "charge" H2 storage if we cannot fulfill demand
            if rest_demand[i] > 0:
                prob += h2st[i] <= h2st[i-1]

        #Cost and emissions
        prob += cost[i] == (grid_el[i] * grid[i]) + \
            (wind_el[i] * wind_cost) + (pv_el[i] * pv_cost)
        prob += emissions[i] == (grid_el[i] * efs[i]) + \
            (wind_el[i] * wind_ef) + (pv_el[i] * pv_ef)

    # objective (minimize the electricity cost and emissions)
    # prob += ((plp.lpSum([grid_el[i] * grid[i] for i in range(len(demand))]) + \
    #     plp.lpSum([wind_el[i] * wind_cost for i in range(len(demand))]) + \
    #         plp.lpSum([pv_el[i] * pv_cost for i in range(len(demand))])) * (1-alpha) / cost_norm) + \
    #     ((plp.lpSum([grid_el[i] * efs[i] for i in range(len(demand))]) + \
    #         plp.lpSum([wind_el[i] * wind_ef for i in range(len(demand))]) + \
    #             plp.lpSum([pv_el[i] * pv_ef for i in range(len(demand))])) * (alpha) / ems_norm)

    c_factor = (1-alpha) / cost_norm
    e_factor = alpha / ems_norm
    h_factor = (c_factor+e_factor) / 2

    prob += ((plp.lpSum([cost[i] for i in range(len(demand))]) - cost_utp) * c_factor) + \
        ((plp.lpSum([emissions[i]for i in range(len(demand))]) - ems_utp) * e_factor) + \
        (plp.lpSum([h2_miss[i] for i in range(len(demand))])
         * 10000000 / h_factor)  # Normalizing this as well
    # Something about this makes it perform weird, in my example run it seems to operate based on costs only?
    # Thought: when it can't fulfill the demand, then something is off?

    # check solution
    solver = plp.GUROBI_CMD()  # plp.getSolver('GUROBI_CMD')
    status = prob.solve(solver)

    e_op = []
    grid_op = []
    wind_op = []
    pv_op = []
    storage = []
    eff = []
    h2_prod = []
    h2_missed = []

    # saving variable solutions
    for i in range(len(demand)):
        e_op.append(elz[i].varValue)
        grid_op.append(grid_el[i].varValue)
        wind_op.append(wind_el[i].varValue)
        pv_op.append(pv_el[i].varValue)
        storage.append(h2st[i].varValue)
        # eff.append(n[i].varValue)
        #h2_prod.append(e_op[i] * eff[i] / h2_hv)
        h2_missed.append(h2_miss[i].varValue)

    # Find nadir and utopia points. Alpha=0 --> utp for cost and nad for ems and vice versa
    if alpha == 0:
        utp = (np.array(e_op) * np.array(grid)).sum() + (np.array(wind_op) *
                                                         np.array(wind_cost)).sum() + (np.array(pv_op) * np.array(pv_cost)).sum()
        nad = (np.array(e_op) * np.array(efs)).sum() + (np.array(wind_op) *
                                                        np.array(wind_ef)).sum() + (np.array(pv_op) * np.array(pv_ef)).sum()
    elif alpha == 1:
        nad = (np.array(e_op) * np.array(grid)).sum() + (np.array(wind_op) *
                                                         np.array(wind_cost)).sum() + (np.array(pv_op) * np.array(pv_cost)).sum()
        utp = (np.array(e_op) * np.array(efs)).sum() + (np.array(wind_op) *
                                                        np.array(wind_ef)).sum() + (np.array(pv_op) * np.array(pv_ef)).sum()
    else:
        nad = 0
        utp = 0
    op = pd.DataFrame(
        {'Electrolyzer operation': e_op,
         'Grid purchases': grid_op,
         'Wind power': wind_op,
         'PV power': pv_op,
         'Electricity price': grid,
         'Storage': storage,
         'Demand': demand,
         # 'Efficiency': eff,
         'Unmet demand': h2_missed,
         'Status': status
         })
    return op, utp, nad


def grid_res_econ_flex(
    demand,
    grid,
    wind,
    pv,
    elz_max,
    elz_min,
    params,
    h2st_max,  # kg?
    h2st_prev,  # storage from previous day
    wind_cost: float = 0,
    pv_cost: float = 0,
    h2_hv: float = 33.3,  # LHV or HHV
    gas_price: float = 50,  # [€/MWh] Value of H2 based on CH4 sales price?
) -> pd.DataFrame:
    """Returns electrolyzer dispatch based on both renewable and grid electricity
    optimized to minimize the cost of electricity for a flexible H2 demand"""
    # What should the cost of renewable electricity be? Free? #Leandro's concept about increasing the cost due to curtailment could be relevant?
    # No electrolyzer startup restrictions currently
    # Part load efficiency?

    # Flexibility: how to enable this?
    # Set a maximum purchasing price for electricity?

    # Define input data
    load_range = params.iloc[:, 0]
    elz_n = params.iloc[:, 1]
    demand = demand.flatten().tolist()
    wind = list(wind)
    pv = list(pv)
    grid = list(grid)
    h2st_prev = h2st_prev * 2.02 / 1000  # mol to kg

    # Define variables
    elz = []
    grid_el = []
    wind_el = []
    pv_el = []
    h2st = []  # unit for this? [kg] now
    h2_use = []
    for i in range(len(demand)):
        # Electrolyzer operation
        elz.append(plp.LpVariable("elz_{}".format(i), 0, elz_max))
        # Purchased grid electricity
        grid_el.append(plp.LpVariable("grid_el_{}".format(i), 0, elz_max))
        # Wind-based operation
        wind_el.append(plp.LpVariable("wind_el_{}".format(i), 0, elz_max))
        # PV-based operation
        pv_el.append(plp.LpVariable("pv_el_{}".format(i), 0, elz_max))
        # Hydrogen storage
        h2st.append(plp.LpVariable("h2st_{}".format(i), 0, h2st_max))
        # Hydrogen use
        h2_use.append(plp.LpVariable("h2_use_{}".format(i), 0, max(demand)))

    # Problem definition
    prob = plp.LpProblem("Electricity_cost_minimization", plp.LpMinimize)

    # Constraints
    for i in range(len(demand)):
        prob += wind_el[i] <= wind[i]
        prob += pv_el[i] <= pv[i]
        prob += elz[i] == wind_el[i] + pv_el[i] + grid_el[i]

        if i == 0:  # using previous day storage value for first hour
            # Storage charging/discharging
            prob += h2st[i] == h2st_prev + (elz[i] * 0.6 / h2_hv) - h2_use[i]
            # Cannot use more hydrogen than demand
            prob += h2_use[i] <= demand[i]
            # Cannot produce more than demand + available storage capacity
            prob += elz[i] <= (demand[i] + h2st_max - h2st_prev) * h2_hv / 0.6
        else:
            # Storage charging/discharging
            prob += h2st[i] == h2st[i-1] + (elz[i] * 0.6 / h2_hv) - h2_use[i]
            # Cannot use more hydrogen than demand
            prob += h2_use[i] <= demand[i]
            # Cannot produce more than demand + available storage capacity
            prob += elz[i] <= (demand[i] + h2st_max - h2st[i-1]) * h2_hv / 0.6

    #h2_prod = sum(elz) * 0.6 / h2_hv

    # objective (minimize the electricity cost)
    prob += plp.lpSum([grid_el[i] * grid[i] for i in range(len(demand))]) + \
        plp.lpSum([wind_el[i] * wind_cost for i in range(len(demand))]) + \
        plp.lpSum([pv_el[i] * pv_cost for i in range(len(demand))]) - \
        plp.lpSum([elz[i] * (0.6 * gas_price) for i in range(len(demand))])

    # check solution
    solver = plp.GUROBI_CMD()
    status = prob.solve(solver)

    e_op = []
    grid_op = []
    wind_op = []
    pv_op = []
    storage = []
    h2_usage = []
    # saving variable solutions
    for i in range(len(demand)):
        e_op.append(elz[i].varValue)
        grid_op.append(grid_el[i].varValue)
        wind_op.append(wind_el[i].varValue)
        pv_op.append(pv_el[i].varValue)
        storage.append(h2st[i].varValue)
        h2_usage.append(h2_use[i].varValue)

    op = pd.DataFrame(
        {'Electrolyzer operation': e_op,
         'Grid purchases': grid_op,
         'Wind power': wind_op,
         'PV power': pv_op,
         'Electricity price': grid,
         'Storage': storage,
         'Demand': demand,
         'H2 usage': h2_usage,
         'Status': status
         })
    return op


def grid_res_econ_demand_pl(
    demand,
    grid,
    wind,
    pv,
    elz_max,  # kW
    elz_min,  # kW
    params,
    h2st_max,  # kg?
    h2st_prev,  # storage from previous day
    wind_cost: float = 0,
    pv_cost: float = 0,
    h2_hv: float = 33.3,  # LHV or HHV
) -> pd.DataFrame:
    """Returns electrolyzer dispatch based on both renewable and grid electricity
    optimized to minimize the cost of electricity for a fixed H2 demand
    Also including a linearized part-load efficiency and minimum load
    Using linearization parameters as proposed in Wirtz et al. 2021"""
    # What should the cost of renewable electricity be? Free? #Leandro's concept about increasing the cost due to curtailment could be relevant?
    # No electrolyzer startup restrictions currently
    # Potential issue: Can be profitable to operate on minimum load due to higher efficiency instead of shutting down...
    # Should use system efficieny and not stack for this. Could assume a constant auxiliary energy consumption?

    # Define input data
    #load_range = params.iloc[:,0]
    #elz_n = params.iloc[:,1]
    demand = demand.flatten().tolist()

    # Determine whether we can fulfill the demand with only the electrolyzer or not
    elz_max_list = np.zeros(24) + (elz_max*0.6/h2_hv)
    h2_demand = np.array([[min(a, b) for a, b in zip(demand, elz_max_list)]])
    rest_demand = list(demand - h2_demand[0])
    h2_demand = list(h2_demand[0])

    wind = list(wind)
    pv = list(pv)
    grid = list(grid)
    h2st_prev = h2st_prev * 2.02 / 1000  # mol to kg

    # Define variables
    elz = []
    grid_el = []
    wind_el = []
    pv_el = []
    h2st = []  # unit for this? [kg] now
    h2_miss = []
    h2_prod = []
    elz_mode = []

    for i in range(len(demand)):
        # Electrolyzer operation
        elz.append(plp.LpVariable("elz_{}".format(i), 0, elz_max))
        elz_mode.append(plp.LpVariable(
            "elz_mode_{}".format(i), 0, 1, plp.LpInteger))
        # Purchased grid electricity
        grid_el.append(plp.LpVariable("grid_el_{}".format(i), 0, elz_max))
        # Wind-based operation
        wind_el.append(plp.LpVariable("wind_el_{}".format(i), 0, elz_max))
        # PV-based operation
        pv_el.append(plp.LpVariable("pv_el_{}".format(i), 0, elz_max))
        # Hydrogen storage
        h2st.append(plp.LpVariable("h2st_{}".format(i), 0, h2st_max))
        # Unmet demand
        h2_miss.append(plp.LpVariable("h2_miss_{}".format(i), 0, None))
        # Hydrogen production
        h2_prod.append(plp.LpVariable("h2_prod_{}".format(i), 0, None))

    # Problem definition
    prob = plp.LpProblem("Electricity_cost_minimization", plp.LpMinimize)

    # Constraints
    for i in range(len(demand)):
        prob += wind_el[i] <= wind[i]
        prob += pv_el[i] <= pv[i]
        prob += elz[i] == wind_el[i] + pv_el[i] + grid_el[i]
        # defining minimum load
        prob += elz[i] >= elz_min * elz_mode[i]
        # defining maximum load
        prob += elz[i] <= elz_max * elz_mode[i]
        # defining start-up
        # if i > 0:
        #     prob += elz_start[i] == elz_mode[i] - elz_mode[i-1] + elz_stop[i]
        # else:
        #     prob += elz_start[i] == elz_mode[i]

        prob += h2_prod[i] == (14.89002 * elz_mode[i]) + \
            (elz[i] * 167.5127 / elz_max)

        if i == 0:  # using previous day storage value for first hour
            # prob += elz[i] >= (h2_demand[i] - h2st_prev) * h2_hv / 0.6 #EFFICIENCY #Converting kg to kWh
           # Excess H2 to storage (should be an additional efficiency loss due to compressor)
            # prob += h2st[i] == h2st_prev + (elz[i] * 0.6 / h2_hv) - h2_demand[i] #How to include compressor losses? Another minus factor converted to kgH2?

            # Unmet demand, what is not produced by the electrolyzer and not supplied through the storage
            prob += h2_miss[i] == demand[i] - \
                h2_prod[i] - (h2st_prev - h2st[i])
            # prob += h2_miss[i] == demand[i] - (elz[i] * 0.6 / h2_hv) - (h2st_prev - h2st[i])

            # Cannot "charge" H2 storage if we cannot fulfill demand
            if rest_demand[i] > 0:
                prob += h2st[i] <= h2st_prev

        else:
            # prob += elz[i] >= (h2_demand[i] - h2st[i-1]) * h2_hv / 0.6 #EFFICIENCY #Converting kg to kWh
            # Excess H2 to storage (should be an additional efficiency loss due to compressor)
            # prob += h2st[i] == h2st[i-1] + (elz[i] * 0.6 / h2_hv) - h2_demand[i] #How to include compressor losses? Another minus factor converted to kgH2?

            # Old attempt:
            # Making sure that storage capacity can be used to fulfill the demand
            # if rest_demand[i] <= h2st[i-1]: #if electrolyzer and storage can fulfill overall demand together, we use that
            #     prob += elz[i] >= (demand[i] - h2st[i-1]) * h2_hv / 0.6 #EFFICIENCY #Converting kg to kWh
            #     #Excess H2 to storage (should be an additional efficiency loss due to compressor)
            #     prob += h2st[i] == h2st[i-1] + (elz[i] * 0.6 / h2_hv) - demand[i] #How to include compressor losses? Another minus factor converted to kgH2?
            # else: #if electrolyzer and storage cannot fulfill the demand, we use as much of the storage as possible (since there is no economic difference as to when we use storage since it does not impact electrolyzer dispatch which is already at max)
            # #     prob += elz[i] >= 0 #EFFICIENCY #Converting kg to kWh
            # #     #Excess H2 to storage (should be an additional efficiency loss due to compressor)
            # #     prob += h2st[i] == 0
            #     prob += elz[i] == elz_max #EFFICIENCY #Converting kg to kWh
            #     #Excess H2 to storage (should be an additional efficiency loss due to compressor). In this case, the storage is emptied
            #     prob += h2st[i] == 0 #How to include compressor losses? Another minus factor converted to kgH2?

            # Unmet demand, what is not produced by the electrolyzer and not supplied through the storage
            prob += h2_miss[i] == demand[i] - \
                h2_prod[i] - (h2st[i-1] - h2st[i])
            # prob += h2_miss[i] == demand[i] - (elz[i] * 0.6 / h2_hv) - (h2st_prev - h2st[i])

            # Cannot "charge" H2 storage if we cannot fulfill demand
            if rest_demand[i] > 0:
                prob += h2st[i] <= h2st[i-1]

    # objective (minimize the electricity cost)
    prob += plp.lpSum([grid_el[i] * grid[i] for i in range(len(demand))]) + \
        plp.lpSum([wind_el[i] * wind_cost for i in range(len(demand))]) + \
        plp.lpSum([pv_el[i] * pv_cost for i in range(len(demand))]) + \
        plp.lpSum([h2_miss[i] * 10000000 for i in range(len(demand))])

    # check solution
    solver = plp.GUROBI_CMD()  # plp.getSolver('GUROBI_CMD')
    #solver = plp.PULP_CBC_CMD()
    status = prob.solve(solver)

    e_op = []
    grid_op = []
    wind_op = []
    pv_op = []
    storage = []
    h2_missed = []
    h2_produced = []

    # saving variable solutions
    for i in range(len(demand)):
        e_op.append(elz[i].varValue)
        grid_op.append(grid_el[i].varValue)
        wind_op.append(wind_el[i].varValue)
        pv_op.append(pv_el[i].varValue)
        storage.append(h2st[i].varValue)
        h2_produced.append(h2_prod[i].varValue)
        h2_missed.append(h2_miss[i].varValue)

    op = pd.DataFrame(
        {'Electrolyzer operation': e_op,
         'Grid purchases': grid_op,
         'Wind power': wind_op,
         'PV power': pv_op,
         'Electricity price': grid,
         'Storage': storage,
         'Demand': demand,
         'H2 prod': h2_produced,
         # 'Efficiency': eff,
         'Unmet demand': h2_missed,
         'Status': status
         })
    return op


def econ_daily_pl(
    demand,
    hr_demand,
    grid,
    wind,
    pv,
    elz_max,  # kW
    elz_min,  # kW
    # params,
    h2st_max,  # kg?
    h2st_prev,  # storage from previous day
    meth_max,  # maximum H2 use [kg/h]
    meth_min,  # minimum H2 use [kg/h]
    elz_eff: float = 0.7,  # HHV
    wind_cost: float = 0,
    pv_cost: float = 0,
    h2_hv: float = 33.3,  # LHV or HHV
) -> pd.DataFrame:
    """Returns electrolyzer dispatch based on both renewable and grid electricity
    optimized to minimize the cost of electricity for a fixed H2 demand
    Also including a linearized part-load efficiency and minimum load
    Using linearization parameters as proposed in Wirtz et al. 2021"""
    # What should the cost of renewable electricity be? Free? #Leandro's concept about increasing the cost due to curtailment could be relevant?
    # No electrolyzer startup restrictions currently
    # Potential issue: Can be profitable to operate on minimum load due to higher efficiency instead of shutting down...
    # Should use system efficiency and not stack for this. Could assume a constant auxiliary energy consumption?
    # Add methanation start-up penalty to ensure a more stable operation?
    # Also storage loss?

    # Define input data
    #load_range = params.iloc[:,0]
    #elz_n = params.iloc[:,1]
    demand = demand.flatten().tolist()

    # Determine whether we can fulfill the demand with only the electrolyzer or not
    # elz_max_list = np.zeros(24) + (elz_max*0.6/h2_hv)
    # h2_demand = np.array([[min(a, b) for a, b in zip(demand, elz_max_list)]])
    # rest_demand = list(demand - h2_demand[0])
    # h2_demand = list(h2_demand[0])
    hr_demand = hr_demand.flatten().tolist()

    wind = list(wind)
    pv = list(pv)
    grid = list(grid)
    h2st_prev = h2st_prev * 2.02 / 1000  # mol to kg
    # h2_max = elz_max

    # linearization parameters
    k = 177.665  # 167.5127
    m = -4.653  # 14.89002

    # Define variables
    elz = []
    grid_el = []
    wind_el = []
    pv_el = []
    h2st = []  # unit for this? [kg] now
    # h2_miss = []
    h2_prod = []
    elz_mode = []
    h2_use = []
    meth_mode = []

    for i in range(len(grid)):
        # Electrolyzer operation
        elz.append(plp.LpVariable("elz_{}".format(i), 0, elz_max))
        elz_mode.append(plp.LpVariable(
            "elz_mode_{}".format(i), 0, 1, plp.LpInteger))
        # Purchased grid electricity
        grid_el.append(plp.LpVariable("grid_el_{}".format(i), 0, elz_max))
        # Wind-based operation
        wind_el.append(plp.LpVariable("wind_el_{}".format(i), 0, elz_max))
        # PV-based operation
        pv_el.append(plp.LpVariable("pv_el_{}".format(i), 0, elz_max))
        # Hydrogen storage
        h2st.append(plp.LpVariable("h2st_{}".format(i), 0, h2st_max))
        # Unmet demand
        # h2_miss.append(plp.LpVariable("h2_miss_{}".format(i),0,None))
        # Hydrogen production
        h2_prod.append(plp.LpVariable("h2_prod_{}".format(i), 0, None))
        # Hydrogen utilization
        h2_use.append(plp.LpVariable("h2_use_{}".format(i), 0, meth_max))
        meth_mode.append(plp.LpVariable(
            "meth_mode_{}".format(i), 0, 1, plp.LpInteger))

    # Problem definition
    prob = plp.LpProblem("Electricity_cost_minimization", plp.LpMinimize)

    # Constraints
    for i in range(len(grid)):
        prob += wind_el[i] <= wind[i]
        prob += pv_el[i] <= pv[i]
        prob += elz[i] == wind_el[i] + pv_el[i] + grid_el[i]
        # defining minimum electrolyzer load
        prob += elz[i] >= elz_min * elz_mode[i]
        # defining maximum electrolyzer load
        prob += elz[i] <= elz_max * elz_mode[i]
        # defining minimum methanation load
        prob += h2_use[i] >= meth_min * meth_mode[i]
        # defining maximum methanation load
        prob += h2_use[i] <= meth_max * meth_mode[i]
        # defining start-up
        # if i > 0:
        #     prob += elz_start[i] == elz_mode[i] - elz_mode[i-1] + elz_stop[i]
        # else:
        #     prob += elz_start[i] == elz_mode[i]
        # Can't exceed hourly CO2 availability
        prob += h2_use[i] <= hr_demand[i]

        # Linearized H2 production efficiency
        prob += h2_prod[i] == (m * elz_mode[i]) + (elz[i] * k / elz_max)

        if i == 0:  # using previous day storage value for first hour
            # Cannot charge H2 storage if we cannot fulfill demand
            # if rest_demand[i] > 0:
            #     prob += h2st[i] <= h2st_prev

            # Storage charging
            prob += h2st[i] == h2_prod[i] - h2_use[i] + h2st_prev

        else:
            #prob += h2_miss[i] == demand[i] - h2_prod[i] - (h2st[i-1] - h2st[i])
            # Cannot "charge" H2 storage if we cannot fulfill demand
            # if rest_demand[i] > 0:
            #     prob += h2st[i] <= h2st[i-1]

            # Storage charging
            prob += h2st[i] == h2_prod[i] - h2_use[i] + h2st[i-1]

    # Fulfill daily demand if technically possible
    # prob += h2_miss == demand - plp.lpSum([h2_prod[i] for i in range(len(grid))])
    # Do no exceed daily demand
    prob += plp.lpSum([h2_use[i] for i in range(len(grid))]) <= demand

    # objective (minimize the electricity cost)
    prob += plp.lpSum([grid_el[i] * grid[i] for i in range(len(grid))]) + \
        plp.lpSum([wind_el[i] * wind_cost for i in range(len(grid))]) + \
        plp.lpSum([pv_el[i] * pv_cost for i in range(len(grid))]) + \
        ((demand - plp.lpSum([h2_use[i] for i in range(len(grid))]))
         * 10000000)  # Fulfill daily demand if technically possible

    # check solution
    solver = plp.GUROBI_CMD()  # plp.getSolver('GUROBI_CMD')
    #solver = plp.PULP_CBC_CMD()
    status = prob.solve(solver)

    e_op = []
    grid_op = []
    wind_op = []
    pv_op = []
    storage = []
    h2_produced = []
    h2_used = []

    # saving variable solutions
    for i in range(len(grid)):
        e_op.append(elz[i].varValue)
        grid_op.append(grid_el[i].varValue)
        wind_op.append(wind_el[i].varValue)
        pv_op.append(pv_el[i].varValue)
        storage.append(h2st[i].varValue)
        h2_produced.append(h2_prod[i].varValue)
        h2_used.append(h2_use[i].varValue)

    h2_missed = np.zeros(24) + demand - sum(h2_used)
    demand_vector = np.zeros(24) + demand

    op = pd.DataFrame(
        {'Electrolyzer operation': e_op,
         'Grid purchases': grid_op,
         'Wind power': wind_op,
         'PV power': pv_op,
         'Electricity price': grid,
         'Storage': storage,
         'Demand': demand_vector,
         'H2 prod': h2_produced,
         'H2 used': h2_used,
         # 'Efficiency': eff,
         'Unmet demand': h2_missed,
         'Status': status
         })
    return op


def econ_daily_pl2(
    demand,
    hr_demand,
    grid,
    wind,
    pv,
    elz_max,  # kW
    elz_min,  # kW
    # params,
    h2st_max,  # kg?
    h2st_prev,  # storage from previous day
    meth_max,  # maximum H2 use [kg/h]
    meth_min,  # minimum H2 use [kg/h]
    startup_cost: float = 0.1,  # fraction of rated power
    standby_cost: float = 0.02,  # fraction of rated power
    # fraction of extra hydrogen lost during methanation startup (only first hour)
    meth_loss_factor: float = 0.1,
    prev_mode: int = 1,  # on/standby/off in last hour of previous day
    prev_mode_meth: int = 1,  # methanation on/off in last hour of previous day
    elz_eff: float = 0.7,  # HHV
    wind_cost: float = 0,
    pv_cost: float = 0,
    h2_hv: float = 33.3,  # LHV or HHV
) -> pd.DataFrame:
    """Returns electrolyzer dispatch based on both renewable and grid electricity
    optimized to minimize the cost of electricity for a fixed H2 demand
    Also including a linearized part-load efficiency and minimum load
    Using linearization parameters as proposed in Wirtz et al. 2021"""
    # What should the cost of renewable electricity be? Free? #Leandro's concept about increasing the cost due to curtailment could be relevant?
    # No electrolyzer startup restrictions currently
    # Potential issue: Can be profitable to operate on minimum load due to higher efficiency instead of shutting down...
    # Should use system efficiency and not stack for this. Could assume a constant auxiliary energy consumption?
    # Add methanation start-up penalty to ensure a more stable operation?
    # Also storage loss?
    # Now: lowest value in each interval has the optimal efficiency it seems
    # What about the start-up time? Can we ignore this since it is very short with PEM?
    # Should startup and standby costs be based on spot price or a constant price?

    demand = demand.flatten().tolist()
    hr_demand = hr_demand.flatten().tolist()

    wind = list(wind)
    pv = list(pv)
    grid = list(grid)
    h2st_prev = h2st_prev * 2.02 / 1000  # mol to kg
    # h2_max = elz_max

    # Define variables
    elz = []
    grid_el = []
    wind_el = []
    pv_el = []
    h2st = []  # unit for this? [kg] now
    # h2_miss = []
    h2_prod = []
    elz_mode = []
    elz_start = []
    elz_stop = []
    elz_off = []
    elz_standby = []
    h2_use = []
    meth_on = []
    meth_start = []
    meth_stop = []
    meth_loss = []
    # Part-load variables
    # Load
    e10 = []
    e20 = []
    e30 = []
    e40 = []
    e50 = []
    e60 = []
    e70 = []
    e80 = []
    e90 = []
    e100 = []
    # On/off
    lambda10 = []
    lambda20 = []
    lambda30 = []
    lambda40 = []
    lambda50 = []
    lambda60 = []
    lambda70 = []
    lambda80 = []
    lambda90 = []
    lambda100 = []

    # Linearization parameters (see "Linearized efficiency" Excel)
    # gradient
    phi10 = 223.3502
    phi20 = 213.1980
    phi30 = 203.0457
    phi40 = 192.8934
    phi50 = 182.7411
    phi60 = 172.5888
    phi70 = 162.4365
    phi80 = 152.2843
    phi90 = 142.1320
    phi100 = 131.9797
    # m-value
    gamma10 = -12.69036
    gamma20 = -11.67513
    gamma30 = -9.64467
    gamma40 = -6.598985
    gamma50 = -2.538071
    gamma60 = 2.5380711
    gamma70 = 8.6294416
    gamma80 = 15.736041
    gamma90 = 23.857868
    gamma100 = 32.994924

    for i in range(len(grid)):
        # Electrolyzer operation
        elz.append(plp.LpVariable("elz_{}".format(i), 0, elz_max))
        elz_mode.append(plp.LpVariable(
            "elz_mode_{}".format(i), 0, 1, plp.LpInteger))
        elz_start.append(plp.LpVariable(
            "elz_start_{}".format(i), 0, 1, plp.LpInteger))
        elz_stop.append(plp.LpVariable(
            "elz_stop_{}".format(i), 0, 1, plp.LpInteger))
        elz_standby.append(plp.LpVariable(
            "elz_standby_{}".format(i), 0, 1, plp.LpInteger))
        elz_off.append(plp.LpVariable(
            "elz_off_{}".format(i), 0, 1, plp.LpInteger))
        # Purchased grid electricity
        grid_el.append(plp.LpVariable("grid_el_{}".format(i), 0, elz_max))
        # Wind-based operation
        wind_el.append(plp.LpVariable("wind_el_{}".format(i), 0, elz_max))
        # PV-based operation
        pv_el.append(plp.LpVariable("pv_el_{}".format(i), 0, elz_max))
        # Hydrogen storage
        h2st.append(plp.LpVariable("h2st_{}".format(i), 0, h2st_max))
        # Unmet demand
        # h2_miss.append(plp.LpVariable("h2_miss_{}".format(i),0,None))
        # Hydrogen production
        h2_prod.append(plp.LpVariable("h2_prod_{}".format(i), 0, None))
        # Hydrogen utilization
        h2_use.append(plp.LpVariable("h2_use_{}".format(i), 0, meth_max))
        # Methanation
        meth_on.append(plp.LpVariable(
            "meth_on_{}".format(i), 0, 1, plp.LpInteger))
        meth_start.append(plp.LpVariable(
            "meth_start_{}".format(i), 0, 1, plp.LpInteger))
        meth_stop.append(plp.LpVariable(
            "meth_stop_{}".format(i), 0, 1, plp.LpInteger))
        meth_loss.append(plp.LpVariable("meth_loss_{}".format(i), 0, elz_max))

        # Part-load variables
        # Load
        e10.append(plp.LpVariable("e10_{}".format(i), 0, None))
        e20.append(plp.LpVariable("e20_{}".format(i), 0, None))
        e30.append(plp.LpVariable("e30_{}".format(i), 0, None))
        e40.append(plp.LpVariable("e40_{}".format(i), 0, None))
        e50.append(plp.LpVariable("e50_{}".format(i), 0, None))
        e60.append(plp.LpVariable("e60_{}".format(i), 0, None))
        e70.append(plp.LpVariable("e70_{}".format(i), 0, None))
        e80.append(plp.LpVariable("e80_{}".format(i), 0, None))
        e90.append(plp.LpVariable("e90_{}".format(i), 0, None))
        e100.append(plp.LpVariable("e100_{}".format(i), 0, None))
        # On/off
        lambda10.append(plp.LpVariable(
            "lambda10_{}".format(i), 0, 1, plp.LpInteger))
        lambda20.append(plp.LpVariable(
            "lambda20_{}".format(i), 0, 1, plp.LpInteger))
        lambda30.append(plp.LpVariable(
            "lambda30_{}".format(i), 0, 1, plp.LpInteger))
        lambda40.append(plp.LpVariable(
            "lambda40_{}".format(i), 0, 1, plp.LpInteger))
        lambda50.append(plp.LpVariable(
            "lambda50_{}".format(i), 0, 1, plp.LpInteger))
        lambda60.append(plp.LpVariable(
            "lambda60_{}".format(i), 0, 1, plp.LpInteger))
        lambda70.append(plp.LpVariable(
            "lambda70_{}".format(i), 0, 1, plp.LpInteger))
        lambda80.append(plp.LpVariable(
            "lambda80_{}".format(i), 0, 1, plp.LpInteger))
        lambda90.append(plp.LpVariable(
            "lambda90_{}".format(i), 0, 1, plp.LpInteger))
        lambda100.append(plp.LpVariable(
            "lambda100_{}".format(i), 0, 1, plp.LpInteger))

    # Problem definition
    prob = plp.LpProblem("Electricity_cost_minimization", plp.LpMinimize)

    # Constraints
    for i in range(len(grid)):
        prob += wind_el[i] <= wind[i]
        prob += pv_el[i] <= pv[i]
        prob += elz[i] == wind_el[i] + pv_el[i] + grid_el[i]
        # defining minimum electrolyzer load
        prob += elz[i] >= elz_min * elz_mode[i]
        # defining maximum electrolyzer load
        prob += elz[i] <= elz_max * elz_mode[i]
        # defining minimum methanation load
        prob += h2_use[i] >= meth_min * meth_on[i]
        # defining maximum methanation load
        prob += h2_use[i] <= meth_max * meth_on[i]

        # Part-load variable constraints
        prob += e10[i] <= elz_max * 0.1 * lambda10[i]
        prob += e10[i] >= elz_max * 0 * lambda10[i]
        prob += e20[i] <= elz_max * 0.2 * lambda20[i]
        prob += e20[i] >= elz_max * 0.1 * lambda20[i]
        prob += e30[i] <= elz_max * 0.3 * lambda30[i]
        prob += e30[i] >= elz_max * 0.2 * lambda30[i]
        prob += e40[i] <= elz_max * 0.4 * lambda40[i]
        prob += e40[i] >= elz_max * 0.3 * lambda40[i]
        prob += e50[i] <= elz_max * 0.5 * lambda50[i]
        prob += e50[i] >= elz_max * 0.4 * lambda50[i]
        prob += e60[i] <= elz_max * 0.6 * lambda60[i]
        prob += e60[i] >= elz_max * 0.5 * lambda60[i]
        prob += e70[i] <= elz_max * 0.7 * lambda70[i]
        prob += e70[i] >= elz_max * 0.6 * lambda70[i]
        prob += e80[i] <= elz_max * 0.8 * lambda80[i]
        prob += e80[i] >= elz_max * 0.7 * lambda80[i]
        prob += e90[i] <= elz_max * 0.9 * lambda90[i]
        prob += e90[i] >= elz_max * 0.8 * lambda90[i]
        prob += e100[i] <= elz_max * 1 * lambda100[i]
        prob += e100[i] >= elz_max * 0.9 * lambda100[i]
        # Total electrolyzer load
        prob += elz[i] == e10[i] + e20[i] + e30[i] + e40[i] + \
            e50[i] + e60[i] + e70[i] + e80[i] + e90[i] + e100[i]
        # Only one part of linearization simultaneously (if not working, test to add an e0 varaible?)
        prob += lambda10[i] == elz_mode[i] - lambda20[i] - lambda30[i] - lambda40[i] - \
            lambda50[i] - lambda60[i] - lambda70[i] - \
            lambda80[i] - lambda90[i] - lambda100[i]
        # prob += elz_mode[i] == lambda10[i] + lambda20[i] + lambda30[i] + lambda40[i] + lambda50[i] + lambda60[i] + lambda70[i] + lambda80[i] + lambda90[i] + lambda100[i]

        # Linearized H2 production efficiency
        prob += h2_prod[i] == ((e10[i]*phi10/elz_max) + (lambda10[i]*gamma10) + (e20[i]*phi20/elz_max) + (lambda20[i]*gamma20) + (e30[i]*phi30/elz_max) + (lambda30[i]*gamma30) +
                               (e40[i]*phi40/elz_max) + (lambda40[i]*gamma40) + (e50[i]*phi50/elz_max) + (lambda50[i]*gamma50) + (e60[i]*phi60/elz_max) + (lambda60[i]*gamma60) +
                               (e70[i]*phi70/elz_max) + (lambda70[i]*gamma70) + (e80[i]*phi80/elz_max) + (lambda80[i]*gamma80) + (e90[i]*phi90/elz_max) + (lambda90[i]*gamma90) + (e100[i]*phi100/elz_max) + (lambda100[i]*gamma100))
        # To add a start-up time, multiply this sum by the assumed fraction of an hour that is lost

        # defining start-up (Baumhof a good source for this)
        if i > 0:
            # Electrolyzer
            prob += elz_start[i] == elz_mode[i] - \
                elz_mode[i-1] - elz_standby[i-1] + elz_stop[i]
            # Methanation
            prob += meth_start[i] == meth_on[i] - meth_on[i-1] + meth_stop[i]
            # Can't go to standby from off
            prob += elz_standby[i] <= 1 - elz_off[i-1]
        else:
            if prev_mode == 1:
                prob += elz_start[i] == 0
            elif prev_mode == 0:
                prob += elz_start[i] == elz_mode[i]

            if prev_mode_meth == 1:
                prob += meth_start[i] == 0
                prob += meth_on[i] == 1 - meth_stop[i]
            elif prev_mode_meth == 0:
                prob += meth_start[i] == meth_on[i]
                prob += meth_on[i] == meth_stop[i]

        # Electrolyzer modes (Baumhof)
        prob += elz_mode[i] + elz_standby[i] + elz_off[i] == 1
        prob += elz_start[i] + elz_stop[i] <= 1
        # Methanation mode
        prob += meth_start[i] + meth_stop[i] <= 1
        # Can't exceed hourly CO2 availability
        prob += h2_use[i] <= hr_demand[i]

        # Linearized H2 production efficiency
        # prob += h2_prod[i] == (14.89002 * elz_mode[i]) + (elz[i] * 167.5127 / elz_max)

        # Storage charging/discharging
        if i == 0:  # using previous day storage value for first hour
            prob += h2st[i] == h2_prod[i] - h2_use[i] + h2st_prev

        else:
            prob += h2st[i] == h2_prod[i] - h2_use[i] + h2st[i-1]

        # Startup time-related losses
        # Methanation
        # want to multiply h2_use with the loss factor to get the lost hydrogen, using big M approach
        prob += meth_loss[i] <= meth_start[i] * 1000000000
        prob += meth_loss[i] >= 0
        prob += meth_loss[i] <= h2_use[i] * meth_loss_factor
        prob += meth_loss[i] >= (h2_use[i] * meth_loss_factor) - \
            ((1-meth_start[i])*1000000000)
        # later converting lost hydrogen to electricity using the efficiency at rated power (assumption)

    # Fulfill daily demand if technically possible
    # prob += h2_miss == demand - plp.lpSum([h2_prod[i] for i in range(len(grid))])
    # Do no exceed daily demand
    prob += plp.lpSum([h2_use[i] for i in range(len(grid))]) <= demand

    # objective (minimize the electricity cost)
    prob += plp.lpSum([grid_el[i] * grid[i] for i in range(len(grid))]) + \
        plp.lpSum([wind_el[i] * wind_cost for i in range(len(grid))]) + \
        plp.lpSum([pv_el[i] * pv_cost for i in range(len(grid))]) + \
        ((demand - plp.lpSum([h2_use[i] for i in range(len(grid))]))*10000000) + \
        plp.lpSum([elz_standby[i] * elz_max*standby_cost*grid[i] for i in range(len(grid))]) + plp.lpSum([elz_start[i] * elz_max*startup_cost*grid[i]
                                                                                                          for i in range(len(grid))]) + plp.lpSum([(meth_loss[i]*39.4/elz_eff) * (sum(grid)/24) for i in range(len(grid))])
    # standby and start-up cost for elz, methanation startup loss defined by converting to energy content, assuming elz efficiency at rated power and daily average electricity price

    # TEST AND COMPARE TO WITHOUT COLD START!

    # check solution
    solver = plp.GUROBI_CMD()  # plp.getSolver('GUROBI_CMD')
    #solver = plp.PULP_CBC_CMD()
    status = prob.solve(solver)

    e_op = []
    grid_op = []
    wind_op = []
    pv_op = []
    storage = []
    h2_produced = []
    h2_used = []
    e_on = []
    e_standby = []
    e_start = []
    e_off = []
    m_on = []
    m_start = []
    m_stop = []
    m_loss = []

    # PL variables
    # E10 = []
    # E20 = []
    # E30 = []
    # E40 = []
    # E50 = []
    # E60 = []
    # E70 = []
    # E80 = []
    # E90 = []
    # E100 = []
    # L10 = []
    # L20 = []
    # L30 = []
    # L40 = []
    # L50 = []
    # L60 = []
    # L70 = []
    # L80 = []
    # L90 = []
    # L100 = []

    # saving variable solutions
    for i in range(len(grid)):
        e_op.append(elz[i].varValue)
        grid_op.append(grid_el[i].varValue)
        wind_op.append(wind_el[i].varValue)
        pv_op.append(pv_el[i].varValue)
        storage.append(h2st[i].varValue)
        h2_produced.append(h2_prod[i].varValue)
        h2_used.append(h2_use[i].varValue)
        e_standby.append(elz_standby[i].varValue)
        e_start.append(elz_start[i].varValue)
        e_off.append(elz_off[i].varValue)
        e_on.append(elz_mode[i].varValue)
        m_on.append(meth_on[i].varValue)
        m_start.append(meth_start[i].varValue)
        m_stop.append(meth_stop[i].varValue)
        m_loss.append(meth_loss[i].varValue)

        # #PL
        # E10.append(e10[i].varValue)
        # E20.append(e20[i].varValue)
        # E30.append(e30[i].varValue)
        # E40.append(e40[i].varValue)
        # E50.append(e50[i].varValue)
        # E60.append(e60[i].varValue)
        # E70.append(e70[i].varValue)
        # E80.append(e80[i].varValue)
        # E90.append(e90[i].varValue)
        # E100.append(e100[i].varValue)
        # L10.append(lambda10[i].varValue)
        # L20.append(lambda20[i].varValue)
        # L30.append(lambda30[i].varValue)
        # L40.append(lambda40[i].varValue)
        # L50.append(lambda50[i].varValue)
        # L60.append(lambda60[i].varValue)
        # L70.append(lambda70[i].varValue)
        # L80.append(lambda80[i].varValue)
        # L90.append(lambda90[i].varValue)
        # L100.append(lambda100[i].varValue)

    h2_missed = np.zeros(24) + demand - sum(h2_used)
    demand_vector = np.zeros(24) + demand

    op = pd.DataFrame(
        {'Electrolyzer operation': e_op,
         'Grid purchases': grid_op,
         'Wind power': wind_op,
         'PV power': pv_op,
         'Electricity price': grid,
         'Storage': storage,
         'Demand': demand_vector,
         'H2 prod': h2_produced,
         'H2 used': h2_used,
         # 'Efficiency': eff,
         'Unmet demand': h2_missed,
         'Status': status,
         'On': e_on,
         'Standby': e_standby,
         'Off': e_off,
         'Cold start': e_start,
         'Meth on': m_on,
         'Meth start': m_start,
         'Meth stop': m_stop,
         'Meth loss': m_loss,
         # 'E10': E10,
         # 'E20': E20,
         # 'E30': E30,
         # 'E40': E40,
         # 'E50': E50,
         # 'E60': E60,
         # 'E70': E70,
         # 'E80': E80,
         # 'E90': E90,
         # 'E100': E100,
         # 'L10': L10,
         # 'L20': L20,
         # 'L30': L30,
         # 'L40': L40,
         # 'L50': L50,
         # 'L60': L60,
         # 'L70': L70,
         # 'L80': L80,
         # 'L90': L90,
         # 'L100': L100,
         })
    return op


def pwl10_byprods_econ(
    # daily_h2_demand, #[kg/d]
    hr_h2_demand, #[kg/h]
    heat_demand, #[kWh/h] hourly
    o2_demand, #[mol/h] hourly
    o2_power, #[kWh avoided/kg O2]
    k_values,
    m_values,
    grid,
    wind,
    pv,
    elz_max,  # kW
    elz_min,  # kW
    aux_cons,  # kW
    h2st_max,  # kg?
    h2st_prev,  # storage from previous day
    meth_max,  # maximum H2 use [kg/h]
    meth_min,  # minimum H2 use [kg/h]
    meth_cons,
    heat_value: float = 0.0, #NOT SURE WHAT THIS IS SUPPOSED TO BE
    startup_cost: float = 0.1,  # fraction of rated power
    standby_cost: float = 0.02,  # fraction of rated power
    # [hour] time until H2 production starts after a cold startup
    elz_startup_time: float = 0.0,
    # [hour] time until electrolyzer has reached operating temperature
    elz_heat_time: float = 0.0,
    # fraction of extra hydrogen lost during methanation startup (only first hour)
    meth_loss_factor: float = 0.1,
    prev_mode: int = 1,  # on/standby/off in last hour of previous day
    prev_mode_meth: int = 1,  # methanation on/off in last hour of previous day
    elz_eff: float = 0.7,  # HHV
    wind_cost: float = 0,  # [€/MWh]
    pv_cost: float = 0,  # [€/MWh]
    bat_cap: float = 0, # [kWh]
    bat_eff: float = 0.95, # battery round trip efficiency
    # h2_hv: float = 39.4,  # LHV or HHV
) -> pd.DataFrame:
    """Returns electrolyzer dispatch based on both renewable and grid electricity
    optimized to minimize the cost of electricity for a fixed H2 demand.
    Also including a linearized part-load efficiency and on/standby/off modes for
    the electrolyzer as well as startup losses for electrolysis and methanation.
    Using linearization parameters as proposed in Wirtz et al. 2021"""
    # Add methanation start-up penalty to ensure a more stable operation?
    # Now: lowest value in each interval has the optimal efficiency it seems
    # What about the start-up time? Can we ignore this since it is very short with PEM?
    # Should startup and standby costs be based on spot price or a constant price?
    # Add methanation standby to encourage its use? And methnanation electricity cost?

    # daily_h2_demand = daily_h2_demand.flatten().tolist()
    hr_h2_demand = hr_h2_demand.flatten().tolist()

    wind = list(wind)
    pv = list(pv)
    grid = list(grid)
    h2st_prev = h2st_prev * 2.02 / 1000  # mol to kg
    # h2_max = elz_max

    # Define variables
    elz = []
    grid_el = []
    wind_el = []
    pv_el = []
    h2st = []  # unit for this? [kg] now
    # h2_miss = []
    h2_prod = []
    elz_mode = []
    elz_start = []
    elz_stop = []
    elz_off = []
    elz_standby = []
    h2_use = []
    meth_on = []
    meth_start = []
    meth_stop = []
    meth_loss = []
    o2_prod = []
    o2_use = []
    bat = []
    bat_out = []
    # Part-load variables
    # Load
    e10 = []
    e20 = []
    e30 = []
    e40 = []
    e50 = []
    e60 = []
    e70 = []
    e80 = []
    e90 = []
    e100 = []
    # On/off
    lambda10 = []
    lambda20 = []
    lambda30 = []
    lambda40 = []
    lambda50 = []
    lambda60 = []
    lambda70 = []
    lambda80 = []
    lambda90 = []
    lambda100 = []

    # Linearization parameters (see "Linearized efficiency" Excel)
    # gradient
    k10 = k_values[0]
    k20 = k_values[1]
    k30 = k_values[2]
    k40 = k_values[3]
    k50 = k_values[4]
    k60 = k_values[5]
    k70 = k_values[6]
    k80 = k_values[7]
    k90 = k_values[8]
    k100 = k_values[9]
    # m-value
    m10 = m_values[0]
    m20 = m_values[1]
    m30 = m_values[2]
    m40 = m_values[3]
    m50 = m_values[4]
    m60 = m_values[5]
    m70 = m_values[6]
    m80 = m_values[7]
    m90 = m_values[8]
    m100 = m_values[9]

    for i in range(len(grid)):
        # Electrolyzer operation
        elz.append(plp.LpVariable("elz_{}".format(i), 0, elz_max))
        elz_mode.append(plp.LpVariable("elz_mode_{}".format(i), 0, 1, plp.LpInteger))
        elz_start.append(plp.LpVariable("elz_start_{}".format(i), 0, 1, plp.LpInteger))
        elz_stop.append(plp.LpVariable("elz_stop_{}".format(i), 0, 1, plp.LpInteger))
        elz_standby.append(plp.LpVariable("elz_standby_{}".format(i), 0, 1, plp.LpInteger))
        elz_off.append(plp.LpVariable("elz_off_{}".format(i), 0, 1, plp.LpInteger))
        # Purchased grid electricity
        grid_el.append(plp.LpVariable("grid_el_{}".format(i), 0, elz_max))
        # Wind-based operation
        wind_el.append(plp.LpVariable("wind_el_{}".format(i), 0, elz_max))
        # PV-based operation
        pv_el.append(plp.LpVariable("pv_el_{}".format(i), 0, elz_max))
        # Hydrogen storage
        h2st.append(plp.LpVariable("h2st_{}".format(i), 0, h2st_max))
        # Unmet demand
        # h2_miss.append(plp.LpVariable("h2_miss_{}".format(i),0,None))
        # Hydrogen production
        h2_prod.append(plp.LpVariable("h2_prod_{}".format(i), 0, None))
        # Hydrogen utilization
        h2_use.append(plp.LpVariable("h2_use_{}".format(i), 0, meth_max))
        # Oxygen production
        o2_prod.append(plp.LpVariable("o2_prod_{}".format(i), 0, None))
        # Oxygen utilization
        o2_use.append(plp.LpVariable("o2_use_{}".format(i), 0, None))
        # Methanation
        meth_on.append(plp.LpVariable("meth_on_{}".format(i), 0, 1, plp.LpInteger))
        meth_start.append(plp.LpVariable("meth_start_{}".format(i), 0, 1, plp.LpInteger))
        meth_stop.append(plp.LpVariable("meth_stop_{}".format(i), 0, 1, plp.LpInteger))
        meth_loss.append(plp.LpVariable("meth_loss_{}".format(i), 0, elz_max))
        #Battery
        bat.append(plp.LpVariable("bat_{}".format(i), 0, bat_cap))
        bat_out.append(plp.LpVariable("bat_out_{}".format(i), 0, bat_cap))
        
        # Part-load variables
        # Load
        e10.append(plp.LpVariable("e10_{}".format(i), 0, None))
        e20.append(plp.LpVariable("e20_{}".format(i), 0, None))
        e30.append(plp.LpVariable("e30_{}".format(i), 0, None))
        e40.append(plp.LpVariable("e40_{}".format(i), 0, None))
        e50.append(plp.LpVariable("e50_{}".format(i), 0, None))
        e60.append(plp.LpVariable("e60_{}".format(i), 0, None))
        e70.append(plp.LpVariable("e70_{}".format(i), 0, None))
        e80.append(plp.LpVariable("e80_{}".format(i), 0, None))
        e90.append(plp.LpVariable("e90_{}".format(i), 0, None))
        e100.append(plp.LpVariable("e100_{}".format(i), 0, None))
        # On/off
        lambda10.append(plp.LpVariable(
            "lambda10_{}".format(i), 0, 1, plp.LpInteger))
        lambda20.append(plp.LpVariable(
            "lambda20_{}".format(i), 0, 1, plp.LpInteger))
        lambda30.append(plp.LpVariable(
            "lambda30_{}".format(i), 0, 1, plp.LpInteger))
        lambda40.append(plp.LpVariable(
            "lambda40_{}".format(i), 0, 1, plp.LpInteger))
        lambda50.append(plp.LpVariable(
            "lambda50_{}".format(i), 0, 1, plp.LpInteger))
        lambda60.append(plp.LpVariable(
            "lambda60_{}".format(i), 0, 1, plp.LpInteger))
        lambda70.append(plp.LpVariable(
            "lambda70_{}".format(i), 0, 1, plp.LpInteger))
        lambda80.append(plp.LpVariable(
            "lambda80_{}".format(i), 0, 1, plp.LpInteger))
        lambda90.append(plp.LpVariable(
            "lambda90_{}".format(i), 0, 1, plp.LpInteger))
        lambda100.append(plp.LpVariable(
            "lambda100_{}".format(i), 0, 1, plp.LpInteger))

    # Problem definition
    prob = plp.LpProblem("Electricity_cost_minimization", plp.LpMinimize)

    # Constraints
    for i in range(len(grid)):
        prob += wind_el[i] <= wind[i]
        prob += pv_el[i] <= pv[i]
        prob += elz[i] == wind_el[i] + pv_el[i] + grid_el[i] + bat_out[i]
        prob += bat[i] <= ((wind[i] + pv[i] - wind_el[i] - pv_el[i]) * bat_eff) - bat_out[i]
        # defining minimum electrolyzer load
        prob += elz[i] >= elz_min * elz_mode[i]
        # defining maximum electrolyzer load
        prob += elz[i] <= elz_max * elz_mode[i]
        # defining minimum methanation load
        prob += h2_use[i] >= meth_min * meth_on[i]
        # defining maximum methanation load
        prob += h2_use[i] <= meth_max * meth_on[i]

        # Part-load variable constraints
        prob += e10[i] <= elz_max * 0.1 * lambda10[i]
        prob += e10[i] >= elz_max * 0 * lambda10[i]
        prob += e20[i] <= elz_max * 0.2 * lambda20[i]
        prob += e20[i] >= elz_max * 0.1 * lambda20[i]
        prob += e30[i] <= elz_max * 0.3 * lambda30[i]
        prob += e30[i] >= elz_max * 0.2 * lambda30[i]
        prob += e40[i] <= elz_max * 0.4 * lambda40[i]
        prob += e40[i] >= elz_max * 0.3 * lambda40[i]
        prob += e50[i] <= elz_max * 0.5 * lambda50[i]
        prob += e50[i] >= elz_max * 0.4 * lambda50[i]
        prob += e60[i] <= elz_max * 0.6 * lambda60[i]
        prob += e60[i] >= elz_max * 0.5 * lambda60[i]
        prob += e70[i] <= elz_max * 0.7 * lambda70[i]
        prob += e70[i] >= elz_max * 0.6 * lambda70[i]
        prob += e80[i] <= elz_max * 0.8 * lambda80[i]
        prob += e80[i] >= elz_max * 0.7 * lambda80[i]
        prob += e90[i] <= elz_max * 0.9 * lambda90[i]
        prob += e90[i] >= elz_max * 0.8 * lambda90[i]
        prob += e100[i] <= elz_max * 1 * lambda100[i]
        prob += e100[i] >= elz_max * 0.9 * lambda100[i]
        # Total electrolyzer load
        prob += elz[i] == e10[i] + e20[i] + e30[i] + e40[i] + \
            e50[i] + e60[i] + e70[i] + e80[i] + e90[i] + e100[i]
        # Only one part of linearization simultaneously (if not working, test to add an e0 varaible?)
        prob += lambda10[i] == elz_mode[i] - lambda20[i] - lambda30[i] - lambda40[i] - \
            lambda50[i] - lambda60[i] - lambda70[i] - \
            lambda80[i] - lambda90[i] - lambda100[i]
        # prob += elz_mode[i] == lambda10[i] + lambda20[i] + lambda30[i] + lambda40[i] + lambda50[i] + lambda60[i] + lambda70[i] + lambda80[i] + lambda90[i] + lambda100[i]

        # Linearized H2 production efficiency
        prob += h2_prod[i] == ((e10[i]*k10/elz_max) + (lambda10[i]*m10) + (e20[i]*k20/elz_max) + (lambda20[i]*m20) + (e30[i]*k30/elz_max) + (lambda30[i]*m30) +
                               (e40[i]*k40/elz_max) + (lambda40[i]*m40) + (e50[i]*k50/elz_max) + (lambda50[i]*m50) + (e60[i]*k60/elz_max) + (lambda60[i]*m60) +
                               (e70[i]*k70/elz_max) + (lambda70[i]*m70) + (e80[i]*k80/elz_max) + (lambda80[i]*m80) + (e90[i]*k90/elz_max) + (lambda90[i]*m90) + (e100[i]*k100/elz_max) + (lambda100[i]*m100))
        # To add a start-up time, multiply this sum by the assumed fraction of an hour that is lost

        # defining start-up (Baumhof a good source for this)
        if i > 0:
            # Electrolyzer
            prob += elz_start[i] == elz_mode[i] - elz_mode[i-1] - elz_standby[i-1] + elz_stop[i]
            # Methanation
            prob += meth_start[i] == meth_on[i] - meth_on[i-1] + meth_stop[i]
            # Can't go to standby from off
            prob += elz_standby[i] <= 1 - elz_off[i-1]
        else:  # considering end of previous day
            if prev_mode == 1:
                prob += elz_start[i] == 0
            elif prev_mode == 0:
                prob += elz_start[i] == elz_mode[i]

            if prev_mode_meth == 1:
                prob += meth_start[i] == 0
                prob += meth_on[i] == 1 - meth_stop[i]
            elif prev_mode_meth == 0:
                prob += meth_start[i] == meth_on[i]
                prob += meth_on[i] == meth_stop[i]

        # Electrolyzer modes (Baumhof)
        prob += elz_mode[i] + elz_standby[i] + elz_off[i] == 1
        prob += elz_start[i] + elz_stop[i] <= 1
        # Methanation mode
        prob += meth_start[i] + meth_stop[i] <= 1
        
        # Can't exceed hourly CO2 availability
        prob += h2_use[i] <= hr_h2_demand[i]

        # Storage charging/discharging
        if i == 0:  # using previous day storage value for first hour
            prob += h2st[i] == h2_prod[i] - h2_use[i] + h2st_prev

        else:
            prob += h2st[i] == h2_prod[i] - h2_use[i] + h2st[i-1]

        # Startup time-related losses
        # Methanation
        # want to multiply h2_use with the loss factor to get the lost hydrogen, using big M approach
        prob += meth_loss[i] <= meth_start[i] * 1000000000
        prob += meth_loss[i] >= 0
        prob += meth_loss[i] <= h2_use[i] * meth_loss_factor
        prob += meth_loss[i] >= (h2_use[i] * meth_loss_factor) - \
            ((1-meth_start[i])*1000000000)
        # later converting lost hydrogen to electricity using the efficiency at rated power (assumption)

        # By-product generation
        # Oxygen
        # prob += o2_prod[i] == h2_prod[i] * (32/2.02) * 0.5  # kg of o2 produced
        # prob += o2_use[i] <= o2_prod[i]
        # prob += o2_use[i] <= o2_demand[i]
        # Heat (NON-LINEAR HEAT RELATIONS, WHAT IS THE IMPACT? If it doesn't work, try a simplified relation, or PWL for heat)
        # elz_heat[i] == ((h2_prod[i]*h2_hv)/(elz[i]-aux_cons))
        # meth_heat[i] == 

    # Fulfill hourly demand if technically possible (using hourly demand solely!)
    # prob += h2_miss == demand - plp.lpSum([h2_prod[i] for i in range(len(grid))])
    # Do no exceed daily demand
    # prob += plp.lpSum([h2_use[i] for i in range(len(grid))]) <= daily_h2_demand
    
    # objective (minimize the electricity cost)
    prob += plp.lpSum([grid_el[i] * grid[i] for i in range(len(grid))]) + plp.lpSum([wind_el[i] * wind_cost for i in range(len(grid))]) + plp.lpSum([pv_el[i] * pv_cost for i in range(len(grid))]) + \
        ((plp.lpSum([hr_h2_demand[i] - h2_use[i] for i in range(len(grid))]))*10000000) + plp.lpSum([elz_standby[i] * elz_max*standby_cost*grid[i] for i in range(len(grid))]) + \
        plp.lpSum([elz_start[i] * elz_max*startup_cost*grid[i] for i in range(len(grid))]) + plp.lpSum([(meth_loss[i]*39.4/elz_eff) * (sum(grid)/24) for i in range(len(grid))])# - \
        #plp.lpSum([o2_use[i] * (o2_power * grid[i] / 1000) for i in range(len(grid))])# + plp.lpSum([meth_on[i] * (meth_cons * grid[i]) for i in range(len(grid))])
    #electricity costs (grid, wind, pv)
    #missed demand "cost" and electrolyzer standby cost
    #start-up cost for elz, methanation startup loss defined by converting to energy content, assuming elz efficiency at rated power and daily average electricity price
    #oxygen income + methanation electricity cost
    
    # TEST AND COMPARE TO WITHOUT COLD START!

    # check solution
    solver = plp.GUROBI_CMD()  # plp.getSolver('GUROBI_CMD')
    #solver = plp.PULP_CBC_CMD()
    status = prob.solve(solver)

    e_op = []
    grid_op = []
    wind_op = []
    pv_op = []
    storage = []
    h2_produced = []
    h2_used = []
    e_on = []
    e_standby = []
    e_start = []
    e_off = []
    m_on = []
    m_start = []
    m_stop = []
    m_loss = []
    bat_state = []
    bat_discharge = []

    # PL variables
    # E10 = []
    # E20 = []
    # E30 = []
    # E40 = []
    # E50 = []
    # E60 = []
    # E70 = []
    # E80 = []
    # E90 = []
    # E100 = []
    # L10 = []
    # L20 = []
    # L30 = []
    # L40 = []
    # L50 = []
    # L60 = []
    # L70 = []
    # L80 = []
    # L90 = []
    # L100 = []

    # saving variable solutions
    for i in range(len(grid)):
        e_op.append(elz[i].varValue)
        grid_op.append(grid_el[i].varValue)
        wind_op.append(wind_el[i].varValue)
        pv_op.append(pv_el[i].varValue)
        storage.append(h2st[i].varValue)
        h2_produced.append(h2_prod[i].varValue)
        h2_used.append(h2_use[i].varValue)
        e_standby.append(elz_standby[i].varValue)
        e_start.append(elz_start[i].varValue)
        e_off.append(elz_off[i].varValue)
        e_on.append(elz_mode[i].varValue)
        m_on.append(meth_on[i].varValue)
        m_start.append(meth_start[i].varValue)
        m_stop.append(meth_stop[i].varValue)
        m_loss.append(meth_loss[i].varValue)
        bat_state.append(bat[i].varValue)
        bat_discharge.append(bat_out[i].varValue)

        # #PL
        # E10.append(e10[i].varValue)
        # E20.append(e20[i].varValue)
        # E30.append(e30[i].varValue)
        # E40.append(e40[i].varValue)
        # E50.append(e50[i].varValue)
        # E60.append(e60[i].varValue)
        # E70.append(e70[i].varValue)
        # E80.append(e80[i].varValue)
        # E90.append(e90[i].varValue)
        # E100.append(e100[i].varValue)
        # L10.append(lambda10[i].varValue)
        # L20.append(lambda20[i].varValue)
        # L30.append(lambda30[i].varValue)
        # L40.append(lambda40[i].varValue)
        # L50.append(lambda50[i].varValue)
        # L60.append(lambda60[i].varValue)
        # L70.append(lambda70[i].varValue)
        # L80.append(lambda80[i].varValue)
        # L90.append(lambda90[i].varValue)
        # L100.append(lambda100[i].varValue)

    h2_missed = list(np.array(hr_h2_demand) - np.array(h2_used))
    demand_vector = hr_h2_demand

    op = pd.DataFrame(
        {'Electrolyzer operation': e_op,
         'Grid purchases': grid_op,
         'Wind power': wind_op,
         'PV power': pv_op,
         'Electricity price': grid,
         'Storage': storage,
         'Demand': demand_vector,
         'H2 prod': h2_produced,
         'H2 used': h2_used,
         # 'Efficiency': eff,
         'Unmet demand': h2_missed,
         'Status': status,
         'On': e_on,
         'Standby': e_standby,
         'Off': e_off,
         'Cold start': e_start,
         'Meth on': m_on,
         'Meth start': m_start,
         'Meth stop': m_stop,
         'Meth loss': m_loss,
         'Battery state': bat_state,
         'Battery discharging': bat_discharge,
         # 'E10': E10,
         # 'E20': E20,
         # 'E30': E30,
         # 'E40': E40,
         # 'E50': E50,
         # 'E60': E60,
         # 'E70': E70,
         # 'E80': E80,
         # 'E90': E90,
         # 'E100': E100,
         # 'L10': L10,
         # 'L20': L20,
         # 'L30': L30,
         # 'L40': L40,
         # 'L50': L50,
         # 'L60': L60,
         # 'L70': L70,
         # 'L80': L80,
         # 'L90': L90,
         # 'L100': L100,
         })
    return op


def pwl10_byprods_econ_stack(
    hr_h2_demand, #[kg/h]
    heat_demand, #[kWh/h] hourly
    o2_demand, #[mol/h] hourly
    o2_power, #[kWh avoided/kg O2]
    k_values,
    m_values,
    grid,
    wind,
    pv,
    elz_max,  # kW
    elz_min,  # kW
    aux_cons,  # kW
    h2st_max,  # kg?
    h2st_prev,  # storage from previous day
    meth_max,  # maximum H2 use [kg/h]
    meth_min,  # minimum H2 use [kg/h]
    meth_cons,
    meth_heat_value: float = 23.837, #MWh heat from methanation/kgH2 in
    startup_cost: float = 0.1,  # fraction of rated power
    standby_cost: float = 0.02,  # fraction of rated power
    # [hour] time until H2 production starts after a cold startup
    elz_startup_time: float = 0.0,
    # [hour] time until electrolyzer has reached operating temperature
    elz_heat_time: float = 0.0,
    # fraction of extra hydrogen lost during methanation startup (only first hour)
    meth_loss_factor: float = 0.1,
    prev_mode: int = 1,  # on/standby/off in last hour of previous day
    prev_mode_meth: int = 1,  # methanation on/off in last hour of previous day
    elz_eff: float = 0.7,  # HHV
    wind_cost: float = 0,  # [€/MWh]
    pv_cost: float = 0,  # [€/MWh]
    dh_winter: float = 56.0,
    dh_spring: float = 37.0,
    dh_summer: float = 24.0,
    dh_autumn: float = 37.0,
    # h2_hv: float = 39.4,  # LHV or HHV
) -> pd.DataFrame:
    """Returns electrolyzer dispatch based on both renewable and grid electricity
    optimized to minimize the cost of electricity for a fixed H2 demand.
    Also including a linearized part-load efficiency and on/standby/off modes for
    the electrolyzer as well as startup losses for electrolysis and methanation.
    Using linearization parameters as proposed in Wirtz et al. 2021"""
    # Add methanation start-up penalty to ensure a more stable operation?
    # Now: lowest value in each interval has the optimal efficiency it seems
    # What about the start-up time? Can we ignore this since it is very short with PEM?
    # Should startup and standby costs be based on spot price or a constant price?
    # Add methanation standby to encourage its use? And methnanation electricity cost?
    # Implemented no heat generation during cold elz start. But this (and 5 min loss) should be only start-up cost?

    #Standard parameters
    stack_min = 0
    stack_max = elz_max - aux_cons

    hr_h2_demand = hr_h2_demand.flatten().tolist()

    wind = list(wind)
    pv = list(pv)
    grid = list(grid)
    h2st_prev = h2st_prev * 2.02 / 1000  # mol to kg
    # h2_max = elz_max

    # Define variables
    elz = []
    stack = []
    grid_el = []
    wind_el = []
    pv_el = []
    h2st = []  # unit for this? [kg] now
    # h2_miss = []
    h2_prod = []
    elz_mode = []
    elz_start = []
    elz_stop = []
    elz_off = []
    elz_standby = []
    h2_use = []
    meth_on = []
    meth_start = []
    meth_stop = []
    meth_loss = []
    o2_prod = []
    o2_use = []
    elz_heat = []
    meth_heat = []
    tot_heat = []
    heat_income = []
    heat_use = []
    # Part-load variables
    # Load
    e10 = []
    e20 = []
    e30 = []
    e40 = []
    e50 = []
    e60 = []
    e70 = []
    e80 = []
    e90 = []
    e100 = []
    # On/off
    lambda10 = []
    lambda20 = []
    lambda30 = []
    lambda40 = []
    lambda50 = []
    lambda60 = []
    lambda70 = []
    lambda80 = []
    lambda90 = []
    lambda100 = []

    # Linearization parameters (see "Linearized efficiency" Excel)
    # gradient
    k10 = k_values[0]
    k20 = k_values[1]
    k30 = k_values[2]
    k40 = k_values[3]
    k50 = k_values[4]
    k60 = k_values[5]
    k70 = k_values[6]
    k80 = k_values[7]
    k90 = k_values[8]
    k100 = k_values[9]
    # m-value
    m10 = m_values[0]
    m20 = m_values[1]
    m30 = m_values[2]
    m40 = m_values[3]
    m50 = m_values[4]
    m60 = m_values[5]
    m70 = m_values[6]
    m80 = m_values[7]
    m90 = m_values[8]
    m100 = m_values[9]

    for i in range(len(grid)):
        # Electrolyzer operation
        elz.append(plp.LpVariable("elz_{}".format(i), 0, elz_max))
        elz_mode.append(plp.LpVariable("elz_mode_{}".format(i), 0, 1, plp.LpInteger))
        elz_start.append(plp.LpVariable("elz_start_{}".format(i), 0, 1, plp.LpInteger))
        elz_stop.append(plp.LpVariable("elz_stop_{}".format(i), 0, 1, plp.LpInteger))
        elz_standby.append(plp.LpVariable("elz_standby_{}".format(i), 0, 1, plp.LpInteger))
        elz_off.append(plp.LpVariable("elz_off_{}".format(i), 0, 1, plp.LpInteger))
        stack.append(plp.LpVariable("stack_{}".format(i), 0, elz_max))
        # Purchased grid electricity
        grid_el.append(plp.LpVariable("grid_el_{}".format(i), 0, elz_max))
        # Wind-based operation
        wind_el.append(plp.LpVariable("wind_el_{}".format(i), 0, elz_max))
        # PV-based operation
        pv_el.append(plp.LpVariable("pv_el_{}".format(i), 0, elz_max))
        # Hydrogen storage
        h2st.append(plp.LpVariable("h2st_{}".format(i), 0, h2st_max))
        # Unmet demand
        # h2_miss.append(plp.LpVariable("h2_miss_{}".format(i),0,None))
        # Hydrogen production
        h2_prod.append(plp.LpVariable("h2_prod_{}".format(i), 0, None))
        # Hydrogen utilization
        h2_use.append(plp.LpVariable("h2_use_{}".format(i), 0, meth_max))
        # Oxygen
        o2_prod.append(plp.LpVariable("o2_prod_{}".format(i), 0, None))
        o2_use.append(plp.LpVariable("o2_use_{}".format(i), 0, None))
        #Heat
        elz_heat.append(plp.LpVariable("elz_heat_{}".format(i), 0, elz_max))
        meth_heat.append(plp.LpVariable("meth_heat_{}".format(i), 0, elz_max))
        tot_heat.append(plp.LpVariable("tot_heat_{}".format(i), 0, elz_max))
        heat_income.append(plp.LpVariable("heat_income_{}".format(i), 0, 2*elz_max))
        heat_use.append(plp.LpVariable("heat_use_{}".format(i), 0, 2*elz_max))
        # Methanation
        meth_on.append(plp.LpVariable("meth_on_{}".format(i), 0, 1, plp.LpInteger))
        meth_start.append(plp.LpVariable("meth_start_{}".format(i), 0, 1, plp.LpInteger))
        meth_stop.append(plp.LpVariable("meth_stop_{}".format(i), 0, 1, plp.LpInteger))
        meth_loss.append(plp.LpVariable("meth_loss_{}".format(i), 0, elz_max))

        # Part-load variables
        # Load
        e10.append(plp.LpVariable("e10_{}".format(i), 0, None))
        e20.append(plp.LpVariable("e20_{}".format(i), 0, None))
        e30.append(plp.LpVariable("e30_{}".format(i), 0, None))
        e40.append(plp.LpVariable("e40_{}".format(i), 0, None))
        e50.append(plp.LpVariable("e50_{}".format(i), 0, None))
        e60.append(plp.LpVariable("e60_{}".format(i), 0, None))
        e70.append(plp.LpVariable("e70_{}".format(i), 0, None))
        e80.append(plp.LpVariable("e80_{}".format(i), 0, None))
        e90.append(plp.LpVariable("e90_{}".format(i), 0, None))
        e100.append(plp.LpVariable("e100_{}".format(i), 0, None))
        # On/off
        lambda10.append(plp.LpVariable(
            "lambda10_{}".format(i), 0, 1, plp.LpInteger))
        lambda20.append(plp.LpVariable(
            "lambda20_{}".format(i), 0, 1, plp.LpInteger))
        lambda30.append(plp.LpVariable(
            "lambda30_{}".format(i), 0, 1, plp.LpInteger))
        lambda40.append(plp.LpVariable(
            "lambda40_{}".format(i), 0, 1, plp.LpInteger))
        lambda50.append(plp.LpVariable(
            "lambda50_{}".format(i), 0, 1, plp.LpInteger))
        lambda60.append(plp.LpVariable(
            "lambda60_{}".format(i), 0, 1, plp.LpInteger))
        lambda70.append(plp.LpVariable(
            "lambda70_{}".format(i), 0, 1, plp.LpInteger))
        lambda80.append(plp.LpVariable(
            "lambda80_{}".format(i), 0, 1, plp.LpInteger))
        lambda90.append(plp.LpVariable(
            "lambda90_{}".format(i), 0, 1, plp.LpInteger))
        lambda100.append(plp.LpVariable(
            "lambda100_{}".format(i), 0, 1, plp.LpInteger))

    # Problem definition
    prob = plp.LpProblem("Electricity_cost_minimization", plp.LpMinimize)

    # Constraints
    for i in range(len(grid)):
        prob += wind_el[i] <= wind[i]
        prob += pv_el[i] <= pv[i]
        prob += elz[i] == wind_el[i] + pv_el[i] + grid_el[i]
        prob += stack[i] == elz[i] - aux_cons
        # defining minimum electrolyzer load
        prob += elz[i] >= elz_min * elz_mode[i]
        prob += stack[i] >= stack_min * elz_mode[i]
        # defining maximum electrolyzer load
        prob += elz[i] <= elz_max * elz_mode[i]
        prob += stack[i] <= stack_max * elz_mode[i]
        # defining minimum methanation load
        prob += h2_use[i] >= meth_min * meth_on[i]
        # defining maximum methanation load
        prob += h2_use[i] <= meth_max * meth_on[i]

        # Part-load variable constraints
        prob += e10[i] <= elz_max * 0.1 * lambda10[i]
        prob += e10[i] >= elz_max * 0 * lambda10[i]
        prob += e20[i] <= elz_max * 0.2 * lambda20[i]
        prob += e20[i] >= elz_max * 0.1 * lambda20[i]
        prob += e30[i] <= elz_max * 0.3 * lambda30[i]
        prob += e30[i] >= elz_max * 0.2 * lambda30[i]
        prob += e40[i] <= elz_max * 0.4 * lambda40[i]
        prob += e40[i] >= elz_max * 0.3 * lambda40[i]
        prob += e50[i] <= elz_max * 0.5 * lambda50[i]
        prob += e50[i] >= elz_max * 0.4 * lambda50[i]
        prob += e60[i] <= elz_max * 0.6 * lambda60[i]
        prob += e60[i] >= elz_max * 0.5 * lambda60[i]
        prob += e70[i] <= elz_max * 0.7 * lambda70[i]
        prob += e70[i] >= elz_max * 0.6 * lambda70[i]
        prob += e80[i] <= elz_max * 0.8 * lambda80[i]
        prob += e80[i] >= elz_max * 0.7 * lambda80[i]
        prob += e90[i] <= elz_max * 0.9 * lambda90[i]
        prob += e90[i] >= elz_max * 0.8 * lambda90[i]
        prob += e100[i] <= elz_max * 1 * lambda100[i]
        prob += e100[i] >= elz_max * 0.9 * lambda100[i]
        # Total electrolyzer load
        prob += elz[i] == e10[i] + e20[i] + e30[i] + e40[i] + \
            e50[i] + e60[i] + e70[i] + e80[i] + e90[i] + e100[i]
        # Only one part of linearization simultaneously (if not working, test to add an e0 varaible?)
        prob += lambda10[i] == elz_mode[i] - lambda20[i] - lambda30[i] - lambda40[i] - \
            lambda50[i] - lambda60[i] - lambda70[i] - \
            lambda80[i] - lambda90[i] - lambda100[i]
        # prob += elz_mode[i] == lambda10[i] + lambda20[i] + lambda30[i] + lambda40[i] + lambda50[i] + lambda60[i] + lambda70[i] + lambda80[i] + lambda90[i] + lambda100[i]

        # Linearized H2 production efficiency
        prob += h2_prod[i] == ((e10[i]*k10/elz_max) + (lambda10[i]*m10) + (e20[i]*k20/elz_max) + (lambda20[i]*m20) + (e30[i]*k30/elz_max) + (lambda30[i]*m30) +
                                (e40[i]*k40/elz_max) + (lambda40[i]*m40) + (e50[i]*k50/elz_max) + (lambda50[i]*m50) + (e60[i]*k60/elz_max) + (lambda60[i]*m60) +
                                (e70[i]*k70/elz_max) + (lambda70[i]*m70) + (e80[i]*k80/elz_max) + (lambda80[i]*m80) + (e90[i]*k90/elz_max) + (lambda90[i]*m90) + (e100[i]*k100/elz_max) + (lambda100[i]*m100))
        
        # To add a start-up time, multiply this sum by the assumed fraction of an hour that is lost

        # defining start-up (Baumhof a good source for this)
        if i > 0:
            # Electrolyzer
            prob += elz_start[i] == elz_mode[i] - elz_mode[i-1] - elz_standby[i-1] + elz_stop[i]
            # Methanation
            prob += meth_start[i] == meth_on[i] - meth_on[i-1] + meth_stop[i]
            # Can't go to standby from off
            prob += elz_standby[i] <= 1 - elz_off[i-1]
        else:  # considering end of previous day
            if prev_mode == 1:
                prob += elz_start[i] == 0
            elif prev_mode == 0:
                prob += elz_start[i] == elz_mode[i]

            if prev_mode_meth == 1:
                prob += meth_start[i] == 0
                prob += meth_on[i] == 1 - meth_stop[i]
            elif prev_mode_meth == 0:
                prob += meth_start[i] == meth_on[i]
                prob += meth_on[i] == meth_stop[i]

        # Electrolyzer modes (Baumhof)
        prob += elz_mode[i] + elz_standby[i] + elz_off[i] == 1
        prob += elz_start[i] + elz_stop[i] <= 1
        # Methanation mode
        prob += meth_start[i] + meth_stop[i] <= 1
        
        # Can't exceed hourly CO2 availability
        prob += h2_use[i] <= hr_h2_demand[i]

        # Storage charging/discharging
        if i == 0:  # using previous day storage value for first hour
            prob += h2st[i] == h2_prod[i] - h2_use[i] + h2st_prev

        else:
            prob += h2st[i] == h2_prod[i] - h2_use[i] + h2st[i-1]

        # Startup time-related losses
        # Methanation
        # want to multiply h2_use with the loss factor to get the lost hydrogen, using big M approach
        prob += meth_loss[i] <= meth_start[i] * 1000000000
        prob += meth_loss[i] >= 0
        prob += meth_loss[i] <= h2_use[i] * meth_loss_factor
        prob += meth_loss[i] >= (h2_use[i] * meth_loss_factor) - \
            ((1-meth_start[i])*1000000000)
        # later converting lost hydrogen to electricity using the efficiency at rated power (assumption)

        # By-product generation
        # Oxygen
        prob += o2_prod[i] == h2_prod[i] * (32/2.02) * 0.5  # kg of o2 produced
        # can use the minimum of production and demand; min(o2_prod[i],o2_demand[i])
        prob += o2_use[i] <= o2_prod[i]
        prob += o2_use[i] <= o2_demand[i]
        # prob += o2_prod[i] <= (100000000 * y1[i]) + o2_demand[i]
        # prob += o2_demand[i] <= (100000000 * (1-y1[i])) + o2_prod[i]
        # prob += o2_use[i] <= o2_prod
        # Heat (NON-LINEAR HEAT RELATIONS, WHAT IS THE IMPACT? If it doesn't work, try a simplified relation, or PWL for heat)
        elz_heat[i] <= stack[i] - (h2_prod[i] * 39.4) - (stack[i] * (10*997/(1000*18.02/2.02)) * 75.3 * (80 - 15) / (3600*1000)) #all "lost" energy in the stack is heat minus water heating
        #No heat production from electrolysis during cold start
        prob += elz_heat[i] <= elz_start[i] * 100000000
        prob += elz_heat[i] >= 0
        prob += elz_heat[i] >= (stack[i] - (h2_prod[i] * 39.4) - (stack[i] * (10*997/(1000*18.02/2.02)) * 75.3 * (80 - 15) / (3600*1000))) - ((1-elz_start[i])*100000000)
        #Other heat
        meth_heat[i] == meth_heat_value * h2_use[i]
        tot_heat[i] == elz_heat[i] + meth_heat[i]
        heat_use[i] <= tot_heat[i]
        heat_use[i] <= heat_demand[i]
        
        
        #Heat income (don't need tot since methanation is not part of dipatch, only need elz)
        if i < 1416 or i >= 8016:
            heat_income[i] == heat_use[i] * dh_winter / 1000
        elif i >= 1416 and i < 3624:
            heat_income[i] == heat_use[i] * dh_spring / 1000
        elif i >= 3624 and i < 5832:
            heat_income[i] == heat_use[i] * dh_summer / 1000
        elif i >= 5832 and i < 8016:
            heat_income[i] == heat_use[i] * dh_autumn / 1000
            
    
    # objective (minimize the electricity cost)
    prob += plp.lpSum([grid_el[i] * grid[i] for i in range(len(grid))]) + plp.lpSum([wind_el[i] * wind_cost for i in range(len(grid))]) + plp.lpSum([pv_el[i] * pv_cost for i in range(len(grid))]) + \
        ((plp.lpSum([hr_h2_demand[i] - h2_use[i] for i in range(len(grid))]))*10000000) + plp.lpSum([elz_standby[i] * elz_max*standby_cost*grid[i] for i in range(len(grid))]) + \
        plp.lpSum([elz_start[i] * elz_max*startup_cost*grid[i] for i in range(len(grid))]) + plp.lpSum([(meth_loss[i]*39.4/elz_eff) * (sum(grid)/24) for i in range(len(grid))]) - \
        plp.lpSum([o2_use[i] * (o2_power * grid[i] / 1000) for i in range(len(grid))]) - plp.lpSum([heat_income[i] for i in range(len(grid))])
        # + plp.lpSum([meth_on[i] * (meth_cons * grid[i]) for i in range(len(grid))])
    #electricity costs (grid, wind, pv)
    #missed demand "cost" and electrolyzer standby cost
    #start-up cost for elz, methanation startup loss defined by converting to energy content, assuming elz efficiency at rated power and daily average electricity price
    #oxygen income + methanation electricity cost (meth cost only relevant if it is sized so that it can be flexible?)
    
    #remember that heat price varies during the year
    
    # TEST AND COMPARE TO WITHOUT COLD START!

    # check solution
    solver = plp.GUROBI_CMD()  # plp.getSolver('GUROBI_CMD')
    #solver = plp.PULP_CBC_CMD()
    status = prob.solve(solver)

    e_op = []
    grid_op = []
    wind_op = []
    pv_op = []
    storage = []
    h2_produced = []
    h2_used = []
    e_on = []
    e_standby = []
    e_start = []
    e_off = []
    m_on = []
    m_start = []
    m_stop = []
    m_loss = []

    # PL variables
    # E10 = []
    # E20 = []
    # E30 = []
    # E40 = []
    # E50 = []
    # E60 = []
    # E70 = []
    # E80 = []
    # E90 = []
    # E100 = []
    # L10 = []
    # L20 = []
    # L30 = []
    # L40 = []
    # L50 = []
    # L60 = []
    # L70 = []
    # L80 = []
    # L90 = []
    # L100 = []

    # saving variable solutions
    for i in range(len(grid)):
        e_op.append(elz[i].varValue)
        grid_op.append(grid_el[i].varValue)
        wind_op.append(wind_el[i].varValue)
        pv_op.append(pv_el[i].varValue)
        storage.append(h2st[i].varValue)
        h2_produced.append(h2_prod[i].varValue)
        h2_used.append(h2_use[i].varValue)
        e_standby.append(elz_standby[i].varValue)
        e_start.append(elz_start[i].varValue)
        e_off.append(elz_off[i].varValue)
        e_on.append(elz_mode[i].varValue)
        m_on.append(meth_on[i].varValue)
        m_start.append(meth_start[i].varValue)
        m_stop.append(meth_stop[i].varValue)
        m_loss.append(meth_loss[i].varValue)

        # #PL
        # E10.append(e10[i].varValue)
        # E20.append(e20[i].varValue)
        # E30.append(e30[i].varValue)
        # E40.append(e40[i].varValue)
        # E50.append(e50[i].varValue)
        # E60.append(e60[i].varValue)
        # E70.append(e70[i].varValue)
        # E80.append(e80[i].varValue)
        # E90.append(e90[i].varValue)
        # E100.append(e100[i].varValue)
        # L10.append(lambda10[i].varValue)
        # L20.append(lambda20[i].varValue)
        # L30.append(lambda30[i].varValue)
        # L40.append(lambda40[i].varValue)
        # L50.append(lambda50[i].varValue)
        # L60.append(lambda60[i].varValue)
        # L70.append(lambda70[i].varValue)
        # L80.append(lambda80[i].varValue)
        # L90.append(lambda90[i].varValue)
        # L100.append(lambda100[i].varValue)

    h2_missed = list(np.array(hr_h2_demand) - np.array(h2_used))
    demand_vector = hr_h2_demand

    op = pd.DataFrame(
        {'Electrolyzer operation': e_op,
         'Grid purchases': grid_op,
         'Wind power': wind_op,
         'PV power': pv_op,
         'Electricity price': grid,
         'Storage': storage,
         'Demand': demand_vector,
         'H2 prod': h2_produced,
         'H2 used': h2_used,
         # 'Efficiency': eff,
         'Unmet demand': h2_missed,
         'Status': status,
         'On': e_on,
         'Standby': e_standby,
         'Off': e_off,
         'Cold start': e_start,
         'Meth on': m_on,
         'Meth start': m_start,
         'Meth stop': m_stop,
         'Meth loss': m_loss,
         # 'E10': E10,
         # 'E20': E20,
         # 'E30': E30,
         # 'E40': E40,
         # 'E50': E50,
         # 'E60': E60,
         # 'E70': E70,
         # 'E80': E80,
         # 'E90': E90,
         # 'E100': E100,
         # 'L10': L10,
         # 'L20': L20,
         # 'L30': L30,
         # 'L40': L40,
         # 'L50': L50,
         # 'L60': L60,
         # 'L70': L70,
         # 'L80': L80,
         # 'L90': L90,
         # 'L100': L100,
         })
    return op


def p2g_wwtp(
    h2_demand, #[kg/h]
    heat_demand, #[kWh/h] hourly
    o2_demand, #[mol/h] hourly
    o2_power, #[kWh avoided/kg O2]
    k_values,
    m_values,
    grid,
    wind,
    pv,
    elz_max,  # kW
    elz_min,  # kW
    aux_cons,  # kW
    h2st_max,  # kg?
    h2st_prev,  # storage from previous day
    meth_max,  # maximum H2 use [kg/h]
    meth_min,  # minimum H2 use [kg/h]
    meth_el: float = 0.0, #electricity demand for methanation and biogas compression [kWh/molCO2]
    heat_value: list = [0.0,0.0,0.0,0.0], #[€/MWh] for spring, summer, autumn and winter respectively
    startup_cost: float = 0.1,  # fraction of rated power
    standby_cost: float = 0.02,  # fraction of rated power
    elz_startup_time: float = 0.0, # [hour] time until H2 production starts after a cold startup
    elz_heat_time: float = 0.0, # [hour] time until electrolyzer has reached operating temperature
    prev_mode: int = 1,  # on/standby/off in last hour of previous day
    elz_eff: float = 0.7,  # HHV
    wind_cost: float = -10,  # [€/MWh]
    pv_cost: float = -11,  # [€/MWh] (only to prioritize ahead of wind and negative prices)
    bat_cap: float = 0, # [kWh]
    bat_eff: float = 0.95, # battery round trip efficiency
    bat_prev: float = 0, # [kWh] previous day battery charge
    meth_spec_heat: float = 0.0, # [kWh/kgH2] methanation heat generation per kg H2 methanized
    usable_heat: float = 0.8, #usable heat fraction
) -> pd.DataFrame:
    """Returns electrolyzer dispatch based on both renewable and grid electricity
    optimized to minimize the cost of electricity for a fixed H2 demand.
    Also including a linearized part-load efficiency and on/standby/off modes for
    the electrolyzer as well as startup losses for electrolysis and methanation.
    Using linearization parameters as proposed in Wirtz et al. 2021"""
    # Now: lowest value in each interval has the optimal efficiency it seems
    # What about the start-up time? Can we ignore this since it is very short with PEM?
    # Should startup and standby costs be based on spot price or a constant price?
    # Methanation ignored since it is based on CO2 availability

    #Hourly hydrogen demand (from CO2 availability)    
    h2_demand = h2_demand.flatten().tolist()
    
    wind = list(wind)
    pv = list(pv)
    grid = list(grid)
    h2st_prev = h2st_prev * 2.02 / 1000  # mol to kg

    # Define variables
    elz = []
    grid_el = []
    wind_el = []
    pv_el = []
    h2st = []
    h2_prod = []
    elz_mode = []
    elz_start = []
    elz_stop = []
    elz_off = []
    elz_standby = []
    h2_use = []
    meth_on = []
    o2_prod = []
    o2_use = []
    elz_heat = []
    heat_use = []
    heat_income = []
    elz_heat_demand = []
    ehd = []
    bat = []
    bat_out = []
    standby_el = []
    # Part-load variables
    # Load
    e10 = []
    e20 = []
    e30 = []
    e40 = []
    e50 = []
    e60 = []
    e70 = []
    e80 = []
    e90 = []
    e100 = []
    # On/off
    lambda10 = []
    lambda20 = []
    lambda30 = []
    lambda40 = []
    lambda50 = []
    lambda60 = []
    lambda70 = []
    lambda80 = []
    lambda90 = []
    lambda100 = []

    # Linearization parameters (see "Linearized efficiency" Excel)
    # gradient
    k10 = k_values[0]
    k20 = k_values[1]
    k30 = k_values[2]
    k40 = k_values[3]
    k50 = k_values[4]
    k60 = k_values[5]
    k70 = k_values[6]
    k80 = k_values[7]
    k90 = k_values[8]
    k100 = k_values[9]
    # m-value
    m10 = m_values[0]
    m20 = m_values[1]
    m30 = m_values[2]
    m40 = m_values[3]
    m50 = m_values[4]
    m60 = m_values[5]
    m70 = m_values[6]
    m80 = m_values[7]
    m90 = m_values[8]
    m100 = m_values[9]

    for i in range(len(grid)):
        # Electrolyzer operation
        elz.append(plp.LpVariable("elz_{}".format(i), 0, elz_max))
        elz_mode.append(plp.LpVariable("elz_mode_{}".format(i), 0, 1, plp.LpInteger))
        elz_start.append(plp.LpVariable("elz_start_{}".format(i), 0, 1, plp.LpInteger))
        elz_stop.append(plp.LpVariable("elz_stop_{}".format(i), 0, 1, plp.LpInteger))
        elz_standby.append(plp.LpVariable("elz_standby_{}".format(i), 0, 1, plp.LpInteger))
        elz_off.append(plp.LpVariable("elz_off_{}".format(i), 0, 1, plp.LpInteger))
        # Purchased grid electricity
        grid_el.append(plp.LpVariable("grid_el_{}".format(i), 0, elz_max))
        # Wind-based operation
        wind_el.append(plp.LpVariable("wind_el_{}".format(i), 0, elz_max))
        # PV-based operation
        pv_el.append(plp.LpVariable("pv_el_{}".format(i), 0, elz_max))
        # Hydrogen storage
        h2st.append(plp.LpVariable("h2st_{}".format(i), 0, h2st_max))
        # Hydrogen production and utilization
        h2_prod.append(plp.LpVariable("h2_prod_{}".format(i), 0, None))
        h2_use.append(plp.LpVariable("h2_use_{}".format(i), 0, meth_max))
        # Methanation
        meth_on.append(plp.LpVariable("meth_on_{}".format(i), 0, 1, plp.LpInteger))
        # Oxygen production and utilization
        o2_prod.append(plp.LpVariable("o2_prod_{}".format(i), 0, None))
        o2_use.append(plp.LpVariable("o2_use_{}".format(i), 0, None))
        # Heat production and utilization
        elz_heat.append(plp.LpVariable("elz_heat_{}".format(i), 0, None))
        heat_use.append(plp.LpVariable("heat_use_{}".format(i), 0, None))
        heat_income.append(plp.LpVariable("heat_income_{}".format(i), 0, None))
        elz_heat_demand.append(plp.LpVariable("elz_heat_demand_{}".format(i), 0, None))
        ehd.append(plp.LpVariable("ehd_{}".format(i), 0, 1, plp.LpInteger))
        #Battery
        bat.append(plp.LpVariable("bat_{}".format(i), 0, bat_cap))
        bat_out.append(plp.LpVariable("bat_out_{}".format(i), 0, bat_cap))
        #Standby electricity consumption
        standby_el.append(plp.LpVariable("sys_el_{}".format(i), 0, None))
        
        # Part-load variables
        # Load
        e10.append(plp.LpVariable("e10_{}".format(i), 0, None))
        e20.append(plp.LpVariable("e20_{}".format(i), 0, None))
        e30.append(plp.LpVariable("e30_{}".format(i), 0, None))
        e40.append(plp.LpVariable("e40_{}".format(i), 0, None))
        e50.append(plp.LpVariable("e50_{}".format(i), 0, None))
        e60.append(plp.LpVariable("e60_{}".format(i), 0, None))
        e70.append(plp.LpVariable("e70_{}".format(i), 0, None))
        e80.append(plp.LpVariable("e80_{}".format(i), 0, None))
        e90.append(plp.LpVariable("e90_{}".format(i), 0, None))
        e100.append(plp.LpVariable("e100_{}".format(i), 0, None))
        # On/off
        lambda10.append(plp.LpVariable(
            "lambda10_{}".format(i), 0, 1, plp.LpInteger))
        lambda20.append(plp.LpVariable(
            "lambda20_{}".format(i), 0, 1, plp.LpInteger))
        lambda30.append(plp.LpVariable(
            "lambda30_{}".format(i), 0, 1, plp.LpInteger))
        lambda40.append(plp.LpVariable(
            "lambda40_{}".format(i), 0, 1, plp.LpInteger))
        lambda50.append(plp.LpVariable(
            "lambda50_{}".format(i), 0, 1, plp.LpInteger))
        lambda60.append(plp.LpVariable(
            "lambda60_{}".format(i), 0, 1, plp.LpInteger))
        lambda70.append(plp.LpVariable(
            "lambda70_{}".format(i), 0, 1, plp.LpInteger))
        lambda80.append(plp.LpVariable(
            "lambda80_{}".format(i), 0, 1, plp.LpInteger))
        lambda90.append(plp.LpVariable(
            "lambda90_{}".format(i), 0, 1, plp.LpInteger))
        lambda100.append(plp.LpVariable(
            "lambda100_{}".format(i), 0, 1, plp.LpInteger))

    # Problem definition
    prob = plp.LpProblem("Operational_cost_minimization", plp.LpMinimize)

    # Constraints
    for i in range(len(grid)):
        # defining electricity supply
        prob += wind_el[i] <= wind[i] #wind
        prob += pv_el[i] <= pv[i] #PV
        prob += elz[i] == wind_el[i] + pv_el[i] + grid_el[i] + bat_out[i] #electrolyzer
        # prob += standby_el[i] == elz_standby[i] * elz_max * standby_cost
        # prob += meth_el[i] == h2_use[i] * meth_el
        # prob += elz[i] == wind_el[i] + pv_el[i] + grid_el[i] + bat_out[i] - standby_el[i] - comp_el[i] #overall system electricity consumption
        # defining minimum electrolyzer load
        prob += elz[i] >= elz_min * elz_mode[i]
        # defining maximum electrolyzer load
        prob += elz[i] <= elz_max * elz_mode[i]
        # defining minimum methanation load
        prob += h2_use[i] >= meth_min * meth_on[i]
        # defining maximum methanation load
        prob += h2_use[i] <= meth_max * meth_on[i]

        # Part-load variable constraints
        prob += e10[i] <= elz_max * 0.1 * lambda10[i]
        prob += e10[i] >= elz_max * 0 * lambda10[i]
        prob += e20[i] <= elz_max * 0.2 * lambda20[i]
        prob += e20[i] >= elz_max * 0.1 * lambda20[i]
        prob += e30[i] <= elz_max * 0.3 * lambda30[i]
        prob += e30[i] >= elz_max * 0.2 * lambda30[i]
        prob += e40[i] <= elz_max * 0.4 * lambda40[i]
        prob += e40[i] >= elz_max * 0.3 * lambda40[i]
        prob += e50[i] <= elz_max * 0.5 * lambda50[i]
        prob += e50[i] >= elz_max * 0.4 * lambda50[i]
        prob += e60[i] <= elz_max * 0.6 * lambda60[i]
        prob += e60[i] >= elz_max * 0.5 * lambda60[i]
        prob += e70[i] <= elz_max * 0.7 * lambda70[i]
        prob += e70[i] >= elz_max * 0.6 * lambda70[i]
        prob += e80[i] <= elz_max * 0.8 * lambda80[i]
        prob += e80[i] >= elz_max * 0.7 * lambda80[i]
        prob += e90[i] <= elz_max * 0.9 * lambda90[i]
        prob += e90[i] >= elz_max * 0.8 * lambda90[i]
        prob += e100[i] <= elz_max * 1 * lambda100[i]
        prob += e100[i] >= elz_max * 0.9 * lambda100[i]
        # Total electrolyzer load
        prob += elz[i] == e10[i] + e20[i] + e30[i] + e40[i] + \
            e50[i] + e60[i] + e70[i] + e80[i] + e90[i] + e100[i]
        # Only one part of linearization simultaneously (if not working, test to add an e0 varaible?)
        prob += lambda10[i] == elz_mode[i] - lambda20[i] - lambda30[i] - lambda40[i] - \
            lambda50[i] - lambda60[i] - lambda70[i] - \
            lambda80[i] - lambda90[i] - lambda100[i]

        # Linearized H2 production efficiency
        prob += h2_prod[i] == ((e10[i]*k10/elz_max) + (lambda10[i]*m10) + (e20[i]*k20/elz_max) + (lambda20[i]*m20) + (e30[i]*k30/elz_max) + (lambda30[i]*m30) +
                               (e40[i]*k40/elz_max) + (lambda40[i]*m40) + (e50[i]*k50/elz_max) + (lambda50[i]*m50) + (e60[i]*k60/elz_max) + (lambda60[i]*m60) +
                               (e70[i]*k70/elz_max) + (lambda70[i]*m70) + (e80[i]*k80/elz_max) + (lambda80[i]*m80) + (e90[i]*k90/elz_max) + (lambda90[i]*m90) + (e100[i]*k100/elz_max) + (lambda100[i]*m100))
        # To add a start-up time, multiply this sum by the assumed fraction of an hour that is lost

        # defining start-up (Baumhof a good source for this)
        if i > 0:
            # Electrolyzer
            prob += elz_start[i] == elz_mode[i] - elz_mode[i-1] - elz_standby[i-1] + elz_stop[i]
            # Can't go to standby from off
            prob += elz_standby[i] <= 1 - elz_off[i-1]
        else:  # considering end of previous day
            if prev_mode == 1:
                prob += elz_start[i] == 0
            elif prev_mode == 0:
                prob += elz_start[i] == elz_mode[i]

        # Electrolyzer modes (Baumhof)
        prob += elz_mode[i] + elz_standby[i] + elz_off[i] == 1
        prob += elz_start[i] + elz_stop[i] <= 1
        
        # Can't exceed hourly CO2 availability
        prob += h2_use[i] <= h2_demand[i]

        # Storage charging/discharging
        if i == 0:  # using previous day storage value for first hour
            #Hydrogen    
            prob += h2st[i] == h2_prod[i] - h2_use[i] + h2st_prev
            #Battery (currently not using grid electricity)
            prob += bat[i] <= ((wind[i] + pv[i] - wind_el[i] - pv_el[i]) * bat_eff) - bat_out[i] + bat_prev
        else:
            #Hydrogen
            prob += h2st[i] == h2_prod[i] - h2_use[i] + h2st[i-1]
            #Battery
            prob += bat[i] <= ((wind[i] + pv[i] - wind_el[i] - pv_el[i]) * bat_eff) - bat_out[i] + bat[i-1]
        
        # By-product generation
        #Oxygen
        prob += o2_prod[i] == h2_prod[i] * (32/2.02) * 0.5  # kg of o2 produced
        prob += o2_use[i] <= o2_prod[i]
        prob += o2_use[i] <= o2_demand[i]
        
        #Heat
        prob += elz_heat[i] == elz[i] - (aux_cons*elz_mode[i]) - (h2_prod[i]*39.4)
        #Electrolyzer heat demand, remove methanation heat (already known from h2_use) (maximum of this heat and zero)
        prob += 100000 * ehd[i] >= heat_demand[i] - (h2_use[i] * meth_spec_heat * usable_heat)
        prob += 100000 * (1 - ehd[i]) >= (h2_use[i] * meth_spec_heat * usable_heat) - heat_demand[i]
        prob += elz_heat_demand[i] <= heat_demand[i] - (h2_use[i] * meth_spec_heat * usable_heat) + (100000 * (1 - ehd[i])) 
        prob += elz_heat_demand[i] <= 0 + 100000 * ehd[i]
        #No heat during cold start
        prob += elz_heat[i] <= (1-elz_start[i]) * 1000000000
        prob += elz_heat[i] >= 0
        prob += elz_heat[i] <= elz[i] - (aux_cons*elz_mode[i]) - (h2_prod[i]*39.4)
        prob += elz_heat[i] >= (elz[i] - (aux_cons*elz_mode[i]) - (h2_prod[i]*39.4)) - (elz_start[i]*1000000000)  
        #Use limited by both production and demand
        prob += heat_use[i] <= usable_heat * elz_heat[i]
        prob += heat_use[i] <= elz_heat_demand[i]
        #Seasonal district heat cost
        if (i < 1416) or (i >= 8016):
            prob += heat_income[i] == heat_use[i] * heat_value[3] / 1000
        elif (i >= 1416) and (i < 3624):
            prob += heat_income[i] == heat_use[i] * heat_value[0] / 1000
        elif (i >= 3624) and (i < 5832):
            prob += heat_income[i] == heat_use[i] * heat_value[1] / 1000
        elif (i >= 5832) and (i < 8016):
            prob += heat_income[i] == heat_use[i] * heat_value[2] / 1000
    
        # Start-up time-related losses (either big M here or include in elz start-up cost)
                  
    
    # objective (minimize the electricity cost)
    prob += plp.lpSum([grid_el[i] * grid[i] for i in range(len(grid))]) + plp.lpSum([wind_el[i] * wind_cost for i in range(len(grid))]) + plp.lpSum([pv_el[i] * pv_cost for i in range(len(grid))]) + \
        ((plp.lpSum([h2_demand[i] - h2_use[i] for i in range(len(grid))]))*10000000) + plp.lpSum([elz_standby[i] * elz_max*standby_cost*grid[i] for i in range(len(grid))]) + \
        plp.lpSum([elz_start[i] * elz_max*startup_cost*grid[i] for i in range(len(grid))]) - plp.lpSum([o2_use[i] * (o2_power * grid[i] / 1000) for i in range(len(grid))]) \
        - plp.lpSum([heat_income[i]*1 for i in range(len(grid))])
    #electricity costs (grid, wind, pv)
    #missed demand "cost" and electrolyzer standby cost
    #start-up cost for elz, oxygen income,
    #heat income
    
    # TEST AND COMPARE TO WITHOUT COLD START!
    
    # check solution
    solver = plp.GUROBI_CMD()
    status = prob.solve(solver)

    e_op = []
    grid_op = []
    wind_op = []
    pv_op = []
    storage = []
    h2_produced = []
    h2_used = []
    e_on = []
    e_standby = []
    e_start = []
    e_off = []
    o_prod = []
    o_use = []
    h_prod = []
    h_use = []
    h_inc = []
    bat_state = []
    bat_discharge = []

    # saving variable solutions
    for i in range(len(grid)):
        e_op.append(elz[i].varValue)
        grid_op.append(grid_el[i].varValue)
        wind_op.append(wind_el[i].varValue)
        pv_op.append(pv_el[i].varValue)
        storage.append(h2st[i].varValue)
        h2_produced.append(h2_prod[i].varValue)
        h2_used.append(h2_use[i].varValue)
        e_standby.append(elz_standby[i].varValue)
        e_start.append(elz_start[i].varValue)
        e_off.append(elz_off[i].varValue)
        e_on.append(elz_mode[i].varValue)
        o_prod.append(o2_prod[i].varValue)
        o_use.append(o2_use[i].varValue)
        h_prod.append(elz_heat[i].varValue)
        h_use.append(heat_use[i].varValue)
        h_inc.append(heat_income[i].varValue)
        bat_state.append(bat[i].varValue)
        bat_discharge.append(bat_out[i].varValue)
        

    h2_missed = list(np.array(h2_demand) - np.array(h2_used))
    demand_vector = h2_demand
    grid_inc = np.array(grid_op) * grid / 1000
    o2_inc = np.array(o_use) * (o2_power * grid[i] / 1000)

    op = pd.DataFrame(
        {'Electrolyzer operation': e_op,
         'Grid purchases': grid_op,
         'Wind power': wind_op,
         'PV power': pv_op,
         'Electricity price': grid,
         'Storage': storage,
         'Demand': demand_vector,
         'H2 prod': h2_produced,
         'H2 used': h2_used,
         # 'Efficiency': eff,
         'Unmet demand': h2_missed,
         'Status': status,
         'On': e_on,
         'Standby': e_standby,
         'Off': e_off,
         'Cold start': e_start,
         'O2 prod': o_prod,
         'O2 use': o_use,
         'Heat prod': h_prod,
         'Heat use': h_use,
         'Battery state': bat_state,
         'Battery discharging': bat_discharge,
         'Heat income': h_inc,
         'Grid income': grid_inc,
         'Oxygen income': o2_inc,
         })
    return op


def p2g_wwtp2(
    h2_demand, #[kg/h]
    heat_demand, #[kWh/h] hourly
    o2_demand, #[mol/h] hourly
    o2_power, #[kWh avoided/kg O2]
    biogas, #[mol/h] CH4 and CO2 flow
    k_values,
    m_values,
    grid,
    wind,
    pv,
    elz_max,  # kW
    elz_min,  # kW
    aux_cons,  # kW
    h2st_max,  # kg?
    h2st_prev,  # storage from previous day
    meth_max,  # maximum H2 use [kg/h]
    meth_min,  # minimum H2 use [kg/h]
    meth_el_factor: float = 0.0, #[kWh/molCO2 converted]
    comp_el_factor: float = 0.0, #[kWh/mol compressed gas]
    meth_el: float = 0.0, #electricity demand for methanation and biogas compression [kWh/molCO2]
    heat_value: list = [0.0,0.0,0.0,0.0], #[€/MWh] for spring, summer, autumn and winter respectively
    startup_cost: float = 0.1,  # fraction of rated power
    standby_cost: float = 0.02,  # fraction of rated power
    elz_startup_time: float = 0.0, # [hour] time until H2 production starts after a cold startup
    elz_heat_time: float = 0.0, # [hour] time until electrolyzer has reached operating temperature
    prev_mode: int = 1,  # on/standby/off in last hour of previous day
    elz_eff: float = 0.7,  # HHV
    wind_cost: float = -10,  # [€/MWh]
    pv_cost: float = -11,  # [€/MWh] (only to prioritize ahead of wind and negative prices)
    bat_cap: float = 0, # [kWh]
    bat_eff: float = 0.95, # battery round trip efficiency
    bat_prev: float = 0, # [kWh] previous day battery charge
    meth_spec_heat: float = 0.0, # [kWh/kgH2] methanation heat generation per kg H2 methanized
    usable_heat: float = 0.8, #usable heat fraction
    h2o_cons: float = 10.0, #water consumption [lH2O/kgH2]
    temp: float = 80.0,
    h2o_temp: float = 15.0,
) -> pd.DataFrame:
    """Returns electrolyzer dispatch based on both renewable and grid electricity
    optimized to minimize the cost of electricity for a fixed H2 demand.
    Also including a linearized part-load efficiency and on/standby/off modes for
    the electrolyzer as well as startup losses for electrolysis and methanation.
    Using linearization parameters as proposed in Wirtz et al. 2021"""
    # Now: lowest value in each interval has the optimal efficiency it seems
    # What about the start-up time? Can we ignore this since it is very short with PEM?
    # Should startup and standby costs be based on spot price or a constant price?
    # Methanation ignored since it is based on CO2 availability

    #Hourly hydrogen demand (from CO2 availability)    
    h2_demand = h2_demand.flatten().tolist()
    
    wind = list(wind)
    pv = list(pv)
    grid = list(grid)
    h2st_prev = h2st_prev * 2.02 / 1000  # mol to kg

    # Define variables
    elz = []
    grid_el = []
    wind_el = []
    pv_el = []
    h2st = []
    h2_prod = []
    h2_prod_start = []
    h2_prod_real = []
    elz_mode = []
    elz_start = []
    elz_stop = []
    elz_off = []
    elz_standby = []
    h2_use = []
    meth_on = []
    o2_prod = []
    o2_use = []
    elz_heat = []
    heat_use = []
    heat_income = []
    elz_heat_demand = []
    ehd = []
    bat = []
    bat_out = []
    standby_el = []
    meth_el1 = []
    comp_el = []
    # elz1 = []
    # Part-load variables
    # Load
    e10 = []
    e20 = []
    e30 = []
    e40 = []
    e50 = []
    e60 = []
    e70 = []
    e80 = []
    e90 = []
    e100 = []
    # On/off
    lambda10 = []
    lambda20 = []
    lambda30 = []
    lambda40 = []
    lambda50 = []
    lambda60 = []
    lambda70 = []
    lambda80 = []
    lambda90 = []
    lambda100 = []

    # Linearization parameters (see "Linearized efficiency" Excel)
    # gradient
    k10 = k_values[0]
    k20 = k_values[1]
    k30 = k_values[2]
    k40 = k_values[3]
    k50 = k_values[4]
    k60 = k_values[5]
    k70 = k_values[6]
    k80 = k_values[7]
    k90 = k_values[8]
    k100 = k_values[9]
    # m-value
    m10 = m_values[0]
    m20 = m_values[1]
    m30 = m_values[2]
    m40 = m_values[3]
    m50 = m_values[4]
    m60 = m_values[5]
    m70 = m_values[6]
    m80 = m_values[7]
    m90 = m_values[8]
    m100 = m_values[9]

    for i in range(len(grid)):
        # Electrolyzer operation
        elz.append(plp.LpVariable("elz_{}".format(i), 0, elz_max))
        # elz1.append(plp.LpVariable("elz1_{}".format(i), 0, None)) #an independent copy
        elz_mode.append(plp.LpVariable("elz_mode_{}".format(i), 0, 1, plp.LpInteger))
        elz_start.append(plp.LpVariable("elz_start_{}".format(i), 0, 1, plp.LpInteger))
        elz_stop.append(plp.LpVariable("elz_stop_{}".format(i), 0, 1, plp.LpInteger))
        elz_standby.append(plp.LpVariable("elz_standby_{}".format(i), 0, 1, plp.LpInteger))
        elz_off.append(plp.LpVariable("elz_off_{}".format(i), 0, 1, plp.LpInteger))
        # Purchased grid electricity
        grid_el.append(plp.LpVariable("grid_el_{}".format(i), 0, None))
        # Wind-based operation
        wind_el.append(plp.LpVariable("wind_el_{}".format(i), 0, None))
        # PV-based operation
        pv_el.append(plp.LpVariable("pv_el_{}".format(i), 0, None))
        # Hydrogen storage
        h2st.append(plp.LpVariable("h2st_{}".format(i), 0, h2st_max))
        # Hydrogen production and utilization
        h2_prod.append(plp.LpVariable("h2_prod_{}".format(i), 0, None))
        h2_prod_start.append(plp.LpVariable("h2_prod_start_{}".format(i), 0, None))
        h2_prod_real.append(plp.LpVariable("h2_prod_real_{}".format(i), 0, None))
        h2_use.append(plp.LpVariable("h2_use_{}".format(i), 0, meth_max))
        # Methanation
        meth_on.append(plp.LpVariable("meth_on_{}".format(i), 0, 1, plp.LpInteger))
        meth_el1.append(plp.LpVariable("meth_el_{}".format(i), 0, None))
        comp_el.append(plp.LpVariable("comp_el_{}".format(i), 0, None))
        # Oxygen production and utilization
        o2_prod.append(plp.LpVariable("o2_prod_{}".format(i), 0, None))
        o2_use.append(plp.LpVariable("o2_use_{}".format(i), 0, None))
        # Heat production and utilization
        elz_heat.append(plp.LpVariable("elz_heat_{}".format(i), 0, None))
        heat_use.append(plp.LpVariable("heat_use_{}".format(i), 0, None))
        heat_income.append(plp.LpVariable("heat_income_{}".format(i), 0, None))
        elz_heat_demand.append(plp.LpVariable("elz_heat_demand_{}".format(i), 0, None))
        ehd.append(plp.LpVariable("ehd_{}".format(i), 0, 1, plp.LpInteger))
        #Battery
        bat.append(plp.LpVariable("bat_{}".format(i), 0, bat_cap))
        bat_out.append(plp.LpVariable("bat_out_{}".format(i), 0, bat_cap))
        #Standby electricity consumption
        standby_el.append(plp.LpVariable("sys_el_{}".format(i), 0, None))
        
        # Part-load variables
        # Load
        e10.append(plp.LpVariable("e10_{}".format(i), 0, None))
        e20.append(plp.LpVariable("e20_{}".format(i), 0, None))
        e30.append(plp.LpVariable("e30_{}".format(i), 0, None))
        e40.append(plp.LpVariable("e40_{}".format(i), 0, None))
        e50.append(plp.LpVariable("e50_{}".format(i), 0, None))
        e60.append(plp.LpVariable("e60_{}".format(i), 0, None))
        e70.append(plp.LpVariable("e70_{}".format(i), 0, None))
        e80.append(plp.LpVariable("e80_{}".format(i), 0, None))
        e90.append(plp.LpVariable("e90_{}".format(i), 0, None))
        e100.append(plp.LpVariable("e100_{}".format(i), 0, None))
        # On/off
        lambda10.append(plp.LpVariable(
            "lambda10_{}".format(i), 0, 1, plp.LpInteger))
        lambda20.append(plp.LpVariable(
            "lambda20_{}".format(i), 0, 1, plp.LpInteger))
        lambda30.append(plp.LpVariable(
            "lambda30_{}".format(i), 0, 1, plp.LpInteger))
        lambda40.append(plp.LpVariable(
            "lambda40_{}".format(i), 0, 1, plp.LpInteger))
        lambda50.append(plp.LpVariable(
            "lambda50_{}".format(i), 0, 1, plp.LpInteger))
        lambda60.append(plp.LpVariable(
            "lambda60_{}".format(i), 0, 1, plp.LpInteger))
        lambda70.append(plp.LpVariable(
            "lambda70_{}".format(i), 0, 1, plp.LpInteger))
        lambda80.append(plp.LpVariable(
            "lambda80_{}".format(i), 0, 1, plp.LpInteger))
        lambda90.append(plp.LpVariable(
            "lambda90_{}".format(i), 0, 1, plp.LpInteger))
        lambda100.append(plp.LpVariable(
            "lambda100_{}".format(i), 0, 1, plp.LpInteger))

    # Problem definition
    prob = plp.LpProblem("Operational_cost_minimization", plp.LpMinimize)

    # Constraints
    for i in range(len(grid)):
        # defining electricity supply
        prob += wind_el[i] <= wind[i] #wind
        prob += pv_el[i] <= pv[i] #PV
        # prob += elz[i] == wind_el[i] + pv_el[i] + grid_el[i] + bat_out[i] #electrolyzer
        prob += standby_el[i] == elz_standby[i] * elz_max * standby_cost
        prob += elz[i] == wind_el[i] + pv_el[i] + grid_el[i] + bat_out[i] - standby_el[i] - meth_el1[i] - comp_el[i] #overall system electricity consumption
        prob += meth_el1[i] == h2_use[i] * meth_el_factor #electricity required for methanation
        if biogas[i,1] > 0:
            prob += comp_el[i] == (((0.25*h2_use[i]*1000/2.02) / biogas[i,1]) * (biogas[i,0] + biogas[i,1])) * comp_el_factor #electricity required for biogas compression
        elif biogas[i,1] == 0:
            prob += comp_el[i] == 0
        # defining minimum electrolyzer load
        prob += elz[i] >= elz_min * elz_mode[i]
        # defining maximum electrolyzer load
        prob += elz[i] <= elz_max * elz_mode[i]
        # defining minimum methanation load
        prob += h2_use[i] >= meth_min * meth_on[i]
        # defining maximum methanation load
        prob += h2_use[i] <= meth_max * meth_on[i]

        # Part-load variable constraints
        prob += e10[i] <= elz_max * 0.1 * lambda10[i]
        prob += e10[i] >= elz_max * 0 * lambda10[i]
        prob += e20[i] <= elz_max * 0.2 * lambda20[i]
        prob += e20[i] >= elz_max * 0.1 * lambda20[i]
        prob += e30[i] <= elz_max * 0.3 * lambda30[i]
        prob += e30[i] >= elz_max * 0.2 * lambda30[i]
        prob += e40[i] <= elz_max * 0.4 * lambda40[i]
        prob += e40[i] >= elz_max * 0.3 * lambda40[i]
        prob += e50[i] <= elz_max * 0.5 * lambda50[i]
        prob += e50[i] >= elz_max * 0.4 * lambda50[i]
        prob += e60[i] <= elz_max * 0.6 * lambda60[i]
        prob += e60[i] >= elz_max * 0.5 * lambda60[i]
        prob += e70[i] <= elz_max * 0.7 * lambda70[i]
        prob += e70[i] >= elz_max * 0.6 * lambda70[i]
        prob += e80[i] <= elz_max * 0.8 * lambda80[i]
        prob += e80[i] >= elz_max * 0.7 * lambda80[i]
        prob += e90[i] <= elz_max * 0.9 * lambda90[i]
        prob += e90[i] >= elz_max * 0.8 * lambda90[i]
        prob += e100[i] <= elz_max * 1 * lambda100[i]
        prob += e100[i] >= elz_max * 0.9 * lambda100[i]
        # Total electrolyzer load
        prob += elz[i] == e10[i] + e20[i] + e30[i] + e40[i] + \
            e50[i] + e60[i] + e70[i] + e80[i] + e90[i] + e100[i]
        # Only one part of linearization simultaneously (if not working, test to add an e0 varaible?)
        prob += lambda10[i] == elz_mode[i] - lambda20[i] - lambda30[i] - lambda40[i] - \
            lambda50[i] - lambda60[i] - lambda70[i] - \
            lambda80[i] - lambda90[i] - lambda100[i]

        # Linearized H2 production efficiency
        prob += h2_prod_real[i] == ((e10[i]*k10/elz_max) + (lambda10[i]*m10) + (e20[i]*k20/elz_max) + (lambda20[i]*m20) + (e30[i]*k30/elz_max) + (lambda30[i]*m30) +
                                (e40[i]*k40/elz_max) + (lambda40[i]*m40) + (e50[i]*k50/elz_max) + (lambda50[i]*m50) + (e60[i]*k60/elz_max) + (lambda60[i]*m60) +
                                (e70[i]*k70/elz_max) + (lambda70[i]*m70) + (e80[i]*k80/elz_max) + (lambda80[i]*m80) + (e90[i]*k90/elz_max) + (lambda90[i]*m90) + (e100[i]*k100/elz_max) + (lambda100[i]*m100))
        # prob += h2_prod[i] == ((e10[i]*k10/elz_max) + (lambda10[i]*m10) + (e20[i]*k20/elz_max) + (lambda20[i]*m20) + (e30[i]*k30/elz_max) + (lambda30[i]*m30) +
        #                         (e40[i]*k40/elz_max) + (lambda40[i]*m40) + (e50[i]*k50/elz_max) + (lambda50[i]*m50) + (e60[i]*k60/elz_max) + (lambda60[i]*m60) +
        #                         (e70[i]*k70/elz_max) + (lambda70[i]*m70) + (e80[i]*k80/elz_max) + (lambda80[i]*m80) + (e90[i]*k90/elz_max) + (lambda90[i]*m90) + (e100[i]*k100/elz_max) + (lambda100[i]*m100))
        
        # To add a start-up time, uncomment this and use other option above
        prob += h2_prod_start[i] == h2_prod_real[i] * (1-elz_startup_time)
        prob += h2_prod[i] >= h2_prod_start[i] - (100000000*(1-elz_start[i]))
        prob += h2_prod[i] <= h2_prod_start[i] + (100000000*(1-elz_start[i]))
        prob += h2_prod[i] >= h2_prod_real[i] - (100000000*elz_start[i])
        prob += h2_prod[i] <= h2_prod_real[i] + (100000000*elz_start[i])
        
        # defining start-up (Baumhof a good source for this)
        if i > 0:
            # Electrolyzer
            prob += elz_start[i] >= elz_mode[i] - elz_mode[i-1] - elz_standby[i-1]
            # Can't start if on or standby during previous hour or on or standby in current hour
            # prob += elz_start[i] <= 1 - elz_mode[i-1] - elz_standby[i-1]
            # prob += elz_start[i] <= 1 - elz_off[i] - elz_standby[i]
            # Can't start without being on
            # prob += elz_start[i] <= elz_mode[i]
            #CANNOT GET STARTS AND STOPS TO WORK
                #BEFORE I HAD SIMPLY ADDED "+ELZ_STOP[I]" TO TOP IN THIS LOOP AND DID NOT HAVE THE ONE JUST ABOVE.
            # Can't go to standby from off or vice versa
            prob += elz_standby[i] <= 1 - elz_off[i-1]
            # prob += elz_off[i] <= 1 - elz_standby[i-1]
        else:  # considering end of previous day
            if prev_mode == 1:
                prob += elz_start[i] == 0
            elif prev_mode == 0:
                prob += elz_start[i] == elz_mode[i]
                prob += elz_standby[i] == 0

        # Electrolyzer modes (Baumhof)
        prob += elz_mode[i] + elz_standby[i] + elz_off[i] == 1
        # prob += elz_start[i] + elz_stop[i] <= 1
        
        # Can't exceed hourly CO2 availability
        prob += h2_use[i] <= h2_demand[i]

        # Storage charging/discharging
        if i == 0:  # using previous day storage value for first hour
            #Hydrogen    
            prob += h2st[i] == h2_prod[i] - h2_use[i] + h2st_prev
            #Battery (currently not using grid electricity)
            prob += bat[i] <= ((wind[i] + pv[i] - wind_el[i] - pv_el[i]) * bat_eff) - bat_out[i] + bat_prev
        else:
            #Hydrogen
            prob += h2st[i] == h2_prod[i] - h2_use[i] + h2st[i-1]
            #Battery
            prob += bat[i] <= ((wind[i] + pv[i] - wind_el[i] - pv_el[i]) * bat_eff) - bat_out[i] + bat[i-1]
        
        # By-product generation
        #Oxygen
        prob += o2_prod[i] == h2_prod[i] * (32/2.02) * 0.5  # kg of o2 produced
        prob += o2_use[i] <= o2_prod[i]
        prob += o2_use[i] <= o2_demand[i]
        
        #Heat
        # prob += elz_heat[i] == elz[i] - (aux_cons*elz_mode[i]) - (h2_prod[i]*39.4)
        #Electrolyzer heat demand, remove methanation heat (already known from h2_use) (maximum of this heat and zero)
        prob += 100000 * ehd[i] >= heat_demand[i] - (h2_use[i] * meth_spec_heat * usable_heat)
        prob += 100000 * (1 - ehd[i]) >= (h2_use[i] * meth_spec_heat * usable_heat) - heat_demand[i]
        prob += elz_heat_demand[i] <= heat_demand[i] - (h2_use[i] * meth_spec_heat * usable_heat) + (100000 * (1 - ehd[i])) 
        prob += elz_heat_demand[i] <= 0 + 100000 * ehd[i]
        #Input water heat consumption
        # h2o = ((h2o_cons * h2_prod[i] * 997 / 1000) * (4.18/3600) * (temp - h2o_temp)) #[kg]*[kWh/kg*K]*[K]
        #No heat during cold start
        prob += elz_heat[i] <= (1-elz_start[i]) * 1000000000
        prob += elz_heat[i] >= 0
        prob += elz_heat[i] <= elz[i] - (aux_cons*elz_mode[i]) - (h2_prod[i]*39.4) - ((h2o_cons * h2_prod[i] * 997 / 1000) * (4.18/3600) * (temp - h2o_temp))
        prob += elz_heat[i] >= (elz[i] - (aux_cons*elz_mode[i]) - (h2_prod[i]*39.4)) - (elz_start[i]*1000000000) - ((h2o_cons * h2_prod[i] * 997 / 1000) * (4.18/3600) * (temp - h2o_temp))  
        #Use limited by both production and demand
        prob += heat_use[i] <= usable_heat * elz_heat[i]
        prob += heat_use[i] <= elz_heat_demand[i]
        prob += heat_use[i] >= 0

        #Seasonal district heat cost
        if (i < 1416) or (i >= 8016):
            prob += heat_income[i] == heat_use[i] * heat_value[3] / 1000
        elif (i >= 1416) and (i < 3624):
            prob += heat_income[i] == heat_use[i] * heat_value[0] / 1000
        elif (i >= 3624) and (i < 5832):
            prob += heat_income[i] == heat_use[i] * heat_value[1] / 1000
        elif (i >= 5832) and (i < 8016):
            prob += heat_income[i] == heat_use[i] * heat_value[2] / 1000
    
    
    # objective (minimize the electricity cost)
    prob += plp.lpSum([grid_el[i] * grid[i] for i in range(len(grid))]) + plp.lpSum([wind_el[i] * wind_cost for i in range(len(grid))]) + plp.lpSum([pv_el[i] * pv_cost for i in range(len(grid))]) - \
            plp.lpSum([o2_use[i] * (o2_power * grid[i] / 1000) for i in range(len(grid))]) - plp.lpSum([heat_income[i]*1 for i in range(len(grid))]) + \
                ((plp.lpSum([h2_demand[i] - h2_use[i] for i in range(len(grid))]))*10000000)# + \
                    # plp.lpSum([elz_start[i] * elz_max*startup_cost*grid[i] for i in range(len(grid))])# + plp.lpSum([elz_standby[i] * elz_max*standby_cost*grid[i] for i in range(len(grid))])

    #electricity costs (grid, wind, pv)
    #electrolyzer standby cost and start-up cost
    #oxygen income, and heat income
        
    # check solution
    solver = plp.GUROBI_CMD()
    status = prob.solve(solver)

    e_op = []
    grid_op = []
    wind_op = []
    pv_op = []
    storage = []
    h2_produced = []
    h2_used = []
    e_on = []
    e_standby = []
    e_start = []
    e_off = []
    o_prod = []
    o_use = []
    h_prod = []
    h_use = []
    h_inc = []
    bat_state = []
    bat_discharge = []
    sys_op = []
    sb_el = []
    m_el = []
    c_el = []
    e_h_dem = []
    h2_produced_start = []
    h2_produced_real = []

    

    # saving variable solutions
    for i in range(len(grid)):
        e_op.append(elz[i].varValue)
        grid_op.append(grid_el[i].varValue)
        wind_op.append(wind_el[i].varValue)
        pv_op.append(pv_el[i].varValue)
        storage.append(h2st[i].varValue)
        h2_produced.append(h2_prod[i].varValue)
        h2_used.append(h2_use[i].varValue)
        e_standby.append(elz_standby[i].varValue)
        e_start.append(elz_start[i].varValue)
        e_off.append(elz_off[i].varValue)
        e_on.append(elz_mode[i].varValue)
        o_prod.append(o2_prod[i].varValue)
        o_use.append(o2_use[i].varValue)
        h_prod.append(elz_heat[i].varValue)
        h_use.append(heat_use[i].varValue)
        h_inc.append(heat_income[i].varValue)
        bat_state.append(bat[i].varValue)
        bat_discharge.append(bat_out[i].varValue)
        sys_op.append(elz[i].varValue + standby_el[i].varValue + meth_el1[i].varValue + comp_el[i].varValue)
        sb_el.append(standby_el[i].varValue)
        m_el.append(meth_el1[i].varValue)
        c_el.append(comp_el[i].varValue)
        e_h_dem.append(elz_heat_demand[i].varValue)
        h2_produced_start.append(h2_prod_start[i].varValue)
        h2_produced_real.append(h2_prod_real[i].varValue)
        

    h2_missed = list(np.array(h2_demand) - np.array(h2_used))
    demand_vector = h2_demand
    grid_inc = np.array(grid_op) * grid / 1000
    o2_inc = np.array(o_use) * (o2_power * grid[i] / 1000)

    op = pd.DataFrame(
        {'Electrolyzer operation': e_op,
         'Grid purchases': grid_op,
         'Wind power': wind_op,
         'PV power': pv_op,
         'Electricity price': grid,
         'Storage': storage,
         'Demand': demand_vector,
         'H2 prod': h2_produced,
         'H2 used': h2_used,
         # 'Efficiency': eff,
         'Unmet demand': h2_missed,
         'Status': status,
         'On': e_on,
         'Standby': e_standby,
         'Off': e_off,
         'Cold start': e_start,
         'O2 prod': o_prod,
         'O2 use': o_use,
         'Heat prod': h_prod,
         'Heat use': h_use,
         'Battery state': bat_state,
         'Battery discharging': bat_discharge,
         'Heat income': h_inc,
         'Grid income': grid_inc,
         'Oxygen income': o2_inc,
          'System operation': sys_op,
          'Standby electricity': sb_el,
          'Methanation electricity': m_el,
          'Compression electricity': c_el,
         'Heat demand': e_h_dem,
         'H2 prod start': h2_produced_start,
         'H2 prod real': h2_produced_real,
         })
    return op


def p2g_wwtp2_res(
    h2_demand, #[kg/h]
    heat_demand, #[kWh/h] hourly
    o2_demand, #[mol/h] hourly
    o2_power, #[kWh avoided/kg O2]
    biogas, #[mol/h] CH4 and CO2 flow
    k_values,
    m_values,
    grid,
    res,
    elz_max,  # kW
    elz_min,  # kW
    aux_cons,  # kW
    h2st_max,  # kg?
    h2st_prev,  # storage from previous day
    meth_max,  # maximum H2 use [kg/h]
    meth_min,  # minimum H2 use [kg/h]
    meth_el_factor: float = 0.0, #[kWh/molCO2 converted]
    comp_el_factor: float = 0.0, #[kWh/mol compressed gas]
    meth_el: float = 0.0, #electricity demand for methanation and biogas compression [kWh/molCO2]
    heat_value: list = [0.0,0.0,0.0,0.0], #[€/MWh] for spring, summer, autumn and winter respectively
    startup_cost: float = 0.1,  # fraction of rated power
    standby_cost: float = 0.02,  # fraction of rated power
    elz_startup_time: float = 0.0, # [hour] time until H2 production starts after a cold startup
    elz_heat_time: float = 0.0, # [hour] time until electrolyzer has reached operating temperature
    prev_mode: int = 1,  # on/standby/off in last hour of previous day
    elz_eff: float = 0.7,  # HHV
    res_cost: float = 0,  # [€/MWh]
    bat_cap: float = 0, # [kWh]
    bat_eff: float = 0.95, # battery round trip efficiency
    bat_prev: float = 0, # [kWh] previous day battery charge
    meth_spec_heat: float = 0.0, # [kWh/kgH2] methanation heat generation per kg H2 methanized
    usable_heat: float = 0.8, #usable heat fraction
    h2o_cons: float = 10.0, #water consumption [lH2O/kgH2]
    temp: float = 80.0,
    h2o_temp: float = 15.0,
) -> pd.DataFrame:
    """Based on p2g_wwtp2 but with only excess local renewables"""
    # Now: lowest value in each interval has the optimal efficiency it seems
    # What about the start-up time? Can we ignore this since it is very short with PEM?
    # Should startup and standby costs be based on spot price or a constant price?
    # Methanation ignored since it is based on CO2 availability

    #Hourly hydrogen demand (from CO2 availability)    
    h2_demand = h2_demand.flatten().tolist()
    
    res = list(res)
    grid = list(grid)
    h2st_prev = h2st_prev * 2.02 / 1000  # mol to kg

    # Define variables
    elz = []
    grid_el = []
    res_el = []
    h2st = []
    h2_prod = []
    h2_prod_start = []
    h2_prod_real = []
    elz_mode = []
    elz_start = []
    elz_stop = []
    elz_off = []
    elz_standby = []
    h2_use = []
    meth_on = []
    o2_prod = []
    o2_use = []
    elz_heat = []
    heat_use = []
    heat_income = []
    elz_heat_demand = []
    ehd = []
    bat = []
    bat_out = []
    standby_el = []
    meth_el1 = []
    comp_el = []
    # elz1 = []
    # Part-load variables
    # Load
    e10 = []
    e20 = []
    e30 = []
    e40 = []
    e50 = []
    e60 = []
    e70 = []
    e80 = []
    e90 = []
    e100 = []
    # On/off
    lambda10 = []
    lambda20 = []
    lambda30 = []
    lambda40 = []
    lambda50 = []
    lambda60 = []
    lambda70 = []
    lambda80 = []
    lambda90 = []
    lambda100 = []

    # Linearization parameters (see "Linearized efficiency" Excel)
    # gradient
    k10 = k_values[0]
    k20 = k_values[1]
    k30 = k_values[2]
    k40 = k_values[3]
    k50 = k_values[4]
    k60 = k_values[5]
    k70 = k_values[6]
    k80 = k_values[7]
    k90 = k_values[8]
    k100 = k_values[9]
    # m-value
    m10 = m_values[0]
    m20 = m_values[1]
    m30 = m_values[2]
    m40 = m_values[3]
    m50 = m_values[4]
    m60 = m_values[5]
    m70 = m_values[6]
    m80 = m_values[7]
    m90 = m_values[8]
    m100 = m_values[9]

    for i in range(len(grid)):
        # Electrolyzer operation
        elz.append(plp.LpVariable("elz_{}".format(i), 0, elz_max))
        # elz1.append(plp.LpVariable("elz1_{}".format(i), 0, None)) #an independent copy
        elz_mode.append(plp.LpVariable("elz_mode_{}".format(i), 0, 1, plp.LpInteger))
        elz_start.append(plp.LpVariable("elz_start_{}".format(i), 0, 1, plp.LpInteger))
        elz_stop.append(plp.LpVariable("elz_stop_{}".format(i), 0, 1, plp.LpInteger))
        elz_standby.append(plp.LpVariable("elz_standby_{}".format(i), 0, 1, plp.LpInteger))
        elz_off.append(plp.LpVariable("elz_off_{}".format(i), 0, 1, plp.LpInteger))
        # Purchased grid electricity
        grid_el.append(plp.LpVariable("grid_el_{}".format(i), 0, None))
        # RES-based operation
        res_el.append(plp.LpVariable("res_el_{}".format(i), 0, None))
        # Hydrogen storage
        h2st.append(plp.LpVariable("h2st_{}".format(i), 0, h2st_max))
        # Hydrogen production and utilization
        h2_prod.append(plp.LpVariable("h2_prod_{}".format(i), 0, None))
        h2_prod_start.append(plp.LpVariable("h2_prod_start_{}".format(i), 0, None))
        h2_prod_real.append(plp.LpVariable("h2_prod_real_{}".format(i), 0, None))
        h2_use.append(plp.LpVariable("h2_use_{}".format(i), 0, meth_max))
        # Methanation
        meth_on.append(plp.LpVariable("meth_on_{}".format(i), 0, 1, plp.LpInteger))
        meth_el1.append(plp.LpVariable("meth_el_{}".format(i), 0, None))
        comp_el.append(plp.LpVariable("comp_el_{}".format(i), 0, None))
        # Oxygen production and utilization
        o2_prod.append(plp.LpVariable("o2_prod_{}".format(i), 0, None))
        o2_use.append(plp.LpVariable("o2_use_{}".format(i), 0, None))
        # Heat production and utilization
        elz_heat.append(plp.LpVariable("elz_heat_{}".format(i), 0, None))
        heat_use.append(plp.LpVariable("heat_use_{}".format(i), 0, None))
        heat_income.append(plp.LpVariable("heat_income_{}".format(i), 0, None))
        elz_heat_demand.append(plp.LpVariable("elz_heat_demand_{}".format(i), 0, None))
        ehd.append(plp.LpVariable("ehd_{}".format(i), 0, 1, plp.LpInteger))
        #Battery
        bat.append(plp.LpVariable("bat_{}".format(i), 0, bat_cap))
        bat_out.append(plp.LpVariable("bat_out_{}".format(i), 0, bat_cap))
        #Standby electricity consumption
        standby_el.append(plp.LpVariable("sys_el_{}".format(i), 0, None))
        
        # Part-load variables
        # Load
        e10.append(plp.LpVariable("e10_{}".format(i), 0, None))
        e20.append(plp.LpVariable("e20_{}".format(i), 0, None))
        e30.append(plp.LpVariable("e30_{}".format(i), 0, None))
        e40.append(plp.LpVariable("e40_{}".format(i), 0, None))
        e50.append(plp.LpVariable("e50_{}".format(i), 0, None))
        e60.append(plp.LpVariable("e60_{}".format(i), 0, None))
        e70.append(plp.LpVariable("e70_{}".format(i), 0, None))
        e80.append(plp.LpVariable("e80_{}".format(i), 0, None))
        e90.append(plp.LpVariable("e90_{}".format(i), 0, None))
        e100.append(plp.LpVariable("e100_{}".format(i), 0, None))
        # On/off
        lambda10.append(plp.LpVariable(
            "lambda10_{}".format(i), 0, 1, plp.LpInteger))
        lambda20.append(plp.LpVariable(
            "lambda20_{}".format(i), 0, 1, plp.LpInteger))
        lambda30.append(plp.LpVariable(
            "lambda30_{}".format(i), 0, 1, plp.LpInteger))
        lambda40.append(plp.LpVariable(
            "lambda40_{}".format(i), 0, 1, plp.LpInteger))
        lambda50.append(plp.LpVariable(
            "lambda50_{}".format(i), 0, 1, plp.LpInteger))
        lambda60.append(plp.LpVariable(
            "lambda60_{}".format(i), 0, 1, plp.LpInteger))
        lambda70.append(plp.LpVariable(
            "lambda70_{}".format(i), 0, 1, plp.LpInteger))
        lambda80.append(plp.LpVariable(
            "lambda80_{}".format(i), 0, 1, plp.LpInteger))
        lambda90.append(plp.LpVariable(
            "lambda90_{}".format(i), 0, 1, plp.LpInteger))
        lambda100.append(plp.LpVariable(
            "lambda100_{}".format(i), 0, 1, plp.LpInteger))

    # Problem definition
    prob = plp.LpProblem("Operational_cost_minimization", plp.LpMinimize)

    # Constraints
    for i in range(len(grid)):
        #Defining electricity supply
        prob += res_el[i] >= 0
        prob += res_el[i] <= 1000000000 * res[i]
        # prob += elz[i] == wind_el[i] + pv_el[i] + grid_el[i] + bat_out[i] #electrolyzer
        prob += standby_el[i] == elz_standby[i] * elz_max * standby_cost
        prob += elz[i] <= res_el[i] - standby_el[i] - meth_el1[i] - comp_el[i] #overall system electricity consumption
        prob += meth_el1[i] == h2_use[i] * meth_el_factor #electricity required for methanation
        if biogas[i,1] > 0:
            prob += comp_el[i] == (((0.25*h2_use[i]*1000/2.02) / biogas[i,1]) * (biogas[i,0] + biogas[i,1])) * comp_el_factor #electricity required for biogas compression
        elif biogas[i,1] == 0:
            prob += comp_el[i] == 0
        # defining minimum electrolyzer load
        prob += elz[i] >= elz_min * elz_mode[i]
        # defining maximum electrolyzer load
        prob += elz[i] <= elz_max * elz_mode[i]
        # defining minimum methanation load
        prob += h2_use[i] >= meth_min * meth_on[i]
        # defining maximum methanation load
        prob += h2_use[i] <= meth_max * meth_on[i]

        # Part-load variable constraints
        prob += e10[i] <= elz_max * 0.1 * lambda10[i]
        prob += e10[i] >= elz_max * 0 * lambda10[i]
        prob += e20[i] <= elz_max * 0.2 * lambda20[i]
        prob += e20[i] >= elz_max * 0.1 * lambda20[i]
        prob += e30[i] <= elz_max * 0.3 * lambda30[i]
        prob += e30[i] >= elz_max * 0.2 * lambda30[i]
        prob += e40[i] <= elz_max * 0.4 * lambda40[i]
        prob += e40[i] >= elz_max * 0.3 * lambda40[i]
        prob += e50[i] <= elz_max * 0.5 * lambda50[i]
        prob += e50[i] >= elz_max * 0.4 * lambda50[i]
        prob += e60[i] <= elz_max * 0.6 * lambda60[i]
        prob += e60[i] >= elz_max * 0.5 * lambda60[i]
        prob += e70[i] <= elz_max * 0.7 * lambda70[i]
        prob += e70[i] >= elz_max * 0.6 * lambda70[i]
        prob += e80[i] <= elz_max * 0.8 * lambda80[i]
        prob += e80[i] >= elz_max * 0.7 * lambda80[i]
        prob += e90[i] <= elz_max * 0.9 * lambda90[i]
        prob += e90[i] >= elz_max * 0.8 * lambda90[i]
        prob += e100[i] <= elz_max * 1 * lambda100[i]
        prob += e100[i] >= elz_max * 0.9 * lambda100[i]
        # Total electrolyzer load
        prob += elz[i] == e10[i] + e20[i] + e30[i] + e40[i] + \
            e50[i] + e60[i] + e70[i] + e80[i] + e90[i] + e100[i]
        # Only one part of linearization simultaneously (if not working, test to add an e0 varaible?)
        prob += lambda10[i] == elz_mode[i] - lambda20[i] - lambda30[i] - lambda40[i] - \
            lambda50[i] - lambda60[i] - lambda70[i] - \
            lambda80[i] - lambda90[i] - lambda100[i]

        # Linearized H2 production efficiency
        # prob += h2_prod_real[i] == ((e10[i]*k10/elz_max) + (lambda10[i]*m10) + (e20[i]*k20/elz_max) + (lambda20[i]*m20) + (e30[i]*k30/elz_max) + (lambda30[i]*m30) +
        #                        (e40[i]*k40/elz_max) + (lambda40[i]*m40) + (e50[i]*k50/elz_max) + (lambda50[i]*m50) + (e60[i]*k60/elz_max) + (lambda60[i]*m60) +
        #                        (e70[i]*k70/elz_max) + (lambda70[i]*m70) + (e80[i]*k80/elz_max) + (lambda80[i]*m80) + (e90[i]*k90/elz_max) + (lambda90[i]*m90) + (e100[i]*k100/elz_max) + (lambda100[i]*m100))
        prob += h2_prod[i] == ((e10[i]*k10/elz_max) + (lambda10[i]*m10) + (e20[i]*k20/elz_max) + (lambda20[i]*m20) + (e30[i]*k30/elz_max) + (lambda30[i]*m30) +
                               (e40[i]*k40/elz_max) + (lambda40[i]*m40) + (e50[i]*k50/elz_max) + (lambda50[i]*m50) + (e60[i]*k60/elz_max) + (lambda60[i]*m60) +
                               (e70[i]*k70/elz_max) + (lambda70[i]*m70) + (e80[i]*k80/elz_max) + (lambda80[i]*m80) + (e90[i]*k90/elz_max) + (lambda90[i]*m90) + (e100[i]*k100/elz_max) + (lambda100[i]*m100))
        
        # To add a start-up time, uncomment this and use other option above
        # prob += h2_prod_start[i] == h2_prod[i] * (1-elz_startup_time)
        # prob += h2_prod[i] >= h2_prod_start[i] - (100000000*(1-elz_start[i]))
        # prob += h2_prod[i] <= h2_prod_start[i] + (100000000*(1-elz_start[i]))
        # prob += h2_prod[i] >= h2_prod_real[i] - (100000000*elz_start[i])
        # prob += h2_prod[i] <= h2_prod_real[i] + (100000000*elz_start[i])
        
        # defining start-up (Baumhof a good source for this)
        if i > 0:
            # Electrolyzer
            prob += elz_start[i] == elz_mode[i] - elz_mode[i-1] - elz_standby[i-1] + elz_stop[i]
            # Can't go to standby from off
            prob += elz_standby[i] <= 1 - elz_off[i-1]
        else:  # considering end of previous day
            if prev_mode == 1:
                prob += elz_start[i] == 0
            elif prev_mode == 0:
                prob += elz_start[i] == elz_mode[i]
                prob += elz_standby[i] == 0

        # Electrolyzer modes (Baumhof)
        prob += elz_mode[i] + elz_standby[i] + elz_off[i] == 1
        prob += elz_start[i] + elz_stop[i] <= 1
        
        # Can't exceed hourly CO2 availability
        prob += h2_use[i] <= h2_demand[i]

        # Storage charging/discharging
        if i == 0:  # using previous day storage value for first hour
            #Hydrogen    
            prob += h2st[i] == h2_prod[i] - h2_use[i] + h2st_prev
            #Battery (currently not using grid electricity)
            # prob += bat[i] <= ((wind[i] + pv[i] - wind_el[i] - pv_el[i]) * bat_eff) - bat_out[i] + bat_prev
        else:
            #Hydrogen
            prob += h2st[i] == h2_prod[i] - h2_use[i] + h2st[i-1]
            #Battery
            # prob += bat[i] <= ((wind[i] + pv[i] - wind_el[i] - pv_el[i]) * bat_eff) - bat_out[i] + bat[i-1]
        
        # By-product generation
        #Oxygen
        prob += o2_prod[i] == h2_prod[i] * (32/2.02) * 0.5  # kg of o2 produced
        prob += o2_use[i] <= o2_prod[i]
        prob += o2_use[i] <= o2_demand[i]
        
        #Heat
        # prob += elz_heat[i] == elz[i] - (aux_cons*elz_mode[i]) - (h2_prod[i]*39.4)
        #Electrolyzer heat demand, remove methanation heat (already known from h2_use) (maximum of this heat and zero)
        prob += 100000 * ehd[i] >= heat_demand[i] - (h2_use[i] * meth_spec_heat * usable_heat)
        prob += 100000 * (1 - ehd[i]) >= (h2_use[i] * meth_spec_heat * usable_heat) - heat_demand[i]
        prob += elz_heat_demand[i] <= heat_demand[i] - (h2_use[i] * meth_spec_heat * usable_heat) + (100000 * (1 - ehd[i])) 
        prob += elz_heat_demand[i] <= 0 + 100000 * ehd[i]
        #Input water heat consumption
        # h2o = ((h2o_cons * h2_prod[i] * 997 / 1000) * (4.18/3600) * (temp - h2o_temp)) #[kg]*[kWh/kg*K]*[K]
        #No heat during cold start
        prob += elz_heat[i] <= (1-elz_start[i]) * 1000000000
        prob += elz_heat[i] >= 0
        prob += elz_heat[i] <= elz[i] - (aux_cons*elz_mode[i]) - (h2_prod[i]*39.4) - ((h2o_cons * h2_prod[i] * 997 / 1000) * (4.18/3600) * (temp - h2o_temp))
        prob += elz_heat[i] >= (elz[i] - (aux_cons*elz_mode[i]) - (h2_prod[i]*39.4)) - (elz_start[i]*1000000000) - ((h2o_cons * h2_prod[i] * 997 / 1000) * (4.18/3600) * (temp - h2o_temp))  
        #Use limited by both production and demand
        prob += heat_use[i] <= usable_heat * elz_heat[i]
        prob += heat_use[i] <= elz_heat_demand[i]
        prob += heat_use[i] >= 0

        #Seasonal district heat cost
        if (i < 1416) or (i >= 8016):
            prob += heat_income[i] == heat_use[i] * heat_value[3] / 1000
        elif (i >= 1416) and (i < 3624):
            prob += heat_income[i] == heat_use[i] * heat_value[0] / 1000
        elif (i >= 3624) and (i < 5832):
            prob += heat_income[i] == heat_use[i] * heat_value[1] / 1000
        elif (i >= 5832) and (i < 8016):
            prob += heat_income[i] == heat_use[i] * heat_value[2] / 1000
    
        # Start-up time-related losses (either big M here or include in elz start-up cost)
                  
    
    # objective (minimize the electricity cost)
    prob += plp.lpSum([grid_el[i] * grid[i] for i in range(len(grid))]) + plp.lpSum([res_el[i] * res_cost for i in range(len(grid))]) - \
            plp.lpSum([o2_use[i] * (o2_power * grid[i] / 1000) for i in range(len(grid))]) - plp.lpSum([heat_income[i]*1 for i in range(len(grid))]) + \
                ((plp.lpSum([h2_demand[i] - h2_use[i] for i in range(len(grid))]))*10000000)# + \
                    # plp.lpSum([elz_standby[i] * elz_max*standby_cost*grid[i] for i in range(len(grid))]) + plp.lpSum([elz_start[i] * elz_max*startup_cost*grid[i] for i in range(len(grid))]) - \
    #electricity costs (grid, wind, pv)
    #electrolyzer standby cost and start-up cost
    #oxygen income, and heat income
        
    # check solution
    solver = plp.GUROBI_CMD()
    status = prob.solve(solver)

    e_op = []
    grid_op = []
    res_op = []
    storage = []
    h2_produced = []
    h2_used = []
    e_on = []
    e_standby = []
    e_start = []
    e_off = []
    o_prod = []
    o_use = []
    h_prod = []
    h_use = []
    h_inc = []
    bat_state = []
    bat_discharge = []
    sys_op = []
    sb_el = []
    m_el = []
    c_el = []
    e_h_dem = []
    

    # saving variable solutions
    for i in range(len(grid)):
        e_op.append(elz[i].varValue)
        grid_op.append(grid_el[i].varValue)
        res_op.append(res_el[i].varValue)
        storage.append(h2st[i].varValue)
        h2_produced.append(h2_prod[i].varValue)
        h2_used.append(h2_use[i].varValue)
        e_standby.append(elz_standby[i].varValue)
        e_start.append(elz_start[i].varValue)
        e_off.append(elz_off[i].varValue)
        e_on.append(elz_mode[i].varValue)
        o_prod.append(o2_prod[i].varValue)
        o_use.append(o2_use[i].varValue)
        h_prod.append(elz_heat[i].varValue)
        h_use.append(heat_use[i].varValue)
        h_inc.append(heat_income[i].varValue)
        bat_state.append(bat[i].varValue)
        bat_discharge.append(bat_out[i].varValue)
        sys_op.append(elz[i].varValue + standby_el[i].varValue + meth_el1[i].varValue + comp_el[i].varValue)
        sb_el.append(standby_el[i].varValue)
        m_el.append(meth_el1[i].varValue)
        c_el.append(comp_el[i].varValue)
        e_h_dem.append(elz_heat_demand[i].varValue)
        

    h2_missed = list(np.array(h2_demand) - np.array(h2_used))
    demand_vector = h2_demand
    grid_inc = np.array(grid_op) * grid / 1000
    o2_inc = np.array(o_use) * (o2_power * grid[i] / 1000)

    op = pd.DataFrame(
        {'Electrolyzer operation': e_op,
         'Grid purchases': grid_op,
         'RES power': res_op,
          'PV power': res_op,
         'Electricity price': grid,
         'Storage': storage,
         'Demand': demand_vector,
         'H2 prod': h2_produced,
         'H2 used': h2_used,
         # 'Efficiency': eff,
         'Unmet demand': h2_missed,
         'Status': status,
         'On': e_on,
         'Standby': e_standby,
         'Off': e_off,
         'Cold start': e_start,
         'O2 prod': o_prod,
         'O2 use': o_use,
         'Heat prod': h_prod,
         'Heat use': h_use,
         'Battery state': bat_state,
         'Battery discharging': bat_discharge,
         'Heat income': h_inc,
         'Grid income': grid_inc,
         'Oxygen income': o2_inc,
         'System operation': sys_op,
         'Standby electricity': sb_el,
         'Methanation electricity': m_el,
         'Compression electricity': c_el,
         'Heat demand': e_h_dem,
         })
    return op


def p2g_wwtp3(
    h2_demand, #[kg/h]
    heat_demand, #[kWh/h] hourly
    o2_demand, #[mol/h] hourly
    o2_power, #[kWh avoided/kg O2]
    biogas, #[mol/h] CH4 and CO2 flow
    k_values,
    m_values,
    grid,
    wind,
    pv,
    elz_max,  # kW
    elz_min,  # kW
    aux_cons,  # kW
    h2st_max,  # kg?
    h2st_prev,  # storage from previous day
    meth_max,  # maximum H2 use [kg/h]
    meth_min,  # minimum H2 use [kg/h]
    meth_el_factor: float = 0.0, #[kWh/molCO2 converted]
    comp_el_factor: float = 0.0, #[kWh/mol compressed gas]
    meth_el: float = 0.0, #electricity demand for methanation and biogas compression [kWh/molCO2]
    heat_value: list = [0.0,0.0,0.0,0.0], #[€/MWh] for spring, summer, autumn and winter respectively
    startup_cost: float = 0.1,  # fraction of rated power
    standby_cost: float = 0.02,  # fraction of rated power
    elz_startup_time: float = 0.0, # [hour] time until H2 production starts after a cold startup
    elz_heat_time: float = 0.0, # [hour] time until electrolyzer has reached operating temperature
    prev_mode: int = 1,  # on/standby/off in last hour of previous day
    elz_eff: float = 0.7,  # HHV
    wind_cost: float = -9,  # [€/MWh]
    pv_cost: float = -10,  # [€/MWh] (only to prioritize ahead of wind and negative prices)
    bat_cap: float = 0, # [kWh]
    bat_eff: float = 0.95, # battery round trip efficiency
    bat_prev: float = 0, # [kWh] previous day battery charge
    meth_spec_heat: float = 0.0, # [kWh/kgH2] methanation heat generation per kg H2 methanized
    usable_heat: float = 0.8, #usable heat fraction
    h2o_cons: float = 10.0, #water consumption [lH2O/kgH2]
    temp: float = 80.0,
    h2o_temp: float = 15.0,
) -> pd.DataFrame:
    """Returns electrolyzer dispatch based on both renewable and grid electricity
    optimized to minimize the cost of electricity for a fixed H2 demand.
    Also including a linearized part-load efficiency and on/standby/off modes for
    the electrolyzer as well as startup losses for electrolysis and methanation.
    Using linearization parameters as proposed in Wirtz et al. 2021"""
    # Now: lowest value in each interval has the optimal efficiency it seems
    # What about the start-up time? Can we ignore this since it is very short with PEM?
    # Should startup and standby costs be based on spot price or a constant price?
    # Methanation ignored since it is based on CO2 availability

    #Hourly hydrogen demand (from CO2 availability)    
    h2_demand = h2_demand.flatten().tolist()
    
    wind = list(wind)
    pv = list(pv)
    grid = list(grid)
    h2st_prev = h2st_prev * 2.02 / 1000  # mol to kg

    # Define variables
    elz = []
    grid_el = []
    wind_el = []
    pv_el = []
    h2st = []
    h2_prod = []
    h2_prod_start = []
    h2_prod_real = []
    elz_mode = []
    elz_start = []
    elz_stop = []
    elz_off = []
    elz_standby = []
    h2_use = []
    meth_on = []
    o2_prod = []
    o2_use = []
    elz_heat = []
    heat_use = []
    heat_income = []
    elz_heat_demand = []
    ehd = []
    bat = []
    bat_out = []
    standby_el = []
    meth_el1 = []
    comp_el = []
    # elz1 = []
    # Part-load variables
    # Load
    e10 = []
    e20 = []
    e30 = []
    e40 = []
    e50 = []
    e60 = []
    e70 = []
    e80 = []
    e90 = []
    e100 = []
    # On/off
    lambda10 = []
    lambda20 = []
    lambda30 = []
    lambda40 = []
    lambda50 = []
    lambda60 = []
    lambda70 = []
    lambda80 = []
    lambda90 = []
    lambda100 = []

    # Linearization parameters (see "Linearized efficiency" Excel)
    # gradient
    k10 = k_values[0]
    k20 = k_values[1]
    k30 = k_values[2]
    k40 = k_values[3]
    k50 = k_values[4]
    k60 = k_values[5]
    k70 = k_values[6]
    k80 = k_values[7]
    k90 = k_values[8]
    k100 = k_values[9]
    # m-value
    m10 = m_values[0]
    m20 = m_values[1]
    m30 = m_values[2]
    m40 = m_values[3]
    m50 = m_values[4]
    m60 = m_values[5]
    m70 = m_values[6]
    m80 = m_values[7]
    m90 = m_values[8]
    m100 = m_values[9]

    for i in range(len(grid)):
        # Electrolyzer operation
        elz.append(plp.LpVariable("elz_{}".format(i), 0, elz_max))
        # elz1.append(plp.LpVariable("elz1_{}".format(i), 0, None)) #an independent copy
        elz_mode.append(plp.LpVariable("elz_mode_{}".format(i), 0, 1, plp.LpInteger))
        elz_start.append(plp.LpVariable("elz_start_{}".format(i), 0, 1, plp.LpInteger))
        elz_stop.append(plp.LpVariable("elz_stop_{}".format(i), 0, 1, plp.LpInteger))
        elz_standby.append(plp.LpVariable("elz_standby_{}".format(i), 0, 1, plp.LpInteger))
        elz_off.append(plp.LpVariable("elz_off_{}".format(i), 0, 1, plp.LpInteger))
        # Purchased grid electricity
        grid_el.append(plp.LpVariable("grid_el_{}".format(i), 0, None))
        # Wind-based operation
        wind_el.append(plp.LpVariable("wind_el_{}".format(i), 0, None))
        # PV-based operation
        pv_el.append(plp.LpVariable("pv_el_{}".format(i), 0, None))
        # Hydrogen storage
        h2st.append(plp.LpVariable("h2st_{}".format(i), 0, h2st_max))
        # Hydrogen production and utilization
        h2_prod.append(plp.LpVariable("h2_prod_{}".format(i), 0, None))
        h2_prod_start.append(plp.LpVariable("h2_prod_start_{}".format(i), 0, None))
        h2_prod_real.append(plp.LpVariable("h2_prod_real_{}".format(i), 0, None))
        h2_use.append(plp.LpVariable("h2_use_{}".format(i), 0, meth_max))
        # Methanation
        meth_on.append(plp.LpVariable("meth_on_{}".format(i), 0, 1, plp.LpInteger))
        meth_el1.append(plp.LpVariable("meth_el_{}".format(i), 0, None))
        comp_el.append(plp.LpVariable("comp_el_{}".format(i), 0, None))
        # Oxygen production and utilization
        o2_prod.append(plp.LpVariable("o2_prod_{}".format(i), 0, None))
        o2_use.append(plp.LpVariable("o2_use_{}".format(i), 0, None))
        # Heat production and utilization
        elz_heat.append(plp.LpVariable("elz_heat_{}".format(i), 0, None))
        heat_use.append(plp.LpVariable("heat_use_{}".format(i), 0, None))
        heat_income.append(plp.LpVariable("heat_income_{}".format(i), 0, None))
        elz_heat_demand.append(plp.LpVariable("elz_heat_demand_{}".format(i), 0, None))
        ehd.append(plp.LpVariable("ehd_{}".format(i), 0, 1, plp.LpInteger))
        #Battery
        bat.append(plp.LpVariable("bat_{}".format(i), 0, bat_cap))
        bat_out.append(plp.LpVariable("bat_out_{}".format(i), 0, bat_cap))
        #Standby electricity consumption
        standby_el.append(plp.LpVariable("sys_el_{}".format(i), 0, None))
        
        # Part-load variables
        # Load
        e10.append(plp.LpVariable("e10_{}".format(i), 0, None))
        e20.append(plp.LpVariable("e20_{}".format(i), 0, None))
        e30.append(plp.LpVariable("e30_{}".format(i), 0, None))
        e40.append(plp.LpVariable("e40_{}".format(i), 0, None))
        e50.append(plp.LpVariable("e50_{}".format(i), 0, None))
        e60.append(plp.LpVariable("e60_{}".format(i), 0, None))
        e70.append(plp.LpVariable("e70_{}".format(i), 0, None))
        e80.append(plp.LpVariable("e80_{}".format(i), 0, None))
        e90.append(plp.LpVariable("e90_{}".format(i), 0, None))
        e100.append(plp.LpVariable("e100_{}".format(i), 0, None))
        # On/off
        lambda10.append(plp.LpVariable(
            "lambda10_{}".format(i), 0, 1, plp.LpInteger))
        lambda20.append(plp.LpVariable(
            "lambda20_{}".format(i), 0, 1, plp.LpInteger))
        lambda30.append(plp.LpVariable(
            "lambda30_{}".format(i), 0, 1, plp.LpInteger))
        lambda40.append(plp.LpVariable(
            "lambda40_{}".format(i), 0, 1, plp.LpInteger))
        lambda50.append(plp.LpVariable(
            "lambda50_{}".format(i), 0, 1, plp.LpInteger))
        lambda60.append(plp.LpVariable(
            "lambda60_{}".format(i), 0, 1, plp.LpInteger))
        lambda70.append(plp.LpVariable(
            "lambda70_{}".format(i), 0, 1, plp.LpInteger))
        lambda80.append(plp.LpVariable(
            "lambda80_{}".format(i), 0, 1, plp.LpInteger))
        lambda90.append(plp.LpVariable(
            "lambda90_{}".format(i), 0, 1, plp.LpInteger))
        lambda100.append(plp.LpVariable(
            "lambda100_{}".format(i), 0, 1, plp.LpInteger))

    # Problem definition
    prob = plp.LpProblem("Operational_cost_minimization", plp.LpMinimize)

    # Constraints
    for i in range(len(grid)):
        # defining electricity supply
        prob += wind_el[i] <= wind[i] #wind
        prob += pv_el[i] <= pv[i] #PV
        # prob += elz[i] == wind_el[i] + pv_el[i] + grid_el[i] + bat_out[i] #electrolyzer
        prob += standby_el[i] == elz_standby[i] * elz_max * standby_cost
        prob += elz[i] == wind_el[i] + pv_el[i] + grid_el[i] + bat_out[i] - standby_el[i] - meth_el1[i] - comp_el[i] #overall system electricity consumption
        prob += meth_el1[i] == h2_use[i] * meth_el_factor #electricity required for methanation
        if biogas[i,1] > 0:
            prob += comp_el[i] == (((0.25*h2_use[i]*1000/2.02) / biogas[i,1]) * (biogas[i,0] + biogas[i,1])) * comp_el_factor #electricity required for biogas compression
        elif biogas[i,1] == 0:
            prob += comp_el[i] == 0
        # defining minimum electrolyzer load
        prob += elz[i] >= elz_min * elz_mode[i]
        # defining maximum electrolyzer load
        prob += elz[i] <= elz_max * elz_mode[i]
        # defining minimum methanation load
        prob += h2_use[i] >= meth_min * meth_on[i]
        # defining maximum methanation load
        prob += h2_use[i] <= meth_max * meth_on[i]

        # Part-load variable constraints
        prob += e10[i] <= elz_max * 0.1 * lambda10[i]
        prob += e10[i] >= elz_max * 0 * lambda10[i]
        prob += e20[i] <= elz_max * 0.2 * lambda20[i]
        prob += e20[i] >= elz_max * 0.1 * lambda20[i]
        prob += e30[i] <= elz_max * 0.3 * lambda30[i]
        prob += e30[i] >= elz_max * 0.2 * lambda30[i]
        prob += e40[i] <= elz_max * 0.4 * lambda40[i]
        prob += e40[i] >= elz_max * 0.3 * lambda40[i]
        prob += e50[i] <= elz_max * 0.5 * lambda50[i]
        prob += e50[i] >= elz_max * 0.4 * lambda50[i]
        prob += e60[i] <= elz_max * 0.6 * lambda60[i]
        prob += e60[i] >= elz_max * 0.5 * lambda60[i]
        prob += e70[i] <= elz_max * 0.7 * lambda70[i]
        prob += e70[i] >= elz_max * 0.6 * lambda70[i]
        prob += e80[i] <= elz_max * 0.8 * lambda80[i]
        prob += e80[i] >= elz_max * 0.7 * lambda80[i]
        prob += e90[i] <= elz_max * 0.9 * lambda90[i]
        prob += e90[i] >= elz_max * 0.8 * lambda90[i]
        prob += e100[i] <= elz_max * 1 * lambda100[i]
        prob += e100[i] >= elz_max * 0.9 * lambda100[i]
        # Total electrolyzer load
        prob += elz[i] == e10[i] + e20[i] + e30[i] + e40[i] + \
            e50[i] + e60[i] + e70[i] + e80[i] + e90[i] + e100[i]
        # Only one part of linearization simultaneously (if not working, test to add an e0 varaible?)
        prob += lambda10[i] == elz_mode[i] - lambda20[i] - lambda30[i] - lambda40[i] - \
            lambda50[i] - lambda60[i] - lambda70[i] - \
            lambda80[i] - lambda90[i] - lambda100[i]

        #Linearized H2 production efficiency
        # prob += h2_prod_real[i] == ((e10[i]*k10/elz_max) + (lambda10[i]*m10) + (e20[i]*k20/elz_max) + (lambda20[i]*m20) + (e30[i]*k30/elz_max) + (lambda30[i]*m30) +
        #                         (e40[i]*k40/elz_max) + (lambda40[i]*m40) + (e50[i]*k50/elz_max) + (lambda50[i]*m50) + (e60[i]*k60/elz_max) + (lambda60[i]*m60) +
        #                         (e70[i]*k70/elz_max) + (lambda70[i]*m70) + (e80[i]*k80/elz_max) + (lambda80[i]*m80) + (e90[i]*k90/elz_max) + (lambda90[i]*m90) + (e100[i]*k100/elz_max) + (lambda100[i]*m100))
        prob += h2_prod[i] == ((e10[i]*k10/elz_max) + (lambda10[i]*m10) + (e20[i]*k20/elz_max) + (lambda20[i]*m20) + (e30[i]*k30/elz_max) + (lambda30[i]*m30) +
                                (e40[i]*k40/elz_max) + (lambda40[i]*m40) + (e50[i]*k50/elz_max) + (lambda50[i]*m50) + (e60[i]*k60/elz_max) + (lambda60[i]*m60) +
                                (e70[i]*k70/elz_max) + (lambda70[i]*m70) + (e80[i]*k80/elz_max) + (lambda80[i]*m80) + (e90[i]*k90/elz_max) + (lambda90[i]*m90) + (e100[i]*k100/elz_max) + (lambda100[i]*m100))
        
        # To add a start-up H2 losses, uncomment this and use other option above
        # prob += h2_prod_start[i] == h2_prod_real[i] * (1-elz_startup_time)
        # prob += h2_prod[i] >= h2_prod_start[i] - (100000000*(1-elz_start[i]))
        # prob += h2_prod[i] <= h2_prod_start[i] + (100000000*(1-elz_start[i]))
        # prob += h2_prod[i] >= h2_prod_real[i] - (100000000*elz_start[i])
        # prob += h2_prod[i] <= h2_prod_real[i] + (100000000*elz_start[i])
        
        # defining start-up (Baumhof a good source for this)
        if i > 0:
            # Electrolyzer
            prob += elz_start[i] >= elz_mode[i] - elz_mode[i-1] - elz_standby[i-1]# + elz_stop[i]
            # Can't start if on or standby during previous hour or on or standby in current hour
            # prob += elz_start[i] <= 1 - elz_mode[i-1] - elz_standby[i-1]
            # prob += elz_start[i] <= 1 - elz_off[i] - elz_standby[i]
            # Can't start without being on
            # prob += elz_start[i] <= elz_mode[i]
            #CANNOT GET STARTS AND STOPS TO WORK
                #BEFORE I HAD SIMPLY ADDED "+ELZ_STOP[I]" TO TOP IN THIS LOOP AND DID NOT HAVE THE ONE JUST ABOVE.
            # Can't go to standby from off or vice versa
            prob += elz_standby[i] <= 1 - elz_off[i-1]
            # prob += elz_off[i] <= 1 - elz_standby[i-1]
        else:  # considering end of previous day
            if prev_mode == 1:
                prob += elz_start[i] == 0
            elif prev_mode == 0:
                prob += elz_start[i] == elz_mode[i]
                prob += elz_standby[i] == 0

        # Electrolyzer modes (Baumhof)
        prob += elz_mode[i] + elz_standby[i] + elz_off[i] == 1
        # prob += elz_start[i] + elz_stop[i] <= 1
        
        # Can't exceed hourly CO2 availability
        prob += h2_use[i] <= h2_demand[i]

        # Storage charging/discharging
        if i == 0:  # using previous day storage value for first hour
            #Hydrogen    
            prob += h2st[i] == h2_prod[i] - h2_use[i] + h2st_prev
            #Battery (currently not using grid electricity)
            prob += bat[i] <= ((wind[i] + pv[i] - wind_el[i] - pv_el[i]) * bat_eff) - bat_out[i] + bat_prev
        else:
            #Hydrogen
            prob += h2st[i] == h2_prod[i] - h2_use[i] + h2st[i-1]
            #Battery
            prob += bat[i] <= ((wind[i] + pv[i] - wind_el[i] - pv_el[i]) * bat_eff) - bat_out[i] + bat[i-1]
        
        # By-product generation
        #Oxygen
        prob += o2_prod[i] == h2_prod[i] * (32/2.02) * 0.5  # kg of o2 produced
        prob += o2_use[i] <= o2_prod[i]
        prob += o2_use[i] <= o2_demand[i]
        
        #Heat
        # prob += elz_heat[i] == elz[i] - (aux_cons*elz_mode[i]) - (h2_prod[i]*39.4)
        #Electrolyzer heat demand, remove methanation heat (already known from h2_use) (maximum of this heat and zero)
        prob += 100000 * ehd[i] >= heat_demand[i] - (h2_use[i] * meth_spec_heat * usable_heat)
        prob += 100000 * (1 - ehd[i]) >= (h2_use[i] * meth_spec_heat * usable_heat) - heat_demand[i]
        prob += elz_heat_demand[i] <= heat_demand[i] - (h2_use[i] * meth_spec_heat * usable_heat) + (100000 * (1 - ehd[i])) 
        prob += elz_heat_demand[i] <= 0 + 100000 * ehd[i]
        #Input water heat consumption
        # h2o = ((h2o_cons * h2_prod[i] * 997 / 1000) * (4.18/3600) * (temp - h2o_temp)) #[kg]*[kWh/kg*K]*[K]
        #No heat during cold start
        prob += elz_heat[i] <= (1-elz_start[i]) * 1000000000
        prob += elz_heat[i] >= 0
        prob += elz_heat[i] <= elz[i] - (aux_cons*elz_mode[i]) - (h2_prod[i]*39.4) - ((h2o_cons * h2_prod[i] * 997 / 1000) * (4.18/3600) * (temp - h2o_temp))
        prob += elz_heat[i] >= (elz[i] - (aux_cons*elz_mode[i]) - (h2_prod[i]*39.4)) - (elz_start[i]*1000000000) - ((h2o_cons * h2_prod[i] * 997 / 1000) * (4.18/3600) * (temp - h2o_temp))  
        #Use limited by both production and demand
        prob += heat_use[i] <= usable_heat * elz_heat[i]
        prob += heat_use[i] <= elz_heat_demand[i]
        prob += heat_use[i] >= 0

        # #Seasonal district heat cost
        if (i < 1416) or (i >= 8016):
            prob += heat_income[i] == heat_use[i] * heat_value[3] / 1000
        elif (i >= 1416) and (i < 3624):
            prob += heat_income[i] == heat_use[i] * heat_value[0] / 1000
        elif (i >= 3624) and (i < 5832):
            prob += heat_income[i] == heat_use[i] * heat_value[1] / 1000
        elif (i >= 5832) and (i < 8016):
            prob += heat_income[i] == heat_use[i] * heat_value[2] / 1000
    
    
    # objective (minimize the electricity cost)
    prob += plp.lpSum([grid_el[i] * grid[i] / 1000 for i in range(len(grid))]) + plp.lpSum([wind_el[i] * wind_cost / 1000 for i in range(len(grid))]) + plp.lpSum([pv_el[i] * pv_cost / 1000 for i in range(len(grid))]) + \
        ((plp.lpSum([h2_demand[i] - h2_use[i] for i in range(len(grid))]))*10000000) + (plp.lpSum([elz_start[i] * elz_startup_time*elz_max*grid[i]/1000 for i in range(len(grid))])) - \
            plp.lpSum([o2_use[i] * (o2_power * grid[i] / 1000) for i in range(len(grid))]) - plp.lpSum([heat_income[i] for i in range(len(grid))])# + \
    #electricity costs (grid, wind, pv)
    #oxygen income, and heat income
    
    # check solution
    solver = plp.GUROBI_CMD()
    status = prob.solve(solver)

    e_op = []
    grid_op = []
    wind_op = []
    pv_op = []
    storage = []
    h2_produced = []
    h2_used = []
    e_on = []
    e_standby = []
    e_start = []
    e_off = []
    o_prod = []
    o_use = []
    h_prod = []
    h_use = []
    h_inc = []
    bat_state = []
    bat_discharge = []
    sys_op = []
    sb_el = []
    m_el = []
    c_el = []
    e_h_dem = []
    h2_produced_start = []
    h2_produced_real = []

    

    # saving variable solutions
    for i in range(len(grid)):
        e_op.append(elz[i].varValue)
        grid_op.append(grid_el[i].varValue)
        wind_op.append(wind_el[i].varValue)
        pv_op.append(pv_el[i].varValue)
        storage.append(h2st[i].varValue)
        h2_produced.append(h2_prod[i].varValue)
        h2_used.append(h2_use[i].varValue)
        e_standby.append(elz_standby[i].varValue)
        e_start.append(elz_start[i].varValue)
        e_off.append(elz_off[i].varValue)
        e_on.append(elz_mode[i].varValue)
        o_prod.append(o2_prod[i].varValue)
        o_use.append(o2_use[i].varValue)
        h_prod.append(elz_heat[i].varValue)
        h_use.append(heat_use[i].varValue)
        h_inc.append(heat_income[i].varValue)
        bat_state.append(bat[i].varValue)
        bat_discharge.append(bat_out[i].varValue)
        sys_op.append(elz[i].varValue + standby_el[i].varValue + meth_el1[i].varValue + comp_el[i].varValue)
        sb_el.append(standby_el[i].varValue)
        m_el.append(meth_el1[i].varValue)
        c_el.append(comp_el[i].varValue)
        e_h_dem.append(elz_heat_demand[i].varValue)
        h2_produced_start.append(h2_prod_start[i].varValue)
        h2_produced_real.append(h2_prod_real[i].varValue)
        

    h2_missed = list(np.array(h2_demand) - np.array(h2_used))
    demand_vector = h2_demand
    grid_inc = np.array(grid_op) * grid / 1000
    o2_inc = np.array(o_use) * (o2_power * grid[i] / 1000)

    op = pd.DataFrame(
        {'Electrolyzer operation': e_op,
         'Grid purchases': grid_op,
         'Wind power': wind_op,
         'PV power': pv_op,
         'Electricity price': grid,
         'Storage': storage,
         'Demand': demand_vector,
         'H2 prod': h2_produced,
         'H2 used': h2_used,
         # 'Efficiency': eff,
         'Unmet demand': h2_missed,
         'Status': status,
         'On': e_on,
         'Standby': e_standby,
         'Off': e_off,
         'Cold start': e_start,
          'O2 prod': o_prod,
          'O2 use': o_use,
          'Heat prod': h_prod,
          'Heat use': h_use,
          'Battery state': bat_state,
          'Battery discharging': bat_discharge,
          'Heat income': h_inc,
          'Grid income': grid_inc,
          'Oxygen income': o2_inc,
           'System operation': sys_op,
           'Standby electricity': sb_el,
           'Methanation electricity': m_el,
           'Compression electricity': c_el,
          'Heat demand': e_h_dem,
          'H2 prod start': h2_produced_start,
          'H2 prod real': h2_produced_real,
         })
    return op