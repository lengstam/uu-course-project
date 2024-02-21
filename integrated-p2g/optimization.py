# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:57:13 2023

@author: Linus Engstam
"""

import P2G.main as main
import pandas as pd
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from time import sleep
from tabulate import tabulate


#Questions: should the RES have a cost in the dispatch? Now they do (the LCOE).
#If there is a by-product demand, should it be included in the dispatch strategy? Probably!
#The environmental benefits of by-products should also be included!


#Issue due to the same solution for alpha=0 and alpha=1 in for configurations.
# """ OPTIMIZATION OF COMPONENT SIZING """

#Define optimization parameters
alpha = 0 #0=cost, 1=emissions
ef_type = "aef"
co2_use = "daily"

#7th run an issue with alpha=0.5 (1.5, 4, 2, 0)
#Define investigated sizes
elz_size = [9] #[MW]
meth_size = [8] #[MW CH4 out]
h2st_size = [2] #[hours] Should it be defined in hours, kg/MWh or related to electrolyzer?
wind_size = [0] #[MW] Should it be defined in relation to elz or MW?
pv_size = [0] #[MW]
bat_size = [0] #hours? NOT IMPLEMENTED

if len(elz_size) == 1 and len(h2st_size) == 1 and len(wind_size) == 1 and len(pv_size) == 1 and len(meth_size) == 1:
    run_type = "single"
else:
    run_type = "optimization"

#Counting etc.
sims = len(elz_size) * len(h2st_size) * len(wind_size) * len(pv_size)
count = 0
# t = time.process_time()
#Progress bar
#pbar_outer = tqdm(total=outer_loop, position=0, leave=True, ncols=80, ascii=True)
# pbar_inner = tqdm(total=sims, position=0, leave=False, ncols=50, ascii=True)
#Create results dataframe
results = pd.DataFrame({'KPIs': ['LCOE', 'NPV', 'MSP', 'Gas eff.', 'Tot eff.', 'Spec.ems. (AEF)', 'Spec. ems. (MEF)', 'Gas loss [%]', 'Test']}).set_index('KPIs')

#Run simulation for all investigated sizes
for e in range(len(elz_size)):#, desc='Electrolyzer', position=0, leave=True):
    for m in range(len(meth_size)):
        for s in range(len(h2st_size)):#, desc='Storage', position=1, leave=True, total=int(sims/(len(wind_size)*len(pv_size)))):
            for w in range(len(wind_size)):#, desc='Wind', position=2, leave=True, total=int(sims/len(pv_size))):
                for p in range(len(pv_size)):#, desc='PV', position=3, leave=True, total=sims):
                    KPIs, test, __ = main.P2G_sim(elz_size=elz_size[e], meth_size=meth_size[m], h2st_size=h2st_size[s], wind_size=wind_size[w], pv_size=pv_size[p], alpha=alpha, ef_type=ef_type, run_type=run_type, co2_use=co2_use)
                    # KPIs, test = main.P2G_sim(elz_size=elz_size[e], h2st_size=h2st_size[s], wind_size=wind_size[w], pv_size=pv_size[p], alpha=alpha, ef_type=ef_type, run_type=run_type, co2_use=co2_use)

                    #Results vector column
                    results['E: {}, S: {}, W: {}, P: {}'.format(elz_size[e],h2st_size[s],wind_size[w],pv_size[p])] = [KPIs[1][0], KPIs[1][1], KPIs[1][2], KPIs[1][3], KPIs[1][4], KPIs[1][5], KPIs[1][6], KPIs[1][7], test]
                    count = count + 1
                    # seconds = time.process_time() - t
                    # minutes = math.floor(seconds/60)
                    # rem_sec = round(seconds - (minutes*60))
                    # elapsed_time = '{} minutes and {} seconds'.format(minutes,rem_sec)
                    # remaining_sec = (sims-count) * (seconds / count)
                    # remaining_min = math.floor(remaining_sec/60)
                    # remaining_rem_sec = round(remaining_sec - (remaining_min*60))
                    # time_remaining = '{} minutes and {} seconds'.format(remaining_min,remaining_rem_sec)
                    # print('{}/{} simulations performed. Elapsed time: {}. Estimated time remaining: {}'.format(count,sims,elapsed_time,time_remaining))
                    print('{}/{} simulations performed'.format(count,sims))
                # pbar_inner.set_description(f"inner iteration {p + 1} of {sims}")
                # pbar_inner.update()
                # sleep(1)

#Determine lowest LCOE configuration
min_lcoe = min(results.iloc[0,:])
index2 = results.columns[results.eq(min_lcoe).any()]

#(Could do a bubble plot to get a thrid KPI as dot size, but perhaps unnecessarily complex?)
#Pareto front
fig, ax1 = plt.subplots()
#Scatter
ax1.plot(results.iloc[2,:], results.iloc[5,:], ls='none', marker='o') #LCOE vs. AEF
#Pareto
sorted_list = sorted([[results.iloc[2,i], results.iloc[5,i]] for i in range(len(results.iloc[2,:]))], reverse=False)
pareto_front = [sorted_list[0]]
for pair in sorted_list[1:]:
    if pair[1] < pareto_front[-1][1]: #could be an issue not to have "<="
                pareto_front.append(pair)
pf_lcoe = [pair[0] for pair in pareto_front]
pf_aef = [pair[1] for pair in pareto_front]
ax1.plot(pf_lcoe, pf_aef, ls='-', marker='o', color='r')

ax1.set_ylabel('Spec. ems (AEF) [kg$_{CO_2}$/MWh$_{CH_4}$]')
ax1.set_xlabel('MSP [€/MWh]')

#Impact of component size



# #Table of single-run results
# # if run_type == "single":
# #     table = [kpis[0], kpis[1]]
# #     print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))


""" PYMOO approach """

#From ChatGPT
#from pymoo.model.problem import Problem
# from pymoo.core.problem import ElementwiseProblem
# from pymoo.algorithms.moo.nsga2 import NSGA2
# from pymoo.optimize import minimize
# from pymoo.operators.sampling.rnd import IntegerRandomSampling
# from pymoo.operators.repair.rounding import RoundingRepair
# from pymoo.operators.crossover.sbx import SBX
# from pymoo.operators.mutation.pm import PM
# from pymoo.visualization.scatter import Scatter

# run_type = "single"

# class MyProblem(ElementwiseProblem):
#     def __init__(self):
#         super().__init__(n_var=1, n_obj=2, n_constr=0, xl=8, xu=10, vtype=int)

#     def _evaluate(self, x, out, *args, **kwargs):
#         #y1, y2 = self.compute_outputs(x) # compute outputs based on inputs x
#         lcoe, aef = main.P2G_sim(elz_size=x, meth_size=8, h2st_size=4, wind_size=4, pv_size=4, alpha=alpha, ef_type=ef_type, run_type=run_type, co2_use=co2_use)
#         #out["F"] = [y1, y2]
#         out["F"] = [lcoe, aef]

#     # def compute_outputs(self, x):
#     #     # compute outputs based on inputs x
#     #     y1 = 2*x[0] + 3*x[1] + x[2]**2
#     #     y2 = x[0]**2 + x[1] + 4*x[2]
#     #     return y1, y2

# problem = MyProblem()

# #algorithm = get_algorithm("nsga2") # choose the algorithm
# algorithm = NSGA2(pop_size=20,
#             sampling=IntegerRandomSampling(),
#             crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
#             mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
#             eliminate_duplicates=True,
#             )

# res = minimize(problem, algorithm, termination=('n_gen', 100))

# print("Best solution found: X = %s, F = %s" % (res.X, res.F))

# plt.scatter(res.F[:, 0], res.F[:, 1], s=30, facecolors='none', edgecolors='b', label="Solutions")
# plt.plot(res.F[:, 0], res.F[:, 1], alpha=0.5, linewidth=2.0, color="red", label="Pareto-front")
# plt.title("Objective Space")
# plt.legend()
# plt.show()

# plot = Scatter()
# plot.add(res.F[:,0], res.F[:,1], s=30, facecolors='none', edgecolors='r')
# plot.add(res.F[:,0], res.F[:,1], plot_type="line", color="black", linewidth=2)
# plot.show()