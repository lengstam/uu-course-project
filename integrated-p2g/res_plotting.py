# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:44:29 2023

@author: enls0001
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as lines
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import waterfall_chart
import matplotlib.pyplot as plt

""" SENSITIVITY ANALYSIS PLOTS """

#Color palettes
teal_dark = sns.dark_palette("teal",5)
teal_light = sns.light_palette("teal",5)
pal2 = sns.dark_palette("orange",10)

#SUBPLOT DEFINITION
# fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)

plt.figure()
ax1 = plt.subplot(2,2,2)
ax2 = plt.subplot(2,2,4)
ax3 = plt.subplot(1,2,1)

#MAIN LCOP2G SENSITIVITY ANALYSIS PLOT
el_price = [183.09*100/194.43,100,205.76*100/194.43]
elz_capex = [188.91*100/194.43,100,199.94*100/194.43]
lifetime = [192.07*100/194.43,100,197.41*100/194.43]
discount = [191.68*100/194.43,100,197.2*100/194.43]
methanation = [191.97*100/194.43,100,196.88*100/194.43]
install_cost = [193.50*100/194.43,100,195.36*100/194.43]
meth_energy = [194.27*100/194.43,100,194.74*100/194.43]
o2_cost = [194.12*100/194.43,100,194.73*100/194.43] #both equipment and piping
pipe_dist = [194.14*100/194.43,100,194.72*100/194.43]
heat_cost = [194.27*100/194.43,100,194.58*100/194.43] #both equipment and piping
h2st_cost = [194.35*100/194.43,100,194.50*100/194.43]
# h2o_cost = [X,100,X]#[194.41,194.43,194.44]
categories = ['Electricity price','Electrolyser CAPEX','System lifetime','Discount rate','Methanation CAPEX','Installation cost',
              'Oxygen integration CAPEX','Piping distance','Methanation energy','Heat integration CAPEX','Hydrogen storage CAPEX']
increase = [el_price[2],elz_capex[2],lifetime[0],discount[2],methanation[2],install_cost[2],o2_cost[2],pipe_dist[2],meth_energy[2],heat_cost[2],h2st_cost[2]]
decrease = [el_price[0],elz_capex[0],lifetime[2],discount[0],methanation[0],install_cost[0],o2_cost[0],pipe_dist[0],meth_energy[0],heat_cost[0],h2st_cost[0]]
increase = increase[::-1]
decrease = decrease[::-1]
categories = categories[::-1]

# fig, ax = plt.subplots()
ax3.barh(categories, increase-(np.zeros(11,)+100), align='center', height = 0.6,facecolor=teal_dark[3],edgecolor=teal_dark[3], lw=0)
ax3.barh(categories, ((np.zeros(11,)+100)-decrease)*-1, align='center', height = 0.6,facecolor=teal_light[3],edgecolor=teal_light[3], lw=0)

# Show the leave % on each  bar
for i, leave in enumerate(increase):
    ax3.text( leave + 8, categories[i],'{:,.0f}%'.format(leave), ha='center', size = 16,color = "black")

# Show the remain % on each  bar
for i, remain in enumerate(decrease):
    ax3.text(( remain*-1) - 8, categories[i],'{} %'.format(remain), ha='center', size = 16,color = "black")
    
# Show the title at the top of each  bar
# As the y axis is a category axis to be able to plot the labels outise the bars, we need to use the get method
for  bar, cat in zip(ax3.patches, categories):
    width = bar.get_width()
    label_y = bar.get_y() + bar.get_height() + 0.06
    label_x = len(cat)*(1.5/22)
    plt.text(-label_x, label_y, s=f'{cat}', size = 12)

ax3.set_xlim(-6,6)
ax3.set_xticks([-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6])
ax3.set_xlabel('LCOP2G change [%]', fontsize=12)

ax3.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    color='white',
    labelcolor='white') # labels along the bottom edge are off

plt.gca().xaxis.grid(True)
ax3.set_axisbelow(True)

#Legend
inc_patch = mpatches.Patch(facecolor=teal_dark[3], edgecolor=teal_dark[3], label='10 % increase', linewidth=0)
dec_patch = mpatches.Patch(facecolor=teal_light[3], edgecolor=teal_light[3], label='10 % decrease', linewidth=0)

legend = ax3.legend(loc='center',
    handles=[dec_patch,inc_patch],
    numpoints=1,
    frameon=False,
    bbox_to_anchor=(0.15, 0.98, 0.7, 0.07), 
    bbox_transform=ax3.transAxes,
    mode='expand', 
    ncol=2, 
    borderaxespad=-.46,
    prop={'size': 12,},
    handletextpad=0.5,
    handlelength=2.3)

ax3.text(0.00,1.02,'(a)', fontsize=12, transform=ax3.transAxes)

#ELECTRICITY PRICE PLOT
#Prices include grid fee
#Year-specific values [lcop2g;avg el price]
#Maybe we must change the price after operation to get point exactly on the line? But doesnät make sense for different years?
ep_2018 = [182.59,53.62]
ep_2019 = [171.26,47.63]
ep_2020 = [137.97,29.87]
ep_2021 = [218.52,72.46]
opt_case_21 = [194.42,59.95]
pv = [166.16,45]

ep = [0,10,20,30,40,50,60,70,80,90,100]
lcop2g = [81.07,99.98,118.89,137.80,156.71,175.62,194.53,213.44,232.35,251.26,270.17]

# fig, ax1 = plt.subplots()

ax1.plot(ep,lcop2g,color='k')
ax1.plot(ep_2018[1],ep_2018[0], marker='o', label='2018 grid', lw=0, color=teal_light[1])
ax1.plot(ep_2019[1],ep_2019[0], marker='o', label='2019 grid', lw=0, color=teal_light[2])
ax1.plot(ep_2020[1],ep_2020[0], marker='o', label='2020 grid', lw=0, color=teal_dark[4])
ax1.plot(ep_2021[1],ep_2021[0], marker='o', label='2021 grid', lw=0, color=teal_dark[2])
ax1.plot(opt_case_21[1],opt_case_21[0], marker='o', label='2021 optimal', lw=0, color=teal_dark[0])

ax1.set_ylabel('LCOP2G [€/MWh]', fontsize=12)
ax1.set_xlabel('Electricity price [€/MWh]', fontsize=12)
ax1.set_ylim(0,300)
ax1.set_xlim(0,100)
ax1.legend()
ax1.text(0.00,1.03,'(b)', fontsize=12, transform=ax1.transAxes)
ax1.grid(True)

#BY-PRODUCT NPV PLOT
inv_cost_heat = [0,1.32,2.05,2.63,3.95,5.27,7.91,9.23,10.55,13.18]
npv_heat = [1882.13,1631.82,1492.43,1381.52,1131.21,880.91,380.30,129.99,-120.31,-620.92]
inv_cost_o2 = [0,1.30,1.82,2.61,3.49,5.21,6.25,7.82,10.42,15.63]
npv_o2 = [335.89,88.51,-10.44,-158.87,-326.37,-653.63,-851.53,-1148.39,-1643.15,-2632.66]

npv_heat_opt = [1381.52,2.63] #[npv,rel. capex to elx and meth]
npv_o2_opt = [-653.63,5.21] #[npv,rel. capex to elx and meth]

# ax2.axhline(y=0, color='grey', linestyle='-', lw=0.5)
ax2.plot(inv_cost_heat,npv_heat,label='Heat',color=teal_dark[3])
ax2.plot(inv_cost_o2,npv_o2,label='Oxygen',color=teal_light[3])
ax2.plot(npv_o2_opt[1],npv_o2_opt[0],label='Cost-optimal system',color='k',lw=0,marker='o')
ax2.plot(npv_heat_opt[1],npv_heat_opt[0],color='k',lw=0,marker='o')

ax2.set_ylabel('By-product NPV [k€]', fontsize=12)
ax2.set_xlabel('Investment cost [% of electrolyser and methanation CAPEX]', fontsize=12)
ax2.set_ylim(-2000,2000)
ax2.set_xlim(0,12)
ax2.legend()
plt.text(0.00,1.03,'(c)', fontsize=12, transform=ax2.transAxes)
ax2.grid(True)


#HEAT UTILIZATION PLOT
#Assuming a heat utilization factor of 0.8
#Case-specific values [lcop2g;heat utilization;efficicnecy]
hu_base = [194.43,25.44,60.02]
hu_thermo = [191.07,43.10,64.92] #doubling of digester heat demand (Zupancic and Ros, 2003)
hu_sanit = [187.97,60.10,69.45] #hu_thermo + assuming an annual need for sanitation (see "Impact of by-products v1.xlsx")
hu_dh = [181.16,100,80.01]

hu = [0,10,20,30,40,50,60,70,80,90,100] #assuming equal sales during each season, i.e. the average price. Also assuming with with heat CAPEX in zero case.
lcop2g = [199.44,197.61,195.78,193.95,192.13,190.30,188.47,186.64,184.81,182.99,181.16]
eff = [53.45,56.11,58.77,61.44,64.10,66.76,69.42,72.09,74.75,77.41,80.01] #total system efficiency, including oxygen use.

fig, ax1 = plt.subplots()

ax1.plot(hu,lcop2g, color='k')
ax2 = ax1.twinx()
ax2.plot(hu,eff, color='grey')
ax1.plot(hu_base[1],hu_base[0], marker='o', label='WWTP Mesophilic', lw=0, color=teal_light[1])
ax1.plot(hu_thermo[1],hu_thermo[0], marker='o', label='WWTP Thermophilic', lw=0, color=teal_light[3])
ax1.plot(hu_sanit[1],hu_sanit[0], marker='o', label='WWTP Thermophilic + Co-digestion', lw=0, color=teal_dark[3])
ax1.plot(hu_dh[1],hu_dh[0], marker='o', label='District heating', lw=0, color=teal_dark[0])
ax2.plot(hu_base[1],hu_base[2], marker='o', label='WWTP Mesophilic', lw=0, color=teal_light[1])
ax2.plot(hu_thermo[1],hu_thermo[2], marker='o', label='WWTP Thermophilic', lw=0, color=teal_light[3])
ax2.plot(hu_sanit[1],hu_sanit[2], marker='o', label='WWTP Thermophilic + Co-digestion', lw=0, color=teal_dark[3])
ax2.plot(hu_dh[1],hu_dh[2], marker='o', label='District heating', lw=0, color=teal_dark[0])

#Colors
ax2.spines['right'].set_color('grey')
ax2.xaxis.label.set_color('grey')
ax2.tick_params(axis='y', colors='grey')

ax1.set_ylabel('LCOP2G [€/MWh]', fontsize=14)
ax1.set_xlabel('Heat utilisation [%]', fontsize=14)
ax2.set_ylabel('Net energy efficiency [%]', color='grey', fontsize=14)
ax1.set_yticks([0,50,100,150,200,250])
ax1.set_xlim(0,100)
ax2.set_yticks([0,20,40,60,80,100])
plt.legend()
ax1.grid(True)
plt.show()

#OXYGEN UTILIZATION PLOT

fig, (ax1, ax2) = plt.subplots(1,2)
#Aeration energy consumption
#Base case [cons,npv,eff,df,lcop2g]
base_o2 = [1/17,-653.6,60.22,25.8691,194.43]

cons = [0,0.1,0.2,0.3,0.4,0.5,0.6]
npv_o2 = [-989.5,-418.2,154.4,727.8,1301.2,1874.3,2450.2]
eff_increase = [59.57,60.68,61.75,62.83,63.92,64.99,66.07]
df_o2 = [25.8741,25.8647,25.8706,25.8702,25.8689,25.8737,25.8745]
lcop2g = [195.49,193.83,192.28,190.82,189.32,187.73,186.29]

l1 = ax1.plot(cons,npv_o2,label='NPV',color='k')
ax3 = ax1.twinx()
l2 = ax3.plot(cons,eff_increase,label='Net efficiency',color=teal_light[3])
l3 = ax1.plot(base_o2[0],base_o2[1],lw=0,marker='o',label='Base case',color='k')
ax3.plot(base_o2[0],base_o2[2],lw=0,marker='o',color='k')

ax1.set_ylabel('NPV$_{O_2}$ [k€]',color='k',fontsize=14)
ax3.set_ylabel('Net energy efficiency [%]',fontsize=14, color=teal_light[3])
ax1.set_xlabel('Aeration energy consumption [kWh/kgO$_2$]',fontsize=14)
ax1.set_ylim(-1000,4000)
ax1.set_yticks([-1000,0,1000,2000,3000,4000])
ax3.set_ylim(0,125)
ax3.set_yticks([0,25,50,75,100,125])
ax1.set_xlim(0,0.6)
#Legend
# added these three lines
lns = l1+l2+l3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)
plt.text(0.00,1.03,'(a) Aeration energy consumption', fontsize=14, transform=ax1.transAxes)
ax1.grid(True)

#Oxygen transfer rate
otr = [0,50,100,150,200,250,300]
npv_o2 = [-653.6,-485.4,-317.9,-155.9,-12.5,100.1,179.5]
df_o2 = [25.88,38.80,51.67,64.04,74.79,83.17,88.92]
uf_o2 = [100.0,99.99,99.87,99.00,96.39,91.86,85.92]
eff_increase = [60.22,60.54,60.86,61.15,61.44,61.64,61.78]
lcop2g = [194.43,194.01,193.58,193.12,192.74,192.47,192.30]

l1 = ax2.plot(otr,npv_o2,label='NPV',color='k')
ax4 = ax2.twinx()
l2 = ax4.plot(otr,df_o2,label='Demand fulfilment',color=teal_light[3],ls='--')
l3 = ax4.plot(otr,uf_o2,label='Utilisation factor',color=teal_light[3])

ax2.set_ylabel('NPV$_{O_2}$ [k€]',color='k',fontsize=14)
ax4.set_ylabel('Oxygen demand and utilisation [%]',fontsize=14, color=teal_light[3])
ax2.set_xlabel('Oxygen transfer rate increase [%]',fontsize=14)
ax2.set_ylim(-1000,4000)
ax2.set_yticks([-1000,0,1000,2000,3000,4000])
ax4.set_ylim(0,125)
ax4.set_yticks([0,25,50,75,100,125])
ax2.set_xlim(0,300)
#Legend
lns = l1+l2+l3
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs, loc=0)
plt.text(0.00,1.03,'(b) Oxygen transfer rate', fontsize=14, transform=ax2.transAxes)

#Colors
ax3.spines['right'].set_color(teal_light[3])
ax4.spines['right'].set_color(teal_light[3])
ax3.xaxis.label.set_color(teal_light[3])
ax4.xaxis.label.set_color(teal_light[3])
ax3.tick_params(axis='y', colors=teal_light[3])
ax4.tick_params(axis='y', colors=teal_light[3])

ax2.grid(True)
plt.show()


""" OPTIMIZATION 2D PLOTS """

#Data
cb_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\Cost breakdown.xlsx'
cb_data = pd.read_excel(cb_read)

#Determine lowest LCOP2G configuration
min_lcop2g = min(results.iloc[0,:])
index_min = results.columns[results.eq(min_lcop2g).any()]   
meth_use = 5 #the methanation fraction value used in elz vs. h2st plot
h2st_use = 400 #the hydrogen storage value used in elz vs. meth plot

#2D COLOR PLOTS
#defining vectors
# elz_size_vector = [6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12] #MW
# meth_scale_vector = [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2] #ratio to elz
# h2st_size_vector = [0,1,2,3,4,5] #hours
#LCOP2G
#Electrolyzer vs. H2 storage
dx = elz_size_vector[1] - elz_size_vector[0]
dy = h2st_size_vector[1] - h2st_size_vector[0]
y, x = np.mgrid[slice(h2st_size_vector[0], h2st_size_vector[-1] + dy, dy),
                slice(elz_size_vector[0], elz_size_vector[-1] + dx, dx)]

meth_index = meth_scale_vector.index(meth_use)

fig, (ax1, ax2) = plt.subplots(1,2)
Z = np.zeros([len(h2st_size_vector),len(elz_size_vector)])
for x in range(len(h2st_size_vector)):
    for y in range(len(elz_size_vector)):
        Z[x,y] = results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size_vector[y],meth_scale_vector[meth_index],h2st_size_vector[x],wind_size_vector[0]*elz_size_vector[y],pv_size_vector[0]*elz_size_vector[y],bat_size_vector[0])][0]
        lcop2g_test = results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size_vector[y],meth_scale_vector[meth_index],h2st_size_vector[x],wind_size_vector[0]*elz_size_vector[y],pv_size_vector[0]*elz_size_vector[y],bat_size_vector[0])][0]
        if lcop2g_test == min_lcop2g:
            elz_mini = elz_size_vector[y]
            h2st_mini = h2st_size_vector[x]
            
levels = np.linspace(190,250,50)
X = elz_size_vector
Y = h2st_size_vector
elz_h2st = ax1.contourf(X, Y, Z, zorder=1, levels=levels)
fig.colorbar(elz_h2st, ticks=[190,200,210,220,230,240,250])
ax1.plot(elz_mini,h2st_mini, color='k', marker='o', zorder=2)
ax1.set_xlabel('Electrolyser [MW$_{el}$]')
ax1.set_ylabel('Hydrogen storage [hours]')
plt.text(0.01,1.02,'(a) Fixed methanation ratio of {}.'.format(meth_use), fontsize=10, transform=ax1.transAxes)

#Electrolyzer vs. methanation reactor
dx = elz_size_vector[1] - elz_size_vector[0]
dy = meth_scale_vector[1] - meth_scale_vector[0]
y, x = np.mgrid[slice(meth_scale_vector[0], meth_scale_vector[-1] + dy, dy),
                slice(elz_size_vector[0], elz_size_vector[-1] + dx, dx)]

h2st_index = h2st_size_vector.index(h2st_use)
# fig, ax = plt.subplots()
Z = np.zeros([len(meth_scale_vector),len(elz_size_vector)])
for x in range(len(meth_scale_vector)):
    for y in range(len(elz_size_vector)):
        Z[x,y] = results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size_vector[y],meth_scale_vector[x],h2st_size_vector[h2st_index],wind_size_vector[0]*elz_size_vector[y],pv_size_vector[0]*elz_size_vector[y],bat_size_vector[0])][0]
        lcop2g_test = results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size_vector[y],meth_scale_vector[x],h2st_size_vector[h2st_index],wind_size_vector[0]*elz_size_vector[y],pv_size_vector[0]*elz_size_vector[y],bat_size_vector[0])][0]
        if lcop2g_test == min_lcop2g:
            elz_min = elz_size_vector[y]
            meth_min = meth_scale_vector[x]
            
levels = np.linspace(190,250,50)
X = elz_size_vector
Y = meth_scale_vector
elz_meth = ax2.contourf(X, Y, Z, zorder=1, levels=levels)
fig.colorbar(elz_meth, ticks=[190,200,210,220,230,240,250])
ax2.plot(elz_min,meth_min, color='k', marker='o', zorder=2)
ax2.set_xlabel('Electrolyser [MW$_{el}$]')
ax2.set_ylabel('Methanation ratio [-]')
# ax2.set_title('LCOP2G')
plt.text(0.01,1.02,'(b) Fixed hydrogen storage capacity of {} hours.'.format(h2st_use), fontsize=10, transform=ax2.transAxes)

fig.suptitle('LCOP2G [€/MWh$_{CH_4}$]', fontsize=15)

#GAS LOSSES
meth_use = 5 #the methanation fraction value used in elz vs. h2st plot
h2st_use = 300 #the hydrogen storage value used in elz vs. meth plot
#Electrolyzer vs. H2 storage
dx = elz_size_vector[1] - elz_size_vector[0]
dy = h2st_size_vector[1] - h2st_size_vector[0]
y, x = np.mgrid[slice(h2st_size_vector[0], h2st_size_vector[-1] + dy, dy),
                slice(elz_size_vector[0], elz_size_vector[-1] + dx, dx)]

meth_index = meth_scale_vector.index(meth_use)

fig, (ax1, ax2) = plt.subplots(1,2)
Z = np.zeros([len(h2st_size_vector),len(elz_size_vector)])
for x in range(len(h2st_size_vector)):
    for y in range(len(elz_size_vector)):
        Z[x,y] = results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size_vector[y],meth_scale_vector[meth_index],h2st_size_vector[x],wind_size_vector[0]*elz_size_vector[y],pv_size_vector[0]*elz_size_vector[y],bat_size_vector[0])][12]
        lcop2g_test = results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size_vector[y],meth_scale_vector[meth_index],h2st_size_vector[x],wind_size_vector[0]*elz_size_vector[y],pv_size_vector[0]*elz_size_vector[y],bat_size_vector[0])][12]
        if lcop2g_test == min_lcop2g:
            elz_mini = elz_size_vector[y]
            h2st_mini = h2st_size_vector[x]
            
levels = np.linspace(0,18,50)
X = elz_size_vector
Y = h2st_size_vector
elz_h2st = ax1.contourf(X, Y, Z, zorder=1, levels=levels)
fig.colorbar(elz_h2st, ticks=[0,2,4,6,8,10,12,14,16,18])#,20,30])
# ax1.plot(elz_mini,h2st_mini, color='k', marker='o', zorder=2)
ax1.set_xlabel('Electrolyser [MW$_{el}$]')
ax1.set_ylabel('Hydrogen storage [hours]')
plt.text(0.01,1.02,'(a) Fixed methanation ratio of {}.'.format(meth_use), fontsize=10, transform=ax1.transAxes)

#Electrolyzer vs. methanation reactor
dx = elz_size_vector[1] - elz_size_vector[0]
dy = meth_scale_vector[1] - meth_scale_vector[0]
y, x = np.mgrid[slice(meth_scale_vector[0], meth_scale_vector[-1] + dy, dy),
                slice(elz_size_vector[0], elz_size_vector[-1] + dx, dx)]

h2st_index = h2st_size_vector.index(h2st_use)
Z = np.zeros([len(meth_scale_vector),len(elz_size_vector)])
for x in range(len(meth_scale_vector)):
    for y in range(len(elz_size_vector)):
        Z[x,y] = results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size_vector[y],meth_scale_vector[x],h2st_size_vector[h2st_index],wind_size_vector[0]*elz_size_vector[y],pv_size_vector[0]*elz_size_vector[y],bat_size_vector[0])][12]
        lcop2g_test = results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size_vector[y],meth_scale_vector[x],h2st_size_vector[h2st_index],wind_size_vector[0]*elz_size_vector[y],pv_size_vector[0]*elz_size_vector[y],bat_size_vector[0])][12]
        if lcop2g_test == min_lcop2g:
            elz_min = elz_size_vector[y]
            meth_min = meth_scale_vector[x]
            
levels = np.linspace(0,18,50)
X = elz_size_vector
Y = meth_scale_vector
elz_meth = ax2.contourf(X, Y, Z, zorder=1, levels=levels)
fig.colorbar(elz_meth, ticks=[0,2,4,6,8,10,12,14,16,18])
# ax2.plot(elz_min,meth_min, color='k', marker='o', zorder=2)
ax2.set_xlabel('Electrolyser [MW$_{el}$]')
ax2.set_ylabel('Methanation ratio [-]')
plt.text(0.01,1.02,'(b) Fixed hydrogen storage capacity of {} hours.'.format(h2st_use), fontsize=10, transform=ax2.transAxes)

fig.suptitle('Gas loss [%]', fontsize=15)

#OTHERS (16/17 for by-product NPV, 3-5 for efficiency)
meth_use = 5 #the methanation fraction value used in elz vs. h2st plot
h2st_use = 300 #the hydrogen storage value used in elz vs. meth plot
#Electrolyzer vs. H2 storage
dx = elz_size_vector[1] - elz_size_vector[0]
dy = h2st_size_vector[1] - h2st_size_vector[0]
y, x = np.mgrid[slice(h2st_size_vector[0], h2st_size_vector[-1] + dy, dy),
                slice(elz_size_vector[0], elz_size_vector[-1] + dx, dx)]

meth_index = meth_scale_vector.index(meth_use)

fig, (ax1, ax2) = plt.subplots(1,2)
Z = np.zeros([len(h2st_size_vector),len(elz_size_vector)])
for x in range(len(h2st_size_vector)):
    for y in range(len(elz_size_vector)):
        Z[x,y] = results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size_vector[y],meth_scale_vector[meth_index],h2st_size_vector[x],wind_size_vector[0]*elz_size_vector[y],pv_size_vector[0]*elz_size_vector[y],bat_size_vector[0])][6]
        lcop2g_test = results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size_vector[y],meth_scale_vector[meth_index],h2st_size_vector[x],wind_size_vector[0]*elz_size_vector[y],pv_size_vector[0]*elz_size_vector[y],bat_size_vector[0])][6]
        if lcop2g_test == min_lcop2g:
            elz_mini = elz_size_vector[y]
            h2st_mini = h2st_size_vector[x]
            
# levels = np.linspace(0,5,50)
X = elz_size_vector
Y = h2st_size_vector
elz_h2st = ax1.contourf(X, Y, Z, zorder=1)#, levels=levels)
fig.colorbar(elz_h2st)#, ticks=[0,2,4,6,8,10,12])#,20,30])
# ax1.plot(elz_mini,h2st_mini, color='k', marker='o', zorder=2)
ax1.set_xlabel('Electrolyser [MW$_{el}$]')
ax1.set_ylabel('Hydrogen storage [hours]')
plt.text(0.01,1.02,'(a) Fixed methanation ratio of {}.'.format(meth_use), fontsize=10, transform=ax1.transAxes)

#Electrolyzer vs. methanation reactor
dx = elz_size_vector[1] - elz_size_vector[0]
dy = meth_scale_vector[1] - meth_scale_vector[0]
y, x = np.mgrid[slice(meth_scale_vector[0], meth_scale_vector[-1] + dy, dy),
                slice(elz_size_vector[0], elz_size_vector[-1] + dx, dx)]

h2st_index = h2st_size_vector.index(h2st_use)
Z = np.zeros([len(meth_scale_vector),len(elz_size_vector)])
for x in range(len(meth_scale_vector)):
    for y in range(len(elz_size_vector)):
        Z[x,y] = results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size_vector[y],meth_scale_vector[x],h2st_size_vector[h2st_index],wind_size_vector[0]*elz_size_vector[y],pv_size_vector[0]*elz_size_vector[y],bat_size_vector[0])][6]
        lcop2g_test = results['E: {}, M: {}, S: {}, W: {}, P: {}, B: {}'.format(elz_size_vector[y],meth_scale_vector[x],h2st_size_vector[h2st_index],wind_size_vector[0]*elz_size_vector[y],pv_size_vector[0]*elz_size_vector[y],bat_size_vector[0])][6]
        if lcop2g_test == min_lcop2g:
            elz_min = elz_size_vector[y]
            meth_min = meth_scale_vector[x]
            
# levels = np.linspace(0,2,50)
X = elz_size_vector
Y = meth_scale_vector
elz_meth = ax2.contourf(X, Y, Z, zorder=1)#, levels=levels)
fig.colorbar(elz_meth)#, ticks=[0,10,20,30,40,50])
# ax2.plot(elz_min,meth_min, color='k', marker='o', zorder=2)
ax2.set_xlabel('Electrolyser [MW$_{el}$]')
ax2.set_ylabel('Methanation ratio [-]')
plt.text(0.01,1.02,'(b) Fixed hydrogen storage capacity of {} hours.'.format(h2st_use), fontsize=10, transform=ax2.transAxes)

fig.suptitle('System efficiency [%]', fontsize=15)





""" COST BREAKDOWN PLOT """

#Data
cb_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\Cost breakdown.xlsx'
cb_data = pd.read_excel(cb_read)
labels = list(cb_data.iloc[0:14,0])
gold_dark = sns.dark_palette("gold",4)
gold_light = sns.light_palette("gold",4)
teal_dark = sns.dark_palette("teal",6)
teal_light = sns.light_palette("teal",6)
pal = gold_dark
colors = [teal_dark[1],teal_dark[3],teal_dark[5],teal_light[3],teal_light[1],'cornflowerblue','indianred','mediumpurple','0.5','orange',gold_dark[2],gold_dark[3],gold_light[2],gold_light[1]]
# colors = [teal_dark[1],teal_dark[2],teal_dark[3],teal_dark[4],'seagreen','mediumseagreen','indianred','mediumpurple','silver','orange',gold_dark[2],gold_dark[3],gold_light[2],gold_light[1]]
edgecolor = colors
hatches = ['','','','','','','','','','','','///','...','xxx']
# colors = sns.color_palette("Set1")
#Plot
elz_cap = [6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12]
elz_cost = list(cb_data.iloc[0,1:14])
stack_cost = list(cb_data.iloc[1,1:14])
h2st_cost = list(cb_data.iloc[2,1:14])
meth_cost = list(cb_data.iloc[3,1:14])
comp_cost = list(cb_data.iloc[4,1:14])
water_cost = list(cb_data.iloc[5,1:14])
heat_cost = list(cb_data.iloc[6,1:14])
o2_cost = list(cb_data.iloc[7,1:14])
inst_cost = list(cb_data.iloc[8,1:14])
flare_cost = list(cb_data.iloc[9,1:14])
grid_cost = list(cb_data.iloc[10,1:14])
pv_cost = list(cb_data.iloc[11,1:14])
wind_cost = list(cb_data.iloc[12,1:14])
curt_cost = list(cb_data.iloc[13,1:14])
o2inc_cost = list(cb_data.iloc[14,1:14])
heatinc_cost = list(cb_data.iloc[15,1:14])

fig, ax = plt.subplots()
stacks = ax.stackplot(elz_cap,elz_cost,stack_cost,h2st_cost,meth_cost,comp_cost,water_cost,heat_cost,o2_cost,inst_cost,flare_cost,grid_cost,pv_cost,wind_cost,curt_cost,labels=labels,colors=colors)#,edgecolors=edgecolor)#,hatch=hatch)
plt.axvline(x=8.5,ls='--',color='k')
# plt.legend(loc='upper left')
# hatches=["\\", "//","+"]
# for stack, hatch in zip(stacks, hatches):
#     stack.set_hatch(hatch)
legend = ax.legend(loc='center',
    # handles=[lcop2g_patch,curt_patch,lcop2g_pv0,curt_pv0,lcop2g_pv25,curt_pv25,lcop2g_pv50,curt_pv50,lcop2g_pv75,curt_pv75,lcop2g_pv100,curt_pv100],
    numpoints=1,
    frameon=False,
    bbox_to_anchor=(0.1, 0.98, 5.5, 2.5), 
    bbox_transform=ax1.transAxes,
    mode='expand', 
    ncol=5, 
    borderaxespad=-.46,
    prop={'size': 9,},
    handletextpad=0.5,
    handlelength=2.3)    

plt.xlabel('Electrolyser capacity [MW]')
plt.ylabel('Share of LCOP2G [%]')
plt.xlim(6,12)
plt.ylim(0,100)
plt.show()

""" ELECTRICITY SUPPLY PLOTS """

teal_dark = sns.dark_palette("teal",5)
teal_light = sns.light_palette("teal",5)
#Data
lcop2g_curt = [[218.5,208.6,199.2,195.3,197.8,204.1,211.5],
               [218.5,210.4,202.0,196.1,194.5,196.9,201.1],
               [218.5,212.2,205.6,199.5,195.6,194.4,195.6],
               [218.5,213.9,209.2,204.7,201.1,199.4,199.2],
               [218.5,215.7,212.8,210.2,208.5,209.0,210.5]]
lcop2g_nocurt = [[218.5,208.5,198.8,192.1,189.0,185.9,185.3],
                 [218.5,210.3,201.9,194.9,190.1,187.9,186.9],
                 [218.5,212.1,205.5,199.3,194.2,190.5,187.1],
                 [218.5,213.9,209.2,204.6,200.6,197.2,194.7],
                 [218.5,215.7,212.8,210.1,207.9,206.6,205.5]]
curt_cost = [list(np.array(lcop2g_curt[0]) - np.array(lcop2g_nocurt[0])),
             list(np.array(lcop2g_curt[1]) - np.array(lcop2g_nocurt[1])),
             list(np.array(lcop2g_curt[2]) - np.array(lcop2g_nocurt[2])),
             list(np.array(lcop2g_curt[3]) - np.array(lcop2g_nocurt[3])),
             list(np.array(lcop2g_curt[4]) - np.array(lcop2g_nocurt[4]))]
curt_amount = [[0,0.26,1.1,6.82,14.93,24.7,30.99],
               [0,0.27,0.39,2.86,8.65,14.76,20.53],
               [0,0.31,0.23,0.49,3.49,7.9,14.22],
               [0,0.22,0.31,0.28,1.62,5.73,10.09],
               [0,0.24,0.18,0.37,2.94,9.66,17.0]]
res_fraction = [[0,21.72,43.16,57.53,66.14,73.23,76.81],
                [0,17.82,37.72,52.26,61.72,67.98,72.59],
                [0,13.92,27.89,41.77,54.05,60.72,67.95],
                [0,9.46,20.06,30.12,39.58,47.42,51.11],
                [0,5.8,12.31,18.42,23.88,26.16,28.83]]
grid_fraction = [list(np.ones(7)*100 - np.array(res_fraction[0])),
             list(np.ones(7)*100 - np.array(res_fraction[1])),
             list(np.ones(7)*100 - np.array(res_fraction[2])),
             list(np.ones(7)*100 - np.array(res_fraction[3])),
             list(np.ones(7)*100 - np.array(res_fraction[4]))]
aefs = [[34.01,29.26,24.69,21.88,20.87,19.44,20.39],
        [34.01,31.75,29.27,27.81,27.31,27.87,29.71],
        [34.01,34.29,34.66,35.08,35.86,36.95,38.45],
        [34.01,36.73,39.69,42.63,45.74,48.61,49.88],
        [34.01,39.06,44.68,50.09,55.09,57.32,59.71]]
mefs = [[1571.68,1230.91,896.34,673.95,542.18,431.55,378.68],
        [1571.68,1292.36,982.09,757.72,613.02,518.37,450.24],
        [1571.68,1354.42,1137.61,923.6,735.23,635.09,525.24],
        [1571.68,1425.05,1260.96,1107.06,964.87,846.04,791.64],
        [1571.68,1482.30,1383.09,1290.84,1212.59,1180.7,1141.13]]

#Parameters
oversizing = [0,0.5,1,1.5,2,2.5,3]
pv_fraction = [0,0.25,0.5,0.75,1]
curt_var = ['No curtailment', 'Curtailment']

#Color palette
pal = sns.dark_palette("teal",8)
pal2 = sns.dark_palette("orange",9)
# pal2 = sns.light_palette("teal",8)
# set width of bars
barWidth = 0.12
# barWidth1 = barWidth - 0.03
 
# set heights of bars
lcop2g1 = lcop2g_nocurt[0]
lcop2g2 = lcop2g_nocurt[1]
lcop2g3 = lcop2g_nocurt[2]
lcop2g4 = lcop2g_nocurt[3]
lcop2g5 = lcop2g_nocurt[4]
curt1 = curt_cost[0]
curt2 = curt_cost[1]
curt3 = curt_cost[2]
curt4 = curt_cost[3]
curt5 = curt_cost[4]
 
# Set position of bar on X axis
r1 = np.arange(len(lcop2g1))
r2 = [x + barWidth + 0.02 for x in r1]
r3 = [x + barWidth + 0.02 for x in r2]
r4 = [x + barWidth + 0.02 for x in r3]
r5 = [x + barWidth + 0.02 for x in r4]

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.grid(axis='y')
ax1.set_axisbelow(True)
# Make the plot
#Left plot
#no curtailment
ax1.bar(r1, lcop2g1, color=pal[3], width=barWidth)#, edgecolor='white', label='0 % PV')
ax1.bar(r2, lcop2g2, color=pal[4], width=barWidth)#, edgecolor='white', label='25 % PV')
ax1.bar(r3, lcop2g3, color=pal[5], width=barWidth)#, edgecolor='white', label='50 % PV')
ax1.bar(r4, lcop2g4, color=pal[6], width=barWidth)#, edgecolor='white', label='75 % PV')
ax1.bar(r5, lcop2g5, color=pal[7], width=barWidth)#, edgecolor='white', label='100 % PV')
#add curtailment
ax1.bar(r1, curt1, bottom=lcop2g1, color='0.3', width=barWidth)#, edgecolor='white')
ax1.bar(r2, curt2, bottom=lcop2g2, color='0.4', width=barWidth)#, edgecolor='white')
ax1.bar(r3, curt3, bottom=lcop2g3, color='0.5', width=barWidth)#, edgecolor='white')
ax1.bar(r4, curt4, bottom=lcop2g4, color='0.6', width=barWidth)#, edgecolor='white')
ax1.bar(r5, curt5, bottom=lcop2g5, color='0.7', width=barWidth)#, edgecolor='white')
 
# Add xticks on the middle of the group bars
ax1.set_xlabel('Oversizing factor')
ax1.set_xticks([r + 2*barWidth for r in range(len(lcop2g1))], ['0', '0.5', '1', '1.5', '2', '2.5', '3'])
ax1.set_ylabel('LCOP2G [€/MWh]')
ax1.set_ylim(0,250)


ax3 = ax1.twinx()
ax3.plot(r1, curt_amount[0], color='k', lw=0, marker='o', markersize=4)
ax3.plot(r2, curt_amount[1], color='k', lw=0, marker='o', markersize=4)
ax3.plot(r3, curt_amount[2], color='k', lw=0, marker='o', markersize=4)
ax3.plot(r4, curt_amount[3], color='k', lw=0, marker='o', markersize=4)
ax3.plot(r5, curt_amount[4], color='k', lw=0, marker='o', markersize=4)

ax3.set_ylabel('Curtailed RES generation [%]')
ax3.set_ylim(0,100)

plt.text(0.01,0.97,'(a)', fontsize=10, transform=ax1.transAxes)

#Right plot
#Energy composition
ax2.grid(axis='y')
ax2.set_axisbelow(True)

ax2.bar(r1, res_fraction[0], color=pal[3], width=barWidth)#, edgecolor='white', label='0 % PV')
ax2.bar(r2, res_fraction[1], color=pal[4], width=barWidth)#, edgecolor='white', label='25 % PV')
ax2.bar(r3, res_fraction[2], color=pal[5], width=barWidth)#, edgecolor='white', label='50 % PV')
ax2.bar(r4, res_fraction[3], color=pal[6], width=barWidth)#, edgecolor='white', label='75 % PV')
ax2.bar(r5, res_fraction[4], color=pal[7], width=barWidth)#, edgecolor='white', label='100 % PV')

ax2.bar(r1, grid_fraction[0], bottom=res_fraction[0], color='0.3', width=barWidth)#, edgecolor='white')
ax2.bar(r2, grid_fraction[1], bottom=res_fraction[1], color='0.4', width=barWidth)#, edgecolor='white')
ax2.bar(r3, grid_fraction[2], bottom=res_fraction[2], color='0.5', width=barWidth)#, edgecolor='white')
ax2.bar(r4, grid_fraction[3], bottom=res_fraction[3], color='0.6', width=barWidth)#, edgecolor='white')
ax2.bar(r5, grid_fraction[4], bottom=res_fraction[4], color='0.7', width=barWidth)#, edgecolor='white')

#Legend
lcop2g_patch = lines.Line2D([0], [0], color='white', lw=0, label='Without curtailment:')#mpatches.Patch(facecolor='white', edgecolor='white', label='Without curtailment:', linewidth=0)
lcop2g_pv0 = mpatches.Patch(facecolor=pal[3], edgecolor=pal[3], label='0 % PV', linewidth=0)
lcop2g_pv25 = mpatches.Patch(facecolor=pal[4], edgecolor=pal[4], label='25 % PV', linewidth=0)
lcop2g_pv50 = mpatches.Patch(facecolor=pal[5], edgecolor=pal[5], label='50 % PV', linewidth=0)
lcop2g_pv75 = mpatches.Patch(facecolor=pal[6], edgecolor=pal[6], label='75 % PV', linewidth=0)
lcop2g_pv100 = mpatches.Patch(facecolor=pal[7], edgecolor=pal[7], label='100 % PV', linewidth=0)
curt_patch = lines.Line2D([0], [0], color='white', lw=0, label='Curtailment cost:')#mpatches.Patch(facecolor='white', edgecolor='white', label='Curtailment cost:', linewidth=0)
curt_pv0 = mpatches.Patch(facecolor='0.3', edgecolor='0.3', label='0 % PV', linewidth=0)
curt_pv25 = mpatches.Patch(facecolor='0.4', edgecolor='0.4', label='25 % PV', linewidth=0)
curt_pv50 = mpatches.Patch(facecolor='0.5', edgecolor='0.5', label='50 % PV', linewidth=0)
curt_pv75 = mpatches.Patch(facecolor='0.6', edgecolor='0.6', label='75 % PV', linewidth=0)
curt_pv100 = mpatches.Patch(facecolor='0.7', edgecolor='0.7', label='100 % PV', linewidth=0)
curt_dot_patch = lines.Line2D([0], [0], color='white', lw=0, label='Curtailed electricity:')
curt_dot = lines.Line2D([0], [0], color='k', marker='o', lw=0, label='')

legend = ax1.legend(loc='center',
    handles=[lcop2g_patch,curt_patch,lcop2g_pv0,curt_pv0,lcop2g_pv25,curt_pv25,lcop2g_pv50,curt_pv50,lcop2g_pv75,curt_pv75,lcop2g_pv100,curt_pv100],
    numpoints=1,
    frameon=False,
    bbox_to_anchor=(-0.022, 0.98, 1, 0.17), 
    bbox_transform=ax1.transAxes,
    mode='expand', 
    ncol=6, 
    borderaxespad=-.46,
    prop={'size': 9,},
    handletextpad=0.5,
    handlelength=2.3)

legend = ax3.legend(loc='center',
    handles=[curt_dot_patch, curt_dot],
    numpoints=1,
    frameon=False,
    bbox_to_anchor=(-0.022, 0.98, 0.3, 0.075), 
    bbox_transform=ax1.transAxes,
    mode='expand', 
    ncol=2, 
    borderaxespad=-.46,
    prop={'size': 9,},
    handletextpad=0.5,
    handlelength=2.3)

#Emissions
ax4 = ax2.twinx()
#AEF
ax4.plot(r1, aefs[0], color='k', lw=0, marker='o', markersize=4)
ax4.plot(r2, aefs[1], color='k', lw=0, marker='o', markersize=4)
ax4.plot(r3, aefs[2], color='k', lw=0, marker='o', markersize=4)
ax4.plot(r4, aefs[3], color='k', lw=0, marker='o', markersize=4)
ax4.plot(r5, aefs[4], color='k', lw=0, marker='o', markersize=4)
#MEF
ax4.plot(r1, mefs[0], color='k', lw=0, marker='D', markersize=3)
ax4.plot(r2, mefs[1], color='k', lw=0, marker='D', markersize=3)
ax4.plot(r3, mefs[2], color='k', lw=0, marker='D', markersize=3)
ax4.plot(r4, mefs[3], color='k', lw=0, marker='D', markersize=3)
ax4.plot(r5, mefs[4], color='k', lw=0, marker='D', markersize=3)

ax2.set_ylabel('Electricity composition [%]')
ax2.set_ylim(0,100)
ax2.set_xticks([r + 2*barWidth for r in range(len(res_fraction[0]))], ['0', '0.5', '1', '1.5', '2', '2.5', '3'])
ax2.set_xlabel('Oversizing factor')
ax4.set_ylabel('Net specific emissions [kgCO$_2$eq/MWh]')
ax4.set_ylim(0,2000)
ax4.set_yticks([0,400,800,1200,1600,2000])

plt.text(0.01,0.97,'(b)', fontsize=10, transform=ax2.transAxes)

#Legend
res_patch = lines.Line2D([0], [0], color='white', lw=0, label='RES share:')#mpatches.Patch(facecolor='white', edgecolor='white', label='Without curtailment:', linewidth=0)
res_pv0 = mpatches.Patch(facecolor=pal[3], edgecolor=pal[3], label='0 % PV', linewidth=0)
res_pv25 = mpatches.Patch(facecolor=pal[4], edgecolor=pal[4], label='25 % PV', linewidth=0)
res_pv50 = mpatches.Patch(facecolor=pal[5], edgecolor=pal[5], label='50 % PV', linewidth=0)
res_pv75 = mpatches.Patch(facecolor=pal[6], edgecolor=pal[6], label='75 % PV', linewidth=0)
res_pv100 = mpatches.Patch(facecolor=pal[7], edgecolor=pal[7], label='100 % PV', linewidth=0)
grid_patch = lines.Line2D([0], [0], color='white', lw=0, label='Grid share:')#mpatches.Patch(facecolor='white', edgecolor='white', label='Curtailment cost:', linewidth=0)
grid_pv0 = mpatches.Patch(facecolor='0.3', edgecolor='0.3', label='0 % PV', linewidth=0)
grid_pv25 = mpatches.Patch(facecolor='0.4', edgecolor='0.4', label='25 % PV', linewidth=0)
grid_pv50 = mpatches.Patch(facecolor='0.5', edgecolor='0.5', label='50 % PV', linewidth=0)
grid_pv75 = mpatches.Patch(facecolor='0.6', edgecolor='0.6', label='75 % PV', linewidth=0)
grid_pv100 = mpatches.Patch(facecolor='0.7', edgecolor='0.7', label='100 % PV', linewidth=0)
ems_dot_patch = lines.Line2D([0], [0], color='white', lw=0, label='Emissions:')
aef_dot = lines.Line2D([0], [0], color='k', lw=0, marker='o', label='Average emissions')
mef_dot = lines.Line2D([0], [0], color='k', lw=0, marker='D', label='Marginal emissions')

legend = ax2.legend(loc='center',
    handles=[res_patch,grid_patch,res_pv0,grid_pv0,res_pv25,grid_pv25,res_pv50,grid_pv50,res_pv75,grid_pv75,res_pv100,grid_pv100],
    numpoints=1,
    frameon=False,
    bbox_to_anchor=(1.2, 0.98, 0.9, 0.17), 
    bbox_transform=ax1.transAxes,
    mode='expand', 
    ncol=6, 
    borderaxespad=-.46,
    prop={'size': 9,},
    handletextpad=0.5,
    handlelength=2.3)

legend = ax4.legend(loc='center',
    handles=[ems_dot_patch,aef_dot,mef_dot],
    numpoints=1,
    frameon=False,
    bbox_to_anchor=(1.2, 0.98, 0.6, 0.075), 
    bbox_transform=ax1.transAxes,
    mode='expand', 
    ncol=3, 
    borderaxespad=-.46,
    prop={'size': 9,},
    handletextpad=0.5,
    handlelength=2.3)

# Create legend & Show graphic
# plt.legend()
plt.show()


""" BY-PRODUCT KPI IMPACT """

# a = ['P2G system','Heat utilisation','Oxygen utilisation']
# b = [196.36,-3.68,1.74]

# my_plot = waterfall_chart.plot(a, b)

fig = make_subplots(rows=2, cols=2, start_cell="top-left")

fig.add_trace(go.Waterfall(
    name = "20", orientation = "v",
    measure = ["relative", "relative", "relative", "total"],
    x = ["P2G system", "Heat utilisation", "Oxygen utilisation", "LCOP2G"],
    textposition = "outside",
    text = ["196.4", "-3.7", "+1.7", "194.4"],
    y = [196.4, -3.7, 1.7, 0],
    connector = {"line":{"color":"rgb(63, 63, 63)"}},
    width = [0.6,0.6,0.6,0.6]
), row=1, col=1)

fig.add_trace(go.Waterfall(
    name = "20", orientation = "v",
    measure = ["relative", "relative", "relative", "total"],
    x = ["P2G system", "Heat utilisation", "Oxygen utilisation", "Net energy efficiency"],
    textposition = "outside",
    text = ["52.8", "+6.8", "+0.6", "60.2"],
    y = [52.81, 6.77, 0.64, 0],
    connector = {"line":{"color":"rgb(63, 63, 63)"}},
    width = [0.6,0.6,0.6,0.6]
), row=1, col=2)

fig.add_trace(go.Waterfall(
    name = "20", orientation = "v",
    measure = ["relative", "relative", "relative", "total"],
    x = ["P2G system", "Heat utilisation", "Oxygen utilisation", "Average net specific emissions"],
    textposition = "outside",
    text = ["51.6", "-14.4", "-0.3", "36.9"],
    y = [51.63, -14.37, -0.31, 0],
    connector = {"line":{"color":"rgb(63, 63, 63)"}},
    width = [0.6,0.6,0.6,0.6]
), row=2, col=1)

fig.add_trace(go.Waterfall(
    name = "20", orientation = "v",
    measure = ["relative", "relative", "relative", "total"],
    x = ["P2G system", "Heat utilisation", "Oxygen utilisation", "Marginal net specific emissions"],
    textposition = "outside",
    text = ["649.1", "-3.8", "-10.2", "635.1"],
    y = [649.09, -3.85, -10.15, 0],
    connector = {"line":{"color":"rgb(63, 63, 63)"}},
    width = [0.6,0.6,0.6,0.6]
), row=2, col=2)

#Axis properties
fig.update_yaxes(title_text="LCOP2G [€/MWh]", range=[0,250], row=1, col=1)
fig.update_yaxes(title_text="Efficiency [%]", range=[0, 100], row=1, col=2)
fig.update_yaxes(title_text="Average emissions [kgCO2eq/MWh]", range=[0,80], row=2, col=1)
fig.update_yaxes(title_text="Marginal emissions [kgCO2eq/MWh]", range=[0,800], row=2, col=2)

fig.update_layout(height=1000, width=1700)

# fig.write_image("images/yourfile.svg") 

import plotly.io as pio
pio.renderers.default = "browser"
fig.show()

#OLD VERSION
#Import data
# local_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\RES local.xlsx'
# local = pd.read_excel(local_read)
# constr_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\RES constrained.xlsx'
# constr = pd.read_excel(constr_read)
# unconstr_read = r'C:\Users\enls0001\Anaconda3\Lib\site-packages\P2G\Data\RES unconstrained.xlsx'
# unconstr = pd.read_excel(unconstr_read)
# local = local.drop(columns=['KPIs'])
# constr = constr.drop(columns=['KPIs'])
# unconstr = unconstr.drop(columns=['KPIs'])
# local.iloc[13,:] = local.iloc[13,:] * 100
# constr.iloc[13,:] = constr.iloc[13,:] * 100
# unconstr.iloc[13,:] = unconstr.iloc[13,:] * 100 

# # #Functions
# # def get_flipped(y_data, y_model):
# #     flipped = y_model - y_data
# #     flipped[flipped > 0] = 0
# #     return flipped

# # def flipped_resid(pars, x, y):
# #     """
# #     For every iteration, everything above the currently proposed
# #     curve is going to be mirrored down, so that the next iterations
# #     is going to progressively shift downwards.
# #     """
# #     y_model = model(x, *pars)
# #     flipped = get_flipped(y, y_model)
# #     resid = np.square(y + flipped - y_model)
# #     #print pars, resid.sum() # uncomment to check the iteration parameters
# #     return np.nan_to_num(resid)

# # # plotting the mock data
# # plt.plot(dset[0], dset[1], '.', alpha=0.2, label = 'Test data')

# # # mask bad data (we accidentaly generated some NaN values)
# # gmask = np.isfinite(dset[1])
# # dset = dset[np.vstack([gmask, gmask])].reshape((2, -1))

# # from scipy.optimize import leastsq
# # guesses =[100, 100, 0]
# # fit_pars, flag = leastsq(func = flipped_resid, x0 = guesses,
# #                          args = (dset[0], dset[1]))
# # # plot the fit:
# # y_fit = model(x_data, *fit_pars)
# # y_guess = model(x_data, *guesses)
# # plt.plot(x_data, y_fit, 'r-', zorder = 0.9, label = 'Edge')
# # plt.plot(x_data, y_guess, 'g-', zorder = 0.9, label = 'Guess')
# # plt.legend(loc = 'lower left')
# # plt.show()

# #Pareto front (MSP vs. MEF)
# fig, ax1 = plt.subplots()
# #Scatter
# # ax1.plot(local.iloc[13,:], local.iloc[2,:], ls='none', marker='o', color='gold') #LCOP2G vs. RES
# # ax1.plot(constr.iloc[13,:], constr.iloc[2,:], ls='none', marker='o', color='blue') #LCOP2G vs. RES
# # ax1.plot(unconstr.iloc[13,:], unconstr.iloc[2,:], ls='none', marker='o', color='k') #LCOP2G vs. RES

# #Pareto
# #Local scenario
# local_sorted = sorted([[local.iloc[13,i], local.iloc[2,i]] for i in range(len(local.iloc[2,:]))], reverse=False)
# local_sorted_arr = np.array(local_sorted)
# local_edge = [local_sorted[0]]
# i = 0
# for pair in local_sorted[1:]:
#     i = i+1
#     if pair[1] < local_edge[-1][1]:
#         local_edge.append(pair)
#     if i < len(local_sorted)-1:
#         if pair[1] < min(local_sorted_arr[i+1:,1]):
#             local_edge.append(pair)
# edge_lcop2g_local = [pair[1] for pair in local_edge]
# edge_res_local = [pair[0] for pair in local_edge]
# ax1.plot(edge_res_local, edge_lcop2g_local, ls='-', marker='o', color='gold', zorder=3, label='Local')
# #Constrained PPA scenario
# constr_sorted = sorted([[constr.iloc[13,i], constr.iloc[2,i]] for i in range(len(constr.iloc[2,:]))], reverse=False)
# constr_sorted_arr = np.array(constr_sorted)
# constr_edge = [constr_sorted[0]]
# i = 0
# for pair in constr_sorted[1:]:
#     i = i+1
#     if pair[1] < constr_edge[-1][1]:
#         constr_edge.append(pair)
#     if i < len(constr_sorted)-1:
#         if pair[1] < min(constr_sorted_arr[i+1:,1]):
#             constr_edge.append(pair)
# edge_lcop2g_constr = [pair[1] for pair in constr_edge]
# edge_res_constr = [pair[0] for pair in constr_edge]
# ax1.plot(edge_res_constr, edge_lcop2g_constr, ls='-', marker='o', color='steelblue', zorder=2, label='Constrained PPA')
# #Unconstrained PPA scenario
# unconstr_sorted = sorted([[unconstr.iloc[13,i], unconstr.iloc[2,i]] for i in range(len(unconstr.iloc[2,:]))], reverse=False)
# unconstr_sorted_arr = np.array(unconstr_sorted)
# unconstr_edge = [unconstr_sorted[0]]
# i = 0
# for pair in unconstr_sorted[1:]:
#     i = i+1
#     if pair[1] < unconstr_edge[-1][1]:
#         unconstr_edge.append(pair)
#     if i < len(unconstr_sorted)-1:
#         if pair[1] < min(unconstr_sorted_arr[i+1:,1]):
#             unconstr_edge.append(pair)
# edge_lcop2g_unconstr = [pair[1] for pair in unconstr_edge]
# edge_res_unconstr = [pair[0] for pair in unconstr_edge]
# ax1.plot(edge_res_unconstr, edge_lcop2g_unconstr, ls='-', marker='o', color='k', zorder=1, label='Unconstrained PPA')
# ax1.set_ylabel('LCOP2G [€/MWh]')
# ax1.set_xlabel('RES fraction [%]')
# ax1.set_ylim(0,600)
# ax1.set_xlim(0,100)
# plt.legend()
# plt.show()



