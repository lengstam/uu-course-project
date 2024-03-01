# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 16:10:29 2024

@author: enls0001
"""

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as lines
import plotly.graph_objects as go
import numpy as np
import math
import plotly.io as pio


def byprods(
        ):
    """
    Returns a normalized bar chart showing the relative impact of heat and oxygen utilization on
    LCOP2G, average and marginal net specific emissions, and net energy efficiency.

    """

def dispatch(electrolyzer, elz_size_degr, h2_flow, elz_h2_max, h2_storage, spot_price, biogas_flow, \
             h2_demand, h2_used, heat_demand_tot, meth_heat, usable_heat, elz_heat, o2_wwtp, o2_flow):
    """
    Returns a weekly sample of system operation. Fig(a) shows hydrogen production, demand and storage.
    Fig(b) shows wind and solar generation as well as spot price. Fig(c) shows by-product generation and demand.
    
    """
    #colors
    teal_dark = sns.dark_palette("teal",5)
    teal_light = sns.light_palette("teal",5)
    orange_dark = sns.dark_palette("orange",5)
    
    x1 = 2208 # Starting hour
    x2 = x1+(24*7) # One week later
    d1 = x1 - x1%24 + 24 # Start of the first new day
    elzload_plot = electrolyzer[x1:x2]*100/(elz_size_degr*1000)
    h2prod_plot = (h2_flow[x1:x2]*2.02/1000)*100/elz_h2_max
    elz_plot = electrolyzer[x1:x2]
    h2st_plot = np.array(h2_storage[x1-1:x2-1])
    ep_plot = spot_price[x1:x2]
    bg_plot = biogas_flow[x1:x2,1]
    h2dem_plot = (h2_demand[x1:x2]*2.02/1000)*100 / elz_h2_max
    h2use_plot = h2_used[x1:x2]*100/elz_h2_max
    htdem_plot = np.array(np.maximum(heat_demand_tot[x1:x2]-(meth_heat[x1:x2]*usable_heat),0))
    htprod_plot = usable_heat*elz_heat[x1:x2]
    o2dem_plot = np.array(o2_wwtp[x1:x2] * 32 / 1000)
    o2prod_plot = np.array(o2_flow[x1:x2] * 32 / 1000)
    x = range(0,x2-x1)
    
    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
    l1 = ax1.plot(x,h2prod_plot, color=teal_dark[4], label='H$_2$ production')
    l2 = ax1.fill_between(x,h2st_plot, color=teal_light[1], label='Hydrogen storage')
    ax3 = ax1.twinx()
    l3 = ax3.plot(x,ep_plot, color=orange_dark[4], label='Electricity price')
    l4 = ax1.plot(x,h2dem_plot, color=teal_dark[2], label='H$_2$ demand', ls='--')
    ax2.set_xlabel('Hour')
    ax1.set_ylabel('Load [%]', color='k')
    ax3.set_ylabel('El. price [€/MWh]', color=orange_dark[4])
    
    # Indicating days
    for i in range(math.floor((x2-x1)/24)):
        if d1 == x1:
            if i % 2 == 1:
                ax1.axvspan(d1-x1+(24*i),d1-x1+((24*(i+1))), facecolor='0.2', alpha=0.2, zorder=-1)
        elif d1 != x1:
            if i % 2 == 0:
                ax1.axvspan(d1-x1+(24*i),d1-x1+((24*(i+1))), facecolor='0.2', alpha=0.2, zorder=-1)
    ax1.set_ylim(0,120)
    ax3.set_ylim(0,250)
    ax1.set_xlim(0,x2-x1-1)
    plt.text(0.00,1.03,'(a)', fontsize=10, transform=ax1.transAxes)
    plt.text(0.00,-0.17,'(b)', fontsize=10, transform=ax1.transAxes)
    
    # Axis colors
    ax3.spines['right'].set_color(orange_dark[4])
    ax3.xaxis.label.set_color(orange_dark[4])
    ax3.tick_params(axis='y', colors=orange_dark[4])
    
    # By-products
    l2 = ax2.plot(x,htdem_plot, color='indianred', ls='--', label='Heat demand')
    l3 = ax2.plot(x,htprod_plot, color='indianred', label='Heat production')
    ax4 = ax2.twinx()
    l5 = ax4.plot(x,o2dem_plot, color='mediumpurple', ls='--', label='O2 demand')
    l6 = ax4.plot(x,o2prod_plot, color='mediumpurple', label='O2 production')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Heat [kW]', color='indianred')
    ax4.set_ylabel('Oxygen [kg/h]', color='mediumpurple')
    lns = l2+l3+l5+l6
    labs = [l.get_label() for l in lns]
    
    # Indicating days
    for i in range(math.floor((x2-x1)/24)):
        if d1 == x1:
            if i % 2 == 1:
                ax2.axvspan(d1-x1+(24*i),d1-x1+((24*(i+1))), facecolor='0.2', alpha=0.2, zorder=-1)
        elif d1 != x1:
            if i % 2 == 0:
                ax2.axvspan(d1-x1+(24*i),d1-x1+((24*(i+1))), facecolor='0.2', alpha=0.2, zorder=-1)
    ax4.set_ylim(0,6000)
    ax2.set_ylim(0,2000)
    ax2.set_xlim(0,x2-x1-1)
    
    # Axis colors
    ax4.spines['right'].set_color('mediumpurple')
    ax4.yaxis.label.set_color('mediumpurple')
    ax4.tick_params(axis='y', colors='mediumpurple')
    ax2.spines['left'].set_color('indianred')
    ax2.yaxis.label.set_color('indianred')
    ax2.tick_params(axis='y', colors='indianred')
    
    # Legend 1
    h2_prod_patch = lines.Line2D([0], [0], color=teal_dark[4], lw=3, label='H$_2$ production')
    h2_dem_patch = lines.Line2D([0], [0], color=teal_dark[2], lw=3, ls='--', label='H$_2$ demand')
    h2st_patch = mpatches.Patch(facecolor=teal_light[1], edgecolor=teal_light[2], label='H$_2$ storage', linewidth=0)
    el_patch = lines.Line2D([0], [0], color=orange_dark[4], lw=3, label='Electricity price')

    legend = ax1.legend(loc='center',
        handles=[h2_prod_patch,h2_dem_patch,el_patch,h2st_patch],
        numpoints=1,
        frameon=False,
        bbox_to_anchor=(0.1, 0.98, 0.8, 0.15), 
        bbox_transform=ax3.transAxes,
        mode='expand', 
        ncol=4, 
        borderaxespad=-.46,
        prop={'size': 9,},
        handletextpad=0.5,
        handlelength=2.3)
    
    # Legend 2
    o2_prod_patch = lines.Line2D([0], [0], color='mediumpurple', lw=3, label='O$_2$ production')
    o2_dem_patch = lines.Line2D([0], [0], color='mediumpurple', lw=3, ls='--', label='O$_2$ demand')
    heat_prod_patch = lines.Line2D([0], [0], color='indianred', lw=3, label='Heat production')
    heat_dem_patch = lines.Line2D([0], [0], color='indianred', lw=3, ls='--', label='Heat demand')

    legend = ax2.legend(loc='center',
        handles=[o2_prod_patch,o2_dem_patch,heat_prod_patch,heat_dem_patch],
        numpoints=1,
        frameon=False,
        bbox_to_anchor=(0.07, 0.98, 0.87, -2.25), 
        bbox_transform=ax3.transAxes,
        mode='expand', 
        ncol=4, 
        borderaxespad=-.46,
        prop={'size': 9,},
        handletextpad=0.5,
        handlelength=2.3)
    
    plt.show()
    
    
def sankey(grid_use, pv_use, wind_use, bat_in_pv, bat_in_wind, electrolyzer, biogas_in, ch4_mol, bg_comp_power, h2_used, h2_kg, \
           flared, meth_el, elz_heat_nonnet, meth_heat, ch4_total, heat_wwtp, o2_energy_savings, aerator_savings, o2_loss):
    """
    Returns a Sankey diagrom of system energy flows.
    
    """
    # Defining flows
    gr_el = grid_use.sum()/1000
    pv_el = (pv_use+bat_in_pv).sum()/1000
    wi_el = (wind_use+bat_in_wind).sum()/1000
    # el_bat = sum(bat_in_wind+bat_in_pv)/1000 + 1000
    # bat_el = el_bat*bat_eff
    # bat_ls= el_bat-bat_el
    el_h2 = electrolyzer.sum()/1000
    bg_cm = (biogas_in[0,:].sum()*ch4_mol/1000)
    el_cm = bg_comp_power.sum()/1000
    h2_ch4 = h2_used.sum()*h2_kg/1000
    bg_ls = flared.sum()*ch4_mol/1000
    cm_ch4 = bg_cm+el_cm
    el_ch4 = meth_el.sum()/1000
    h2_ht = elz_heat_nonnet.sum()/1000
    ch4_ht = meth_heat.sum()/1000
    ch4_ch4 = ch4_total
    ht_ww = heat_wwtp.sum()/1000
    ht_ls = h2_ht+ch4_ht-ht_ww
    o2_ww = o2_energy_savings.sum()/1000
    o2_ls = aerator_savings * o2_loss * 32 / (1000*1000)
    h2_ls = el_h2-h2_ch4-h2_ht
    ch4_ls = h2_ch4+cm_ch4+el_ch4-ch4_ch4-ch4_ht
    
    # Plot
    pio.renderers.default='browser'
    opacity = 0.4
    fig = go.Figure(data=[go.Sankey(
    node = dict(
       pad = 15,
       thickness = 10,
       line = dict(color = "black", width = 0.5),
       label = ["Grid", "PV", "Wind", "Biogas", "Electricity", "Battery", "Electrolysis", "Compression", "Oxygen", "Methanation", "Heat", "Losses", "WWTP", "Unused heat", "Methane"],
       color = "gray"
     ),
     link = dict(
       source = [0, 1, 2, 4, 5, 5, 4, 3, 4, 6, 7, 4, 6, 9, 9, 10, 10, 8, 8, 3, 6, 9], # indices correspond to labels.
       target = [4, 4, 4, 5, 4, 11, 6, 7, 7, 9, 9, 9, 10, 10, 14, 12, 11, 12, 11, 11, 11, 11],
       value = [gr_el,pv_el,wi_el,el_bat,bat_el,bat_ls,el_h2,bg_cm,el_cm,h2_ch4,cm_ch4,el_ch4,h2_ht,ch4_ht,ch4_ch4,ht_ww,ht_ls,o2_ww,o2_ls,bg_ls,h2_ls,ch4_ls],
       # label =  [],
       color =  ['gold','gold','gold','gold','gold','gold','gold','seagreen','gold','steelblue','seagreen','gold','indianred','indianred','coral','indianred','indianred','purple','purple','seagreen','steelblue','coral']
       ))])

    fig.update_layout(title_text="P2G energy flows", font_size=10)
    fig.show()
    
    
    
    
    