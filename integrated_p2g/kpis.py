# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 14:37:45 2023

@author: Linus Engstam
"""

import pandas as pd
import numpy as np

def lcoe(
    opex,
    capex,
    stack,
    dr,
    lt,
    ch4,
    stack_reps,
    rep_years,
) -> pd.DataFrame:
    """Returns the LCOE for current simulation"""
    
    #Define total parameters
    full_opex = 0
    full_ch4 = 0
    full_stack = 0
    #Define stack replacement parameters
    if stack_reps > 0:
        stack_cost = stack / stack_reps
    else:
        stack_cost = 0

    #Total OPEX and CH4 production
    for y in range(lt):
        full_opex = full_opex + (opex / pow(1 + (dr/100),y))
        full_ch4 = full_ch4 + (ch4 / pow(1 + (dr/100),y))
    #Discounting stack replacements
    for i in range(stack_reps):
        full_stack = full_stack + (stack_cost / pow(1 + (dr/100),rep_years[i]))
    #LCOE
    lcoe = (capex + full_opex + full_stack) / full_ch4
    
    return lcoe


def npv(
    opex,
    income,
    capex,
    stack,
    dr,
    lt,
    stack_reps,
    rep_years,
) -> pd.DataFrame:
    """Returns the NPV for current simulation"""
    
    annual_flow = 0
    total_stack = 0
    if stack_reps > 0:
        stack_cost = stack / stack_reps
    else:
        stack_cost = 0
    for y in range(lt):
        annual_flow = annual_flow + ((income-opex) / pow(1 + (dr/100),y))
        
    for i in range(stack_reps):
        total_stack = total_stack + (stack_cost / pow(1 + (dr/100),rep_years[i]))
        
    npv = annual_flow - capex - total_stack
    
    return npv
    

def msp(
    opex,
    capex,
    stack,
    dr,
    lt,
    ch4,
    stack_reps,
    rep_years,
) -> pd.DataFrame:
    """Returns the MSP for current simulation"""
    
    annual_flow = 0
    total_ch4 = 0
    total_stack = 0
    if stack_reps > 0:
        stack_cost = stack / stack_reps
    else:
        stack_cost = 0
    for y in range(lt):
        annual_flow = annual_flow + ((opex) / pow(1 + (dr/100),y))
        total_ch4 = total_ch4 + ((ch4) / pow(1 + (dr/100),y))
    for i in range(stack_reps):
        total_stack = total_stack + (stack_cost / pow(1 + (dr/100),rep_years[i]))
        
    total_costs = annual_flow + capex + total_stack
    msp = total_costs / total_ch4 #[€/MWh CH4]
    
    return msp
    
    
    
    
    
    
    
    