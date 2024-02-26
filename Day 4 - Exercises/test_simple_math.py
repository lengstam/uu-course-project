# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:28:09 2024

@author: enls0001
"""

import simple_math

def test_simple_add():
    assert simple_math.simple_add(1,2) == 3
    
def test_simple_sub():
    assert simple_math.simple_sub(3,1) == 2
    
def test_simple_mult():
    assert simple_math.simple_mult(2,3) == 6
    
def test_simple_div():
    assert simple_math.simple_div(6,3) == 2
    
def test_poly_first():
    assert simple_math.poly_first(3,1,2) == 7
    
def test_poly_second():
    assert simple_math.poly_second(3,1,2,2) == 25