"""
A collection of simple math operations
"""

def simple_add(a,b):
    """
    The sum of two numbers.
    
    Parameters
    ----------
    a: first number to add
        Input number 'a'. This can be any number, or an array.
    b: second number to add
        Input number 'b'. This can be any other number, or the same.
        
    Returns
    -------
    Depends on input.
        Returns the sum of inputs 'a' and 'b'.
    """
    return a+b

def simple_sub(a,b):
    return a-b

def simple_mult(a,b):
    return a*b

def simple_div(a,b):
    return a/b

def poly_first(x, a0, a1):
    return a0 + a1*x

def poly_second(x, a0, a1, a2):
    return poly_first(x, a0, a1) + a2*(x**2)

# Feel free to expand this list with more interesting mathematical operations...
# .....
