#from .multiplication import multiplication # relative path 상대경로

from Pack.calculator import multiplication # absolute path 절대경로

def add_and_multiply(a, b):
    return multiplication.multiply(a, b) + (a + b)