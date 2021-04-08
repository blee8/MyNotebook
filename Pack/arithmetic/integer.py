# integer.py

#import util # use when excute in the scipt --> absolute path
from . import util  # use when excute in arith.py --> relative path

print ('top in integer ', __name__)

def div(a, b):
    """
    >>> div(3, 2)
    1
    """

    result = a // b
    util.log("dividing...")

    return result

#if __name__ == "__main__" or __name__ == "integer":
#    import util
#else:
#    from . import util