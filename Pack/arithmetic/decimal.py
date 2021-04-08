# decimal.py

#import util

#print (__name__)

def div(a, b):
    """
    >>> div(3, 2)
    1.5
    """

    result = a / b
    util.log("dividing...")
    util.log(result)

    return result

if __name__ == "__main__" or __name__ == "decimal":
    # Do something only when directly executed as a script.
    import util             # absolute import 절대경로

    print('절대경로 in decimal',__name__, "__main__")

    import doctest
    doctest.testmod()


else:
    from . import util      # relative import 상대경로, when called by arith.py should use relative import

    print('상대경로 in decimal', __name__, "__main__")