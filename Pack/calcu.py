#from Pack.calculator.add_and_multiply import add_and_multiply  # absolute import   __name__ == __main__

from calculator.add_and_multiply import add_and_multiply # relative import   ---> error !!!

if __name__ == '__main__':
    print(add_and_multiply(1, 2))