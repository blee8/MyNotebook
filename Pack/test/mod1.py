
# mod1.py
def add(a, b):
    return a+b

def sub(a, b):
    return a-b

print(f'__name__= : {__name__}')

if __name__ == "__main__":
    print(f'absolute import __name__= "mod1" : {__name__}')

    print(add(1, 4))
    print(sub(4, 2))