
import numpy as np


class Person:    # 클래스


    def __init__(self, name, age, address):
        self.name = name
        self.age = age
        self.address = address


    def greeting(self):
        print('안녕하세요. 저는 {0}입니다.'.format(self.name))



class Person2:    # 클래스


    def __init__(self, name, age, address):
        self.name = name
        self.age = age
        self.address = address


    def greeting(self):
        print('안녕하세요. 저는 {0}입니다.'.format(self.name))
        a = np.zeros(10)
        b = np.zeros(5)
        print('np check a= {1}'.format(a, b))
        print(f'np check a= {a}')
