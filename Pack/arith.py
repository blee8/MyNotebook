# arith.py

#from arithmetic import integer, decimal   # . 으로 시작하지 않았으므로 절대경로, OK

from Pack.arithmetic import *


print("3 / 2 is " + str(decimal.div(3, 2)))

print("3 / 2 is " + str(integer.div(3, 2)))

if __name__ == "__main__"  :
    print ('absolute path in arith.py', __name__, "__main__")

'''
PEP8

Explicit relative imports are an acceptable alternative to absolute imports.
Implicit relative imports should never be used and have been removed in Python3.

import 경로가 . 으로 시작 -->  상대경로

Python의 창시자, Guido 
The only use case seems to be running scripts that happen to be living inside 
a module’s directory, which I’ve always seen as an antipattern. To make me change
my mind you’d have to convince me that it isn’t.


상대경로 import를 포함한 python 모듈은 단독 실행이 불가능하다.
패키지 내에서 절대경로 import는 쉽지 않다.
패키지 안에 들어있는 모듈을 단독으로 실행하는 것이 python 철학에는 맞지 않는 모양인 것 같은데, 
모듈만 실행하고 싶을 때가 종종 있다. 당장 위처럼 doctest를 모듈 단위로 돌리고 싶을 때가 있다.

https://blog.potados.com/dev/python3-import/
https://stackoverflow.com/questions/16981921/relative-imports-in-python-3
'''
