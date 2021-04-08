from Pack.arithmetic import *



__all__ = ['decimal', 'integer']

#__all__ = ['decimal' ] # error, imported all, but except integer

'''
특정 디렉터리의 모듈을 *를 이용하여 import할 때에는 해당 디렉터리의 __init__.py 파일에 
__all__이라는 변수를 설정하고 import할 수 있는 모듈을 정의해 주어야 합니다.
__all__로 정의하지 않으면 인식되지 않습니다.
https://nesoy.github.io/articles/2018-07/Python-init-all

패키지(Packages)는 도트(.)를 사용하여 파이썬 모듈을 계층적(디렉터리 구조)으로 관리할 수 
있게 해준다. 예를 들어 모듈 이름이 A.B인 경우에 A는 패키지 이름이 되고 B는 A 패키지의 B모듈이 된다.

※ python3.3 버전부터는 __init__.py 파일이 없어도 패키지로 인식한다(PEP 420). 하지만 
하위 버전 호환을 위해 __init__.py 파일을 생성하는 것이 안전한 방법이다.

 __all__이 의미하는 것은 sound 디렉터리에서 * 기호를 사용하여 import할 경우 
 이곳에 정의된 echo 모듈만 import된다는 의미이다.
https://wikidocs.net/1418


'''
