#import matplotlib as mpl
#import matplotlib.pyplot as plt
#import t_Mod
#from t_Mod.plots_t import plot_curve, eval, add_curve   # relative path, 함수까지 import

#from t_Mod.Class_t import WinGen, WindowGenerator, plot, plot_batch, plot_xy
#from t_Mod.compile import compile_and_fit, compile_2, fit_2
# 도트 연산자(.)를 사용해서 import a.b.c처럼 import할 때 가장 마지막 항목인 c는 반드시 모듈
# 또는 패키지여야만 한다.



from t_Mod import *   # absolute path, import 함수
# 특정 디렉터리의 모듈을 *를 사용하여 import할 때에는 다음과 같이 해당 디렉터리의 __init__.py 파일에
# __all__ 변수를 설정하고 import할 수 있는 모듈을 정의해 주어야 한다.


#from t_Mod.plots_t import *  # absolute path, import 함수
#from t_Mod.Class_t import *
#from t_Mod.compile import *
# ※ 착각하기 쉬운데 from game.sound.echo import * 는 __all__과 상관없이 무조건 import된다.
# 이렇게 __all__과 상관없이 무조건 import되는 경우는 from a.b.c import * 에서 from의
# 마지막 항목인 c가 모듈인 경우이다.

# 모듈에서 함수를 사용하는 경우 상대경로로 불러와야 한다.
# 절대경로는 함수 자체를 실행하는 경우, 예 : doctest 같은,경우에만 적용
# 원칙 : 모듈 내  함수는 모듈 내에서 실행, 상대경로 사용 !!!! 2021.04.01 by blee


#from t_Mod import plots_t    # absolute path, 모듈까지 import
#from t_Mod import Class_t
#from t_Mod import compile



__all__ = ['plots_t', 'compile', 'Class_t'  ]
