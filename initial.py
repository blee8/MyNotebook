%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()

def xyplot(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.asnumpy(), y_vals.asnumpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')

#--------
%load_ext watermark
#%watermark -a "Sebastian Raschka" -u -d -p numpy,pandas,matplotlib
%watermark -a "blee" -u -d -p numpy,pandas,matplotlib

#---
from IPython.display import Image
Image(filename='./images/02_04.png', width=600)

#---
from mxnet import nd
from mxnet import autograd
