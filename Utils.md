
# 通用处理函数


```python
import mxnet as mx
from mxnet import ndarray as nd


def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx


def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad
```
