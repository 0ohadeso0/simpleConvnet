import mxnet as mx
import mxpth
def convFactory(data,num_filter,kernel,stride = (1,1),pad = (0,0) ,name = None, suffix = ''):
    conv = mx.sym.Convolution(data = data,num_filter = num_filter,kernel = kernel,stride = stride,pad = pad,name = 'conv_%s%s'%(name,suffix))
    bn = mx.sym.BatchNorm(data=conv, name='bn_%s%s'%(name, suffix))
    act = mx.sym.Activation(data=bn, act_type='relu', name='relu_%s%s'%(name,suffix))
    pooling = mx.sym.Pooling(data = act, pool_type = 'max', kernel = (2,2), stride = stride, name = 'pooling_%s%s'%(name,suffix))
    return pooling

def fcFactory(data, num_hidden, name = None, sufiix = ''):
    fc = mx.sym.FullyConnected(data = data, name = 'FC_%s%s'%(name,sufiix) , num_hidden = num_hidden)
    return fc
