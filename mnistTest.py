import mxnet as mx
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import sys
from collections import namedtuple
#for 1 picture test, wo define a simple dataiter
Batch = namedtuple('Batch', ['data'])
#for testdata set



#test on testset
def test():
    wholeTest_data = mx.nd.array(np.load('test.npy'))
    wholeTest_label = mx.nd.array(np.load('test_label.npy'))
    test_iter = mx.io.NDArrayIter(data=wholeTest_data, label=wholeTest_label, batch_size=100)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix='mymxnetConv_', epoch=9)
    mod = mx.mod.Module(symbol=sym, context=mx.cpu())
    mod.bind(for_training=False, data_shapes=test_iter.provide_data, label_shapes=test_iter.provide_label)
    mod.set_params(arg_params, aux_params, allow_missing=True)
    test_prob = mod.predict(test_iter)
    acc = mx.metric.Accuracy()
    mod.score(test_iter, acc)
    print(acc)
    print test_prob.shape


#one picture test
def predict(show = False):
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix='mymxnetConv_', epoch=9)
    mod = mx.mod.Module(symbol=sym, context=mx.cpu())
    mod.bind(for_training=False, data_shapes=[('data',(1,1,28,28))], label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)
    img = cv2.imread('./mnist/mnist_images/mnist_test_train_36.bmp')
    # compute the predict probabilities
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    test_img = np.array(img)
    test_img = test_img.reshape((1,1,28,28))
    print test_img.shape
    mod.forward(Batch([mx.nd.array(test_img)]))
    prob = mod.get_outputs()[0].asnumpy()
    prob1 = mod.get_outputs()
    print (prob1)
    # print the top-5
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    print a[0]
    if show:
        cv2.imshow('a',img)
        cv2.waitKey(0)


test()
#pridict one picture
# if you want to show the image at the same time, otherwise set show = False
#predict(show=True)