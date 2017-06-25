import mxnet as mx
import LayerFactory
import numpy as np
from PIL import Image
import logging
logging.getLogger().setLevel(logging.DEBUG)

wholeTrain_data = np.load("train.npy")
wholeTrain_lebel = np.load("train_label.npy")
wholeTrain_data = mx.nd.array(wholeTrain_data)
wholeTrain_lebel = mx.nd.array(wholeTrain_lebel)

#separate  all train data into training and cross vildlidation and define data iterations

train_data = wholeTrain_data[0:50000]
train_label = wholeTrain_lebel[0:50000]

cv_data = wholeTrain_data[50000:]
cv_label = wholeTrain_lebel[50000:]

train_data_iter = mx.io.NDArrayIter(data=train_data,label=train_label,batch_size=100,shuffle=True)

cv_data_iter = mx.io.NDArrayIter(data=cv_data,label=cv_label,batch_size=100)

#CNN layer structure
net  = mx.sym.Variable('data')
conv1 = LayerFactory.convFactory(data = net,num_filter=64,kernel=(3,3),stride=(1,1),name='',suffix='')
conv2 = LayerFactory.convFactory(data = conv1,num_filter=128,kernel=(3,3),stride=(1,1))
flaten = mx.sym.flatten(data = conv2)
fc1 = LayerFactory.fcFactory(data = flaten,num_hidden=500,name = '',sufiix='')
fc1_relu = mx.sym.Activation(data=fc1, act_type="relu")
fc2 = mx.sym.FullyConnected(data=fc1_relu, num_hidden=10)
out = mx.sym.SoftmaxOutput(data = fc2,name = 'softmax')

mnist_conv_model = mx.mod.Module(symbol=out,context=mx.gpu(),data_names=['data'],label_names=['softmax_label'])

mnist_conv_model.bind(data_shapes=train_data_iter.provide_data, label_shapes=train_data_iter.provide_label)

mnist_conv_model.init_params(initializer=mx.init.Uniform(scale=0.07))

mnist_conv_model.init_optimizer(optimizer='sgd',
                                optimizer_params=(('learning_rate',0.01),))

metric = mx.metric.create('acc')

#train 10 epochs and after each epoch , do cross validation

for epoch in range(10):
    train_data_iter.reset()
    metric.reset()
    for batch in train_data_iter:
        mnist_conv_model.forward(batch,is_train=True)
        mnist_conv_model.update_metric(metric,batch.label)
        mnist_conv_model.backward()
        mnist_conv_model.update()
    print("epoch %d, Training %s" %(epoch,metric.get()))
    #mnist_conv_model.save_params('./parms/mnistConvparm_epoch_%d'%epoch)
    mnist_conv_model.save_checkpoint(prefix='mymxnetConv_',epoch=epoch, save_optimizer_states=True)
    cv_test_iter = mx.io.NDArrayIter(data=cv_data,label= None,batch_size=100)
    predict = mnist_conv_model.predict(cv_test_iter)
    accuracy = mx.metric.Accuracy()
    score = mnist_conv_model.score(cv_data_iter,['mse','acc'])
    print('epoch %d cv test accuracy '%epoch , score)










