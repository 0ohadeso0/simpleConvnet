
import mnistLoad as mst
import numpy as np
from PIL import Image

#change to you own path
data_path = '/home/hades/MxnetProject/mnist/'
train_data_name = 'train-images-idx3-ubyte'
train_label_name = 'train-labels-idx1-ubyte'
test_data_name = 't10k-images-idx3-ubyte'
test_label_name = 't10k-labels-idx1-ubyte'


#load Trian images
train_data_reader = mst.Load_mnist_data(data_path+train_data_name)
print data_path+train_data_name
trainImage_matrix = train_data_reader.getImage()
print trainImage_matrix
train_img = trainImage_matrix.reshape((trainImage_matrix.shape[0],1,28,28))
#img = Image.fromarray(np.uint8(train_img[0]))
#img.show()
np.save('train.npy',train_img)


train_label_reader = mst.Load_mnist_data(data_path+train_label_name)
train_label_vec = train_label_reader.getlable()
np.save('train_label.npy',train_label_vec)

#save Test images
test_data_reader = mst.Load_mnist_data(data_path+test_data_name)
testImage_matrix = test_data_reader.getImage()
test_img = testImage_matrix.reshape((testImage_matrix.shape[0],1,28,28))
np.save('test.npy',test_img)


test_label_reader = mst.Load_mnist_data(data_path+test_label_name)
test_label_vec = test_label_reader.getlable()
np.save('test_label.npy',test_label_vec)