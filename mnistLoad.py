#coding=utf-8
import numpy as np
import struct
from PIL import Image

tag = '>'  #使用大端读取
twoBytes = 'II'  #读取数据格式是两个整数
fourBytes = 'IIII'  # 读取的数据格式是四个整数
pictureBytes = '784B'  # 读取的图片的数据格式是784个字节，28*28
lableByte = '1B'  # 标签是1个字节
msb_twoBytes = tag + twoBytes
msb_fourBytes = tag + fourBytes
msb_pictureBytes = tag + pictureBytes
msb_lableByte = tag + lableByte

class Load_mnist_data():


    def __init__(self,data_path):
        self.filename = data_path


    def getImage(self):
        binfile = open(self.filename, 'rb') #以二进制读取的方式打开文件
        buf = binfile.read() #获取文件内容缓存区
        binfile.close()
        index = 0 #偏移量
        numMagic, numImgs, numRows, numCols = struct.unpack_from(msb_fourBytes, buf, index)
        index += struct.calcsize(fourBytes)
        images = []
        for i in xrange(numImgs):
            imgVal  = struct.unpack_from(msb_pictureBytes, buf, index)
            index += struct.calcsize(pictureBytes)

            imgVal  = list(imgVal)
            #for j in range(len(imgVal)):
            #   if imgVal[j] > 1:
            #       imgVal[j] = 1
            images.append(imgVal)
        return np.array(images)

    def getlable(self) :
        binfile = open(self.filename, 'rb')
        buf = binfile.read() #获取文件内容缓存区
        binfile.close()
        index = 0 #偏移量
        numMagic, numItems = struct.unpack_from(msb_twoBytes,buf, index)
        index += struct.calcsize(twoBytes)
        labels = []
        for i in range(numItems):
            value = struct.unpack_from(msb_lableByte, buf, index)
            index += struct.calcsize(lableByte)
            labels.append(value[0]) #获取值的内容
        return np.array(labels)