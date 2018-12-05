# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#展示MNIST数据集前二十张图片及对应标签
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)
save_dir='MNIST_data/raw/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for i in range(20):
    image_array=mnist.train.images[i,:]
    image_array=image_array.reshape(28,28)
    filename=save_dir+'mnist_train_%d.jpg' % i
    scipy.misc.toimage(image_array,cmin=0.0,cmax=1.0).save(filename)

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)
for i in range(20):
    one_hot_label=mnist.train.labels[i,:]
    label=np.argmax(one_hot_label)
    print('mnist_train_%d.jpg label:%d'%(i,label))