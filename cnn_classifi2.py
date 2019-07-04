#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 17:14:53 2019

@author: liuhongbing
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 加载数据集
def read_data(file_path):
    column_names = ['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    data = pd.read_csv(file_path, header=None, names=column_names)
    data['z-axis'] = data['z-axis'].apply(lambda x : str(x).split(";")[0])
    data['z-axis'] = data['z-axis'].astype('float32')

    return data

# 数据标准化
def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    print('mu:',mu)
    sigma = np.std(dataset, axis=0)
    print('sigma:',sigma)
    return (dataset - mu) / sigma


# 创建时间窗口，90 × 50ms，也就是 4.5 秒，每次前进 45 条记录，半重叠的方式。
def windows(data, size):
    start = 0
    while start < data.count():
        yield start, start + size
        start += (size / 2)

# 创建输入数据，每一组数据包含 x, y, z 三个轴的 90 条连续记录，
# 用 `stats.mode` 方法获取这 90 条记录中出现次数最多的行为
# 作为该组行为的标签，这里有待商榷，其实可以完全使用同一种行为的数据记录
# 来创建一组数据用于输入的。
        

def segment_signal(data, window_size=128):
    segments = np.empty((0, window_size, 3))
    labels = np.empty((0))
    print (len(data['timestamp']))
    count = 0
    for (start, end) in windows(data['timestamp'], window_size):
        print (count)
        start = int(start)
        end = int(end)
        count += 1
        x = data["x-axis"][start:end]
        y = data["y-axis"][start:end]
        z = data["z-axis"][start:end]
        
        if (len(data['timestamp'][start:end]) == window_size):
            segments = np.vstack([segments, np.dstack([x, y, z])])
            labels = np.append(labels, stats.mode(data["activity"][start:end])[0][0])
        else:
            return segments, labels

    return segments, labels



def get_train_test():
    root = "/Users/liuhongbing/Documents/tensorflow/data/WISDM_ar_v1.1/"
    dataset2 = read_data(root +'WISDM_ar_v1.1_raw.txt')
    dataset2.fillna(0, inplace=True)
    print("dataset2:", len(dataset2))
    dataset = dataset2[:300000]
    print("dataset:", len(dataset))

    dataset['x-axis'] = feature_normalize(dataset['x-axis'])
    
    dataset['y-axis'] = feature_normalize(dataset['y-axis'])
    
    dataset['z-axis'] = feature_normalize(dataset['z-axis'])
    
    
    segments, labels = segment_signal(dataset)
    
    
    labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)
    
    # 创建输入
    ## [batch_size, height, width, chanles]
    
    reshaped_segments = segments.reshape(len(segments), 1, 128, 3)
    
    train_x, test_x,train_y, test_y = train_test_split(reshaped_segments, labels, test_size = 0.3)
    
    return train_x, test_x,train_y, test_y



class CNN_classify:
    
    def __init__(self):
        # 定义输入数据的维度和标签个数
        self.input_height = 1
        self.input_width = 128
        self.num_labels = 4  # 6
        self.num_channels = 3
        
        ## width_conv
        self.kernel_size = 30
#        self.channel_multiplier = 60
        self.depth = 60
        
        # 隐藏层神经元个数
        self.num_hidden = 1000
        self.learning_rate = 0.0001
        # 降低 cost 的迭代次数
        self.training_epochs = 8
        
    
        # 初始化神经网络参数
    def _weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    # 初始化神经网络参数
    def _bias_variable(self,shape):
        initial = tf.constant(0.0, shape=shape)
        return tf.Variable(initial)
    
    # 执行卷积操作
    def _depthwise_conv2d(self,x, W):
        
        #tf.nn.depthwise_conv2d(input,filter,strides,padding,rate=None,name=None,data_format=None)
        #input：是4维Tensor，具有[batch, height, width, in_channels]
        #filter：是4维Tensor，具有[filter_height, filter_width, in_channels, channel_multiplier]
        return tf.nn.depthwise_conv2d(input = x, filter = W, strides = [1, 1, 1, 1], padding='VALID')
    
    # 为输入数据的每个 channel 执行一维卷积，并输出到 ReLU 激活函数
    def _apply_depthwise_conv(self, x, kernel_size, num_channels, depth):
        
        ## conv_filter_trainable = [filter_height, filter_width, in_channels, channel_multiplier]
        weights = self._weight_variable([1, kernel_size, num_channels, depth])
        biases = self._bias_variable([depth * num_channels])
        return tf.nn.relu(tf.add(self._depthwise_conv2d(x, weights), biases))
    
    # 在卷积层输出进行一维 max pooling
    def _apply_max_pool(self,x, kernel_size, stride_size):
        
        ## max_pool(value, ksize, strides, padding, name=None)
        ## ksize = [batch, height, width, chanels]
        ## strides = [1, height, width, 1]
        return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1],
                              strides=[1, 1, stride_size, 1], padding='VALID')


    
    def model(self, X):
        
        c = self._apply_depthwise_conv(X, self.kernel_size, self.num_channels, self.depth)
        p = self._apply_max_pool(c,20,2)
        
        print("C_1:", c)
        print("p_1:",p)
        c = self._apply_depthwise_conv(p,6,self.depth*self.num_channels,self.depth//10)
        print("C_2:",c)
        shape = c.get_shape().as_list()
        c_flat = tf.reshape(c, [-1, shape[1] * shape[2] * shape[3]])
        
        print("C_flat:", c_flat)
        print("shape:", shape)
        f_weights_l1 = self._weight_variable([shape[1] * shape[2] * self.depth * self.num_channels * (self.depth//10), self.num_hidden])
        
        print("f_wights_l1:", shape[1] * shape[2] * self.depth * self.num_channels * (self.depth//10))
        f_biases_l1 = self._bias_variable([self.num_hidden])
        f = tf.nn.tanh(tf.add(tf.matmul(c_flat, f_weights_l1),f_biases_l1))
        
        out_weights = self._weight_variable([self.num_hidden, self.num_labels])
        out_biases = self._bias_variable([self.num_labels])
        y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)
        
        return y_
        
    def train(self, X, Y):
        
        y_ = self.model(X)
        loss = -tf.reduce_sum(Y * tf.log(y_))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(loss)
        correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        cost_history = np.empty(shape=[1], dtype=float)
        
        return cost_history, optimizer, accuracy,loss, y_

        
    

training_epochs = 10
batch_size = 64

train_x, test_x,train_y, test_y = get_train_test()
ccc = CNN_classify()

X = tf.placeholder(tf.float32, shape=[None,ccc.input_height, ccc.input_width,ccc.num_channels])
Y = tf.placeholder(tf.float32, shape=[None,ccc.num_labels])
cost_history, optimizer, accuracy, loss, y_ = ccc.train(X,Y)
total_batchs = train_x.shape[0] // batch_size

with tf.Session() as session:
    tf.initialize_all_variables().run()
    # 开始迭代
    for epoch in range(training_epochs):
        for b in range(total_batchs):
            offset = (b * batch_size) % (train_y.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size), :, :, :]
            batch_y = train_y[offset:(offset + batch_size), :]
            print(batch_x.shape)
            print(batch_y.shape)
            _, c = session.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
            cost_history = np.append(cost_history, c)
        print("Epoch {}: Training Loss = {}, Training Accuracy = {}".format(
            epoch, c, session.run(accuracy, feed_dict={X: train_x, Y: train_y})))
    y_p = tf.argmax(y_, 1)
    y_true = np.argmax(test_y, 1)
    final_acc, y_pred = session.run([accuracy, y_p], feed_dict={X: test_x, Y: test_y})
    print("Testing Accuracy: {}".format(final_acc))
    temp_y_true = np.unique(y_true)
    temp_y_pred = np.unique(y_pred)
    np.save("y_true", y_true)
    np.save("y_pred", y_pred)
    print("temp_y_true", temp_y_true)
    print( "temp_y_pred", temp_y_pred)
    # 计算模型的 metrics
    print( "Precision", precision_score(y_true.tolist(), y_pred.tolist(), average='weighted'))
    print( "Recall", recall_score(y_true, y_pred, average='weighted'))
    print( "f1_score", f1_score(y_true, y_pred, average='weighted'))
    print( "confusion_matrix")
    print( confusion_matrix(y_true, y_pred))
    
    
    
    
    
        


