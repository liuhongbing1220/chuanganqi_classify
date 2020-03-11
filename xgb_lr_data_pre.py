#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 21:06:43 2019

@author: liuhongbing
"""
import numpy as np
import os
import pandas as pd

train_X_path = "/Users/liuhongbing/Documents/tensorflow/data/UCI_HAR_Dataset/train/Inertial Signals"


X_trainS1_x = pd.DataFrame(np.loadtxt(os.path.join(train_X_path, "body_acc_x_train.txt")))
X_trainS1_y = pd.DataFrame(np.loadtxt(os.path.join(train_X_path, "body_acc_y_train.txt")))
X_trainS1_z = pd.DataFrame(np.loadtxt(os.path.join(train_X_path, "body_acc_z_train.txt")))
x_columns = ['x_'+ str(i) for i in range(1, 129)]
y_columns = ['y_'+ str(i) for i in range(1, 129)]
z_columns = ['z_'+ str(i) for i in range(1, 129)]
X_trainS1_x.columns = x_columns
X_trainS1_y.columns = y_columns
X_trainS1_z.columns = z_columns


label = pd.Series(np.loadtxt(os.path.join(train_X_path, "y_train.txt")))


data = pd.concat([X_trainS1_x,X_trainS1_y,X_trainS1_z], axis=1)




train_x, test_x, train_y, test_y = train_test_split(data, label, test_size = 0.3, random_state=0)
