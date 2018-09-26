#  Code challenge of Sai Peri
#  The task is completed in Python using a Neural Network with Keras library and Tensorflow for backend 

import random
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import rmsprop
from keras.regularizers import l2

data_dir = 'F:\Jobs\MoruLabs\splice.data'     # reading the data 

# The idea is to read the data file as two columns as Labels and Features where first column is the Labels 
# and third column is Features ( Second column is ignored as the donor data is not considered ) 
 
with open(data_dir + 'data_orig', 'r') as f:
    data_lines = f.readlines()

labels_str = [x.strip().split()[0] for x in data_lines]

a = []
for x in data_lines:
    a.append(x.strip().split()[2])
a = list(set([x for y in a for x in y]))

# replaced the string variables to numbers for easier analysis while modelling using Keras

feats = [list(x.strip().split()[2].replace('A', '0').replace('C', '1').replace('G', '2').replace('T', '3').replace('N', '4').replace('S','5').replace('D', '6').replace('R','7')) for x in data_lines]

labels = []
for lab in labels_str:
    if lab.strip(',') == 'EI':
        labels.append(['0', '0', '1'])    # assigned [0,0,1] for EI class
    elif lab.strip(',') == 'IE':
        labels.append(['0','1','0'])      # assigned [0,1,0] for IE class
    else:
        labels.append(['1','0','0'])      # assigned [1,0,0] for Neither class


zipped = list(zip(labels, feats))
random.shuffle(zipped)

labels, feats = zip(*zipped)

# Divding the dataset into training and test data in the ration of 80:20 
# I chose this ratio as the availability of data is limited 
labels_tr = np.array(labels[0:2552])   # training labels
feats_tr = np.array(feats[0:2552])    # training features

labels_test = np.array(labels[2552:])  # test labels
feats_test = np.array(feats[2552:])    # test features 

# I chose L2 regularization parameter to be 0.0001 for my Neural Network approach
# I chose a simple neural network with one input layer, one hidden layer and the output layer
# The model is chosen to be sequential and the input layer has 32 units (chosen after trail and error method to maximize training efficiency)
# The dropout factor has been chosen after trial and error method to maximize training efficiency
l2_reg = l2(0.0001)
model = Sequential()
model.add(Dense(32, input_dim=60, activation='relu', activity_regularizer=l2_reg))
model.add(Dropout(0.3))

# The hidden layer has 32 units (chosen after trail and error method to maximize training efficiency)
model.add(Dense(32, activation='relu', activity_regularizer=l2_reg))
#model.add(Dropout(0.3))

model.add(Dense(3, activation='sigmoid'))      # The output unit

# I used the RMSprop optimizer for my adaptive learning rate 
opt = rmsprop(lr=0.0001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# I used 32 as my batch size so that each batch has 10 elements and epochs as 50 
# for optimum fitting of the curve  
model.fit(feats_tr, labels_tr, epochs=50, batch_size=32)
# the training efficiency is about 85% 
out = model.predict(feats_test)  # output prediction 

out_lab = np.argmax(out,axis=1)  

out_labels = []
for outs in out_lab:
    if outs == 0:
        out_labels.append('0 0 1')
    elif outs == 1:
        out_labels.append('0 1 0')
    elif outs == 2:
        out_labels.append('1 0 0')

target_labels = ["{0} {1} {2}".format(x[0], x[1], x[2]) for x in labels_test]

# Since the output is of the form a vector of 0 and 1. I used the sum as a criteria to calculate the efficiency of the model
sum = 0
for idx, lab in enumerate(out_labels):
    if lab == target_labels[idx]:
       sum += 1

print(sum/len(out_labels))

# The test data gives the efficiency of the model as 29.3% 
# I would have loved to play around more with the data but given my time constriants ( I am a graduate student currently in school)
# I could not spend more time with the data. 
# I look forward to your response 