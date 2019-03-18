#!/usr/bin/env python3

# a script to load 4 taxon fasta files following code at 
# https://github.com/SchriderLab/

from keras.utils.np_utils import to_categorical
from keras.layers import Dense
import pandas as pd
from lineage import Lineage
import numpy as np
from individual import Individual
import pickle
import zipfile
import tarfile
from fasta2numeric import *
    

# load train, test, validation data
trainfile = "train.tar.gz"
testfile = "test.tar.gz"
valfile = "val.tar.gz"

train_tar = tarfile.open(trainfile, "r:gz")
train_tar.extractall()
train_tar.close()

test_tar = tarfile.open(testfile, "r:gz")
test_tar.extractall()
test_tar.close()

val_tar = tarfile.open(valfile, "r:gz")
val_tar.extractall()
val_tar.close()


train_data1, valid_data1, test_data1 = tv_parse("TRAIN", "VAL", "TEST", 4)    

#Reshape for Keras
train_data1 = train_data1.reshape(train_data1.shape[0],train_data1.shape[1],train_data1.shape[2],1)
valid_data1 = valid_data1.reshape(valid_data1.shape[0],valid_data1.shape[1],valid_data1.shape[2],1)
test_data1  = test_data1.reshape(test_data1.shape[0],test_data1.shape[1],test_data1.shape[2],1)

# generate labels
# note that this assumes 
Nlabels = 3

train_label=to_categorical(np.repeat(range(0,Nlabels),len(train_data1)/Nlabels), num_classes=None)
valid_label=to_categorical(np.repeat(range(0,Nlabels),len(valid_data1)/Nlabels), num_classes=None)
test_label=to_categorical(np.repeat(range(0,Nlabels),len(test_data1)/Nlabels), num_classes=None)



# set up model parameters

output_config = Dense(units = 3, activation = 'softmax').get_config()

# input shape is (number_of_taxa, alignment_length, 1)
input_shape = (4, 1000, 1)
loss = 'categorical_crossentropy'


# Now we establish a lineage
p1 = Lineage(input_shape, output_config, loss, train_data1, train_label, valid_data1, valid_label, trainsize = 1000, valsize = 1000)
p1.initialise(10)
p1.evolve([10]*3)
