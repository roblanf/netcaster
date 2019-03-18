# an example
from keras.utils.np_utils import to_categorical
from keras.layers import Dense
import pandas as pd
from lineage import Lineage
import numpy as np
from individual import Individual
import pickle
import zipfile


# load the data
train = pd.read_csv('mnist_train.csv')

# for fashion mnist
#with zipfile.ZipFile("fashion-mnist_train.csv.zip","r") as zip_ref:
#    zip_ref.extractall()

#train = pd.read_csv('fashion-mnist_train.csv')

Y_train = train[['label']]
X_train = train.drop(train.columns[[0]], axis=1)

#Reshape the training set
X_train = np.array(X_train)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

#Padding the images by 2 pixels since in the paper input images were 32x32
X_train = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')

#Standardization
mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)
X_train = (X_train - mean_px)/(std_px)

#One-hot encoding the labels
Y_train = to_categorical(Y_train)

# Here's where we start doing things for the evolutionary algorithm
# First we split off a validation set for assessing fitness
# we'll use the validation set to calculate network fitness

X_val = X_train[0:10000]
Y_val = Y_train[0:10000]

X_train = X_train[10000:]
Y_train = Y_train[10000:]


# We need a few things to establish a population

# what will the output layer look like?
output_config = Dense(units = 10, activation = 'softmax').get_config()

# shape of the input data
input_shape = (32, 32, 1)

# name of the loss function to use
loss = 'categorical_crossentropy'


# test mcmc code, delete
test = Lineage(input_shape, output_config, loss, X_train, Y_train, X_val, Y_val, trainsize = 3000, valsize = 900)
test.initialise(2)
test.evolve([2]*1)





# test code, delete
hillclimb = Lineage(input_shape, output_config, loss, X_train, Y_train, X_val, Y_val, trainsize = 3199, valsize = 500)
hillclimb.initialise(2)
# these settings keep the best individual around, and just breed one offspring from it in each generation
hillclimb.evolve([2]*4, num_parents = 1, kill = 1, keep=1)



# Now we're ready to set up a population and let it evolve. Here are some examples

# 1. A simple way of doing standard evolution 
lineage = Lineage(input_shape, output_config, loss, X_train, Y_train, X_val, Y_val, trainsize = 49999, valsize = 10000)
lineage.initialise(40) # start with 20 random genotypes
lineage.evolve([40]*100) # 50 generations of evolution with 20 individuals in each


# 2. A simple hill climbing algorithm
hillclimb = Lineage(input_shape, output_config, loss, X_train, Y_train, X_val, Y_val, trainsize = 31999, valsize = 5000)
hillclimb.initialise(5)
# these settings keep the best individual around, and just breed one offspring from it in each generation
hillclimb.evolve([2]*10, num_parents = 1, kill = 1, keep=1)



# 3. Let's see if we can use evolution to improve on the classic lenet-5 network

# Let's specify the lenet-5 network first...
conv1 =    {"type": "conv",
            "filters": 6,
            "kernel_h": 5,
            "kernel_w": 5,
            "strides_h": 1,
            "strides_w": 1,
            "padding": "valid",
            "mutrate": 0.1,
            "dropout": 0.0,
            "norm": 0}

pool1 =    {"type": "pool",
                  "pool_size_h": 2,
                  "pool_size_w": 2,
                  "strides_h": 2,
                  "strides_w": 2,
			"padding": "valid",
			"mutrate": 0.1}

conv2 =    {"type": "conv",
            "filters": 16,
            "kernel_h": 5,
            "kernel_w": 5,
            "strides_h": 1,
            "strides_w": 1,
            "padding": "valid",
            "mutrate": 0.1,
            "dropout": 0.0,
            "norm": 0}

pool2 =    {"type": "pool",
                  "pool_size_h": 2,
                  "pool_size_w": 2,
                  "strides_h": 2,
                  "strides_w": 2,
			"padding": "valid",
			"mutrate": 0.1}

full1 =    {"type": "full",
		    "units": 120,
		    "mutrate": 0.1,
		    "dropout": 0.0,
		    "norm": 0
            }

full2 =    {"type": "full",
		    "units": 84,
		    "mutrate": 0.1,
		    "dropout": 0.0,
		    "norm": 0
            }

network = [conv1, pool1, conv2, pool2, full1, full2]

# train for 10 epochs with a batch size of 128
training = [128]*10

# hyperparameters of the model and the evolutioary algorithm
params =   {"mutation": 0.1,
            "recomb": 1.0, 
            "optimiser": "Adam",
            "learning_rate": 0.001,
            "learning_rate_mutrate": 0.1, # mutation size for learning rate
            "slip": 0.1, # chance of adding new layers to network
            "epoch_mutrate": 0.1, # mutation rate applied to # epochs
            "batchsize_mutrate": 0.1 # mutation rate for batch size
            }

lenet5 = {'params': params, 'network': network, 'training': training}

# now let's write this out as a population
# a population is a list of tuples: (fitness, training time, genotype)
pop = [[(0.0, 0.0, lenet5, 0.0)]] 
with open('lenet5.pkl', 'wb') as myfile:
    pickle.dump(pop, myfile)

# now we'll make a lineage that starts with lenet5
lenetx = Lineage(input_shape, output_config, loss, X_train, Y_train, X_val, Y_val, trainsize = 31999, valsize = 5000)
# the recalculate fitness flag means that we'll start by evaluating the lenet5 genotype
lenetx.initialise('lenet5.pkl', recalculate_fitness = True)

# now we can evolve it how we like, here are two options
# this is hill climbing
lenetx.evolve([2], num_parents = 1, keep=1) # expand the population size to two 
lenetx.evolve([2]*1000, num_parents = 1, kill = 1, keep=1) 


# and this is standard evolution, but we'll keep the best network around in each generation
lenetx.evolve([50], num_parents = 1, keep=1) # expand the population size to 50
lenetx.evolve([50*20], keep=1) # 20 generations of 50 individuals, always keeping the fittest 
