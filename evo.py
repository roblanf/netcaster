import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from pprint import pprint
import pickle
from time import time
import random
import csv

class Individual(object):
    def __init__(self, input_shape, output_layer, loss, settings, parents=None):
        """An individual network"""
        self.settings = settings            # dict of global settings for defining new individuals
        self.parents = parents              # a list of P parents genotypes, from which we can generate offspring
        self.input_shape = input_shape      # shape of input data, from np.shape(X_train), where dim 0 is N
        self.output_layer = output_layer    # keras layer to put on as the ouput layer at the end
        self.loss = loss                    # loss function, e.g. "categorical crossentropy"
        self.fitness = None

        if parents != None:
            for p in parents:
                p.model = None
                p.parents = None # otherwise we recursively keep a lot of data.

    def make_genotype(self):
        if self.parents==None:
            self.random_genotype()
        else:
            self.offspring_genotype()

        print("\nOffspring genotype")
        self.print_genotype()

    def build_network(self):
        # build network from genotype
        self.model = Sequential()


        # add layers one by one
        for layer_num in range(len(self.genotype["network"])):
            self.add_layer(layer_num)

        # flatten if last layer isn't full
        if self.genotype["network"][layer_num]["type"] in ["conv", "pool"]:
            self.model.add(Flatten())


        self.model.add(self.output_layer)  

        optimiser = self.get_optimiser()
        self.model.compile(optimizer = optimiser, 
                           loss = self.loss, 
                           metrics = ['accuracy'])
        

    def get_optimiser(self):
        p = self.genotype["params"]

        if p["optimiser"] == "SGD":
            optimiser = keras.optimizers.SGD(lr=p["learning_rate"])
        elif p["optimiser"] == "RMSprop":
            optimiser = keras.optimizers.RMSprop(lr=p["learning_rate"])
        elif p["optimiser"] == "Adagrad":
            optimiser = keras.optimizers.Adagrad(lr=p["learning_rate"])
        elif p["optimiser"] == "Adadelta":
            optimiser = keras.optimizers.Adadelta(lr=p["learning_rate"])
        elif p["optimiser"] == "Adam":
            optimiser = keras.optimizers.Adam(lr=p["learning_rate"])
        elif p["optimiser"] == "Adamax":
            optimiser = keras.optimizers.Adamax(lr=p["learning_rate"])
        elif p["optimiser"] == "Nadam":
            optimiser = keras.optimizers.Nadam(lr=p["learning_rate"])
        else: #WTF
            print("Oh Dear")

        return(optimiser)


    def train_network(self, X_train, Y_train):
        # train network based on genotype

        # training is a list, where each entry is an epoch and the value
        # is the batch size

        start_time = time()

        for epoch_batch_size in self.genotype["training"]: 
            self.model.fit(X_train, Y_train, batch_size = epoch_batch_size, epochs = 1, shuffle = True)

        end_time = time()

        self.training_time = start_time - end_time

    def test_network(self, X_test, Y_test):
        # test network based on genotype
        # return accuracy and CPU/GPU time

        start_time = time()

        evals = self.model.evaluate(X_test, Y_test)

        end_time = time()

        self.test_time = start_time - end_time

        self.loss = evals[0]
        self.fitness = evals[1] #fitness is just the test accuracy

    def get_fitness(self, X_train, Y_train, X_test, Y_test, genotype = None):
        # a general function to return the fitness of any individual
        if self.fitness:
            return(self.fitness)

        if self.parents == None:
            # keep generating random genotypes until one works
            while(self.fitness == None):
                try:
                    self.make_genotype()
                    self.build_network()
                    self.train_network(X_train, Y_train)
                    self.test_network(X_test, Y_test)
                except:
                    # we do this because sometimes
                    # we build impossible networks
                    pass

        else: # we get the genotype from the parents
            try:
                if genotype == None: # a catch because this allows us to pass in an old genotype if we want
                    self.make_genotype()
                self.build_network()
                self.train_network(X_train, Y_train)
                self.test_network(X_test, Y_test)
            except:
                # the network didn't work
                # so it gets zero fitness
                self.fitness = 0

        print("fitness: ", self.fitness)
        return(self.fitness)


    def add_layer(self, layer_num):

        layer = self.genotype["network"][layer_num]
        layer_type = layer["type"]

        # sometimes we need the type of the previous layer
        if layer_num > 0:
            prev_type = self.genotype["network"][layer_num - 1]["type"]
        else:
            prev_type = 0

        # use input shape for first layer, and not for future layers
        if len(self.model.layers)==0 and layer_type in ["conv", "pool"]:
            input_shape = self.input_shape
        elif len(self.model.layers)==0 and layer_type in ["full"]:
            input_shape = (np.product(self.input_shape),)
        else:
            input_shape = False
        if layer_type == "conv":
            new_layer = self.add_conv_layer(layer, input_shape)
        if layer_type == "pool":
            new_layer = self.add_pool_layer(layer, input_shape)
        if layer_type == "full":
            if prev_type in ["conv", "pool", 0]:
                self.model.add(Flatten())
            new_layer = self.add_full_layer(layer, input_shape)

        self.model.add(new_layer)

        # add dropout, norm, residual 
        if layer.get("dropout", False):
            # dropout at a rate determined by genotype
            self.model.add(Dropout(layer["dropout"]))

        if layer.get("norm", False):
            # we just add normalization with default params
            self.model.add(BatchNormalization())

        # TODO add residual connection, e.g. by a flag
        # where we store this layer and a value (e.g. +2)
        # in a list of residual connections, so that at some
        # future layer we can add a layer like this:
        # z = keras.layers.add([x, y])

    def offspring_genotype(self):
        # make a genotype from N parents
        self.genotype = {} # reset the genotype

        N = len(self.parents)

        current_parent = np.random.choice(self.parents)

        mutrate = current_parent.genotype["params"]["mutation"] # start with the mutation rate from the current parent

        # params
        mutrate = self.mutate_addition(current_parent.genotype["params"]["mutation"], size = 0.1, limits = [0, 1], mutrate = mutrate)

        if np.random.uniform(0, 1) < current_parent.genotype["params"]["recomb"]:
            current_parent = np.random.choice(self.parents)

        recomb = self.mutate_addition(current_parent.genotype["params"]["recomb"], size = 0.1, limits = [0, 1], mutrate = mutrate)

        # recombination looks like this - just switch parent
        if np.random.uniform(0, 1) < recomb:
            current_parent = np.random.choice(self.parents)

        optimiser, learning_rate = self.mutate_optimiser(current_parent.genotype["params"]["optimiser"], 
                                                         current_parent.genotype["params"]["learning_rate"],
                                                         size = 3, # change the learning rate up to a factor of this
                                                         limits = [0, 1], # limits on the learning rate
                                                         mutrate = mutrate)

        if np.random.uniform(0, 1) < current_parent.genotype["params"]["recomb"]:
            current_parent = np.random.choice(self.parents)

        sliprate = self.mutate_addition(current_parent.genotype["params"]["sliprate"], size = 0.1, limits = [0, 1], mutrate = mutrate)

        params =   {"mutation": mutrate,
                    "recomb": recomb,
                    "optimiser": optimiser,
                    "learning_rate": learning_rate,
                    "sliprate": sliprate}

        self.genotype["params"] = params

        # network
        offspring_network = self.get_offspring_network()
        # training
        offspring_training = self.get_offspring_training()

        self.genotype["network"] = offspring_network
        self.genotype["training"] = offspring_training


    def get_offspring_network(self):

        mutrate = self.genotype["params"]["mutation"]
        recomb = self.genotype["params"]["recomb"]
        parents = self.parents.copy()
        sliprate = self.genotype["params"]["sliprate"]

        print("\n\nParent genotype(s)")
        for p in parents:
            p.print_genotype()
        print("\n")

        # we'll start with teh conv and pool layers, then rinse and repeat for the dense layers
        # choose a parent
        current_parent = np.random.choice(parents)
        
        # get the conv/pool layers from the parent network
        cp_layers_parent = [layer for layer in current_parent.genotype["network"] if layer["type"] in ["conv", "pool"]]

        # start by assuming that the offspring will look like this parent
        num_cp_layers = len(cp_layers_parent)

        # here's where we see if we'll change the number of layers
        # we change by at most 2, with a bias towards adding layers
        new_cp_nums = [] #placeholders for the index of new layers, if we get them
        if np.random.uniform(0, 1) < sliprate:
            layer_change = np.random.choice([-1, 1, 2])
            num_cp_layers = max(0, (num_cp_layers + layer_change))
            if layer_change > 0:
                all_cp_nums = list(range(num_cp_layers))
                new_cp_nums = random.sample(all_cp_nums, layer_change)

            print("changing cp layers by: ", layer_change)

            if layer_change < 0:
                # we need to lose layer_change layers at random
                # from the parent genotype
                try:
                    for deletion in range(layer_change*-1):
                        del cp_layers_parent[np.random.randint(0, len(cp_layers_parent))]
                except:
                    pass # we've run out of parent layers to delete, so don't worry

        # now we'll make an offspring genotype
        offspring_cp_layers = []
        p_counter = 0 # count parent layers as we use them up
        for i in range(num_cp_layers):

            if i in new_cp_nums:
                # this is a layer we added
                # if the parents have cp layers, we'll just randomly pick one with mutation
                if len(cp_layers_parent) > 0:
                    offspring_cp_layers.append(np.random.choice(cp_layers_parent))
                else:
                    # otherwise we add a random cp layer
                    offspring_cp_layers.append(self.random_cp_layer())
            else:
                offspring_cp_layers.append(cp_layers_parent[p_counter])
                p_counter += 1

        # rinse and repeat for the full layers
        # start by allowing recombination
        if np.random.uniform(0, 1) < recomb:
            current_parent = np.random.choice(parents)
        
        # get the dense layers from the parent network
        d_layers_parent = [layer for layer in current_parent.genotype["network"] if layer["type"] in ["full"]]

        # start by assuming that the offspring will look like this parent
        num_d_layers = len(d_layers_parent)

        # here's where we see if we'll change the number of layers
        # we change by at most 2, with a bias towards adding layers
        new_d_nums = [] #placeholders for the index of new layers, if we get them
        if np.random.uniform(0, 1) < sliprate:
            layer_change = np.random.choice([-1, 1, 2])
            num_d_layers = max(0, len(d_layers_parent) + layer_change)
            if layer_change > 0:
                all_d_nums = list(range(num_d_layers))
                new_d_nums = random.sample(all_d_nums, layer_change)

            print("changing full layers by: ", layer_change)

            if layer_change < 0:
                # we need to lose layer_change layers at random
                # from the parent genotype
                try:
                    for deletion in range(layer_change*-1):
                        del d_layers_parent[np.random.randint(0, len(d_layers_parent))]
                except:
                    pass # we've run out of parent layers to delete, so don't worry


        # now we'll make an offspring genotype
        offspring_d_layers = []
        p_counter = 0 # count parent layers as we use them up
        for i in range(num_d_layers):

            if i in new_d_nums:
                # this is a layer we added
                # if the parents have cp layers, we'll just randomly pick one with mutation
                if len(d_layers_parent) > 0:
                    offspring_d_layers.append(np.random.choice(d_layers_parent))
                else:
                    # otherwise we add a random cp layer
                    offspring_d_layers.append(self.random_full_layer())
            else:
                offspring_d_layers.append(d_layers_parent[p_counter])
                p_counter += 1

        # now we'll have the option to mutate all the layers
        pre_mutation = offspring_cp_layers + offspring_d_layers
        post_mutation = []
        for layer in pre_mutation:
            post_mutation.append(self.mutate_layer(layer))


        return(post_mutation)


    def get_offspring_training(self):

        mutrate = self.genotype["params"]["mutation"]
        recomb = self.genotype["params"]["recomb"]
        parents = self.parents.copy()

        # choose a parent
        current_parent = np.random.choice(parents)

        # get the number of epochs from the current parent
        # mutate it up or down by up to 10%
        num_epochs = self.mutate_int(len(current_parent.genotype["training"]), size = 0.1, limits = [1, 100], mutrate = mutrate)

        offspring_training = []
        while len(offspring_training) < num_epochs:

            # recombination
            if np.random.uniform(0, 1) < recomb:
                current_parent = np.random.choice(parents)

            if len(current_parent.genotype["training"]) > len(offspring_training):
                # current parent has enough epochs
                next_batchsize = current_parent.genotype["training"][len(offspring_training)]
                next_batchsize = self.mutate_epochs(next_batchsize, mutrate)
                offspring_training.append(next_batchsize)

            else: # current parent doesn't have enough epochs
                
                # remove that parent
                parents.remove(current_parent) # this parent is no more use to us
                
                # choose another parent if there is one
                if len(parents)>0:
                    current_parent = np.random.choice(parents)
                else: # no more parents, fill training with random epochs
                    break

        # if we get to here, the offspring still needs epochs, and the parents have run out
        # so we add random epochs
        if len(offspring_training) < num_epochs:
            while len(offspring_training) < num_epochs:
                offspring_training.append(np.random.choice([2,4,8,16,32,64,128]))

        return(offspring_training)


    def mutate_optimiser(self, optimiser, learning_rate, size, limits, mutrate):
        # mutate the optimiser. if the optimiser changes, we set the learning rate
        # to the appropriate default
        # if it doesn't, we might mutate the current learning rate

        old_optimiser = optimiser

        if np.random.uniform(0, 1) < mutrate:
            
            # uncomment to bring back optimiser mutation change to a new optimiser
            #optimisers = ["SGD", "RMSprop", "Adagrad", "Adadelta", "Adam", "Adamax", "Nadam"]
            #optim_num = np.random.randint(0, 6) # choose an optimiser
            #optimiser = optimisers[optim_num]

            if optimiser != old_optimiser: # start with the default learning rate
                lr_defaults = [0.01, 0.001, 0.01, 1.0, 0.001, 0.002, 0.002] # defaults from keras
                learning_rate = lr_defaults[optim_num]

        else: # we didn't change the optimiser, so try mutating the learning rate
            learning_rate = self.mutate_product(learning_rate, size, [0, 1], mutrate)

        return(optimiser, learning_rate)

    def mutate_epochs(self, batchsize, mutrate):
        # change a batch size up or down by a factor of 2

        if np.random.uniform(0, 1) < mutrate:
            new_batchsize = np.int(np.random.choice([0.5, 2]) * batchsize)
        else:
            new_batchsize = batchsize
        if new_batchsize < self.settings["min_minibatch"]: new_batchsize = self.settings["min_minibatch"]
        if new_batchsize > self.settings["max_minibatch"]: new_batchsize = self.settings["max_minibatch"]

        return new_batchsize

    def mutate_int(self, value, size, limits, mutrate):
        # change an int by at least 1..., up to size %

        if np.random.uniform(0, 1) < mutrate:

            min_mut = 1
            max_mut = np.int(np.random.uniform(1.0, (1.0+size)) * value - value)

            # choose a size from -size to +size
            if min_mut < max_mut:
                mutation_size = np.random.randint(min_mut, max_mut)
            else:
                mutation_size = min_mut

            if np.random.uniform(0, 1) < 0.5: 
                mutation_size = mutation_size * -1

            # mutate the value
            value = value + mutation_size

            # check the limits
            if value < limits[0]: value = limits[0]
            if value > limits[1]: value = limits[1]

        return value


    def mutate_addition(self, value, size, limits, mutrate):
        # mutate a float of value, by amount size, within limits
 
        if np.random.uniform(0, 1) < mutrate:

            # choose a size from -size to +size
            mutation_size = np.random.uniform(0.0, size)
            if np.random.uniform(0, 1) < 0.5: mutation_size = mutation_size * -1

            # mutate the value
            value = value + mutation_size

            # check the limits
            if value < limits[0]: value = limits[0]
            if value > limits[1]: value = limits[1]

        return value

    def mutate_product(self, value, size, limits, mutrate):
        # mutate a float of value, by amount size, within limits

        if np.random.uniform(0, 1) < mutrate:
    
            # choose a size from -size to +size
            mutation_size = np.random.uniform(1.0, size)
            if np.random.uniform(0, 1) < 0.5:
                value = value / mutation_size
            else:
                value = value * mutation_size

            # check the limits
            if value < limits[0]: value = limits[0]
            if value > limits[1]: value = limits[1]

        return value


    def random_genotype(self):

        # we have a certain number of conv and pooling layers
        num_cp_layers = np.random.randint(2, 6)

        # followed by a certain numbe of dense layers
        num_d_layers = np.random.randint(0, 2)

        # general network parameters
        # optim_num = np.random.randint(0, 6) # choose an optimiser
        optim_num = 4 # if you want to initliase with Adam
        optimisers = ["SGD", "RMSprop", "Adagrad", "Adadelta", "Adam", "Adamax", "Nadam"]
        lr_defaults = [0.01, 0.001, 0.01, 1.0, 0.001, 0.002, 0.002] # defaults from keras
        params =   {"mutation": np.random.uniform(0.1, 0.4),
                    "recomb": np.random.uniform(0.7, 1.0),
                    "optimiser": optimisers[optim_num],
                    "learning_rate": lr_defaults[optim_num],
                    "sliprate": np.random.uniform(0.0, 0.3)}

        network = []
        for i in range(num_cp_layers):
            new_layer = self.random_cp_layer()
            network.append(new_layer)

        for i in range(num_d_layers):
            network.append(self.random_full_layer())

        epochs = np.random.randint(5, 10)

        training = []
        for i in range(epochs):
            training.append(np.random.choice([32, 64])) # add minibatch sizes for epochs

        # genotype has recombination rate, then network arch, then training
        genotype = {'params': params, 'network': network, 'training': training}

        self.genotype = genotype

    def mutate_layer(self, layer):
        if layer["type"] == "full":
            layer = self.mutate_full_layer(layer)

        if layer["type"] == "conv":
            layer = self.mutate_conv_layer(layer)

        if layer["type"] == "pool":
            layer = self.mutate_pool_layer(layer)

        return(layer)


    def print_layer(self, layer):

        if layer["type"] == "full":
            print("full: %d"  % layer["units"])

        if layer["type"] == "conv":
            print("conv: %dx%dx%d, s=%d, p=%s" %(layer["kernel"], layer["kernel"], layer["filters"], layer["strides"], layer["padding"]))

        if layer["type"] == "pool":
            print("pool: %dx%d, s=%d, p=%s" %(layer["pool_size"], layer["pool_size"], layer["strides"], layer["padding"]))

    def print_genotype(self):

        print("Network: %d layers" %(len(self.genotype["network"])))
        for layer in self.genotype["network"]:
            self.print_layer(layer)

        print(len(self.genotype["training"]), "epochs")
        print(self.genotype["training"])

        print("mutation: ", self.genotype["params"]["mutation"])
        print("recombination: ", self.genotype["params"]["recomb"])
        print("learning_rate: ", self.genotype["params"]["learning_rate"])
        print("optimiser: ", self.genotype["params"]["optimiser"])
        print("sliprate: ", self.genotype["params"]["sliprate"])

    def random_cp_layer(self):

        # this makes sure we don't try and add conv/pool
        # layers after dense layers
        r = np.random.choice(["conv", "pool"])

        if r == "conv":
            return(self.random_conv_layer())
        if r == "pool":
            return(self.random_pool_layer())

    def add_conv_layer(self, layer, input_shape):

        if input_shape == False:
            new_layer = Conv2D(filters = layer["filters"],
                               kernel_size = layer["kernel"],
                               strides = layer["strides"],
                               padding = layer["padding"],
                               activation = "relu")
        else:
            new_layer = Conv2D(filters = layer["filters"],
                               kernel_size = layer["kernel"],
                               strides = layer["strides"],
                               padding = layer["padding"],
                               activation = "relu",
                               input_shape = input_shape)

        return(new_layer)

    def mutate_conv_layer(self, layer):

        layer["filters"] = self.mutate_int(layer["filters"], 0.1, [1, 1000], layer["mutrate"])
        layer["kernel"] = self.mutate_int(layer["kernel"], 0.1, [1, 1000], layer["mutrate"])
        layer["strides"] = self.mutate_int(layer["strides"], 0.1, [1, 100], layer["mutrate"])

        if np.random.uniform(0, 1) < layer["mutrate"]:
            layer["padding"] = np.random.choice(["valid", "same"])

        layer["dropout"] = self.mutate_addition(layer["dropout"], 0.1, [0.0, 1.0], layer["mutrate"])

        if np.random.uniform(0, 1) < layer["mutrate"]:
            layer["norm"] = np.random.choice([0, 1])

        if np.random.uniform(0, 1) < layer["mutrate"]:
            layer["residual"] = np.random.choice([0, 1])

        layer["mutrate"] = self.mutate_addition(layer["mutrate"], 0.1, [0.0, 1.0], layer["mutrate"])

        return(layer)


    def random_conv_layer(self):

        conv_layer =    {"type": "conv",
                         "filters": np.random.randint(2,24),
                         "kernel": np.random.randint(1,4),
                         "strides": np.random.randint(1,3),
                         "padding": np.random.choice(["valid", "same"]),
                         "mutrate": np.random.uniform(0.01, 0.3),
                         "dropout": np.random.uniform(0.0, 0.99),
                         "norm": np.random.choice([0, 1]),
                         "residual": np.random.choice([0, 1])}

        return(conv_layer)




    def add_pool_layer(self, layer, input_shape):

        if input_shape == False:
            new_layer = MaxPooling2D(pool_size = layer["pool_size"],
                                     strides = layer["strides"],
                                     padding = layer["padding"])

        else:
            new_layer = MaxPooling2D(pool_size = layer["pool_size"],
                                     strides = layer["strides"],
                                     padding = layer["padding"],
                                     input_shape = input_shape)

        return(new_layer)


    def mutate_pool_layer(self, layer):

        layer["pool_size"] = self.mutate_int(layer["pool_size"], 0.1, [2, 1000], layer["mutrate"])
        layer["strides"] = self.mutate_int(layer["strides"], 0.1, [1, 100], layer["mutrate"])

        if np.random.uniform(0, 1) < layer["mutrate"]:
            layer["padding"] = np.random.choice(["valid", "same"])

        layer["mutrate"] = self.mutate_addition(layer["mutrate"], 0.1, [0.0, 1.0], layer["mutrate"])

        return(layer)

    def random_pool_layer(self):

        pool_layer =    {"type": "pool",
                         "pool_size": np.random.randint(2,4),
                         "strides": np.random.randint(1,3),
                         "padding": np.random.choice(["valid", "same"]),
                         "mutrate": np.random.uniform(0.01, 0.3)}

        return(pool_layer)

    def add_full_layer(self, layer, input_shape):

        if input_shape == False:
            new_layer = Dense(units = layer["units"],
                              activation = "relu")

        else:
            new_layer = Dense(units = layer["units"],
                              activation = "relu",
                              input_shape = input_shape)

        return(new_layer)

    def mutate_full_layer(self, layer):

        layer["units"] = self.mutate_int(layer["units"], 0.1, [1, 10000], layer["mutrate"])
        layer["dropout"] = self.mutate_addition(layer["dropout"], 0.1, [0.0, 1.0], layer["mutrate"])

        if np.random.uniform(0, 1) < layer["mutrate"]:
            layer["norm"] = np.random.choice([0, 1])

        if np.random.uniform(0, 1) < layer["mutrate"]:
            layer["residual"] = np.random.choice([0, 1])


        layer["mutrate"] = self.mutate_addition(layer["mutrate"], 0.1, [0.0, 1.0], layer["mutrate"])

        return(layer)


    def random_full_layer(self):

        full_layer =    {"type": "full",
                         "units": np.random.randint(4,200),
                         "mutrate": np.random.uniform(0.01, 0.3),
                         "dropout": np.random.uniform(0.0, 0.99),
                         "norm": np.random.choice([0, 1]),
                         "residual": np.random.choice([0, 1])}

        return(full_layer)



# an example

# load the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

Y_train = train[['label']]
X_train = train.drop(train.columns[[0]], axis=1)
X_test = test

#Reshape the training and test set
X_train = np.array(X_train)
X_test = np.array(X_test)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

#Padding the images by 2 pixels since in the paper input images were 32x32
X_train = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

#Standardization
mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)
X_train = (X_train - mean_px)/(std_px)

#One-hot encoding the labels
from keras.utils.np_utils import to_categorical
Y_train = to_categorical(Y_train)

# split into test and train
X_test = X_train[0:10000]
Y_test = Y_train[0:10000]

X_train = X_train[10001:40000]
Y_train = Y_train[10001:40000]



# basic settings for a particular input dataset

N = X_train.shape[0] # number of training examples


settings = {"min_cp_layers": 2,
            "max_cp_layers": 8,
            "min_d_layers": 1,
            "max_d_layers": 2,
            "min_epochs": 4,
            "max_epochs": 4,
            "min_minibatch": 16,
            "max_minibatch": 128,
            "mutation_rate": 0.1}

output_layer = Dense(units = 10, activation = 'softmax')
input_shape = (32, 32, 1)
loss = 'categorical_crossentropy'


# make a population

pop = [] # list of tuples (fitness, individual)

for i in range(20):
    random_ind = Individual(input_shape, Dense(units = 10, activation = 'softmax'), loss, settings, parents=None)
    random_ind.get_fitness(X_train, Y_train, X_test[0:1000], Y_test[0:1000])

    pop.append((random_ind.fitness, random_ind))


# 1 generation

num_generations = 30
keep = 0 # keep this many fittest individuals each generation. 
kill = 20 # kill this many of the least fit individuals
N = 50 # number of individuals in population
P = 2 # number of parents
remove_slowest = 20

# sort population
pop.sort(key=lambda tup: tup[0])

founders = pop.copy()

def choose_n_parents(pop, N, type = 'weighted'):
    # randomly choose parents based on weighted fitness
    parent_pop = pop.copy()

    parents = []
    for p in range(N):

        if type == 'weighted':
            fitness = [x[0] for x in parent_pop]
        elif type == 'rank':
            # rank selection: the least fit is eliminated with fitness zero like this
            fitness = list(range(len(parent_pop)))

            # we can square the ranked fitness to be more extreme
            fitness = np.multiply(fitness, fitness)

            # if you don't like killing the least fit, do this:
            #sfitness = list(range(1, len(parent_pop)+1))

        choice = np.random.uniform(0, np.sum(fitness))
        cs = choice>np.cumsum(fitness)
        try:
            parent_index = max(np.where(cs == True)[0])
        except:
            parent_index = 0

        print(parent_index)
        parents.append(parent_pop[parent_index][1])
        
    return(parents)



for g in range(num_generations):
    pprint(pop)

    # use a small random sample of the test data in each generation
    test_sample = np.random.choice(list(range(X_test.shape[0])), size = 2000, replace = False)
    X_test_sample = X_test[test_sample, :]
    Y_test_sample = Y_test[test_sample, :]

    train_sample = np.random.choice(list(range(X_train.shape[0])), size = 20000, replace = False)
    X_train_sample = X_train[train_sample, :]
    Y_train_sample = Y_train[train_sample, :]

    # record fitness
    fitness = [x[0] for x in pop]
    fitness.insert(0,g)
    with open('output.txt', 'a') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(fitness)

    genotypes = [x[1].genotype for x in pop]
    with open('genotypes%d.pkl' %(g), 'wb') as myfile:
        pickle.dump(genotypes, myfile)


    offspring = []

    # keep the fittest keep individuals in the population
    for i in range(len(pop)-keep, len(pop)):
        offspring.append(pop[i])

    # kill the worst ones: NB this assumes that the pop is sorted w.r.t. fitness
    for i in range(kill):
        del pop[i]


    print("breeding from")
    print(pop)

    # breed the rest
    while len(offspring) < N:
        f1 = Individual(input_shape, Dense(units = 10, activation = 'softmax'), loss, settings, parents=choose_n_parents(pop, P, type = 'rank'))
        f1.get_fitness(X_train_sample, Y_train_sample, X_test_sample, Y_test_sample)
        offspring.append((f1.fitness, f1))

    pop = offspring.copy()
    pop.sort(key=lambda tup: tup[0])



# example to load genotypes and continue
with open('genotypes.pkl', 'rb') as myfile:
        genotypes = pickle.load(myfile)

pop = []
for g in genotypes:
    ind = Individual(input_shape, Dense(units = 10, activation = 'softmax'), loss, settings, parents=None)
    ind.get_fitness(X_train[0:20000], Y_train[0:20000], X_test[0:1000], Y_test[0:1000], g)
    pop.append((ind.fitness, ind))


# TODO 

# Make the population just a list of genotypes and fitness, instead of individuals. That's easier to load and deal with

# Change everything so that there's a global mutation rate, and then each parameter has its own mutation size.
# This allows much finer parameter tuning.
# change sliprate to slip
# add a flip parameter - which would do random inversions on lists of layers
