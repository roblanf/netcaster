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
    def __init__(self, input_shape, out_config, loss, settings, parents=None, genotype = None):
        """An individual network"""
        self.parents = parents              # a list of P parents genotypes, from which we can generate offspring
        self.input_shape = input_shape      # shape of input data, from np.shape(X_train), where dim 0 is N
        self.out_config = out_config    # keras layer to put on as the ouput layer at the end
        self.loss = loss                    # loss function, e.g. "categorical crossentropy"
        self.genotype = genotype            # a genotype, which can be passed in
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

        output_layer = Dense.from_config(self.out_config)
        self.model.add(self.output_layer)  

        optimiser = self.get_optimiser()
        self.model.compile(optimizer = optimiser, 
                           loss = self.loss, 
                           metrics = ['accuracy'])
        

    def train_network(self, X_train, Y_train):
        # train network based on genotype

        # training is a list, where each entry is an epoch and the value
        # is the batch size

        start_time = time()

        for epoch_batch_size in self.genotype["training"]: 
            self.model.fit(X_train, Y_train, batch_size = epoch_batch_size, epochs = 1, shuffle = True)

        end_time = time()

        self.training_time = start_time - end_time

    def test_network(self, X_val, Y_val):
        # test network based on genotype
        # return accuracy and CPU/GPU time
        # testing is done on a validation set

        start_time = time()

        evals = self.model.evaluate(X_val, Y_val)

        end_time = time()

        self.test_time = start_time - end_time

        self.loss = evals[0]
        self.fitness = evals[1] #fitness is just the test accuracy

    def get_fitness(self, X_train, Y_train, X_test, Y_test, genotype = None):
        # a general function to return the fitness of any individual
        # and calculate it if it hasn't already been calculated

        if self.fitness:
            return(self.fitness)

        if self.parents == None:

            if self.genotype == None
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
            else: 
                try:
                    self.build_network()
                    self.train_network(X_train, Y_train)
                    self.test_network(X_test, Y_test)
                except:
                    # that genotype was shit
                    self.fitness = 0

        else: # we get the genotype from the parents
            try:
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

        slip = self.mutate_addition(current_parent.genotype["params"]["slip"], size = 0.1, limits = [0, 1], mutrate = mutrate)

        params =   {"mutation": mutrate,
                    "recomb": recomb,
                    "optimiser": optimiser,
                    "learning_rate": learning_rate,
                    "slip": slip}

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
        slip = self.genotype["params"]["slip"]

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
        if np.random.uniform(0, 1) < slip:
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
                if len(cp_layers_parent) > 0:
                    # 50/50 split: choose a layer from the parent (with mutation)
                    # vs. add a random layer
                    if np.random.uniform(0, 1) < 0.5:
                        offspring_cp_layers.append(mutate_layer(np.random.choice(cp_layers_parent)))
                    else:
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
        if np.random.uniform(0, 1) < slip:
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
                    # 50/50 split: choose a layer from the parent (with mutation)
                    # vs. add a random layer
                    if np.random.uniform(0, 1) < 0.5:
                        offspring_d_layers.append(mutate_layer(np.random.choice(d_layers_parent)))
                    else:
                        offspring_d_layers.append(self.random_d_layer())
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
        print("slip: ", self.genotype["params"]["slip"])





