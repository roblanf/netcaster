import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from pprint import pprint
from time import time
import random
from layers import * 
from mutation import * 
from optimisers import *

class Individual(object):
    def __init__(self, input_shape, out_config, loss, parents=None, genotype = None, mcmc = False):
        """An individual network"""
        self.parents = parents              # a list of P parents genotypes, from which we can generate offspring
        self.input_shape = input_shape      # shape of input data, from np.shape(X_train), where dim 0 is N
        self.out_config = out_config        # keras layer to put on as the ouput layer at the end
        self.loss = loss                    # loss function, e.g. "categorical crossentropy"
        self.genotype = genotype            # a genotype, which can be passed in
        self.fitness = None
        self.training_time = np.Inf         # in case there's an error during fitting
        self.test_time = np.Inf
        self.mcmc = mcmc                    # a flag because we evolve differently for hill climbing and mcmc

    def make_genotype(self):
        if self.parents==None:
            self.random_genotype()
        else:
            self.offspring_genotype()

    def random_genotype(self, mutrate = 0.2, indel=0.3):
        # generate a random CNN genotype with sensible defaults

        # number of conv and pooling layers at the start
        num_cp_layers = np.random.randint(2, 8)

        # number of fully connected layers
        num_d_layers = np.random.randint(1, 3)

        # genotype hyperparameters
        # to get a random optimiser:
        opt = random_optimiser()

        # to use Adam:
        opt = ("Adam", 0.001)

        params =   {"mutation": mutrate,
                    "optimiser": opt[0],
                    "learning_rate": opt[1],
                    "indel": indel, # chance of inserting/deleting layers in the network
                    }

        network = []
        for i in range(num_cp_layers):
            new_layer = random_cp_layer()
            network.append(new_layer)

        for i in range(num_d_layers):
            network.append(random_full_layer())

        epochs = np.random.randint(5, 10)

        training = []
        for i in range(epochs):
            # add minibatch sizes for epochs
            training.append(np.random.choice([512, 1024])) 

        # genotype has parameters, then network architecture, then training epochs
        genotype = {'params': params, 'network': network, 'training': training}

        self.genotype = genotype


    def build_network(self):
        # build a keras network from genotype

        self.model = Sequential()

        # add layers one by one
        for layer_num in range(len(self.genotype["network"])):
            self.add_layer(layer_num)

        # flatten if last layer isn't full
        if self.genotype["network"][layer_num]["type"] in ["conv", "pool"]:
            self.model.add(Flatten())

        # add the output layer (this is user-specified)
        output_layer = Dense.from_config(self.out_config)
        output_layer.name = "output_layer"        
        self.model.add(output_layer)  

        # add an optimiser and compile the model
        optimiser = get_optimiser(self.genotype["params"]["optimiser"], self.genotype["params"]["learning_rate"])
        self.model.compile(optimizer = optimiser, 
                           loss = self.loss, 
                           metrics = ['accuracy'])
        
    def train_network(self, X_train, Y_train):
        # train network based on genotype

        # training is a list, where each entry is an epoch and the value
        # is the batch size


        start_time = time()

        # train over each epoch
        for epoch_batch_size in self.genotype["training"]: 
            self.model.fit(X_train, Y_train, batch_size = epoch_batch_size, epochs = 1, shuffle = True, verbose = 0)

        end_time = time()

        self.training_time = end_time - start_time

    def test_network(self, X_val, Y_val):
        # test network based on genotype
        # return accuracy and CPU/GPU time
        # testing is done on a validation set

        start_time = time()

        evals = self.model.evaluate(X_val, Y_val, verbose = 0)

        end_time = time()

        self.test_time = end_time - start_time

        self.loss = evals[0]
        self.fitness = evals[1] #fitness is just the test accuracy

    def get_fitness(self, X_train, Y_train, X_val, Y_val):
        # a general function to return the fitness of any individual
        # and calculate it if it hasn't already been calculated

        if self.fitness:
            return(self.fitness)

        if self.parents == None:

            if self.genotype == None:
                while(self.fitness == None):
                    try:
                        self.make_genotype()
                        self.build_network()
                        self.train_network(X_train, Y_train)
                        self.test_network(X_val, Y_val)
                    except:
                        # we do this because sometimes
                        # we build impossible networks
                        
                        pass
            else: # genotype is already specified, e.g. by loading it in
                try:
                    self.build_network()
                    self.train_network(X_train, Y_train)
                    self.test_network(X_val, Y_val)
                except:
                    # that genotype was shit
                    self.fitness = 0

        else: # we get the genotype from the parents
            try:
                self.make_genotype()
                self.build_network()
                self.train_network(X_train, Y_train)
                self.test_network(X_val, Y_val)
            except:
                # that offspring didn't work
                self.fitness = 0


        return(self.fitness)


    def add_layer(self, layer_num):

        layer = self.genotype["network"][layer_num]
        layer_type = layer["type"]

        # sometimes we need the type of the previous layer
        # so we can flatten before the first Dense layer in the network
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
            new_layer = add_conv_layer(layer, input_shape)
            new_layer = Conv2D.from_config(new_layer)
        if layer_type == "pool":
            new_layer = add_pool_layer(layer, input_shape)
            new_layer = MaxPooling2D.from_config(new_layer)
        if layer_type == "full":
            if prev_type in ["conv", "pool", 0]:
                self.model.add(Flatten())
            new_layer = add_full_layer(layer, input_shape)
            new_layer = Dense.from_config(new_layer)

        self.model.add(new_layer)


        # TODO add residual connection, e.g. by a flag
        # where we store this layer and a value (e.g. +2)
        # in a list of residual connections, so that at some
        # future layer we can add a layer like this:
        # z = keras.layers.add([x, y])

    def offspring_genotype(self):
        # make a genotype from N parents
        self.genotype = {} # reset the genotype

        # params
        offspring_params = self.get_offspring_params()
        self.genotype["params"] = offspring_params
        # network
        offspring_network = self.get_offspring_network()
        self.genotype["network"] = offspring_network
        # training
        offspring_training = self.get_offspring_training()
        self.genotype["training"] = offspring_training


    def get_offspring_params(self):
        # get a parameter dictionary from the parents
        # run through all params, with recombination and mutation

        # mutation rate
        current_parent = np.random.choice(self.parents)
        mutrate = current_parent["params"]["mutation"] # start with the mutation rate from the current parent
        if self.mcmc==False:
            mutrate = mutate_product(mutrate, size = 1.05, limits = [0, 1], mutrate = mutrate)

        # optimiser and learning rate
        current_parent = self.recombination(current_parent)
        optimiser, learning_rate = mutate_optimiser(current_parent["params"]["optimiser"], current_parent["params"]["learning_rate"], size = 1.05, limits = [0, 1], mutrate = current_parent["params"]["learning_rate_mutrate"])

        # indel rate
        indel = current_parent["params"]["indel"]
        if self.mcmc==False:
            current_parent = self.recombination(current_parent)
            indel = mutate_product(indel, size = 1.05, limits = [0, 1], mutrate = mutrate)

        params =   {"mutation": mutrate,
                    "optimiser": optimiser,
                    "learning_rate": learning_rate,
                    "indel": indel,
                    }

        return(params)


    def get_offspring_network(self):

        parents = self.parents.copy()

        if self.mcmc==False:            
            mutrate = self.genotype["params"]["mutation"]
            indel = self.genotype["params"]["indel"]
        else:
            mutrate = 0.1
            indel = 0.5

        # we'll start with the conv and pool layers, then rinse and repeat for the dense layers
        # choose a parent
        current_parent = np.random.choice(parents)
        
        # get the conv/pool layers from the parent network
        cp_layers_parent = [layer for layer in current_parent["network"] if layer["type"] in ["conv", "pool"]]
        cp_layers_parent = cp_layers_parent.copy()

        # start by assuming that the offspring will look like this parent
        num_cp_layers = len(cp_layers_parent)

        # here's where we see if we'll change the number of layers
        # we change by at most 1
        new_cp_nums = [] #placeholders for the index of new layers, if we get them
        if np.random.uniform(0, 1) < indel:
            layer_change = np.random.choice([-1, 1])
            num_cp_layers = max(0, (num_cp_layers + layer_change))
            if layer_change > 0:
                all_cp_nums = list(range(num_cp_layers))
                new_cp_nums = random.sample(all_cp_nums, layer_change)


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
                    # vs. add a random cp layer
                    if np.random.uniform(0, 1) < 0.5:
                        offspring_cp_layers.append(mutate_layer(np.random.choice(cp_layers_parent.copy())))
                    else:
                        offspring_cp_layers.append(random_cp_layer())
                else:
                    offspring_cp_layers.append(random_cp_layer())

            else:
                offspring_cp_layers.append(cp_layers_parent[p_counter].copy())
                p_counter += 1

        # rinse and repeat for the full layers
        current_parent = self.recombination(current_parent)
        
        # get the dense layers from the parent network
        d_layers_parent = [layer for layer in current_parent["network"] if layer["type"] in ["full"]]
        d_layers_parent = d_layers_parent.copy()

        # start by assuming that the offspring will look like this parent
        num_d_layers = len(d_layers_parent)

        # here's where we see if we'll change the number of layers
        # we change by at most 2, with a bias towards adding layers
        new_d_nums = [] #placeholders for the index of new layers, if we get them
        if np.random.uniform(0, 1) < indel:
            layer_change = np.random.choice([-1, 1])
            num_d_layers = max(0, len(d_layers_parent) + layer_change)
            if layer_change > 0:
                all_d_nums = list(range(num_d_layers))
                new_d_nums = random.sample(all_d_nums, layer_change)

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
                        offspring_d_layers.append(mutate_layer(np.random.choice(d_layers_parent.copy())))
                    else:
                        offspring_d_layers.append(random_full_layer())
                else:
                    offspring_d_layers.append(random_full_layer())
            else:
                offspring_d_layers.append(d_layers_parent[p_counter].copy())
                p_counter += 1

        # now we'll have the option to mutate all the layers
        pre_mutation = offspring_cp_layers.copy() + offspring_d_layers.copy()
        post_mutation = []
        for layer in pre_mutation:
            post_mutation.append(mutate_layer(layer.copy(), mutrate))


        return(post_mutation)

    def recombination(self, current_parent, parents = None):

        if parents == None:
            parents = self.parents.copy()

        if len(parents) == 1:
            return(current_parent) # there's only one parent, so no recombination
        else:
            # return a parent that's not the current parent
            p = parents.copy()
            p.remove(current_parent)
            return np.random.choice(p)

    def get_offspring_training(self):

        parents = self.parents.copy()

        # choose a parent
        current_parent = np.random.choice(parents)

        # get the number of epochs from the current parent
        # mutate it up or down by up to 1 epoch
        num_epochs = len(current_parent["training"])
        num_epochs = mutate_int_fixed(num_epochs, 1, [1, 100], self.genotype["params"]["epoch_mutrate"])

        offspring_training = []
        while len(offspring_training) < num_epochs:

            current_parent = self.recombination(current_parent, parents)

            if len(current_parent["training"]) > len(offspring_training):
                # current parent has enough epochs
                next_batchsize = current_parent["training"][len(offspring_training)]
                next_batchsize = mutate_batchsize(next_batchsize, self.genotype["params"]["batchsize_mutrate"])
                offspring_training.append(next_batchsize)

            else: # current parent doesn't have enough epochs
                
                # remove that parent
                parents.remove(current_parent) # this parent is no more use to us
                
                # choose another parent if there is one
                if len(parents)>0:
                    current_parent = np.random.choice(parents)
                else: # no more parents epochs to do in order
                    break

        # if we get to here, the offspring still needs epochs, and the parents have run out
        # epochs of batch sizes randomly selected from the two parents
        if len(offspring_training) < num_epochs:
            parent_epochs = []
            for p in self.parents:
                parent_epochs += p["training"]

            while len(offspring_training) < num_epochs:
                offspring_training.append(np.random.choice(parent_epochs))

        return(offspring_training)

    def print_genotype(self):

        print("Network: %d layers" %(len(self.genotype["network"])))
        for layer in self.genotype["network"]:
            print_layer(layer)
        print(len(self.genotype["training"]), "epochs")
        print(self.genotype["training"])
        print("mutation: ", self.genotype["params"]["mutation"])
        print("learning_rate: ", genotype["params"]["learning_rate"])
        print("optimiser: ", self.genotype["params"]["optimiser"])
        print("indel: ", self.genotype["params"]["indel"])




def print_genotype(genotype):

        print("Network: %d layers" %(len(genotype["network"])))
        for layer in genotype["network"]:
            print_layer(layer)
        print(len(genotype["training"]), "epochs")
        print(genotype["training"])
        print("mutation: ", genotype["params"]["mutation"])
        print("learning_rate: ", genotype["params"]["learning_rate"])
        print("optimiser: ", genotype["params"]["optimiser"])
        print("indel: ", genotype["params"]["indel"])

