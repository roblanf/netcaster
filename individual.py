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
        self.loss_function = loss                    # loss function, e.g. "categorical crossentropy"
        self.genotype = genotype            # a genotype, which can be passed in
        self.fitness = None
        self.accuracy = None
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


        # genotype hyperparameters
        # to get a random optimiser:
        opt = random_optimiser()

        # to use Adam:
        opt = ("Adam", 0.001)

        params =   {"mutation": mutrate,
                    "optimiser": opt[0],
                    "learning_rate": opt[1],
                    "indel": indel}

        # number of conv and pooling layers at the start
        num_cp_layers = np.random.randint(2, 5)

        # number of fully connected layers
        num_d_layers = np.random.randint(2, 5)

        network = []
        for i in range(num_cp_layers):
            # always start with a cp layer
            if i==0:
                new_layer = random_cp_layer()
            else:
                new_layer = random_cpbd_layer()

            network.append(new_layer)

        for i in range(num_d_layers):
            network.append(random_d_layer())

        epochs = np.random.randint(3, 5)
        #epochs = np.random.randint(1, 2) # for quick testing

        training = [1024]*epochs # let's start simple

        # genotype has parameters, then network architecture, then training epochs
        genotype = {'params': params, 'network': network, 'training': training}

        self.genotype = genotype

    def build_network(self):
        # build a keras network from genotype

        self.model = Sequential()

        # add layers one by one
        for layer_num in range(len(self.genotype["network"])):
            self.add_layer(layer_num)

        # add a flatten layer if there were no full layers in the network
        layer_types = [x["type"] for x in self.genotype["network"]]
        if 'full' not in layer_types:
            self.model.add(Flatten())

        # add the output layer (this is user-specified)
        output_layer = Dense.from_config(self.out_config)
        output_layer.name = "output_layer"        
        self.model.add(output_layer)  

        # add an optimiser and compile the model
        optimiser = get_optimiser(self.genotype["params"]["optimiser"], self.genotype["params"]["learning_rate"])

        self.model.compile(optimizer = optimiser, 
                           loss = self.loss_function, 
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
        self.accuracy = evals[1] #fitness is just the test accuracy

    def get_fitness(self, X_train, Y_train, X_val, Y_val):
        # a general function to return the fitness of any individual
        # and calculate it if it hasn't already been calculated
        
        if self.fitness:
            return(self.fitness)

        if self.parents == None:
            if self.genotype == None:
                while(self.accuracy == None):
                    try: 
                        self.make_genotype()
                        self.build_network()
                        self.train_network(X_train, Y_train)
                        self.test_network(X_val, Y_val)                        
                    except:
                        # keep trying until you find a random genotype that works
                        pass

            else: # genotype is already specified, e.g. by loading it in
                
                try:
                    self.build_network()
                    self.train_network(X_train, Y_train)
                    self.test_network(X_val, Y_val)
                except:
                    # that genotype was shit
                    self.accuracy = 0

        else: # we get the genotype from the parents
            try:
                self.make_genotype()
                self.build_network()
                self.train_network(X_train, Y_train)
                self.test_network(X_val, Y_val)
            except:
                self.accuracy = 0
                
        # define fitness
        # this is 90% accuracy, and 10% training time
        # where we don't reward training times lower than min_time seconds
        min_time = 5
        #self.fitness = self.accuracy * 0.9 + (min_time/np.max([self.training_time, min_time])) * 0.1

        # or we can just have fitness == accuracy
        self.fitness = self.accuracy

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

        any_full = 0
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
            if prev_type != "full" and any_full == 0:
                self.model.add(Flatten())
                any_full = 1
            new_layer = add_full_layer(layer, input_shape)
            new_layer = Dense.from_config(new_layer)
        if layer_type == "dropout":
            new_layer = add_dropout_layer(layer)
            new_layer = Dropout.from_config(new_layer)
        if layer_type == "batchnorm":

            if prev_type in ["conv", "pool"]:
                new_layer = BatchNormalization(axis = 3) # normalise across channels
            else:
                new_layer = BatchNormalization()

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
            # just so the mutation rate never actually stays at zero forever
            if mutrate<0.1:
                min_mutrate = 0.1
            else:
                min_mutrate = mutrate
            mutrate = mutate_float_fixed(mutrate, size = 0.1, limits = [0.1, 1], mutrate = min_mutrate)
        else:
            mutrate = 0.2

        # optimiser and learning rate
        current_parent = self.recombination(current_parent)
        optimiser, learning_rate = mutate_optimiser(current_parent["params"]["optimiser"], current_parent["params"]["learning_rate"], size = 2.0, limits = [0, 1], mutrate = mutrate)

        # indel rate
        indel = current_parent["params"]["indel"]
        if self.mcmc==False:
            current_parent = self.recombination(current_parent)
            indel = mutate_float_fixed(indel, size = 0.1, limits = [0.2, 1], mutrate = mutrate)

        params =   {"mutation": mutrate,
                    "optimiser": optimiser,
                    "learning_rate": learning_rate,
                    "indel": indel}

        return(params)


    def get_offspring_network(self):

        if self.mcmc==False:            
            mutrate = self.genotype["params"]["mutation"]
            indel = self.genotype["params"]["indel"]
        else:
            mutrate = 0.2
            indel = 0.5

        # choose a parent
        current_parent = np.random.choice(self.parents)
        
        # get the total number of layers and the position of the first full layer from this parent
        num_layers = len(current_parent["network"])
        layer_types = [x["type"] for x in current_parent["network"]]
        try:
            first_full = layer_types.index("full")
        except:
            first_full = np.Inf

        try: 
            last_cp = np.max[layer_types.index("conv"), layer_types.index("pool")]
        except:
            last_cp = np.Inf
            
        # now iterate over the layers, choosing layers with recombination 
        # from the two parents
        # but don't allow full layers until first_full
        current_parent = self.recombination(current_parent)
        offspring = []
        
        
        while len(offspring) < num_layers:

            # keep trying the parents at random until you get one with this layer index
            while len(current_parent["network"]) < len(offspring) + 1:
                current_parent = self.recombination(current_parent)

            # check that it's not a full layer if we can't have one yet
            new_layer = current_parent["network"][len(offspring)]

            if new_layer["type"] == "full" and len(offspring)<first_full:
                current_parent = self.recombination(current_parent)
            else:
                offspring.append(new_layer)

        # now we can add or delete a layer at random
        if np.random.uniform(0, 1) < indel:

            change = np.random.choice([-1, 1])

            if change == -1:
                del offspring[np.random.randint(0, len(offspring))]

            if change == +1:
                insertion_point = np.random.randint(0, len(offspring)+1) # +1 to insert at the end
                rd = random_d_layer()
                rcpbd = random_cpbd_layer()
                
                if insertion_point >= last_cp: # after the last conv/pool layer, anything goes
                    insertion_layer = np.random.choice([rd, rcpbd])
                elif insertion_point >= first_full: # after a full layer, you have to have full, batchnorm, drop
                    insertion_layer = random_d_layer()
                else: # before the last conv/pool layer, you have to have conv, pool, batch, drop
                    insertion_layer = random_cpbd_layer()

                offspring.insert(insertion_point, insertion_layer)
        
        # finally, we'll mutate all the layers
        post_mutation = []
        for layer in offspring:
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
        mutrate = self.genotype["params"]["mutation"]

        # choose a parent
        current_parent = np.random.choice(parents)

        # get the number of epochs from the current parent
        # mutate it up or down by up to 1 epoch
        num_epochs = len(current_parent["training"])
        num_epochs = mutate_int_fixed(num_epochs, 1, [1, 100], mutrate)

        offspring_training = []
        while len(offspring_training) < num_epochs:

            current_parent = self.recombination(current_parent, parents)

            if len(current_parent["training"]) > len(offspring_training):
                # current parent has enough epochs
                next_batchsize = current_parent["training"][len(offspring_training)]
                next_batchsize = mutate_batchsize(next_batchsize, mutrate)
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
        # so we'll just add more epochs that look like the final epoch
        if len(offspring_training) < num_epochs:

            while len(offspring_training) < num_epochs:
                offspring_training.append(offspring_training[-1])

        return(offspring_training)

    def print_genotype(self):

        print("Network: %d layers" %(len(self.genotype["network"])))
        for layer in self.genotype["network"]:
            print_layer(layer)
        print(len(self.genotype["training"]), "epochs")
        print(self.genotype["training"])
        print("mutation: ", self.genotype["params"]["mutation"])
        print("learning_rate: ", self.genotype["params"]["learning_rate"])
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

