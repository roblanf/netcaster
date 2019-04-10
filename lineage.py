from individual import *

# these are to clear sessions to deal with this memory leak: https://github.com/keras-team/keras/issues/2102
import keras.backend.tensorflow_backend
import tensorflow as tf
from keras.backend import clear_session


import csv
import numpy as np
from pprint import pprint
import pickle
import os

class Lineage(object):
    def __init__(self, input_shape, out_config, loss_function, X_train, Y_train, X_val, Y_val, 
                 trainsize = None, valsize = None, overlapping = False, outputdir = '', name = 'lineage'):
        """An lineage of individuals that can evolve"""

        self.input_shape = input_shape
        self.out_config = out_config    # the keras config of the output layer (e.g.  'Dense(units = 10, activation = 'softmax').get_config()')
        self.loss = loss_function
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.trainsize = trainsize      # train on a subset of the data. None = use all data
        self.valsize = valsize          # test  on a subset of the data. None = use all data
        self.overlapping = overlapping  # True or False. True = overlapping generation. 
        self.lineage = []               # a list to contain the lists of genotype / fitness tuples that represent each population
        self.outputdir = outputdir    # filepath for saving output
        self.name = name                # name for this lineage, pertains to output files
        self.cache = {}                 # a cache for genotypes and data

    def initialise(self, founders = None, recalculate_fitness = False):
        # founders is a file path to a lineage.pkl file - a pkl file containing a list of lists of genotypes
        if type(founders) == str:

            # we have an input file of a lineage.pkl file
            # load it and set it to this lineage .lineage
            self.load_lineage(founders)

            if recalculate_fitness:
                # useful if you have different train / test data
                pop = self.recalculate_last_gen()
            
        else:
            # founders is an int of a population size
            self.random_population(founders)

    def add_ind_to_pop(self, ind, pop, X_train_sample, Y_train_sample, X_val_sample, Y_val_sample, notes = ""):
        # add an individual to a population, and cache it too

        # the individual ind has just a genotype, and no fitness yet
        if ind.g_md5 not in self.cache:
            # ind is not already in the cache
            # which means we need to calculate fitness etc. then add to cache
            ind.get_fitness(X_train_sample, Y_train_sample, X_val_sample, Y_val_sample)

            ind_summary = (ind.fitness, 
                           ind.training_time, 
                           ind.genotype, 
                           ind.test_time, 
                           ind.accuracy, 
                           notes)

            self.cache[ind.g_md5] = ind_summary

        else:

            ind_summary = self.cache[ind.g_md5]

        # add the individual
        pop.append(ind_summary)

        return(pop)



    def clean_up(self):
        # clean up a keras memory leak following https://stackoverflow.com/questions/48796619/why-is-tf-keras-inference-way-slower-than-numpy-operations
        clear_session()
        if keras.backend.tensorflow_backend._SESSION:
            tf.reset_default_graph()
            keras.backend.tensorflow_backend._SESSION.close()
            keras.backend.tensorflow_backend._SESSION = None

    def random_population(self, N):
        # make a population of random individuals
        X_val_sample, Y_val_sample = self.subsample_val()
        X_train_sample, Y_train_sample = self.subsample_train()

        pop = []
        for i in range(N):
            random_ind = Individual(self.input_shape, self.out_config, self.loss)
            
            random_ind.make_genotype()
            pop = self.add_ind_to_pop(random_ind, pop, X_train_sample, Y_train_sample, X_val_sample, Y_val_sample, notes = "random_initialisation")
            
            self.clean_up()
            del random_ind

        pop.sort(key=lambda tup: tup[0])
        self.lineage.append(pop)
        self.save_lineage()

        return(pop)
    
    def recalculate_last_gen(self):
        # make a population from a list of one or more genotypes
        pop = self.lineage[-1]
        genotypes = [x[2] for x in pop]

        X_val_sample, Y_val_sample = self.subsample_val()
        X_train_sample, Y_train_sample = self.subsample_train()

        pop = []
        for g in genotypes:
            ind = Individual(self.input_shape, self.out_config, self.loss, genotype = g)

            ind.make_genotype()
            pop = self.add_ind_to_pop(ind, pop, X_train_sample, Y_train_sample, X_val_sample, Y_val_sample, notes = "from_last")

            self.clean_up()
            del ind

        pop.sort(key=lambda tup: tup[0])
        self.lineage[-1] = pop

    def load_lineage(self, lineagepkl):
        with open(lineagepkl, 'rb') as myfile:
            self.lineage = pickle.load(myfile)

        # now we rebuild the cache from the loaded individuals
        for pop in self.lineage:
            for ind_summary in pop:
                g_md5 = reduced_genotype_md5(ind_summary[2])
                self.cache[g_md5] = ind_summary

            
    def subsample_val(self):
        # use a small random sample of the test data in each generation
        if self.valsize != None:
            val_sample = np.random.choice(list(range(self.X_val.shape[0])), size = self.valsize, replace = False)
            X_val_sample = self.X_val[val_sample, :]
            Y_val_sample = self.Y_val[val_sample, :]
        else:
            X_val_sample = self.X_val
            Y_val_sample = self.Y_val

        return(X_val_sample, Y_val_sample)

    def subsample_train(self):
        # use a small random sample of the test data in each generation
        if self.trainsize != None:
            train_sample = np.random.choice(list(range(self.X_train.shape[0])), size = self.trainsize, replace = False)
            X_train_sample = self.X_train[train_sample, :]
            Y_train_sample = self.Y_train[train_sample, :]
        else:
            X_train_sample = self.X_train
            Y_train_sample = self.Y_train

        return(X_train_sample, Y_train_sample)

    def choose_n_parents(self, pop, num_parents, selection):
        # randomly choose N parents from pop based on their fitness
        # return a list of genotypes
        parent_pop = pop.copy()

        fitness = [x[0] for x in parent_pop]

        parents = []
        for p in range(num_parents):

            if selection == 'weighted':
                fitness = [x[0] for x in parent_pop]
            elif selection == 'rank':
                # rank selection: the least fit is eliminated with fitness zero like this
                fitness = list(range(len(parent_pop)))
            elif selection == 'rank2':
                # rank fitness squared - more extreme advantage for the fittest
                fitness = list(range(len(parent_pop)))
                fitness = np.multiply(fitness, fitness)

            # now we choose a parent. Note that parents are chosen with replacement.
            choice = np.random.uniform(0, np.sum(fitness))
            cs = choice>np.cumsum(fitness)
            try:
                parent_index = max(np.where(cs == True)[0]) + 1
            except:
                parent_index = len(parent_pop) - 1 # if anything goes wrong, choose the fittest

            # parent is a tuple of (fitness, training_time, genotype)
            parent = parent_pop[parent_index]

            parents.append(parent[2])
            
        return(parents)

    def save_lineage(self):
        # record fitness and genotypes

        # write the fitness, training time, etc.
        pop = self.lineage[-1].copy()
        generation = len(self.lineage)

        with open(os.path.join(self.outputdir, '%s.tsv' %(self.name)), 'a') as csvout:
            for p in pop:
                csvout.write("%d\t%f\t%f\t%f\t%s\t%f\t%s\n" %(generation, p[4],p[1],p[3],p[2],p[0],p[5]))

        with open(os.path.join(self.outputdir, '%s.pkl') %(self.name), 'wb') as pklout:
            pickle.dump(self.lineage, pklout)

    def evolve(self, generations, num_parents = 2, keep = 0, kill = 0, kill_slow = 0, selection = "rank2", max_time = np.inf):
        # evolve a population
        # generations is a list of population sizes
        gen = 1
        for g in generations:

            # start with the most recent lineage
            pop = self.lineage[-1].copy()

            # subsample the validation and training data in each generation
            X_val_sample, Y_val_sample = self.subsample_val()
            X_train_sample, Y_train_sample = self.subsample_train()

            offspring = []


            
            # keep the fittest keep individuals in the population
            for i in range(len(pop)-keep, len(pop)):
                offspring.append(pop[i])

            # kill the worst ones
            for i in range(kill):
                del pop[0]

            # kill the slowest ones:
            pop.sort(key=lambda tup: tup[1])
            for i in range(kill_slow):
                print("killing training time:", pop[-1][1])
                del pop[-1]
            
            # kill anything with a training time > max_time
            for k, ind in enumerate(pop):
                if ind[1] < max_time:
                    del pop[k]

            # re-sort to fitness
            pop.sort(key=lambda tup: tup[0])

            # always kill individuals with fitness == 0; otherwise they can still participate in rank selection
            kill_zero = 0
            for i in range(len(pop)):
                if pop[i][0] == 0: kill_zero += 1
            
            for i in range(kill_zero):
                del pop[0]
                    
            # breed the rest of offspring from what's left of pop
            while len(offspring) < g:
                parents = self.choose_n_parents(pop, num_parents, selection)
                f1 = Individual(self.input_shape, self.out_config, self.loss, parents=parents)
                f1.make_genotype()
                offspring = self.add_ind_to_pop(f1, offspring, X_train_sample, Y_train_sample, X_val_sample, Y_val_sample, notes = selection)

                self.clean_up()
                del f1
                
            offspring.sort(key=lambda tup: tup[0])
            self.lineage.append(offspring)
            self.save_lineage()

            gen += 1
            current_ind = self.lineage[-1][-1]

    def explore(self, iterations, min=1.0):
        # evolve a lineage using hill climbing
        
        # get the best genotype
        current_ind = self.lineage[-1][-1]
        print("Starting genotype\n") 
        print_genotype(current_ind[2])
        print("Accuracy: ", current_ind[4])
        print("Traintime: ", current_ind[1])
        print("Fitness: ", current_ind[0])

        for i in range(iterations):
            # each iteration involves accepting a new genotype
            print("\n\niteration: ", i)

            # subsample the validation and training data in each generation
            X_val_sample, Y_val_sample = self.subsample_val()
            X_train_sample, Y_train_sample = self.subsample_train()

            # keep choosing a new individual until there's at least one mutation
            offspring_genotype = current_ind[2]
            current_ind_genotype = current_ind[2]
            while(offspring_genotype == current_ind_genotype):
                offspring = Individual(self.input_shape, self.out_config, self.loss, parents=[current_ind_genotype], mcmc=True)
                offspring.make_genotype()
                offspring_genotype = offspring.genotype

            pop = self.add_ind_to_pop(offspring, [], X_train_sample, Y_train_sample, X_val_sample, Y_val_sample, notes = "explore" + "_" + str(min))

            # get the offspring back out of the population
            offspring = pop[-1]

            acceptance_ratio = offspring[0] / current_ind[0]


            print_genotype(offspring[2])
            print("Accuracy: ", offspring[4])
            print("Traintime: ", offspring[1])
            print("Fitness: ", offspring[0])

            if acceptance_ratio > np.random.uniform(min, 1):
                
                current_ind = offspring
                print("Acceptance ratio: ", acceptance_ratio)
                print("New offspring accepted")

            self.lineage.append([current_ind])
            self.save_lineage()
            self.clean_up()
            del offspring


