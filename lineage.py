from individual import Individual

# these are to clear sessions to deal with this memory leak: https://github.com/keras-team/keras/issues/2102
import keras.backend.tensorflow_backend
import tensorflow as tf
from keras.backend import clear_session


import csv
import numpy as np
from pprint import pprint
import pickle


class Lineage(object):
    def __init__(self, input_shape, out_config, loss_function, X_train, Y_train, X_val, Y_val, 
                 trainsize = None, valsize = None, overlapping = False):
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

    def initialise(self, founders = None, recalculate_fitness = False):
        # founders is a file path to a population.pkl file - a pkl file containing a list of genotypes
        if type(founders) == str:
            # we have an input file of a population.pkl file
            if recalculate_fitness:
                # useful if you have different train / test data
                pop = self.founder_population(founders)
            else:
                pop = self.load_genotypes(founders)
        else:
            # founders is an int of a population size
            pop = self.random_population(founders)

        # sort on fitness
        pop.sort(key=lambda tup: tup[0])

        self.lineage.append(pop) # our first generation

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
            random_ind.get_fitness(X_train_sample, Y_train_sample, X_val_sample, Y_val_sample)
            pop.append((random_ind.fitness, random_ind.training_time, random_ind.genotype))

            self.clean_up()
            del random_ind

        return(pop)


    def founder_population(self, founders):
        # make a population from a list of one or more genotypes
        pop = self.load_genotypes(founders)
        genotypes = [x[2] for x in pop]

        X_val_sample, Y_val_sample = self.subsample_val()
        X_train_sample, Y_train_sample = self.subsample_train()

        pop = []
        for g in genotypes:
            ind = Individual(self.input_shape, self.out_config, self.loss, genotype = g)
            ind.get_fitness(X_train_sample, Y_train_sample, X_val_sample, Y_val_sample)
            pop.append((ind.fitness, ind.training_time, ind.genotype))

            self.clean_up()
            del ind

        return(pop)

    def load_genotypes(self, founder_pop):
        with open(founder_pop, 'rb') as myfile:
            pop = pickle.load(myfile)

        return(pop)


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
                parent_index = len(parent_pop) # if anything goes wrong, choose the fittest

            # parent is a tuple of (fitness, training_time, genotype)
            parent = parent_pop[parent_index]

            parents.append(parent[2])
            
        return(parents)

    def write_output(self, generation, pop):
        # record fitness and genotypes
        fitness = [x[0] for x in pop]
        fitness.insert(0,generation)
        with open('fitness.txt', 'a') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(fitness)

        with open('pop%d.pkl' %(generation), 'wb') as myfile:
            pickle.dump(pop, myfile)


    def evolve(self, generations, num_parents, keep = 0, kill = 0, selection = "rank2"):
        # evolve a population
        # generations is a list of population sizes

        for g in generations:

            print("Generation: ", len(self.lineage)) 

            # start with the most recent lineage
            pop = self.lineage[-1]
            fitness = [x[0] for x in pop]

            # subsample the validation and training data in each generation
            X_val_sample, Y_val_sample = self.subsample_val()
            X_train_sample, Y_train_sample = self.subsample_train()

            self.write_output(len(self.lineage), pop)

            offspring = []

            # keep the fittest keep individuals in the population
            for i in range(len(pop)-keep, len(pop)):
                offspring.append(pop[i])

            # kill the worst ones
            for i in range(kill):
                del pop[i]

            # breed the rest of offspring from what's left of pop
            while len(offspring) < g:
                fitness = [x[0] for x in pop]

                parents = self.choose_n_parents(pop, num_parents, selection)
                f1 = Individual(self.input_shape, self.out_config, self.loss, parents=parents)
                f1.get_fitness(X_train_sample, Y_train_sample, X_val_sample, Y_val_sample)
                offspring.append((f1.fitness, f1.training_time, f1.genotype))
                self.clean_up()
                del f1

            offspring.sort(key=lambda tup: tup[0])
            self.lineage.append(offspring)

            print("Fitness of previous generation")
            print(fitness)

            fitness = [x[0] for x in offspring]
            print("Fitness of offspring")
            print(fitness)


