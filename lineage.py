
import Individual

# these are to clear sessions to deal with this memory leak: https://github.com/keras-team/keras/issues/2102
import keras.backend.tensorflow_backend
import tensorflow as tf
from keras.backend import clear_session

class Lineage(object):
    def __init__(self, N, out_config, X_train, Y_train, X_val, Y_val, keep = 0, kill = 0,  
    			 trainsize = None, testsize = None, overlapping = False, selection = 'rank2'):
        """An lineage of individuals that can evolve"""
        self.N = N 				# Number of individuals per generation
        self.out_config = out_config	# the keras config of the output layer (e.g.  'Dense(units = 10, activation = 'softmax').get_config()')
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.keep = keep                # number of fittest individuals to keep in each generation
        self.kill = kill                # number of least fit individuals to kill in each generation
        self.trainsize = trainsize		# train on a subset of the data. None = use all data
        self.valsize = valsize		# test  on a subset of the data. None = use all data
        self.overlapping = overlapping  # True or False. True = overlapping generation. 
        self.selection = selection		# type of selection: "weighted", "rank", or "rank2" (rank squared)


        self.lineage = []				# a list to contain the lists of genotype / fitness tuples that represent each population

        def initialise(self, founders = None):
            # founders is a file path to a genotypes.pkl file - a pkl file containing a list of genotypes
        	if founders:
        		pop = self.founder_population(founderss)
        	else:
        		pop = self.random_population()

			# sort on fitness
			pop.sort(key=lambda tup: tup[0])
	
        	lineage.append(pop) # our first generation

        def clean_up(self):
        	# clean up a keras memory leak following https://stackoverflow.com/questions/48796619/why-is-tf-keras-inference-way-slower-than-numpy-operations
        	clear_session()
			if keras.backend.tensorflow_backend._SESSION:
			    tf.reset_default_graph()
			    keras.backend.tensorflow_backend._SESSION.close()
			    keras.backend.tensorflow_backend._SESSION = None

        def random_population(self):
        	# make a population of random individuals
        	pop = []
			for i in range(self.N):
			    random_ind = Individual(self.input_shape, self.out_config, self.loss)
			    random_ind.get_fitness(self.X_train, self.Y_train, self.X_val, self.Y_val)
			    pop.append((random_ind.fitness, random_ind.training_time, random_ind.genotype))

			    self.clean_up()
			    del random_ind

			return(pop)


		def founder_population(self, founders):
			# make a population from a list of one or more genotypes
            genotypes = load_genotypes(founders)
			pop = []
			for g in genotypes:
			    ind = Individual(self.input_shape, self.out_config, self.loss, genotype = g)
			    ind.get_fitness(X_train, Y_train, X_val, Y_val)
			    pop.append((ind.fitness, ind.training_time, ind.genotype))

			    self.clean_up()
			    del ind

			return(pop)

        def load_genotypes(self, founders):
            with open(founders, 'rb') as myfile:
                genotypes = pickle.load(myfile)

            return(genotypes)


        def subsample_val():
            # use a small random sample of the test data in each generation
            if self.valsize != None:
                val_sample = np.random.choice(list(range(X_val.shape[0])), size = self.valsize, replace = False)
                X_val_sample = self.X_val[val_sample, :]
                Y_val_sample = self.Y_val[val_sample, :]
            else:
                X_val_sample = X_val
                Y_val_sample = Y_val

            return(X_val_sample, Y_val_sample)

        def subsample_train():
            # use a small random sample of the test data in each generation
            if self.trainsize != None:
                train_sample = np.random.choice(list(range(X_train.shape[0])), size = self.trainsize, replace = False)
                X_train_sample = self.X_train[train_sample, :]
                Y_train_sample = self.Y_train[train_sample, :]
            else:
                X_val_sample = X_val
                Y_val_sample = Y_val

            return(X_val_sample, Y_val_sample)

		def choose_n_parents(self, pop):
		    # randomly choose N parents from pop based on their fitness
            # return a list of genotypes
		    parent_pop = pop.copy()

		    parents = []
		    for p in range(N):

		        if type == 'weighted':
		            fitness = [x[0] for x in parent_pop]
		        elif type == 'rank':
		            # rank selection: the least fit is eliminated with fitness zero like this
		            fitness = list(range(len(parent_pop)))
		        elif type == 'rank2':
                    # rank fitness squared - more extreme advantage for the fittest
		            fitness = list(range(len(parent_pop)))
                    fitness = np.multiply(fitness, fitness)

                # now we choose a parent. Note that parents are chosen with replacement.
		        choice = np.random.uniform(0, np.sum(fitness))
		        cs = choice>np.cumsum(fitness)
		        try:
		            parent_index = max(np.where(cs == True)[0])
		        except:
		            parent_index = 0

                # parent is a tuple of (fitness, training_time, genotype)
                parent = parent_pop[parent_index]

		        print('chose parent', parent_index, 'with fitness', parent[0], 'training_time', parent[1])

		        parents.append(parent[2])
		        
		    return(parents)

        def write_output(generation, pop):
            # record fitness and genotypes
            fitness = [x[0] for x in pop]
            fitness.insert(0,g)
            with open('fitness.txt', 'a') as myfile:
                wr = csv.writer(myfile)
                wr.writerow(fitness)

            genotypes = [x[2] for x in pop]
            with open('genotypes%d.pkl' %(g), 'wb') as myfile:
                pickle.dump(genotypes, myfile)


        def evolve(self, num_generations):
            # evolve a population

            for g in range(num_generations):
                
                # start with the most recent lineage
                pop = self.lineage[-1]
                pprint(pop)

                # subsample the validation and training data in each generation
                X_val_sample, Y_val_sample = self.subsample_val()
                X_train_sample, Y_train_sample = self.subsample_train()

                self.write_output(g, pop)

                offspring = []

                # keep the fittest keep individuals in the population
                for i in range(len(pop)-keep, len(pop)):
                    offspring.append(pop[i])

                # kill the worst ones
                for i in range(kill):
                    del pop[i]

                # breed the rest
                while len(offspring) < N:
                    parents = self.choose_n_parents(pop, self.parents, self.selection)
                    f1 = Individual(self.input_shape, self.out_config, self.loss, parents=parents)
                    f1.get_fitness(X_train_sample, Y_train_sample, X_val_sample, Y_val_sample)
                    offspring.append((f1.fitness, f1.training_time, f1.genotype))
                    self.clean_up()
                    del f1

                pop.sort(key=lambda tup: tup[0])
                self.lineage.append(pop)
