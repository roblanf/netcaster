# what an individual's genotype looks like
# contains hard-coded defaults for initialisatino

class Genotype(object):
    def __init__(self):
        """An individual network"""
        self.cplayers = None
        self.network = None
        self.epochs = None

    def generate_random(self):
    	# generate a random CNN genotype with sensible defaults

        # number of conv and pooling layers at the start
        num_cp_layers = np.random.randint(2, 4)

        # number of fully connected layers
        num_d_layers = np.random.randint(1, 2)

        # genotype hyperparameters
        # to get a random optimiser:
        opt = random_optimiser()

        # to use Adam:
        opt = ("Adam", 0.001)

        params =   {"mutation": 0.05,
                    "recomb": 1.0, # generally we expect higher recombination to be better
                    "optimiser": opt[0],
                    "learning_rate": opt[1],
                    "learning_rate_ms": 1.1, # mutation size for learning rate
                    "slip": 0.05, # chance of adding new layers
                    "slip_ms", 0.05,
                    "epoch_ms", 0.05 # mutation rate applied to epochs
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
            training.append(np.random.choice([16, 32, 64, 128])) # add minibatch sizes for epochs

        # genotype has recombination rate, then network arch, then training
        genotype = {'params': params, 'network': network, 'training': training}

        self.genotype = genotype
