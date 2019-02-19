# availalbe optimisers in keras
import numpy as np
import keras

# a list of tuples of optimiser names and default learning rates

def all_optimisers():

	optimisers = [("SGD", 0.01),
				  ("RMSprop", 0.001),
				  ("Adagrad", 0.01),
				  ("Adadelta", 1.0),
				  ("Adam", 0.001),
				  ("Adamax", 0.002),
				  ("Nadam", 0.002)]

	return(optimisers)

def random_optimiser();
	
	optimiser = np.random.choice(all_optimisers())

	return(optimiser)

def get_optimiser(name):

    if name == "SGD":
        optimiser = keras.optimizers.SGD(lr=p["learning_rate"])
    elif name == "RMSprop":
        optimiser = keras.optimizers.RMSprop(lr=p["learning_rate"])
    elif name == "Adagrad":
        optimiser = keras.optimizers.Adagrad(lr=p["learning_rate"])
    elif name == "Adadelta":
        optimiser = keras.optimizers.Adadelta(lr=p["learning_rate"])
    elif name == "Adam":
        optimiser = keras.optimizers.Adam(lr=p["learning_rate"])
    elif name == "Adamax":
        optimiser = keras.optimizers.Adamax(lr=p["learning_rate"])
    elif name == "Nadam":
        optimiser = keras.optimizers.Nadam(lr=p["learning_rate"])
    else: #WTF
        print("Oh Dear")

    return(optimiser)
