# availalbe optimisers in keras
import numpy as np
import keras

# a list of tuples of optimiser names and default learning rates



def random_optimiser():
    
    optimisers = [("SGD", 0.01),
                  ("RMSprop", 0.001),
                  ("Adagrad", 0.01),
                  ("Adadelta", 1.0),
                  ("Adam", 0.001),
                  ("Adamax", 0.002),
                  ("Nadam", 0.002)
                  ]

    o = np.random.choice(range(6))

    optimiser = optimisers[o]

    return(optimiser)

def get_optimiser(name, learning_rate):

    if name == "SGD":
        optimiser = keras.optimizers.SGD(lr=learning_rate)
    elif name == "RMSprop":
        optimiser = keras.optimizers.RMSprop(lr=learning_rate)
    elif name == "Adagrad":
        optimiser = keras.optimizers.Adagrad(lr=learning_rate)
    elif name == "Adadelta":
        optimiser = keras.optimizers.Adadelta(lr=learning_rate)
    elif name == "Adam":
        optimiser = keras.optimizers.Adam(lr=learning_rate)
    elif name == "Adamax":
        optimiser = keras.optimizers.Adamax(lr=learning_rate)
    elif name == "Nadam":
        optimiser = keras.optimizers.Nadam(lr=learning_rate)
    else: #WTF
        print("Oh Dear")

    return(optimiser)
