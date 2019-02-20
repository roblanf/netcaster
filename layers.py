from mutation import *
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
import numpy as np

base_mr = 0.1 # a simple global default for initalising all mutation rates

def mutate_layer(layer):
    if layer["type"] == "full":
        layer = mutate_full_layer(layer)

    if layer["type"] == "conv":
        layer = mutate_conv_layer(layer)

    if layer["type"] == "pool":
        layer = mutate_pool_layer(layer)

    return(layer)



def add_conv_layer(layer, input_shape):

    if input_shape == False:
        new_layer = Conv2D(filters = layer["filters"],
                           kernel_size = (layer["kernel"], layer["kernel"]),
                           strides = (layer["strides"], layer["strides"]),
                           padding = layer["padding"],
                           activation = "relu")
    else:
        new_layer = Conv2D(filters = layer["filters"],
                           kernel_size = (layer["kernel"], layer["kernel"]),
                           strides = (layer["strides"], layer["strides"]),
                           padding = layer["padding"],
                           activation = "relu",
                           input_shape = input_shape)

    return(new_layer)

def mutate_conv_layer(layer):

    # add or subtract a filter
    layer["filters"] = mutate_int_fixed(layer["filters"], 1, [1, 1000], layer["mutrate"])
    layer["kernel"] = mutate_int_fixed(layer["kernel"], 1, [1, 1000], layer["mutrate"])
    layer["strides"] = mutate_int_fixed(layer["strides"], 1, [1, 1000], layer["mutrate"])

    if np.random.uniform(0, 1) < layer["mutrate"]:
        layer["padding"] = np.random.choice(["valid", "same"])

    layer["dropout"] = mutate_product(layer["dropout"], 1.05, [0.0, 1.0], layer["mutrate"])

    if np.random.uniform(0, 1) < layer["mutrate"]/10: # 1/10th of the mutation rate, because this is a big change
        layer["norm"] = np.random.choice([0, 1])

    layer["mutrate"] = mutate_product(layer["mutrate"], 1.05, [0.0, 1.0], layer["mutrate"])

    return(layer)


def random_conv_layer():

    conv_layer =    {"type": "conv",
                     "filters": np.random.randint(2,24),
                     "kernel": np.random.randint(1,10),
                     "strides": np.random.randint(1,10),
                     "padding": np.random.choice(["valid", "same"]),
                     "mutrate": base_mr,
                     "dropout": np.random.uniform(0.0, 0.99),
                     "norm": np.random.choice([0, 1])}

    return(conv_layer)



def random_cp_layer():

    # this makes sure we don't try and add conv/pool
    # layers after dense layers
    r = np.random.choice(["conv", "pool"])

    if r == "conv":
        return(random_conv_layer())
    if r == "pool":
        return(random_pool_layer())



def add_pool_layer(layer, input_shape):

    if input_shape == False:
        new_layer = MaxPooling2D(pool_size = (layer["pool_size"], layer["pool_size"]),
                                 strides = (layer["strides"], layer["strides"]),
                                 padding = layer["padding"])

    else:
        new_layer = MaxPooling2D(pool_size = (layer["pool_size"], layer["pool_size"]),
                                 strides = (layer["strides"], layer["pool_size"]),
                                 padding = layer["padding"],
                                 input_shape = input_shape)

    return(new_layer)


def mutate_pool_layer(layer):

    layer["pool_size"] = mutate_int_fixed(layer["pool_size"], 1, [1, 1000], layer["mutrate"])
    layer["strides"] = mutate_int_fixed(layer["strides"], 1, [1, 1000], layer["mutrate"])

    if np.random.uniform(0, 1) < layer["mutrate"]:
        layer["padding"] = np.random.choice(["valid", "same"])

    layer["mutrate"] = mutate_product(layer["mutrate"], 1.05, [0.0, 1.0], layer["mutrate"])

    return(layer)

def random_pool_layer():

    pool_layer =    {"type": "pool",
                     "pool_size": np.random.randint(2,4),
                     "strides": np.random.randint(1,3),
                     "padding": np.random.choice(["valid", "same"]),
                     "mutrate": base_mr}

    return(pool_layer)

def add_full_layer(layer, input_shape):

    if input_shape == False:
        new_layer = Dense(units = layer["units"],
                          activation = "relu")

    else:
        new_layer = Dense(units = layer["units"],
                          activation = "relu",
                          input_shape = input_shape)

    return(new_layer)

def mutate_full_layer(layer):

    layer["units"] = mutate_int(layer["units"], 0.1, [1, 10000], layer["mutrate"])
    layer["dropout"] = mutate_product(layer["dropout"], 1.05, [0.0, 1.0], layer["mutrate"])

    if np.random.uniform(0, 1) < layer["mutrate"]/10:
        layer["norm"] = np.random.choice([0, 1])

    layer["mutrate"] = mutate_product(layer["mutrate"], 1.05, [0.0, 1.0], layer["mutrate"])

    return(layer)


def random_full_layer():

    full_layer =    {"type": "full",
                     "units": np.random.randint(4,200),
                     "mutrate": base_mr,
                     "dropout": np.random.uniform(0.0, 0.99),
                     "norm": np.random.choice([0, 1])
                     }

    return(full_layer)

def print_layer(layer):

    if layer["type"] == "full":
        print("full: %d; d %.2f; n %d; m %.2f"  %(layer["units"], layer["dropout"], layer["norm"], layer["mutrate"]))

    if layer["type"] == "conv":
        print("conv: %dx%dx%d, s=%d, p=%s; d %.2f; n %d; m %.2f" %(layer["kernel"], layer["kernel"], layer["filters"], layer["strides"], layer["padding"], layer["dropout"], layer["norm"], layer["mutrate"]))

    if layer["type"] == "pool":
        print("pool: %dx%d, s=%d, p=%s; m %.2f" %(layer["pool_size"], layer["pool_size"], layer["strides"], layer["padding"], layer["mutrate"]))
