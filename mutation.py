import numpy as np

def mutate_optimiser(optimiser, learning_rate, size, limits, mutrate):
    # mutate the optimiser. if the optimiser changes, we set the learning rate

    if np.random.uniform(0, 1) < 0: # Turn this back on if you want to explore other optimisers than Adam
        # get a new optimiser and get the default learning rate for that.
        optimiser, learning_rate = optimisers.random_optimiser()

    if np.random.uniform(0, 1) < mutrate:
        learning_rate = mutate_product(learning_rate, size, [0, 1], mutrate)

    return(optimiser, learning_rate)

def mutate_batchsize(batchsize, mutrate, min_size = 8, max_size = 8192):
    # change a batch size up or down by a factor of 2
    # within bounds set by min/max _size

    if np.random.uniform(0, 1) < mutrate:
        new_batchsize = np.int(np.random.choice([0.5, 2]) * batchsize)
    else:
        return(batchsize)

    if new_batchsize < min_size: new_batchsize = min_size
    if new_batchsize > max_size: new_batchsize = max_size

    return new_batchsize

def mutate_int_fixed(value, size, limits, mutrate):
    # mutate an integer up or down by a value up to size, within limits
    change = 0

    if np.random.uniform(0, 1) < mutrate:
        change = list(range(-1*size, size + 1))
        change.remove(0) # because we've already agreed we;ll mutate
        change = np.random.choice(change)

    value = value + change

    if value < limits[0]: value = limits[0]
    if value > limits[1]: value = limits[1]

    return(value)

def mutate_float_fixed(value, size, limits, mutrate):
    # mutate a float by + or - size

    change = 0

    if np.random.uniform(0, 1) < mutrate:
        change = [-1*size, size]
        change = np.random.choice(change)

    value = value + change

    if value < limits[0]: value = limits[0]
    if value > limits[1]: value = limits[1]

    return(value)


def mutate_int(value, size, limits, mutrate):
    # change an int by at least 1..., up to proportion size

    if np.random.uniform(0, 1) < mutrate:

        min_mut = 1
        max_mut = np.int(np.random.uniform(1.0, (1.0+size)) * value - value)

        # choose a size from -size to +size
        if min_mut < max_mut:
            mutation_size = np.random.randint(min_mut, max_mut)
        else:
            mutation_size = min_mut

        if np.random.uniform(0, 1) < 0.5: 
            mutation_size = mutation_size * -1

        # mutate the value
        value = value + mutation_size

        # check the limits
        if value < limits[0]: value = limits[0]
        if value > limits[1]: value = limits[1]

    return value



def mutate_addition(value, size, limits, mutrate):
    # mutate a float of value, by amount size, within limits

    if np.random.uniform(0, 1) < mutrate:

        # choose a size from -size to +size
        mutation_size = np.random.uniform(-1*size, size)

        # mutate the value
        value = value + mutation_size

        # check the limits
        if value < limits[0]: value = limits[0]
        if value > limits[1]: value = limits[1]

    return value

def mutate_product(value, size, limits, mutrate, output="float", dist="uniform"):
    # mutate a float of value, by amount size, within limits
    if np.random.uniform(0, 1) < mutrate:

        # choose a size from -size to +size
        # equal probability - and +
        if dist=="uniform":
            mutation_size = np.random.uniform(1.0, size)

        if dist=="exponential":
            # we add 1 so that it's always >1, then we * or / later
            mutation_size = 1 + np.random.exponential(scale = size)

        if np.random.uniform(0, 1) < 0.5:
            value = value / mutation_size
        else:
            value = value * mutation_size

        # check the limits
        if value < limits[0]: value = limits[0]
        if value > limits[1]: value = limits[1]

        if output == "int":
            np.int(np.round(value))

    return value

def mutate_layer(layer, mutrate):
    if layer["type"] == "full":
        layer = mutate_full_layer(layer, mutrate)

    if layer["type"] == "conv":
        layer = mutate_conv_layer(layer, mutrate)

    if layer["type"] == "pool":
        layer = mutate_pool_layer(layer, mutrate)

    if layer["type"] == "dropout":
        layer = mutate_dropout_layer(layer, mutrate)

    return(layer)

def mutate_dropout_layer(layer, mutrate):

    layer["dropout"] = mutate_float_fixed(layer["dropout"], 0.1, [0.1, 0.9], mutrate)

    return(layer)

def mutate_conv_layer(layer, mutrate):

    layer["filters"] = mutate_product(layer["filters"], 2, [1, 2000], mutrate, output="int", dist="exponential")
    layer["kernel_h"] = mutate_product(layer["kernel_h"], 2, [1, 2000], mutrate, output="int", dist="exponential")
    layer["kernel_w"] = mutate_product(layer["kernel_w"], 2, [1, 2000], mutrate, output="int", dist="exponential")
    layer["strides_h"] = mutate_product(layer["strides_h"], 2, [1, 2000], mutrate, output="int", dist="exponential")
    layer["strides_w"] = mutate_product(layer["strides_w"], 2, [1, 2000], mutrate, output="int", dist="exponential")

    if np.random.uniform(0, 1) < mutrate:
        layer["padding"] = np.random.choice(["valid", "same"])

    return(layer)


def mutate_pool_layer(layer, mutrate):

    layer["pool_size_h"] = mutate_product(layer["pool_size_h"], 2, [1, 2000], mutrate, output="int", dist="exponential")
    layer["pool_size_w"] = mutate_product(layer["pool_size_w"], 2, [1, 2000], mutrate, output="int", dist="exponential")
    layer["strides_h"] = mutate_product(layer["strides_h"], 2, [1, 2000], mutrate, output="int", dist="exponential")
    layer["strides_w"] = mutate_product(layer["strides_w"], 2, [1, 2000], mutrate, output="int", dist="exponential")

    if np.random.uniform(0, 1) < mutrate:
        layer["padding"] = np.random.choice(["valid", "same"])

    return(layer)


def mutate_full_layer(layer, mutrate):

    layer["units"] = mutate_int(layer["units"], 0.2, [1, 10000], mutrate)

    return(layer)


