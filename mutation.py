import optimisers

def mutate_optimiser(optimiser, learning_rate, size, limits, mutrate):
    # mutate the optimiser. if the optimiser changes, we set the learning rate

    if np.random.uniform(0, 1) < mutrate:
        return(random_optimiser)

    else: # we didn't change the optimiser, so try mutating the learning rate
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

    if np.random.uniform(0, 1) < mutrate:
        change = list(range(-1*size, size + 1))
        change.remove(0) # because we've already agreed well mutate

    value = value + change

    if value < limits[0]: value = limits[0]
    if value > limits[1]: value = limits[1]

    return(value)

def mutate_int(self, value, size, limits, mutrate):
    # change an int by at least 1..., up to size %

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


def mutate_addition(self, value, size, limits, mutrate):
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

def mutate_product(self, value, size, limits, mutrate):
    # mutate a float of value, by amount size, within limits

    if np.random.uniform(0, 1) < mutrate:

        # choose a size from -size to +size
        mutation_size = np.random.uniform(1.0, size)
        if np.random.uniform(0, 1) < 0.5:
            value = value / mutation_size
        else:
            value = value * mutation_size

        # check the limits
        if value < limits[0]: value = limits[0]
        if value > limits[1]: value = limits[1]

    return value
