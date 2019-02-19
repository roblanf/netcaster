def mutate_layer(self, layer):
    if layer["type"] == "full":
        layer = self.mutate_full_layer(layer)

    if layer["type"] == "conv":
        layer = self.mutate_conv_layer(layer)

    if layer["type"] == "pool":
        layer = self.mutate_pool_layer(layer)

    return(layer)



def add_conv_layer(self, layer, input_shape):

    if input_shape == False:
        new_layer = Conv2D(filters = layer["filters"],
                           kernel_size = layer["kernel"],
                           strides = layer["strides"],
                           padding = layer["padding"],
                           activation = "relu")
    else:
        new_layer = Conv2D(filters = layer["filters"],
                           kernel_size = layer["kernel"],
                           strides = layer["strides"],
                           padding = layer["padding"],
                           activation = "relu",
                           input_shape = input_shape)

    return(new_layer)

def mutate_conv_layer(self, layer):

    # add or subtract a filter
    layer["filters"] = mutate_int_fixed(layer["filters"], 1, [1, 1000], layer["mutrate"])
    layer["kernel"] = mutate_int_fixed(layer["kernel"], 1, [1, 1000], layer["mutrate"])
    layer["strides"] = mutate_int_fixed(layer["strides"], 1, [1, 1000], layer["mutrate"])

    if np.random.uniform(0, 1) < layer["mutrate"]:
        layer["padding"] = np.random.choice(["valid", "same"])

    layer["dropout"] = self.mutate_addition(layer["dropout"], 0.05, [0.0, 1.0], layer["mutrate"])

    if np.random.uniform(0, 1) < layer["mutrate"]/10: # 1/10th of the mutation rate, because this is a big change
        layer["norm"] = np.random.choice([0, 1])

    layer["mutrate"] = self.mutate_addition(layer["mutrate"], 0.05, [0.0, 1.0], layer["mutrate"])

    return(layer)


def random_conv_layer(self):

    conv_layer =    {"type": "conv",
                     "filters": np.random.randint(2,24),
                     "kernel": np.random.randint(1,10),
                     "strides": np.random.randint(1,10),
                     "padding": np.random.choice(["valid", "same"]),
                     "mutrate": 0.05,
                     "dropout": np.random.uniform(0.0, 0.99),
                     "norm": np.random.choice([0, 1])}

    return(conv_layer)






def random_cp_layer(self):

    # this makes sure we don't try and add conv/pool
    # layers after dense layers
    r = np.random.choice(["conv", "pool"])

    if r == "conv":
        return(self.random_conv_layer())
    if r == "pool":
        return(self.random_pool_layer())



def add_pool_layer(self, layer, input_shape):

    if input_shape == False:
        new_layer = MaxPooling2D(pool_size = layer["pool_size"],
                                 strides = layer["strides"],
                                 padding = layer["padding"])

    else:
        new_layer = MaxPooling2D(pool_size = layer["pool_size"],
                                 strides = layer["strides"],
                                 padding = layer["padding"],
                                 input_shape = input_shape)

    return(new_layer)


def mutate_pool_layer(self, layer):

    layer["pool_size"] = self.mutate_int(layer["pool_size"], 0.1, [2, 1000], layer["mutrate"])
    layer["strides"] = self.mutate_int(layer["strides"], 0.1, [1, 100], layer["mutrate"])

    if np.random.uniform(0, 1) < layer["mutrate"]:
        layer["padding"] = np.random.choice(["valid", "same"])

    layer["mutrate"] = self.mutate_addition(layer["mutrate"], 0.1, [0.0, 1.0], layer["mutrate"])

    return(layer)

def random_pool_layer(self):

    pool_layer =    {"type": "pool",
                     "pool_size": np.random.randint(2,4),
                     "strides": np.random.randint(1,3),
                     "padding": np.random.choice(["valid", "same"]),
                     "mutrate": np.random.uniform(0.01, 0.3)}

    return(pool_layer)

def add_full_layer(self, layer, input_shape):

    if input_shape == False:
        new_layer = Dense(units = layer["units"],
                          activation = "relu")

    else:
        new_layer = Dense(units = layer["units"],
                          activation = "relu",
                          input_shape = input_shape)

    return(new_layer)

def mutate_full_layer(self, layer):

    layer["units"] = self.mutate_int(layer["units"], 0.1, [1, 10000], layer["mutrate"])
    layer["dropout"] = self.mutate_addition(layer["dropout"], 0.1, [0.0, 1.0], layer["mutrate"])

    if np.random.uniform(0, 1) < layer["mutrate"]:
        layer["norm"] = np.random.choice([0, 1])

    if np.random.uniform(0, 1) < layer["mutrate"]:
        layer["residual"] = np.random.choice([0, 1])


    layer["mutrate"] = self.mutate_addition(layer["mutrate"], 0.1, [0.0, 1.0], layer["mutrate"])

    return(layer)


def random_full_layer(self):

    full_layer =    {"type": "full",
                     "units": np.random.randint(4,200),
                     "mutrate": np.random.uniform(0.01, 0.3),
                     "dropout": np.random.uniform(0.0, 0.99),
                     "norm": np.random.choice([0, 1]),
                     "residual": np.random.choice([0, 1])}

    return(full_layer)

def print_layer(self, layer):

    if layer["type"] == "full":
        print("full: %d"  % layer["units"])

    if layer["type"] == "conv":
        print("conv: %dx%dx%d, s=%d, p=%s" %(layer["kernel"], layer["kernel"], layer["filters"], layer["strides"], layer["padding"]))

    if layer["type"] == "pool":
        print("pool: %dx%d, s=%d, p=%s" %(layer["pool_size"], layer["pool_size"], layer["strides"], layer["padding"]))
