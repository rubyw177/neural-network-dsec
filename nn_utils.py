import numpy as np
import matplotlib.pyplot as plt

def normalize_image(image):
    normalized_image = image/255.0
    return normalized_image

def relu(z):
    a = np.maximum(0, z)
    return a

def tanh(x):
    return np.tanh(x)

def softmax(x):
    e = np.exp(x)
    return e/np.sum(e, axis = 0)

def backward_tanh(x):
    return (1 - np.power(np.tanh(x), 2))

def backward_relu(x):
    return np.array(x > 0, dtype = np.float32)

def init_parameters(network_dims):
    np.random.seed(69)
    parameters = {}
    L = len(network_dims)

    for i in range(1, L):
        parameters["w" + str(i)] = np.random.randn(network_dims[i], network_dims[i-1]) * 0.01
        parameters["b" + str(i)] = np.zeros((network_dims[i], 1))

    return parameters

def forward_propagation(X, parameters):
    forward_cache = {}
    L = len(parameters) // 2
    forward_cache["a0"] = X
    a_prev = forward_cache["a0"]
    
    # loop through layer 2 to last layer
    for i in range(1, L+1):
        a_prev = forward_cache["a" + str(i-1)]
        w = parameters["w" + str(i)]
        b = parameters["b" + str(i)]

        if i == L:
            z = np.dot(w, a_prev) + b
            a = softmax(z)
            forward_cache["z" + str(L)] = z
            forward_cache["a" + str(L)] = a

        else:
            z = np.dot(w, a_prev) + b
            a = tanh(z)
            forward_cache["z" + str(i)] = z
            forward_cache["a" + str(i)] = a

        a_prev = a

    return forward_cache["a" + str(L)], forward_cache

def compute_cost(last_a, y):
    assert(last_a.shape == y.shape)
    m = y.shape[1]

    # compute categorical crossentropy cost
    cost = (-1./m) * np.sum(y * np.log(last_a + 1e-15))  # add 1e-15 so that we don't get log(0)
    cost = np.squeeze(cost)
    return cost

def backward_propagation(aL, y, parameters, forward_cache):
    gradients = {}
    L = len(parameters) // 2
    m = aL.shape[1]

    # compute gradients for output layer
    gradients["dz" + str(L)] = aL - y
    gradients["dw" + str(L)] = (1./m) * np.dot(gradients["dz" + str(L)], forward_cache["a" + str(L-1)].T)
    gradients["db" + str(L)] = (1./m) * np.sum(gradients["dz" + str(L)], axis=1, keepdims=True)

    for i in reversed(range(1, L)):
        gradients["dz" + str(i)] = np.dot(parameters["w" + str(i+1)].T, gradients["dz" + str(i+1)]) * backward_tanh(forward_cache["a" + str(i)])
        gradients["dw" + str(i)] = (1./m) * np.dot(gradients["dz" + str(i)], forward_cache["a" + str(i-1)].T)
        gradients["db" + str(i)] = (1./m) * np.sum(gradients["dz" + str(i)], axis=1, keepdims=True)

    return gradients


def update_parameters(parameters, gradients, learning_rate=0.01):
    L = len(parameters) // 2
 
    for i in range(1, L+1):
        parameters["w" + str(i)] = parameters["w" + str(i)] - learning_rate * gradients["dw" + str(i)]
        parameters["b" + str(i)] = parameters["b" + str(i)] - learning_rate * gradients["db" + str(i)]
    
    return parameters

def plot_cost(costs, learning_rate, title):
    plt.figure(figsize=(10, 4))
    plt.plot(np.squeeze(costs), color="#f39530")
    plt.ylabel("Cost (average loss)")
    plt.xlabel("Epoch (iterations)")
    plt.title(f"{title} using learning rate {learning_rate}")
    plt.show()

def create_model(X_train, X_val, y_train, y_val, network_dims, epochs, learning_rate=0.01, verbose=True, plot=True):
    epoch_color = "\033[94m"  # Yellow color
    train_cost_color = "\033[92m"  # Green color
    val_cost_color = "\033[93m"  # Blue color
    reset_color = "\033[0m"  # Reset color to default
    train_costs = []
    val_costs = []

    # initialize parameters
    parameters = init_parameters(network_dims)
    L = len(parameters) // 2

    for epoch in range(1, epochs+1):
        aL_train, forward_cache_train = forward_propagation(X_train, parameters)
        aL_val, forward_cache_val = forward_propagation(X_val, parameters)

        # compute cost
        train_cost = compute_cost(aL_train, y_train)
        val_cost = compute_cost(aL_val, y_val)

        # compute gradients
        gradients = backward_propagation(aL_train, y_train, parameters, forward_cache_train)

        # update parameters using gradient descent
        parameters = update_parameters(parameters, gradients, learning_rate)

        train_costs.append(train_cost)
        val_costs.append(val_cost)

        if verbose and ((epoch % 50 == 0) or (epoch == epochs)):
            print(f"epoch: {epoch_color}{epoch}{reset_color} -- training cost: {train_cost_color}{round(train_cost, 4)}{reset_color} -- validation cost: {val_cost_color}{round(val_cost, 4)}{reset_color}")

    # plot the cost
    if plot:
        plot_cost(train_costs, learning_rate, "Training cost")
        plot_cost(val_costs, learning_rate, "Validation cost")

    return parameters

def predict(X, parameters):
    # preprocess data before feeding it to model
    X_flatten = X.reshape(X.shape[0], 1)

    # forward feed the preprocessed data to model
    aL, forward_cache = forward_propagation(X_flatten, parameters)
    prediction = np.argmax(aL, 0)
    return prediction[0]



    





    


        
    