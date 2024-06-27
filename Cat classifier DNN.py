import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from dnn_functions import *

# Load data
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Dataset parameters
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

# Example of a picture
index = 10
plt.imshow(train_x_orig[index])
print ("y = " + str(train_y[:, index]) + ", it's a '" + classes[np.squeeze(train_y[:, index])].decode("utf-8") +  "' picture.")
plt.show()

# Flatten and standardize the dataset
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

# Set plotting parameters
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Define the architecture of the neural network
layers_dims = [12288, 20, 7, 5, 1] # 4-layer model

def dnn(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    costs = [] # keeping track of cost using an array
    
    # Parameters initialization
    parameters = initialize_params(layers_dims)
    
    # Loop (gradient descent)
    for i in range(num_iterations):
        # Forward propagation
        AL, caches = forward(X, parameters)
        
        # Compute cost
        cost = compute_cost(AL, Y)
        
        # Backward propagation
        grads = backward(AL, Y, caches)
        
        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the cost every 100 iterations
        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)
    
    return parameters, costs

# Train the model
parameters, costs = dnn(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)

# Predict on training and test sets
pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)

# Example of predicting with your own image
my_image = "cat2.jpg"
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)

fname = "images/" + my_image
image = np.array(Image.open(fname).resize((num_px, num_px)))
plt.imshow(image)
plt.show()

image = image / 255.
image = image.reshape((1, num_px * num_px * 3)).T
my_predicted_image = predict(image, my_label_y, parameters)

print ("y = " + str(np.squeeze(my_predicted_image)) + ", model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
