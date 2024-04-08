import numpy as np

#Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def sigmoid_derivative(x):
    return x * (1 - x)

#Initializing network parameters
input_layer_size = 2  # Number of features in the input data
hidden_layer_size = 2  # Can be varied
output_layer_size = 1  # For binary classification

np.random.seed(42)  # For reproducibility
weights_input_hidden = np.random.uniform(size=(input_layer_size, hidden_layer_size))
weights_hidden_output = np.random.uniform(size=(hidden_layer_size, output_layer_size))
bias_hidden = np.random.uniform(size=(1, hidden_layer_size))
bias_output = np.random.uniform(size=(1, output_layer_size))

#Forward propagation function
def forward_propagate(inputs):
    hidden_layer_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_layer_activation = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_activation, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)
    return predicted_output

#Backpropagation function (Going back and adjusting weights based on the error
#of the output compared to the expected result)
def train(inputs, expected_output, learning_rate, iterations):
    global weights_input_hidden, weights_hidden_output, bias_hidden, bias_output
    for _ in range(iterations):
        # Forward propagation
        hidden_layer_input = np.dot(inputs, weights_input_hidden) + bias_hidden
        hidden_layer_activation = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_activation, weights_hidden_output) + bias_output
        predicted_output = sigmoid(output_layer_input)
        
        # Calculating the error
        error = expected_output - predicted_output
        d_predicted_output = error * sigmoid_derivative(predicted_output)
        
        # Backpropagation
        error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_activation)
        
        # Updating Weights and Biases
        weights_hidden_output += hidden_layer_activation.T.dot(d_predicted_output) * learning_rate
        weights_input_hidden += inputs.T.dot(d_hidden_layer) * learning_rate
        bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
        bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

#Training the neural network
inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
expected_output = np.array([[0], [1], [1], [0]])

learning_rate = 0.1
iterations = 10000

train(inputs, expected_output, learning_rate, iterations)

#Test the network after training
print(forward_propagate(inputs))
