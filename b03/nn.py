import numpy as np
import progressbar

input_data = np.array([[1, 2, 3, 4],
			[1, 2, 3, 5],
			[1, 2, 3, 6],
			[1, 2, 3, 7], 
			[1, 2, 3, 8]])
output_data = np.array([4, 5, 6, 7, 8])

print("Input Data : ", input_data)
print("Output Data : ", output_data)


"""
Input Layer -> hidden layer -> output_layer
"""

def buildNetwork(input_shape, hidden_layer_size, output_shape):
	"""
	input: 
		input_shape - tuple/list of length 1
		hidden_layer_size - tuple/list of number of nodes in the hidden layer,
					lenght of list = number of hiddden layers
		output_shape - tuple/list of length 1

	builds a NN with random weights and bias
	"""
	weights = list()
	bias = list()
	weights.append(np.random.rand(input_shape, hidden_layer_size[0]))
	for index in range(len(hidden_layer_size)):
		if index != len(hidden_layer_size) - 1: 
			weights.append(np.random.rand(hidden_layer_size[index], hidden_layer_size[index + 1]))
		else:
			weights.append(np.random.rand(hidden_layer_size[index], output_shape))
			
		bias.append(np.random.rand(hidden_layer_size[index]))

	return weights, bias
	
weights, bias = buildNetwork(4, [8, 6], 1)

"""
for _ in weights: 
	print("Weights : ", _)
for _ in bias:
	print("bias : ", _)

"""
def relu(data):
	return np.array([max(0, x) for x in data])

def feedForward(input_data, weights, bias):
	feedData = input_data
	activation = relu
	for w, b in zip(weights, bias):
		output = activation(np.dot(feedData, w) + b)
		feedData = output
	return output

print("1st Iteration : ", feedForward(input_data[0], weights, bias))
