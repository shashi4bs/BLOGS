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
	
	bias.append(output_shape)
	return weights, bias
	
weights, bias = buildNetwork(4, [8, 6], 1)

def relu(x):
	return max(0, x)

def diffRelu(x):
	if x < 0:
		return 0
	else:
		return 1
def applyDiffRelu(data):
	return np.array([diffRelu(x) for x in data])

def applyRelu(data):
	return np.array([relu(x) for x in data])

def feedForward(input_data, weights, bias):
	feedData = input_data
	activation = applyRelu
	network = list()
	for w, b in zip(weights, bias):
		z = np.dot(feedData, w) + b
		output = activation(z)
		feedData = output
		network.append(z)
	return output, np.array(network, dtype='object')


print("1st Iteration : ", feedForward(input_data[0], weights, bias))


def get_cost(output, index):
	return (output_data[index] - output) ** 2

def fix_weights_and_bias(cost, network, weights, bias):
	'''
		calculate gradient -> updated weights and bias
	'''
	print(weights[-1])
	learning_rate = 0.01
	new_weights, new_bias = list(), list()
	delta_w = (cost) * network[-1] * applyDiffRelu(network[-2])
	for index in range(network.shape[0] - 1, -1, -1):
		weights[index] = weights[index] - (learning_rate * delta_w)
		print("test: ", weights[index])
		delta_w = delta_w * network[index] * applyDiffRelu(network[index - 1])					
	return new_weights, new_bias 

num_iterations = 1
for _ in range(num_iterations):
	cost = 0
	network = None
	for i in range(input_data.shape[0]): 
		output, network = feedForward(input_data[i], weights, bias)
		cost += get_cost(output[0], i)
	print("Cost : ", cost)
	new_weights, new_bias = fix_weights_and_bias(cost, network, weights, bias)
	weights, bias = new_weights, new_bias

