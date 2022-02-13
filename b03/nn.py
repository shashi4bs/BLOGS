import numpy as np
import progressbar

input_data = np.array([[1, 2, 3, 4],
			[1, 2, 3, 5],
			[1, 2, 3, 6],
			[1, 2, 3, 7], 
			[1, 2, 3, 8]])
output_data = np.array([4, 5, 6, 7, 8])

#output_data = output_data.reshape(-1, 1)
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
			weights.append(np.random.rand(hidden_layer_size[index]))
			
		bias.append(np.random.rand(hidden_layer_size[index]))
	
	bias.append(np.array(output_shape))
	return np.array(weights), np.array(bias)
	
weights, bias = buildNetwork(4, [8, 6], 1)

def relu(x):
	return max(0, x)

def diffRelu(x):
	if x < 0:
		return 0
	else:
		return 1

def applyDiffRelu(data):
	if type(data) == np.float64:
		return np.array([diffRelu(data)])
	else:
		return np.array([diffRelu(x) for x in data])

def applyRelu(data):
	if type(data) == np.float64:
		return np.array([relu(data)])
	else:
		return np.array([relu(x) for x in data])

def feedForward(input_data, weights, bias):
	feedData = input_data
	activation = applyRelu
	network = [np.array(input_data)]
	for w, b in zip(weights, bias):
		z = np.dot(feedData, w) + b
		output = activation(z)
		feedData = output
		network.append(z)
	return output, np.array(network, dtype='object')


print("1st Iteration : ", feedForward(input_data[0], weights, bias))


def get_cost(output, index):
	return (output_data[index] - output) ** 2


def fix_weights_and_bias(output, network, cost, weights, bias, lambda_w, lambda_b):
	"""
	calculate gradient and add it to lambda_w and lambda_b
	"""
	z = (2 * (network[-1] - output)) * applyDiffRelu(network[-1])
	lambda_b[-1] += z[0]
	lambda_w[-1] += np.dot(lambda_b[-1], network[-2])
		
	for index in range(2, weights.shape[0] + 1):
		if index == 2:
			z = weights[-index + 1].T * applyDiffRelu(network[-index])
		else:
			z = np.sum(weights[-index+1].T * applyDiffRelu(network[-index]), axis=0)	
		lambda_b[-index] += z
		lambda_w[-index] += network[-index-1].reshape(-1, 1) * z.reshape(1, -1) 
	for index in range(weights.shape[0]):
		weights[index] -= (lr * lambda_w[index])
		bias[index] = bias[index] - (lr * lambda_b[index])
	return weights, bias


num_iterations = 10
for _ in range(num_iterations):
	cost = 0
	network = None
	lr = 0.001
	lambda_w = [np.zeros(w.shape) for w in weights]
	lambda_b = [np.zeros(b.shape) for b in bias]
	for i in range(input_data.shape[0]): 
		output, network = feedForward(input_data[i], weights, bias)
		cost = get_cost(output[0], i)
		print("Cost : ", cost)
		weights, bias = fix_weights_and_bias(output_data[i], network, cost, weights, bias, lambda_w, lambda_b)
	print("Weights: ", weights)
	print("Bias: ", bias)	
