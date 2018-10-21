
from numpy import random,  dot, exp, array

class NeuralNetwork(object):
	random.seed(1)
	def __init__(self):
		self.synaptic_weigthts = 2 * random.random((3, 1)) - 1

	def __sigmoid(self, x):
		return 1/(1+exp(-x))

	def __sigmoid_derivate(self,x):
		return x * (1-x)

	def train(self, training_set_input,training_set_outputs, number_of_training_iterations):
		for iteration in range(number_of_training_iterations):
			output = self.think(training_set_input)
			error = training_set_outputs - output
			adjustment = dot(training_set_input.T, error * self.__sigmoid_derivate(output))
			
			self.synaptic_weigthts += adjustment

	def think(self,inputs):
		return self.__sigmoid(dot(inputs,self.synaptic_weigthts))		


if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print ("Random starting synaptic weights: ")
    print (neural_network.synaptic_weigthts)

    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    neural_network.train(training_set_inputs,training_set_outputs,100000)
    print("Para un nuevo caso [1,0,0]")
    print(neural_network.think(array([1,0,0])))
 