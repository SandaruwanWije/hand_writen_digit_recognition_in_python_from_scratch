import math
from pathlib import Path
import ast
from matrix import Matrix

class NeuralNetwork:
    num_of_input_nurons = None
    num_of_hiddden_nurons = None
    num_of_output_nurons = None
    error = 0
    lr = 0.1
#initialize the  neural network layers, weights and biases
    def __init__(self, no_of_input, no_of_hidden, no_of_output, path, weight_file="weights.txt"):
        self.num_of_input_nurons = no_of_input;
        self.num_of_hiddden_nurons = no_of_hidden;
        self.num_of_output_nurons = no_of_output;
#check if exist the weights of the local drive
        isWeightsExist = Path(path).exists()
        if(isWeightsExist):
            f = open(weight_file, "r")
            f1 = f.readlines()
            w = []
            w_count = 0
            b = []
            b_count = 0
            w_i_h = []
            w_h_o = []
            b_h = []
            b_o = []
            for text in f1:
                if(text.find("w")) != -1:
                    w.append(ast.literal_eval(text.split(":")[1]))
                elif(text.find("b")) != -1:
                    b.append(ast.literal_eval(text.split(":")[1]))
            for i in range(no_of_hidden):
                w_i_h.append([])
                for j in range(no_of_input):
                    w_i_h[i].append(w[w_count])
                    w_count = w_count + 1
            for i in range(no_of_hidden):
                b_h.append([])
                for j in range(1):
                    b_h[i].append(b[b_count])
                    b_count = b_count + 1
            for i in range(no_of_output):
                w_h_o.append([])
                for j in range(no_of_hidden):
                    w_h_o[i].append(w[w_count])
                    w_count = w_count + 1
            for i in range(no_of_output):
                b_o.append([])
                for j in range(1):
                    b_o[i].append(b[b_count])
                    b_count = b_count + 1
            self.weight_iput_to_hidden = Matrix.array_to_vector(w_i_h)
            self.weight_hidden_to_output = Matrix.array_to_vector(w_h_o)
            self.bias_of_hidden = Matrix.array_to_vector(b_h)
            self.bias_of_output = Matrix.array_to_vector(b_o)
        else:
            self.weight_iput_to_hidden = Matrix(no_of_hidden, no_of_input)
            self.weight_hidden_to_output = Matrix(no_of_output, no_of_hidden)
            self.weight_iput_to_hidden.randomize()
            self.weight_hidden_to_output.randomize()
            self.bias_of_hidden = Matrix(no_of_hidden, 1)
            self.bias_of_output = Matrix(no_of_output, 1)
            self.bias_of_hidden.randomize()
            self.bias_of_output.randomize()
#sigmoid activation function
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
#derivatives of the sigmoid activation function
    @staticmethod
    def d_sigmoid(y):
        return y * (1 - y)
#this is the method which predict the result
    def predict(self, input_array):
        inputs = Matrix.array_to_vector(input_array)
        hidden = Matrix.dot_product(self.weight_iput_to_hidden, inputs)
        hidden.add(self.bias_of_hidden)
        hidden.map(NeuralNetwork.sigmoid)
        output = Matrix.dot_product(self.weight_hidden_to_output, hidden)
        output.add(self.bias_of_output)
        output.map(NeuralNetwork.sigmoid)
        return output.to_array()
#this is the method which train the network with backpropagation in gradient decent algorithm
    def train(self, input_array, target_array):
#converts the input array to a vector
        inputs = Matrix.array_to_vector(input_array)
        hidden = Matrix.dot_product(self.weight_iput_to_hidden, inputs)
        hidden.add(self.bias_of_hidden)
        hidden.map(NeuralNetwork.sigmoid)
        output = Matrix.dot_product(self.weight_hidden_to_output, hidden)
        output.add(self.bias_of_output)
        output.map(NeuralNetwork.sigmoid)

        target_Matrix = Matrix.array_to_vector(target_array)
#calculate error
        error = Matrix.sub(target_Matrix, output)
        self.error = error
        weight_hidden_to_output_t = Matrix.transpose(self.weight_hidden_to_output)
        hidden_error = Matrix.dot_product(weight_hidden_to_output_t, error)
#calculate the gradient of the hidden layer
        hidden_gradient_matrix = Matrix.map(output, NeuralNetwork.d_sigmoid)
        hidden_gradient_matrix.scale(error)
        hidden_gradient_matrix.scale(self.lr)
        delta_weights_ho = Matrix.dot_product(output, Matrix.transpose(hidden))
#update weights and biases
        self.weight_hidden_to_output.add(delta_weights_ho)
        self.bias_of_output.add(hidden_gradient_matrix)
#calculate the gradient of the output layer
        output_gradient_matrix = Matrix.map(hidden, NeuralNetwork.d_sigmoid)
        output_gradient_matrix.scale(hidden_error)
        output_gradient_matrix.scale(self.lr)
        delta_weights_ih = Matrix.dot_product(hidden, Matrix.transpose(inputs))
#update weights and biases
        self.bias_of_hidden.add(output_gradient_matrix)
        self.weight_iput_to_hidden.add(delta_weights_ih)