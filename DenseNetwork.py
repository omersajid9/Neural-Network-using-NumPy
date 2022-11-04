import numpy as np
from numpy import argmax
from Layer import Layer, Input_Layer

class DenseNetwork:
    class Weight:
        def __init__(self, n_input, n_output):
            self.data = np.random.randn(n_output, n_input) / np.sqrt(n_input/2)
            # self.data = np.random.normal(loc = 0.0, scale = 0.5, size = (n_output, n_input)) / np.sqrt(n_input/2)
            self.delta = np.zeros((n_output, n_input))
            
        def reset_delta(self):
            m, n = self.delta.shape
            self.delta = np.zeros((m, n))

        def update_data(self, alpha):
            self.data = self.data - alpha * self.delta
            if self.isNaN() > 0:
                print("Nan occured in updated weight")
                exit()
            self.reset_delta()
        
        def isNaN(self):
            return np.sum(np.isnan(self.data))
        
        def __str__(self):
            return f"Weight Data :{self.data}, \n Weight Delta: {self.delta}"
        
    def __init__(self, n_layers, n_nodes_list):
        self.n_hidden = n_layers - 2
        self.weights_list = list()
        self.layers_list = list()
        self.n_nodes_list = n_nodes_list
        
    def initialize_weights(self):
        for i in range(self.n_hidden + 1):
            cur_nodes = self.n_nodes_list[i]
            nex_nodes = self.n_nodes_list[i + 1]
            self.weights_list.append(self.Weight(cur_nodes, nex_nodes))
            
    def initialize_layers(self, data):
        self.layers_list.append(Input_Layer(data, self.weights_list[0]))
        for j in range(1, 1 + self.n_hidden):
            self.layers_list.append(Layer(self.n_nodes_list[j], "ReLU", input_weight = self.weights_list[j - 1], output_weight = self.weights_list[j]))
        self.layers_list.append(Layer(self.n_nodes_list[-1], "softmax", input_weight = self.weights_list[-1]))
        
    def link_layers(self):
        self.layers_list[0].set_output_layer(self.layers_list[1])
        for k in range(1, 1 + self.n_hidden):
            self.layers_list[k].set_input_layer(self.layers_list[k -1])
            self.layers_list[k].set_output_layer(self.layers_list[k + 1])
        self.layers_list[-1].set_input_layer(self.layers_list[-2])
        
    def forward_feed(self):
        for b in self.layers_list:
            b.forward_propogation()
            
    def backward_feed(self, actual):
        for c in self.layers_list[::-1]:
            c.backward_propogation(actual)
            
    def update_network(self, alpha):
        for d in self.layers_list:
            d.update_parameters(alpha)

    def accuracy(act, pred):
        res = sum(argmax(pred, axis=0) == argmax(act, axis=0)) / pred.shape[1]
        return res
        
            
    def mean_squared_error(act, pred):
        diff = pred - act
        differences_squared = diff ** 2
        mean_diff = differences_squared.mean()
        return mean_diff
    
    def L_i_vectorized(act, pred):
        delta = 1.0
        margins = np.maximum(0, pred - pred[act==1] + delta)
        margins[act==1] = 0
        loss_i = np.sum(margins)
        return loss_i/pred.shape[1]

    def printStatement(self, act, iteration, epoch):
        output = self.layers_list[-1].A
        statement = "Loss :" + str(DenseNetwork.L_i_vectorized(act, output)) + "; MSE: " + str(DenseNetwork.mean_squared_error(act, output)) + "; Accuracy: " + str(DenseNetwork.accuracy(act, output)) + " " + str(iteration) + " / " + str(epoch)
        return statement

    def prediction(self, test_image):
        temp = self.layers_list[0].A
        self.layers_list[0].A = test_image
        self.forward_feed()
        output = argmax(self.layers_list[-1].A, axis=0)
        self.layers_list[0].A = temp
        print("I predict it as " + str(output))
