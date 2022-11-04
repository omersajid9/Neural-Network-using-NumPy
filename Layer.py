import numpy as np


class Bias:
        def __init__(self, n_nodes):
            self.data = np.random.normal(loc = 0.0, scale = 0.5, size = (n_nodes, 1)) / np.sqrt(n_nodes)
            self.delta = np.zeros((1))
            
        def update_data(self, alpha):
            self.data = self.data - alpha * self.delta
            if self.isNaN() > 0:
                print("Nan occured in updated bias")
                exit()
            self.reset_delta()
            
        def reset_delta(self):
#             m,  = self.delta.shape
            self.delta = np.zeros((1))
            
        def isNaN(self):
            return np.sum(np.isnan(self.data))
            
        def __str__(self):
            return f"Bias Data :{self.data}, \n Bias Delta: {self.delta}"

class Layer:
    
        
    def __init__(self, n_nodes, activation_function, input_weight=None, output_weight=None):
        self.bias = Bias(n_nodes)
        self.input_weight = input_weight
        self.output_weight = output_weight
        if activation_function == "softmax":
            self.activate_layer = self.activate_softmax
            self.deactivate_layer = self.deactivate_softmax
        else:
            self.activate_layer = self.activate_ReLU
            self.deactivate_layer = self.deactivate_ReLU
            
    def deactivate_softmax(self, actual):
        print("softmax deactivate")
        self.dZ = self.A - actual
        
    def deactivate_ReLU(self):
        print("deactivate relu")
        self.dZ = self.output_weight.data.T.dot(self.output_layer.dZ) * Layer.deriv_ReLU(self.Z)
        
    def activate_softmax(self):
        print("activate softmax")
        self.A = Layer.softmax(self.Z)
        
    def activate_ReLU(self):
        print("activate_ReLU")
        self.A = Layer.ReLU(self.Z)
        
    def softmax(Z):
        print("softmax")
#         expZ = np.exp(Z - np.max(Z))
#         return expZ / np.sum(expZ, 0)
        return np.exp(Z)/np.sum(np.exp(Z), 0)

    def deriv_ReLU(Z):
        print("deriv_ReLU")
        return Z > 0
    
    def ReLU(Z):
        print("ReLU")
        return np.maximum(Z, 0)
        
    def set_input_layer(self, input_layer):
        if hasattr(self, 'input_weight'):
            self.input_layer = input_layer
    
    def set_output_layer(self, output_layer):
        if hasattr(self, 'output_weight'):
            self.output_layer = output_layer

    def forward_propogation(self):
        if hasattr(self, 'input_layer'):
            self.Z = self.input_weight.data.dot(self.input_layer.A) + self.bias.data
            self.activate_layer()
            
            
    def backward_propogation(self, actual):
        si = actual.size
        print(si)
        if hasattr(self, 'input_layer'):
            if hasattr(self, 'output_layer'):
                self.deactivate_ReLU()
            else:
                self.deactivate_softmax(actual)
            print("weight delta")
            self.input_weight.delta = self.dZ.dot(self.input_layer.A.T) / si
            print("bias delta")
            self.bias.delta = np.sum(self.dZ) / si
            
            
    def update_parameters(self, alpha):
        if hasattr(self, 'input_layer'):
            self.input_weight.update_data(alpha)
            self.bias.update_data(alpha)
            
class Input_Layer(Layer):
    def __init__(self, train_data, output_weight):
        self.Z = train_data.T/255.
        self.A = train_data.T/255.
        self.output_weight = output_weight
