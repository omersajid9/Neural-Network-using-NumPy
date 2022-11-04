import numpy as np
import pandas as pd

class Data:
    def __init__(self, data_loc):
        batch_data = np.array(pd.read_csv(data_loc), dtype=float)
        self.batch_data = batch_data
        
    def shuffle_data(self):
        np.random.shuffle(self.batch_data)
        
    def divide_data(self, train_percentage):
        num_train = int(train_percentage * self.batch_data.shape[0])
        self.train_data = self.batch_data[:num_train, :]
        self.test_data = self.batch_data[self.train_data.shape[0]:, :]
        
    def y_to_vector(y_data):
        temp = np.zeros((y_data.shape[0], max(y_data) + 1))
        temp[np.arange(y_data.shape[0]), y_data] = 1
        return temp

    def parse_data(self):
        self.y_train = np.array(self.train_data[:, 0], dtype=int)
        self.y_train_vector = Data.y_to_vector(self.y_train)
        self.x_train = self.train_data[:, 1:]
        self.y_test = np.array(self.test_data[:, 0], dtype=int)
        self.y_test_vector = Data.y_to_vector(self.y_test)
        self.x_test = self.test_data[:, 1:]

    def normalize_data(self):
        self.x_train -= np.mean(self.x_train, axis = 0)
        self.x_train /= np.std(self.x_train, axis = 0)
        
    def prepare_data(self, train_percentage):
        self.shuffle_data()
        self.divide_data(train_percentage)
        self.parse_data()
        # self.normalize_data()