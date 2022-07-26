import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
import numpy as np


class neuralNetwork:

    def __int__(self, input_nodes, hidden_nodes, output_nodes):

