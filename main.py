from data_preprocessor import DataPreprocessor
from neural_network import NeuralNetwork
import numpy as np
from tkinter.filedialog import askopenfilename
from tkinter import *
import operator
import matplotlib.pyplot as plt


if __name__ == "__main__":
    dataset = DataPreprocessor().gather()
    Tk().withdraw()
    weight_file = askopenfilename()
    nn = NeuralNetwork(784,64,10,weight_file)
    if (weight_file == ""):
        for i in range(1000):
            normalize_image = dataset["train_images"][i]/255
            nn.train(normalize_image, dataset["train_labels"][i])

    newimg = plt.imread("E:\\Projects\\ANN\MNIST_Classifire\\testing\\9.png")
    normalize_image = newimg[:, :, 1]
    guess = nn.predict(normalize_image)
    index, value = max(enumerate(guess), key=operator.itemgetter(1))
    lable = ""
    if (index == 0):
        lable = "Zero"
    elif (index == 1):
        lable = "One"
    elif (index == 2):
        lable = "Two"
    elif (index == 3):
        lable = "Three"
    elif (index == 4):
        lable = "Four"
    elif (index == 5):
        lable = "Five"
    elif (index == 6):
        lable = "Six"
    elif (index == 7):
        lable = "Seven"
    elif (index == 8):
        lable = "Eight"
    elif (index == 9):
        lable = "Nine"
    print(lable)
