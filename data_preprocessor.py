import os,codecs,numpy
import pickle
from pathlib import Path
import urllib.request
import gzip
import ast

class DataPreprocessor:
    URL = "http://yann.lecun.com/exdb/mnist/"
    train_images = "train-images-idx3-ubyte.gz"
    train_labels = "train-labels-idx1-ubyte.gz"
    test_images = "t10k-images-idx3-ubyte.gz"
    test_lables = "t10k-labels-idx1-ubyte.gz"
    def gather(self):
        datapath = 'data/'
        if(not Path(datapath).exists()):
            os.mkdir(datapath)
        files = os.listdir(datapath)
        if(len(files) != 4):
            urllib.request.urlretrieve(self.URL + self.train_images, datapath + "/" + self.train_images)
            urllib.request.urlretrieve(self.URL + self.train_labels, datapath + "/" + self.train_labels)
            urllib.request.urlretrieve(self.URL + self.test_images, datapath + "/" + self.test_images)
            urllib.request.urlretrieve(self.URL + self.test_lables, datapath + "/" + self.test_lables)

            files = os.listdir(datapath)
            for file in files:
                input = gzip.GzipFile("data/" + file, 'rb')
                s = input.read()
                input.close()
                output = open("data/" + file.split(".")[0]+".idx3-ubyte", 'wb')
                output.write(s)
                output.close()
                os.remove(datapath+"/"+file)
                files = os.listdir(datapath)
        def get_int(b):
            return int(codecs.encode(b, 'hex'), 16)
        data_dict = {}
        for file in files:
            if file.endswith('ubyte'):  # FOR ALL 'ubyte' FILES
                print('Reading ', file)
                with open(datapath + file, 'rb') as f:
                    data = f.read()
                    type = get_int(data[:4])  # 0-3: THE MAGIC NUMBER TO WHETHER IMAGE OR LABEL
                    length = get_int(data[4:8])  # 4-7: LENGTH OF THE ARRAY  (DIMENSION 0)
                    if (type == 2051):
                        category = 'images'
                        num_rows = get_int(data[8:12])  # NUMBER OF ROWS  (DIMENSION 1)
                        num_cols = get_int(data[12:16])  # NUMBER OF COLUMNS  (DIMENSION 2)
                        parsed = numpy.frombuffer(data, dtype=numpy.uint8,
                                                  offset=16)  # READ THE PIXEL VALUES AS INTEGERS
                        parsed = parsed.reshape(length, num_rows,
                                                num_cols)  # RESHAPE THE ARRAY AS [NO_OF_SAMPLES x HEIGHT x WIDTH]
                    elif (type == 2049):
                        category = 'labels'
                        parsed = numpy.frombuffer(data, dtype=numpy.uint8,
                                                  offset=8)  # READ THE LABEL VALUES AS INTEGERS
                        parsed = parsed.reshape(length)  # RESHAPE THE ARRAY AS [NO_OF_SAMPLES]
                    if (length == 10000):
                        set = 'test'
                    elif (length == 60000):
                        set = 'train'
                    data_dict[set + '_' + category] = parsed  # SAVE THE NUMPY ARRAY TO A CORRESPONDING KEY
        sets = ['train', 'test']

        for set in sets:  # FOR TRAIN AND TEST SET
            images = data_dict[set + '_images']  # IMAGES
            labels = data_dict[set + '_labels']  # LABELS
            no_of_samples = images.shape[0]  # NUBMER OF SAMPLES
            for indx in range(no_of_samples):  # FOR EVERY SAMPLE
                print(set, indx)
                image = images[indx]  # GET IMAGE
                label = labels[indx]  # GET LABEL

        # DUMPING THE DICTIONARY INTO A PICKLE
        with open(datapath + 'MNISTData.pkl', 'wb') as fp:
            pickle.dump(data_dict, fp)