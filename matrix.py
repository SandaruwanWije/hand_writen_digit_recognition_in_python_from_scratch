import  numpy as np
import random
class Matrix:
    rows = None
    cols = None
    matrix = None

#initialize the num of rows, num of cols and the empty matrix
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.matrix = []
        for i in range(rows):
            self.matrix.append([])
            for j in range(cols):
                self.matrix[i].append(0)
#initialize the matrix with random numbers
    def randomize(self):
            for i in range(len(self.matrix)):
                for j in range(len(self.matrix[i])):
                    self.matrix[i][j] = random.uniform(-1, 1)
#add a numbet with matrix or add another with the matrix
    def add(self, n):
#check if the input is a matrix or not
        if (isinstance(n, Matrix)):
            if (self.rows == n.rows and self.cols == n.cols):
                for i in range(len(self.matrix)):
                    for j in range(len(self.matrix[i])):
                        self.matrix[i][j] = self.matrix[i][j] + n.matrix[i][j]
            else:
                raise Exception('input matrix is not in same dimension')
        else:
            for i in range(len(self.matrix)):
                for j in range(len(self.matrix[0])):
                    self.matrix[i][j] = self.matrix[i][j] + n
# add a numbet with matrix or add another with norther matrix
    @staticmethod
    def sub(m, n):
        output = Matrix(m.rows, m.cols)
        if (isinstance(n, Matrix)):
            if (m.rows == n.rows and m.cols == n.cols):
                for i in range(m.rows):
                    for j in range(m.cols):
                        output.matrix[i][j] = m.matrix[i][j] - n.matrix[i][j]
                return output
            else:
                raise Exception('input matrix is not in same dimension.')
        else:
            for i in range(m.rows):
                for j in range(m.cols):
                    output.matrix[i][j] = m.matrix[i][j] - n
            return output
#scale the matrix from a scaler
    def scale(self, n):
        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i][j] = self.matrix[i][j] * n
#element vise multiplication of the matrx
    def multiply(self, n):
        if (isinstance(n, Matrix)):
            if (self.rows == n.rows and self.cols == n.cols):
                for i in range(self.rows):
                    for j in range(self.cols):
                        self.matrix[i][j] = self.matrix[i][j] * n.matrix[i][j]
            else:
                raise Exception('input matrix is not in same dimension.')
        else:
            raise Exception('input  is not a matrix.')
#matrix dot product
    @staticmethod
    def dot_product(m, n):
        if (isinstance(m, Matrix) and isinstance(n, Matrix) and m.cols == n.rows):
            product_matrix = Matrix(m.rows, n.cols)
            product_matrix.matrix = []
            for i in range(m.rows):
                product_matrix.matrix.append([])
                for j in range(n.cols):
                    sum = 0
                    for k in range(len(n.matrix)):
                        sum += m.matrix[i][k] * n.matrix[k][j]
                    product_matrix.matrix[i].append(sum)
            return product_matrix
        else:
            raise Exception('inputs are not matrices or is not in correct dimension.')
#take the transpose of a matrix
    @staticmethod
    def transpose(m):
        if (isinstance(m, Matrix)):
            transposed_matrix = Matrix(m.cols, m.rows)
            transposed_matrix.matrix = []
            for i in range(m.cols):
                transposed_matrix.matrix.append([])
                for j in range(m.rows):
                    transposed_matrix.matrix[i].append(m.matrix[j][i])
            return transposed_matrix
        else:
            raise Exception('inputs are not matrices.')
#map the all element of the matrix to a function
    def map(self, func):
        for i in range(self.rows):
            for j in range(self.cols):
                result = func(self.matrix[i][j])
                self.matrix[i][j] = result
#static vertion of the map function
    @staticmethod
    def map_s(m, func):
        for i in range(m.rows):
            for j in range(m.cols):
                result = func(m.matrix[i][j])
                m.matrix[i][j] = result
        return m
#convert array to a one colum matrix(vector)
    @staticmethod
    def array_to_vector(m):
        if (isinstance(m, list)):
            result = Matrix(len(m), 1)
            for i in range(len(m)):
                result.matrix[i][0] = m[i]
            return result
        elif isinstance(m, (np.ndarray, np.generic) ):
            result = Matrix(m.size, 1)
            index = 0
            for i in range(len(m)):
                for j in range(m.shape[1]):
                    result.matrix[index][0] = m[i][j]
                    index = index + 1
            return result
        else:
            raise Exception('input is not a list or numpy array.')

    @staticmethod
    def array_to_matrix(input_array, rows, cols):
        result = Matrix(rows, cols)
        for i in range(rows):
            for j in range(cols):
                result.matrix[i][j] = input_array[i][j]
        return result
#convert the matrix to an array
    def to_array(self):
        result = []
        for i in range(self.rows):
            for j in range(self.cols):
                result.append(self.matrix[i][j])
        return result
