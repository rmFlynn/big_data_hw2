# Author: Rory Flynn, sinned: 10/19/2019
import numpy as np
import gc

class line_reader(object):

    def parse(self, string):
        """Parses a line to ints"""
        line = string.split(',')
        return [int(i) for i in line]

    def __init__(self, file_path):
        """ Reads the file and sets information """
        self.f = open(file_path)
        # make some clean counters
        self.line_count = 0
        self.user_count = 0
        self.item_count = 0
        self.f.seek(0)
        for line in self.f:
            values = self.parse(line)
            # count things
            if values[0] > self.user_count:
                self.user_count = values[0]
            if values[1] > self.item_count:
                self.item_count = values[1]
            self.line_count+=1
        self.f.seek(0)

    def get_file(self):
        """Return the file making sure we are at the start"""
        self.f.seek(0)
        return self.f

    def next(self):
        return self.parse(self.f.readline().strip())




def stocat_grad_dec(l, a, l_tool, P, Q, max_rating=5):
    """Perform one round of stochastic gradient decent"""
    for line in l_tool.get_file():
        # put variables into useable formats with names.
        x, i, rxi = l_tool.parse(line.strip())
        px = np.asmatrix(P[x])
        qi = np.asmatrix(Q[i])
        # calculate all values
        exi = np.asscalar(2*(rxi - np.dot(qi, px.T)))
        qi = qi + a*(exi*px - 2*l*qi)
        px = px + a*(exi*qi - 2*l*px)
        # put variables back in the matrix
        Q[i] = qi
        P[x] = px
    return P, Q

def error_calk(l, l_tool, P, Q, max_rating=5):
    """Calculate the error"""
    error = 0
    for line in l_tool.get_file():
        i, u, riu = l_tool.parse(line.strip())
        pi = np.asmatrix(P[i])
        qu = np.asmatrix(Q[u])
        # Literally the equations from the presentation as close as possible.
        error += (riu - np.dot(pi, qu.T))**2
    # normalize
    normalizer = l*(np.sum(Q**2) + np.sum(P**2))
    error += normalizer
    return np.asscalar(error)

def latent_factor_recommnder(data_path, regularization_factor, learning_rate, iterations, k, keep_error=True, max_rating = 5, lr_reduce=False):
    """The main control point of the program"""
    l_tool = line_reader(data_path)
    # make the matrices
    P = np.random.rand(l_tool.user_count + 1, k)*np.sqrt(max_rating/k)
    Q = np.random.rand(l_tool.item_count + 1, k)*np.sqrt(max_rating/k)
    # get first error
    iter_error = [error_calk(regularization_factor, l_tool, P, Q)]
    print("Start error: {:0.0f}".format(iter_error[0]))
    for i in range(iterations):
        # Option for random testing
        if lr_reduce:
            learning_rate = learning_rate/(i + 1)
        # perform one loop over the file
        P, Q = stocat_grad_dec(l=regularization_factor, a=learning_rate, l_tool=l_tool, P=P, Q=Q)
        print("Iteration: {}".format(i+1))
        # Calculate the error if you want to
        if keep_error:
            iter_error.append(error_calk(regularization_factor, l_tool, P, Q))
            print("    Error: {:0.0f}".format(iter_error[-1]))
    if not keep_error:
        iter_error.append(error_calk(regularization_factor, l_tool, P, Q))
    # Print last
    print("  End error: {:0.0f}".format(iter_error[-1]))
    return iter_error

gc.collect()
