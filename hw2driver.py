# Author: Rory Flynn, sinned: 10/06/2019

from pyspark import SparkContext
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sc = SparkContext()

##
## PART 1
##
from part1 import kmeans, plot_cost


#### var #### declaration #### start #### var #### declaration #### start #### var #### declaration #### start ####

# determines the number of iterations
Iterations = 20
# this is the absolute path to the dataset file
dataset_path = './data.txt'
# this contains the absolute path the centroid file that being use (either c1.txt or c2.txt)
centroids_path = './c1.txt'
# centroids_path = './c2.txt'
# or False
euclidean_distance = True

# data and centroids to rdd. data needed more than defalt slices on my system
data = sc.parallelize(np.genfromtxt(dataset_path), numSlices=30)
centroids = sc.parallelize(np.genfromtxt(centroids_path))

# Do the k means operation for c1
cost_euclid_c1 = kmeans(data, centroids, Iterations, euclidean_distance)
cost_manhat_c1 = kmeans(data, centroids, Iterations, False)

# Do the k means operation for c2
centroids_path = './c2.txt'
centroids = sc.parallelize(np.genfromtxt(centroids_path))
cost_euclid_c2 = kmeans(data, centroids, Iterations, euclidean_distance)
cost_manhat_c2 = kmeans(data, centroids, Iterations, False)

# Save work
np.save("m-e_c-1", cost_euclid_c1)
np.save("m-m_c-1", cost_manhat_c1)
np.save("m-e_c-2", cost_euclid_c2)
np.save("m-m_c-2", cost_manhat_c2)

#load work
cost_euclid_c1 = np.load("m-e_c-1.npy")
cost_manhat_c1 = np.load("m-m_c-1.npy")
cost_euclid_c2 = np.load("m-e_c-2.npy")
cost_manhat_c2 = np.load("m-m_c-2.npy")

plot_cost(cost_euclid_c1, cost_euclid_c2, "Euclidean")
plot_cost(cost_manhat_c1, cost_manhat_c2, "Manhatan")


##
## PART 2
##

#### var #### declaration #### end #### var #### declaration #### end #### var #### declaration #### end ####

iterations = 40
k = 20
regularization_factor = 0.1 # l
# it is not clear if you want k to be 20 or 1 if k = 20 this rate is too high, if k = 1 it is ok
#learning_rate = 0.1 # n
learning_rate = 0.01 # n
data_path = './ratings.csv'

#### var #### declaration #### start #### var #### declaration #### start #### var #### declaration #### start ####

# Load the code
from part2 import latent_factor_recommnder

# Train the models
sgd_01 = latent_factor_recommnder(data_path, regularization_factor, learning_rate, iterations, k)
sgd_01rf = latent_factor_recommnder(data_path, 0.01, 0.01, iterations, k)

# save the errors
#np.save("sgd_01", sgd_01)
#np.save("sgd_01rf", sgd_01)

# load the errors
best_a_error = np.load("sgd_01.npy")
best_a_error_w_rf = np.load("sgd_01rf.npy")

# make the plot, using pandas
e1 = pd.DataFrame(best_a_error, columns=["err"])
e2 = pd.DataFrame(best_a_error_w_rf, columns=["err"])
e1['model'] = "Best Alpha"
e2['model'] = "Best Alpha: with less regularization"
e = e1.append(e2, sort=False)
e['Iteration'] = e.index
# Make a plot of the results, epoch vs losses
plot1 = sns.lineplot(x='Iteration',
        y='err',
        hue='model',
        data=e,
        linewidth=2.5,
        marker="o")
plot1.set(xlabel = "Iteration")
plot1.set(ylabel = "Error Values")
plot1.set(title = "Error over Iterations")
plot1.figure.savefig('plot_part2.png')
# Clear the stage
plt.clf()

