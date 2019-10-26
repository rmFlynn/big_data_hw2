# Author: Rory Flynn, sinned: 10/19/2019
import numpy as np
import numpy as np
from pyspark import SparkContext
# plot function tools
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


#sc = SparkContext()

def kmeans(data, centroids, Iterations, euclidean_distance=True):
    # get distance metric
    if euclidean_distance:
        # Calculate euclidean distance from 2 vectors
        dist_metric = lambda v1, v2: np.linalg.norm(v1 - v2)
        metric_name = "Euclidean Distance"
    else:
        # Calculate Manhattan distance from 2 vectors
        dist_metric = lambda v1, v2: np.abs(v1 - v2).sum(-1)
        metric_name = "Manhattan Distance"
    # Format the centroids into a usable form
    cents=centroids.collect()
    nums = list(range(len(cents)))
    cents=[(n, c) for n, c in zip(nums, cents)]
    # special centroid computation for the first centroids
    def first_centroid(x):
        dist = [dist_metric(x, c[1]) for c in cents]
        min_d = min(dist)
        cent_id = cents[dist.index(min_d)][0]
        return min_d, cent_id, x
    # fined first centroids
    data_clst = data.map(first_centroid)
    # Print variables like in example
    print("Now performing {} Iterations with {} as the metric:".format(Iterations, metric_name))
    costs = []
    num = 0
    for i in range(Iterations):
        # calculate the cost
        costs.append(data_clst.map(lambda x : x[0]).sum())
        print("Iteration {}. Cost is now {}.".format(num, costs[num]))
        num += 1
        # separate the centroid count and point
        cent_count = data_clst.map(lambda x : (x[1], 1))# better way?
        cent_point = data_clst.map(lambda x : (x[1], x[2]))
        # reduce and sum the centroid count
        cent_count = cent_count.reduceByKey(lambda e, i : e + i)
        cent_point = cent_point.reduceByKey(lambda e, i : e + i)
        print(cent_count.collect())
        # join
        cent_count_point = cent_count.join(cent_point)
        # divide
        cent_new = cent_count_point.map(lambda x : (x[0], x[1][1]/x[1][0]))
        # I am the centroid now - like Captain Philips meme
        cents = cent_new.collect()
        # I am the function now - like Captain Philips meme
        def get_centroid(y):
            x = y[2]
            dist = [dist_metric(x, c[1]) for c in cents]
            min_d = min(dist)
            cent_id = cents[dist.index(min_d)][0]
            return min_d, cent_id, x
        # apply the centroid function
        data_clst_new = data_clst.map(get_centroid)
        # I am the data now - like Captain Philips meme
        data_clst = data_clst_new
    # calculate the cost one last time
    costs.append(data_clst.map(lambda x : x[0]).sum())
    print("Iteration {}. Cost is now {}.".format(num, costs[num]))
    return costs


def plot_cost(cost1, cost2, metric):
    # Make a data frame of the results, epoch vs losses
    cost1 = pd.DataFrame(cost1, columns=["cost"])
    cost2 = pd.DataFrame(cost2, columns=["cost"])
    cost1['cent'] = "Centroid Set 1"
    cost2['cent'] = "Centroid Set 2"
    cost = cost1.append(cost2)
    cost['Iteration'] = cost.index
    # Make a plot of the results, epoch vs losses
    plot1 = sns.lineplot(x='Iteration',
            y='cost',
            hue='cent',
            data=cost,
            linewidth=2.5,
            marker="o")
    plot1.set(xlabel = "Iteration")
    plot1.set(ylabel = "Cost Values")
    plot1.set(title =  metric)
    plot1.figure.savefig('plot_' + metric + '.png')
    print("the figure has been generated!")
    # Clear the stage
    plt.clf()
